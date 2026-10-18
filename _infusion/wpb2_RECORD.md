# WP-B2 — gsplat `main` AOT build in an isolated venv — RECORD (2026-10-17/18)

**Verdict: GATE PASSED.** gsplat `main` (1.6.0, unreleased) builds AOT for sm_120 on Windows
after 4 small source patches + 1 CUDA-header overlay, lives only in
`D:\REPOS\HYPERclone\venvs\gsplat_main`, and our trainer runs under it **unmodified** and
reproduces the production result on Laptop within run-to-run noise (masked LPIPS 0.2791 vs
0.2794 / 0.2800 for two gsplat-1.5.3 runs of the same argv+seed). MCMC and Default strategy
smoke tests pass. The system-site gsplat 1.5.3 was never touched
(`python -c "import gsplat"` -> 1.5.3, `...\Python311\Lib\site-packages\gsplat`). No speed gain on
this recipe (8195 s vs 8211 s, run concurrently) — the step time is dominated by LPIPS / I/O, not
rasterization.

## 1. Pins
| item | value |
|---|---|
| gsplat source | `D:\REPOS\HYPERclone\gsplat_main` — `git clone --recursive https://github.com/nerfstudio-project/gsplat` |
| **gsplat commit** | **`512d366b67073d77ca099ede742683c165dfc23b`** (2026-09-19, "[perf] 3DGS training (simple_trainer.py) on 8x B200 ... (#1062)"), `__version__ = "1.6.0"` |
| submodules | glm `2d4c4b4dd31fde06cfffad7915c2b3006402322f`, googletest `d72f9c8aea6817cdf1ca0ac10887f328de7f3da2` |
| venv | `D:\REPOS\HYPERclone\venvs\gsplat_main` (`python -m venv --system-site-packages`, system Python 3.11) |
| torch | inherited **2.11.0+cu128** (`...\Python311\Lib\site-packages\torch`, NOT reinstalled); numpy 2.4.3 inherited |
| venv-local packages | setuptools 70.2.0 + pip 26.0.1 (aligned to system, `-c constraints.txt`), gsplat 1.6.0, ppisp 1.3.0 (WP-F5) |
| build | `tools/blackwell_session_2026-06/build_gsplat_main.bat` — VS2022 vcvars64 (MSVC 14.44), CUDA v12.9, `TORCH_CUDA_ARCH_LIST=12.0`, `DISTUTILS_USE_SDK=1`, `--no-build-isolation --no-deps --ignore-installed -c constraints.txt`, `BUILD_EXPERIMENTAL=0` |
| deps not installed | `jaxtyping`, `nvtx` (declared but unused / optional: `gsplat/trace.py` falls back to no-ops). Nothing pulled flash-attn or touched torch/numpy pins. |

## 2. Windows build fixes (all recorded; the funded port must re-apply or upstream them)
1. **CUDA 12.9 / CCCL 2.8.2 header bug (toolkit NOT modified).** `cuda/__ptx/instructions/generated/clusterlaunchcontrol.h`
   binds `long2` (32-bit `long` on LLP64 Windows) to the 64-bit `"l"` asm constraint ->
   `asm operand type size(4) does not match ... 'l'` in every sm_100+/sm_120 TU that includes
   `<cub/cub.cuh>` (IntersectTile*.cu, Projection2DGSPacked, ProjectionUT3DGSFused). Fix: a
   patched copy of that ONE header (`long2` -> `longlong2`, the upstream CCCL fix) at
   `tools/patches/cuda129_win_overlay/cuda/__ptx/instructions/generated/clusterlaunchcontrol.h`,
   put first on nvcc's include path via `NVCC_PREPEND_FLAGS=-I<overlay>` (also used for PPISP).
2. `gsplat/cuda/include/Cameras.cuh:782` — `std::isfinite` in device code resolves to MSVC's
   host-only template -> replaced with `fabsf(x) <= FLT_MAX` (identical semantics, NaN-false).
3. `RasterizeToPixelsFromWorld3DGSParallelBatchBwd.cu` — `__builtin_clzll` in a host function
   (MSVC has none) -> portable bit-length loop; and `at::cuda::cub::radix_sort_pairs<int64,int32>`
   -> `detail::radix_sort_pairs_impl<int64,4>` is not exported by torch_cuda.dll on Windows
   (LNK2001) -> direct `CUB_WRAPPER(cub::DeviceRadixSort::SortPairs, ...)` (same pattern gsplat
   already uses in `SparseTileLayout.cu`).
4. `SphericalHarmonicsL1PlusCUDA.cu:707` — `std::array<T, 0>::operator[]` (DEGREE 0 -> MAX_K 0) is
   host-only in MSVC's STL -> plain C array `opmath_t acc[MAX_K > 0 ? MAX_K : 1]`.
5. `RasterizeToPixelsFromWorld3DGS.h:120` — declaration `bool fwd_only` vs definition
   `const bool fwd_only`: MSVC mangles the top-level const -> LNK2001 -> header made `const bool`.
6. `BUILD_EXPERIMENTAL=0`: the experimental HiGS inference extension is Linux-only code (GCC
   builtins in `experimental/.../gaussian_inference/Utils.h`); unused by us.

Full diff (CRLF source files): `tools/patches/gsplat_main_512d366_win.diff`.
Fixes 3-5 are all in the 3DGUT / sparse paths we do not use; they were needed only to link.

## 3. API changes vs 1.5.3 (what our code touches)
- **Our trainer needs NO shim**: `gsplat.rendering.rasterization` (same kwargs we pass),
  `gsplat.strategy.{MCMCStrategy,DefaultStrategy}` (same constructor fields, same
  `step_pre_backward` / `step_post_backward(..., lr=)` hooks; new optional `scene=`),
  `gsplat.export_splats` (same signature), `gsplat.cuda._wrapper.fully_fused_projection`,
  `gsplat.cuda._torch_impl._eval_sh_bases_fast` / `_quat_to_rotmat` all still exist.
- **"Everything now lives under the `gsplat` namespace"** (README): the public names we use did
  not move; `gsplat.strategy` is still the import path.
- **inject_noise change:** MCMC noise is now a native CUDA kernel
  (`gsplat.strategy.ops.inject_noise_to_position(..., noise_scale=None, *, scaler=None, t=0.005, k=100)`;
  `GSPLAT_MCMC_BACKEND=torch` forces the PyTorch fallback). Same maths as 1.5.3
  (`sigmoid(-k (opacity - t))`, k 100, t 0.005); different RNG stream, so seeded runs are
  statistically, not bit-wise, comparable. `MCMCStrategy` gained `noise_injection_stop_iter`,
  `noise_opacity_t/k` (we do not set them; our PPISP freeze skips `step_post_backward` instead,
  which works on both versions).
- **BREAKING for non-trainer code: `gsplat.spherical_harmonics`** changed from
  `(degree, dirs, coeffs, masks)` to `(degree, means, viewmats, coeffs, masks, ...)` (view
  direction fused into the kernel). Shim: **`core/training/gsplat_compat.py::spherical_harmonics_dirs`**
  (identity view matrix => `means - campos == dirs`), tested on BOTH versions by
  `tests/test_gsplat_compat.py` (5/5 pass in each env). Callers still on the old signature (not
  edited here — they belong to the concurrent WP-SH4 work): `tools/align_ply.py` (SH rotation
  oracle), `tests/test_sh_degree4.py`, `tests/test_sh_ply_io.py`. Under the venv those 20 tests
  fail with `TypeError: spherical_harmonics() missing 1 required positional argument: 'coeffs'`;
  routing them through the shim fixes it. Under system gsplat 1.5.3: **343 passed**.

## 4. Gate — Laptop production stack, opacity_reg 0, seed 0
Same argv as `_infusion/opreg0_RECORD.md` §3 with `--opacity_reg 0`; arms differ ONLY in the
interpreter (`_infusion/opreg0_run.sh <arm> [python]`):
| arm | interpreter / gsplat | wall clock |
|---|---|---|
| `or0` | system python, gsplat 1.5.3 (STEP 0 run) | 8631 s (concurrent with or01) |
| `or0r` | system python, gsplat 1.5.3 — seed-0 REPEAT of or0 (noise floor) | 8211 s (concurrent with b2) |
| **`b2`** | `venvs/gsplat_main` python, **gsplat 1.6.0 @512d366** | **8195 s** (concurrent with or0r) |

All three are evaluated by the SAME evaluator (system qa_report / gsplat 1.5.3 rasterizer), so
render differences come from the checkpoints only. `_infusion/wpb2_laptop_gate.json`:

| arm | mLPIPS | mPSNR | mSSIM | deck LPIPS | crossscore | nova | tsed | raft | jod | op<0.05 | fog-layer α | needle% | giant% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| or0 (1.5.3) | 0.2794 | 23.037 | 0.8311 | 0.2400 | 0.8651 | 0.1180 | 0.970 | 0.0418 | -2.859 | 0.267 | 0.049 | 8.87 | 11.16 |
| or0r (1.5.3) | 0.2800 | 23.000 | 0.8315 | 0.2419 | 0.8621 | 0.1203 | 0.939 | 0.0411 | -2.866 | 0.268 | 0.048 | 8.82 | 11.28 |
| **b2 (main)** | **0.2791** | 23.008 | 0.8312 | 0.2407 | 0.8648 | 0.1209 | 0.939 | 0.0404 | -2.861 | 0.267 | 0.050 | 8.80 | 11.27 |

Run-to-run spread of the two 1.5.3 runs: 0.0006 LPIPS, 0.037 dB, 0.0019 deck LPIPS, 0.031 tsed.
b2 sits inside that spread on every metric (closer to or0 than or0r is). Off-path sheets
(`qa/opreg0_laptop_offpath_or0_or0r_b2/sheets/`, 21 viewpoints) are visually indistinguishable.
Pruned PLY: `_dataset/Laptop/.../opreg0/b2/output/ply/point_cloud_29999_pruned_op05.ply`
(26.7 % removed). Trainer-reported held-out (raw, unmasked): or0 23.04/0.8311, or0r 22.99/0.8314,
b2 23.00/0.8312.

**MCMC smoke (3k steps, Laptop factor 4, cap 500k, seed 0):** gsplat main 22.38 dB / 0.8165,
126->132 it/s; gsplat 1.5.3 22.36 dB / 0.8161, 124->127 it/s; identical Gaussian-count trajectory
(118 853 / 193 592 / 300 318). **Default strategy smoke (2k steps):** runs, 21.05 dB, 636 717 GS.
Logs: `_dataset/Laptop/.../opreg0/{b2_mcmc_smoke,v153_mcmc_smoke,b2_default_smoke}/train.log`.

## 5. Rollback
Nothing to roll back: production keeps the system gsplat 1.5.3. Delete
`venvs/gsplat_main` to remove 1.6.0 entirely.

## 6. Persisted artifacts
`_infusion/wpb2_RECORD.md`, `_infusion/wpb2_laptop_gate.json`, `_infusion/opreg0_run.sh`,
`_infusion/opreg0_eval.sh`, `tools/blackwell_session_2026-06/build_gsplat_main.bat`,
`tools/patches/gsplat_main_512d366_win.diff`, `tools/patches/cuda129_win_overlay/`,
`core/training/gsplat_compat.py`, `tests/test_gsplat_compat.py`,
`qa/opreg0_laptop_{or0,or0r,b2}/`, `qa/opreg0_laptop_offpath_or0_or0r_b2/offpath.json`.
