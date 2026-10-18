# WP-F5 — PPISP Photometric Compensation (`--ppisp` / `--ppisp_controller`) — RECORD (2026-10-18)

**Verdict: INFRASTRUCTURE + VS-BILATERAL-GRID GATE PASSED; VS-UNCOMPENSATED (`base`) CONDITIONAL (DEFAULT OFF).**
1. **PPISP vs Bilateral Grid (`--bilateral_grid`):** PPISP (`--ppisp`) **decisively beats `--bilateral_grid` on all 3 scenes and every primary metric** — raw PLY held-out masked LPIPS improves **-11.0% on Laptop** (`0.3154 -> 0.2805`), **-17.2% on GPU_V4** (`0.4498 -> 0.3724`), and **-6.6% on 360** (`0.5928 -> 0.5536`), with **+3.39 dB / +5.90 dB / +3.90 dB** higher masked PSNR. Mechanism: our vendored BilaRF grid (`lib_bilagrid.py`) has only a spatial TV regularizer and **no identity/mean anchor**, so it absorbs a large global colour/brightness shift into the discarded grid (`raw_mean_rgb` drifts by up to 0.11 vs GT), leaving the exported PLY with a heavy warm/dark colour cast. PPISP's zero-mean exposure/white-balance regularizer anchors the Gaussians' canonical radiance to the capture's mean tone.
2. **PPISP vs Uncompensated (`base`) on our 3 banked captures:** all three banked captures have **very low exposure variation** (`exp_sd = 0.056` EV on Laptop, `0.035` EV on GPU_V4, `0.059` EV on 360 `theater_027_8k`). On such well-controlled captures:
   - **With PPISP applied at novel views (`frame_idx=-1`, `isp_lpips` / `isp_psnr`):** `pp`/`ppc` **match or beat `base` on all 3 scenes** (Laptop `isp_lpips` `0.2783` vs `base` `0.2790`; GPU_V4 `ppc` `isp_psnr` `24.78 dB` vs `base` `24.74 dB`; 360 `ppc` `isp_lpips` `0.5847` / `18.63 dB` vs `base` `0.5857` / `18.58 dB`).
   - **On the raw exported PLY (what Postshot displays, without the camera's learned vignetting & CRF):** `base` leads `pp` by **0.0013–0.0026 masked LPIPS** (`0.31–0.63 dB` PSNR) because PPISP factors per-camera vignetting and toe/shoulder CRF out of the Gaussians; after per-image affine colour correction (`cc_lpips`, isolating structure from tone), `pp` and `base` are tied on Laptop/GPU_V4 and **`pp`/`ppc` beat `base` on 360** (`cc_lpips` `0.6235` / `0.6223` vs `0.6251`, `cc_psnr` `+0.12 dB`).
   - **On No-Reference / Temporal / Off-Path Fog metrics (raw PLY):** `pp` **beats `base` on Laptop** (CrossScore `0.8667` vs `0.8648`, NOVA `0.1169` vs `0.1209`, RAFT-warp `0.0394` vs `0.0404`, off-path fog-layer alpha `0.0464` vs `0.0498`, `prune_delta_mae` `0.183` vs `0.333`, `op<0.05` `26.35%` vs `26.74%`); on **GPU_V4**, `pp` reduces off-path fog-layer alpha (`0.0790` vs `0.0841`) and RAFT-warp (`0.0139` vs `0.0154`), while `ppc` leads CrossScore (`0.8387` vs `0.8273`), RAFT-warp (`0.0133`), and `op<0.05` (`30.59%` vs `32.68%`); on **360**, `pp` cuts spatial outliers beyond 3x bbox by **-16.9%** (`3,649` vs `4,393`).
3. **Controller distillation (`--ppisp_controller`, `ppc` vs `pp`):** freezing splats at 80% (`step 24001`) to train the novel-view exposure/WB controller helps **ISP-rendered novel views** (`isp_psnr` `ppc > pp` on all 3 scenes) but **hurts raw PLY LPIPS** slightly (`+0.0030` on Laptop, `+0.0067` on GPU_V4, `+0.0014` on 360) because static PLY consumers (Postshot, SuperSplat) never run the controller and the Gaussians lose 6,000 fine-phase optimization steps.
4. **Operational rule:** Keep `--ppisp` **default-OFF** (production unchanged on controlled captures). Whenever photometric compensation is needed (auto-exposure / variable-WB captures), **use `--ppisp` (without `--ppisp_controller` for PLY/Postshot delivery) and never `--bilateral_grid`**.

---

## 1. Build, Wiring & Memory Leak Check

| Item | Detail |
|---|---|
| Upstream source | `nv-tlabs/ppisp` @ **`a247368047651db2e03e6f97a36e4f5bebbe5c5f`** (2026-09-23, `v1.3.0` + 5 fixes; Apache-2.0), cloned to `D:\REPOS\HYPERclone\ppisp` |
| Build recipe | `tools/blackwell_session_2026-06/build_ppisp.bat` — VS2022 vcvars64, CUDA v12.9, `TORCH_CUDA_ARCH_LIST=12.0`, `NVCC_PREPEND_FLAGS=-I.../tools/patches/cuda129_win_overlay`, `--no-build-isolation --no-deps`, installed into `D:\REPOS\HYPERclone\venvs\gsplat_main` only |
| Leak check | `_infusion/wpf5_leakcheck.py` -> `_infusion/wpf5/leakcheck.json`: 3,000 fwd+bwd+step iterations at 1920x1080 (266 frames, controller ON); CUDA allocated memory **72.02 MB -> 67.12 MB (delta -4.90 MB, 0 leak)** (`v1.3.0` includes the PR #7 `py::return_value_policy::move` fix) |
| Trainer wiring | `core/training/gsplat_trainer.py`: setup `:1197-1245`, controller freeze `:1456-1470` (freezes splats + poses and skips `strategy.step_pre_backward` / `step_post_backward` so MCMC relocation + noise stop), render hook `:1504-1511` (unclamped radiance into `ppisp_mod`), reg loss `:1796-1800`, optimizer/scheduler step `:1846-1851`, per-loss log `:1891-1895`, ckpt side-car `:1912-1922`, novel-view eval `:1986-2003`, CLI flags `:2224-2238` (`--ppisp`, `--ppisp_controller`, `--ppisp_controller_start 0.8`) |
| Backend wiring | `core/training/gsplat_backend.py`: defaults `:305-310` (`ppisp: False`, `ppisp_controller: False`, `trainer_python: None`), subprocess argv `:443, :542-547` |
| Evaluation bug fix | `core/qa/masked_metrics.py:104-106`: added `lp.reset()` inside `_lpips_window` after `float(lp(r, g))` (verified bit-identical; fixes `torchmetrics` `all_scores` list accumulation that caused $O(N^2)$ slowdown across multi-arm 360 tiled evaluations, speeding up 144-crop eval from 4,021 s to 64 s) |

### Negative probe recorded: baking PPISP's CRF into PLY SH coefficients (`_infusion/wpf5_crf_bake_probe.py`)
Tested applying the learned per-camera toe/shoulder/gamma CRF curve directly to each Gaussian's `sh0` (and first-order slope to `shN`) on Laptop `wpf5/pp` (`_infusion/wpf5/laptop_tone_crfbaked.json`). **Result: NEGATIVE (do not ship).** Held-out raw LPIPS regressed `0.2805 -> 0.2863` and PSNR fell `22.69 -> 22.23 dB`; mean RGB overshot to `0.519` (vs GT `0.507`, unbaked `0.497`) because PPISP also absorbed lens vignetting (`alpha_0 ~ -0.28`, darkening image corners) which is image-space and cannot be baked into 3D Gaussians. Exporting the unbaked canonical radiance is strictly better.

---

## 2. Experimental Matrix & Training Logs

Drivers: `_infusion/wpf5_run.sh <laptop|gpu|360> <base|bil|pp|ppc>`, `_infusion/wpf5_chain.sh <A|B>`, `_infusion/wpf5_eval.sh <laptop|gpu|360>`.
All 12 arms ran in `D:\REPOS\HYPERclone\venvs\gsplat_main` (Python 3.11, torch 2.11.0+cu128, gsplat 1.6.0 @ `512d366`, ppisp 1.3.0 @ `a247368`), `seed 0`, `opacity_reg 0`.
- **Laptop & GPU_V4 recipe (WP-S1 STACK):** `--data_factor 2 --max_steps 30000 --test_every 8 --seed 0 --sh_degree 3 --ssim_lambda 0.2 --strategy mcmc --cap_max 3000000 --opacity_reg 0 --scale_reg 0.01 --antialiased --normalize_world_space --pose_opt --pose_opt_lr 1e-5 --coarse_factor 4 --coarse_until 15000 --lpips_lambda 0.2 --lpips_patch 512 --lpips_start 3000 --lpips_gated --mono_depth_lambda 0.05 --mono_depth_start 2000 --mono_depth_use_weight --erank_lambda 0.01 --erank_start 7000`. (Laptop `base` reuses the identical WP-B2 run `_dataset/Laptop/apex_images/training/opreg0/b2`.)
- **360 recipe (WP-O5 arm D):** `--data_dir _dataset/360/train/wpo5_theater_027_8k --holdout_list .../holdout_test.txt --max_steps 30000 --seed 0 --data_factor 1 --strategy mcmc --cap_max 3000000 --antialiased --opacity_reg 0 --scale_reg 0.01 --ssim_lambda 0.2 --sh_degree 3 --loss_mask --coarse_factor 0 --erank_lambda 0.01 --erank_start 7000`.

### Per-loss & training summary at step 30,000

| Scene : Arm | Cams / Train views | Step-30k train loss / PSNR | `erank` | `mono` | `ppisp_reg` | `exp_sd` (EV) | `exp_rng` (EV) | Raw Eval PSNR / SSIM | PPISP Novel Eval PSNR / SSIM | Wall time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| `laptop:base` | 1 / 266 | 0.0746 / 25.30 dB | 2.67e-4 | 0.2081 | — | — | — | 23.00 / 0.8312 | — | 8,195 |
| `laptop:bil` | 1 / 266 | 0.0932 / 24.76 dB | 2.38e-4 | 0.1646 | — | — | — | 19.31 / 0.8107 | — | 8,852 |
| `laptop:pp` | 1 / 266 | 0.0707 / 26.09 dB | 2.62e-4 | 0.1949 | 1.68e-5 | 0.056 | [-0.22, +0.16] | 22.69 / 0.8300 | 22.98 / 0.8314 | 9,276 |
| `laptop:ppc` | 1 / 266 | 0.0822 / 25.09 dB | 2.53e-4 | 0.2356 | 1.98e-5 | 0.056 | [-0.22, +0.15] | 22.60 / 0.8294 | 23.00 / 0.8309 | 8,589 |
| `gpu:base` | 1 / 497 | 0.0762 / 27.34 dB | 1.82e-4 | 0.1423 | — | — | — | 24.74 / 0.8438 | — | 8,982 |
| `gpu:bil` | 1 / 497 | 0.1137 / 24.92 dB | 1.54e-4 | 0.1225 | — | — | — | 18.22 / 0.7304 | — | 9,597 |
| `gpu:pp` | 1 / 497 | 0.0758 / 28.11 dB | 1.74e-4 | 0.1475 | 1.49e-5 | 0.035 | [-0.08, +0.48] | 24.11 / 0.8356 | 24.61 / 0.8431 | 9,582 |
| `gpu:ppc` | 1 / 497 | 0.0753 / 27.47 dB | 1.83e-4 | 0.1251 | 2.93e-5 | 0.021 | [-0.07, +0.05] | 24.15 / 0.8310 | 24.78 / 0.8445 | 10,929 |
| `360:base` | 12 / 984 | 0.0889 / 23.30 dB | 7.87e-5 | — | — | — | — | 19.34 / 0.7291 | — | 1,195 |
| `360:bil` | 12 / 984 | 0.1100 / 21.50 dB | 5.37e-5 | — | — | — | — | 16.20 / 0.7123 | — | 2,590 |
| `360:pp` | 12 / 984 | 0.0778 / 24.41 dB | 7.86e-5 | — | 1.73e-6 | 0.059 | [-0.22, +0.24] | 19.05 / 0.7270 | 19.28 / 0.7290 | 1,230 |
| `360:ppc` | 12 / 984 | 0.0846 / 23.27 dB | 7.18e-5 | — | 1.81e-6 | 0.059 | [-0.21, +0.24] | 18.95 / 0.7268 | 19.24 / 0.7290 | 1,153 |

*(Note: Chains A and B ran concurrently on the 96 GB GPU; on the non-LPIPS 360 runs, PPISP added only +3% wall time — 24.6 it/s vs 25.4 it/s — whereas `bil` cut throughput in half to 11.6 it/s.)*

---

## 3. Gate Tables — Held-Out, Off-Path Fog & Tone Split

### 3.1 Laptop (38 held-out views, factor 2; `_infusion/wpf5/laptop_gate.json` & `laptop_tone.json`)

| Arm | mLPIPS | mPSNR | mSSIM | Deck LPIPS | CrossScore | NOVA | TSED | RAFT-warp | JOD | `op<0.05` | `foglayerA` | `prune_dMAE` | `needle%` | `giant%` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **base** | **0.2791** | **23.008** | **0.8312** | **0.2407** | 0.8648 | 0.1209 | 0.939 | 0.0404 | **-2.861** | 26.74% | 0.0498 | 0.333 | 8.80 | 11.27 |
| bil | 0.3154 | 19.308 | 0.8107 | 0.2684 | 0.8558 | 0.1245 | **0.970** | **0.0292** | -2.864 | 28.09% | 0.0494 | **0.139** | **8.08** | 10.55 |
| **pp** | 0.2805 | 22.694 | 0.8300 | 0.2422 | **0.8667** | **0.1169** | 0.939 | 0.0394 | -2.891 | 26.35% | **0.0464** | 0.183 | 8.68 | 10.96 |
| ppc | 0.2835 | 22.608 | 0.8294 | 0.2440 | 0.8629 | 0.1211 | **0.970** | 0.0390 | -2.875 | **25.32%** | 0.0566 | 0.195 | 8.37 | **9.81** |

**Tone vs structure split (`_infusion/wpf5/laptop_tone.json`, full-frame):**
| Arm | `raw_lpips` / `raw_psnr` (PLY as-is) | `cc_lpips` / `cc_psnr` (affine colour-corrected) | `isp_lpips` / `isp_psnr` (PPISP `frame_idx=-1`) | `raw_mean_rgb` (GT = `[0.507, 0.500, 0.606]`) |
|---|---|---|---|---|
| base | 0.2790 / **23.000** | **0.2806** / **22.896** | — | `[0.509, 0.503, 0.609]` |
| bil | 0.3153 / 19.306 | 0.3042 / 21.410 | — | `[0.455, 0.421, 0.542]` (severe drift) |
| **pp** | 0.2805 / 22.693 | 0.2809 / 22.834 | **0.2783** / 22.982 | `[0.497, 0.493, 0.597]` |
| ppc | 0.2834 / 22.603 | 0.2840 / 22.756 | 0.2804 / **23.003** | `[0.496, 0.493, 0.597]` |

---

### 3.2 GPU_V4 (72 held-out views, factor 2; `_infusion/wpf5/gpu_gate.json` & `gpu_tone.json`)

| Arm | mLPIPS | mPSNR | mSSIM | Shiny LPIPS | CrossScore | NOVA | TSED | RAFT-warp | JOD | `op<0.05` | `foglayerA` | `prune_dMAE` | `needle%` | `giant%` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **base** | **0.3698** | **24.744** | **0.8437** | **0.2997** | 0.8273 | **0.0232** | 1.000 | 0.0154 | **-2.384** | 32.68% | 0.0841 | 0.356 | 5.91 | 6.82 |
| bil | 0.4498 | 18.217 | 0.7303 | 0.4008 | 0.7841 | 0.0410 | 1.000 | **0.0113** | -2.505 | 34.32% | **0.0790** | 0.219 | **5.10** | **6.59** |
| **pp** | 0.3724 | 24.115 | 0.8356 | 0.3045 | 0.8232 | 0.0255 | 1.000 | 0.0139 | -2.396 | 33.21% | **0.0790** | 0.326 | 5.90 | 6.74 |
| ppc | 0.3792 | 24.152 | 0.8309 | 0.3138 | **0.8387** | 0.0262 | 1.000 | 0.0133 | -2.429 | **30.59%** | 0.0896 | **0.202** | 5.79 | 6.81 |

**Tone vs structure split (`_infusion/wpf5/gpu_tone.json`, full-frame):**
| Arm | `raw_lpips` / `raw_psnr` (PLY as-is) | `cc_lpips` / `cc_psnr` (affine colour-corrected) | `isp_lpips` / `isp_psnr` (PPISP `frame_idx=-1`) | `raw_mean_rgb` (GT = `[0.489, 0.378, 0.278]`) |
|---|---|---|---|---|
| base | **0.3689** / **24.744** | **0.3692** / **24.599** | — | `[0.491, 0.379, 0.280]` |
| bil | 0.4493 / 18.217 | 0.4460 / 20.706 | — | `[0.492, 0.332, 0.203]` (warm/orange cast) |
| **pp** | 0.3715 / 24.115 | 0.3713 / 24.201 | **0.3690** / 24.612 | `[0.498, 0.387, 0.289]` |
| ppc | 0.3785 / 24.152 | 0.3779 / 24.206 | 0.3746 / **24.783** | `[0.496, 0.388, 0.293]` |

---

### 3.3 360 Walkthrough (`wpo5_theater_027_8k`, 144 held-out crops across 12 panos, factor 1; `_infusion/wpf5/eval360/scorecard.json` & `360_tone.json`)

| Arm | Masked LPIPS (tiled) | Masked LPIPS (bbox) | Masked PSNR | Masked SSIM | Full LPIPS | Full PSNR | `op<0.05` | Mean Opacity | Outliers >3x bbox | `needle%` (erank<2) |
|---|---|---|---|---|---|---|---|---|---|---|
| **base** | **0.5524** | **0.5819** | **21.420** | **0.7146** | **0.5810** | **19.335** | 19.54% | 0.4100 | 4,393 | 9.40% |
| bil | 0.5928 | 0.6171 | 17.046 | 0.6978 | 0.6141 | 16.203 | 20.22% | **0.4135** | 3,774 | **9.00%** |
| **pp** | 0.5536 | 0.5836 | 20.950 | 0.7128 | 0.5823 | 19.050 | 20.07% | 0.4062 | **3,649** | 9.52% |
| ppc | 0.5550 | 0.5836 | 20.817 | 0.7127 | 0.5824 | 18.946 | **19.19%** | 0.4060 | 5,142 | 10.74% |

**Tone vs structure split (`_infusion/wpf5/360_tone.json`, 48 held-out views, full-frame):**
| Arm | `raw_lpips` / `raw_psnr` | `cc_lpips` / `cc_psnr` (structure-only) | `isp_lpips` / `isp_psnr` (PPISP `frame_idx=-1`) | `raw_mean_rgb` (GT = `[0.590, 0.566, 0.493]`) |
|---|---|---|---|---|
| base | **0.5857** / **18.583** | 0.6251 / 16.031 | — | `[0.581, 0.563, 0.487]` |
| bil | 0.6208 / 15.753 | 0.6414 / 15.705 | — | `[0.476, 0.468, 0.381]` (dark/greenish cast) |
| **pp** | 0.5876 / 18.251 | 0.6235 / **16.148** | 0.5854 / 18.492 | `[0.561, 0.542, 0.465]` |
| **ppc** | 0.5872 / 18.210 | **0.6223** / 16.144 | **0.5847** / **18.632** | `[0.558, 0.539, 0.462]` |

---

## 4. Visual Inspection (`_infusion/wpf5/sheets/` & `_infusion/wpf5/eval360/`)

- **Laptop & GPU_V4 off-path sheets (`_infusion/wpf5/sheets/{laptop,gpu}_cmp_*.jpg`):**
  - `bil` shows a pronounced colour shift on both scenes (cooler/darker on Laptop patio; heavy yellow-orange cast on GPU_V4's white polka-dot cloth and shroud).
  - `base`, `pp`, and `ppc` are visually close across all 21 off-path viewpoints (dolly-in, pull-out, elevation +25/-20 deg, lateral, azimuth gap). On Laptop, `pp` has slightly cleaner silhouette edges under pull-out/elevation (`fog_layer_alpha_mean` 0.0464 vs 0.0498). On GPU_V4, neutral whites on the polka-dot cloth and metallic lettering on the shroud are crisp and well-matched across `base` and `pp`.
- **360 off-path & fog-pair sheets (`_infusion/wpf5/eval360/offpath/cmp_*.jpg`, `fog/fogpair_*.jpg`):**
  - At lateral +-1.0 m (`cmp_pano_00031_lateral_08.jpg`), `bil` is visibly murkier and greenish; `base`, `pp`, and `ppc` preserve room geometry and wall/floor tone similarly.
  - At the unvisited interior viewpoint 4.62 m from any capture station (`cmp_unvisited_unvisited_28.jpg`), all four arms remain occluded by unvisited-space floaters — confirming that photometric compensation does not cure missing parallax on a walkthrough.
  - In the full-vs-pruned (`op>=0.05`) fog pairs (`fogpair_{base,pp}_lateral_0.jpg`), pruning ~20% low-opacity Gaussians leaves the rendered image visually identical in all arms.

---

## 5. Persisted Artifacts

- **Record & scripts:** `_infusion/wpf5_RECORD.md` (this file), `_infusion/wpf5_run.sh`, `_infusion/wpf5_chain.sh`, `_infusion/wpf5_eval.sh`, `_infusion/wpf5_tone_eval.py`, `_infusion/wpf5_crf_bake_probe.py`, `_infusion/wpf5_leakcheck.py`, `tools/blackwell_session_2026-06/build_ppisp.bat`.
- **Committable gate JSONs & sheets:** `_infusion/wpf5/{laptop_gate.json,laptop_tone.json,laptop_tone_crfbaked.json,gpu_gate.json,gpu_tone.json,360_tone.json,leakcheck.json}`, `_infusion/wpf5/eval360/{scorecard.json,base/metrics.json,bil/metrics.json,pp/metrics.json,ppc/metrics.json,offpath/*.jpg,fog/*.jpg}`, `_infusion/wpf5/sheets/*.jpg`.
- **QA directories:** `qa/wpf5_{laptop,gpu}_{base,bil,pp,ppc}/` (`scorecard.json`, `region_*.json`, `floaters.txt`, `prune_stats.json`), `qa/wpf5_360_{base,bil,pp,ppc}/prune_stats.json`, `qa/wpf5_{laptop,gpu}_offpath/offpath.json`.
- **Pruned deliverable PLYs (gitignored `_dataset/`):**
  - `_dataset/Laptop/apex_images/training/wpf5/{bil,pp,ppc}/output/ply/point_cloud_29999_pruned_op05.ply` (+ `opreg0/b2` for `base`)
  - `_dataset/GPU_V4/apex_images/training/wpf5/{base,bil,pp,ppc}/output/ply/point_cloud_29999_pruned_op05.ply`
  - `_dataset/360/train/wpf5_{base,bil,pp,ppc}/output/ply/point_cloud_29999_pruned_op05.ply`
