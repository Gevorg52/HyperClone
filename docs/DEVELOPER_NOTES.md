# Apex Reality — Developer Notes & Engineering Log

Canonical "read this first" reference for anyone working on this SfM→3DGS pipeline.
Captures the build environment, the trainer's quality knobs, and — most valuably —
the **empirical learnings and dead-ends** from the June 2026 quality investigation,
so you don't relearn them the hard way.

Companion docs (deeper detail):
- `docs/quality-roadmap.md` — the full quality investigation, ranked levers, and
  every empirical update (cap_max, geo-reg, LPIPS, WD-R), with the reasoning.
- `docs/research-findings.md` — literature/SOTA research (extractors, matchers,
  trainers, enhancement, June 2026).
- `docs/brief.txt`, `docs/rules.txt` — original product brief and working rules.

---

## 1. Pipeline overview

GUI (Gradio) pipeline, JPEG stills (or extracted video frames) → COLMAP-format
sparse reconstruction → 3D Gaussian Splatting → optional diffusion enhancement.
Target: max-quality, artifact-free splats from ordinary footage. RTX 4090 (24 GB)
now; Blackwell Pro 6000 (96 GB) later. **Quality ≫ speed. RESEARCH-ONLY** — license
restrictions are not a selection filter (we reimplement best methods as our own OSS).

10 stages (`gui/stages/`): 1 import/QC · 2 ALIKED extraction · 3 LightGlue matching
(+ optional RoMa v2 dense rescue) · 4-5 COLMAP recon (incremental or GLOMAP global) ·
6 final BA · 7 export (+ gravity alignment) · 8 masking (BiRefNet + SAM 2.1) ·
9 gsplat training · 10 enhancement (NVIDIA Fixer + distill-back loop).
Entry point: `run.py` (launches the Gradio app). Core logic in `core/`; the trainer
is a self-contained subprocess (`core/training/gsplat_trainer.py`).

---

## 2. Environment & build (CRITICAL — read before touching CUDA code)

> ### ⚡ MACHINE MIGRATION — Blackwell RTX PRO 6000 / 96 GB (2026-06-29). This supersedes the 4090/conda details below.
> Verified rebuilt + working on the new machine (see `docs/BLACKWELL_FRONTIER.md`, memory `blackwell-build`):
> - **GPU:** RTX PRO 6000 Blackwell, **sm_120 / cap (12,0)**, 96 GB, driver 576.57.
> - **Python:** **system Python 3.11** at `C:\Users\gevor\AppData\Local\Programs\Python\Python311\python.exe` — **NO conda on this machine** (the anaconda `base` below was the OLD 4090 box). All pip installs use a constraints file pinning torch to avoid clobbering it.
> - **Torch:** **2.11.0+cu128** (cu128 build, `arch_list` includes sm_120) — pre-installed, drives the card, **do not reinstall**.
> - **CUDA Toolkit:** **v12.9** only (`...\CUDA\v12.9`). ≥12.8 needed for native sm_120 cubins — satisfied.
> - **MSVC:** VS2022 **BuildTools 14.44** — **no `-vcvars_ver` pin needed** now (the 14.34 pin was only for CUDA<12.4; we're on 12.9). Just `call vcvars64.bat`.
> - **gsplat:** rebuilt **AOT v1.5.3 for sm_120** with `TORCH_CUDA_ARCH_LIST="12.0"`, `CUDA_HOME=...\v12.9`, `DISTUTILS_USE_SDK=1`. Verified: `from gsplat import csrc` + real rasterization (Mip-Splatting antialiased + MCMC) runs on the card. Build script: scratchpad `build_gsplat.bat`.
> - **Installed + verified:** pycolmap **4.1.0**, hloc 1.5 (+lightglue, editable from `../Hierarchical-Localization`), kornia, imageio, torchmetrics+lpips (gated-LPIPS VGG cached), tensorly, opencv 4.13, gradio 6.11.
> - **Native 24 MP training validated**: `--data_factor 1` (our loader reads full-res `images/` directly — NO stock-3DGS 1600px cap exists in our code). GPU_V4 24 MP: ~6.3 GB at 123k GS, headroom huge.
> - **WSL2 Ubuntu-24.04 present** (for ArtiFixer's Linux/Docker env). **Docker NOT installed.**

- **(OLD 4090 box — historical)** There is NO named conda env. The whole stack lives in the **anaconda `base`
  env** (`C:\Users\gevor\anaconda3`, **Python 3.11.5**). Activate:
  `& C:\Users\gevor\anaconda3\shell\condabin\conda-hook.ps1; conda activate base`.
- **Torch:** 2.3.0+cu121. **CUDA toolkits installed:** v11.7, v11.8 (default
  `CUDA_PATH`!), v12.1. For any CUDA build, set `CUDA_HOME`/`CUDA_PATH` to v12.1.
- **MSVC pin:** VS 2022's default toolset (14.4x) STL hard-errors with CUDA < 12.4
  (`STL1002`). Use `vcvars64.bat -vcvars_ver=14.34` (toolsets 14.16–14.38 installed).
- **gsplat** is built **AOT from source** (the PyPI wheel is pure-Python = JIT-only,
  which needs `cl.exe` in *every* process → unusable for the GUI). Recipe:
  ```
  set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
  set "CUDA_PATH=%CUDA_HOME%"
  set "PATH=C:\Users\gevor\anaconda3;...\Scripts;...\Library\bin;%CUDA_HOME%\bin;%PATH%"
  set "TORCH_CUDA_ARCH_LIST=8.9+PTX"   (4090 = sm_89)
  set "DISTUTILS_USE_SDK=1"            (required when VC env active, else torch build aborts)
  set "MAX_JOBS=16"
  call "...\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.34
  pip install --force-reinstall --no-deps --no-build-isolation --no-cache-dir \
    "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3"
  ```
  Verify: `python -c "from gsplat import csrc"`. (Current pin: gsplat **1.5.3**.)
  - `core/training/gsplat_jit_shim.py` fixes a gsplat<=1.5.3 bug where it passes
    GCC-style flags to MSVC (D8021) — imported before gsplat everywhere.
- **pycolmap 4.0.4** (COLMAP 4.0: GLOMAP merged, native ALIKED+LightGlue, faster BA,
  deterministic `random_seed`). NOTE: pycolmap 4.x made `Image.cam_from_world` a
  **method** (was a property in 3.x) and moved BA solver opts to
  `BundleAdjustmentOptions.ceres.solver_options` — see `core/alignment._cam_from_world`
  and `gui/stages/export._run_standalone_ba` for the version-robust handling.
- **Gotchas seen:** cv2 "DLL load failed" → `pip install --force-reinstall --no-deps
  opencv-contrib-python==4.11.0.86`. `ninja` must be on PATH for JIT. In `cmd /c`
  chains, `%PATH%` expands at parse time — set PATH BEFORE `call vcvars64.bat`.

### Running it
- GUI: `python run.py`.
- Headless SfM (stages 1-7): see `run_pipeline_laptop.py` for a worked example —
  drives the real stage handlers with a no-op `progress` stub.
- Training is a subprocess; flags below. Tests: `pytest tests` (134 passing;
  `tests/smoke_gsplat.py` needs a VS dev shell for first JIT).

---

## 3. The 3DGS trainer — quality knobs & recommended settings

`core/training/gsplat_trainer.py` (CLI; driven by `GsplatBackend`). Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--data_factor` | 2 | training resolution = source / factor. **Relative to source!** 24MP→factor2=6MP; 4K→factor1=8MP. Pick by target MP, not habit. |
| `--strategy` | mcmc | `mcmc` (MCMC-3DGS, better quality + budget control) or `default`. |
| `--cap_max` | 1e6 | MCMC Gaussian budget. **~3M is near-optimal for a single object+scene; more HURTS** (see §6). |
| `--antialiased` | on | Mip-Splatting (`rasterize_mode`). Keep on. |
| `--pose_opt` | off→on(GUI) | SE(3) per-camera refinement, lr 1e-5. |
| `--coarse_factor`/`--coarse_until` | — | coarse-to-fine: densify at lower res, finish at full. |
| `--opacity_reg`/`--scale_reg` | 0.01 | MCMC regularizers. **On a many-view walkthrough `opacity_reg` must be exactly 0 — 0.01 AND 0.001 both destroy the model (WP-O3, WP-O5); "small" is not safe, see §6.** `scale_reg` is harmless — keep it. |
| `--erank_lambda`/`--erank_start` | 0/7000 | Anti-needle hinge (WP-Q2). On the WP-O5 8K walkthrough `0.01 @ 7k` cut needles (erank<2) **43.8 % → 9.7 %** at identical held-out PSNR/SSIM/LPIPS, identical opacity health and +3 % wall clock, and was the **only** arm that improved free-viewpoint stability (8/8 sweeps). Cheap; consider it default-ON for walkthroughs. |
| `--holdout_list` | off | File of COLMAP image names to use as the test set instead of the every-`--test_every` rule. For captures whose views come in inseparable GROUPS (the 12 rig crops of one 360 pano). WP-O3. |
| `--loss_mask` | off | Average L1 **and the per-pixel SSIM map** over the VALID mask only. Use when the mask removes a large part of the frame; `--rgb_object_masked` (which composites over black) is the WP-D1 object-refit lever and is the wrong tool there — §6. WP-O3. |
| `--geo_reg` | off | edge-aware curvature depth smoothness. **No effect on textureless mush — leave off** (§6). |
| `--lpips_lambda` | 0 | perceptual loss weight. 0=off. ~0.2 at factor≥2, gated (below). |
| `--lpips_gated` | on | **gate LPIPS toward under-resolved (high-error) crops** so it doesn't over-texture well-resolved regions. Makes LPIPS safe at any resolution. |
| `--lpips_start`/`--lpips_patch` | 3000/512 | start step / random-crop size (bounds 4K VRAM). |

**Recommended by scenario:**
- **Object/scene, 24MP stills:** factor 2, mcmc, cap 1-3M, antialiased, pose_opt,
  coarse-to-fine. (GPU_V4 run: 25.3 PSNR, 6.8 GB VRAM.)
- **4K video:** factor 1 (don't waste the 4K — see §5), cap ~3M, coarse 2→1.
- **Perceptual sharpness:** `--lpips_lambda 0.2 --lpips_gated`; on full-res
  coarse-to-fine the GUI auto-starts LPIPS in the fine phase only.

VRAM at 4K/3M ≈ 6.8 GB (peak set by the KNN init `cdist`, not training) — lots of
headroom on a 4090, but **don't spend it on more Gaussians** (§6).

---

## 4. Key learnings (evidence-based, June 2026)

- **SfM is not the bottleneck.** ALIKED+LightGlue+COLMAP registers ~100% with ~1.5px
  reproj even on hard subjects. IMC-2025 winners use the same stack.
- **MCMC densification + Mip-Splatting are the proven training-quality base.**
- **3M Gaussians is near-optimal**; the detail ceiling is the *capture*, not the
  primitive budget (§6).
- **Decoupled resolution holds:** SfM features at 1600px (COLMAP thresholds are
  ~1080p-calibrated); train 3DGS at full res. **Never super-resolve input frames**
  (per-frame SR → view-inconsistent → 3DGS averages to blur + Gaussian bloat).
- **Perceptual (LPIPS) loss helps perceived sharpness** — clean win at ≤1080p /
  under-resolved content; at full 4K a *global* LPIPS is a wash (over-textures
  well-resolved regions). **Gating it by relative crop error fixes that** → safe
  at any resolution. Judge perceptual changes by **LPIPS/crops, never PSNR**
  (perceptual loss deliberately trades PSNR).
- **Textureless/flat/dark/semi-gloss surfaces are capture-limited.** They're
  Gaussian-starved because MCMC samples relocation by **opacity only**
  (`gsplat/strategy/ops.py`: `probs = opacities.flatten()`) → drains low-error flat
  regions by design. You can't reconstruct detail the photons didn't carry; the
  fixes are capture-side (cross-pol, matte spray, RAW) or generative synthesis.
- **Diffusion enhancement = synthesis, not measurement.** Our NVIDIA Fixer (=Cosmos
  NuRec Fixer) + the Difix3D+-style distill-back loop (`core/enhancement/distill.py`)
  add *plausible* detail; reserve for cosmetic polish.

---

## 5. Video input

Pipeline expects stills. Extract frames with ffmpeg (`-r 1` ≈ 1 fps gave ~300 frames
from a 5-min 4K clip — plenty for a single-object orbit). **Train at factor 1 for 4K
source** (factor 2 throws away 75% of captured pixels). For a continuous orbit,
retrieval (NetVLAD) matching is the designed path for >300 frames.

**KNOWN BUG (TODO):** the Stage-1 sharpness auto-cull default (threshold 200) is
tuned for 24MP DSLR stills and is **wrong for video** — 4K video frames score much
lower (median ~138), so the default would cull ~74% of them, and the metric
conflates "dark/low-texture" with "blurry" (it flagged sharp-but-dark lid close-ups).
Fix: relative/data-driven threshold, or Tenengrad windowed sharpest-per-partition
selection. For now, for video, **don't auto-cull** (let COLMAP's geometric
verification drop unregisterable frames).

**Reference implementation of the correct behaviour (WP-O1, 2026-08-04):**
`core/ingest_360.py` samples video by *window-relative maximum* only — window =
`round(fps / --target_fps)`, sharpest frame per window, **no threshold at any point**
(the report even records `"absolute_threshold": null`). A regression test pins it:
`tests/test_ingest_360.py::TestSampling::test_selection_is_window_relative_not_absolute`
asserts that a globally dark/low-texture stretch still contributes frames. Stage-1's
own cull is still unfixed — see §7.

---

## 6. What was tried and did NOT work (do not repeat)

- **Raising `cap_max` above ~3M:** 8M *diverged* (PSNR 23→10 as count grew past 3M)
  and OOM'd (overlapping giant Gaussians → tile-intersection blowup, 33 GB). More
  Gaussians actively hurt this scene. 3M is near-optimal; budget is not the ceiling.
  - **RE-TESTED 2026-07-02 (WP-X1, RTX PRO 6000 96 GB, WITH erank hinge 0.01@7k +
    scale_reg 0.02, seed 0, full 30k C2F production recipe on GPU_V4; evidence:
    `_infusion/wpx1_RECORD.md` + `qa/wpx1_cap{3,4,6,8}m/scorecard.json`):
    the *divergence/OOM mechanism is FIXED* — 8M now trains to completion (no
    collapse, 13.2 GB live-tensor peak, and the *fewest* needles/SOR/giants of
    any arm) — but the **3M ceiling stands**: masked PSNR/SSIM/LPIPS degrade
    monotonically with cap (LPIPS 0.4101→0.4105→0.4166→0.4503 at 3/4/6/8M; PSNR
    −0.2/−1.0/−3.9 dB). Failure mode is now benign *dilution* (low-opacity dead
    fraction 30.6%→63.2%), not blowup. Residual 8M harm is MCMC noise injection at
    high caps: the 8M train-PSNR sag recovers +3.5 dB the moment noise stops at
    refine_stop=25000 — so a lowered `noise_lr` (not exposed by our trainer; gsplat
    default 5e5) is the one untested lever before any future >3M attempt. Verdict:
    still do not raise cap_max above 3M for quality; >3M is merely *safe* now, and
    only worth revisiting with a `noise_lr` flag or on a capture that starves 3M.
- **`--opacity_reg 0.01` (a PRODUCTION DEFAULT) on a large many-view scene: destroys the
  model, silently.** Found by WP-O3 (2026-08-05, 1 584-view 360 walkthrough; evidence
  `_infusion/wpo3_RECORD.md` §3 + `_infusion/wpo3/diag_collapse{,2,3}.json`, an 8-arm ×
  5 000-step controlled sweep). The run completes with no error, a plausible loss (0.08), a
  plausible logged PSNR and the Gaussian count pinned at the cap — and every render is a
  **flat colour**: mean opacity 2.5e-4, **99.87 % of Gaussians dead**, means blown to
  **±2.1e6** with the camera cloud normalised to radius 1. Held-out masked LPIPS 0.695.
  - **Mechanism:** `opacity_reg` presses every Gaussian's opacity down on *every* step,
    while the photometric counter-pressure only acts when that Gaussian is rendered. Our
    object-orbit captures render each Gaussian in most of ~200 views; a walkthrough with
    1 584 views renders it in a handful, so opacity loses. gsplat MCMC then injects
    **maximum** positional noise at near-zero opacity **by design**
    (`op_sigmoid(1-opacity)`, k=100, x0=0.995 — that is how dead Gaussians explore), so the
    whole cloud random-walks apart.
  - **`scale_reg 0.01` is NOT the problem** — isolated: arm G (opacity_reg 0, scale_reg
    0.01) healthy; arm H (opacity_reg 0.01, scale_reg 0) 99.9 % dead. Keep scale_reg.
  - **Fix:** `--opacity_reg 0` on such scenes. Worth **+3.34 dB / −20.5 % masked LPIPS** at
    an identical step budget.
  - **"Just use a smaller value" DOES NOT WORK — measured, WP-O5 2026-08-05.** `0.001` (10×
    below the destructive value) reaches the *same* end, roughly 10× later: **99.9 % of
    Gaussians dead and mean opacity 0.0077 by step 10 000**, ‖mean‖ p99 blown 9.0 → 36.8,
    then the rasterizer climbs **4.6 → 61.7 GB** over ~500 steps and the process dies with
    `CUDA error: an illegal memory access was encountered` (~step 11 500). Train PSNR was
    already 6 dB below the healthy arm while the loss looked ordinary. **The safe value is 0.**
  - **The crash is the mechanism, not GPU contention.** WP-O3 could only say its crashed arms
    had shared the GPU with an eval. WP-O5's arm C ran alone in a sequential sweep and died
    identically, so collapse → positional runaway → OOM-class crash is one causal chain.
  - **How to detect it:** the training log will NOT tell you. Check the checkpoint:
    `sigmoid(opacities).mean()` (healthy ≈0.3–0.6) and `means.norm(dim=1).median()`
    (healthy ≈1–2 in normalised units). `_infusion/wpo3_diag_collapse.py::ckpt_stats` does it.
  - **Scope caveat (review 2026-08-05):** the many-view generalization is a mechanism-based
    inference from ONE scene under `--strategy mcmc`; the MCMC noise-injection step is
    load-bearing in the runaway. Not yet replicated on a second capture or under
    `--strategy default` — hedge accordingly until a second scene confirms.
- **`--rgb_object_masked` with a mask that removes most of the frame:** wrong lever. It
  composites BOTH sides over black and takes a plain frame mean, so a 33 %-valid mask
  divides the photometric term by 3 (regularizers untouched → they dominate) and
  manufactures hard black edges at the mask boundary for SSIM to chase. Under live MCMC that
  compounds the opacity collapse above into a rasterizer memory blow-up (19.6 → 57.9 GB,
  `CUDA error: an illegal memory access`). It was built for WP-D1's object-extract refit,
  where the mask keeps most of the object and densification is OFF (`refine_stop_iter=1`) —
  that is the only regime it is validated in. Use **`--loss_mask`** instead (averages L1 and
  the SSIM map over the valid pixels; neither rescales the loss nor creates edges). The two
  flags are independent and must not be combined.
- **30 k steps is a step *count*, not a schedule — check your epochs.** 30 k over 1 584
  views is **19 epochs**; production object runs give each of ~200 views 100–250. Also
  ~6.5× fewer pixel-gradients than a GPU_V4 run (30 k × 0.92 MP vs 30 k × 6 MP). WP-O3 ran
  200 k (~126 epochs) — but honestly, most of that was wasted: the last 170 k steps bought
  only −5.5 % LPIPS and +0.29 dB held-out while train PSNR climbed 18.0 → 20.8 (i.e. it was
  mostly fitting the training views). Budget ~60–100 k for this shape of capture.
- **Post-hoc PRUNING to fix free-viewpoint mush: cannot work. Both variants measured.**
  WP-O5 (2026-08-05) took the WP-O3 splat the user rejected in Postshot ("hot mess… only from
  key angles everything lines up") and ablated it without retraining
  (`_infusion/wpo5_needle_probe.py`, evidence `_infusion/wpo5/needle_probe_wpo3/`):
  - **Deleting low-opacity "fog" does nothing.** 33.3 % of its 3 M Gaussians sit below
    opacity 0.05; deleting all 1 000 308 of them leaves the render **visually identical** at
    on-path, lateral-off-path and never-visited viewpoints alike. So an opacity-pruned PLY is
    a smaller file (708 → 472 MB), **not** a quality fix — do not ship one as a rescue.
  - **Deleting needles is catastrophic.** The same checkpoint is **52.4 % needles**
    (effective rank < 2; 27.8 % below 1.5; median axis aspect **6.9 : 1**), but dropping them
    dissolves the walls into confetti **even at on-path viewpoints where the full render is
    fine**. Needles here are not a contaminant on top of a good surface — they *are* the
    surface representation.
  - **Therefore the only admissible intervention is at TRAINING time** (grow fatter Gaussians:
    `--erank_lambda`), never at export time. `tools/prune_ply.py` reports
    `needle_frac_erank_lt_2` / `erank_mean` / `aspect_ratio_median` next to the opacity stats
    so this is one command to check on any checkpoint.
  - **Caveat kept explicit:** "needles cause the off-path failure" is a *hypothesis*. The
    ablation proves only that needles cannot be removed after the fact; it cannot prove they
    are the cause, because they are load-bearing where the render is good too.
- **Estimating gravity from mean CAMERA UP on a hand-held 360 stick: systematically wrong.**
  WP-O5 addendum (2026-08-05). The mean camera up is the mean *stick* direction, and an
  operator who leans the stick biases it consistently rather than randomly — measured **8.3
  deg** off on `_027`, which put the "floor" at a 2.76 m median camera height (impossible) and
  smeared the point-height histogram into a single hump with no floor/ceiling structure. Use
  instead: **normal of the plane through the capture stations** (the walking plane; check
  planarity S[1]/S[2] first) as the seed, then an **iterated RANSAC floor plane** — which put
  camera height at a plausible 2.15 m median. `tools/align_ply.py` does all three and records
  each. **Also a negative:** picking the direction that maximises point-height histogram
  peakiness wanders onto a wall/corridor axis (50 deg off, camera-height spread 8x worse).
- **Rotating a 3DGS model: the SH coefficients MUST be rotated too.** Means/scales/quats are
  the obvious ones; spherical harmonics are a function of the *world-space view direction*, so
  a global rotation rotates every Gaussian's view-dependent lobe. Skipping it leaves the model
  looking right from one direction and wrong as you orbit. No Wigner-D implementation is
  needed: each band is a rotation-closed (2l+1)-dim space, so solve `A D = B` over sampled
  directions with the basis evaluated by **gsplat's own `spherical_harmonics`** (convention
  matches by construction). `tools/align_ply.py::sh_rotation_matrix`, and
  `tests/test_align_ply.py` asserts the render CHANGES if you drop it.
- **Edge-aware depth-smoothness (`--geo_reg`):** zero effect on textureless mush —
  it's a *geometry* lever, the mush is lost *appearance* detail. Gradients verified
  to flow (not a no-op); just the wrong lever. Kept off by default.
- **Global LPIPS at full 4K:** over-textures already-well-resolved regions (keyboard
  LPIPS 0.274→0.327 @ λ0.2; PSNR −2 dB). → led to **gated** LPIPS.
- **Apple WD-R (arXiv 2603.23297):** not worth it — code unreleased (project page
  only), ships spatially-uniform (σ=4, same blind spot as global LPIPS), full-image
  4K VGG hostile to 24 GB, gains benchmarked vs blurry baselines not LPIPS-as-loss.
- **Depth/normal priors for *detail*:** they fix geometry, not albedo sharpness
  (DN-Splatter/DNGaussian treat depth↔color as a trade-off).
- **2DGS/GOF/RaDe-GS for sharpness:** over-smooth appearance (~−1 dB).
- **PlanarGS planar package at the paper's lambdas (WP-F4, 2026-07-04):** DO NOT
  copy PlanarGS's lambda VALUES (L_p 0.5 / L_rd 0.05 / L_rn 0.2) into our trainer.
  Our losses are mask-sum + median normalized (PlanarGS divides by full-frame px in
  metric depth), so the values don't transfer. A/B (Laptop 15k, dense_init+mono_depth
  base): geometry FLATNESS improved a lot (deck plane RMS −18.9%, all-planes −42.1%)
  but appearance CRASHED (masked-LPIPS +15.3%, PSNR −1.37 dB, tex-LPIPS +17.6%,
  giants/low-opa floaters up). Culprit = **L_rn**: its term is ~0.34–0.72, so ×λ0.2
  ≈ 0.07–0.14 > the whole photometric loss (~0.09) → total loss triples + PSNR crashes
  the moment L_rn activates. The lever is mechanically correct (flatness is real, code
  reviewed + unit-tested); it needs λ recalibration (λ_rn ÷~10, λ_rd ÷~2.5, keep λ_p;
  target each per-loss contribution ≲0.02) before use. Also: restrict plane masks to
  foreground/object planes (background got labeled). Evidence: `_infusion/wpf4_RECORD.md`.

---

## 7. Open items / future work

- Fix the Stage-1 video sharpness-cull bug (§5).
- ~~Validate gated LPIPS at 4K~~ DONE: 4-way 4K A/B confirms gated λ0.2 improves
  the deck (LPIPS 0.318→0.313, sharper) with the keyboard EXACTLY unharmed (0.274 =
  original), vs global λ0.2 which wrecked the keyboard (0.327). Gated LPIPS is the
  production default for the perceptual lever; deck gain is modest (capture-limited)
  but free and safe at any resolution.

### Artifact correction (June 2026)
- **DONE — faithful floater cleanup** (`core/training/prune.py`, Stage 9 GUI
  "Floater cleanup"): removes near-invisible (opacity<0.01) + spatial-outlier
  Gaussians. Validated lossless (3M→1.34M, PLY 708→316 MB, held-out PSNR/LPIPS
  unchanged). **Do this first** for any clip — faithful, no model, no Docker.
- **DONE + TESTED — native Difix3D+ backend** (`core/enhancement/difix_backend.py`
  + `difix_runner.py`, registered as `difix`): WSL-FREE (the Docker dep was the
  *separate* Cosmos NuRec Fixer). Runs as a **subprocess in an isolated conda env**
  `difix` (`conda run -n difix`) so its pinned deps stay out of the main env. Setup
  in the backend docstring; **needs `numpy<2`** (torch 2.3 `from_numpy` breaks on
  NumPy 2.x). Drops into the distill-back loop. Validated end-to-end on Windows.
  - **Empirical (laptop, 8 held-out views):** full-image LPIPS **0.366→0.350**
    (−4.5%, 7/8 views; biggest gains on under-observed orbit views) — Difix
    genuinely cleans soft/under-observed regions. BUT on the well-resolved SUBJECT
    it over-sharpens / **hallucinates**: deck LPIPS 0.256→0.279, keyboard
    0.194→0.234 (sharper but *less faithful* — looks better, measures worse).
  - **Verdict (ungated):** same lesson as global LPIPS — global enhancement can't
    tell under- from well-resolved regions.
- **DONE + TESTED — sharpness-GATED Difix (`gated`, ON by default).** Fixes the
  above: a per-pixel FreeFix-style 2D blend `final = (1-w)·render + w·difix`
  (`_gate_blend` in `difix_backend.py`). `w` is high where the render is locally
  SOFT, ~0 where it is already SHARP, with the pivot at the image's **median**
  local sharpness (Sobel-magnitude, Gaussian-smoothed σ=7) — so it is
  **resolution-robust** (no absolute threshold). This keeps Difix's gains on soft
  regions and protects the well-resolved subject.
  - **Why a sharpness gate (validated, the RIGHT metric is patch-LPIPS not L1):**
    binning 224 patches by render sharpness, Difix's LPIPS improvement is
    **+0.036 / +0.033 (Q1–Q2 softest, ~69% helped)** but **−0.014 (Q5 sharpest,
    only 20% helped)**. The "mushy" deck has HIGH local gradient (keys/labels/
    edges) → lands in the skip bucket too, so one signal guards BOTH failure modes
    (over-sharpening the keyboard AND hallucinating the deck).
  - **Empirical (laptop, 8 views) — gated beats render AND ungated Difix:**
    full-image LPIPS **render 0.366 / ungated 0.350 / GATED 0.343**. Region LPIPS
    (render | difix | gated): deck 0.256 | 0.279 | **0.254**, keyboard 0.194 |
    0.234 | **0.199** (both back to ~render fidelity), background 0.350 | 0.351 |
    **0.318** (soft-region gain kept). So Difix is now safe on by default.
  - Research basis (verified vs arXiv): FreeFix (2601.20857) per-pixel confidence
    blend; HAD (2605.16873) for the hallucination nuance. A stronger Tier-2 gate
    (alpha-accumulation + reference-coverage evidence guard) is the next step if a
    clip has truly unobserved regions; sharpness alone sufficed here. Set
    `gated=False` (or untick the Stage-10 checkbox) for the raw ungated output.
- Enhancement backends now: `difix` (native, isolated-env subprocess — **default**,
  gated, preferred WSL-free diffusion path) and `fixer` (Cosmos, Docker — avoid).

### Distill-back: transferring Difix detail INTO the PLY (v1 vs v2 — WORKING)

The gated-Difix *renders* look clean, but baking that into the 3D Gaussians is the
hard part. Full validated run on the Laptop 4K clip (base train: factor 1, MCMC
3M, gated LPIPS, 60 min):

- **v1 (conservative, `run_distill_laptop.py`):** `refine_stop_iter=1` (no
  densification) + L1+SSIM only. Geometry frozen + no perceptual loss → **the PLY
  barely changed** (held-out LPIPS only −3.6%) even though renders looked great.
  Lesson: nothing could *carry* the 2D detail into 3D.
- **v2 (aggressive, `--aggressive`) — WORKING, the recommended recipe:** *free
  appearance, constrain geometry.* MCMC relocation @ **`cap_max=3M`** (GOTCHA: the
  1M default would prune a 3M model down by ⅔ — must match current count) + **gated
  LPIPS 0.2 in the distill loss** + `geo_reg`, 3 rounds × 120 views × 8000 steps.
  Floater-safe because MCMC *at the cap* relocates spare Gaussians onto
  under-densified **real** surfaces (the "mush" is sparse-but-real, not empty
  holes) and never spawns in empty space (unlike DefaultStrategy clone/split).
  - **Results (base → v1 → v2, held-out @ factor 2):** LPIPS 0.355 → 0.342 →
    **0.324 (−8.7%, ~2.4× v1)**; SOR-outlier floaters 0.04% → 0.02% → **0.04%
    (= base, no new floaters)**; giant-blobs 8.3% → 37% → 9.9% (v1's no-densify
    L1-nudging *bloated* Gaussians; v2 far healthier); dead low-opacity 58% → 84%
    → 43% (MCMC kept Gaussians working). Visual (`compare/`): v2 deck/keys/sticker
    crisper, background floater-free.
  - **Honest caveat:** distilled models get extra same-resolution training, so
    +PSNR is inflated — judge by **LPIPS + visual + floater metrics**, not PSNR.
- **Tools:** `floater_report.py` (opacity/SOR/blob health per ckpt),
  `render_compare.py` (GT|base|v1|v2 side-by-side), `eval_compare.py` (fair
  same-res PSNR/SSIM/LPIPS).
### v3 (Gram + cap→4M growth) — TESTED, did NOT beat v2 on this scene

Built the next levers and ran them (`run_distill_laptop.py --v3`): v2 recipe +
**Gram/style loss** (`--gram_lambda 0.5`, VGG16 Gram on the shared LPIPS crop,
Difix3D+ 2503.01774) + **`cap_max` 3M→4M** (MCMC *grows* +1M Gaussians to fill
holes) + `scale_reg` 0.02. Pc-confidence-downweight (3DGS-Enhancer 2410.16266)
was deliberately NOT used — it down-weights under-reconstructed regions, which
*fights* hole-filling; floater control came from MCMC + rails + prune instead.

- **Result (held-out @ factor2): v3 REGRESSED vs v2.** LPIPS base 0.355 → v2
  **0.324** → v3 0.337 (−5.1%, *worse* than v2's −8.7%). PSNR/SSIM also down
  slightly (22.82 / 0.836 vs 23.08 / 0.838). Visual: v3 ≈ v2 but marginally
  softer.
- **Floaters stayed controlled** (goal held): v3 grew to exactly 4.0M with
  SOR-outlier 0.02% + giant 8.0% — the *cleanest* of all four, below base. So the
  +1M growth created NO floaters; it just **diluted/softened** the well-covered
  regions. Confirms the cap-divergence finding: **3M is the sweet spot; more
  Gaussians hurt** even when floater-free.
- **Why:** the laptop is a well-captured orbit with ~no real holes/extreme views,
  so the cap-raise had nothing to fill and the extra Gaussians under-trained →
  softer. **v3's levers are for UNDER-captured / holey / extreme-novel-view
  scenes**, not well-covered ones. Held-out LPIPS also can't *see* hole-filling
  (test views are well-covered), so on a sparse-capture scene v3 may still win —
  re-test there before dismissing.
- **Verdict: v2 (`--aggressive`) is the working best for well-captured clips.**
  Gram (`--gram_lambda`) + cap-raise remain available, OFF by default. Tag
  `working-distill-v2`. Untested further levers: ArtifactWorld LPIPS
  artifact-heatmap gating (2604.12251); inverted-Pc enhance-where-coarse.
  MCMC is floater-resistant by construction (2404.09591).
- Re-pull `nvidia/Fixer` V2 weights; run the distill-back loop (needs Docker up).
- Optional installs (research-only, not yet wired): RoMa v2 dense rescue
  (`core/roma_rescue.py`), Faster-GS backend (`docs/research-findings.md` has setup).
- Capture playbook for hard surfaces: cross-polarization, matte/scanning spray, RAW
  expose-for-shadows — highest ROI for textureless/glossy subjects.
- Watch list (no code yet): GloSplat, SSR-GS, 3DGRUT/3DGUT (Blackwell-era).

---

## 8. Versions (June 2026)
gsplat 1.5.3 (AOT) · pycolmap 4.0.4 · torch 2.3.0+cu121 · HLoc (ALIKED-n16 +
LightGlue) · BiRefNet + SAM 2.1 · Gradio 5.x · Python 3.11.5 (anaconda base).
