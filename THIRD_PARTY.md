# THIRD_PARTY.md — provenance ledger

Lightweight component → source → license ledger for the future funded rebuild
(POC stance: licenses are NOT a selection filter; this exists purely so a
commercial port knows exactly what to re-license or re-implement). See
`docs/EXECUTION_PLAN.md` §0 for the stance.

Format: what we ship · where the idea/code came from · how we adopted it · license of the source.

---

## WP-Q2 — Quick trainer wins (added 2026-07-01)

### (a) Effective-rank anti-needle loss — `core/training/gsplat_trainer.py::erank_loss`
- **Idea/source:** "Effective Rank Analysis and Regularization for Enhanced 3D
  Gaussian Splatting", Hyung et al., **NeurIPS 2024, arXiv:2406.11672**.
- **Adoption:** **RE-IMPLEMENTED from the paper (Eq. 7–10)**, cross-checked against
  the official repo snapshot in `tools/reference_sources/erank_gs/`
  (`utils/get_stats.py::get_effective_rank`, `train.py` erank_loss line). Our
  version adds `.clamp_min(1e-30)` on the eigenvalue sum and `.clamp_min(1e-12)`
  inside `log(p)` — these are OURS, needed because gsplat MCMC relocates dead
  Gaussians to log-scale ≈ −inf whose `exp` underflows to 0 and would NaN the
  entropy. `w_thin` stays 0 (MCMC `scale_reg` already shrinks size). No source
  code copied — a ~10-line pure-PyTorch function.
- **Source repo license:** the official erank_gs repo is built on the Inria
  GRAPHDECO 3DGS base (`loss_utils.py` carries the Inria "free for non-commercial,
  research and evaluation use" header). Since we re-implemented the method rather
  than vendoring the code, the funded port only needs to re-derive from the paper
  (method itself is not the licensed artifact). **Re-license/re-derive: trivial.**

### (b) SA-GS eval-time eps2d anti-aliasing rule — `compute_eps2d` + `render_splats(eps2d=)`
- **Idea/source:** "SA-GS: Scale-Adaptive Gaussian Splatting for Training-Free
  Anti-Aliasing", Song et al., **arXiv:2403.19615**.
- **Adoption:** **IDEA ONLY.** SA-GS ships a custom CUDA rasterizer; we do NOT use
  it. gsplat already exposes the exact same 2D screen-space dilation knob
  (`rasterization(eps2d=...)`, default 0.3), so we implement the scale-adaptive
  rule as a ~2-line pure-Python helper `compute_eps2d(...) = 0.3 * r²` and pass it
  through at zoom/eval render time (+ optional `eval_ss` supersampling). No SA-GS
  code copied. Reference snapshot (their CUDA + train scripts, **Apache-2.0**) at
  `tools/reference_sources/sags/` was consulted for the formula only.
- **Source repo license:** Apache-2.0. **Re-license: none needed** (we wrote our own
  code; the idea is unpatentable math over gsplat's existing API).

---

## WP-Q1 phase 2 — external QA scorers (added 2026-07-02)

Clones live at `D:\REPOS\HYPERclone\qa_repos\`; pinned/shadowing deps live in the
`D:\REPOS\HYPERclone\venvs\qa` venv (never the main env). Integration code is
ours under `core/qa/scorers/` + `core/qa/venv_drivers/` + `core/qa/vendor/`.

**Pinned clone commits (verified 2026-07-02):**
| repo | commit |
|---|---|
| ActiveVisionLab/CrossScore | `01cc66c8459b41ae86858e6663da267a39a319a5` |
| stootaghaj/NOVA | `e01be9ad6ea7a2895b355dc232bc9df2b5129a03` |
| VincentQQu/NVS-SQA | `8565fc1a921fcaf7df1108ac20f2f5b3bb09add9` |

**Local patches to the CrossScore clone** (exist as uncommitted working-tree edits
there; committable copies live in `tools/patches/`). After any re-clone, re-apply:
```
git -C D:/REPOS/HYPERclone/qa_repos/CrossScore apply \
    D:/REPOS/HYPERclone/apex-sfm/tools/patches/crossscore_mkdir.patch \
    D:/REPOS/HYPERclone/apex-sfm/tools/patches/crossscore_winpath.patch
```
- `crossscore_mkdir.patch` — `utils/io/images.py::metric_map_write` mkdir-before-write.
- `crossscore_winpath.patch` — `utils/io/score_summariser.py` Windows `\\`→`/` path
  normalisation (without it the summariser IndexErrors and no CSV is written).
NOVA and NVS-SQA clones are UNMODIFIED (drivers import them read-only).

### CrossScore — `core/qa/scorers/crossscore.py`
- **Source:** ActiveVisionLab/CrossScore (ECCV'24), BSD-3-Clause. Cloned as-is;
  ckpt `CrossScore-v1.0.0.ckpt` (123 MB in-repo LFS) used unmodified.
- **Adoption:** RUN AS-SHIPPED via its Hydra CLI `task/predict.py` (subprocess to
  qa venv). We stage renders (query) + real train photos (reference) into the deep
  dir layout its summariser parses and read the per-frame mean predicted-SSIM CSV.
  Determinism overrides: `this_main.force_batch_size=True`,
  `data.neighbour_config.deterministic=True` (default picks 5 RANDOM refs),
  `...write.config.score_map_colour_mode=gray`, `trainer.precision=32-true`.
  Uses `Dinov2Model.from_pretrained` (HF download, cached). We DISABLE the batch
  image/map writer (`...write.flag.batch=False`) — it mirrors the query subpath into
  many leaf dirs it never mkdir's (FileNotFoundError) and we don't need it: the
  machine-readable per-frame mean SSIM comes from the summariser CSV, which runs
  unconditionally. (A one-line `os.makedirs` was also added to the clone's
  `utils/io/images.py::metric_map_write` so per-pixel gray maps CAN be re-enabled;
  finishing the other writer paths' mkdirs is left as a nicety.) The issue-#3
  `utils/misc/image.py` img_std slice is LEFT ALONE
  (it only de-normalises images for the vis PNGs; the machine-read score_map means
  come straight from model output, so HEAD↔released-ckpt stay consistent).
- **Re-license:** BSD-3 (permissive); re-derive: re-clone.

### NOVA — `core/qa/scorers/nova.py` + `venv_drivers/nova_driver.py`
- **Source:** stootaghaj/NOVA (WACV'26), MIT. Cloned as-is; ckpt `NOVA_NVS.pt`
  (661 MB LFS, dict `base`+`embedding_head`) used unmodified.
- **Adoption:** import NOVA's own `load_model` + `compute_cosine_distance` from its
  `nova.py` in the qa venv (timm DINOv2). Our driver FAILS HARD if the ckpt is
  missing (NOVA silently degrades to vanilla DINOv2) and patches `torch.load` to
  `weights_only=False` for the trusted dict. Reference = nearest real photo by
  COLMAP pose distance excluding self (non-aligned).
- **Re-license:** MIT.

### NVS-SQA — `core/qa/scorers/nvs_sqa.py` + `venv_drivers/nvs_sqa_driver.py`
- **Source:** VincentQQu/NVS-SQA (TPAMI), CC0. Cloned as-is; 18 MB in-repo ckpt.
- **Adoption:** we import ONLY `models.sequence_models` (bypassing
  `utils.train_utils`, which drags in `qa_methods`→`skvideo` — a training-only dep
  that breaks under numpy 2) and replicate the encoder-load path. The 385→1 head
  was a pickled sklearn Ridge (sklearn<1.7 pickle); we loaded it ONCE and extracted
  `coef_` (384) + `intercept_` (15.164) — VENDORED at
  `core/qa/vendor/nvs_sqa_linear_head.npz` (in apex-sfm, so a re-clone of the
  NVS-SQA repo cannot lose it) — so runtime is pure numpy
  (`jod = feat@coef + intercept`): no sklearn dep, fully deterministic. JOD is
  within-scene-delta only.
- **Re-license:** CC0 (public domain).

### TSED — `core/qa/vendor/tsed_epipolar.py` (VENDORED ~100 lines)
- **Source/idea:** Photoconsistent-NVS (yorkucvil), MIT — Thresholded Symmetric
  Epipolar Distance.
- **Adoption:** RE-IMPLEMENTED from the method (SIFT + Lowe ratio + median symmetric
  epipolar distance), NOT vendored code: their pose plumbing is c2w OpenGL + hard
  256² intrinsics; ours is w2c OpenCV with real per-view K. F comes entirely from
  known poses (`F = K2^-T [t_rel]_x R_rel K1^-1`, no axis flip). numpy+opencv only,
  runs in the main env. Threshold per plan: ≥10 matches AND median < 2px·(H/256).
- **Re-license:** trivial (method re-derive).

### Temporal flickering — `core/qa/vendor/vbench_flicker.py` (VENDORED ~60 lines)
- **Source:** Vchitect/VBench `vbench/temporal_flickering.py`, Apache-2.0.
- **Adoption:** VBench is NEVER pip-installed (pins numpy<2 + transformers 4.33).
  We copy the one metric verbatim in spirit: `(255 - mean consecutive-frame
  MAE)/255` (cv2.absdiff), higher=less-flicker. STATIC-camera only — our scorer
  wraps it with a hard guard that refuses moving-camera clips. main env, numpy+cv2.
- **Re-license:** Apache-2.0.

### RAFT warp-error — `core/qa/scorers/raft_warp.py`
- **Source:** torchvision `raft_large(weights=Raft_Large_Weights.DEFAULT)` (~85 MB
  auto-download), BSD. Used as the flow model; the warp-error metric (forward-
  backward occlusion mask + masked L1) is ours. main env, torchvision 0.26.
- **Re-license:** BSD (torchvision).

### qa venv deps (venv-local, shadowing — never in main env)
- Added: `timm`, `lightning` (2.5.6), `hydra-core` 1.3.2, `omegaconf` 2.3.0,
  `wandb`, `torchsummary`. Inherited from main env via `--system-site-packages`:
  torch 2.11.0+cu128, torchvision 0.26, transformers 4.57.3, numpy 2.4, opencv,
  sklearn 1.8 (used ONCE offline to extract the NVS-SQA head, not at runtime).
  NOTE: the plan's precautionary transformers~4.4x / sklearn<1.7 pins proved
  UNNECESSARY — CrossScore's `Dinov2Model.from_pretrained` and NOVA's timm path
  run clean under transformers 4.57.3, and the Ridge head was extracted to numpy
  so no sklearn pickle is unpickled at runtime.

---

## WP-F1 — MoGe-2 metric depth+normal prior cache (added 2026-07-02)

### MoGe-2 — geometric prior (depth + normals) — `core/priors.py`
- **Idea/source:** "MoGe-2: Accurate Monocular Geometry with Metric Scale and
  Sharp Details", Wang et al., **Microsoft**. Model `Ruicheng/moge-2-vitl-normal`
  (metric depth + camera-frame normals).
- **Adoption:** **USED AS A LIBRARY (weights + code), UNMODIFIED.** Installed via
  `pip install git+https://github.com/microsoft/MoGe.git` into the main env
  (pure PyTorch/SDPA, keeps its pinned `utils3d`; verified torch 2.11.0+cu128 and
  numpy 2.x untouched post-install). We call `MoGeModel.infer(image, fov_x=...,
  resolution_level=9, use_fp16=True, apply_mask=True)` per view; everything
  downstream (scale-first metric alignment against COLMAP sparse points, RANSAC +
  weighted-LS refit, per-view/global mode selection, conf + inverse-Canny edge
  masks, the aligned prior cache layout) is **OURS**. API snapshot consulted at
  `tools/reference_sources/moge/`.
- **License:** **MIT** (code + weights) — permissive, no re-license needed for the
  funded port; a commercial rebuild can ship the weights directly.
- **WP-N1 addition (2026-07-05, OUR code, no new component):** `core/priors.py` gained
  a default-OFF `--depth_only` build mode (drops per-view normal/img retention + skips
  the normal/conf/edge_mask writes) so a **native factor-1** cache fits disk/RAM
  (~256 GB → ~51 GB, ~400 GB → ~68 GB RAM) for mono-depth-only runs. Same MoGe-2
  library call, unchanged; only our cache builder changed.
- **WP-T2 (2026-07-08, NO new third-party component):** reused the `--depth_only` mode to build a
  **factor-2** cache for the Sony shoe native-detail variant (588/588 reliable, median rel-residual
  0.0146 — same alignment quality as factor-3). Also reused (all already logged): gsplat MCMC + erank,
  dense_init RoMa sparse (WP-F2), gated-LPIPS, BiRefNet delivery masks, and the SplatSuRe
  view-count/ρ diagnosis machinery (`core/qa/visibility.py`). New code is OURS/analysis-only
  (`_infusion/wpt2_{diagnose,regions,gap}.py`, `_dataset/SONY_TEST/{downsample_f2.py,train_sony_t2.sh}`).
- **Alignment references (idea only, no code copied):** DN-Splatter
  `align_depth.py` (closed-form scale/shift; the COLMAP reproj/track/depth filter
  thresholds) and the Inria hierarchical-3DGS `make_depth_scale.py` (median-ratio
  scale) — snapshots at `tools/reference_sources/{dnsplatter,gs}/`. DN-Splatter is
  Apache-2.0; the Inria script carries the GRAPHDECO non-commercial research
  header (**re-derive from first principles for a funded port — the alignment is
  standard robust least-squares**).

### Dense mono-depth loss consumer — `gsplat_trainer.py` (`--mono_depth_lambda`)
- **Idea/source:** standard monocular-depth supervision for 3DGS (DN-Splatter /
  MonoSDF-style aligned depth L1). **Written from scratch** as a masked L1 between
  the aligned MoGe prior depth and the rendered expected-depth, normalized by the
  median prior depth (scale/resolution invariant, matching the existing sparse
  `--depth_lambda`). No external code. Default OFF.

---

## WP-F2 — EDGS-style dense-matcher triangulation init (added 2026-07-02)

New module `core/dense_init.py`; HLoc additions `hloc/matchers/roma.py` + a `roma`
entry in `hloc/match_dense.confs`; a `--dense_init` opt-in in
`core/training/gsplat_backend.py::prepare_data`. Default OFF (production unchanged).

### RoMa dense matcher — pip `romatch` — `hloc/matchers/roma.py`
- **Idea/source:** "RoMa: Robust Dense Feature Matching", Edstedt et al.,
  **CVPR 2024, arXiv:2305.15404**. Used as EDGS's dense correspondence engine.
- **Adoption:** **USED AS A LIBRARY (weights + code), UNMODIFIED.** `pip install
  romatch` (v0.1.2) into the main env with the torch constraints file
  (`tools/blackwell_session_2026-06/constraints.txt`) — torch 2.11.0+cu128 /
  torchvision 0.26 / numpy 2.4.3 verified untouched post-install. We call
  `roma_outdoor(device=...)` (weights auto-download via torch.hub: DINOv2 ViT-L/14
  + Parskatt release, ~1.1 GB), then `.match()` / `.sample()` /
  `.to_pixel_coordinates()`. Our thin `hloc/matchers/roma.py` wrapper (~90 lines,
  OURS) adapts RoMa into hloc's `match_dense` matcher interface; everything else
  (pairs, triangulation, merge) is our pipeline.
- **License:** romatch code **MIT**. NOTE for the funded port: RoMa's DINOv2
  backbone weights carry Meta's DINOv2 license (permissive research; review before
  commercial use). POC stance: not a filter.
- **Env side effect (recorded + PINNED):** installing romatch pulled
  `albumentations` → `opencv-python-headless 5.0.0.93` ALONGSIDE the pre-existing
  `opencv-python 4.13.0.92`. With both wheels present, whichever installed LAST
  wins `import cv2` — an install-order-dependent outcome, so nothing guaranteed
  the validated build stayed active. We validated the trainer/hloc undistort APIs
  (`getOptimalNewCameraMatrix`, `initUndistortRectifyMap`, `remap`) under the
  headless 5.0.0 build and **pinned `opencv-python-headless==5.0.0.93` +
  `romatch==0.1.2` in `requirements.txt`** so a fresh install reproduces the
  validated resolution instead of silently flipping. Also pulled `timm 1.0.27`,
  `poselib`, `loguru`, `wandb`, `simsimd` into the main env. A funded rebuild
  that wants a clean env can `pip install romatch --no-deps` and supply only
  `einops`/`kornia` (already present).

### EDGS sampling parameters — provenance (idea only, NO EDGS code run/copied)
- **Source:** CompVis/EDGS `source/corr_init.py` (snapshot at
  `tools/reference_sources/edgs/source_corr_init.py`). EDGS is a graphdeco-based
  trainer with in-memory-only output (no points3D/PLY export, issue #15), numpy<2
  pin, unresolved Windows issue #9, and an H/W-swap triangulation bug (#14) — a
  poor technical fit, so we did NOT run or vendor it.
- **What we took (parameters/method, re-derived):** match at RoMa's coarse
  resolution with `upsample_preds=False` (560×560, fp16); certainty sample
  threshold **0.05** (= RoMa `sample_thresh`, EDGS's `upper_thresh`); sample
  10k–20k correspondences/pair (`init_wC.matches_per_ref=15_000`); color from the
  matched pixel → SH DC; isotropic scale ≈ dist·`scaling_factor` (EDGS
  `scaling_factor=0.001`). We DIVERGE from EDGS on the geometry backend: instead of
  EDGS's per-pixel in-Python `triangulate_points` (which yields ~3M points but ships
  the #14 H/W bug and no COLMAP export), we triangulate through **hloc + pycolmap**
  against the existing poses (poses preserved, epipolar-verified, proper COLMAP
  `points3D` with tracks) and merge with the original ALIKED sparse. Yields ~0.6–1.2M
  points — fewer than EDGS's ~3M but a valid, exportable, multi-view-consistent cloud
  that MCMC re-densifies.
- **License:** EDGS code is **research/non-commercial** (graphdeco base). Since we
  re-derived the parameters and used our own hloc/pycolmap geometry, the funded port
  owes nothing to EDGS's code — re-derive the sampling constants from the paper.

### Selectivity — CoMapGS finding (idea only)
- **Source/idea:** CoMapGS (co-visibility map guided 3DGS) — random point injection
  HURTS (−0.2..−0.7 dB); density should be added SELECTIVELY in under-covered regions.
- **Adoption:** we keep ALL triangulated points by default but (a) REPORT per-region
  density (`_infusion/wpf2_density_eval.py`) and (b) expose a `low_density_only`
  flag that restricts EXTRA seeding to coarse cells the original sparse cloud
  under-covers. No CoMapGS code used.

### hloc patches (committable copies in `tools/patches/`)
The `Hierarchical-Localization` clone is an editable install outside apex-sfm git.
After any re-clone, re-apply:
```
git -C D:/REPOS/HYPERclone/Hierarchical-Localization apply \
    D:/REPOS/HYPERclone/apex-sfm/tools/patches/hloc_match_dense_roma_conf.patch \
    D:/REPOS/HYPERclone/apex-sfm/tools/patches/hloc_roma_matcher.patch
```
- `hloc_roma_matcher.patch` — new file `hloc/matchers/roma.py` (RoMa matcher).
- `hloc_match_dense_roma_conf.patch` — adds the `roma` conf to `match_dense.confs`
  (`max_error: 2, cell_size: 4`, RGB preprocessing).
No other hloc files are modified (the DataLoader worker count is overridden at call
time from `core/dense_init.py`, not patched, to keep hloc edits minimal).

---

## WP-C1 — GaMO-pattern gated generative completion (added 2026-07-02)

### GaMO opacity-mask latent-gate mechanism — `core/enhancement/latent_gate.py`
- Source: arXiv 2512.25073 (GaMO) — mechanism re-implemented clean-room from the
  verified description in EXECUTION_PLAN WP-C1 (~50 lines; no GaMO code copied —
  their repo is not cloned on this machine).
- License: n/a (mechanism only); our implementation.
- Deviation: RePaint 3-iteration resample harmonization approximated by SDXL's
  iterative refinement + trigger-point injection + feathered paste-back
  (`callback_on_step_end` cannot re-run a sampler step).

### SDXL-Inpaint backbone — WSL runner `_infusion/gamo_fill.py`
- Model: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (HF, ungated).
- License: **OpenRAIL++** (use-restriction, NOT permissive) — fine for the POC per
  the §0 stance; the funded rebuild should swap the backbone (gate is agnostic).
- Weights: WSL `~/.cache/huggingface/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1` (~20 GB).
- Env: WSL `~/artifixer-venv` (diffusers 0.37.1, torch 2.11.0+cu128) — no new env.
- **Known-region injection timestep convention** (WP-C1 closeout M2 fix): our GaMO
  `callback_on_step_end` now noises the known latent to `timesteps[i+1]` (the noise
  level of the post-`scheduler.step` latents), matching diffusers'
  `StableDiffusionXLInpaintPipeline` own known-region re-injection
  (`pipeline_stable_diffusion_xl_inpaint.py`: `noise_timestep = timesteps[i+1]`).
  Convention adopted from the diffusers source (Apache-2.0); ~2-line indexing fix,
  no code copied.

---

## WP-F4 — Planar-prior losses (PlanarGS port, added 2026-07-03)

### Co-planarity / prior-depth / prior-normal losses + plane-mask builder
- **What we ship:**
  - `core/training/gsplat_trainer.py::co_planar_depth` + `normal_from_depth` and
    the three losses `--planar_lambda` (L_p), `--prior_depth_lambda` (L_rd),
    `--prior_normal_lambda` (L_rn) — all default OFF.
  - `core/plane_masks.py` — per-view plane-instance label maps
    (`priors/plane_mask/{stem}.npy`) via the geometric-inspection recipe.
- **Idea/source:** "PlanarGS: ..." — **github.com/SJTU-ViSYS-team/PlanarGS**
  (cloned to `D:\REPOS\HYPERclone\PlanarGS`, reference-only, not run). Ported:
  `planar/co_planar.py::ProjectDepth` (regularized LS plane fit
  `param=(PPᵀ+0.01I)⁻¹P·1`, plane depth `1/(paramᵀK⁻¹ũ)`),
  `common_utils/graphics_utils.py::{Depth2Pointscam, pcd2normal, NormalFromDepth}`,
  and the `train.py` loss wiring (L_p masked-L1; L_rd = conf×canny×depth_weight
  gated depth loss; L_rn = L1 + (1−cos) of depth-derived vs prior normal). Lambdas
  0.5 / 0.05 / 0.2 and the 7k/14k/20k stagger are PlanarGS's `arguments/__init__.py`
  defaults.
- **Adoption / deviations (POC stance — copied freely, adapted to our stack):**
  - Base trainer is **gsplat MCMC (ours)**, not PlanarGS's graphdeco + custom
    `diff-plane-rasterization`. So L_dn (needs a rasterized Gaussian-normal map)
    is **SKIPPED**; L_rn instead compares the prior normal to a **depth-derived**
    normal (`normal_from_depth`, = their pcd2normal) — no extra render. PlanarGS's
    `lambda_scale=100` min-scale flatten is **SKIPPED** (our erank hinge covers
    anti-needle; aggressive flatten biases MCMC relocation).
  - Geometric prior is **our MoGe-2 cache** (WP-F1), not their DUSt3R
    `run_geomprior.py`. Plane masks come from the **MoGe-2 normal+depth cache**
    (K-means k=5 normal split → 8-conn CC → 0.05 plane-residual prune →
    largest-CC, min 2000 px AND ≥0.5% frame), NOT from SAM-2.1 instances
    (production plane masks were never preserved; running SAM AMG fresh would
    contend with the GPU and still need this same inspection). Disjoint clusters
    → no overlap/containment merge needed (a documented simplification).
  - Depths are normalized-world units (arbitrary scale), so L_p/L_rd are
    **median-normalized** (their lambda stays scene-scale-invariant, matching our
    existing sparse/mono depth losses). MoGe normals point toward the camera,
    matching pcd2normal — verified, **no sign flip**.
  - **Mask normalization deviation:** all three losses divide by their *active
    mask pixel count* (`m.sum()`), whereas PlanarGS's `loss_utils.l1_loss/l2_loss`
    divide by the *full-frame* pixel count. Ours is coverage-invariant (planar
    coverage varies a lot per view — median 58.7% of frame); combined with the
    median depth-normalization this means the PlanarGS lambda VALUES (0.5/0.05/0.2)
    are re-interpreted, not transferred 1:1 — the A/B gate is what calibrates
    effective strength (tune if L_p fights MCMC).
  - **Border deviation fixed (post-review):** we zero the 8-px border in the loss
    mask too (PlanarGS zeros it in-place inside `co_planar`, so its `planar_mask`
    drops it) — keeps numerator/denominator consistent. Degenerate-denominator
    guard is sign-PRESERVING (a fixed +eps would map grazing rays to ~1e6).
  - L_p fit is kept **attached** by default (faithful to PlanarGS, which does not
    `.detach()`); `--planar_detach` exposes the detached variant.
- **Source repo license:** MIT (`PlanarGS/LICENSE`). Permissive; the funded port
  can vendor the ported functions directly. (POC stance: license was not a factor.)

---

## Pre-existing components (recorded for completeness — not added by WP-Q2)
- **gsplat** v1.5.3 (AOT sm_120) — nerfstudio-project/gsplat — Apache-2.0. Core rasterizer/MCMC.
- **Mip-Splatting** antialiased mode — via gsplat `rasterize_mode="antialiased"` — used as-is.

---

## WP-S1 — Stacked-levers production run (2026-07-04)
No new third-party component. WP-S1 is a production run that combines four already-logged
levers on the user's two real captures: RoMa dense-init (`romatch` MIT, logged under WP-F2),
MoGe-2 mono-depth prior (`Ruicheng/moge-2-vitl-normal` MIT, logged under WP-F1), the
effective-rank hinge (paper 2406.11672 / erank_gs, logged under WP-Q2), and gsplat's gated
LPIPS (torchmetrics VGG). Eval reuses the WP-Q1 6-scorer stack (crossscore/nova/tsed/
raft_warp/nvs_sqa/flicker) — all previously logged. Evidence: `_infusion/wps1_RECORD.md`.

---

## WP-G1 — Glossy/specular lever, phase-1 feasibility probe (2026-07-04)
Ran the UPSTREAM 3DGS-DR trainer (NOT ported into gsplat) on our two scenes to measure
the specular ceiling before deciding on a gsplat port.
- **3DGS-DR** ("3D Gaussian Splatting with Deferred Reflection", SIGGRAPH'24, arXiv 2404.18454)
  — repo `gapszju/3DGS-DR` @ commit `4077f1d4b45891885e6ebd1089da7bc51bee8afc` (cloned to
  `D:\REPOS\HYPERclone\3DGS-DR`). Deferred shading: per-pixel normal propagation + a learned
  cubemap environment map + deferred specular composite `(1-refl)*base + refl*env(reflect(view,n))`.
  License: Inria "Gaussian-Splatting License" (non-commercial research/eval), inherited from the
  graphdeco 3DGS base it forks. Used as a separate upstream trainer; nothing vendored into
  `apex-sfm/` yet (phase-2 port would re-implement the losses/kernels in gsplat).
  - **CUDA submodules built AOT for sm_120** (house recipe: VS2022 vcvars64, CUDA v12.9,
    `TORCH_CUDA_ARCH_LIST=12.0`, `--no-build-isolation`, constraints — torch 2.11.0+cu128
    untouched): `diff-gaussian-rasterization_c7` (7-channel deferred rasterizer),
    `diff-gaussian-rasterization_c3` (vanilla base pass), `cubemapencoder` (env map),
    `simple-knn`. Build recipe: `_infusion/wpg1_build.bat`.
  - **sm_120 / torch-2.11 build fixes** (recorded so the funded port can re-apply): (1) removed
    `.data<T>()` → `.data_ptr<T>()` (removed API); (2) the `.cu`-reachable headers included the
    full `<torch/extension.h>`/`<torch/torch.h>`, which under nvcc+MSVC pulls
    `torch/csrc/dynamo/compiled_autograd.h` → `C2872 'std' ambiguous` — replaced with a
    lightweight pure-ATen shim (`#include <ATen/ATen.h>` + `<ATen/Dispatch.h>` +
    `namespace torch { using namespace at; }`) in the 6 nvcc-compiled files, keeping
    `<torch/extension.h>` only in the cl-compiled pybind `.cpp` (mirrors gsplat's own pattern);
    (3) `torch::kFloat32`/`kInt32` (torch-C++-API aliases, not in ATen) → `torch::kFloat`/`kInt`;
    (4) made `open3d` import optional in `utils/general_utils.py` (used only by an unused
    PLY-writer). No torch/CUDA reinstall.
- **Provenance of inputs:** trained on our exported undistorted factor-2 PINHOLE frames
  (`_infusion/wpg1_export_colmap.py` reuses our own `load_colmap_data`+`load_image` so GT is
  byte-identical to the WP-S1 stack eval). Init = sparse ALIKED COLMAP points (3DGS-DR's
  designed regime). Evidence: `_infusion/wpg1_RECORD.md`.

---

## WP-S2 — Object-mode background study, object masks (2026-07-04)

### BiRefNet object matting — `core/masking.py::run_birefnet_batch` (pre-existing masking engine)
- **Idea/source:** "Bilateral Reference for High-Resolution Dichotomous Image Segmentation",
  Zheng et al., **CAAI AIR 2024, arXiv 2401.03407**. Weights: HuggingFace
  **`ZhengPeng7/BiRefNet`** ('general' variant, 1024×1024 input), loaded via
  `transformers.AutoModelForImageSegmentation(..., trust_remote_code=True)` (the repo ships its
  own `birefnet.py`/`BiRefNet_config.py` modeling code, auto-downloaded to the HF cache).
- **Adoption:** used AS-IS through the project's existing headless masking engine
  (`core/masking.py`, already in the tree) to auto-matte the GPU_V4 object on the factor-2
  training frames — the same BiRefNet auto-matte the GUI masking stage exposes. Soft alpha,
  threshold=0, erosion=0. No model code copied; weights + remote modeling code fetched from HF.
- **License:** BiRefNet code **MIT** (github.com/ZhengPeng7/BiRefNet); the `ZhengPeng7/BiRefNet`
  checkpoint is released for research use. (POC stance: not a filter — logged for the funded
  rebuild, which may swap in a commercially-cleared matting model.)
- **Note:** SAM 2.1 (the masking engine's interactive-correction path) is **not installed** in the
  Windows env, so BiRefNet auto-matte was the sole mask source (quality spot-checked across the
  orbit — `qa/wps2_mask_overlays/`). Evidence: `_infusion/wps2_RECORD.md`.

### `--mono_depth_object_masked` trainer lever (new, default OFF)
- **What we ship:** `core/training/gsplat_trainer.py` + `gsplat_backend.py` — an object-mode
  toggle that restricts the WP-F1 mono-depth loss to inside the object mask (`pv &= mask>0.5`),
  so object-only training never anchors Gaussians to background depth. OURS (~4 lines); no
  external source. Default OFF → production (full-frame mono-depth) unchanged.

## WP-D1 — Delivery pass (+ WP-S2 synthesis: delivery-time object extraction, 2026-07-04)

### PlayCanvas splat-transform — the delivery filter/decimate/SOG-compress CLI
- **Idea/source:** **`@playcanvas/splat-transform` v2.7.1** (npmjs.com/package/@playcanvas/splat-transform,
  github.com/playcanvas/splat-transform), the reference PlayCanvas/SuperSplat 3DGS delivery tool.
- **Adoption:** used AS-IS as an external Node CLI (`npm install -g`, node ≥18), shelled from
  `core/delivery.py::run_splat_transform`. Actions used: `--filter-nan`, `--filter-floaters`
  (voxel edge scaled per-scene to world units = `floater_frac × robust_scene_diagonal`),
  `--filter-cluster --seed-pos <object-center>` (keep the connected object cluster),
  `--decimate N%` (pairwise-merge simplify — KEEPS N%), `--filter-harmonics 2` (drop SH>2 at
  final export), `--morton-order`, and SOG compression (`-i` iters). GPU (WebGPU) path used for
  filter-floaters/-cluster/SOG on the Blackwell adapter (`-g 0`); `-g cpu` fallback documented.
  No code copied; invoked as a subprocess. Verified WebGPU works on sm_120 (3 NVIDIA adapters
  enumerated via `--list-gpus`).
- **License:** **MIT** (github.com/playcanvas/splat-transform). Output SOG / compressed-PLY open
  in SuperSplat (supersplat.play.canvas.com) and any PlayCanvas engine viewer.

### `core/delivery.py` — delivery orchestrator + delivery-time object extraction (OURS)
- **What we ship:** a single-file CLI (default full-scene delivery; `--extract_object` opt-in)
  that chains prune → splat-transform filters/decimate → short self-refit → SOG export → QA
  scorecard. The object-extraction mode selects object Gaussians by **multi-view BiRefNet-mask
  voting** on projected Gaussian centers. OURS; the projection loop reuses the
  `fully_fused_projection(eps2d=0)` machinery from `_infusion/val_splatsure.py` /
  `core/qa/visibility.py`, scene scale from `core/qa/scorers/common.py::robust_scene_scale`,
  object center from `core/enhancement/coverage.py::object_center_from_cameras`, floater cleanup
  from `core/training/prune.py`. No external code.

### `--rgb_object_masked` trainer lever (new, default OFF)
- **What we ship:** `core/training/gsplat_trainer.py` — for the delivery object-extract self-refit,
  composites BOTH render and GT over black by the object mask before the L1+SSIM loss, so a short
  refit of an already-extracted object is not pulled by background pixels. OURS (~4 lines); no
  external source. Default OFF → production photometric loss unchanged.

---

## WP-P1 — Generative-polish bake-off (2026-07-04/05)

Bake-off of released enhancers to polish the WP-S1 stack. All ran as SEPARATE upstream tools
(nothing vendored into `apex-sfm/`); each got its own isolated venv so torch 2.11 is never
clobbered. Verdict: none beat the no-op splat (see `_infusion/wpp1_RECORD.md`).

### GeoQuery — 2D geometry-guided diffusion refiner (measured; NOT adopted)
- **Source:** "GeoQuery: ..." SIGGRAPH 2026, **arXiv 2605.12399**, github.com/Xiaoc7/GeoQuery
  (cloned `D:\REPOS\HYPERclone\GeoQuery`, shallow @ latest). **License: MIT.** Weights
  `DIG-UESTC/GeoQuery` `geoquery_window_size3_down1.pkl` (5.2 GB) + auto-pull `stabilityai/sd-turbo`
  (~5 GB; the `.pkl` is a LoRA/skip delta over sd-turbo's VAE+UNet, so sd-turbo is a hard runtime dep).
- **Adoption:** used AS-SHIPPED as a 2D refiner (`src.model.GeoQuery.sample`) driven through OUR distill
  machinery (rendered targets + nearest-train reference + MoGe depth + poses -> refined image -> resume-
  train our gsplat stack), NOT its bundled gsplat example (avoids its pycolmap-fork + fused-ssim Windows
  builds and guarantees normalization parity with our ckpt). No GeoQuery code modified. Our drivers
  (`_infusion/wpp1_geoquery_*.py`) are ours.
- **Env (venv `venvs/geoquery`, --system-site-packages, torch 2.11 untouched):** diffusers==0.25.1
  (pinned - its UNet2DConditionModel structure must match the `.pkl` state_dict), transformers==4.38.0,
  peft==0.9.0, huggingface-hub==0.25.1, **cupy-cuda12x==13.4.1** (its `softsplat.py` forward-warp needs
  cupy; cupy 14.x fails via a stray `Python311\cuda.py` shadowing the `cuda` namespace pkg -> pin 13.4.1),
  `xformers` stripped from requirements (unused on the refiner path). **Re-license:** MIT (permissive);
  sd-turbo weights carry an SAI community license (not a POC filter).

### SwiftVR — Wan2.2-TI2V-5B temporal video restorer (measured frame-level; NOT adopted)
- **Source:** "SwiftVR: ..." **arXiv 2606.09516**, github.com/H-oliday/SwiftVR (cloned
  `D:\REPOS\HYPERclone\SwiftVR`). **License: Apache-2.0.** Weights `H-oliday/SwiftVR` (~20 GB,
  self-contained: a Wan2.2-TI2V-5B-derived DiT + REAE + precomputed empty-prompt embedding).
- **Adoption:** run AS-SHIPPED via `scripts/inference.py` (`SwiftVRPipeline.restore_video`, image-folder
  in -> PNG out, `--attention_backend sdpa`). No SwiftVR code modified. Restored a rendered orbit; scored
  vs GT with our eval (`_infusion/wpp1_swiftvr_frame_eval.py`, ours).
- **Env (venv `venvs/swiftvr`, --system-site-packages, torch 2.11 untouched):** SwiftVR installed
  `pip install -e . --no-deps` (its `torch==2.10.0` pin must NOT downgrade 2.11) + diffusers==0.36.0,
  decord==0.6.0, accelerate/safetensors/einops/imageio(-ffmpeg). **venv-local patch:**
  `venvs/swiftvr/Lib/site-packages/diffusers/models/attention_dispatch.py` - wrapped the eager
  `flash_attn`/`flash_attn_interface` imports in try/except so a BROKEN system-site-packages flash_attn_3
  (the sm_120 landmine) falls back to SDPA. **Re-license:** Apache-2.0 (permissive).

### ArtiFixer3D-bidir — incumbent (characterized from existing evidence, not re-run this WP)
- **Source:** nv-tlabs/artifixer (NVIDIA, arXiv 2603.00492); WSL env `~/artifixer`, weights
  `nvidia/ArtiFixer artifixer-14b.pt` (67.6 GB; code Apache-2.0, weights NVIDIA-NC). Already logged in
  memory `artifixer-adoption` / `docs/ARTIFIXER_INVESTIGATION.md`. Not adopted (architecturally replaces
  a splat rather than polishing our gsplat ckpt; causal path blocked on sm_120 block-sparse flash-attn).

### FlashVSR — EXCLUDED (not cloned/run)
- Hard block-sparse "locality-constrained sparse attention" dependency = the known sm_120 killer.
  Excluded up-front per the researcher's landmine flag.

---

## UI-1 — GUI 2.0 wizard skeleton (`gui2/`, added 2026-07-07)

**No NEW third-party component.** UI-1 is a fresh wizard SHELL that reuses the UI
framework and internal modules already logged above; nothing external was vendored.

- **Gradio 6.11** — the wizard app (`gui2/app.py`) + progress plumbing. Same framework
  already shipped for the legacy `gui/`. **License: Apache-2.0.** Built-in web app —
  no API layer needed for the POC (a React/FastAPI rebuild is the funded phase, per
  `docs/GUI_REDESIGN.md` §1). The interactive splat viewer it lacks (PlayCanvas
  SuperSplat iframe) is UI-2, not UI-1.
- **Everything the runner drives is our own / previously-logged code:** the WP-T1 SfM
  stage handlers (`gui/stages/*`, HLoc ALIKED + LightGlue + NetVLAD + pycolmap),
  MoGe-2 priors (WP-F1), RoMa dense-init (WP-F2), the gsplat MCMC trainer, and
  `core/delivery.py` + PlayCanvas splat-transform (WP-D1). `gui2/` (defaults, qc,
  runner, app, theme, `run2.py`) is entirely OURS.
- **Re-license:** none needed (Apache-2.0 UI framework + our orchestration code).

---

## UI-2 — Live training preview + interactive viewer (`gui2/`, added 2026-07-08)

### PlayCanvas SuperSplat viewer — VENDORED (`gui2/viewer/`)
- **What we ship:** the **prebuilt** static bundle of `@playcanvas/supersplat-viewer`
  **v1.27.1** (npmjs.com/package/@playcanvas/supersplat-viewer,
  github.com/playcanvas/supersplat-viewer) — the same open-source viewer that powers
  superspl.at scene pages. Copied its published `public/` output verbatim:
  `gui2/viewer/{index.html, index.css, index.js}` (index.js = the ~2.9 MB Rollup
  bundle with the PlayCanvas engine v2.20.6 baked in; the 6.4 MB `index.js.map` was
  dropped and its trailing `//# sourceMappingURL=` comment stripped). Obtained via
  `npm pack @playcanvas/supersplat-viewer` + `tar -xzf` (no build step — the npm
  package already includes the built `public/`).
- **Adoption:** used AS-IS, **self-hosted same-origin**. `gui2/app.py::launch()` mounts
  it at `/viewer` (FastAPI `StaticFiles` via `gr.mount_gradio_app`) alongside `/scenes`
  (= `_dataset`), so the wizard, the viewer, and the exported `.sog` all share the
  127.0.0.1 Gradio origin. The iframe is `<gr.HTML>` pointing at
  `/viewer/index.html?content=/scenes/<scan>/preview/viewer_<step>.sog` (the viewer's
  documented `?content=` param). No viewer code modified.
- **Why self-hosted, not the HOSTED superspl.at viewer (verified 2026-07-08):** the
  hosted viewer at `superspl.at` **cannot** load a localhost file — mixed content is NOT
  the blocker (browsers exempt `127.0.0.1`/`localhost` loopback), but **CORS is**: a
  fetch from the `https://superspl.at` origin to `http://127.0.0.1:...` is cross-origin,
  and Gradio's `CustomCORSMiddleware` deliberately withholds `Access-Control-Allow-Origin`
  for external origins hitting a local server (exactly to block remote-page-reads-local-
  file). Self-hosting the same viewer same-origin sidesteps CORS entirely. `.sog` loads
  natively (PlayCanvas engine ≥2.11 reads SOG directly).
- **License:** **MIT** (Copyright (c) 2011-2026 PlayCanvas Ltd.) — permissive; the funded
  port can vendor or rebuild freely. splat-transform (already logged under WP-D1) produces
  the `.sog` it reads.

### `gui2/preview.py` — turntable strip renderer + on-demand SOG export (OURS)
- **What we ship:** a subprocess CLI (`python -m gui2.preview strip|export`) that (a)
  renders a 6-frame turntable STRIP from a training checkpoint and (b) exports a
  viewer-ready `.sog` on demand. All OURS; **reuses already-logged internals**:
  `core.training.gsplat_trainer.{load_colmap_data, render_splats}` (same render path as
  `core/enhancement/distill.py`), object-centering + orbit basis from
  `core.enhancement.coverage.{object_center_from_cameras, estimate_up, _basis}`, and the
  PLY→SOG conversion via `core.delivery.{run_splat_transform, export_ply}` (PlayCanvas
  splat-transform, WP-D1). No new external component. Runner hooks
  (`gui2/runner.py`: checkpoint-save → debounced strip subprocess; `preview_now`,
  `list_checkpoints`, `export_viewer`) and the Step-② panel wiring (`gui2/app.py`) are
  OURS. Default behavior unchanged for every non-gui2 caller.

### SAM 2.1 (Segment Anything 2) — interactive + video mask review (UI-3)
- **Source:** Meta `facebookresearch/sam2` — local clone `D:\REPOS\HYPERclone\sam2`
  (repo dated 2026-02-27), checkpoint `sam2.1_hiera_large.pt` (898 MB, already downloaded).
- **License:** **Apache-2.0** (code) + the sam2.1 weights (Apache-2.0 per the repo). Permissive.
- **Install (2026-07-09):** `SAM2_BUILD_CUDA=0 pip install -e . --no-build-isolation -c
  tools/blackwell_session_2026-06/constraints.txt`. torch/torchvision UNTOUCHED
  (2.11.0+cu128 already satisfies `torch>=2.5.1`); pip only pulled the pure-python deps
  (hydra-core, omegaconf, iopath, portalocker). The `_C` connected-components CUDA
  extension is intentionally **NOT built** on sm_120 — it is only used for hole/sprinkle
  fill, which we disable (`predictor.fill_hole_area = 0`; image predictor defaults
  max_hole/sprinkle=0). Everything else in SAM 2 is plain attention / **SDPA, which works
  on sm_120** (verified: image predictor forward 0.33 s cold / 5 ms warm, 1.4 GB VRAM;
  video predictor full 588-frame track, 8.6 GB VRAM, no `_C` error).
- **Adoption (UI-3 Mask Review):**
  - `core/masking.py` (pre-existing) SAM 2.1 **image predictor** helpers `load_sam2` /
    `predict_mask_sam2` drive the interactive per-frame edit (click include / exclude).
  - `gui2/sam_service.py` (OURS) wraps the SAM 2.1 **video predictor**
    (`build_sam2_video_predictor` → `init_state` / `add_new_points_or_box` /
    `propagate_in_video`) to track the clicked object across the whole orbit sequence
    (frames ordered by orbit index = a video). Frames are downsampled to `--max_side`
    (=1024, the model's internal image size) for light VRAM; masks come back at that
    resolution and delivery/trainer `cv2.resize` them to the image, so resolution is
    irrelevant downstream. Multi-anchor propagation: each edited frame is tracked
    independently and the **nearest seed (linear frame distance, per-seed claim radius)
    wins** per frame — so a correction on a hard view fixes its local window without
    hijacking the well-tracked far frames (SAM 2 propagation is sequential from the min
    conditioning frame, so naive multi-conditioning regresses; this merge is OURS).
  - No SAM 2 source modified; used as an installed package. `gui2/review.py` +
    `gui2/app.py` Step-③ wiring are OURS.

### UI-5 GUI 2.0 design overhaul (`gui2/theme.py`) — theme is OURS; playwright is a dev-only dep
- **Theme system (`gui2/theme.py`, NEW, OURS):** a `build_theme()` that subclasses Gradio's
  `Soft` theme and pins one dark palette for both browser color schemes. No external component;
  it only calls `gradio.themes.Soft(...).set(**tokens)` (Gradio, Apache-2.0, already logged
  under UI-1). `theme.css` rewrite + copy sweep are OURS. No new weights, no new runtime dep.
- **Playwright — VALIDATION-ONLY dev dependency (not shipped, not imported by the app or tests):**
  `pip install playwright` (v1.61.0) + `python -m playwright install chromium`, installed with the
  torch constraints file (`tools/blackwell_session_2026-06/constraints.txt`; torch 2.11.0+cu128
  untouched — playwright pulled only `greenlet`, `pyee`). Used solely by
  `_infusion/ui5_visual_check.py` to drive the wizard headless and audit contrast / em-dashes /
  full-width. **License: Apache-2.0** (Microsoft); the bundled Chromium is BSD-family. A funded
  rebuild keeps this in dev/test tooling only.

### UI-4 Deliver screen + Utilities — NO new external component
- **Adoption (UI-4):** `gui2/deliver.py` (Deliver-screen logic) + `gui2/app.py` Step-④
  and the Utilities menu are OURS. Every capability **reuses already-logged internals**:
  the delivery record + object-region QA schema from `core/delivery.py` (WP-D1); the
  angular-coverage / re-capture report from `core/enhancement/coverage.py`; faithful
  floater pruning from `core/training/prune.py`; the standard visibility-split scorecard
  from `qa_report.py` (`core/qa/*`); turntable-strip rendering from `gui2/preview.py`
  (UI-2); the interactive viewer from the vendored PlayCanvas `supersplat-viewer` (UI-2,
  MIT) served over the same `/scenes` static mount; and BiRefNet object mattes via
  `core/masking.py` (already logged). No new weights, no new dependency, torch untouched.

---

## WP-O1 — 360°/Insta360 ingest (`core/ingest_360.py`, added 2026-08-04)

### (a) ffmpeg `v360` dual-fisheye → equirectangular stitch
- **Idea/source:** ffmpeg's own `v360` filter (`dfisheye` input format → `e` output,
  `ih_fov`/`iv_fov`/`yaw`/`w`/`h`), official filter documentation. The exact command
  (`[0:0][0:1]hstack=inputs=2 → v360=dfisheye:e:ih_fov=200:iv_fov=200`) was validated on
  this footage in PLAN_360 §2.
- **Adoption:** invoked as a **system binary**, no code copied, no library linked. Our
  contribution is the surrounding pipeline: layout probe, `select`-before-`v360` frame
  gating, the raw-video pipe, and the ERP mask geometry.
- **License:** ffmpeg is LGPL-2.1+/GPL-2+ depending on build (here: the gyan.dev
  full build, 8.0). Binary invocation only — a funded rebuild can keep the same call or
  swap in Equilib/py360convert (already logged as candidates in PLAN_360 §8).

### (b) SAM 2.1 video predictor — operator tracking
- **Idea/source:** Meta **SAM 2.1** (`facebookresearch/sam2`), clone at
  `D:\REPOS\HYPERclone\sam2`, checkpoint `checkpoints/sam2.1_hiera_large.pt` (898 MB) +
  config `configs/sam2.1/sam2.1_hiera_l.yaml`. **Already in the stack** (`core/masking.py`
  image predictor; `gui2/sam_service.py` video predictor, both logged under UI-3).
- **Adoption:** `core/ingest_360.py::sam2_operator_masks` calls
  `build_sam2_video_predictor` → `init_state` → `add_new_points_or_box` →
  `propagate_in_video` on the SELECTED ERP frames written as a JPEG folder, reusing
  `core.masking._sam2_resolve_path` for checkpoint resolution. **No SAM 2 source modified**
  (used as an installed package; its optional `_C` extension is absent, so upstream skips
  `fill_holes_in_mask_scores` — harmless here). OURS: the ERP prompt geometry (rear lens at
  ERP centre), the rear-hemisphere/pitch clamp, the area/lost guards and the fallback box.
- **License:** Apache-2.0 (code), SAM 2.1 checkpoints Apache-2.0.

### (c) Window-relative sharpest-frame selection
- **Source:** OURS — `core.frame_select.select_indices` (pre-existing in this repo) reused
  unchanged for the window argmax; the var-of-Laplacian proxy is textbook. No third party.

### No new dependencies
Nothing was pip-installed for WP-O1: `cv2` 5.0.0, `numpy`, ffmpeg/ffprobe on PATH, and the
already-present sam2 clone. torch 2.11.0+cu128 untouched; no CUDA build.

---

## WP-O2 — 360° SfM lane (`core/sfm_360.py`, added 2026-08-04)

### (a) COLMAP panorama SfM — `panorama_sfm.py` / `pycolmap/panorama.py`
- **Idea/source:** COLMAP's own 360° pipeline. Two upstream revisions were fetched into
  `tools/reference_sources/colmap_panorama/` (see its README for exact paths/refs):
  the **self-contained `python/examples/panorama_sfm.py` at tag `4.1.0`** — the revision that
  matches our installed binary (`COLMAP 4.1.0 Commit fa8e3b3 on 2026-06-26 without CUDA`) —
  and the `main` @ `319c6e6` split into `python/pycolmap/panorama.py` + a thin CLI.
  **Neither ships in our pycolmap 4.1.0 wheel** (`site-packages/pycolmap/` has only
  `__init__.py`, `_core.pyd`, `utils.py`, `cost_functions/`, `pyceres/`), so the logic had to
  be vendored rather than imported.
- **Adoption:** `core/sfm_360.py` re-implements the geometry with upstream's math kept
  faithful — `create_virtual_camera`, `get_virtual_camera_rays`, `spherical_img_from_cam`,
  `get_virtual_rotations`, `create_pano_rig_config`, `PanoProcessor.convert_to_equirectangular`
  — and calls the same pycolmap entry points (`extract_features` with
  `CameraMode.PER_FOLDER` + `mask_path`, `apply_rig_config`, `match_sequential` with
  `rig_verification` + `skip_image_pairs_in_same_frame`, `incremental_mapping` with
  `ba_refine_sensor_from_rig/focal_length/principal_point/extra_params = False`).
  OURS: compositing the WP-O1 operator/nadir/seam masks INTO the exclusive crop masks;
  building the 12 remap tables and exclusive masks once instead of per pano; the
  solid-angle sphere-coverage accounting; cv2 IO at q95 instead of PIL defaults; the
  stage heartbeat + hard `--timeout_s` watchdog for colmap/colmap#3599; the metrics /
  trajectory / report layer. Upstream's covisibility-`.npz` filtering is NOT ported.
- **License:** BSD-3-Clause (colmap/colmap). Snapshot files carry upstream's headers.

### (b) COLMAP C++ sources read to pin down behaviour (reference only, nothing vendored into runtime)
- `src/colmap/controllers/image_reader.cc` and `controllers/pairing.cc` at tag `4.1.0`,
  snapshotted next to the Python files. They establish the mask-file naming fallback
  (`<stem>.png` works for `<stem>.jpg`), the "keypoint kept iff mask pixel != 0" rule that
  makes WP-O1's 255=VALID convention directly usable, and the rig-aware
  `expand_rig_images` behaviour of the sequential pair generator. BSD-3-Clause.

### (c) COLMAP vocabulary tree (loop detection)
- `vocab_tree_faiss_flickr100K_words256K.bin`, auto-downloaded and cached by COLMAP itself
  from `github.com/colmap/colmap/releases/download/3.11.1/` into
  `%LOCALAPPDATA%\..\.cache\colmap\` on the first `--loop_detection 1` run. Binary asset
  published by the COLMAP project (BSD-3-Clause project; the tree is trained on Flickr100K).
  Not redistributed by us.

### No new dependencies
Nothing was pip-installed for WP-O2: pycolmap 4.1.0, cv2, numpy and matplotlib were already
present. torch 2.11.0+cu128 untouched; no CUDA build. NOTE: our pycolmap wheel is built
**without CUDA**, so SIFT extraction and matching run on the CPU in both arms (recorded as
`colmap.has_cuda: false` in every `sfm_report.json`).

## WP-O3 — 360° training lane (`core/train_360_prep.py` + trainer flags, added 2026-08-05)

**Nothing was vendored and nothing was installed.** WP-O3 is 100 % our own code on top of
components already logged above. Recorded here only so the future funded rebuild sees the
full dependency surface of the 360 lane end to end.

### Components reused (all previously logged)
- **gsplat** v1.5.3 AOT sm_120 (Apache-2.0) — the rasterizer/MCMC that trains the splat, via
  our own `core/training/gsplat_trainer.py`. `rasterize_mode="antialiased"` = Mip-Splatting.
  Its MCMC noise schedule (`op_sigmoid(1-opacity)`, k=100, x0=0.995) is the mechanism behind
  the WP-O3 ARM 1 divergence — read from gsplat's source, not modified.
- **pycolmap** 4.1.0 (BSD-3-Clause) — reads the WP-O2 crop-rig model, rewrites the image
  names (`Image.name` is writable) and writes the packaged `sparse/0`.
- **LPIPS (VGG) via torchmetrics** — the masked-LPIPS gate metric, through the existing
  `core/qa/masked_metrics.py` (WP-Q1 component 1, already logged).
- **OpenCV** (Apache-2.0) — image IO and the figure strips. **ffmpeg/libx264** (LGPL/GPL,
  `C:\ffmpeg\bin\ffmpeg.exe`, system binary) — encodes `walkthrough.mp4` from rendered
  frames; no filter graph, plain image2 → H.264.
- The dataset it consumes is WP-O2 ARM A's, so COLMAP's panorama geometry (BSD-3-Clause,
  logged under WP-O2) is upstream of every number here.

### New code, all ours
`core/train_360_prep.py` (crop-rig model → trainer-shaped COLMAP dataset, flat names,
mask-name convention, pano-grouped held-out split), `gsplat_trainer.py --holdout_list`
(explicit test-set membership for view GROUPS that must not be split) and
`--loss_mask` (average L1 + the SSIM map over the valid mask instead of compositing over
black), `_infusion/wpo3_eval.py`, `_infusion/wpo3_diag_collapse.py`.

### No new dependencies
No pip install of any kind. torch 2.11.0+cu128 untouched, gsplat not rebuilt.
