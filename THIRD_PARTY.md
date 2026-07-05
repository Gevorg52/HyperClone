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
