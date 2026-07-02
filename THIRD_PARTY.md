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
  `coef_` (384) + `intercept_` (15.164) to `checkpoints/v0.0.1/regr/linear_head.npz`
  so runtime is pure numpy (`jod = feat@coef + intercept`) — no sklearn dep, fully
  deterministic. JOD is within-scene-delta only.
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

## Pre-existing components (recorded for completeness — not added by WP-Q2)
- **gsplat** v1.5.3 (AOT sm_120) — nerfstudio-project/gsplat — Apache-2.0. Core rasterizer/MCMC.
- **Mip-Splatting** antialiased mode — via gsplat `rasterize_mode="antialiased"` — used as-is.
