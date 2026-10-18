# Apex Reality — Clean-Splat EXECUTION PLAN (July 2026)

**Audience: the coding agent (Opus 4.8) executing the Clean-Splat Program.** This doc is the
implementation handoff: environment contract, work packages with verified APIs/pins/formulas,
validation gates, and risks. Strategy + evidence live in `PERFECT_SPLAT_PROGRAM.md` (rev 2);
read that first for WHY. Everything below was verified against fetched primary sources or the
local codebase on 2026-07-01 (5 research agents + 2 implementation-dossier agents + local recon).
Tags: [V] verified · [PC] paper-claim/inference · [U] unverified.

**Hardware decision (user, 2026-07-01): stay LOCAL on the RTX PRO 6000 (sm_120). Every WP below
runs locally. The H100 causal-ArtiFixer probe (Annex A) is optional and cheap — not excluded,
not blocking anything.**

**POC phase (user, 2026-07-01): non-commercial proof of concept — licenses are noted for the
future funded port but are NOT a filter and must not drive any design choice (§0).**

---

## 0. Working rules (restored `docs/rules.txt` — follow constantly)

- Don't make assumptions, always research and make educated decisions
- Break complex problems into smaller pieces
- Check answers from multiple perspectives · Think through the code logically
- Always double check before saying you're done and it's working
- Don't be lazy. Be thorough. Be methodical. Be smart.
- Don't add complexity for marginal performance gains; simpler-but-slightly-slower is often better
- Only optimize when profiling shows a real bottleneck

Project-specific additions (from six months of validated practice):
- Every lever ships **default-OFF** behind a CLI flag; current production behavior unchanged until opted in.
- Every WP has a **validation gate** — do not advance past a failing gate; record results (incl.
  negative) in the relevant doc.
- **GATE EVIDENCE RULE (added 2026-07-02 after two audit gaps): every number in a gate claim must
  be PERSISTED to a committable artifact at measurement time** (scorecard JSON, a `*_RECORD.md`
  outside gitignored dirs, or a metrics JSON + the script that produced it). A/B records must
  include both arms' full argv + seed. "I computed it ad hoc" does not close a gate — reviewers
  verify from repo state only.
- Add losses ONE at a time with per-loss logging; keep a no-priors A/B baseline.
- Judge perceptual work by LPIPS/region-metrics/visual, never PSNR alone; judge FILL by the WP0
  visibility-split suite, never by GT metrics on unobserved regions.
- **POC LICENSE STANCE (user directive, stated twice — supersedes any license-driven choice
  below): non-commercial proof of concept. Licenses are NEVER a selection filter or design
  driver — use/vendor/run the best code directly, whatever its license (NC, no-license, gated
  weights all fine). Keep a lightweight `THIRD_PARTY.md` ledger (component → source → license)
  purely so the future funded rebuild knows what to re-license or re-implement.
  `tools/reference_sources/` snapshots are fair game to copy from.

## 1. Environment contract (verified 2026-07-01)

- **Windows 11**, system **Python 3.11** (`C:\Users\gevor\AppData\Local\Programs\Python\Python311\python.exe`).
  NO conda. All pip installs use a constraints file pinning torch (`tools/blackwell_session_2026-06/constraints.txt`).
- **torch 2.11.0+cu128** — drives the card, **never reinstall/downgrade**. numpy ≥ 2 in main env.
- **GPU**: RTX PRO 6000 Blackwell, **sm_120 / cc 12.0**, 96 GB, driver 576.57. Block-sparse
  flash-attention does NOT exist for sm_120; plain SDPA (all 4 backends) + cuDNN SDPA verified fine.
  NEVER `pip install flash-attn-4` (poisons both attention paths — see ARTIFIXER_INVESTIGATION).
- **CUDA Toolkit v12.9** only. Any new CUDA extension: build AOT in a VS2022 vcvars64 shell with
  `TORCH_CUDA_ARCH_LIST=12.0`, `DISTUTILS_USE_SDK=1`, `--no-build-isolation` (recipe:
  `tools/blackwell_session_2026-06/build_gsplat.bat`).
- **gsplat v1.5.3** AOT-built for sm_120 (verified working: Mip-Splatting antialiased + MCMC).
  The local `../gsplat` clone is STALE (v1.4.0) — don't trust it for upstream reference.
- **HLoc v1.5** editable from `../Hierarchical-Localization`; **pycolmap 4.1.0** (`cam_from_world`
  is a METHOD; BA opts under `.ceres.solver_options`).
- **WSL2 Ubuntu-24.04**: venv `~/artifixer-venv` (torch 2.11.0+cu128, diffusers 0.37.1,
  transformers 5.5.0, moge 2.0.0); repo `~/artifixer` (+3DGRUT, flex patch `patch_flex.py` must be
  re-applied on fresh clones); ArtiFixer weights `D:\REPOS\HYPERclone\artifixer-checkpoints\artifixer-14b.pt`.
  Do NOT use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` on WSL2.
- **Datasets**: `_dataset/GPU_V4/apex_images/{export,sfm,training}` (2M-GS factor-2 30k ckpt) and
  `_dataset/Laptop/apex_images/training/gsplat_prod` (Laptop ckpts dir was cleared; PLY remains;
  production masks come from the SAM 2.1 stage). **CORRECTED 2026-07-04 (WP-G1 discovery): the
  `gsplat_prod/data/images_2` training frames are FULL SCENES, NOT pre-multiplied black-bg**
  (Laptop ~0% near-black outdoor patio; GPU_V4 16.8% warm indoor) — production training has been
  full-scene all along; WP-S2 tests whether object-only masking should become the object-mode default.
- **Trainer integration points** (verified): losses join at `core/training/gsplat_trainer.py:1040–1129`
  (l1/ssim → alpha → opacity/scale/vis regs → depth@1097 → geo → gated-LPIPS/gram); flags at
  argparse block; backend plumbing pattern in `core/training/gsplat_backend.py` (`get_default_config()`
  + subprocess argv). Gaussian init: `init_gaussians_from_sfm` (line ~514). Renderer provides
  RGB + expected-depth (`render_mode="RGB+ED"`) + accumulated alpha.
- Prior-session assets salvaged to `tools/blackwell_session_2026-06/` (WSL ArtiFixer scripts, eval
  scripts, cuDNN SDPA tests, build recipe) and `tools/reference_sources/` (fetched source snapshots
  incl. `gsplat_pr858.diff`, MoGe API, alignment references).

---

## 2. WORK PACKAGES

Two parallel tracks. **Track Q (quality measurement + quick wins)** and **Track F (faithful
package)** are independent; **Track C (completion)** should start after WP-Q1 exists so its output
is judged correctly. Within a track, order is load-bearing.

═══════════════════════════════════════════════════════════════════════
### WP-Q1 — QA harness v1 ("clean & usable" metrics) — ~1.5–2 weeks — FIRST
═══════════════════════════════════════════════════════════════════════
> **STATUS: PHASE 1 (components 1+8) ✅ CLOSED 2026-07-02** (implement → review → fix → re-validate).
> `core/qa/` + `qa_report.py` shipped; deterministic (byte-identical JSON); corruption-sensitive
> (floaters crater all metrics; blur correctly degrades ONLY LPIPS); sliver-robust masked LPIPS
> (review-found crash fixed, wedge split verified end-to-end on a 45.2%-observed view). Baselines
> banked: GPU_V4 masked 25.278/0.8514/0.4125 · Laptop 22.319/0.8120/0.3540 (`qa/README.md`).
> **PHASE 2 ✅ CLOSED 2026-07-02** (implement → review → fix → re-validate): all 6 scorers live
> (crossscore/nova/tsed/raft_warp/nvs_sqa/flicker), byte-deterministic, every scorer has a
> demonstrated TRUE POSITIVE (incl. `--floater_jitter` temporal control: RAFT 11× worse, TSED
> 1.0→0.70). Review-found stale-render-cache bug FIXED (always-fresh staging, corruption-checked
> merge). Clones hash-pinned + patches in `tools/patches/`. venv `D:\REPOS\HYPERclone\venvs\qa`;
> full 6-scorer scorecard ≈3.5 min/scene. Gate table: `qa/README.md`.
**Goal:** replace bare held-out-GT LPIPS with a visibility-split suite; new module
`core/qa/` + CLI `qa_report.py` producing a per-scene scorecard (JSON + HTML strip).

**Environment:** one new Windows venv "qa" on torch 2.11 stack (no custom CUDA in the whole set).
Frictions [V]: sklearn <1.7 (NVS-SQA pickle; or re-fit its 385→1 linear head once);
transformers pinned ~4.4x in THIS venv for `Dinov2Model`; torch 2.11 `weights_only=True` default
breaks NOVA/CrossScore-Lightning loads — use `weights_only=False` for these trusted files, fail
hard on missing ckpts (NOVA silently degrades to vanilla DINOv2 otherwise).

**Components (all verified, all permissive):**
1. **Visibility masks** (core, build first): OGGSplat protocol [V] — unobserved =
   `render_alphas < 0.3` + `cv2.MORPH_CLOSE` 25×25 + `dilate` 5×5 ×3. Optionally intersect with
   per-Gaussian <k=3 training-view counts (reuse `val_splatsure.py` projection machinery).
   OBSERVED → masked PSNR/SSIM/LPIPS (crop bounding boxes for LPIPS — not pixelwise-decomposable);
   UNOBSERVED → FID + NR metrics ONLY, never PSNR/LPIPS.
2. **CrossScore** (BSD-3) [V]: `task/predict.py` with `trainer.ckpt_path_to_load=ckpt/CrossScore-v1.0.0.ckpt`
   (123 MB in-repo LFS), `data.dataset.query_dir=<renders> reference_dir=<real>`,
   `this_main.force_batch_size=True`, `data.neighbour_config.deterministic=True` (else 5 RANDOM refs),
   gray output mode for machine-readable maps. Output = predicted per-pixel SSIM [0,1] higher=better.
   Check issue #3 (img_std slice bug) — verify HEAD↔ckpt consistency before "fixing".
3. **NOVA** (MIT, WACV'26) [V]: `python nova.py --image-a <nearest_real> --image-b <render>
   --checkpoint weights/NOVA_NVS.pt` (661 MB LFS; state-dict dict with `base`+`embedding_head`).
   We pick the nearest held-out real photo by COLMAP pose distance. Output cosine distance [0,2]
   lower=better — calibrate per-scene, use deltas.
4. **TSED — vendor ~100 lines** (MIT) [V]: SIFT (Lowe ratio) + median symmetric epipolar distance
   from poses; pair consistent iff ≥10 matches AND median < 2px·(H/256). GOTCHAS: repo poses are
   c2w OpenGL — ours are w2c OpenCV (invert, NO axis flip); real K replaces hardcoded 256² intrinsics;
   0-match pairs count inconsistent (report match counts so textureless ≠ geometry failure); eps-guard
   epipole division.
5. **RAFT warp-error** (primary temporal metric, moving cameras) [V]: torchvision
   `raft_large(weights=Raft_Large_Weights.DEFAULT)` + `weights.transforms()`, take flow iterate
   `[-1]`, backward-warp t+1→t via grid_sample, forward-backward occlusion mask
   (`0.01·(|f|²+|b|²)+0.5`), masked L1. ~85 MB auto-download, batch 6–8 at 1080p.
6. **Vendored `temporal_flickering`** (Apache, ~60 lines from VBench) — STATIC-camera clips only
   (`(255−MAE)/255` on consecutive absdiff). **Never `pip install vbench`** (pins numpy<2,
   transformers 4.33 — poisons env).
7. **NVS-SQA** (CC0, TPAMI) [V]: folder-of-sorted-frames in → JOD score (higher=better, ~negative
   scale). WITHIN-scene comparisons only. Quote the checkpoint path (contains `(`/`+`).
   Frame ORDER matters (fusion="last") — zero-pad numbering.
8. **Scorecard**: per scene emit {masked-PSNR/SSIM/LPIPS, CrossScore mean+map, NOVA, TSED ratio,
   RAFT warp-err, flicker (static only), NVS-SQA, unobserved-area %}. Maintain a **baseline bank**
   (GPU_V4 + Laptop prod renders) — all gates are per-scene DELTAS vs this bank, not absolute
   thresholds.

**Deferred to v1.5** (isolated WSL venvs, offline diagnostics — deferral is BUILD FRICTION, not
licenses): MEt3R (timm==0.4.12 pin + pytorch3d/FeatUp source builds); ArtifactWorld heatmaps
(LTX-Video 13B ~27 GB base, pins diffusers 0.33.1 — separate venv; input 960×544, frames 8n+1).
**Request the gated ArtifactWorld weights on HF now** so access is approved before v1.5 starts.

**Validation gate:** harness runs end-to-end on GPU_V4 + Laptop prod checkpoints; scores reproduce
run-to-run (deterministic flags verified); a deliberately-corrupted render (add floaters/blur)
scores measurably worse on the right axes.

**Then immediately: the re-judgment.** Re-score the ArtiFixer wedge artifacts
(`artifixer-checkpoints/` PLYs + renders, easy-case AND wedge) and the autocomplete results under
the suite. Deliverable: a table in ARTIFIXER_INVESTIGATION.md — does ArtiFixer3D's wedge fill
win on cleanliness/consistency despite losing on GT-LPIPS? This decides how hard we push the
completion lane.

═══════════════════════════════════════════════════════════════════════
### WP-Q2 — Quick trainer wins — ~1 day total
═══════════════════════════════════════════════════════════════════════
> **STATUS: ✅ CLOSED 2026-07-02** (implement → review → fix → re-validate). **HEADLINE FINDING:
> 89.63% of Gaussians in the validated production recipe are needle-shaped (erank<2); erank 0.01@7k
> collapses that to 16.59%** (mean erank 1.53→2.05, SOR 0.12→0.08%, fixed-bar giants 8.64→5.80%) at
> ~equal quality (masked PSNR 26.96→26.87, LPIPS 0.4205→0.4230 — persisted `qa/wpq2_ab_{off,on}/`).
> Independently reproduced by review. Flags: `--erank_lambda/--erank_start/--erank_end` (default OFF),
> `--seed` (now controls Prefetcher view order — original A/B arms were statistically matched only,
> see `wpq2_ab/AB_RECORD.md`). eps2d rule: zoom-out +9.9 dB vs band-limited ref (COMBINED with ss4 —
> see `_infusion/wpq2_sags/NOTES.md`); zoom-in = clean A/B, suppresses needle streaks (which erank
> also kills at the source — the levers compose). `floater_report.py` gained the needle metric.
> **Implication: WP-X1 (>3M master splat) is re-armed** — needles/blobs were the 8M divergence mechanism.
**(a) Effective-rank anti-needle loss** (paper 2406.11672 Eq. 7–10; drop-in below, cross-checked
against the official erank_gs code — consult/vendor that repo freely):
```python
def erank_loss(log_scales, w_erank=0.01, w_thin=0.0, eps=1e-7):
    lam = torch.exp(log_scales * 2)                              # s_i^2 = cov eigenvalues
    p = lam / lam.sum(dim=1, keepdim=True).clamp_min(1e-30)
    H = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
    erank = torch.exp(H)                                         # in [1,3]; needle→1
    hinge = torch.clamp(-torch.log(erank - 1.0 + eps), min=0.0)  # active iff erank<~2
    thin = torch.exp(log_scales).min(dim=1).values
    return w_erank * hinge.mean() + w_thin * thin.mean()
```
Flag `--erank_lambda 0.01 --erank_start 7000`, **w_thin=0 under MCMC** (scale_reg already
shrinks; hinge is the orthogonal anti-needle term). Clamps are OURS (MCMC dead Gaussians
underflow → NaN without them). Gate: floater/needle metrics (floater_report.py) improve on
GPU_V4 at equal PSNR/LPIPS.

**(b) SA-GS eps2d rule at eval** (pure Python — their CUDA port is unnecessary; gsplat already
exposes the same knob): gsplat's
dilation is the `eps2d` arg of `rasterization()` (default 0.3) [V]. At eval/zoom rendering compute
`r = (render_W/train_W)·(render_f/train_f)·(train_dist/render_dist)` per view and pass
`eps2d=0.3*r*r`; for zoom-OUT quality add `--eval_ss K`: render at K× res with K-scaled Ks and
`eps2d=0.3*(rK)^2`, box-downsample `F.avg_pool2d`. Keep `rasterize_mode="antialiased"` (its
compensation uses the same blurred covariance — coherent [V]). Gate: zoomed turntable renders of
existing ckpts show reduced aliasing/dilation artifacts (visual + RAFT warp-err at varied zoom).

═══════════════════════════════════════════════════════════════════════
### WP-F1 — MoGe-2 prior cache (SHARED INFRASTRUCTURE) — 1–2 days
═══════════════════════════════════════════════════════════════════════
> **STATUS: ✅ CLOSED 2026-07-02** (implement → review [gate initially invalid: evidence not
> persisted — origin of the §0 GATE EVIDENCE RULE] → fix → re-validate; all numbers reproduced
> bit-for-bit). `core/priors.py` + caches for Laptop/GPU_V4 (per-view alignment, 1.1–1.4%
> residuals, pixel-alignment independently re-derived by review) + `--mono_depth_lambda` consumer.
> **A/B (Laptop 15k, seed 0): held-out depth residuals −72% med / −87% p90 vs SfM sparse on
> held-out views at not-worse masked-LPIPS (0.3546→0.3540)** — the geometry foundation for WP-F3/F4.
> Evidence: `_infusion/wpf1_AB_RECORD.md` + `qa/wpf1_ab_{off,on}/`. Caveats recorded: pose_opt=false
> in both arms; depth metric reads as "agrees better with SfM on unseen views", not GT depth.
> Priors stored in the undistorted training frame; conf channel is a heuristic until F3.
Install: `pip install git+https://github.com/microsoft/MoGe.git` (keep its pinned
`utils3d @ 3fab839`). Pure PyTorch/SDPA, numpy≥2 OK, Windows-clean, MIT (code+weights) [V].
Model: **`Ruicheng/moge-2-vitl-normal`** (metric depth + normals).
```python
from moge.model.v2 import MoGeModel
out = model.infer(image_3HW_float01, fov_x=fov_from_colmap_deg, resolution_level=9,
                  use_fp16=True, apply_mask=True)
# out: points(H,W,3 OpenCV cam) · depth(H,W metric z; inf=invalid!) · intrinsics(3,3 NORMALIZED)
# · mask(H,W bool) · normal(H,W,3 unit cam-frame)
```
ALWAYS pass per-image `fov_x` from COLMAP intrinsics; ALWAYS filter by `mask` (inf poisons losses).

**Per-view cache** `data_dir/priors/` (at training-factor resolution, stem-keyed — mirrors the
`masks_{factor}` convention): `depth/{stem}.npy` (aligned METRIC, float32, inf=invalid) ·
`normal/{stem}.npy` (cam-frame unit) · `conf/{stem}.npy` · `plane_mask/{stem}.npy` (uint16
instance labels, 0=bg; WP-F4) · `edge_mask/{stem}.png` (inverse-Canny 50/150, 5px dilate) ·
`align.json` (per image: scale, shift, n_inliers, median_rel_residual, reliable, depth_weight).

**Alignment (metric ⇒ scale-first)** [V components]: per view — sparse samples from COLMAP
`image.xys`+`point3D_ids` (filters: reproj error ≤2.5px, track ≥5, depth∈[0.001,1000]); bilinear-
sample D_mono, drop invalid; percentile-prune 0.1/99.9; `s = median(z_i/d_i)` → ~100-round 1-point
RANSAC (relative inlier |s·d−z|/z < 0.05–0.10) → weighted LS refit (w=1/reproj_err); add shift b
ONLY if it cuts median residual >20% and |b| < 0.2·median z; gates: <10 samples ⇒ view excluded;
per-view s outside [0.2,5]×median(s) ⇒ unreliable flag. MoGe-2 is metric ⇒ scales should cluster —
high dispersion ⇒ use one GLOBAL pooled scale. Reference impls in `tools/reference_sources/
{dnsplatter,gs}/` (read-only).

**First consumer:** dense masked depth-L1 next to the existing sparse `--depth_lambda`
(flag `--mono_depth_lambda`, default 0). Gate: on Laptop, deck-region depth/geometry improves,
held-out masked-LPIPS not worse; alignment report sane (inliers, residuals).

═══════════════════════════════════════════════════════════════════════
### WP-F2 — EDGS-style dense init via HLoc — 2–3 days
═══════════════════════════════════════════════════════════════════════
> **STATUS: ✅ CLOSED 2026-07-03** (implement → review → fix → re-validate). `core/dense_init.py`
> + `hloc/matchers/roma.py` (patches in `tools/patches/`, apply cleanly to fresh clone) +
> `--dense_init` backend opt-in (default OFF). **Laptop: 93k→707k points (7.59×), deck density
> +4.72×, new-point reproj 1.42px, poses preserved (Δt=0.0, asserted+persisted). A/B (15k, seed 0):
> global masked-LPIPS −3.5% (0.3549→0.3425), deck-LPIPS −2.3%, +0.12 dB, needles 91→83%** — the
> biggest single faithful lever landed so far, ~free at init. Seeded matching now bitwise-
> deterministic; certainty-filter edge-case bug fixed + regression test; cv2-headless 5.0.0 pinned.
> Evidence: `_infusion/wpf2_RECORD.md` + `qa/wpf2_ab_*`. Composability with WP-F1 safe by
> construction (dense points carry empty tracks — invisible to point_obs/alignment); both-flags-on
> A/B not yet run (future stack test). Fallback (vendor EDGS sampler) unused — density sufficed.
Primary route: reproduce via our HLoc stack (below) — EDGS-as-shipped is a poor fit for TECHNICAL
reasons (graphdeco base trainer, in-memory-only output — no PLY/points3D export, issue #15;
numpy<2 pin; unresolved Windows issue #9; H/W-swap triangulation bug #14 [V]). POC option if the
gate shows density-limited gains: vendor/adapt EDGS `source/corr_init.py` directly (clone
github.com/CompVis/EDGS) — its per-pixel certainty sampling yields ~3M points vs HLoc's
~100k–1M; fix bug #14 and add a points3D export. New module `core/dense_init.py`:

1. **Pairs**: `hloc.pairs_from_covisibility.main(model, output, num_matched=5)` [V signature].
2. **Matcher**: add `hloc/matchers/roma.py` (~60 lines) wrapping pip `romatch` (MIT) —
   `roma_outdoor` weights (Parskatt/storage releases + DINOv2 ViT-L/14, torch-hub auto-download);
   return warp-sampled keypoint pairs + certainties in match_dense's expected dict. EDGS-informed
   params [V]: match at 560×560 fp16 (`upsample_preds=False`), certainty threshold 0.05,
   `roma_model.sample(warp, certainty, num=10_000–20_000)` per pair. Conf entry in
   `match_dense.confs`: `max_error: 2, cell_size: 4`. Fallback matcher: shipped `loftr` conf.
3. **match_dense.main(...)**: pass `max_kps=None` or ≥50k — the default **8192 caps keypoints/image**
   and would silently gut the dense init [V].
4. **Triangulation**: `hloc.triangulation.main(sfm_dir, reference_model, image_dir, pairs, features,
   matches)` — **Python API only; the CLI `__main__` is broken** (undefined `mapper_options` pop) [V].
   Verified: DB rebuilt from reference with fixed IDs, epipolar check vs reference poses
   (max_error 4px), `pycolmap.triangulate_points` — poses PRESERVED, points only [V]. Optionally
   relax `mapper_options={"triangulation": {...}}` merge/complete reproj errors for density.
5. **Merge + seed**: concat new points3D with original sparse (dedup by voxel ≈ median-NN distance);
   EDGS-style attributes at seeding — color from track mean → SH DC; isotropic scale
   ≈ 0.5·0.001·cam-distance (or our existing KNN init); pipe into `init_gaussians_from_sfm` via a
   `--dense_init` flag on prepare_data.
6. **Selectivity** (CoMapGS finding: random injection HURTS −0.2..−0.7 dB [V]): keep ALL triangulated
   points but report region density; optionally restrict extra seeding to low-density cells.

Expect 100k–1M new points (hloc quantization sparsifies vs EDGS's ~3M — fine; MCMC re-densifies)
[V]. Watch DB size on >300-image scenes (limit num_matched=5).
**Gate (the Laptop deck testbed):** deck SfM-point density (was 0.28× init / 0.16× trained vs
keyboard) rises materially; retrain → deck-region LPIPS improves; global held-out masked-LPIPS
not worse; report point count + mean reproj error.

═══════════════════════════════════════════════════════════════════════
### WP-S1 — Stacked-levers production run (the four faithful levers, first co-run)
═══════════════════════════════════════════════════════════════════════
> **STATUS: ✅ CLOSED 2026-07-04** (production runs, no new code beyond eval scripts).
> **GATE PASSED.** First-ever stack of all four validated faithful levers (dense-init +
> mono-depth 0.05 + erank 0.01@7k + gated-LPIPS 0.2) vs the production control, on the user's
> two real captures, 4× 30k @ factor 2, cap 3M, pose_opt, C2F 4→2, seed 0. **Stack beats
> control on held-out masked-LPIPS (Laptop 0.3143→0.3096 −1.5%; GPU_V4 0.4106→0.4071 −0.85%)
> AND on the flat/shiny region crops (Laptop deck 0.2773→0.2739 −1.2%; GPU_V4 shiny backplate
> box(0.28,0.72,0.30,0.55) 0.3527→0.3473 −1.5%) on BOTH scenes**, and collapses needle
> Gaussians 89→15% / 92→11% (mean-erank 1.5→2.2). NO negative lever interaction (stack ≥ ctrl
> on every primary metric — the mono×dense×erank triple composes positively; drop-one probe
> unneeded). Honest caveats: giant-blob +3–4pp (benign erank isotropization); masked-PSNR
> −0.03/−0.11 dB (LPIPS↔PSNR trade); improvements MODEST (well-captured orbits — payoff is
> geometric cleanup + slight sharpening). GPU_V4 dense-init built this WP (123k→883k, 7.16×,
> reproj 1.39px, seed 0). **Key unsolved: view-dependent specular reflections (glossy laptop
> lid mirror; GPU metal sheen) render as mushy smears in BOTH arms — the faithful levers add
> no specular modeling → clear signal for a dedicated specular lever.** Bank comparison
> confounded (GPU bank = 2M/low-LPIPS → +2.3 dB PSNR but worse LPIPS; even WP-S1 ctrl sits
> 2.2 dB under it — recipe, not levers). Evidence: `_infusion/wps1_RECORD.md` +
> `qa/wps1_{laptop,gpu}_{ctrl,stack}/` + `_infusion/wps1_{laptop,gpu}_region.json` + strips.

═══════════════════════════════════════════════════════════════════════
### WP-F3 — DIM-masked depth supervision (In-Depth-We-Trust) — 2 days + ablation
═══════════════════════════════════════════════════════════════════════
No code exists [V] — implement from formulas (~80 lines; full paper text in
`tools/reference_sources/paper.txt`). **Semantic: the DIM marks where the RENDER is multi-view
inconsistent; absolute mono-depth loss applies ONLY inside M=1** (well-reconstructed regions get
ZERO depth penalty — protects good geometry). Gradient loss is global.

DIM (Eq. 8–10): virtual stereo cam `P^r=[R|t+b]` (b along cam x-axis); render right-eye depth
each iteration (no_grad); back-project right pixels; re-project to left; scatter z-buffer
(`scatter_reduce(amin)`); `M=1` iff |D̂^{l←r}−D̂^l| ≥ ε OR no depth landed.
Losses: `L_abs = L1(M⊙D_mono_aligned, M⊙D̂)`; `L_rel = L1(∂xD,∂xD̂)+L1(∂yD,∂yD̂)` (single-scale,
unmasked, linear depth). Enable after iter 3000 [V].
**ε, |b|, α, β are UNPUBLISHED** [V] — implement relative: ε = k_ε·median scene depth,
|b| = k_b·median inter-camera baseline; grid {k_ε, k_b, α, β} 4-point on one scene. Start
α≈0.1·depth_lambda-scale, β≈0.5α [U]. DIM costs ~+40–60%/step (extra render) — compute every K
steps + cache [U]. Multiply mask by MoGe validity ∩ alignment-reliable.
Flags: `--dim_depth_lambda --dim_grad_lambda --dim_eps_rel --dim_baseline_rel --dim_every`.
**Gate:** beats WP-F1's plain masked dense-depth baseline on Laptop (depth metrics + floaters +
masked-LPIPS); if not, fall back to conf-masked L_abs + gradient loss without DIM (still principled).

═══════════════════════════════════════════════════════════════════════
### WP-F4 — Planar-prior losses (PlanarGS port) — 3–5 days
═══════════════════════════════════════════════════════════════════════
> **STATUS: ✅ CLOSED 2026-07-05 — CONDITIONAL LEVER (gate failed twice, recorded honestly).**
> As-specced (PlanarGS lambdas): flatness −18.9% deck-RMS, appearance +15.3% LPIPS (λ
> mis-calibration, L_rn > photometric). RECALIBRATED (λ_rn 0.02/λ_rd 0.02/λ_p 0.5, FG-filtered
> masks, contributions ~18% of photometric, no spike): flatness improves MORE (deck −19.9%,
> all-planes −48.2%) but appearance still regresses (+4.0% deck / +4.3% masked-LPIPS, −0.45 dB)
> → the flatness↔appearance trade is INTRINSIC on this scene, not calibration. No third attempt
> per discipline. Lever ships DEFAULT-OFF as a conditional tool: use where geometric flatness
> outranks perceptual fidelity (meshing/geometry deliverables/worlds walls), not perceptual object
> delivery. Evidence: `_infusion/wpf4_ab_flatness_recal.json`, `qa/wpf4_ab_on_recal/`. Shipped: `core/plane_masks.py` (plane-instance maps from
> the MoGe-2 cache — SAM route dropped: masks never preserved + GPU contention;
> geometric inspection instead) built for Laptop (304/304 views, 8.57 planes/view).
> Trainer + backend flags (all default OFF): `--planar_lambda` (L_p), `--prior_depth_lambda`
> (L_rd), `--prior_normal_lambda` (L_rn), `--*_start`, `--planar_detach` (default off =
> faithful, no detach). Helpers `co_planar_depth`+`normal_from_depth` ported from
> PlanarGS (MIT; THIRD_PARTY.md). L_dn + lambda_scale flatten SKIPPED per plan. CPU +
> GPU per-loss mechanism checks PASSED; **146 tests pass** (7 new); reviewer pass fixed
> a HIGH sign-preserving degenerate-denom bug + border/coverage/normalization items.
> **A/B (Laptop 15k, seed 0, cap 1M, dense_init+mono_depth base): deck plane RMS
> −18.9% / all-planes −42.1% (FLATNESS improves) BUT masked-LPIPS +15.3%
> (0.2572→0.2966), masked PSNR −1.37 dB, deck-LPIPS +11.3%, tex-LPIPS +17.6%, giants
> +1.6pp / low-opa +3.9pp → GATE FAILED on every appearance leg.** Root cause (loss
> curves): the PlanarGS lambda VALUES don't transfer to our mask-sum+median
> normalization — **L_rn term ~0.34–0.72 × λ0.2 ≈ 0.07–0.14 > the whole photometric
> loss (~0.09)**; total loss triples + PSNR crashes the instant L_rn activates @10k.
> Lever is mechanically sound (flatness real); needs λ recalibration (λ_rn ÷10≈0.02,
> λ_rd ÷2.5, keep λ_p). Evidence: `_infusion/wpf4_RECORD.md` + `qa/wpf4_ab_{off,on}/` +
> `_infusion/wpf4_ab_{flatness.json,floaters.txt}`.
Port ONLY the losses (its base trainer is graphdeco — we keep gsplat MCMC). **Geometric prior:
our MoGe-2 cache** (reuses WP-F1 infra, metric depth, one fewer heavy dep) [PC]; running their
DUSt3R-based `run_geomprior.py` as-in-paper is a fine fallback if MoGe-prior results diverge.
Plane instance maps: derive from our existing SAM 2.1 masks (downstream consumes only
masks/boxes/geometry — class labels never used [V]); skip GroundingDINO entirely (avoids its
Windows `_C` build). Port order:
1. **L_p co-planarity** (needs only label map + rendered depth + inv-K): per instance, regularized
   LS plane fit on unprojected rendered depth `param=(PPᵀ+0.01I)⁻¹P·1`, reconstruct
   `plane_depth = 1/(paramᵀK⁻¹ũ)`, masked L1 to rendered depth. λ=0.5 @14k [V]. Verify whether to
   detach the fit [PC].
2. **L_rd / L_rn** on the MoGe cache (conf>thresh × inverse-Canny edge mask × per-view
   depth_weight from alignment residual): λ_depth=0.05 @7k, λ_normal=0.2 @20k [V].
3. **L_dn** last/optional (needs a rasterized Gaussian normal map — smallest-scale axis of R
   flipped toward camera, rendered as 3 extra channels; ~15 lines) — or skip initially.
4. Plane-mask builder `core/plane_masks.py`: SAM 2.1 instances → geometric inspection
   (plane-distance outliers thresh 0.05 · K-means k=5 normal split · containment/overlap/largest-CC,
   min 2000 px) → `priors/plane_mask/{stem}.npy`. PlanarGS's `lambda_scale=100` flatten: SKIP
   (erank hinge covers anti-needle; aggressive flatten biases MCMC relocation sampling [PC]).
Respect the stagger (7k/14k/20k) — it exists to let coarse geometry form.
**Gate:** Laptop deck + a scene with real walls/table: planar regions visually flat + polished
(normal maps clean), deck LPIPS/depth improves, textured regions untouched (region metrics),
no MCMC instability (loss curves + floater report).

═══════════════════════════════════════════════════════════════════════
### WP-B2 — gsplat `main` AOT rebuild in an isolated venv (added 2026-10-17, RESEARCH_REFRESH §6 #1)
═══════════════════════════════════════════════════════════════════════
> **STATUS: ✅ CLOSED 2026-10-18 — GATE PASSED.** gsplat main **1.6.0 @ `512d366b6707`** built AOT
> for sm_120 into `D:\REPOS\HYPERcloneenvs\gsplat_main` (`--system-site-packages`; torch 2.11.0+cu128
> inherited, system gsplat 1.5.3 untouched). Windows needed 4 MSVC source patches
> (`tools/patches/gsplat_main_512d366_win.diff`), `BUILD_EXPERIMENTAL=0`, and a one-header overlay
> for a CUDA 12.9 `clusterlaunchcontrol.h` LLP64 bug (`tools/patches/cuda129_win_overlay`, via
> `NVCC_PREPEND_FLAGS`). Recipe: `tools/blackwell_session_2026-06/build_gsplat_main.bat`. **Our trainer
> runs unmodified** and reproduces the Laptop production stack within run-to-run noise: masked LPIPS
> **0.2791** (main) vs 0.2794 / 0.2800 (two gsplat-1.5.3 runs, same argv + seed 0), PSNR 23.008 vs
> 23.037 / 23.000, identical fog/needle stats, indistinguishable off-path sheets; MCMC + Default
> strategy smokes pass. No wall-clock gain on this LPIPS-bound recipe (8195 s vs 8211 s). API: only
> `gsplat.spherical_harmonics` broke (now `(deg, means, viewmats, coeffs)`) -> shim
> `core/training/gsplat_compat.py::spherical_harmonics_dirs` (tested in both envs); MCMC
> `inject_noise` is now a native CUDA kernel (same maths, different RNG stream; new
> `noise_injection_stop_iter`). Production stays on 1.5.3; the venv is where PPISP (WP-F5) runs.
> Evidence: `_infusion/wpb2_RECORD.md`, `_infusion/wpb2_laptop_gate.json`, `qa/opreg0_laptop_{or0,or0r,b2}/`.

═══════════════════════════════════════════════════════════════════════
### WP-F5 — PPISP exposure/WB/vignetting/CRF — 2–3 days
═══════════════════════════════════════════════════════════════════════
Apache-2.0 [V, triple-checked]. Install v1.2.1 (NOT the v1.0.0 pin in gsplat PR #858):
`TORCH_CUDA_ARCH_LIST=12.0 pip install "ppisp @ git+https://github.com/nv-tlabs/ppisp@v1.2.1"
--no-build-isolation` in a vcvars64 shell; **apply PR #7's two-line `py::return_value_policy::move`
patch first (issue #6: ~6 MB GPU leak per forward — OOMs long runs)** [V]. Pure-PyTorch reference
at `tests/torch_reference.py` = escape hatch if the Windows build fights.
Wiring (reference: `tools/reference_sources/gsplat_pr858.diff`): apply ISP to the rendered image
IMMEDIATELY after rasterization, before loss: `rgb = ppisp(rgb, resolution=(W,H), camera_idx=c,
frame_idx=k)`; add `ppisp.get_regularization_loss()`; `create_optimizers()` (Adam lr 2e-3,
eps 1e-15, warmup 500) **+ `create_schedulers()` — MANDATORY** (forward reads `last_epoch`).
Controller distillation at 80% of steps: freeze splats AND **manually stop MCMC noise/relocation**
(v1.5.3 lacks `noise_injection_stop_iter` — add an explicit `if step >= act_step: skip
strategy.step_post_backward/noise` branch; freezing grads does NOT stop MCMC noise [V]).
Eval: `frame_idx=-1` (controller predicts exposure/color from the render), real `camera_idx`.
**Export gotcha** [V]: ISP is image-space — baked PLY = mean-anchored radiance MINUS per-camera
CRF/vignetting; ship `export_ppisp_report()` JSON + apply a chosen frame's exposure/color + CRF
as a viewer LUT; `camera_idx=None, frame_idx=None` renders the exact baked-PLY appearance.
Fallback: fused-bilagrid CURRENT MAIN (MIT; MSVC narrowing bug fixed only after the PR-858-pinned
commit [V]); drop-in for our vendored `lib_bilagrid.py` (same signatures [V]).
**Gate:** on an auto-exposure clip (capture one with a phone if none on hand), PPISP beats
bilateral-grid and no-compensation on held-out masked metrics + visual patchiness; smoke-test 3k
steps watching `torch.cuda.memory_allocated` (leak check).

> **STATUS: ✅ CLOSED 2026-10-18 — BUILT (`ppisp 1.3.0` @ `a247368` sm_120 in `venvs/gsplat_main`), WIRED (`--ppisp`, `--ppisp_controller`, `--ppisp_controller_start`, default-OFF), GATED ACROSS 3 SCENES × 4 ARMS (`base`, `bil`, `pp`, `ppc`).**
> - **Build & leak check:** upstream `v1.3.0` already merged PR #7 (`py::return_value_policy::move`); built via `tools/blackwell_session_2026-06/build_ppisp.bat`. 3,000-step GPU memory slope `-0.00177 MB/step` (`-4.90 MB` phase-matched delta, `_infusion/wpf5/leakcheck.json`).
> - **Gate 1 (`pp` vs `bil` on exported PLY — PASS 3/3 scenes):** PPISP decisively eliminates the severe unanchored colour/brightness drift of `--bilateral_grid` (`lib_bilagrid.py` has only TV smoothness, no identity anchor):
>   - **Laptop:** mLPIPS `0.3154 -> 0.2805` (**-11.0 %**), mPSNR `19.31 -> 22.69 dB` (**+3.39 dB**), deck LPIPS `0.2684 -> 0.2422`.
>   - **GPU_V4:** mLPIPS `0.4498 -> 0.3724` (**-17.2 %**), mPSNR `18.22 -> 24.12 dB` (**+5.90 dB**), shiny LPIPS `0.4008 -> 0.3045`, floaters `53 -> 17`.
>   - **360 (WP-O5 rig):** tiled mLPIPS `0.5928 -> 0.5536` (**-6.6 %**), mPSNR `17.05 -> 20.95 dB` (**+3.90 dB**), >3× bbox outliers `3,962 -> 3,649`.
> - **Gate 2 (`pp`/`ppc` vs `base`):** All three banked captures have minimal auto-exposure drift (`exp_sd = 0.035–0.059` EV). Under **ISP novel-view rendering** (`frame_idx=-1`), `pp`/`ppc` match or beat `base` on all 3 scenes (Laptop `isp_lpips` `0.2783` vs `0.2790`; GPU_V4 `ppc` `isp_psnr` `24.78` vs `24.74 dB`; 360 `ppc` `isp_lpips` `0.5847` / `18.63 dB` vs `0.5857` / `18.58 dB`). On **raw exported PLYs** (which lack per-camera vignetting + CRF), `base` leads `pp` by `0.0013–0.0026` mLPIPS (`0.31–0.63 dB` mPSNR), while affine colour-corrected structure (`cc_lpips`) is tied on Laptop/GPU_V4 (`±0.0004`) and **better on 360** (`pp` `0.6235`, `ppc` `0.6223` vs `base` `0.6251`). On raw-PLY NR/consistency/fog metrics, `pp` beats `base` on Laptop CrossScore (`0.8667` vs `0.8648`), NOVA (`0.1169` vs `0.1209`), RAFT-warp (`0.0394` vs `0.0404`), and off-path `fog_layer_alpha_mean` (`0.0464` vs `0.0498`, `prune_dMAE` `0.183` vs `0.333`); on GPU_V4 `fog_layer_alpha_mean` (`0.0790` vs `0.0841`) and RAFT-warp (`0.0139` vs `0.0154`, `ppc` CrossScore `0.8387` vs `0.8273`); on 360 >3× bbox outliers (`3,649` vs `4,393`, **-16.9 %**).
> - **Controller (`--ppisp_controller`, `ppc`) & CRF baking probe:** Freezing Gaussians at 80% (`step 24,001`) helps ISP-rendered novel views (`isp_psnr` `ppc > pp` on all 3 scenes) but hurts raw PLY LPIPS by `0.0014–0.0067` because static PLYs forfeit 6,000 fine-phase Gaussian steps and cannot run the controller MLP. Baking the learned CRF into SH (`_infusion/wpf5_crf_bake_probe.py`) hurts PLY PSNR/LPIPS (`22.69 -> 21.84 dB`) because CRF is non-commutative with alpha compositing and decoupled from vignetting.
> - **Policy:** Keep `--ppisp` default-OFF for fixed-exposure studio orbits delivering raw PLYs; use `--ppisp` (not `--bilateral_grid`, and without `--ppisp_controller` when delivering static PLYs) whenever photometric compensation is needed.
> - **Evidence:** `_infusion/wpf5_RECORD.md`, `_infusion/wpf5/{leakcheck,laptop_gate,laptop_tone,laptop_tone_crfbaked,gpu_gate,gpu_tone,360_tone}.json`, `_infusion/wpf5/eval360/scorecard.json`, `_infusion/wpf5/sheets/`.

═══════════════════════════════════════════════════════════════════════
### WP-C1 — GaMO-pattern latent gate (completion lane v1) — WSL — after WP-Q1
═══════════════════════════════════════════════════════════════════════
> **RE-JUDGMENT INPUT (2026-07-02, `_infusion/wedge_rejudgment_RECORD.md` + ARTIFIXER_INVESTIGATION
> §re-judgment):** the June wedge was **100% observed** — an angular-EXTRAPOLATION test, NOT a
> coverage hole; June's GT-LPIPS verdict was legitimate. Product-metric re-score: ArtiFixer3D-bidir
> wedge = qualified win (6/9: cleanliness/NR up, **TSED/RAFT consistency DOWN**); easy case 9/9.
> → ArtiFixer3D-bidir is a **sharpen-observed tool only**; the causal/generative completion path
> (this WP + Annex-A probe + GENA3D) stays on the critical path for true holes; the opacity-mask
> gate is mandatory so enhancement can't inject consistency regressions onto observed pixels.
> **WP-C1's gate testbed must be a GENUINE hole** (back-of-object holdout or slab-deletion void) —
> the Laptop wedge cannot provide one.
> **STATUS: v1 ✅ CLOSED 2026-07-02** (evidence `_infusion/{hole_testbed_RECORD,wpc1_RECORD}.md` +
> `qa/hole_{baseline,gated,ungated}/` + `_infusion/hole/`). (1) **Genuine-hole testbed built**:
> GPU_V4 footprint-aware depth-column deletion → observed 80.6% on a 23-view arc (`wedge:293:316`);
> pure view-withholding REFUTED on our dense domes (Laptop 100%/GPU_V4 95% hemisphere coverage —
> deletion is the only controllable void). (2) **Gate mechanism live**: `core/enhancement/
> latent_gate.py` (backbone-agnostic) + WSL `_infusion/gamo_fill.py` on **SDXL-Inpaint** (FLUX/conda
> envs don't exist on this machine — see backbone note in wpc1_RECORD). **2D gate integrity
> decisive: gated 55.8 dB observed-pixels vs ungated 33.5 dB (37×)**. (3) **Completion loop passes
> vs base**: fills distilled back (arc held out of real supervision) → observed% 80.6→100, CrossScore
> +0.061, RAFT better, TSED 1.0 held, fixed-mask observed-LPIPS (TILED, H1-corrected) IMPROVES
> 0.4791→0.4722 (gated<ungated 0.4805 → gate leg holds). (4) **New finding**: real-view-dominant
> distill is itself a secondary gate — splat-level gated≈ungated (ungated rescued by 364 real views);
> the latent gate's mandatory role = the 2D guarantee + fill-dominant/low-overlap/2D-delivery regimes.
> **CLOSEOUT (2026-07-03, review fixes — `wpc1_RECORD.md` §CLOSEOUT):** H1 — the banked fixed-mask
> LPIPS was accidentally FULL-FRAME (bbox crop = whole frame around the central-void annulus);
> re-banked TILED (tile256/cover0.8, full res) into `gate_splat_*.json`, gate leg intact. H2 —
> a NO-FILL CONTROL distill (identical config, zero fills, 364 real views) reaches observed% 100
> AND BEATS both fills on every splat-level axis (tiled-LPIPS 0.4648, CrossScore 0.7697, NOVA
> 0.2116): on this dense-dome/real-view-dominant testbed MCMC-regrowth+grazing views (v0060 93.9%,
> v0540 90.8%) achieve the whole completion — generative fill adds no splat-level value here; a
> never-captured hole is needed to make it strictly necessary. Floater four-way: SOR-outlier
> 0.11–0.24% (all clean, content not haze). M2 — injection off-by-one (`t`→`timesteps[i+1]`) fixed
> in gamo_fill.py; banked fills kept (feather paste-back masks the effect). **M3 ratification: the
> beats-ungated gate leg holds on the gate-integrity axis (2D 55.8 vs 33.5 dB; corrected tiled LPIPS
> gated < ungated) and NOT on splat-level plausibility scores (ungated edges gated on
> CrossScore/NOVA/RAFT there — the distill-as-secondary-gate regime finding). Closure ratified on
> that basis; the latent gate remains mandatory for the 2D structural guarantee and
> fill-dominant/low-real-overlap/2D-delivery regimes.** Next: FLUX.1-Fill / Wan-I2V / causal-ArtiFixer
> backbone swaps into the same scripted loop; a fresh never-captured-hole capture for a strict test.
**Re-implement the ~50-line mechanism on our WSL diffusers stack, copying freely from their
pipeline source** (`gamo/my_diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_
multiview_inverse.py`). The reason not to run GaMO as-shipped is technical, not legal: its env
pins torch 2.5.1 (no sm_120 kernels), SD2.1/MVGenMaster backbone + deepspeed/lietorch, 3 conda
envs [V] — our Wan stack is the right host. Mechanism (verified from their pipeline source): render alpha → `mask = (alpha < 0.6)` → **max-pool** (not
bilinear) to latent grid (8× for SD; 4t×8s 3D-pool for Wan/LTX video VAEs), binarize ≥0.5 →
progressive dilation (`F.max_pool2d` k=5, ×2/×1/×0) → VAE-encode the render; at trigger points
30/50/70% of denoising: noise render-latent to current level (DDPM `scheduler.add_noise`; for
flow-matching (FLUX/Wan) use `scale_noise` / `(1−σ)x₀+σε`), composite `x = m·x + (1−m)·x_known`
(widest mask first), 3-iteration RePaint resample harmonization; feathered pixel paste-back
(INSET 0.6, FEATHER 0.08). Implement as `callback_on_step_end` or a copied denoise loop.
Backbone order: **Wan2.1 (already local)** first; FLUX.1-Fill stills variant second (already our
default Type-3 inpainter). Integration: this becomes the gated engine inside
`core/enhancement/` next to autocomplete (InFusion seed stays the faithful-reconstruction arm);
unify the mask source with WP-Q1's visibility field.
**Gate (WP-Q1 metrics, wedge testbed):** filled wedge beats base render AND beats ungated
generation on {CrossScore, NOVA, TSED, RAFT-warp} with observed-region masked-LPIPS UNCHANGED
(the gate's whole point). Compare against the ArtiFixer3D wedge result re-judged in WP-Q1.

═══════════════════════════════════════════════════════════════════════
### WP-C2 (optional) — GENA3D object-back completion — WSL — after WP-C1
═══════════════════════════════════════════════════════════════════════
License-absence no longer blocks (POC stance). Remaining risk is technical [V]: TRELLIS
sparse-attention accepts only xformers/flash_attn backends — test xformers-cu128 memEff on sm_120
first, else patch `gena3d/modules/sparse/attention/full_attn.py` to an SDPA segment loop; spconv
likely needs a source build (`TORCH_CUDA_ARCH_LIST=12.0`). Weights HF `Colezwhy/GENA3D` (~6 GB,
auto-download); clone Pi3 manually to `dependency/Pi3`; patch the DINOv2 `force_reload=True`;
needs an external 2D amodal frontend (OAAD) for `amodal_completion.png` inputs. Output =
canonical-unit-cube Gaussians (`save_ply` → standard 3DGS PLY) → similarity-register back into
the scene. **Gate:** beats WP-C1 on the same held-out backs under WP-Q1 metrics.

═══════════════════════════════════════════════════════════════════════
### WP-D1 — Delivery pass (+ WP-S2 synthesis: delivery-time object extraction)
═══════════════════════════════════════════════════════════════════════
> **STATUS: ✅ CLOSED 2026-07-05 — GATE PASSED (both arms).** New `core/delivery.py` (CLI, default
> full-scene; `--extract_object` opt-in) + `--rgb_object_masked` trainer lever (default OFF).
> splat-transform v2.7.1 verified on Blackwell (WebGPU adapter 0; filter-floaters/-cluster/SOG all
> run — `-g cpu` fallback not needed; `--decimate N%` keeps N% [V]). Deliverables (SuperSplat-openable):
> `deliverables/{gpu,laptop}_{full,object}.sog` (+ `.ply`). **(a) FULL-SCENE:** decimation to ~29–34%
> of the 3M master is NEAR-LOSSLESS (reduced SH3: +1.1–1.3% masked-LPIPS, PSNR≈master); shipped SH2
> deliverable within a **+5% LPIPS bound** (GPU +2.8%, Laptop +4.4%). **Self-refit is OFF by default
> for full-scene**: on the WP-S1 LPIPS-tuned masters the L1+SSIM refit trades LPIPS for PSNR (GPU
> +2.07 dB PSNR but LPIPS worse; refit-SH3 0.4155 > decimate-only 0.4124) — it can't "recover ≥half
> the decimation loss" in LPIPS because decimation is already ~lossless. **(b) OBJECT-EXTRACT:** clean
> cut — multi-view BiRefNet-mask voting → object 99.98–99.99% spatially compact (66–77 stray Gaussians
> of ~0.5–0.9M beyond 0.5×scene-scale), out-of-mask alpha ≈0.025 (GPU) vs 0.99 master control; the
> object-masked self-refit makes the object-region metrics BETTER than the master's object region
> (obj-LPIPS −0.0105/−0.0008, obj-PSNR +4.2/+2.4 dB). Objects ship SH3 (SH2 was the whole LPIPS
> regression on the glossy laptop lid). Caveat: Laptop 2D-mask bleed (0.05) is a BiRefNet
> under-segmentation artifact on ~2–4 outdoor frames (spatial-floater metric confirms clean). Evidence:
> `_infusion/wpd1_RECORD.md` + `deliverables/*_delivery.json` + `qa/wpd1_*` (scorecards, strips,
> refit-variant JSONs). Tests: 146 passed. THIRD_PARTY.md updated (splat-transform MIT).

`npm install -g @playcanvas/splat-transform` (node ≥18, MIT, v2.7.1 [V]). New `core/delivery.py`
+ Stage-10 button: run `floater_report.py` → prune.py →
`splat-transform -w in.ply --filter-nan --filter-floaters 0.05,0.1,0.004 --filter-cluster
--filter-harmonics 2 --decimate 60% --morton-order out.sog -i 15` → 2–5k-step self-refit on
original views (PUP recovery pattern) → QA scorecard (WP-Q1) attached to the deliverable.
GOTCHAS [V]: `--filter-floaters` voxel edge is in WORLD units (0.05 default) — rescale per scene
scale; `--filter-value` uses raw PLY column semantics (run `-m/--summary` first); verify SOG
orientation visually on first export; `-g cpu` fallback if the Node WebGPU adapter misbehaves on
Blackwell. **Gate:** decimated SOG vs master PLY on the QA suite — define the acceptable delta;
self-refit recovers ≥ half of any decimation loss.

═══════════════════════════════════════════════════════════════════════
### WP-X1 — Master-splat re-test (>3M) — after WP-Q2a — 1 day of runs
═══════════════════════════════════════════════════════════════════════
> **STATUS: ✅ CLOSED 2026-07-02** (experiment-only, no code; evidence `_infusion/wpx1_RECORD.md` +
> `qa/wpx1_cap*/`; DEVELOPER_NOTES §6 amended). **Verdict: 3M ceiling RE-CONFIRMED under stable
> conditions; divergence mechanism FIXED.** 4-arm ladder (3/4/6/8M, identical recipe + erank,
> seed 0): masked PSNR 25.05→24.86→24.03→21.16, monotonic dilution (dead Gaussians 31%→63%);
> 8M trains to completion, no OOM (13.2 GB), CLEANEST floater metrics — failure mode changed from
> catastrophic to benign. Mechanistic: 8M's sag recovered +3.5 dB when noise injection stopped at
> refine_stop → residual >3M harm is MCMC-noise-driven. **One cheap untested lever remains: a
> `--noise_lr` trainer flag** (gsplat default 5e5 was used; not exposed). Keep cap 3M; revisit
> only with noise_lr exposed or on a capture that genuinely starves 3M.
Re-run cap_max 4M/6M/8M on GPU_V4 WITH erank hinge + raised scale_reg (0.02) + lowered MCMC
noise_lr, coarse-to-fine. The 8M divergence mechanism was giant-blob tile blowup [V] — erank +
scale bound attack exactly that. Judge at fixed eval res by masked-LPIPS + floater metrics + VRAM.
Honest expectation: modest single-object gain; the point is an unconstrained MASTER splat for
delivery-decimation (Hyperscape lesson). Gate: >3M beats 3M on the QA suite or the ceiling is
re-confirmed and recorded in DEVELOPER_NOTES §6.

---

## 3. Sequencing summary

Week 1: WP-Q1 (harness core: masks + masked metrics + CrossScore) ∥ WP-Q2 (erank + eps2d, 1 day)
∥ WP-F1 (MoGe cache).
Week 2: WP-Q1 finish (TSED/RAFT/NOVA/NVS-SQA + scorecard + baseline bank + RE-JUDGMENT) ∥ WP-F2
(dense init).
Week 3: WP-F3 (DIM) ∥ WP-X1 (master-splat runs in background) ∥ WP-C1 start (gate prototype on Wan).
Week 4: WP-F4 (planar) ∥ WP-C1 finish (wedge validation).
Week 5: WP-F5 (PPISP) ∥ WP-D1 (delivery pass) → integrate: full imperfect-capture dry-run on a
fresh phone-video capture, triage→rescue→deliver, scorecard attached.

**Top risks (cross-WP):** (1) unpublished DIM hyperparams — mitigation: relative parameterization
+ 4-point grid + fallback path; (2) provenance drift — keep the THIRD_PARTY.md
ledger current so the funded rebuild knows exactly what to re-license/re-implement; (3) PPISP leak + MCMC-noise-freeze
interaction — patch PR#7 + explicit noise stop + memory smoke test; (4) hloc dense-match
underdelivery — raise max_kps, density report gate before integration; (5) loss stacking
destabilizes MCMC — one loss at a time, per-loss scalars, `--no_priors` A/B always available.

---

═══════════════════════════════════════════════════════════════════════
### WP-S2 — Object-mode background study (✅ CLOSED 2026-07-04, GPU_V4)
═══════════════════════════════════════════════════════════════════════
> **STATUS: ✅ CLOSED 2026-07-04 — GATE PASSED (clear per-mode recommendation).** Verified the data
> mode FIRST: GPU_V4 training frames are **FULL_SCENE** (11.4% near-black, corner-lum 89.5 — confirms
> the WP-G1 correction), so **arm (a) full-scene = CURRENT PRODUCTION (control)** and object-only is
> the NEW thing under test. Masks: BiRefNet 'general' (headless `core.masking`, SAM2 not installed),
> 569 frames, mean 20.7% coverage, orbit-wide spot-check clean. 3 arms × 15k, seed 0, cap 1M, factor
> 2, WP-S1 stacked config (dense_init 882k + mono 0.05 + erank + gated-LPIPS + pose_opt). New lever
> `--mono_depth_object_masked` (default OFF) for correct object-mode depth sup. **RESULT: object-only
> LOSES on object-region quality — obj-LPIPS 0.2277→0.2824 (+24%), obj-PSNR 23.5→20.9 (−2.6 dB),
> obj-SSIM 0.737→0.667; alpha-loss (c) does NOT recover it (c≈b).** The black-opaque-floater hazard
> barely materializes: of 882,902 dense-init pts, 421,347 (47.7%) are background/black-supervised, but
> MCMC+opacity_reg drive them to NEAR-TRANSPARENT (low-opa 58% vs a's 12%) not black-opaque — arm (b)
> WITHOUT alpha has only **7** black-opaque out-of-hull GS (a has 4,740 legit COLORED bg); alpha (c)
> → 0, a marginal cleanup. Pose-drift regression CONFIRMED (modest): object-only +33% translation
> drift (background anchors pose_opt). **RECOMMENDATION: OBJECT scans → KEEP background at training
> (full-scene = prod); do NOT switch to object-only. Alpha-loss lever = leave OFF (unnecessary on this
> recipe). WORLDS mode unaffected.** Evidence: `_infusion/wps2_RECORD.md` + `wps2_obj_metrics.json` +
> `wps2_floater_analysis.json` + `qa/wps2_*`. Deviations: 15k/cap-1M/no-C2F fast A/B (not 30k prod),
> single scene/seed (GPU_V4 = the one scene with a real dark background).

**Product context (user, 2026-07-03): HYPERclone has two functions — scanning WORLDS and scanning
OBJECTS. Both sample captures are objects → background is a means (SfM anchors), not a deliverable.**
Pipeline has the masking stage (Stage 8, after SfM), but **WP-G1 discovered production training
frames are FULL SCENES (correction 2026-07-04)** — so full-scene IS current prod and object-only
is the NEW mode under test. Three-arm A/B on GPU_V4 (production/stacked recipe,
seed 0, 15k): (a) full-scene training (= current prod, control) · (b) object-only (pre-multiplied
masks, NEW) · (c) object-only + `--alpha_loss_lambda 0.1` (the shipped-but-never-production-tested
black-opaque-floater killer, commit 99781f5). Evaluate object-region masked metrics (mask-scoped
QA), floater report (esp. black-opaque floaters vs (b)), background-region artifact check on (a),
and NEW-LEVER INTERACTIONS: does `--dense_init` seed useless background points in object mode
(count in/out of object mask); do MoGe priors degrade on black-bg frames (F1 conf/mask stats
per arm). GATE: a clear recommendation per mode — object scans: keep/drop background at training +
alpha-loss verdict; worlds mode unaffected. Evidence per §0.

═══════════════════════════════════════════════════════════════════════
### WP-G1 — Glossy/specular lever (PHASE 1 CLOSED; scope decided by WP-S1's residual diagnosis)
═══════════════════════════════════════════════════════════════════════
> **STATUS: PHASE-1 CLOSED 2026-07-04 — 3DGS-DR ceiling probe = GATE FAILS on both scenes →
> do NOT port 3DGS-DR (phase-2 negative).** Ran the UPSTREAM 3DGS-DR (2404.18454, deferred
> reflection) on our two scenes vs the WP-S1 stack on the specular crops. Built its 4 CUDA
> submodules AOT for sm_120 (torch 2.11 untouched) — new trap logged: `.cu` files pulling
> `torch/extension.h` → `compiled_autograd.h` `C2872 'std' ambiguous` under nvcc+MSVC; fix =
> pure-ATen shim in the 6 nvcc files (mirrors gsplat). **Region-LPIPS (the gate) is WORSE for
> 3DGS-DR, not better: Laptop deck 0.2739→0.3675 (+34%, + broken outdoor background — out of its
> bounded-scene domain); GPU_V4 backplate 0.3473→0.3580 (+3.1%) WITH env_scope.** 3DGS-DR does
> win region-PSNR on GPU_V4 (+4.55 dB = better semi-gloss *tone*) but that is NOT a perceptual
> (LPIPS) win — it softens etched microtext + blooms the chrome logo; the glossy laptop lid stays
> mushy. No sharp view-consistent reflection. Judged by LPIPS (never PSNR alone, §0) = no win.
> **Fallback per the research ranking below: (1) Spec-Gaussian ASG (MIT, color-function-only swap
> into gsplat MCMC — RECOMMENDED next specular lever, stays in our stack, no domain fragility);
> (2) accept-limitation + generative polish (GeoQuery/SwiftVR bake-off, also fixes the shared
> microtext softness). 3DGS-DR full port NOT recommended.** ⚠️ Data correction found this WP:
> the `gsplat_prod/data/images_2` frames are **FULL SCENES, not black-bg pre-multiplied** (both
> stack + DR render full backgrounds) — contradicts DEVELOPER_NOTES; **WP-S2 must verify the real
> production data mode before its background A/B.** Evidence: `_infusion/wpg1_RECORD.md` +
> `qa/wpg1_{laptop,gpu}/scorecard.json` + strips; THIRD_PARTY.md updated.

Research verdict (2026-07-03, verified primary-source): **"Apple SHARP" (2512.10685, apple/ml-sharp)
identified and REJECTED** — single-image→3DGS for near-view synthesis, explicitly no view-dependent/
reflective modeling (its own limitations section), torch 2.8 pin conflicts; name collision with our
need, skip. Specular ranking for single glossy objects on gsplat MCMC: (0) zero-cost sanity A/B
first — `app_opt=True` + confirm `sh_degree` saturation (flags already in gsplat); (1) **highest
ceiling = 3DGS-DR port** (2404.18454, github.com/gapszju/3DGS-DR — the ONLY specular lever with
public code + object-centric validation: Helmet/Bell/Ball +3.5–6 dB, LPIPS ~halved; medium port:
deferred shading + per-pixel normal propagation kernels, sm_120 AOT rebuild discipline applies);
NASGabor deprioritized (sm_120 buildability unverified, built on Beta-Splatting not MCMC);
Ref-DGS/MaterialRefGS/RGS-DR have NO code; EnvGS = incompatible ray-tracing backend.
Generative-sharpening bake-off candidates (nothing has verified FINAL-SPLAT evidence — judge like
WP-C1): **GeoQuery** (2605.12399, code + gsplat example released, +1.09 dB over Difix3D+ verified
in its Table 1) and **SwiftVR** (2606.09516, Wan2.2-5B, documented plain-SDPA). ArtiFixer repo
unchanged since Jun; no 1.3B variant exists. FlashVSR: check its "locality-constrained sparse
attention" against the sm_120 landmine before any use.

═══════════════════════════════════════════════════════════════════════
### WP-P1 — Generative-polish bake-off (CLOSED 2026-07-05 — GATE FAILS all candidates)
═══════════════════════════════════════════════════════════════════════
> **STATUS: CLOSED 2026-07-05 — NO released enhancer polishes our current-best (dense, well-observed)
> stack → honest negative; do NOT adopt a generative polish stage for this regime.** Bake-off on the
> WP-S1 Laptop STACK, judged so frame-level gains only count if they survive into a splat. No-op bar:
> masked-LPIPS 0.3096 / deck 0.2739 / lid 0.2472. **GeoQuery (2605.12399, MIT, gsplat-native):
> feasible on sm_120 (hand-rolled local attn, no flash-attn; smoke-verified), but FRAME-LEVEL NEGATIVE
> (+10–19% LPIPS, robust across factor-2 & 960p) and SPLAT-LEVEL GATE FAIL — distilled into the stack
> it REGRESSES masked-LPIPS 0.3096→0.3245 (+4.8%; deck +1.3%, lid +5.5%) while raising PSNR/tsed
> (smoother, not perceptually faithful). This is (to our knowledge) the FIRST splat-level evidence for
> GeoQuery = a clean negative on dense captures.** **SwiftVR (2606.09516, Apache-2.0, Wan2.2-5B):
> feasible on sm_120 (EMPIRICALLY smoke-restored, 6.94 fps; new trap: diffusers 0.36 eagerly imports a
> broken system flash_attn_3 → venv-local try/except guard → SDPA), but FRAME-LEVEL STRONGLY NEGATIVE
> (+32–59% LPIPS; hallucinates a denser/distorted on-screen checker) → splat distill not run (foregone
> larger regression).** **ArtiFixer3D-bidir (incumbent): architecturally REPLACES the splat (builds its
> own 720p 3DGRUT), cannot ingest our gsplat ckpt; its ~0.306 masked-LPIPS (720p) only ≈ our no-op at
> half res = not a win; plausible-not-faithful on well-observed views.** **FlashVSR: excluded (block-
> sparse attn = sm_120 killer).** ROOT CAUSE = regime mismatch: these tools target sparse/degraded/
> compressed inputs with heavy artifacts; our stack is near-artifact-free, so their re-synthesis trades
> perceptual fidelity for smoothness/hallucination. Generative enhancement stays reserved for the
> COMPLETION lane (obs%<100, GT invalid), NOT polish of well-observed splats. Residuals → in-stack
> Spec-Gaussian ASG (WP-G1 phase-2) + higher train res. Evidence: `_infusion/wpp1_RECORD.md` +
> `qa/wpp1_*` + `qa/wpp1_geoquery/scorecard.json`; THIRD_PARTY.md updated.

═══════════════════════════════════════════════════════════════════════
### WP-O — 360°/Insta360 capture lane (added 2026-08-04) — plan: `docs/PLAN_360.md`
═══════════════════════════════════════════════════════════════════════
> New input modality: Insta360 X4/X5 360° video (test asset:
> `D:\HYPERCLONE\FOOTAGE\FlowerShop\FLOWERSHOP.mov` — in-camera dual-fisheye 2×1920²,
> 92 s; low-res proxy-class, original .insv/8K requested). Full analysis, verified
> capability audit, architecture decision, capture protocol and gates live in
> **`docs/PLAN_360.md`**. Key verified facts [V, 2026-08-04]: pycolmap 4.1.0 ships
> COLMAP's new EQUIRECTANGULAR model + rig-based `panorama_sfm` pipeline (12
> rig-constrained 90° crops/pano, fixed rig/intrinsics in BA); ERP-direct SfM probe on
> FlowerShop PASSED (40/40 @ 1.22 px, corridor-shaped trajectory, on our own automated
> ffmpeg v360 dfisheye→ERP stitch); gsplat 1.5.3 sm_120 build already exposes
> `camera_model="fisheye"`. NO new CUDA builds needed anywhere in the lane.
- **WP-O1 — 360 ingest module** (~2–3 d) — **✅ GATE PASSED 2026-08-04**: `core/ingest_360.py`,
  `--capture-type insta360` default-OFF; auto-detect dual-fisheye vs stitched ERP;
  window-relative sharp-frame sampling (absolute cull stays dead); v360 stitch; SAM 2.1
  operator mask + nadir/seam bands. Gate: **151** ERP frames @3840×1920 from `_077.insv`
  (sharpness p10 454 / median 1691), operator masked on **20/20** random frames, seam+nadir
  bands present; 8K `_027.insv` → **7680×3840, 94 frames, same code** (105 s);
  `FLOWERSHOP.mov` auto-detects (184 frames); 22 new tests (suite 151→**173**).
  Evidence `_infusion/wpo1_RECORD.md` + `_infusion/wpo1/`. Findings for WP-O2/O3: PLAN_360
  §7's asset labels are wrong (the flower shop is only in the low-res `FLOWERSHOP.mov`), and
  bystanders are unmasked (operator only).
  **Review fixes 2026-08-04** (gate not re-run, defaults unchanged; RECORD §6): re-run guard
  (non-empty `out_dir` without `--overwrite` → exit 2, writes nothing), clean exit-1 +
  `INCOMPLETE` marker + no report on mid-pipeline ffmpeg/SAM failure, and
  `--operator_stream {0,1}` — **the lens/stream order is an ASSUMPTION verified on the
  2026-07-21 clips only; WP-O2 must spot-check one frame per stream for any new
  camera/firmware**. Tests 22→32 (suite **183**).
- **WP-O2 — 360 SfM lane** (~2–3 d) — **✅ GATE PASSED 2026-08-04** on ARM A (crop-rig):
  `core/sfm_360.py`, `--capture-type 360` default-OFF, two arms
  (`--arm crop_rig|spherical`), WP-O1 masks composited into COLMAP's exclusive crop masks,
  `--use_masks 0` no-priors path, colmap#3599 watchdog (heartbeat + `--timeout_s` → exit 3;
  **the hang did not reproduce**, so the vendored-upstream path shipped). Gate on the 151-pano
  `_077` dataset: **151/151 registered (100%)**, mean reproj **1.2313 px** (crop model) /
  **1.3419 px** (after `convert_to_equirectangular()`), 1 reconstruction, corridor-shaped
  track with **0** steps > 5× median. **A/B: ARM B spherical also hits 151/151 at 1.2694 px
  ERP in 20.7 min vs ARM A's 4 h 14 m — accuracy is a dead heat and ARM B is nominally ahead
  in the one comparable unit; what ARM A actually buys is 3.7× the points (622k vs 167k) and
  training-ready pinhole crops.** ih_fov sweep = **negative result**: 196/198/200 differ by
  less than COLMAP's own run-to-run noise (mapper is NOT bit-reproducible at fixed seed; sd
  ≈0.003–0.008 px) and the 20-frame winner (198) FLIPS at 151 frames — WP-O1's default stays
  200, but the delivered poses are on the 198 re-ingest, so **WP-O3 must use
  `_dataset/360/flowershop_077_ihfov198`**. Other findings: panos are **not upright** (zenith
  tilt median 11°/p90 27°/max 49° — PLAN_360 §6's Horizon-Lock rule now has a number); ceres
  CHOLMAP-not-positive-definite warnings reproduce at scale in BOTH arms (157 ERP vs 36
  crop-rig) with **no** quality cost; pycolmap wheel is **without CUDA** so all SIFT is CPU.
  28 new tests (suite 183→**211**). Evidence `_infusion/wpo2_RECORD.md` + `_infusion/wpo2/`
  (A/B scorecard, poses CSVs, rig JSON, trajectory plots, sweep + noise JSON); upstream
  snapshots in `tools/reference_sources/colmap_panorama/`.
- **WP-O3 — 360 training lane** (~2–3 d) — **✅ GATE PASSED 2026-08-05** on ARM 2, after
  ARM 1 (the production config) **FAILED** it. `core/train_360_prep.py` (`--capture-type 360`
  default-OFF) packages WP-O2 ARM A → trainer layout (flat names in file layout AND model,
  trainer mask-name convention, pano-grouped split): **1 812 crops = 1 584 train / 228 test
  (19 panos × 12, zero intra-pano leakage)**. Two default-OFF trainer flags:
  `--holdout_list` (explicit test-set membership for view GROUPS the every-Nth rule cannot
  express) and `--loss_mask` (average L1 + the SSIM **map** over the VALID pixels instead of
  compositing over black). 21 new tests (+3 in the ea5758f housekeeping commit; suite 211→**235**).
  **⚠ HEADLINE NEGATIVE — `--opacity_reg 0.01` (a production default) DESTROYS this scene.**
  ARM 1 ran 30 k steps to completion with a normal-looking loss and a full 3 M cap, and
  produced a model whose every render is a flat colour: held-out masked LPIPS **0.6949**,
  masked PSNR **13.86**, mean opacity **0.00025**, **99.87 %** of Gaussians dead, means blown
  to **±2.1 e6** (camera-cloud radius = 1). An 8-arm × 5 k-step controlled sweep isolates it:
  arm G (`opacity_reg 0`, `scale_reg 0.01`) healthy, arm H (`opacity_reg 0.01`,
  `scale_reg 0`) 99.9 % dead — **`opacity_reg` alone is the cause, `scale_reg` is harmless.**
  Mechanism: 1 584 views ⇒ each Gaussian is rendered rarely, so the every-step opacity
  penalty beats the photometric term; gsplat MCMC then injects **maximum** noise at
  near-zero opacity by design (`op_sigmoid(1-opacity)`, k=100, x0=0.995) → positional
  runaway. `--rgb_object_masked` amplifies it (÷3 loss magnitude on 33 %-valid crops + black
  boundary edges) into an outright rasterizer memory blow-up (19.6→57.9 GB). **This is NOT
  360-specific — expect it on any large many-view walkthrough** (scope caveat, review
  2026-08-05: mechanism-based inference from ONE scene under `--strategy mcmc`; not yet
  replicated on a second capture or under `--strategy default`).
  **ARM 2** (`--opacity_reg 0 --loss_mask --max_steps 200000`, else identical): 200 k steps,
  **74 min 32 s**, 3 M Gaussians, mean opacity 0.555, dead 21.3 %, ‖mean‖ p99 4.20 — healthy.
  Held-out (228 crops): **masked LPIPS 0.5222** (tiled) / 0.5572 (bbox), masked PSNR
  **17.483**, SSIM 0.4334; full-frame 0.5587 / 16.720 / 0.4406. **Step-count ablation inside
  ARM 2: at the same 30 k budget the fix alone gives +3.34 dB / −20.5 % LPIPS; the next 170 k
  steps add only −5.5 % LPIPS and +0.29 dB** ⇒ the fix is nearly the whole win, the long
  schedule buys perceptual sharpness only. Visual gate all PASS: **no operator ghost** (he
  fills the GT of the most-excluded held-out crops and is simply absent from the render),
  **no seam-band phantom geometry** (crops 5/7 look straight down the v360 seam), corridor
  walkable (`walkthrough.mp4`, 151 poses). Splat is structurally correct but **soft** —
  capture-limited by the 1 920²/lens proxy source, which re-arms PLAN_360 §7's 8K request.
  Honest negatives: the skin-tone bystander heuristic is useless here (cream paper ⇒ 228/228
  false positives — bystander check is visual over ~25 crops only, not exhaustive); the
  held-out split is INTERPOLATION (neighbours ~0.4 units away stay in training) so the
  numbers are optimistic; ARM 2 moves 3 levers at once (the 5 k sweep isolates two of them).
  Evidence `_infusion/wpo3_RECORD.md` + `_infusion/wpo3/` (both arms' metrics/scorecards,
  22 figures, MP4, `ckpt_health.json`, `diag_collapse{,2,3}.json`); PLY
  `_dataset/360/train/wpo3_run_arm2/ply/point_cloud_199999.ply`.
- **WP-O4 (optional)** — fisheye-native A/B via gsplat `camera_model="fisheye"` +
  OPENCV_FISHEYE 2-cam rig; adopt only on clear win vs WP-O3.
- **WP-O5 — 8K validation + free-viewpoint quality** — **✅ GATE PASSED 2026-08-05**
  (SfM half outright; training/free-view half with a bounded claim). Commissioned after the
  user's Postshot verdict on the WP-O3 `_077` splat: UNUSABLE free-view ("hot mess... only from
  key angles everything lines up") while its interpolative holdout said LPIPS 0.5222 — exactly
  the failure the WP-O3 record warned its split could not see.
  **SfM (8K `_027`, 94 panos @ 7680×3840 → 1 128 crops of 1920² f=960 — `core/sfm_360.py` is
  confirmed resolution-agnostic): 94/94 panoramas registered (100 %, all 12 crops each), mean
  reproj 1.2096 px crop / 1.3308 px ERP, 1 reconstruction, 63 min 46 s = 4.0× faster than
  WP-O2's 4 h 14 m at 4× the pixels per crop.**
  **⚠ The SIFT-parity guard FAILED for ALIKED+LightGlue, so SIFT shipped.** New default-OFF
  `--features aliked` (+ `core/sfm_360_hloc.py`: HLoc ALIKED/LightGlue/NetVLAD into the same
  COLMAP DB, our own mask filter standing in for COLMAP's `MaskFeatures`, and
  `pycolmap.geometric_verification(rig_verification=True)` — hloc's own verifier takes no rig
  options). Controlled to **exactly 11 476 identical pairs** in every arm: ALIKED is +0.166 px
  reproj (~20× the mapper's noise floor), worse in **every** track-length stratum, worse still
  on a matched track population (+0.46 px), unchanged at native resolution (hypothesis
  refuted), and **2.3× slower** end to end. **The speed premise that motivated the WP is also
  false: SIFT extraction at 8K is 198.7 s of a 3 826 s run (3 %) — WP-O2's cost was matching +
  mapping, never SIFT.** Honest counter-evidence: ALIKED yields **4.0× the points / 3.8× the
  inlier matches**, and the two stacks **agree on camera poses to 0.08 of an inter-panorama
  step**, so the regression is narrow.
  **Training: 4 arms × 30 k** (A baseline / B +pose_opt / C +opacity_reg 1e-3 / D +erank
  0.01@7k; **D added mid-WP on measurement**, recorded as a deviation).
  **⚠ HEADLINE NEGATIVE #1 — `--opacity_reg 0.001` DESTROYS THE MODEL TOO; the safe value is
  0, not "small".** Arm C reached **99.9 % dead / mean opacity 0.0077 / ‖mean‖ p99 36.8 by step
  10 000**, then the rasterizer climbed 4.6 → 61.7 GB and died with a CUDA illegal memory
  access at ~11 500 (exit 3). It had the GPU to itself, which upgrades WP-O3's hedged "the
  crash may have been contention" into a clean mechanism claim.
  **⚠ HEADLINE NEGATIVE #2 — post-hoc PRUNING cannot fix free view.** On the rejected `_077`
  splat and on all three healthy 8K arms, deleting every sub-0.05-opacity Gaussian (18–33 % of
  the model) leaves the render **visually identical at every viewpoint**; deleting the needles
  instead (52 % of `_077`) is catastrophic **even on-path** — they *are* the surface. Only
  training-time levers are admissible. New `tools/prune_ply.py` (PLY reader proven bit-exact
  against `gsplat.export_splats`; opacity/bbox/erank pruning; `splat_stats` reports
  `needle_frac_erank_lt_2` / `erank_mean` / `aspect_ratio_median`).
  **Interpolative metrics cannot rank the arms** — masked PSNR 21.469 / 21.366 / 21.456 and
  LPIPS 0.5505 / 0.5499 / 0.5517 for A/B/D, all within noise, while D has **4.5× fewer
  needles**. So `_infusion/wpo5_viewstab.py` adds a GT-free **viewpoint-stability** metric
  (LPIPS between consecutive renders of a small lateral sweep; valid only because on-path
  fidelity is equal): **arm D wins 8/8 sweeps (−3.2 %, p ≈ 0.004); `--pose_opt` is a measured
  NO-OP (3/8, +0.00001)** — closing the WP-O3 hand-off's ranked-first question with a negative.
  8K vs the `_077` proxy: masked **PSNR 17.483 → 21.469 (+3.99 dB)**, SSIM **0.4334 → 0.7140**.
  **Free-view verdict (bounded, honest): the usable envelope is now ≈ ±1–1.5 m around the
  capture path** (at lateral ±1.0 m = 1.67 inter-panorama steps rooms are readable and walls
  planar in A/B/D, where `_077` degenerated to paint-smear) **but a genuinely unvisited
  viewpoint 4.62 m off-path is still incoherent in every arm.** Final arbiter: the user in Postshot.
  **Also found: the 8K theatre yields a 10.1× sparser SfM cloud per panorama than `_077`
  (corrected per review 2026-08-05; raw total-point ratio was 16× but conflates capture length)**
  (38 450 pts / 94 panos vs 622 325 / 151). Detection is identical (~4 000 kpts/crop);
  **MATCHING collapses — 39.2 → 4.7 inliers per verified pair.** Texture-poor repetitive scene
  and/or SIFT at 1920², not isolated. **Highest-value follow-up: the ALIKED-init A/B** (≈5×
  denser init here) — the parity guard's accuracy criterion may be the wrong criterion for
  this class of scene.
  **ADDENDUM 2026-08-05 (user retested arm D in Postshot: still "hot mess... none of this is
  usable... maybe I'm just not finding the right spaces"):** `tools/align_ply.py` bakes ONE
  lossless similarity transform into the model — gravity-aligned, floor at 0, recentred,
  metric (**4.2286 m/unit** from the measured 0.15372-unit median station step and a stated
  0.65 m/step assumption), **up = −Y** (`y_down`, the INRIA/gsplat/COLMAP convention Postshot
  itself exports). It rotates means, scales, quats **and the spherical harmonics** — SH via
  per-band least squares against **gsplat's own `spherical_harmonics`**, so no Wigner-D and no
  convention to get wrong. **Verified lossless: min PSNR 74.2 dB / mean abs diff 7.8e-6 over
  4 views** rendered from correspondingly transformed cameras. Gravity needed three
  estimators — the mean camera up is the mean STICK direction and is **systematically 8.3°
  off**; the trajectory-plane normal seeds an iterated RANSAC floor fit, which puts camera
  height at a plausible **2.15 m** median (vs an impossible 2.76 m before). Two recorded
  negatives: a point-height-entropy gravity search wanders 50° onto a wall axis, and the 12
  rig crops sharing an optical centre make `median(diff(camera_centres))` **0**, which
  silently produced `scale = 1.0` until `distinct_centres()` was added. Plus
  `_infusion/wpo5_guide.py` → `GUIDE.jpg`: metric top-down floor plan + 8 numbered
  best-supported stops with renders (bearing rule corrected — "most points in front" picks the
  nearest WALL every time) and `README_POSTSHOT.md`. Aligned PLYs at
  `_dataset/360/train/wpo5_run_D/ply/point_cloud_29999_aligned{,_pruned_op05}.ply`.
  **This makes the splat level and navigable; it does not add parallax — the ±1.5 m envelope
  is unchanged.**
  28 new tests (suite 235 → 263; count corrected per review 2026-08-05), +14 in the addendum
  (**277**). Evidence `_infusion/wpo5_RECORD.md` + `_infusion/wpo5/`
  (3 SfM scorecards, reproj-split, trajectory-agreement, train_runs, viewstab, 28 off-path
  sheets, 8 needle-probe sheets, fog pairs, 2 MP4s). PLYs (gitignored): arm D
  `_dataset/360/train/wpo5_run_D/ply/point_cloud_29999{,_pruned_op05}.ply`; retro-pruned `_077`
  `_dataset/360/train/wpo3_run_arm2/ply/point_cloud_199999_pruned_op05.ply`.

═══════════════════════════════════════════════════════════════════════
### WP-SH4 — SH degree 4 (25 coeffs/ch, 72 `f_rest`) — PHASE A & PHASE B ✅ CLOSED 2026-10-18
═══════════════════════════════════════════════════════════════════════
> **STATUS: PHASE A (readiness) ✅ 2026-10-17 · PHASE B (quality A/B) ✅ CLOSED 2026-10-18 (`KEEP_SH3_DEFAULT_OPT_IN_SH4_WARM_START_ONLY`).**
> The trainer default stays `--sh_degree 3`; `--sh_degree 4` is supported end-to-end and recommended
> **only via fine-phase warm-start (`--init_ckpt`)** when targeting the SH4 Unreal Engine plugin.
> Evidence: `_infusion/wpsh4/phaseA_gate.json`, `_infusion/wpsh4/phaseB_gate.json`,
> `_infusion/wpsh4_phaseB_RECORD.md`, `_infusion/wpsh4_phaseB_eval.py`.
> Tests: `tests/test_sh_degree4.py` (19 passed, incl. `resize_shN` pad/truncate), `test_sh_ply_io.py`
> (23), `test_spz_io.py` (13), `test_sh4_splat_transform_guard.py` (7).
>
> **Review 2026-10-17 (phase A + WP-D2): PASS-with-fixes.** The spz `splat-types.h` flipSh[23]
> band-4 sign bug was **independently confirmed** by the reviewer. Fixes applied (all defaults
> unchanged; trainer/backend untouched): (1) `core/delivery.py` prints a WARNING with count + %
> when the SPZ drops out-of-range Gaussians, prints the SPZ clip fractions, and lists every
> warning in the delivery JSON (`warnings`) and again at the end of the run; (2)
> `core/spz_io.write_spz` record gains `clip_frac` {scales outside [−10, 5.9375] log, sh0
> |c| > 3.333, shN outside [−1, 1)} (fraction of values, measured on the packed RUB data) and
> `warnings` (drops; scale clip > 0.5 % — note the WP-D2 smoke models at 1.6–2.0 % WILL warn);
> (3) new `tests/test_sh4_splat_transform_guard.py` (7 tests, `run_splat_transform` stubbed):
> delivery hands splat-transform only 45-f_rest PLYs (exact values) from a 72-f_rest master and
> still writes the 72-f_rest `_sh4.ply`; `gui2/preview.export_viewer_sog` hands it one exact
> degree-2 (24 f_rest) PLY from a degree-4 source with no `--filter-harmonics`; drop/clip
> reporting through `write_spz` and the delivery manifest/stdout. **The preview test found a
> real bug:** the degree>3 path wrote `<out>.sh2.ply` before creating `out_sog`'s directory
> (FileNotFoundError on a fresh output dir) — fixed by creating the directory first; (4)
> `phaseA_gate.json` `splat_transform.version` was "" (the binary prints `--version` to stderr):
> capture fixed in `_infusion/wpsh4_phaseA_gate.py`, and the JSON back-filled without re-running
> the gate (`--patch_version`, adds `version_note`) = **splat-transform v2.7.1 (b42b60c)**;
> (5) splat-transform degree-4 wording softened to what was measured (below). Tests **338
> passed** (331 + 7).

**A1 — kernel (gsplat 1.5.3 AOT sm_120).** CUDA `spherical_harmonics(d, dirs, coeffs)` vs the
torch oracle `_torch_impl._spherical_harmonics` (`_eval_sh_bases_fast`), N=4096 random fp32,
unit and un-normalised dirs, forward + grads wrt coeffs and dirs. Max abs error, degree 0/1/2/3/4:
fwd 0 / 3.6e-7 / 9.5e-7 / 1.9e-6 / **3.8e-6** (raw dirs 5.7e-6; outputs O(5)); d_coeffs
0 / 2.4e-7 / 9.5e-7 / 1.4e-6 / **3.8e-6**; d_dirs relative ≤ **5.5e-7** at every degree. Also:
band-4 oracle basis orthonormal on the sphere (Gram max|G−I| 2.3e-7, 200k-point quadrature);
CUDA basis = oracle basis column-by-column; oracle passes float64 `gradcheck` at degree 4;
evaluating degree d<4 on a K=25 tensor ignores (and sends zero grad to) the extra coeffs.
**Gate: fp32 tolerance — PASS.** Trainer (read only): `init_gaussians_from_sfm` allocates
`(sh_degree+1)^2-1` → **24** rest coeffs at `--sh_degree 4` (`gsplat_trainer.py:615`, asserted
in a test); the trainer renders at the full degree from step 0 (no progressive-SH schedule).

**A2 — PLY I/O + rotation.** `f_rest` is channel-major (INRIA) through `gsplat.export_splats`,
`tools/prune_ply.read_gsplat_ply` and `qa_report.load_splats` — asserted with per-(coeff,
channel) tagged values at degree 4. `tools/align_ply.py` SH rotation is now degree-agnostic
(degree read from the model; was `SH_DEGREE = 3` at :61 and a hard `!= 16` check): D is the
numerically-solved real-SH Wigner-D (lstsq per band over a Fibonacci sphere) on gsplat's float64
oracle basis, CPU, no GPU needed. Measured: f'(Rd) vs f(d) evaluated by the **CUDA kernel**
(independent of D's basis) max error 2.4e-7 / 6.0e-7 / 1.7e-6 / **3.3e-6** at degree 1..4;
off-band blocks exactly 0; per-band orthogonality 2.9e-15; D(R1R2)=D(R1)D(R2) to 1e-10; band 1
equals the closed form P R Pᵀ to 1e-12; a degree-4 model aligns losslessly in a real render
(<5e-3) and the render breaks (>5e-2) if the SH rotation is skipped. Degree bookkeeping
centralised in new `core/sh_utils.py` (exact, raises on non-(d+1)²−1 counts); callers fixed:
`core/delivery.py::sh_degree_of` (rounded silently), `gui2/preview.py` live preview (fragile
quadratic formula) and `_ply_sh_degree`, `qa_report.py --sh_degree` **default now = the
model's own degree** (was 3 → a degree-4 model was silently scored truncated; still 3 for every
production model, so scorecards are unchanged), `gui2/app.py` QA tab SH field → dropdown
default "auto". New `tools/truncate_sh_ply.py` (+ `core.sh_utils.truncate_ply_file`): exact
byte-level truncation, other properties bit-identical, identity truncation byte-identical;
render(truncate(m,3), deg 3) == render(m, deg 3) **exactly (0.0)**. Round-trip test: synthetic
deg-4 PLY → align → PLY → truncate to 3 → reload == expected bit-exactly, and align∘truncate =
truncate∘align (1e-5).

**Finding (negative, load-bearing): splat-transform v2.7.1 does not handle degree-4 PLYs.**
Measured on the smoke model (72 f_rest; splat-transform v2.7.1 (b42b60c)): passthrough
lossless; `--filter-harmonics 3` is a **no-op** (keeps all 72 coefficients, values unchanged);
`--filter-harmonics 2` → **51** f_rest and `0` → **27** (f_rest_45..71 pass through) — **invalid
property counts**, not a valid degree; SOG from a degree-4 PLY keeps **3 bands** (`meta.json`
`bands: 3`). Which 45 coefficients the SOG writer takes was not measured; by its degree-3
layout assumption it would read f_rest_0..44, which in a degree-4 channel-major file are not
the degree-3 subset (inference, not measured). The vendored SuperSplat viewer
(`gui2/viewer/index.js:35022`, reads ≤45 f_rest) has the same assumption (code reading).
**Rule: truncate with our tool before any splat-transform/SOG/viewer step.**
`core/delivery.py` now does this automatically for a degree>3 master (exact truncation to 3
for the splat-transform path, plus `<name>_<mode>_sh<d>.ply` = pruned full-degree master);
`gui2/preview.py` SOG preview truncates degree>3 sources to 2 itself. Degree≤3 paths unchanged.
Delivery smoke on the degree-4 smoke ckpt (`_infusion/wpsh4/delivery_smoke_*`): 500k → prune
446k → SOG/PLY at SH2 (265k), `_sh4.ply` 153.5 MB, SPZ SH4 21.6 MB, rc 0.

**A3 — smoke train** (`_infusion/wpsh4/run_smoke.sh`; Laptop `gsplat_prod/data`, factor 4,
5000 steps, MCMC cap 500k, `--opacity_reg 0.01 --scale_reg 0.01` explicit, seed 0;
output `_dataset/Laptop/apex_images/training/wpsh4_smoke/sh{4,3}`): `--sh_degree 4` runs end to
end, exports **72 f_rest** (172.0 MB PLY vs 118.0 MB at SH3, +46 %), logged peak 2952 vs
2942 MB. Wall time 113 s (SH4) vs 163 s (SH3) is NOT a speed result — the GPU was shared with
another agent's job (78 GB / 88 % util at launch); measure throughput in phase B on an idle GPU.
Held-out PSNR (23.35 vs 23.18) is recorded, not judged: 5k steps, no A/B protocol.

**Trainer diff (APPLIED 2026-10-18):**
(a) `--sh_degree` → `choices=[0,1,2,3,4]` (`gsplat_trainer.py:2068`).
(b) `--init_ckpt` warm-start: `core.sh_utils.resize_shN` pads `shN` with zeros (or truncates) to
`--sh_degree` after checkpoint load (`gsplat_trainer.py:1088–1104`), so a converged SH3 master can
be fine-tuned at SH4 with zero MCMC divergence at step 0.
(c) `--sh_degree_interval N` (default `0` = render at full degree from step 0): progressive SH
schedule `min(step // N, sh_degree)` (`gsplat_trainer.py:1510–1515, 2070`).
(d) `gsplat_backend.py:741–743` uses `core.sh_utils.sh_degree_of(splats)` instead of the inline
quadratic formula.

**Phase B — Quality A/B Gate Results (`_infusion/wpsh4_phaseB_RECORD.md`, `_infusion/wpsh4/phaseB_gate.json`, 2026-10-18):**
- **Contiguous 38° arc holdout (`--holdout_list`):** 32/299 views (`37.8°`) on `GPU_V4`
  (`_infusion/wpsh4/phaseB/gpu_holdout_arc.txt`) and 32/304 views (`37.3°`) on `Laptop`
  (`_infusion/wpsh4/phaseB/laptop_holdout_arc.txt`).
- **Experiment B1 (5,000-step fine-phase warm-start from 30k SH3 masters, strict unseen arc holdout):**
  - `GPU_V4`: SH4 improves **specular top-10% PSNR by `+0.203 dB`** (`20.435 → 20.638 dB`, spec LPIPS
    `0.4227 → 0.4226`) and overall masked PSNR by `+0.068 dB` (`23.621 → 23.688 dB`, mLPIPS
    `0.3717 → 0.3716`), while the truncated degree-3 Postshot twin is **lossless** (`23.679 dB`,
    `+0.058 dB` vs native SH3).
  - `Laptop`: SH4 improves **specular top-10% PSNR by `+0.089 dB`** (`16.705 → 16.794 dB`), but
    widens the train–holdout PSNR gap by `+0.259 dB` (`+2.201` vs `+1.942 dB`); truncating band 4
    back to degree 3 on held-out arc views **jumps masked PSNR by `+0.132 dB`** (`21.416 → 21.547 dB`),
    confirming band-4 angular overfitting across the 38° unseen sector on 304-view orbits.
- **Experiment B2 (15,000-step C2F from-scratch arc-withheld on `GPU_V4`, `--sh_degree_interval 1000`):**
  - SH4 matches SH3 on held-out mLPIPS (`0.4150` vs `0.4153`) and improves off-path `fog_alpha`
    (`0.0449` vs `0.0522`), **but co-adapts bands $L \le 3$ and $L=4$**: truncating the from-scratch
    SH4 PLY to SH3 for Postshot drops held-out mPSNR by **`-0.500 dB`** (`23.663 → 23.164 dB`) and
    specular PSNR by **`-0.880 dB`**.
- **Costs:** PLY size `+46 %` (`708.5 → 1,032.9 MB` at 3M); throughput `-6 %` to `-14 %`; SPZ v4
  (8/8 bits) round-trip PSNR `56.60 dB` (`GPU_V4`) and `51.84 dB` (`Laptop`), both `≥ 50 dB` (`PASS`).
- **Verdict:** Keep `--sh_degree 3` as the production default; use `--sh_degree 4` **only via
  fine-phase warm-start (`--init_ckpt`)** when targeting the SH4 Unreal Engine plugin.

═══════════════════════════════════════════════════════════════════════
### WP-D2 — SPZ v4 writer (+ SH cap per consumer) — ✅ 2026-10-17 (UE/Postshot import pending)
═══════════════════════════════════════════════════════════════════════
> **STATUS: GATE PASSED (round trip at quantization level, deg 3 and 4); consumer import
> checks still open.** Default OFF: `python -m core.delivery ... --spz [--spz_sh_degree D]
> [--spz_sh1_bits 8 --spz_sh_rest_bits 8]`, or standalone `python -m core.spz_io in.ply
> out.spz [--sh_degree D]`; `qa_report.py` now also loads `.spz`. Library: Niantic
> **nianticlabs/spz @ affd0ec** (MIT, lib 1.1.0, format v4) Python bindings built locally
> (`tools/spz/build_spz.bat`; NOT the unrelated PyPI `spz` 0.0.1). Evidence
> `_infusion/wpsh4/phaseA_gate.json` (`spz` section) + `tests/test_spz_io.py`.

Gate (real 500k-Gaussian smoke models, 12 held-out Laptop views at factor 4, gsplat render
original vs decoded at full degree):

| model | SH bits (b1/b2+) | PLY → SPZ | size ratio | render PSNR mean / min |
|---|---|---|---|---|
| SH4 | 5/4 (spz default) | 172.0 → 10.7 MB | 16.1× | 30.41 / 28.53 dB |
| SH4 | **8/8 (our default)** | 172.0 → 24.1 MB | **7.1×** | **53.83 / 52.27 dB** |
| SH3 | 5/4 | 118.0 → 10.1 MB | 11.7× | 33.69 / 31.16 dB |
| SH3 | 8/8 | 118.0 → 18.3 MB | 6.4× | 54.11 / 52.29 dB |

Per-attribute max error equals the format's quantization bound exactly (derived from
`load-spz.cc`, asserted in tests): position 1.22e-4 (=2⁻¹³), log-scale 0.03125 (in range),
sigmoid-alpha 1.96e-3 (=1/510), DC 0.01307, SH 0.0039 (8 bit) / 0.0352 band-1 + 0.0664 bands
2–4 (5/4 bit), quaternion ≤0.22°. SPZ's defaults (5/4) cost ~20 dB render PSNR — **the SH
bucketing is the whole loss**, larger at SH4 (more 4-bit coeffs) → our default is 8/8.
Format limits found and handled: (i) positions beyond ±2048 units **wrap** (24-bit
`& 0xffffff`), so `write_spz` drops them, records the count and warns (delivery prints and
records the warning since the 2026-10-17 review; clip fractions are in the record's
`clip_frac`) (8 of 500k far floaters on the
SH4 smoke, 3 on SH3); (ii) log-scale clips to [−10, 5.94] (1.6–2.0 % of these models' scale values fall
outside it — tiny, sub-pixel Gaussians; renders unaffected at 53 dB); (iii) SH clips to [−1, 1) (0 % here,
p99 |c| 0.11–0.13); DC clips beyond ±3.33 (0.07–0.09 %).

**Upstream bug found — degree 4 coordinate conversion.** spz's converter (`splat-types.h`
flipSh[23]) gives the band-4 m=+4 (cos 4φ) coefficient the sign of `y`; gsplat's basis parity
says it is even in x and y (sign 1). So spz's own RDF→RUB conversion mis-signs exactly that
coefficient at degree 4 (pinned by `test_upstream_spz_converter_band4_bug_is_pinned`); bands
0–3 are right. `core/spz_io.py` therefore performs the RDF↔RUB frame change itself (exact ±1
parity signs computed from gsplat's basis, cross-checked against align_ply's Wigner-D) and packs
with `UNSPECIFIED`. **Consequence for our UE plugin:** if it decodes SPZ through Niantic's
library with a coordinate conversion, its band-4 m=+4 coefficient will be mis-signed too
unless it applies the same fix — check on the first import. Consider reporting upstream.

Consumer recipe (delivery): degree-4 master → `<name>_<mode>_sh4.ply` + SPZ v4 SH4 (UE
plugin); exact degree-3 twin via `tools/truncate_sh_ply.py` (Postshot; 0.686× the master PLY);
SOG/web stays ≤SH3 (SOG v2 cap). For a degree>3 request delivery writes the SPZ from the
pruned master (the decimated path is SH-capped by splat-transform) and records the source.
Open: (1) UE plugin round-trip of the SPZ and the 72-f_rest PLY (frame + SH sign check);
(2) Postshot import of the SH3 twin (and whether Postshot rejects/garbles a 72-f_rest PLY).

═══════════════════════════════════════════════════════════════════════
### WP-C3 — TIDI-GS Cross-View Evidence, Importance & Isolation Floater Pruner — ✅ CLOSED 2026-10-18
═══════════════════════════════════════════════════════════════════════
> **STATUS: GATE PASSED across all 3 production scenes (`Laptop`, `GPU_V4`, `360`).**
> Default OFF: standalone CLI `python tools/prune_tidi.py --ckpt ... --data_dir ... --out_ply ...`
> and delivery opt-in `python -m core.delivery ... --prune_tidi --tidi_data_dir <dir> [--tidi_data_factor F] [--tidi_mode balanced|conservative|aggressive]`.
> Evidence: `_infusion/wpc3_tidi_RECORD.md`, `_infusion/wpc3/tidi_gate.json`, `_infusion/wpc3_tidi_eval.py`,
> `tests/test_prune_tidi.py` (`4 passed`).

- **Root Cause Discovered ("Mip-Splatting Zombie Dust"):** Under `--antialiased` (Mip-Splatting 2D filter)
  + `--scale_reg 0.01` + `--opacity_reg 0`, `--scale_reg` shrinks unused Gaussians' 3D scales to
  `1e-8..1e-7`. Mip-Splatting's 2D amplitude compensation scales effective 2D opacity by
  $\sqrt{\det(\Sigma_{2D}) / \det(\Sigma_{2D} + \epsilon_{2D} I)} < 1/255$, so `gsplat` culls them
  (`radii == 0`, `proj_views == 0`) in every view. Once culled, they receive zero gradient while
  keeping raw `sigmoid(opacity) >= 0.05` (`0.05..0.50`), escaping both MCMC relocation (`op <= 0.005`)
  and `tools/prune_ply.py` (`op < 0.05`). They account for **20.3 %** (`GPU_V4`), **28.5 %** (`Laptop`),
  and **48.4 % (`1,452,071` splats!)** of the 3M budget on the 360 walkthrough.
- **Exact Forward Alpha-Blending Accumulation:** Using a 1-channel dummy color $c_{i,0}=1$ with
  `render_mode="RGB+ED"` and backpropagating $\sum_p \text{renders}[p,0]$ gives exact forward
  transmittance-weighted alpha contributions $w_i(v) = \sum_p T_{p,i}\alpha_{p,i}$ (`colors.grad[:,0]`)
  in 1 pass/view (~7 s on 267 views; ~24 s on 1,260 views).
- **3-Scene Gate (`balanced` mode vs `prune_op05`):**
  - **Laptop:** `2,198,875 → 1,551,279` splats (**`-29.5 %` vs `prune_op05`**, `-48.3 %` vs raw 3M;
    PLY `519.2 → 366.1 MB`), held-out mPSNR `+0.0004 dB`, mLPIPS `+0.0003`, residual mid-opacity
    fog `0.0365 → 0.0000`.
  - **GPU_V4:** `2,018,622 → 1,602,034` splats (**`-20.7 %` vs `prune_op05`**, `-46.6 %` vs raw 3M;
    PLY `476.6 → 378.1 MB`), held-out mPSNR `0.0000 dB`, mLPIPS `+0.0002`, residual mid-opacity
    fog `0.0497 → 0.0000`.
  - **360 (FlowerShop 8K rig):** `2,412,922 → 793,497` splats (**`-67.1 %` vs `prune_op05`**,
    `-73.5 %` vs raw 3M; PLY **`569.7 → 187.3 MB`**), held-out 144-crop mPSNR `-0.0008 dB`,
    mLPIPS **`0.0000`**, residual mid-opacity fog `0.1558 → 0.0000`, `>10×` bbox outliers `488 → 0`.

═══════════════════════════════════════════════════════════════════════
### WP-SfM2 — COLMAP / pycolmap 4.2.1 + Native GPU `LOMA_B` Evaluation — ✅ CLOSED 2026-10-18
═══════════════════════════════════════════════════════════════════════
> **STATUS: GATE PASSED (`PASS`).** Isolated in `D:\REPOS\HYPERclone\venvs\colmap421` (`pycolmap 4.2.1`
> + standalone CUDA/ONNX `colmap.exe 4.2.1` commit `bd1fcf6` at `venvs/colmap421/colmap_bin/bin/colmap.exe`);
> system `pycolmap 4.1.0` untouched. Evidence: `_infusion/wpsfm2_RECORD.md`, `_infusion/wpsfm2/gate.json`,
> `_infusion/wpsfm2_worker.py`, `_infusion/wpsfm2_eval.py`.

- **B1 — 40-frame ERP-Direct Probe (`EQUIRECTANGULAR`):** `4.2.1` incremental mapper registers `40/40`
  frames (`100 %`), increases verified inliers by **`+7.6 %..+8.2 %`**, increases 3D points by
  **`+6.8 %..+7.3 %`** (`27,256 → 29,105`), and maps **`1.26×` faster** (`11.63 s` vs `14.70 s`),
  with `0.0094 m` (`0.097°`) Sim3 agreement against `4.1.0`.
- **Negative Finding (`core/sfm_360.py:725` TODO resolved — DO NOT enable `guided_matching=True` on ERP):**
  In `4.2.1`, `#4054` sets `config = PLANAR_OR_PANORAMIC` (`6`) with `F = nullopt` on 194/780 pure-rotation
  ERP pairs; `MatchGuided` (`sift.cc:829`) then fails `CHECK(two_view_geometry.F.has_value())` on pair
  `(1, 10)` and aborts the process. Keep `guided_matching` **OFF** on `EQUIRECTANGULAR`.
- **B2 — 151-frame Spherical ERP DB & B3 — 94-pano (1,128-crop) WP-O5 Rig:** `151/151` and `1,128/1,128`
  registered (`100 %`), reproj error `1.2694 → 1.2583 px` (B2) and `0.7336 px` (B3), Sim3 agreement
  `0.0276 m` / `0.0033 m`. Note: `run_global_mapper` (GLOMAP) does not yet model multi-sensor rig
  extrinsics in its rotation averaging (`39.29°` Sim3 residual on the 12-crop rig — use incremental mapper
  for rigs).
- **B4 — Native GPU `LOMA_B` (20-pano / 240-crop rig):** Standalone `colmap.exe 4.2.1` runs native GPU
  ONNX `LOMA_B` on RTX PRO 6000 (`sm_120`) when `torch\lib` (`cudnn64_9.dll`) is prepended to `PATH`.
  Honors masks (`0` violations), yields **`550,585` rig-verified inliers** (`7.0×` SIFT, `1.86×` ALIKED+LG)
  and **`73,713` 3D points** (`7.55×` SIFT, `1.87×` ALIKED+LG) at `0.0202 m` Sim3 RMSE (`0.110°` rot)
  vs HLoc ALIKED+LG. Neither the wheel nor zip compiles Caspar (`CASPAR_ENABLED=OFF`) or Ceres CUDA BA.

═══════════════════════════════════════════════════════════════════════
### WP-R1 — Observed-Preservation Gate for Generative Supervision — ✅ CLOSED 2026-10-18 (`HOLD_DEFAULT_OFF`)
═══════════════════════════════════════════════════════════════════════
> **STATUS: GATE EXECUTED — `HOLD_DEFAULT_OFF` (Beats `distill_control` on observed regions & NoVA,
> but 3D frustum bleed from dark SDXL inpaint still degrades the partially-observed fringe).**
> Default OFF: `--extra_unobserved_only`, `--extra_anchor_observed`, `--extra_loss_weight`,
> `--extra_obs_tau`, `--extra_obs_gamma` in `core/training/gsplat_trainer.py`, `core/training/gsplat_backend.py`,
> and `run_distill_laptop.py`. Evidence: `_infusion/wpr1_RECORD.md`, `_infusion/wpr1/gate.json`,
> `_infusion/wpr1_eval.py`, `tests/test_wpr1_observed_gate.py` (`6 passed`).

- **3-Zone Fringe Discovery:** Partitioning the 23 hole arc views (`293..315` in `_infusion/hole/`) into
  `solid_obs_99` ($\alpha_{\text{init}} \ge 0.99$, 57.4 %), `fringe_obs_03_99` ($0.30 \le \alpha_{\text{init}} < 0.99$,
  23.2 %), and `unobs` ($\alpha_{\text{init}} < 0.30$, 19.4 %) revealed why WP-C1's `distill_gated` lost to
  `distill_control`: `gamo_fill.py` only inpainted $\alpha_{\text{init}} < 0.30$ and pasted the raw
  `10.86 dB` holed render back onto `fringe_obs_03_99`, actively supervising the fringe to stay broken
  (`12.48 dB`) whereas `distill_control` let the 364 real training views heal it to `18.19 dB`.
- **Results (`distill_r1_gated`: `--extra_unobserved_only --extra_anchor_observed 1.0 --extra_loss_weight 0.25 --extra_obs_tau 0.30`):**
  - Beats `distill_control` on **observed tiled LPIPS (`0.4603` vs `0.4632`)** and **solid-observed
    PSNR (`20.50` vs `20.19 dB`, LPIPS `0.4342` vs `0.4480`)**, and cuts **NoVA artifacts by `-18.0 %`**
    (`0.3004` vs `0.3663`).
  - However, because SDXL Inpaint hallucinated a dark patch (`mean RGB 0.078` vs `GT 0.581`) inside the
    hole, 3D Gaussians spawned in the hole project across adjacent arc views into `fringe_in_box`
    (`11.56` vs `16.51 dB`) and `unobs` (`6.83` vs `15.50 dB`). Keep default-OFF until paired with a
    multi-view consistent 3D-aware completion backbone (`WP-C4`).

═══════════════════════════════════════════════════════════════════════
### WP-B1 Revisit (`RealVDeblur`) & Spirula Studio Recon — ✅ CLOSED 2026-10-18
═══════════════════════════════════════════════════════════════════════
> **STATUS: RECON COMPLETED (`_infusion/wpb1_spirula_recon.md`).**
> - **RealVDeblur (`arXiv:2607.20628`):** Code (`github.com/OpenImagingLab/RealVDeblur` / `RBJin/RealVDeblur`,
>   commit `63a2eb0`, Apache-2.0) and HuggingFace weights (`huggingface.co/RBJin/RealVDeblur`:
>   `realvdeblur_dmd.safetensors` on `Wan-AI/Wan2.1-T2V-1.3B`) are now **publicly available**. Parked
>   only until a motion-blurred video capture arrives in `_dataset/`.
> - **Spirula Studio (`github.com/harry7557558/spirula-studio`, commit `74a92ef`, GPL-3.0):** Confirmed
>   ships Windows/Linux/macOS binaries with a headless CLI (`spirula train --help`) on Vulkan (primary)
>   and CUDA, supporting 360 equirectangular, fisheye, LoMa, and PPISP-style appearance, capped at SH3.

## Annex A — H100 causal-ArtiFixer probe (optional, ~$1, not excluded)
Rent an H100 spot (sm_90: stock FA2/FA3 works). Steps: clone nv-tlabs/artifixer + weights
(67.6 GB — provision download time or use HF cache on the box), copy one prepared scene
(`wsl_wedge_pipeline.sh` shows the prep), run `run_inference` WITHOUT `--inference_pipeline
bidirectional` (causal AR default, `frames_per_block` set) on the SAME wedge scene → bring
frames home → judge with WP-Q1 vs our bidirectional results. Settles: does causal fill gaps +
make 3D+ win (ARTIFIXER_INVESTIGATION §7 Q1)? If YES by a wide margin, revisit the local
cuDNN-SDPA causal build (feasibility already proven: causal+sliding-window correct on sm_120;
remaining piece = attention sink — two-pass LSE merge / cuDNN 9.23 block-mask / window-only test;
see `tools/blackwell_session_2026-06/cudnn_sdpa_test.py`).

## Annex B′ — MiniMax H3 assessment (researched 2026-08-04, day after release — do not build yet)
33B dense DiT (FL2VA + Ref2VA ckpts, Qwen3-VL-32B encoder; ~124 GB bf16; int8/pruned combos
42.5–67 GB fit our 96 GB card without offload). Open release is FULL ATTENTION ONLY (sparse-MSA
withheld) — diffusers path (unmerged PR #14355) plausibly runs plain-SDPA on sm_120 [PC, no
confirmed sm_120 run exists anywhere yet]. Lane verdict: **completion — NO** (zero masked/region
completion capability; no VACE-equivalent; keeps GaMO/SDXL-Inpaint → FLUX.1-Fill/Wan-I2V order
unchanged); **polish — NO** (same risk class that failed WP-P1; Elo #1 "video editing" is
preference, not faithfulness; gate stays closed); **NVS/gap-fill — NO** (text-tag camera only, no
pose conditioning); **causal unlock — NO** (DiT, not AR). Only plausible fit: **restoration of
DEGRADED sources** (e.g. the 360 proxy footage) via its edit-preserve behavior — BUT the
faithfulness layer (H3-Context-IR) is hosted-API-only, not in the open weights. Cheapest gate:
Stage-0 hosted-API probe (~$2, FlowerShop clip, "enhance clarity, keep content identical") →
only if faithful, Stage-1 local pruned-int8 SDPA run in a FRESH venv (never the pinned ones; no
flash-attn of any kind; vllm-omni `[fa4]` extra FORBIDDEN on this machine). License note
(log-only per POC stance): "Community License" with US in Excluded Territories + outputs may not
train other models — record in THIRD_PARTY.md if ever vendored. Watch: sparse-MSA release,
Context-IR open-sourcing, first sm_120 runs, any 3DGS ecosystem uptake (none exists yet).

## Annex B″ — Generative coverage-densification survey (researched 2026-08-05)
User question: can generative models synthesize missing walkthrough coverage (H3+equirect
plugin / Apple SHARP / HunyuanWorld)? Verdicts [researcher, primary-sourced]:
**H3 — REJECT** (the "seamless equirect ComfyUI plugin for H3" does NOT exist — likely
conflated with `cedarconnor/ComfyUI-DiT360`, an experimental FLUX image LoRA node; Ref2VA
preserves subject identity NOT scene geometry; camera control still text-tag only).
**Apple SHARP (2512.10685, apple/ml-sharp) — REJECT** (single image → 1.2M Gaussians <1s;
explicitly designed to NOT extrapolate unseen regions — wrong problem).
**HY-World-2.0 (2604.14268, Tencent, Apr-2026) — bounded WSL spike only**: closest
use-case match ("capture a video, get a digital twin", direct 3DGS out) but flash-attn
hard-pinned (sm_120 risk) and ZERO fidelity evidence.
**Hard evidence vs the naive generate-then-SfM route**: WorldPrompter (2504.02045)
measured up to **89.7% COLMAP failure on generated 360° video**; COLMAP silently drops
inconsistent generated content (WorldScore 2504.00983). Unanchored generation = dead.
**Viable shortlist (anchored densification of REAL captures)**: (1) **SEVA / Stable
Virtual Camera** (Stability, ICCV'25, N≤32 posed imgs + transforms.json → pose-conditioned
novel views; real-indoor GT eval exists; flash-attn-on-Windows question open — half-day
smoke test w/ existing poses); (2) **Difix3D+** (nv-tlabs, CVPR'25 Oral, SD-Turbo
single-step artifact fixer on OUR renders; no flash-attn, gsplat integration, real-capture
LPIPS 0.33→0.26 — same-day smoke, complementary not densifying); (3) HY-World-2.0 env
spike. Closed/no-code: CAT3D/CAT4D/Bolt3D/Wonderland/ReconX/MultiDiff/CamPVG/Pantheon360.
Down-weighting generated views in 3DGS training exists (CAT3D App.D, AugSplat 2606.31556)
but NO paper ablates it and NONE implements visibility-split eval — our WP-Q1 gate is
ahead of the literature. **Meta-finding [V]: arXiv 2607.01556 proves interpolative holdout
understates true 3DGS error by mean 5.84 dB (8.4 dB on generated views) — independent
confirmation of the WP-O5 off-path-eval lesson.** Any pilot must run under the completion
-lane discipline: generated views ONLY where no real observation exists, gated + judged by
visibility-split, never global metrics.

## Annex B — Watch list (do not build; check monthly)
GENA3D training code (inference promoted to WP-C2) · CompleteSplat code (Niantic) ·
ArtifactWorld training set + full release · CHROMA code (ICLR'26 review) · GaussianGPT object
checkpoints · PR-IQA code · GeoQuery (claims +1.09 dB over Difix3D+ [PC]) · gsplat upstream
(3DGUT integration state at our next rebuild) · EDGS CVPR-2026 code update · Casual3DHDR repo ·
Native-ERP 3DGS rasterizers (Seam360GS code, Spherical-GOF release, PFGS360 maturity — see
PLAN_360.md §3; all parked, no sm_120 evidence) · colmap#3599 panorama_sfm hang fix.
