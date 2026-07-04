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
> **STATUS: CODE COMPLETE + reviewed/fixed; GATE RAN 2026-07-04 → FAILED as-specced
> (PlanarGS lambdas), diagnosed + fixable. Recalibration A/B recommended, NOT run
> (GPU handed to WP-S2).** Shipped: `core/plane_masks.py` (plane-instance maps from
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
### WP-D1 — Delivery pass — 1–2 days
═══════════════════════════════════════════════════════════════════════
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
### WP-S2 — Object-mode background study (QUEUED; runs when GPU frees after WP-S1/F4)
═══════════════════════════════════════════════════════════════════════
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

## Annex B — Watch list (do not build; check monthly)
GENA3D training code (inference promoted to WP-C2) · CompleteSplat code (Niantic) ·
ArtifactWorld training set + full release · CHROMA code (ICLR'26 review) · GaussianGPT object
checkpoints · PR-IQA code · GeoQuery (claims +1.09 dB over Difix3D+ [PC]) · gsplat upstream
(3DGUT integration state at our next rebuild) · EDGS CVPR-2026 code update · Casual3DHDR repo.
