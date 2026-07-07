# WP-T1 — Full end-to-end pipeline integration test on a FRESH capture — RECORD

**Capture:** 588 Sony 60.2 MP stills (`DSC01203.JPG`…`DSC01790.JPG`, 9504×6336, EXIF
orientation 1 landscape, ~17 MB each, 9.9 GB) at `D:\HYPERCLONE\CAM TEST\SONY` (OUTSIDE
the repo). All outputs under `D:\REPOS\HYPERclone\apex-sfm\_dataset\SONY_TEST\`.
**Do NOT commit** (`_dataset/` is gitignored; this record + `qa/wpt1_*` are the committable evidence).

## What the capture IS
Product-photography orbit of an **ASICS GEL-KAYANO 14 running shoe** (silver/white mesh
upper, metallic-green medial panel, reflective silver TPU overlays, white laces, dark
rubber tread sole). The shoe is clamped on a thin dark rod / wooden-base stand with a
white 3D-printed fixture; the camera orbits a **full sphere** around the fixed mounted
shoe (sole-from-below, top-down, both side profiles). Background is a **fixed but heavily
defocused** studio room (cabinets, LED light strip, wood floor, office chair) — shallow
depth-of-field bokeh. Indoor. Single rigid mounting (SfM-friendly static scene, moving camera).
**Stack-relevant challenges:** (1) semi-gloss metallic/specular surfaces (the WP-S1
specular-failure case), (2) thin structures (laces + mount rod → needle Gaussians, erank
lever target), (3) shallow-DoF bokeh background (mushy but view-consistent).

## Environment
RTX PRO 6000 Blackwell sm_120 96 GB, driver 576.57, CUDA 12.9; torch 2.11.0+cu128; system
Python 3.11; gsplat AOT sm_120. GPU idle at start (1.8 GB desktop apps only). Sequential jobs.

---

## STAGE 1 — import/QC  (ROBUSTNESS FINDING)
Kept ALL 588 frames (no auto-cull); QC sharpness probe run separately to REPORT cull behavior.

**Metric** = pipeline's own auto-cull metric (90th-percentile of 8×8-tile Laplacian variance,
full native 60 MP), default threshold **200**. Script: `_dataset/SONY_TEST/qc_sharpness.py`,
JSON: `_dataset/SONY_TEST/qc_sharpness.json`.

| stat | value |
|---|---|
| n | 588 |
| min / median / mean / max | 84.8 / 120.9 / 124.8 / 221.4 |
| p05 / p95 | 102.0 / 160.8 |
| **would cull (< 200)** | **585 / 588 (99.5%)** |
| borderline (200–300) | 3 |
| sharp (> 300) | 0 |

**FINDING (default auto-cull misfires catastrophically on high-MP stills):** threshold-200
would cull 585/588. This EXTENDS the known Stage-1 cull bug (DEVELOPER_NOTES §5, previously
documented only for 4K video). Root cause: at 60 MP the lens oversamples the sensor, so
per-pixel second-derivative (Laplacian) variance is LOW even though every frame is perfectly
sharp — the threshold-200 default was calibrated on 24 MP DSLR where pixel-level detail is
crisper. Higher-MP sensors paradoxically score LOWER on this absolute metric. **Correct action
taken:** bypass auto-cull, keep all 588, let COLMAP geometric verification drop unregisterable
frames. (Reinforces the §7 open item: the cull needs a relative/data-driven threshold.)

## STAGE 2-7 — SfM (driver `_dataset/SONY_TEST/run_sfm_sony.py`)
Decoupled resolution: ALIKED-n8192 @ resize_max 1600, native 60 MP intrinsics for recon.
Retrieval (NetVLAD top-50) matching = designed >300-frame path (588). Incremental SfM, OPENCV
single camera, final BA 200, export binary + up=y alignment + centering (no masks, no IMU).

| stage | result |
|---|---|
| Stage 2 ALIKED @1600 | avg **2322** kpts/img (low — bokeh bg starves features; concentrate on sharp shoe+stand), 362 s |
| Stage 3 retrieval match | **29,400 pairs**, avg **867.9 matches/pair** (excellent), 2651 s (~44 min) |
| **Stage 4-5 incremental SfM** | **registered 588/588 = 100.0%** ✓ (gate ≥95%); **158,791** points; **mean reproj 1.541 px**; 2810 s (~47 min) |
| Stage 6 final BA (200) | pending |
| Stage 7 export | pending |

**FINDING (robustness — PASS):** the stack registered 100% of a fresh, unseen 60 MP capture
with a heavily defocused (shallow-DoF) background and semi-gloss subject, at 1.54 px reproj —
matching DEVELOPER_NOTES §4 ("SfM is not the bottleneck; ~100% reg, ~1.5 px"). Low avg keypoints
(2322 vs a textured scene's higher) did NOT hurt registration because avg matches/pair is very
high (867.9) — the sharp shoe + rigid stand carry strong, repeatable correspondences.

**OPERATIONAL FINDING (durability):** the first background launch (Bash `run_in_background`) was
**killed** by the harness reaping the job tracker mid-matching (~53 min in) — NOT the benign
orphan-survival case in WP-S1 (0 python left, log frozen). Fix: relaunch via PowerShell
`Start-Process -WindowStyle Hidden` (detached/reparented) — survives the hourly reap. HLoc's h5
skip made the resume free (matched pairs persisted; only the remaining ~9k re-run). All
subsequent long jobs (prep/dense_init, training) use the same detached pattern.

---

## PHASE B — training prep  (driver `_dataset/SONY_TEST/prep_sony.py`)
`--data_factor 3`: 9504×6336 = 60.2 MP → /3 per side = **3168×2112 = 6.69 MP**, the nearest
landing to the validated ~6 MP regime (Laptop/GPU_V4 = 24 MP/factor 2 = 6 MP). coarse_factor
**6** = the 2×-of-fine C2F analog of the validated "4→2" schedule. dense_init ON (RoMa outdoor,
560×560 fp16, num_matched 5, 15000 samples/pair, cert 0.05, voxel 1.0, seed 0) baked into
data/sparse/0. Mono-depth prior cache: `python -m core.priors <data_dir> --factor 3 --depth_only`.

- **prepare_data (54.2 min):** images/ (9.9 GB copy) + images_3 (588 @ 3168×2112 = 6.69 MP) +
  images_6 (588 @ 1584×1056). dense_init RoMa: 1822 covisible pairs, triangulated 792,879 pts
  (4.2 min), **158,791 → 738,340 points (+579,549, 4.65×)**, new-point mean reproj **1.03 px**,
  **poses preserved exactly** (max|Δt|=0, max|ΔR|=1.7e-6°). 4.65× is lower than GPU_V4's 7.16×
  — expected: the shallow-DoF background yields fewer dense covisible matches, so the extra
  points concentrate on the sharp shoe (where they help most).
- **priors (--depth_only, 179 s):** MoGe-2 (`Ruicheng/moge-2-vitl-normal`), factor 3,
  **588/588 views reliable**, median reliable rel-residual **0.0146** (clean MoGe↔SfM alignment
  despite the specular subject), scale span [0.715, 2.07], peak VRAM 2.5 GB.

## PHASE C — train validated stacked recipe  (`priors_then_train.bat` / `train_sony.sh`)
Setup lines confirmed: mono-depth loss ON (λ0.05, start auto-raised to 15001 = fine phase),
**514/514 train views covered**, 588/588 reliable; erank hinge ON (λ0.01 @7k); C2F factor 6→3
@ step 15000; init **738,340 Gaussians** from the dense cloud → MCMC cap 3.0 M; pose_opt on;
gated-LPIPS λ0.2 @3k. Peak train VRAM ~**23.3 GB** (of 96). ~12–16 it/s.
Exact WP-S1 stacked production argv adapted to factor 3 (persisted in train_sony.sh):
```
--data_factor 3 --max_steps 30000 --test_every 8 --seed 0 --sh_degree 3 --ssim_lambda 0.2
--strategy mcmc --cap_max 3000000 --opacity_reg 0.01 --scale_reg 0.01
--antialiased --normalize_world_space --pose_opt --pose_opt_lr 1e-5
--coarse_factor 6 --coarse_until 15000
--lpips_lambda 0.2 --lpips_patch 512 --lpips_start 3000 --lpips_gated
--mono_depth_lambda 0.05 --mono_depth_start 2000 --mono_depth_use_weight
--erank_lambda 0.01 --erank_start 7000  --save_steps 7000 30000
```
(dense_init has no trainer flag — the augmented sparse/0 IS the dense init.)
**RESULTS (from `output/stats/results.json` = the every-8 eval split):** final model hit the
**MCMC cap = 3,000,000 Gaussians** at step 30000; **PSNR 23.928 / SSIM 0.8949** (identical to the
Phase-D scorecard — same split). Peak train VRAM ~23.3 GB of 96 (recorded above), ~12–16 it/s →
~35–42 min est. (wall-clock not separately logged; prior agent died at a spend limit AFTER training
completed — ckpt/PLY at steps 6999 & 29999 both on disk). Config confirmed in results.json:
mcmc, antialiased, pose_opt, data_factor 3, coarse_factor 6.

---

## PHASE D — eval  (finished 2026-07-07; GPU free, sequential)
Ckpt `…\output\ckpts\ckpt_29999_rank0.pt`, data `…\training\gsplat\data`, factor 3, every-8 split
(**74 held-out**, 100% observed area). All 6 phase-2 scorers ran.

### Scorecard  (`qa/wpt1_sony/scorecard.json` + report.html)
| metric | value |
|---|---|
| masked PSNR | **23.928** |
| masked SSIM | **0.8949** |
| masked LPIPS (↓) | **0.4096** |
| observed area | 100.0% |
| render_sanity | ok |

| scorer | result | note |
|---|---|---|
| CrossScore (↑ predicted-SSIM) | mean **0.849** / median 0.864 | self-sup, no GT |
| NoVA (↓ cosine-dist) | mean **0.0175** / median 0.0127 | feature drift vs neighbor |
| TSED (↑ consistent-ratio) | **1.0** (33/33 pairs) | median epi 0.050 px, 723 matches/pair |
| RAFT-warp (↓ L1) | mean **0.0123** / median 0.0099 | valid_frac 0.982 |
| NVS-SQA (↑ JOD) | **−2.71** | no-ref perceptual |
| Flicker (↑ score) | **0.9972**, static=True | max motion frac 1.5e-4 → no popping |

**Per-view spread (robustness):** whole-frame PSNR 19.4 (idx496) … 27.5 (idx8). The low-PSNR /
low-CrossScore cluster is indices **400–512** (CrossScore dips to 0.54–0.68) — these are the
**sole-from-below + top-down heavy-specular angles** with the least texture and the most
view-dependent metallic reflection. TSED=1.0 and flicker static=True confirm the model is
geometrically consistent and non-flickering across the whole orbit (SfM/pose quality holds).

### Floater report  (`_infusion/wpt1_floaters.txt`)
`sony  3,000,000  needle 12.20%  SOR-outlier 0.04%  giant 12.02%  low-opa 41.5%  med-opa 0.018  mean-erank 2.169`
- **SOR-outlier 0.04%** = essentially no isolated floaters (dense-init + erank hinge working).
- **low-opa 41.5%** (opacity<0.01) and **med-opa 0.018** = characteristic MCMC-at-cap padding: the
  fixed 3.0M budget is ~40% near-transparent Gaussians. **Cross-check:** the full-scene delivery
  pruned **1,244,654 / 3,000,000 = 41.49%** as low-opacity — an exact match to the floater metric,
  i.e. ~40% of the master is disposable weight that compresses away losslessly at delivery.
- needle 12.20% / giant 12.02% / mean-erank 2.169: erank sits just above the needle threshold (2.0),
  the hinge is holding the distribution off the degenerate wall but many Gaussians are still thin
  (expected for a mesh-upper + thin laces/rod subject).

### Zoom strips  (`qa/wpt1_strips/`, GT | render, full + 2× crop)
`strip_vp44_laces_mesh.jpg`, `strip_vp32_silver_logo_specular.jpg`, `strip_vp28_green_medial_specular.jpg`
**Visual verdict:** renders are faithful — white laces, tongue "ASICS GEL" text, hexagonal mesh
weave, reflective silver TPU swoosh + "ASICS" logo, iridescent metallic-green medial panel with its
sheen gradient, embossed "GEL-KAYANO 14" script, translucent sole gel-window, and the thin mount
rod are all reproduced. Limits: the very finest mesh micro-texture softens slightly, and
view-dependent specular highlights on the metallic panels shift a touch (the WP-S1 specular case —
handled well here, not failed). The high whole-frame LPIPS (0.41) is dominated by the
**heavily-defocused bokeh background**, NOT the shoe (object-region LPIPS is 0.176 — see Phase E).

---

## PHASE E — deliver  (`core/delivery.py`, both runs finished 2026-07-07)
Two variants, both factor 3, decimate 60%. Records: `deliverables/sony_*_delivery.json`.

| deliverable (`deliverables/`) | mode | Gaussians | SH | PLY MB | SOG MB | runtime |
|---|---|---|---|---|---|---|
| **sony_shoe_object.sog/.ply** | object-extract: BiRefNet masks (588/588) → mask-vote → prune → decimate60% → **refit 3k** | 939,315 | 3 | 211.41 | **13.37** | 703.9 s |
| **sony_full_full.sog/.ply** | full-scene: prune → floater-filter → decimate60%, SH→2, no refit | 1,049,779 | 2 | 152.18 | **14.01** | 245.7 s |

(File sizes are the on-disk `deliverables/` values — authoritative; the JSON records list ~5% higher
because they measure the pre-copy `_work` intermediate.)

**Object extract (the product deliverable) — faithfulness IMPROVED by extraction+refit:**
- object-region LPIPS **0.1841 → 0.1760** (Δ −0.0082); PSNR 22.70→**25.42**; SSIM 0.775→**0.878**.
- background bleed (out-of-mask alpha) **0.9351 → 0.0229** (max 0.204) — background cleanly removed.
- spatial cleanliness: **99.988%** of delivered Gaussians within 0.5× scene-scale of the object
  center (p99 radial 0.136×); only 112 beyond, 64 beyond 1.0× — a clean carve.
- vote kept 1,882,165/3.0M (62.7%) inside mask; 896,909 never-observed dropped; prune −302,538 low-opa.
- SH3 kept (WP-D1 finding: SH2 truncation is the LPIPS regression on glossy objects — keep specular).

**Full-scene — decimation near-lossless:** whole-frame LPIPS master 0.4096 → reduced 0.4153 →
delivered 0.4239 (**+3.5% vs master**, within the QA bound). SH truncated to 2 (halves payload,
still in-bound for full-scene). No refit (WP-D1: refit trades LPIPS for PSNR on an LPIPS-tuned master).

---

## VERDICT (WP-T1 end-to-end)
1. **Capture quality: strong and SfM-friendly** — a rigid full-sphere product orbit registered
   100% (588/588) at 1.54 px with 868 matches/pair despite semi-gloss metallic surfaces and a
   shallow-DoF bokeh background; the only Stage-1 snag was the known auto-cull misfire on 60 MP
   stills (bypassed, all frames kept — see Stage 1).
2. **Model quality: good, and the perceptually-relevant object is faithful** — whole-frame metrics
   (PSNR 23.9 / LPIPS 0.41) look mediocre but are dragged down by the defocused background; the
   **object-region LPIPS is 0.176** and the zoom strips show faithful laces, mesh, text, and even
   the specular metallic panels (the WP-S1 failure case is handled, not failed). Direct comparison
   to the Laptop/GPU_V4 baselines is **not valid** (different scene, different subject, factor-3 vs
   factor-2) — flagged per the different-scene caveat; qualitatively this sits in the same
   ~24 dB / TSED-1.0 / non-flickering regime as those runs.
3. **Robustness findings:** (a) the full stack is end-to-end healthy on a fresh unseen capture —
   TSED 1.0, flicker static, SOR-outlier 0.04%, delivery carve 99.99% clean; (b) MCMC-at-cap leaves
   ~41.5% near-transparent Gaussians that the delivery prune reclaims losslessly (floater low-opa ==
   delivery prune, exact); (c) the hardest views are the low-texture sole-from-below and top-down
   heavy-specular angles (idx 400–512), the natural weak spot for a metallic subject on a thin mount.

