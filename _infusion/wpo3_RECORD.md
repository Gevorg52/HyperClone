# WP-O3 — 360° training lane — GATE RECORD (2026-08-05)

Plan of record: `docs/PLAN_360.md` §5 (WP-O3). Stub: `docs/EXECUTION_PLAN.md` §WP-O.
Input: WP-O2 ARM A (`_dataset/360/sfm/wpo2_crop_rig`) — 1 812 pinhole crops (151 panos ×
12), SIMPLE_PINHOLE 960×960 f=480, 622 325 points, mean reproj 1.2313 px, on the ih_fov-198
re-ingest `_dataset/360/flowershop_077_ihfov198`.

Deliverable: `core/train_360_prep.py` (dataset packaging, default-OFF), two default-OFF
trainer flags (`--holdout_list`, `--loss_mask`), the first 360-derived Gaussian splat, and
the evidence below.

**This WP produced a headline NEGATIVE result before it produced a splat, and the negative
result is the more valuable half: the production trainer config DIVERGES on this capture,
and the single cause is `--opacity_reg 0.01`.** Both arms are recorded in full.

---

## 1. What was built

| piece | what it does |
|---|---|
| `core/train_360_prep.py` | WP-O2 crop-rig run dir → trainer-shaped COLMAP dataset. **Default-OFF**: `main()` returns 2 without `--capture-type 360`; nothing in production imports it. |
| `gsplat_trainer.py --holdout_list FILE` | explicit test-set membership by COLMAP image name. Default `None` ⇒ the existing every-`--test_every` rule is byte-unchanged. |
| `gsplat_trainer.py --loss_mask` | average L1 **and the per-pixel SSIM map** over the VALID mask only. Default OFF ⇒ production unchanged. (`ssim()` gained `return_map=`; its default return is unchanged.) |
| `_infusion/wpo3_eval.py` | held-out masked PSNR/SSIM/LPIPS + figure strips + walk-through MP4 + scorecard. |
| `_infusion/wpo3_diag_collapse.py` | the 8-arm short-run sweep that found the root cause. |
| `tests/test_train_360_prep.py` | 21 tests (naming, split, layout, trainer consumption). Suite **235 passing** (211 → 214 via WP-O2 housekeeping (+3) → 235 (+21); counts corrected per review 2026-08-05). |

### 1a. Why a packaging step at all — the two shape mismatches

`load_colmap_data` (`core/training/gsplat_trainer.py:116`) resolves images as
`os.path.join(data_dir, "images", <colmap image name>)` and masks as
`os.path.join(data_dir, "masks", os.path.splitext(<name>)[0] + ".png")`. WP-O2 writes
`images/pano_cameraN/pano_XXXXX.jpg` and `masks/pano_cameraN/pano_XXXXX.jpg.png`.

1. **Mask filename.** WP-O2 uses COLMAP's own `<image_name>.png` convention
   (`image_reader.cc:148-171`); the trainer wants `<stem>.png`. Renamed on copy.
2. **Per-camera subdirectories.** `os.path.join` would actually resolve the nested name
   correctly, but (a) every tool that keys on `os.path.basename` (prior caches, the QA and
   render scripts) would then collide 12 ways on one stem, and (b) the trainer sorts views
   by name, so nested names order the dataset **camera-major** while the held-out split
   needs **pano-major**. Names are therefore flattened to `pano_XXXXX_camNN.jpg` in the
   file layout **and** in the COLMAP model (`pycolmap.Image.name` is writable; the model is
   re-written to `sparse/0`).

Pixels are untouched: images and masks are byte-copied, no resize, no re-encode; camera
intrinsics, poses, rig, frames and points3D carry over verbatim. Every transformation is
listed in `prep_report.json → transformations`.

### 1b. Held-out split

Every 8th **panorama** by sorted-name rank, **all 12 of its crops together**. The 12 crops
of a panorama share an exact optical centre, so splitting one leaks the held-out view into
training. Eligibility uses the strict `panos_fully_registered` accounting added to
`core.sfm_360.reconstruction_metrics` by the WP-O2 housekeeping commit — all 151 panos have
all 12 crops registered here, so nothing was dropped.

Result (verified by re-loading the packaged model through the trainer's own loader):
**1 812 images = 1 584 train (132 panos) + 228 test (19 panos), 12/12 crops per held-out
pano, ZERO panorama appears in both splits.** Lists: `holdout_test.txt`,
`holdout_train.txt`, `split_panos.json`.

The trainer's own `--test_every` rule cannot express this: it is periodic over images, and
with 12 crops/pano `12p + c ≡ 0 (mod 8)` picks 1–2 crops out of *every* panorama — maximal
leakage. Hence `--holdout_list`.

---

## 2. ARM 1 — the production config — **DIVERGED**

`_infusion/wpo3_train.cmd` (committed, argv verbatim), log
`_dataset/360/train/wpo3_train.log`, evidence `_infusion/wpo3/arm1_baseline_diverged/`.

Config: production `get_default_config()` with four recorded deviations — `data_factor` 2→1
(crops are already 960², throwing away pixels makes no sense), `cap_max` 1M→3M (the
validated sweet spot, DEVELOPER_NOTES §6), `coarse_factor` 4→0 (C2F needs a pre-built
`images_4/`, and at 960² it would densify at 240²), `pose_opt` on→off (the 12 crops of a
pano share an optical centre **by construction** — BA fixed the rig — and per-crop SE(3)
freedom breaks exactly that). Masks active via `--rgb_object_masked`; `alpha_loss_lambda` /
`alpha_obj_lambda` left at 0 because masked-out pixels here are duplicated-coverage /
operator / seam, **not background** — alpha must not be pushed to 0 there.

It ran to completion (30 000 steps, 890.8 s, 3 000 000 Gaussians) without raising, and the
result is worthless:

| held-out (228 crops, 19 panos) | ARM 1 |
|---|---|
| masked PSNR | **13.86** dB |
| masked SSIM | 0.4095 |
| **masked LPIPS (tiled, 128 px, ≥80 % valid)** | **0.6949** |
| masked LPIPS (bbox) | 0.6984 |
| full-frame PSNR / SSIM / LPIPS | 13.588 / 0.4439 / 0.6916 |

The trainer's own final eval printed `PSNR=13.59` — my eval path reproduces it to 3 decimal
places, which is the cross-check that `wpo3_eval.py` is measuring the same thing.

**Every render is a flat colour wash** — see `closeup_thin_pano_00024_cam03.jpg`,
`grid_heldout_pano_*.jpg`. Checkpoint forensics say why:

| | step 7 000 | step 30 000 |
|---|---|---|
| mean opacity | 0.0099 | **0.0002** |
| fraction with opacity < 0.01 | 67.6 % | **99.87 %** |
| median max-axis scale | 1.117 | 0.135 |
| median ‖mean‖ (camera cloud radius = 1) | 122.6 | **5 909** |
| ‖mean‖ p99 | 5 473 | 176 021 |
| means bbox | ±2.4 e5 | **±2.1 e6** |

The model did not "train badly" — the Gaussians went transparent and then random-walked
out to a million camera-radii. Train PSNR peaked at step ~1 000 (with 794 k Gaussians) and
got monotonically *worse* as MCMC grew the count.

---

## 3. Root cause — an 8-arm controlled sweep

`_infusion/wpo3_diag_collapse.py`, results `_infusion/wpo3/diag_collapse{,2,3}.json`
(argv + per-arm log path + checkpoint statistics for every arm). All arms: 5 000 steps,
seed 0, same dataset, same holdout list, MCMC cap 3 M, antialiased, ssim 0.2, factor 1.

| arm | mask path | opacity_reg | scale_reg | outcome | mean opacity | dead <0.01 | ‖mean‖ p99 | median scale | train PSNR |
|---|---|---|---|---|---|---|---|---|---|
| A | `rgb_object_masked` | 0.01 | 0.01 | **CUDA OOM-crash** | — | — | — | — | 19.04 |
| A2 (re-run, alone on GPU) | `rgb_object_masked` | 0.01 | 0.01 | **collapsed** | 0.00057 | 99.6 % | 10.06 | 0.378 | 18.03 |
| B | none | 0.01 | 0.01 | degraded | 0.0199 | 64.5 % | 4.14 | 0.0146 | 13.35 |
| C | `rgb_object_masked` | 0.0033 | 0.0033 | **CUDA OOM-crash** | — | — | — | — | 18.59 |
| D | `rgb_object_masked` | **0** | **0** | healthy | 0.3780 | 5.2 % | 3.78 | 0.0069 | 21.84 |
| E | `loss_mask` | **0** | **0** | healthy | 0.3785 | 5.3 % | 3.78 | 0.0069 | 16.82 |
| F | `loss_mask` | 0.01 | 0.01 | **CUDA OOM-crash** | — | — | — | — | 13.50 |
| **G** | `loss_mask` | **0** | 0.01 | **healthy, best** | **0.4379** | **4.6 %** | 3.95 | 0.0072 | **17.15** |
| H | `loss_mask` | 0.01 | **0** | **collapsed** | 0.00053 | 99.9 % | **6 774** | 21.53 | 13.08 |

PSNR columns are not directly comparable across mask paths: A/A2/C/D print PSNR on the
**black-composited** frame (inflated by ~10·log₁₀(1/0.331) ≈ **4.8 dB** because 66.9 % of
each frame is exactly 0 on both sides), B prints full-frame, E/F/G/H print the **masked**
PSNR (the `--loss_mask` path now logs the honest number). D 21.84 composited ≈ 17.0 masked
≈ E 16.82 — the two mask paths agree once you undo the offset.

**G vs H is the whole finding.** Same code, same data, one flag apart:
`opacity_reg=0, scale_reg=0.01` → healthy. `opacity_reg=0.01, scale_reg=0` → 99.9 % dead,
positions at 6 774 camera-radii. **`opacity_reg` at the production value is the single
cause; `scale_reg` at the production value is harmless here (and 0.3 dB better than
zeroing it).**

### Mechanism (read from gsplat 1.5.3 source, not guessed)

1. `opacity_reg · mean(sigmoid(opacity))` applies a **constant downward pressure to every
   Gaussian on every step**. The counter-pressure is photometric and only acts when a
   Gaussian is actually rendered into a training view.
2. This capture is 1 584 training views of a 12 m corridor + two side rooms, each view a
   90° crop of one panorama. Any given Gaussian is visible in a tiny fraction of views, and
   at 30 000 steps each view is drawn only **≈19 times** (a production GPU_V4 run gives each
   of ~200 views 100–250 draws). So the photometric pressure per Gaussian is roughly two
   orders of magnitude weaker than on the object-orbit captures `opacity_reg=0.01` was
   validated on. Opacity loses.
3. gsplat's MCMC injects noise into positions scaled by `op_sigmoid(1 - opacity)` with
   `k=100, x0=0.995` — i.e. **maximum noise for near-zero-opacity Gaussians**, by design,
   so that dead Gaussians explore. Once step 2 has driven opacity below ~0.005 for
   essentially every Gaussian, every Gaussian gets full noise every step and the cloud
   random-walks apart. That is the ±2 e6 bbox.
4. Masking *amplifies* it. `--rgb_object_masked` composites both sides over black and then
   takes a plain frame mean, so on these ~33 %-valid crops the photometric term is divided
   by ~3 while the regularizers are untouched — which is why the masked arms blow up hard
   enough to crash the rasterizer (tile-intersection blowup, 19.6 → 57.9 GB) while the
   unmasked arm B merely degrades.

Honest caveat: arms A, C and F crashed with `CUDA error: an illegal memory access` while an
eval job shared the GPU. A2 re-ran arm A **alone** and did not crash — it collapsed instead
(opacity 5.7e-4, 99.6 % dead). So the *collapse* is reproducible and load-independent; the
*crash* is the collapse plus memory pressure, and I am not claiming more than that.

---

## 4. ARM 2 — the corrected run

`_infusion/wpo3_train_arm2.cmd` (committed), log `_dataset/360/train/wpo3_train_arm2.log`.
Three changes from ARM 1, each with an arm of §3 behind it:

| change | from | to | evidence |
|---|---|---|---|
| `--opacity_reg` | 0.01 | **0** | arms G vs H |
| mask path | `--rgb_object_masked` | **`--loss_mask`** | correct semantics: no ÷3 magnitude loss, no manufactured black edges at the mask boundary; arms E/G |
| `--max_steps` | 30 000 | **200 000** | 1 584 views: 30 k = 19 epochs vs 100–250 in production; also ~6.5× fewer pixel-gradients than a GPU_V4 run (30 k × 0.92 MP vs 30 k × 6 MP). 200 k ≈ 126 epochs |

`--scale_reg` stays at the production 0.01 (arm G). Everything else is ARM 1's config.

**It trained to completion: 200 000 steps, 4 471.8 s (74 min 32 s), 3 000 000 Gaussians,
19 599 MB peak, no divergence, no crash.** Masked train PSNR climbed monotonically the whole
way — 14.87 (1 k) → 16.82 (5 k) → 17.77 (10 k) → 18.35 (14 k) → 19.55 (39 k) → 20.06 (86 k) →
20.78 (122 k) → ~20.8 (200 k) — where ARM 1 peaked at step ~1 000 and fell.

### 4a. Held-out results — the gate numbers

228 crops / 19 panoramas, never seen in training. `masked_*` uses the WP-O2 crop mask
(mean 33.19 % valid, min 14.2 %). LPIPS "tiled" = 128 px tiles with ≥80 % valid coverage
(the honest observed-region number); "bbox" = the tight bounding box of the valid region
(includes invalid pixels, reported for completeness). All 228 views produced a value in
every metric. Full per-view rows: `_infusion/wpo3/*/metrics.json`.

| held-out (228 crops) | **ARM 1** 30 k, prod config | **ARM 2** 30 k | **ARM 2** 200 k |
|---|---|---|---|
| **masked LPIPS (tiled)** ↓ | 0.6949 | 0.5524 | **0.5222** |
| masked LPIPS (bbox) ↓ | 0.6984 | 0.5825 | **0.5572** |
| masked PSNR ↑ | 13.856 | 17.193 | **17.483** |
| masked SSIM ↑ | 0.4095 | **0.4352** | 0.4334 |
| full-frame LPIPS ↓ | 0.6916 | 0.5817 | **0.5587** |
| full-frame PSNR ↑ | 13.588 | 16.561 | **16.720** |
| full-frame SSIM ↑ | 0.4406 | **0.4544** | 0.4406 |

The trainer's own final eval printed `PSNR=16.72 SSIM=0.4406` for ARM 2 and `PSNR=13.59` for
ARM 1 — `wpo3_eval.py` reproduces both to 3 decimals, which is the cross-check that the two
measurement paths agree.

**Step-count ablation, inside ARM 2 (same run, its own 30 k checkpoint):** at the *same*
30 000-step budget as ARM 1, the `opacity_reg` + `loss_mask` fix alone is worth
**+3.34 dB masked PSNR and −20.5 % masked LPIPS**. The following 170 000 steps add
**−5.5 % LPIPS** but only **+0.29 dB PSNR**, and SSIM goes very slightly *down*.
So: **the fix is nearly the whole win; the long schedule buys perceptual sharpness only.**
Train PSNR meanwhile went 18.0 → 20.8 over those steps, i.e. the extra epochs increasingly
fit the training views — mild overfitting with a still-improving perceptual metric. A future
run can reasonably stop at ~60–100 k.

### 4b. Checkpoint health — the direct divergence check

`_infusion/wpo3/ckpt_health.json`. Camera-cloud radius is normalised to 1.

| | ARM 1 @30 k | ARM 2 @30 k | ARM 2 @200 k |
|---|---|---|---|
| mean opacity | 0.00025 | 0.4967 | **0.5550** |
| fraction opacity < 0.01 | 99.87 % | 12.17 % | **21.3 %** |
| median ‖mean‖ | 5 909 | 1.698 | **1.699** |
| ‖mean‖ p99 | 176 021 | 4.18 | **4.20** |
| max ‖mean‖ | 2 115 274 | 24 356 | 24 321 |
| median max-axis scale | 0.1355 | 0.00214 | 0.00257 |

ARM 2's dead fraction (21.3 %) is *below* the 30.6 % a healthy 3 M GPU_V4 production run
shows (DEVELOPER_NOTES §6), and its position spread is stable between 30 k and 200 k — the
cloud is not drifting.

---

## 5. Gate

PLAN_360 §5's WP-O3 gate is: *trains to completion; held-out masked-LPIPS recorded (baseline
number, no pass bar — this is the first 360 splat); visual checks persisted: no operator
ghosts, no seam-band phantom geometry, corridor walkable in viewer.*

**GATE: PASSED on ARM 2. ARM 1 (the production config) FAILED it and is recorded in full.**

| gate item | verdict | evidence |
|---|---|---|
| trains to completion, no divergence | **PASS (ARM 2)** — 200 000 steps, 74 min 32 s, 3 M Gaussians, mean opacity 0.555, ‖mean‖ p99 4.20. **FAIL (ARM 1)** — completed without raising but produced a transparent, exploded model | `ckpt_health.json`, §4b |
| held-out masked-LPIPS recorded | **PASS — 0.5222** (tiled) / 0.5572 (bbox) on 228 crops of 19 withheld panoramas; masked PSNR 17.483, SSIM 0.4334 | `arm2_final/metrics.json` + `scorecard.json` |
| no operator ghosts | **PASS** | `arm2_final/closeup_operator_*.jpg` — on the 3 most-excluded held-out crops (15–20 % valid) the operator fills the GT frame and **is simply absent from the render**: corridor wall, ceiling and drawings appear where he stood. The region is soft (unsupervised from that viewpoint, filled only by other panoramas) but there is no person-shaped geometry |
| no seam-band phantom geometry | **PASS** | `arm2_final/closeup_seam_*.jpg` — crops 5 and 7, whose optical axis IS the ffmpeg-v360 stitch seam (yaw ±90°). Walls, ceiling tiles, lights and a door reconstruct in the right places; no duplicated wall, no slab, no discontinuity at the seam |
| corridor structure walkable | **PASS** | `arm2_final/walkthrough.mp4` (151 frames, 12 fps, the equator forward crop at every SfM pose, held-out frames labelled). Coherent, stable corridor + rooms: doorways, ceiling grid, fluorescent fixtures, drawing-covered walls, consistent across the whole walk |
| bystanders (WP-O1/O2 carry-over) | **no ghost seen, NOT exhaustively audited** | see below |

**Quality in plain words:** the splat is *structurally correct and soft*. Geometry, layout
and materials are right and stable; fine texture (the charcoal drawings that cover every
surface) is smeared. That is consistent with the input — a 1 920²-per-lens in-camera proxy,
continuously moving, motion-blurred, resampled fisheye→ERP→crop — and it is why PLAN_360 §7's
request for an 8K original still matters. **Do not read 17.5 dB / 0.52 LPIPS as a trainer
verdict; read it as this capture's ceiling under this recipe.**

**Bystander check — honest negative on the method.** `_infusion/wpo3_bystander_check.py`
tries to find held-out crops containing an unmasked second person by skin-tone (YCrCb)
blobs inside the valid region. **The heuristic is useless on this scene**: the walls are
covered in warm cream paper, so it reports 228/228 crops "hit" with skin fractions up to
0.999, and the top-ranked crop is a drawing-covered floor. Recorded as a failed method, not
a result. The actual bystander check is therefore **visual over the ~25 held-out crops
rendered into `arm2_final/`** — no person-shaped ghost appears in any of them. A systematic
audit would need the bystander segmented (SAM 2.1 on all people, not just the operator),
which is the obvious WP-O1 follow-up.

### 5a. Artifacts

| what | where |
|---|---|
| **final PLY** | `_dataset/360/train/wpo3_run_arm2/ply/point_cloud_199999.ply` (708.0 MB, 3 M Gaussians) |
| final checkpoint | `_dataset/360/train/wpo3_run_arm2/ckpts/ckpt_199999_rank0.pt` (708.0 MB) |
| intermediate ckpts/PLYs | same dirs, steps 29 999 / 59 999 / 99 999 / 149 999 |
| **walk-through video** | `_infusion/wpo3/arm2_final/walkthrough.mp4` (151 frames, 12 fps, 7.6 MB) |
| gate metrics + scorecard | `_infusion/wpo3/arm2_final/{metrics,scorecard,figures}.json` |
| step-count ablation | `_infusion/wpo3/arm2_step30k/{metrics,scorecard}.json` |
| ARM 1 failure evidence | `_infusion/wpo3/arm1_baseline_diverged/` (metrics, scorecard, 10 figures) |
| root-cause sweep | `_infusion/wpo3/diag_collapse{,2,3}.json` + `_dataset/360/train/wpo3_diag/*.log` |
| checkpoint health | `_infusion/wpo3/ckpt_health.json` |
| packaged dataset | `_dataset/360/train/wpo3_flowershop_077/` (gitignored; `prep_report.json`, `holdout_{test,train}.txt`, `split_panos.json` regenerate from the argv) |
| training logs | `_dataset/360/train/wpo3_train{,_arm2}.log` (gitignored; head+tail embedded in each scorecard) |

---

## 6. Negative findings / deviations — recorded honestly

1. **The production trainer config does not work on this capture, and that is the WP's most
   valuable output.** ARM 1 ran to completion with no error and produced a model whose every
   render is a flat colour. Nothing in the trainer warns about it: the loss looks reasonable
   (0.08), the printed PSNR looks reasonable (19.2 — because `--rgb_object_masked` inflates
   it by ~4.8 dB), and the Gaussian count sits at exactly the 3 M cap the whole way. The only
   honest signals were the held-out numbers and the checkpoint statistics. **Any future 360 /
   large-walkthrough run must check mean opacity and ‖means‖ in the checkpoint, not the log.**
2. **`opacity_reg 0.01` — a six-month-old production default — is scene-dependent and can be
   catastrophic.** It was validated on object-orbit captures where every Gaussian is rendered
   in most views. On a 1 584-view walkthrough it wins against a photometric term that reaches
   each Gaussian rarely, and gsplat's MCMC noise schedule (max noise at min opacity) turns
   the resulting transparency into positional runaway. This is NOT a 360-specific finding —
   it should be expected on any large multi-room scene with many views.
3. **`--rgb_object_masked` is the wrong lever for a large mask.** It was built for WP-D1's
   object-extract refit, where the mask keeps most of the object and densification is off
   (`refine_stop_iter=1`). With a 33 %-valid mask under live MCMC it divides the photometric
   term by three and adds black boundary edges; arms A/A2/C/F show the result. `--loss_mask`
   is the correct-semantics replacement. **The two are independent flags and must not be
   combined.**
4. **ARM 2 changes three things at once** (opacity_reg, mask path, step count), which
   violates one-lever-at-a-time. Partial mitigation: the 5 000-step sweep isolates
   opacity_reg (G vs H) and the mask path (D vs E) on their own, and the step count is
   isolated *within* ARM 2 by evaluating its own 30 000-step checkpoint against its
   200 000-step one (§4). What is NOT isolated is any interaction between them.
5. **The held-out split is an INTERPOLATION split, not extrapolation.** Every 8th panorama
   is withheld, and its neighbours (~0.4 SfM units away, ≈ 0.5 s of walking) remain in
   training, so the held-out views are well covered by nearby training views. The numbers
   here are therefore optimistic relative to a genuine novel viewpoint. That is the right
   split for "did the trainer fit the capture", and the wrong one for "can it synthesise
   unseen regions" — use the WP-Q1 visibility-split suite for the latter.
6. **Held-out PSNR/SSIM/LPIPS here are masked by the WP-O2 crop mask**, which composites the
   exclusive nearest-camera tiling with WP-O1's operator/nadir/seam exclusions. So "masked"
   means "on the pixels this crop is responsible for and that are real scene". Comparing
   these numbers to the GPU_V4/Laptop baseline bank in `qa/README.md` is **not** valid — the
   scene, the resolution, the mask semantics and the split are all different. They are a
   360-lane baseline for future 360 runs only.
7. **Arms A, C and F crashed with a CUDA illegal-memory-access while an eval job shared the
   GPU.** A2 re-ran arm A alone: no crash, but the same collapse. So the collapse is
   reproducible and load-independent; the crash is collapse + memory pressure, and I do not
   claim more than that.
8. **Deviations from `get_default_config()` in BOTH arms**, argued at the top of
   `_infusion/wpo3_train.cmd`: `data_factor` 2→1, `cap_max` 1M→3M, `coarse_factor` 4→0,
   `pose_opt` on→off. Of these, **`pose_opt` off is a genuine open question** — enabling it
   would give each of the 12 crops of a panorama independent SE(3) freedom and break the
   shared-optical-centre property that BA deliberately fixed, but it might also absorb real
   stitching error. It is the cleanest single follow-up A/B.
9. **Not attempted:** raising `cap_max` above 3 M (DEVELOPER_NOTES §6 says it hurts — but
   that was measured on a single object+scene, and this is a corridor plus two rooms, so
   3 M may genuinely be starving here); scaling `refine_stop_iter` with the longer schedule
   (left at the 25 000 default, so densification runs for 12.5 % of ARM 2); LPIPS loss;
   erank; densified re-crops; ALIKED/LightGlue on crops.
10. **Bystanders are still unmasked** (WP-O1 §3.4 / WP-O2 §4.12 carry-over): WP-O1 masks the
    operator only, and a second person walks through several panoramas. See §5 for what the
    renders actually show.

---

## 7. Reproduce

```
# 1. package the WP-O2 crop-rig model as a trainer dataset (default-OFF without the flag)
python -m core.train_360_prep --capture-type 360 \
  --sfm _dataset/360/sfm/wpo2_crop_rig \
  --out_dir _dataset/360/train/wpo3_flowershop_077 --overwrite --progress 25

# 2a. ARM 1 (the diverged production-config arm) — 890.8 s
_infusion\wpo3_train.cmd
# 2b. ARM 2 (the corrected arm)
_infusion\wpo3_train_arm2.cmd

# 3. the root-cause sweep (8 arms x 5000 steps)
python _infusion/wpo3_diag_collapse.py --data_dir _dataset/360/train/wpo3_flowershop_077 \
  --work _dataset/360/train/wpo3_diag --out _infusion/wpo3/diag_collapse.json --steps 5000

# 4. gate evidence (metrics + figures + walk-through mp4 + scorecard)
python _infusion/wpo3_eval.py --ckpt <ckpt.pt> \
  --data_dir _dataset/360/train/wpo3_flowershop_077 --out _infusion/wpo3/<arm> \
  --train_log <log>

python -m pytest tests -q          # 235 passed (corrected per review 2026-08-05)
```

---

## 8. Hand-off

**If you train ANY large multi-view scene (360 or not) with this trainer, read finding §6.2
first.** `--opacity_reg 0.01` is a production default validated on object orbits; on captures
where each Gaussian is rendered in few views it silently destroys the model. Check
`sigmoid(opacities).mean()` and `means.norm(dim=1).median()` in the checkpoint — the training
log will not tell you.

Ranked follow-ups, cheapest first:

1. **`pose_opt` A/B** (~75 min) — the one deviation in both arms with no evidence behind it.
   ON gives each crop independent SE(3) freedom, which breaks the shared-optical-centre
   property BA fixed, but may absorb real v360 stitching error. Same argv otherwise.
2. **Shorter schedule** (~25 min) — §4a says 170 k of the 200 k steps bought only −5.5 %
   LPIPS. Re-measure at 60 k and 100 k from the checkpoints already on disk (`wpo3_eval.py
   --do metrics`) before spending 75 min on any future arm.
3. **`cap_max` > 3 M** — DEVELOPER_NOTES §6's 3 M ceiling was measured on a *single
   object+scene*; this is a corridor plus two rooms and 3 M may be starving it. Now safe to
   test: WP-X1 showed the divergence mechanism is fixed, and ARM 2's dead fraction is only
   21 %. Pair with `--erank_lambda 0.01` (WP-Q2).
4. **Mask ALL people, not just the operator** — WP-O1's SAM 2.1 stage tracks the operator
   only. That is the last known unmasked contaminant and it also blocks a proper bystander
   audit (§5).
5. **Sharpness is capture-limited** — the 8K originals (`_027`/`_028`) or an 8K re-shoot per
   PLAN_360 §6 will move this baseline far more than any trainer lever. `_027` is already
   proven through WP-O1 at 7680×3840.
6. **Densified re-crops** (PLAN_360 §5) — more yaw steps / extra pitch rows share the pano
   pose, so no new SfM is needed; more views at the same resolution, and the exclusive masks
   would keep a larger fraction of each crop.

For WP-O4 (fisheye-native A/B): compare against **ARM 2's** numbers
(`_infusion/wpo3/arm2_final/metrics.json`), on the **same 19 held-out panoramas**
(`holdout_test.txt` is a committed list of names — regenerate the dataset with
`core.train_360_prep` and the split is bit-identical), and judge by masked LPIPS + visual,
never PSNR alone.
