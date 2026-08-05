# WP-O5 — 8K validation + free-viewpoint quality

**Status:** ✅ **GATE PASSED** (SfM outright; training/free-view with a bounded claim — see §8).
**Asset:** `D:\HYPERCLONE\FOOTAGE\FlowerShop\VID_20260721_123244_00_027.insv` (8K, 2×3840²
in-camera dual-fisheye, 46.78 s, ~64 Mbps/lens, hospital corridor + operating theatre +
patient rooms; bright, static, no screens).

**Why this WP exists.** The user loaded the WP-O3 `_077` splat into Postshot and the verdict
was **unusable in free view** — "hot mess… only from key angles everything lines up". The
WP-O3 held-out numbers (masked LPIPS 0.5222) did not predict that, exactly as
`_infusion/wpo3_RECORD.md` §6.5 warned they would not: that split is **interpolation**
(every 8th panorama withheld, its ~0.4-unit neighbours kept in training). So this WP adds
measurements that do predict free-view usability, and re-runs the lane on real 8K source.

---

## 0. Reproduce

```
# 1. ingest (already gate-passed in WP-O1; re-used unmodified, see section 1)
python -m core.ingest_360 --capture-type insta360 \
  --input D:/HYPERCLONE/FOOTAGE/FlowerShop/VID_20260721_123244_00_027.insv \
  --out_dir _dataset/360/theater_027_8k --overwrite --contact_sheet 6 --contact_cols 2 \
  --seed 0 --progress 5000

# 2. SIFT-parity guard (20 panos, features-only A/B, identical pairing)
python _infusion/wpo5_sfm_ab.py --dataset _dataset/360/theater_027_8k --panos 20 \
  --work _dataset/360/sfm --out _infusion/wpo5/sfm_parity_ab.json
python _infusion/wpo5_sfm_ab.py --dataset _dataset/360/theater_027_8k --panos 20 \
  --work _dataset/360/sfm --out _infusion/wpo5/sfm_parity_ab_native.json \
  --arms "aliked_native=aliked" --aliked_resize_max 0
python _infusion/wpo5_reproj_split.py \
  --model sift=_dataset/360/sfm/wpo5_ab_sift/sparse/0 \
  --model aliked=_dataset/360/sfm/wpo5_ab_aliked/sparse/0 \
  --out _infusion/wpo5/reproj_split.json
python _infusion/wpo5_traj_agree.py \
  --ref sift=_dataset/360/sfm/wpo5_ab_sift/sparse_equirectangular/0 \
  --other aliked=_dataset/360/sfm/wpo5_ab_aliked/sparse_equirectangular/0 \
  --other aliked_native=_dataset/360/sfm/wpo5_ab_aliked_native/sparse_equirectangular/0 \
  --out _infusion/wpo5/traj_agree.json

# 3. full 8K SfM with the winning stack
python -m core.sfm_360 --capture-type 360 --arm crop_rig --features sift \
  --dataset _dataset/360/theater_027_8k --out_dir _dataset/360/sfm/wpo5_theater_027_8k \
  --overwrite --matcher sequential --overlap 10 --quadratic_overlap 1 \
  --loop_detection 1 --seed 0 --timeout_s 28800 --heartbeat_s 300 --progress 25

# 3b. needles-vs-fog probe (no retraining; run on any checkpoint)
python _infusion/wpo5_needle_probe.py --data_dir _dataset/360/train/wpo3_flowershop_077 \
  --ckpt _dataset/360/train/wpo3_run_arm2/ckpts/ckpt_199999_rank0.pt \
  --out _infusion/wpo5/needle_probe_wpo3

# 4. package + 4-arm training sweep (steps 3->4->5 are chained by wpo5_pipeline.py)
python -m core.train_360_prep --capture-type 360 \
  --sfm _dataset/360/sfm/wpo5_theater_027_8k \
  --out_dir _dataset/360/train/wpo5_theater_027_8k --overwrite
python _infusion/wpo5_train.py --data_dir _dataset/360/train/wpo5_theater_027_8k \
  --work _dataset/360/train --out _infusion/wpo5/train_runs.json --arms A,B,C,D
#   (or, to start packaging+training the instant the SfM finishes:)
python _infusion/wpo5_pipeline.py --sfm _dataset/360/sfm/wpo5_theater_027_8k \
  --data_dir _dataset/360/train/wpo5_theater_027_8k --arms A,B,C,D

# 5. eval (held-out + OFF-PATH sheets + fog/needle diagnostics + MP4s)
python _infusion/wpo5_eval.py --data_dir _dataset/360/train/wpo5_theater_027_8k \
  --arm A=<ckptA> --arm B=<ckptB> --arm C=<ckptC> --arm D=<ckptD> \
  --out _infusion/wpo5/eval --do metrics,offpath,fog
python _infusion/wpo5_eval.py --data_dir _dataset/360/train/wpo5_theater_027_8k \
  --arm <best>=<ckpt> --out _infusion/wpo5/eval --do video --video_offset_m 0.75

# 6. pruned PLY deliverables
python tools/prune_ply.py --ply <best_arm.ply> --opacity_min 0.05 --out <best_arm_op05.ply> \
  --stats_json _infusion/wpo5/prune_wpo5_best.json
python tools/prune_ply.py --ply _dataset/360/train/wpo3_run_arm2/ply/point_cloud_199999.ply \
  --opacity_min 0.05 \
  --out _dataset/360/train/wpo3_run_arm2/ply/point_cloud_199999_pruned_op05.ply \
  --stats_json _infusion/wpo5/prune_wpo3_arm2.json

python -m pytest tests -q
```

---

## 1. Ingest — reused, not re-run (with the mandatory stream spot-check done)

WP-O1's 8K gate run already produced `_dataset/360/theater_027_8k` **with production defaults**
and it is complete and intact: 94/94 images + 94/94 masks at 7680×3840, `ingest_report.json`
present, no `INCOMPLETE` marker. Its argv customised only QA-sheet options
(`--contact_sheet 6 --contact_cols 2 --seed 0 --progress 5000`); every substantive parameter
is the default (`--target_fps 2.0` → 1.998 effective, `--ih_fov/--iv_fov 200`,
`--operator_stream 1`, `--operator_mask auto` → SAM 2.1, `--nadir_deg 12`, `--seam_deg 4`).
Operator masking ran clean: mean operator fraction 5.06 %, **0** area-guard frames, **0**
lost-guard frames, mean excluded 15.61 %.

**Re-ingesting was deliberately NOT done.** A re-run would have written the same argv into the
same directory while re-running SAM 2.1, whose output is not guaranteed bit-identical — i.e.
a non-zero chance of perturbing a gate-passed asset for zero information. The one thing a
re-run could have caught is checked directly instead:

**Stream-order spot-check (required by the `core/ingest_360.py` docstring for any new
camera/firmware) — PASSED.** One frame per fisheye stream extracted at n=300
(`ffmpeg -map 0:0 / -map 0:1`) and inspected: **stream 0 is the clean forward lens** (theatre,
bed, IV racks, whiteboard, no person); **stream 1 carries the operator** (man with backpack
holding the stick). The default `--operator_stream 1` is therefore correct for this clip, and
the mask contact sheet confirms the operator is fully covered on all 6 sampled panoramas
(`_dataset/360/theater_027_8k/qa/contact_sheet.jpg`).

**Carry-over negative (WP-O1/O2/O3, still open): bystanders are not masked.** Panoramas 57 and
75 of the contact sheet each contain a second person (a man in a blue shirt; a seated person)
who is **not** in any mask, because WP-O1 segments the operator only.

---

## 2. SIFT-parity guard — **ALIKED+LightGlue FAILS the guard; SIFT ships**

The premise handed to this WP was that CPU SIFT is the WP-O2 bottleneck and 8K crops (4× the
pixels) make it untenable, so the GPU stack should take over — *provided* it costs no quality.
Both halves of that premise were tested. **Both came out negative.**

### 2a. What was made comparable

`core/sfm_360.py --features aliked` (new, default is `sift`, production path untouched) swaps
**only** the keypoints/descriptors/matcher. Shared by construction between the arms: the same
94-panorama ingest, the same 12-crop rig geometry and fixed SIMPLE_PINHOLE intrinsics
(1920×1920, f=960), the same WP-O1 masks, the same `SequentialPairingOptions`
(`overlap=10, quadratic, loop_detection=0, expand_rig_images`), the same
`rig_verification=True`, the same `skip_image_pairs_in_same_frame`, the same
`IncrementalPipelineOptions` and the same seed 0.

Two facts prove the control held rather than merely being intended:
- **Both arms matched exactly 11 476 image pairs.** The ALIKED arm generates its pairs by
  running COLMAP's own `pycolmap.SequentialPairGenerator` on its own database with the same
  options object the SIFT matcher uses, so the pair sets are identical, not merely similar.
- **Mask integrity re-checked independently on the ALIKED arm: 0 keypoints on masked-out
  pixels** out of 133 250 checked over 60 crops. HLoc has no mask support, so
  `core/sfm_360_hloc.py` re-implements what COLMAP's `MaskFeatures` does for SIFT; this is
  the WP-O2 §"mask integrity" check repeated for the new path.

### 2b. Result (20 panoramas = 240 crops, seed 0, 11 476 pairs in every arm)

| arm | reg panos | mean reproj (crop px) | mean reproj (ERP px) | points3D | observations | kpts/crop | inlier matches | wall clock |
|---|---|---|---|---|---|---|---|---|
| **sift** | **20/20 (100 %)** | **1.2023** | **1.3095** | 9 761 | 37 871 | 3 776 | 78 518 | **173.7 s** |
| aliked (resize 1600) | 20/20 (100 %) | 1.3685 | 1.4904 | **39 330** | **138 298** | 2 387 | **296 730** | 387.6 s |
| aliked (native 1920) | 20/20 (100 %) | 1.3585 | 1.4842 | 38 792 | 135 588 | 2 755 | 293 768 | 392.0 s |

Per-stage: SIFT render 2.3 / extract 46.7 / match 105.5 / map 13.0 s. ALIKED render 2.2 /
extract 15.0 / (import 4.7 + LightGlue 301.0 + import 4.9 + verify 2.9) / map 44.4 s.

### 2c. Why this is a fail and not noise

The guard was "not worse on registration, and reproj within noise". Registration ties at
100 %. Reproj does not:

1. **The gap is ~20× COLMAP's own noise floor.** WP-O2 measured this mapper as *not*
   bit-reproducible at fixed seed, with sd ≈ 0.003–0.008 px between identical re-runs
   (`_infusion/wpo2_RECORD.md`). ALIKED is **+0.166 px** (+13.8 %) worse.
2. **It is not a "more tracks ⇒ higher mean residual" artifact.** ALIKED found 4.0× the
   points, which normally inflates mean reproj on its own, so the comparison was split by
   track length (`_infusion/wpo5/reproj_split.json`, `_infusion/wpo5_reproj_split.py`):

   | model | points | mean | median | track ≥ 4 | track = 2 | mean track len |
   |---|---|---|---|---|---|---|
   | sift | 9 761 | 1.2023 | 1.1586 | 1.3606 | 0.9054 | 3.880 |
   | aliked | 39 330 | 1.3685 | 1.4010 | 1.6554 | 1.0344 | 3.516 |

   ALIKED is worse in **every** stratum, and on a **matched population** (the top 9 761 points
   by track length in each model) the gap **widens** to **1.6656 vs 1.2023 (+0.46 px)**. Its
   mean track length is also *lower* (3.52 vs 3.88) despite 4× the points.
3. **It is not my resize choice.** Hypothesis: ALIKED ran at `resize_max 1600` on 1920² crops,
   so keypoints are detected on a downscaled image and rescaled up by 1.2× — which would
   inject localisation error directly. Tested at native resolution: **1.3585 vs 1.3685 px,
   a 0.010 px change, i.e. at the noise floor. Hypothesis refuted** — the localisation
   difference is intrinsic to ALIKED on this data, not an artifact of the pipeline.

### 2c-bis. But the two stacks do NOT disagree about the camera poses

Reprojection error is a keypoint-localisation statistic, not a pose statistic, and this
capture has no ground-truth trajectory — so before reading "+0.17 px" as "worse poses", the
two trajectories were aligned directly (`_infusion/wpo5_traj_agree.py`, Umeyama similarity on
the 20 panorama centres of `sparse_equirectangular/0`, `_infusion/wpo5/traj_agree.json`):

| vs sift | similarity scale | RMSE | max |
|---|---|---|---|
| aliked (resize 1600) | 1.0068 | **0.081 inter-pano steps** | 0.182 steps |
| aliked (native 1920) | 1.0049 | **0.065 inter-pano steps** | 0.194 steps |

**The stacks agree on where the camera was to within ~8 % of one inter-panorama step.** So the
regression is narrow: ALIKED places keypoints slightly less precisely *inside* the image, and
that shows up in the BA residual, but it does not move the reconstruction. This is recorded
because it materially weakens the "ALIKED is worse" reading, and a reviewer should see it.

### 2d. The speed premise is also false — and this is the more useful finding

**CPU SIFT at 8K is not the bottleneck, and the GPU stack is 2.3× SLOWER end to end here**
(173.7 s vs 387.6 s on identical work). SIFT *extraction* on 240 crops of 1920² took **46.7 s**;
WP-O2's whole-run 4 h 14 m was **not** extraction (92 s there) — it was matching (5 696 s) and
mapping (9 393 s), and the matching cost was driven by `--loop_detection 1` inflating the pair
count, not by SIFT's per-image speed. LightGlue's 301 s over 11 476 pairs (38 pairs/s) is
slower than COLMAP's CPU SIFT matcher at 109 pairs/s on the same pairs, because these crops
are only ~33 % valid after masking, so each image carries few keypoints and LightGlue's
per-pair overhead dominates.

### 2e. Verdict and what is left open

**SIFT ships for the full run**, and the decision rests on the *combination*, not on the
reproj number alone:

- ALIKED buys **no measurable pose improvement** (§2c-bis: the trajectories agree to 0.08
  steps), so there is no accuracy upside to pay for.
- It costs **+0.17 px of BA residual**, ~20× the mapper's own noise floor, in every track
  stratum and at matched track populations.
- It is **2.3× slower** end to end, so the speed argument that motivated the swap is gone.

Given no upside, a measurable (if narrow) downside, and a wall-clock penalty, the untouched
production path wins. `--features aliked` stays in the repo, default-OFF and fully tested.

**Honest counter-evidence, recorded rather than buried:** ALIKED produced **4.0× the 3D points
and 3.8× the inlier matches** from the same panoramas. WP-O2 chose the crop-rig arm over the
spherical arm for exactly this kind of advantage (3.7× points), and a denser SfM cloud is a
known 3DGS init lever — most relevant precisely in the textureless regions where free-view
mush originates. Whether 4× denser-but-noisier init produces a *better splat* is a
training-level question this SfM-accuracy guard cannot answer: it needs a two-arm training
A/B on identical panoramas, and that is the clean next experiment. **Nothing here says ALIKED
is a worse choice for the splat — only that it is worse at BA residual, no better at poses,
and slower.**

**Not tested (stated so it is not mistaken for tested):** ALIKED with a keypoint budget
matched *after* masking. ALIKED detects its 8 192 over the whole crop and the mask filter
removes ~67 %, leaving ~2 400–2 800 usable vs SIFT's 3 776 (SIFT is masked *at extraction*, so
its whole budget lands inside the valid region). Raising `--aliked_max_keypoints` to ~24 000
would equalise usable counts. It would not plausibly *improve* localisation precision — extra
keypoints are weaker ones — and the matched-track-population test already controls for
quantity, but it was not run.

---

## 3. Retro-analysis of the WP-O3 `_077` splat — what the user actually saw

Before touching the 8K run, the new eval was pointed at the exact artefact the user rejected
(`_dataset/360/train/wpo3_run_arm2/ckpts/ckpt_199999_rank0.pt`). This both validates the
harness and answers "what broke".

**The off-path sheet reproduces the complaint immediately** (smoke run, not committed as gate
evidence — the committed sheets are the 8K ones): at `pano_00000` the on-path reference is
already smeared, and lateral ±1 m degenerates into paint-smear; at `pano_00150` the on-path
view is decent and the laterals hold up better; and the **unvisited viewpoint (4.91 m from any
capture position) is a haze of streaks and black spikes**. That is "hot mess… only from key
angles everything lines up", measured.

**Fog is NOT the cause — recorded as a negative before it could be sold as a fix.** The WP-O3
checkpoint has **33.3 % of its 3 000 000 Gaussians below opacity 0.05** and 2 653 beyond 3× the
robust point bbox, which looks like a fog problem. It is not: rendering the same off-path and
unvisited viewpoints with those **1 000 308 low-opacity Gaussians deleted produces a visually
near-identical image** (`fogpair_*`). Whatever the mush is made of, it is not a transparent
haze.

**The model is majority-needle, and the off-path artifacts look like needles — but read the
ablation below before calling that the cause.** With the unvisited viewpoint constrained to
the convex hull of the walked positions (2.46 m from the nearest capture, i.e. inside the
space the operator walked around but never stood in), the render is a mass of **spikes**,
which is what elongated Gaussians look like from off-axis. Measured on the same checkpoint
(`_infusion/wpo5/wpo3_arm2_needle_stats.json`, `tools/prune_ply.py::effective_rank`, the
WP-Q2 / arXiv:2406.11672 metric):

| | WP-O3 ARM 2 (what the user rejected) |
|---|---|
| mean effective rank (3 = isotropic, 1 = needle) | 2.176 |
| median effective rank | 1.981 |
| **fraction erank < 2 (needles)** | **52.4 %** |
| fraction erank < 1.5 (severe needles) | 27.8 % |
| median axis aspect ratio | **6.86 : 1** |
| fraction opacity < 0.05 | 33.3 % |

For scale, WP-Q2 measured the *object* production recipe at 89.6 % needles and showed
`--erank_lambda 0.01 --erank_start 7000` collapsing that to 16.6 % at equal PSNR/LPIPS. So
this splat is a majority-needle model and the repo already ships a validated, default-OFF
lever aimed at exactly that.

**Four-way ablation, no retraining** (`_infusion/wpo5_needle_probe.py`,
`_infusion/wpo5/needle_probe_wpo3/` — 8 committed sheets + `needle_probe.json`). The same
viewpoints rendered from the same checkpoint with each population deleted:

| variant | Gaussians kept | render, off-path AND on-path |
|---|---|---|
| all | 3 000 000 (100 %) | the mush the user rejected |
| no-fog (opacity ≥ 0.05) | 1 999 692 (66.7 %) | **indistinguishable from "all"** everywhere |
| no-needles (erank ≥ 2.0) | 1 429 490 (47.6 %) | **catastrophically worse** — walls dissolve into confetti, **including at on-path viewpoints where the full render is fine** |
| neither | 432 587 (14.4 %) | worse still |

Two conclusions, and the second one is a correction to the obvious reading of the needle
statistic:

- **Fog is ruled out.** Deleting a third of the model changes nothing at any viewpoint.
- **Needles are load-bearing, so "52 % needles" is a description of how this model represents
  surfaces, NOT a localised defect that could be cut out.** They are equally essential at
  on-path viewpoints, where the render is acceptable. Therefore **"needles cause the
  off-path failure" remains a HYPOTHESIS, not a demonstrated fact** — the ablation shows only
  that they cannot be removed after the fact.

What follows from this is a constraint, not an answer: **no post-hoc prune can fix this
splat.** Only a training-time change that grows fatter Gaussians in the first place could,
and post-hoc pruning cannot simulate that. Arm D is the experiment that actually tests it.

Three consequences, all load-bearing for the rest of this WP:

1. **The pruned `_077` PLY deliverable will NOT fix the user's Postshot experience — this is
   measured, not assumed.** It is a 33 % smaller file (708.0 → 471.9 MB, 3 000 000 →
   1 999 692 Gaussians) that renders **the same** at every viewpoint tested, on-path and off.
   It is delivered because it was asked for and it is free (loads faster, less VRAM), but it
   is **not** a rescue and must not be handed over as one.
2. **Arm C's prior weakens.** `--opacity_reg 0.001` is aimed at low-opacity floaters, and on
   this scene low-opacity floaters are demonstrably not what breaks free view. Arm C is still
   run — it is a cheap test of a plausible mechanism and it bounds the safe range of a
   dangerous production default — but it is not expected to be the fix.
3. **A fourth arm was added on this evidence** (deviation from the plan, recorded in §5):
   **arm D = arm A + `--erank_lambda 0.01 --erank_start 7000`**, aimed at the measured cause
   instead of the assumed one. It is one already-implemented, already-validated, default-OFF
   lever changed against arm A, so it keeps one-lever-at-a-time.

Evidence: `_infusion/wpo5/prune_wpo3_arm2.json`, `_infusion/wpo5/wpo3_arm2_needle_stats.json`.

---

## 4. Full 8K SfM — **GATE PASSED**, with one finding that matters more than the gate

`_dataset/360/sfm/wpo5_theater_027_8k/sfm_report.json` (argv + seed + all options inside).
94 panoramas → **1 128 crops of 1920×1920, f = 960** — `core/sfm_360.py` derives the crop
geometry from the panorama size (`create_virtual_camera`, lines 174–185), and it did:
3840-high ERP gave 960² crops in WP-O2, 3840-high×2 gives 1920². Resolution-agnostic
confirmed on real 8K.

| gate item | target | measured | verdict |
|---|---|---|---|
| panoramas registered | ≥ 95 % | **94/94 = 100 %**, and **94/94 fully registered** (all 12 crops) | **PASS** |
| mean reproj (native crop px) | ≤ 1.5 px | **1.2096 px** (1.3308 px after `convert_to_equirectangular`) | **PASS** |
| trajectory sane | corridor-shaped, smooth | 1 reconstruction, `qa/trajectory.png` | **PASS** |
| wall clock | "dramatically under WP-O2's 4 h 14 m" | **63 min 46 s** (3 825.7 s) | **PASS — 4.0× faster** |

Stages: render 9.9 s · SIFT extract **198.7 s** · match (sequential overlap 10, quadratic,
**loop_detection 1**) 3 470.6 s · mapping **132.4 s** · convert 14 s. Masks behaved exactly as
in WP-O2: `valid_frac_mean` **0.3304** (WP-O2: 0.3312), min 0.1457, sphere coverage 98.1 %.

**Note on the WP-O2 4 h 14 m, since this WP was commissioned on the belief that CPU SIFT was
the bottleneck:** at 4× the pixels per crop, SIFT extraction here was **198.7 s for 1 128
crops** — 3 % of the run. Matching was 91 % of it and mapping only 132 s (WP-O2's was
9 393 s). So WP-O2's cost was matching + mapping at 1 812 crops, never SIFT's per-image speed.

### 4a. The real finding: **matching yield collapsed 8.3× on this capture, and it is not a bug**

| | WP-O2 `_077` (960² crops) | WP-O5 `_027` 8K (1920² crops) |
|---|---|---|
| images | 1 812 | 1 128 |
| **SIFT keypoints / crop (mean)** | 3 962 | **4 034 — the same** |
| verified image pairs | 211 963 | 127 202 |
| **inlier matches per verified pair** | **39.2** | **4.7** |
| points3D | 622 325 | **38 450** |
| observations / registered image | 1 393 | **126** |

**Detection is fine; matching is what fell over.** The same number of SIFT keypoints per crop
produce 8.3× fewer inliers per pair, so the triangulated cloud is **16× sparser per
panorama** than the scene WP-O3 trained on. Two plausible causes, not separated here: this
scene is texture-poor and repetitive (mint-green tiling, plain painted walls, plain ceiling)
where `_077`'s walls were covered edge-to-edge in charcoal drawings; and SIFT descriptors at
1920² are less viewpoint-repeatable than at 960².

**This retroactively changes how §2's ALIKED verdict should be read.** On the identical
20-panorama subset ALIKED+LightGlue produced **25.9 inliers per verified pair against SIFT's
6.8 (3.8×)** and 4.0× the points. The parity guard judged the stacks on BA residual, where
SIFT wins by 0.17 px; on *matching yield on a texture-poor scene* — which is what actually
determines how densely the splat is initialised — ALIKED wins by ~4×. **SIFT still ships
because that is what the gate asked and it passed, but the guard's criterion may simply be the
wrong criterion for this class of scene, and that is now the single highest-value open
experiment** (§10).

Packaged for training (`core.train_360_prep`, `prep_report.json`): 1 128 crops =
**984 train / 144 test** (12 withheld panoramas × 12 crops, 94/94 fully registered so every
panorama was eligible, zero intra-panorama leakage), 12 SIMPLE_PINHOLE cameras 1920² f=960,
38 450 points carried over verbatim.

---

## 5. Training sweep — 4 arms, one lever each

`_infusion/wpo5_train.py` (full argv per arm in `_infusion/wpo5/train_runs.json`). All arms:
the 8K crop set, `--data_factor 1` (1920² native — the whole point of the 8K run),
`--loss_mask`, `--holdout_list`, MCMC, `--cap_max 3000000`, `--antialiased`,
`--scale_reg 0.01`, `--ssim_lambda 0.2`, `--sh_degree 3`, `--coarse_factor 0`, seed 0,
**30 000 steps** (WP-O3's own ablation showed the last 170 k of its 200 k bought only −5.5 %
LPIPS, so 4 arms × 30 k buys more information than 1 arm × 200 k).

| arm | lever vs A | result |
|---|---|---|
| **A** | — (WP-O3 ARM-2 recipe) | baseline |
| **B** | `--pose_opt` | the WP-O3 hand-off's cleanest untested A/B |
| **C** | `--opacity_reg 0.001` | 10× below the value that destroyed WP-O3 ARM 1 |
| **D** | `--erank_lambda 0.01 --erank_start 7000` | **added mid-WP on §3's measurement** |

### 5a. Checkpoint forensics — the WP-O3 detection recipe, run on every arm

| arm | rc | wall | Gaussians | mean opacity | dead < 0.05 | ‖mean‖ p99 | **needles (erank<2)** | erank mean | median aspect |
|---|---|---|---|---|---|---|---|---|---|
| A | 0 | 16 m 47 s | 3 000 000 | 0.4076 | 18.4 % | 9.00 | 43.8 % | 2.164 | 7.49 : 1 |
| B | 0 | 18 m 03 s | 3 000 000 | 0.3889 | 19.7 % | 9.53 | 43.1 % | 2.164 | 7.20 : 1 |
| **C** | **3 (CRASHED)** | 6 m 59 s | 3 000 000 | **0.0077** | **99.9 %** | **36.77** | 25.3 % | 2.341 | 2.44 : 1 |
| **D** | 0 | 17 m 18 s | 3 000 000 | 0.4090 | 19.8 % | 9.34 | **9.7 %** | **2.313** | **5.15 : 1** |

(C's row is its **step-10 000** checkpoint — it never reached 30 000. A/B/D rows are step 30 000.)

### 5b. **ARM C — `opacity_reg 0.001` DESTROYS THE MODEL TOO. The safe value is 0, not "small".**

This is the most transferable result in the WP. WP-O3 established that `--opacity_reg 0.01`
(a production default) silently kills a many-view walkthrough; the obvious hope was that a
10× smaller value would give the floater control without the collapse. **It does not — it
buys about 10× more steps before the same end.**

- WP-O3 at 0.01: collapsed by step ~1 000.
- WP-O5 at 0.001: **99.9 % of Gaussians dead by step 10 000** (mean opacity **0.0077**),
  ‖mean‖ p99 blown from arm A's 9.00 to **36.77** — the identical positional runaway.
- It then escalated exactly as WP-O3 §6.7 saw: rasterizer memory climbed **4.6 GB (arm A) →
  58.8 → 60.9 → 61.7 GB** over steps 11 000–11 500 and the process died with
  `CUDA error: an illegal memory access was encountered` at step ~11 500 (**exit 3, 419 s**).
  Train PSNR was already degrading (15.8–16.0 vs arm A's 22.8) while the loss looked
  unremarkable — again, **the log does not tell you**.
- **This run also removes WP-O3's doubt about whether that crash was contention.** WP-O3
  could only say "arms A/C/F crashed while an eval shared the GPU". Here arm C had the GPU
  to itself, in a sequential sweep, and crashed the same way. **Collapse → runaway → OOM-class
  crash is the mechanism, not a scheduling artifact.**

Evidence: `_dataset/360/train/wpo5_run_C.log` (gitignored; the trace and the memory ramp are
quoted above), `_infusion/wpo5/train_runs.json` (`ckpt_health_mid` for arm C),
`_dataset/360/train/wpo5_run_C/ckpts/ckpt_9999_rank0.pt`.

### 5c. ARM D — the erank hinge does exactly what it says, at zero health cost

**Needles collapse 43.8 % → 9.7 %** (4.5× fewer), median axis aspect 7.49 : 1 → 5.15 : 1,
mean effective rank 2.164 → 2.313 — while opacity (0.409 vs 0.408), dead fraction (19.8 % vs
18.4 %), position spread (‖mean‖ p99 9.34 vs 9.00) and wall clock (17 m 18 s vs 16 m 47 s) are
all indistinguishable from arm A. This reproduces WP-Q2's object-scene result (89.6 % → 16.6 %)
on a 360 walkthrough. Whether it *helps free view* is §6's question, not this one's.

### 5d. ARM B — pose drift is small but real

`--pose_opt` moved the BA-fixed poses by a **camera-frame translation of 0.0046 units mean /
0.0150 max** and a 6D-rotation offset of ≤ 0.0104. The median inter-panorama step in this
scene is ~0.154 normalised units, so the largest camera moved ~**10 % of one inter-panorama
step**. Small — but note the eval renders held-out and off-path views at the **original SfM
poses** (test cameras' embeddings are never trained, so they stay at identity), which
slightly disadvantages arm B in §6. The drift magnitude above is recorded so that penalty can
be judged rather than guessed.

---

## 6. Free-viewpoint eval

`_infusion/wpo5_eval.py` → `_infusion/wpo5/eval/` (scorecard.json + per-arm metrics.json +
28 sheets + 2 MP4s). Arm C appears only as its step-10 000 checkpoint (`C10k`) to show the
collapse; it deliberately has no held-out metrics.

**Methodology correction made mid-eval, recorded because it changed the pictures.** The first
sheets were rendered down equator crop **4** (yaw 0), which `pano_frames` uses as the panorama
frame because its `cam_from_pano` is identity. On this capture yaw 0 points **straight at the
operator**, whose pixels are masked out of *every* training view and are therefore
unsupervised everywhere — so those sheets were measuring the one direction the splat was never
taught, and every arm looked like mush. Re-rendered with `--frame_cam 6` (yaw 180, the
scene-facing side; verified against the source crop). The flag only rotates the look
direction — `up` is yaw-invariant and the offsets are built from travel × up, so the camera
**positions are identical**. All committed sheets are the `--frame_cam 6` ones.

### 6a. Held-out (interpolative) metrics — 144 crops of 12 withheld panoramas

| arm | masked LPIPS (tiled) ↓ | masked PSNR ↑ | masked SSIM ↑ | full LPIPS ↓ | full PSNR ↑ |
|---|---|---|---|---|---|
| **A** | 0.5505 | **21.469** | 0.7140 | 0.5799 | 19.385 |
| **B** (+pose_opt) | **0.5499** | 21.366 | 0.7120 | 0.5793 | 19.379 |
| **D** (+erank) | 0.5517 | 21.456 | **0.7144** | 0.5809 | 19.379 |

**All three are the same model as far as these numbers can tell** — 0.10 dB, 0.0024 SSIM and
0.0018 LPIPS separate them. **That is the WP's thesis demonstrated:** arm D has **4.5× fewer
needles** than arm A and the interpolative split cannot see it at all. This is exactly why the
WP-O3 numbers did not predict the Postshot verdict.

**8K vs the WP-O3 proxy** — same lane, same recipe family, different capture:
masked PSNR **17.483 → 21.469 (+3.99 dB)** and masked SSIM **0.4334 → 0.7140 (+0.28)**. Masked
LPIPS went 0.5222 → 0.5505, i.e. nominally *worse*, and that comparison is **not valid**: the
tile is 128 px in both runs but the crops are 960² there and 1920² here, so the same tile
covers 4× less scene and scores much finer texture; the scene, the split and the mask
semantics also differ. Read PSNR/SSIM as the real 8K gain and ignore the LPIPS delta across
runs (as `wpo3_RECORD.md` §6.6 already instructed for cross-run comparisons).

### 6b. OFF-PATH sheets — the qualitative gate

24 matched viewpoints per arm (3 trajectory points × {on-path, lateral ±0.5 / ±1.0 m,
vertical +0.5 / −0.3 m} + 3 unvisited), rendered at identical poses for all four arms.
`_infusion/wpo5/eval/offpath/` — `cmp_*.jpg` is one viewpoint × all arms (the decisive
layout), `sheet_arm_*.jpg` is one arm × all viewpoints.

- **Near-path is now usable.** At **lateral ±1.0 m (= 1.67 inter-panorama steps)** the room
  stays structurally correct in A, B and D: the tiled wall stays planar, the stainless
  cabinet, the surgical light, the ceiling grid and the floor line all land in the right
  places. It is soft, but it is a room. The equivalent view on the WP-O3 splat degenerated
  into paint-smear (§3).
- **Genuinely unvisited viewpoints are still not usable.** The hull-restricted unvisited
  point is **4.62 m from the nearest capture position** and every arm renders incoherent
  blur there. Honest verdict: this WP moved the usable envelope out to roughly ±1–1.5 m
  around the capture path; it did not make the scene freely navigable.
- **The failure mode changed.** The WP-O3 splat produced a mass of black **spikes** at
  unvisited viewpoints; the 8K arms produce smooth **blur**. Consistent with 52 % → 44 % (A)
  and → 10 % (D) needles, though not isolated to that cause.
- Arm **C10k** renders flat grey everywhere — the collapse, visible.

### 6c. **Which arm wins off-path — measured, not eyeballed**

The sheets show A, B and D looking very similar, so "which arm wins" was made objective.
`_infusion/wpo5_viewstab.py` → `_infusion/wpo5/viewstab.json`: slide the camera sideways in
12 equal steps spanning 2 inter-panorama steps, at 8 evenly spaced panoramas, and measure
**LPIPS between consecutive renders**. A needle is correct only from the direction it was
fitted at, so it appears, streaks and vanishes as the camera moves; a well-conditioned model
changes smoothly. **The confound is stated in the script and holds here: a blurrier model also
scores lower, so this is only comparable across arms of equal on-path fidelity — which §6a
establishes.**

| arm | mean consecutive LPIPS ↓ | vs A | paired per-sweep result |
|---|---|---|---|
| A | 0.13043 | — | — |
| B | 0.13045 | +0.0 % | **3/8 sweeps — a coin flip** |
| **D** | **0.12621** | **−3.2 %** | **8/8 sweeps better than A** (mean −0.00423, sd 0.00326) |

**Arm D wins.** The margin is small (−3.2 %) but it is **consistent on every one of the 8
sweeps** — a paired sign test gives p ≈ 0.004 — and it comes with 4.5× fewer needles at
identical held-out fidelity, identical opacity health and +3 % wall clock. **`--pose_opt` is a
measured no-op** on this capture: 3/8 sweeps, mean difference +0.00001, and −0.10 dB held-out.
That closes the WP-O3 hand-off's ranked-first open question with a negative.

---

## 7. Fog / floater diagnostics

Per-arm checkpoint statistics (`_infusion/wpo5/eval/fog/ckpt_stats.json`) and, for each arm, a
render pair at an off-path and an unvisited viewpoint with and without the sub-0.05-opacity
Gaussians (`_infusion/wpo5/eval/fog/fogpair_*.jpg`).

| arm | Gaussians | mean opacity | **% below 0.05** | beyond 3× bbox | beyond 10× bbox | needles (erank<2) |
|---|---|---|---|---|---|---|
| A | 3 000 000 | 0.4076 | 18.4 % | 4 899 | 902 | 43.8 % |
| B | 3 000 000 | 0.3889 | 19.7 % | 4 124 | 764 | 43.1 % |
| **D** | 3 000 000 | 0.4090 | 19.8 % | 4 763 | 865 | **9.7 %** |
| C10k | 3 000 000 | **0.0077** | **99.9 %** | **7 781** | 61 | 25.3 % |

**All three healthy arms carry ~19 % low-opacity Gaussians and ~4–5 k spatial outliers, and
deleting them changes the render at no viewpoint** — the same negative §3 established on the
WP-O3 splat, now reproduced on three independent 8K models. Low-opacity fog is simply not the
free-viewpoint failure mode of this pipeline. The 8K arms carry *less* of it than WP-O3 did
(18–20 % vs 33.3 %) without anyone targeting it.

---

## 8. Gate

| gate item | verdict | evidence |
|---|---|---|
| SfM: ≥ 95 % panoramas registered | **PASS — 94/94 = 100 %**, all 12 crops each | §4, `sfm_report.json` |
| SfM: mean reproj ≤ 1.5 px (native crop px) | **PASS — 1.2096 px** (1.3308 px ERP) | §4 |
| SfM: trajectory sane | **PASS** — 1 reconstruction, corridor+rooms | `qa/trajectory.png` |
| SfM: wall clock ≪ WP-O2's 4 h 14 m | **PASS — 63 min 46 s, 4.0× faster** at 4× the pixels/crop | §4 |
| SIFT-parity guard before committing | **RUN — ALIKED FAILED it; SIFT kept** | §2, 3 scorecards |
| training arms complete without collapse | **PASS for A, B, D** (3 M, opacity 0.39–0.41, ‖mean‖ p99 ≈ 9). **FAIL for C** — collapsed to 99.9 % dead by 10 k and crashed at 11.5 k | §5a/§5b, `train_runs.json` |
| checkpoint forensics per arm | **PASS** — opacity, ‖mean‖, needle fraction for every arm incl. the crashed one | §5a |
| off-path sheets: best arm structurally coherent | **PASS near-path** (rooms readable, walls planar, no dominant fog at ±1 m = 1.67 steps). **FAIL at genuinely unvisited viewpoints** (4.62 m off-path is incoherent) | §6b, 28 sheets |
| fog diagnostics + pruned/unpruned pair per arm | **PASS** — and the answer is a negative: fog is not the cause | §7 |
| walkthrough MP4 + honest off-path MP4 | **PASS** | §9 |
| pruned PLY deliverables (8K best arm + retro `_077`) | **PASS** | §9 |

**GATE: PASSED on the SfM half outright. On the training/free-view half: PASSED with a
bounded claim** — the usable free-view envelope is now ≈ ±1–1.5 m around the capture path,
and beyond that the splat is still not navigable. **The final arbiter is the user in
Postshot** (§9 lists exactly what to open).

---

## 9. Deliverables — what to open

| what | path |
|---|---|
| **8K best arm (D) — full PLY** | `D:\REPOS\HYPERclone\apex-sfm\_dataset\360\train\wpo5_run_D\ply\point_cloud_29999.ply` (708 MB, 3 M) |
| **8K best arm (D) — opacity-pruned PLY** | `D:\REPOS\HYPERclone\apex-sfm\_dataset\360\train\wpo5_run_D\ply\point_cloud_29999_pruned_op05.ply` (**567.5 MB, 2 404 841 Gaussians**, −19.8 %) |
| 8K baseline (A) PLY | `_dataset\360\train\wpo5_run_A\ply\point_cloud_29999.ply` |
| **retro: WP-O3 `_077` opacity-pruned PLY** | `D:\REPOS\HYPERclone\apex-sfm\_dataset\360\train\wpo3_run_arm2\ply\point_cloud_199999_pruned_op05.ply` (**471.9 MB, 1 999 692 Gaussians**, −33.3 %) |
| walk-through MP4 (arm D, 94 SfM poses) | `_infusion/wpo5/eval/walkthrough_D.mp4` |
| **honest OFF-PATH MP4** (same walk, displaced 0.75 m laterally) | `_infusion/wpo5/eval/offpath_lateral_D.mp4` |
| off-path contact sheets (28) | `_infusion/wpo5/eval/offpath/` |
| fog pairs + per-arm checkpoint stats | `_infusion/wpo5/eval/fog/` |
| scorecards / metrics | `_infusion/wpo5/eval/scorecard.json`, `_infusion/wpo5/eval/{A,B,D}/metrics.json`, `_infusion/wpo5/viewstab.json`, `_infusion/wpo5/train_runs.json` |
| SfM A/B evidence | `_infusion/wpo5/sfm_parity_ab{,_native}.json`, `reproj_split.json`, `traj_agree.json` |
| WP-O3 retro probe | `_infusion/wpo5/needle_probe_wpo3/` (8 sheets + JSON), `prune_wpo3_arm2.json`, `wpo3_arm2_needle_stats.json` |

**Read before opening the pruned PLYs (§3):** pruning is a **file-size** change, not a quality
change. Measured on both scenes, deleting every sub-0.05-opacity Gaussian leaves the render
visually identical at every viewpoint tested. The pruned `_077` will **not** fix the free-view
experience the user rejected.

---

## 9b. ADDENDUM (2026-08-05) — gravity-aligned metric PLY + a viewpoint guide

User feedback on the arm-D PLY in Postshot: still a *"hot mess, none of this is usable…
maybe I'm just not finding the right spaces."* Two causes were addressable without
retraining — the model was in raw SfM coordinates (tilted floor, arbitrary origin and scale),
and nothing told the user where the reconstruction is actually supported.

### 9b.1 `tools/align_ply.py` — one lossless similarity transform

`x' = s R (x - c)`, baked into the full Gaussian model. What moves, and the trap in each:
**means** (`s R (x-c)`), **scales** (log-space `+= log s`; rotation does not touch them),
**quats** (`q_R (x) q`, global rotation on the LEFT, gsplat's `(w,x,y,z)` order), and — the
one that is usually skipped — **the spherical harmonics**. SH are a function of the
world-space view direction, so rotating the scene rotates every Gaussian's view-dependent
lobe; leaving them alone silently breaks specular/lighting behaviour as you orbit.

**SH rotation without hand-rolling Wigner-D.** Each band `l` spans a (2l+1)-dimensional space
closed under rotation, so `D_l` is the solution of a small linear system:
`A D = B` with `A[i,n] = Y_n(u_i)`, `B[i,m] = Y_m(R^T u_i)` over 512 Fibonacci-sphere
directions in float64, giving `c' = D c`. The basis `Y` is evaluated with **gsplat's own
`spherical_harmonics`**, so the convention matches the rasterizer by construction rather than
by assumption. `D` comes out block-diagonal per band — asserted in the tests, and itself a
correctness signal.

**Gravity, in three escalating steps, because the obvious one is wrong here:**

| estimator | result | why it was not enough |
|---|---|---|
| mean camera up (the "up" the trainer's normalisation already left at −Y) | `[0, −1, 0]` | it is the mean **stick** direction; an operator who leans the stick biases it **systematically** — measured **8.34°** off |
| normal of the plane through the 94 capture stations (the walking plane) | `[0.138, −0.989, 0.043]` | planarity ratio 7.0, so well-conditioned; halves the camera-height spread (0.644 → 0.391 u) |
| **RANSAC floor plane, re-seeded from that and iterated** | **`[0.126, −0.991, −0.047]`** | **shipped.** 1 948 inliers, 14.7 cm thickness. The iteration oscillates ~2°, so the best-inlier round wins, not the last |

A **point-height-entropy** search was also tried (maximise floor/ceiling peakiness) and is a
**negative result**: it wandered 50° away onto a wall/corridor axis and made the camera-height
spread 8× worse. Recorded so nobody retries it.

Sanity after alignment: camera height above the fitted floor **median 2.15 m** (p5 1.07,
p95 3.10) — right for a selfie stick — and a walked area of **18.2 × 13.9 m**. Under the raw
camera-up estimate the same numbers were a physically impossible 2.76 m median.

**Up axis: `y_down` (up = −Y).** Researched: Jawset's docs never state Postshot's PLY up
convention, but a PlayCanvas core developer confirms in a solved thread that a
**Postshot-exported** PLY has "positive Y axis go into the ground", and SuperSplat/PlayCanvas
apply a fixed 180° Z rotation to match that. So `y_down` ships exactly the convention Postshot
itself emits. Documented as needing a 2-minute empirical confirmation by the user;
`--up_axis {y_down,y_up,z_up,z_down}` makes the alternative one flag away.

**Scale: 4.2286 m per normalised unit**, from the measured median inter-station step
(0.15372 u) and the stated assumption that one step = 0.5 s of walking = **0.65 m**. A latent
trap was caught here: the 12 rig crops of a panorama share an optical centre exactly, so a
naive `diff(camera_centres)` is mostly zeros and its median is **0** — which silently produced
`scale = 1.0` on the first run. `distinct_centres()` collapses co-located cameras and the
report prints "94 distinct stations from 1128 cameras" so it cannot happen quietly again.

**VERIFICATION — the transform is invisible.** Renders of the same views from the original
model and from the aligned model with correspondingly transformed cameras
(`c2w' = [R·R_c | s·R·(t−c)]`):

| view | max abs diff | mean abs diff | PSNR |
|---|---|---|---|
| pano_00000_cam00 | 1.506e-02 | 7.81e-06 | 74.2 dB |
| pano_00031_cam03 | 1.787e-02 | 7.79e-07 | 92.4 dB |
| pano_00062_cam07 | 4.210e-03 | 4.93e-07 | 98.5 dB |
| pano_00093_cam11 | 7.527e-03 | 2.79e-07 | 104.8 dB |

Worst case **74.2 dB / mean 7.8e-6** — float32 rasterizer noise (tile depth-sort ties flip
after any float transform), not a modelling error. 14 new tests (`tests/test_align_ply.py`)
including a real gsplat render before/after, **and a guard that asserts the render DOES change
if the SH rotation is dropped**, so a future "simplification" cannot quietly remove it.

### 9b.2 `_infusion/wpo5_guide.py` → `_infusion/wpo5/eval/GUIDE.jpg`

A metric top-down floor plan of the 94 capture stations with **8 numbered stops** and a bearing
arrow at each, beside the render you get standing there (eye height 2.15 m — the capture's own
median, which is where the splat is supported — 75° FOV). Stops are chosen greedily by
surrounding station density with a 2.5 m minimum separation.

The bearing rule is a corrected second attempt, worth recording: scoring each direction by
"most reconstructed points in front" picks the **nearest wall** every time and produced eight
useless blurry close-ups. The shipped rule scores geometry at a *useful* distance
(3 m .. 9 m) and penalises anything inside 1.8 m, which turns the stops into corridor and
room views. 7 of 8 stops now show recognisable space; stop 4 is still a washed-out wall.

`README_POSTSHOT.md` sits beside the PLYs (and committed at `_infusion/wpo5/`): which file to
load, the up-axis and what to do if it loads wrong, the metric-scale assumption, where the good
spaces are, and the honest envelope (usable ~±1.5 m from the path; incoherent well beyond it;
the pruned file is smaller, not better).

### 9b.3 Deliverables

| what | path |
|---|---|
| **aligned + pruned (start here)** | `_dataset/360/train/wpo5_run_D/ply/point_cloud_29999_aligned_pruned_op05.ply` (567.5 MB, 2 404 841) |
| aligned, full | `_dataset/360/train/wpo5_run_D/ply/point_cloud_29999_aligned.ply` (708.0 MB, 3 M) |
| transform + verification | `_infusion/wpo5/align_armD.json` |
| viewpoint guide | `_infusion/wpo5/eval/GUIDE.jpg` + `GUIDE.json` |
| Postshot readme | `_infusion/wpo5/README_POSTSHOT.md` (copy beside the PLYs) |

**What this addendum does NOT fix:** the splat is still capture-limited. Aligning it makes it
navigable and level; it does not add the parallax that free-viewpoint navigation needs. The
user's "none of this is usable" verdict is expected to improve to "usable near the path" —
not to "freely navigable".

---

## 10. Negative findings, deviations and follow-ups — recorded honestly

1. **`--opacity_reg 0.001` destroys the model too (§5b).** The safe value is **0**, not
   "small". It reaches the same 99.9 %-dead state as 0.01, just ~10× later, then crashes the
   rasterizer. This also converts WP-O3's hedged "the crash may have been GPU contention" into
   a clean statement: arm C had the GPU to itself and crashed identically.
2. **`--pose_opt` is a measured no-op here (§6c)** — 3/8 sweeps on view-stability,
   +0.00001 mean, −0.10 dB held-out. Caveat: the eval renders at the original SfM poses while
   pose_opt moved the training cameras by up to 10 % of one inter-panorama step (§5d), so arm
   B carries a small structural handicap; the honest reading is "no benefit detectable", not
   "proven harmful".
3. **ALIKED failed the parity guard on BA residual but wins 3.8× on matching yield (§2, §4a)
   — and matching yield may be the criterion that matters on this scene.** SIFT shipped
   because that is what the gate specified and it passed. **This is the single highest-value
   follow-up:** run the full ALIKED SfM (~5× the init points on this capture: 4.0× measured on
   the 20-panorama subset) and train arm D on both inits. One command each; the code is
   already in the repo and tested.
4. **The 8K theater capture yields a 16× sparser SfM cloud per panorama than WP-O3's `_077`
   (§4a)** — 38 450 points for 94 panoramas vs 622 325 for 151. Detection is identical
   (~4 000 keypoints/crop); *matching* collapses (39.2 → 4.7 inliers per verified pair). Cause
   not isolated between "texture-poor repetitive scene" and "SIFT descriptors less repeatable
   at 1920²". A 3 M-Gaussian model seeded from 38 k points is a plausible contributor to the
   remaining off-path softness, and follow-up 3 tests it directly.
5. **Post-hoc pruning cannot fix free view (§3, §7).** Fog-pruning is a no-op at every
   viewpoint on four independent models; needle-pruning is catastrophic even on-path. Only
   training-time levers are admissible.
6. **"Needles cause the off-path failure" is still a hypothesis, not a proven cause.** Arm D
   cuts needles 4.5× and wins view-stability 8/8, which supports it — but the margin is
   −3.2 %, far from a fix, and the ablation in §3 showed needles are load-bearing where the
   render is good too.
7. **A fourth arm (D) was added mid-WP**, deviating from the assigned 3-arm sweep. It was
   added on measurement (§3), it changes exactly one already-shipped default-OFF flag against
   arm A, and A/B/C were all run as assigned. Recorded as a deviation.
8. **Held-out metrics for arm C were deliberately not computed.** Its checkpoint is 99.9 %
   dead; the forensics already establish the failure and the eval was costing ~34 min/arm.
   The first eval pass was stopped after arm D for this reason; A/B/D metrics are the full
   144-crop computation and are reused by hash-free file reuse (`--skip_existing_metrics`).
9. **The metric scale is not reliably recoverable on these captures (§6, `scale` in the
   scorecard).** Two anchors disagree by 5.7× and both are individually implausible (under
   one, the camera height varies 2.4 m over the walk; under the other, a 94-panorama walk
   fits in a 2.7 × 1.4 m room). The likely cause is the world-up estimate — WP-O2 measured
   zenith tilt median 11° / max 49° on these panoramas. **All metre labels in this record are
   therefore indicative;** the exact, assumption-free statement is
   `offset_in_steps = offset_m / 0.6`, i.e. "lateral 1.0 m" means "1.67 inter-panorama steps".
   An ingest levelling pass would fix the up estimate and is a cheap prerequisite for ever
   quoting metres.
10. **Bystanders are still unmasked** (WP-O1/O2/O3 carry-over). Panoramas 57 and 75 of the
    ingest contact sheet each contain a second, unmasked person.
11. **Not tested:** ALIKED with a post-mask-matched keypoint budget (§2e); `cap_max` above
    3 M on a 38 k-point init; `refine_stop_iter` scaled to the 30 k schedule (left at the
    25 000 default, so densification runs for 83 % of these runs, unlike WP-O3's 12.5 %);
    longer schedules for arm D; erank at other λ.
