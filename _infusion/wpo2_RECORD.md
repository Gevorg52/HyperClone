# WP-O2 — 360° SfM lane — GATE RECORD (2026-08-04)

Plan of record: `docs/PLAN_360.md` §5 (WP-O2) + §3 (COLMAP panorama mechanics).
Stub: `docs/EXECUTION_PLAN.md` §WP-O. Input: WP-O1's 151-frame ERP dataset.
Deliverable: `core/sfm_360.py` (+ CLI `python -m core.sfm_360`, tests `tests/test_sfm_360.py`,
sweep driver `_infusion/wpo2_ihfov_sweep.py`, evidence collector `_infusion/wpo2_evidence.py`).

**GATE: PASSED on ARM A (crop-rig).** 151/151 panos registered (**100.00 %**, bar ≥ 95 %),
mean reprojection error **1.2313 px** in the native crop model / **1.3419 px** after folding
back to equirectangular (bar ≤ 1.5 px, both under), single reconstruction, corridor-shaped
trajectory with no step above 5× the median. Evidence in `_infusion/wpo2/` (committed);
heavy outputs (1812 crops, `database.db`, `sparse/*.bin`) live in the gitignored
`_dataset/360/sfm/*` and are reproducible from the argv in every `sfm_report.json`.

---

## 1. What was built

`core/sfm_360.py` — one WP-O1 pano dataset → a COLMAP sparse model, two selectable arms.
**Default-OFF**: `main()` returns 2 and does nothing unless `--capture-type 360` is passed;
nothing outside `tests/` and `_infusion/` imports the module, so the production
reconstruction path (`gui/stages/reconstruction.py` → HLoc) is untouched.

| arm | what it does |
|---|---|
| **A `crop_rig`** | COLMAP's `PERSPECTIVE_OVERLAPPING`: 12 virtual pinholes per pano (4 yaw steps × pitch rows −35/0/+35°, 90°×90°, half-step yaw stagger on the +35 row) sharing the pano's optical centre; the 12 crops of one pano are ONE rig frame (`RigConfig` → `apply_rig_config`) with pure-rotation `cam_from_rig`; exclusive nearest-camera per-pixel masks; matching with `rig_verification=True` + `skip_image_pairs_in_same_frame=True`; BA with `ba_refine_sensor_from_rig/focal_length/principal_point/extra_params = False`. Writes `sparse/0` (crop space) **and** `sparse_equirectangular/0` via `convert_to_equirectangular()`. |
| **B `spherical`** | single `EQUIRECTANGULAR` camera straight on the ERP frames (params `(w,h)` only — nothing to refine in BA). Replicates the PLAN_360 §2 probe at the full frame count. |

Both arms feed the **WP-O1 masks** into feature extraction. Verified against COLMAP 4.1.0
source (snapshots in `tools/reference_sources/colmap_panorama/`): `MaskFeatures`
(`feature_extraction.cc:60`) keeps a keypoint iff the mask pixel is non-zero — exactly
WP-O1's `255 = VALID / 0 = EXCLUDED`, no inversion — and `image_reader.cc:148-171` resolves
`masks/pano_00000.png` for `images/pano_00000.jpg` through its extension fallback, so WP-O1's
files are consumed as-is. In ARM A the pano mask is warped through the SAME remap as the
image (bilinear, kept only where the warp returns exactly 255 ⇒ all four ERP taps valid ⇒
conservative at the boundary) and AND-ed into the exclusive crop mask, so a pano pixel masked
by WP-O1 yields features in **no** crop. `--use_masks 0` is the no-priors A/B path
(smoke-tested on 6 panos, both arms).

Hang protection for **colmap/colmap#3599**: a stage heartbeat prints every `--heartbeat_s`
and `--timeout_s` hard-exits 3 after dumping every Python thread's stack.
**The bug did not reproduce on this box** — the upstream logic ran to completion in both
arms, so the vendored-upstream path shipped and the "our own crop generator" fallback was
never needed (our implementation *is* our own generator anyway — see §5.1).

## 2. Gate results

### 2a. A/B — crop-rig vs spherical (151 panos, seed 0, same dataset)

Full table + argv + mask integrity: `_infusion/wpo2/AB_TABLE.md`,
machine-readable `_infusion/wpo2/ab_scorecard.json`.

| metric | **ARM A crop-rig** | **ARM B spherical** | ARM B control (ih_fov 200) |
|---|---|---|---|
| panos registered | **151/151 (100.00 %)** | 151/151 (100.00 %) | 151/151 (100.00 %) |
| mean reproj err, native px | **1.2313** (960² crops) | 1.2694 (ERP) | 1.2588 (ERP) |
| mean reproj err, ERP px | 1.3419 | 1.2694 | 1.2588 |
| 3D points | **622 325** | 166 670 | 166 459 |
| observations | **2 525 083** | 675 169 | 662 428 |
| mean track length | 4.057 | 4.051 | 3.980 |
| obs / registered pano | 16 722 | 4 471 | 4 387 |
| reconstructions produced | 1 | 1 | 1 |
| images fed to COLMAP | 1 812 | 151 | 151 |
| matcher | sequential (overlap 10, quadratic, rig-expanded, loop detection ON) | exhaustive (11 325 pairs) | exhaustive |
| wall clock | **15 213 s (4 h 14 m)** | **1 244 s (20.7 m)** | 1 233 s |
| ├ render crops | 4 s | — | — |
| ├ extract SIFT | 92 s | 47 s | 47 s |
| ├ match | 5 696 s | 544 s | 542 s |
| └ incremental mapping | 9 393 s | 652 s | 643 s |

Trajectory (recomputed for all arms by `_infusion/wpo2_evidence.py` from the poses on disk,
so the same code describes every arm):

| | ARM A | ARM B | control |
|---|---|---|---|
| floor-plan extent (camera "up" removed) | 11.93 × 5.41 | 11.70 × 5.32 | 11.63 × 5.33 |
| floor-plan aspect (short/long) | 0.454 | 0.455 | 0.458 |
| height extent / std | 3.23 / 0.926 | 3.11 / 0.912 | 3.10 / 0.913 |
| median inter-pano step | 0.4079 | 0.4065 | 0.4172 |
| step p90 / median · max / median | 2.33 · 4.17 | 2.29 · 4.07 | 2.13 · 3.93 |
| **steps > 5× median** | **0** | **0** | **0** |
| path length / end-to-end displacement | 23.41 | 22.55 | 22.90 |
| long axis / median step | 29.1 | 28.7 | 27.8 |
| best-fit-plane residual rms / max | 0.882 / 2.430 | 0.864 / 2.349 | 0.862 / 2.349 |

Plot: `_infusion/wpo2/trajectory_crop_rig.jpg` (+ `_spherical`, `_spherical_ihfov200`) —
top-down floor plan coloured by pano index plus the step-size histogram. It reads exactly as
the capture did: enter at the far end, walk the corridor, two room loops on the right, walk
back along the corridor, stop mid-way. **The two arms agree on the geometry to ~2 %** on every
shape statistic despite being independent formulations (rig of pinholes vs one ERP camera) —
the strongest available cross-check that neither is drifting.

Scale reference: PLAN_360 §2's 40-frame probe reported bbox 11.4 × 1.5 × 3.8 with
"path length ≈ 10× end-to-end displacement". At 151 frames we get long axis 11.9 and
path/displacement 23.4 — same corridor, more of the walk (and more room loops), same shape.

### 2b. Trajectory sanity — verdict and the honest caveat

PASS: corridor-shaped in plan (aspect 0.45 = corridor + the two side rooms), long axis 29×
the median step, monotone smooth stepping (no step above 5× the median; p90 only 2.3×), one
single reconstruction (no fragmentation), and both arms independently agree.

Caveat recorded honestly: **the walk is genuinely non-planar** — the camera centres sit
0.88 (rms) / 2.43 (max) SfM units off their own best-fit plane, ≈ 2.2 median steps. This is
NOT an artifact of the up-vector estimate (the best-fit-plane normal is within 10° of the
mean camera zenith) and it is NOT reconstruction-specific (both arms measure it identically).
Frame-to-frame it is smooth — the median height change between neighbouring panos is 0.061,
15 % of the median step — so it reads as a hand-carried stick being raised and lowered over a
75 s walk, not as jitter or scale drift. WP-O1 §3.5 already noted the operator holding the
rig low on some frames.

### 2c. Masks actually did their job

`_infusion/wpo2/ab_scorecard.json → arms.*.mask_integrity` — an independent re-open of each
COLMAP database, 40 randomly sampled images per arm, every keypoint tested against the mask
file COLMAP was given:

| arm | keypoints checked | on masked-out pixels |
|---|---|---|
| crop_rig | 138 102 | **0** |
| spherical | 454 287 | **0** |
| spherical control | 462 963 | **0** |

ARM A crop geometry: 12 crops of 960×960 at f = 480 (the ERP equator's own angular
resolution, 0.094°/px). The exclusive masks keep 27.2 % of each equator crop and 40.9 % of
each ±35° crop by pixel area, which is **98.01 % of the full sphere by solid angle** (pixel
area ≠ solid angle on a pinhole: the retained centre of a crop carries far more sphere than
its discarded corners — the missing ~2 % is two small polar caps). Compositing the WP-O1
masks on top leaves **33.1 % mean valid area per crop (min 11.3 %)** = 91.1 % of what the
exclusive masks alone would keep, i.e. WP-O1's operator/nadir/seam bands remove a further
8.9 % of the sphere.

### 2d. ih_fov calibration sweep (deferred from WP-O1 §3.7) — NEGATIVE RESULT

`_infusion/wpo2/ihfov_sweep.json` + `.md`, produced by `_infusion/wpo2_ihfov_sweep.py`.
20 frames (identical source frames for every value — selection happens on the raw fisheye
stream and is independent of the stitch), re-ingested per value through `core.ingest_360`,
scored with ARM B (masks ON, exhaustive), rule = registration % first, mean reproj second.

| ih_fov = iv_fov | registered | mean reproj px | points3D | track | obs/img |
|---|---|---|---|---|---|
| 192 | 20/20 | 1.1086 | 17 678 | 3.679 | 3 252 |
| 194 | 20/20 | 1.1146 | 17 889 | 3.731 | 3 337 |
| 196 | 20/20 | 1.0845 | 17 742 | 3.717 | 3 298 |
| **198** | 20/20 | **1.0672** | 17 379 | 3.721 | 3 234 |
| 200 (WP-O1 default) | 20/20 | 1.0793 | 17 010 | 3.694 | 3 142 |

**The winner does not survive its own noise floor.** `_infusion/wpo2/ihfov_noise.json`
(`wpo2_ihfov_sweep.py --collect_noise`): re-running the *identical* argv on the *identical*
dataset gives ih_fov 198 → mean 1.07667 over 4 runs, range **0.0179**, sd 0.0081; ih_fov 200 →
mean 1.08202, range 0.0058, sd 0.0028. COLMAP's incremental mapper is multi-threaded and is
**not bit-reproducible even at a fixed `--seed`**. The 198-vs-200 gap (0.012 px) is inside
that spread; 192/194 (≈ +0.03 px) are the only values separated from the rest.

Confirmed at full scale: ARM B on all 151 frames gives **1.2694 px at ih_fov 198 vs 1.2588 px
at ih_fov 200** — i.e. the ranking *flips*, again by less than the noise.

Consequences, decided and recorded:
- **WP-O1's `--ih_fov/--iv_fov` default stays 200.** Nothing in the sweep justifies changing
  a gate-validated default; 200 is also the nominal lens spec.
- The brief instructed "if the winner ≠ 200, re-ingest the full 151 before the main runs", so
  the main runs WERE done on a full 151-frame re-ingest at **198**
  (`_dataset/360/flowershop_077_ihfov198`, argv in
  `_infusion/wpo2/dataset_integrity.json`). Both arms used the same dataset, so the A/B is
  internally consistent, and the ih_fov-200 control above shows the choice is immaterial.
- **WP-O3 must consume `_dataset/360/flowershop_077_ihfov198`**, because that is the dataset
  the delivered poses belong to. The two datasets are otherwise identical (same 151 source
  frames, same sharpness histogram, mean excluded 15.42 % vs 15.46 %).
- Cheap future sweep design: score on ≥ 3 repeats per value, or on a metric that is not
  BA-noise limited (e.g. seam-band photometric continuity).

### 2e. Deliverables on disk

| what | where |
|---|---|
| ARM A `sparse/0` (crop space, 12 SIMPLE_PINHOLE cams + 151 rig frames) | `_dataset/360/sfm/wpo2_crop_rig/sparse/0` (2.3 GB run dir) |
| ARM A `convert_to_equirectangular()` artifact | `_dataset/360/sfm/wpo2_crop_rig/sparse_equirectangular/0` |
| ARM B `sparse/0` | `_dataset/360/sfm/wpo2_spherical/sparse/0` (429 MB run dir) |
| committed: 151 pano poses per arm | `_infusion/wpo2/poses_{crop_rig,spherical,spherical_ihfov200}.csv` |
| committed: the 12 virtual cameras + fixed rig extrinsics (what WP-O3 needs) | `_infusion/wpo2/rig_crop_rig.json` |
| committed: ERP camera of each model | `_infusion/wpo2/cameras_*.txt` |

`rig_crop_rig.json` confirms BA honoured the fixed-rig contract: all 12 cameras still read
`SIMPLE_PINHOLE 960 960 480 480 480` and every `cam_from_rig` translation is exactly
`[0, 0, 0]`.

### 2f. PLAN_360 §2 ERP probe — promoted, not re-run

The 40-frame ERP-direct probe that unblocked this lane was reconstructed in the planning
session; its sparse model survived in that session's scratchpad. Its **metrics were
re-derived from the existing model with WP-O2's own code** (no SfM was re-run) into
`_infusion/wpo2/probe_erp_summary.json`, and its two driver scripts are committed at
`_infusion/wpo2/probe_erp_scripts/`. The probe reproduces PLAN_360 §2 exactly — 40/40
registered, 13 408 points, **1.2243 px** (§2 quotes 1.22), mean track 3.944 (§2 quotes 3.9),
long axis 11.41 (§2 quotes bbox 11.4) — and now sits on the same axes as the WP-O2 arms:

| | §2 probe (40 fr, 1 fps, **unmasked**) | ARM A (151 fr, 2 fps, masked) | ARM B (151 fr, masked) |
|---|---|---|---|
| registered | 40/40 | 151/151 | 151/151 |
| mean reproj px | 1.2243 | 1.2313 crop / 1.3419 ERP | 1.2694 |
| points3D | 13 408 | 622 325 | 166 670 |
| floor-plan extent | 11.41 × 3.92 | 11.93 × 5.41 | 11.70 × 5.32 |
| median step | 0.875 (1 fps) | 0.408 (2 fps) | 0.407 |
| steps > 5× median | 0 | 0 | 0 |
| plane residual rms | 0.242 | 0.882 | 0.864 |
| pano tilt med/p90/max ° | 5.7 / 10.0 / 15.9 | 11.0 / 27.0 / 48.6 | 11.2 / 26.9 / 48.3 |

Two things fall out. (i) The scales agree across independent runs because COLMAP normalises
each reconstruction, so these extents really are comparable. (ii) The probe looked *tamer*
than reality: at 1 fps over a 40 s slice it never saw the room loops, which is where both the
non-planarity (0.24 → 0.88) and the pano tilt (max 16° → 49°) actually come from. Good
reminder that a probe's caveats scale worse than its headline number.

### 2g. Tests

`tests/test_sfm_360.py` — 28 new tests (CLI default-OFF gate, re-run guard, clean exit-1 on a
missing image dir, the 12-rotation layout incl. the +pitch-row yaw stagger, virtual-camera
intrinsics, rig config being pure-rotation with one reference sensor, the ERP projection
landmarks, the analytic one-crop solid angle, exclusive-mask uniqueness, sphere coverage,
mask compositing being an AND that can only shrink a crop mask, the −35° row losing more to a
nadir band than the +35° row, `split_image_name` round-trip, trajectory stats). Full suite
**183 → 211 passing**; nothing pre-existing broken.

## 3. Which arm wins

**ARM A crop-rig is the production candidate** (as PLAN_360 §4 predicted) — but the reason is
NOT accuracy, and this record will not pretend otherwise:

- **Registration is a dead heat**: both arms register 151/151.
- **Reprojection error is a dead heat, and ARM B is nominally ahead in the one comparable
  unit** (ERP px: 1.2694 vs 1.3419). COLMAP's own source calls SPHERICAL "generally faster
  but less accurate" — on this scene, at this resolution, that claim is not reproduced.
  (ARM A's native 1.2313 px is measured in 960² crop pixels; the 1.3419 px figure is the same
  model after `convert_to_equirectangular()` + `update_point_3d_errors()`, which folds the
  crop→ERP mapping into the residual. Neither number is "the" accuracy; both are ≤ 1.5 px.)
- **What ARM A actually buys, measured**: 3.7× the 3D points (622 325 vs 166 670) and 3.7× the
  observations, from the same 151 panoramas — because 12 masked 960² crops carry far more
  SIFT keypoints than one 3840×1920 ERP frame capped at `max_num_features 8192`. Plus it
  natively emits pinhole crops + poses, which is precisely the input WP-O3's perspective
  gsplat trainer wants, with no extra SfM.
- **What it costs**: 12.2× the wall clock (4 h 14 m vs 20.7 m) and 5.4× the disk.

So: **ARM A ships as the 360 production lane; ARM B stays as the fast arm** — a ~20-minute
sanity/preview reconstruction, a fallback if the rig path ever misbehaves, and a genuinely
useful independent cross-check of the trajectory.

## 4. Negative findings / deviations — recorded honestly

1. **The ih_fov sweep is a negative result** (§2d): 196–200 are indistinguishable, the
   "winner" flips at full scale, and the effect is smaller than COLMAP's own run-to-run
   noise. Only 192/194 are measurably worse. WP-O1's default is unchanged.
2. **COLMAP is not bit-reproducible at a fixed seed.** `--seed` is recorded and set
   (`pycolmap.set_random_seed` + `random_seed` in the pipeline options), but the
   multi-threaded mapper still gives a different answer run to run (mean reproj sd ≈ 0.003–
   0.008 px on a 20-frame problem). Any future A/B on this pipeline must repeat runs.
3. **The panos are NOT "approximately upright"** — the precondition COLMAP states in its own
   panorama source (PLAN_360 §3). Measured from the reconstruction itself, each pano's zenith
   sits **median 11.0°, p90 27.0°, max 48.6°** off the mean zenith (both arms agree to 0.3°).
   WP-O1 applies no horizon levelling and this clip was shot without Horizon Lock. ARM A still
   registered 151/151 at 1.23 px, so the assumption is softer than it reads — but PLAN_360 §6's
   "Horizon Lock ON" instruction is now backed by a number, and a future levelling pass in
   ingest is a cheap, testable improvement.
4. **Ceres/CHOLMOD warnings reproduce at scale, in BOTH arms.** PLAN_360 §2 flagged
   "dense Cholesky failure" on the 40-frame ERP probe. Counted here (they print as
   `levenberg_marquardt_strategy.cc … Linear solver failure … CHOLMOD warning: Matrix not
   positive definite`): **ARM B 157, ARM B control 220, ARM A 36**. LM simply retries with a
   larger damping term. Quality did **not** suffer in either arm (100 % registered, ≤ 1.35 px,
   one reconstruction, physically sane trajectory), and the crop-rig arm sees ~5× fewer — so
   the warning correlates with the EQUIRECTANGULAR model, as the probe suspected, but it is
   cosmetic at this scale. Do not chase it; do re-check it if a future scene fragments.
5. **The walk is non-planar by ~0.88 rms SfM units** (§2b). Real capture behaviour, not
   reconstruction error — recorded so WP-O3 does not read it as drift.
6. **Sequential matching dominates ARM A's cost and is only ~partly useful.**
   `--overlap 10 --quadratic_overlap 1` places neighbours at 1,2,4,…,512 panos; on a
   151-pano sequence the 256 and 512 offsets never fire and 128 almost never does, so ~20 %
   of the 5 696 s is structurally wasted. Kept at upstream's default so ARM A is "COLMAP's
   pipeline as shipped"; `--overlap` is a flag if a future run wants the time back.
   Loop detection auto-downloaded `vocab_tree_faiss_flickr100K_words256K.bin` and spent
   ~380 s indexing all 1 812 crops before matching started.
7. **GPU SIFT is unavailable**: our pycolmap 4.1.0 wheel reports
   `COLMAP 4.1.0 Commit fa8e3b3 on 2026-06-26 **without CUDA**` and `pycolmap.has_cuda ==
   False`. Everything above is CPU (32 threads); the 96 GB card sat idle. `--use_gpu` exists,
   is recorded in the report, and is a documented no-op in this build. Getting GPU SIFT would
   need a CUDA-enabled pycolmap build — NOT attempted (it risks the pinned environment for a
   speedup that only affects a one-off preprocessing step).
8. **`panorama.py` is not in our wheel.** PLAN_360 §3 read it from `main`; the installed
   pycolmap 4.1.0 package contains only `__init__.py`, `_core.pyd`, `utils.py`,
   `cost_functions/`, `pyceres/`. At tag `4.1.0` upstream shipped the logic as a
   self-contained `python/examples/panorama_sfm.py` (646 lines) and only split it into a
   library later. Both revisions are snapshotted in
   `tools/reference_sources/colmap_panorama/`; the implementation follows the **4.1.0**
   revision because that is what our binary is.
9. **colmap#3599 did not reproduce** on this box in ~5 h of panorama-pipeline execution.
   The watchdog stayed as permanent insurance rather than being removed — it costs one
   thread and it is the only thing that would tell us a hang from a slow global BA.
10. **The first ARM A attempt was killed at 57 % of matching by the session harness**, not by
    any code fault. The run reported above is a single clean run from scratch (relaunched
    detached); the killed attempt contributed nothing to it.
11. **Exclusive-mask uniqueness is exact only in the continuum.** At the 1-pixel Voronoi
    boundary two neighbouring crops can both claim a direction: measured 0.57 % of random
    directions at a 256-wide pano, 0.155 % at 1 024, **0.045 % at our 3 840** (test
    `test_double_claims_are_a_boundary_effect_that_shrinks_with_resolution`). Inherent to
    upstream's per-pixel argmax, negligible at production resolution.
12. **Not addressed here (WP-O1 §3.4 carry-over): bystanders are still unmasked.** A second
    person walks through several frames. They do not appear to have broken the SfM (no
    fragmentation, no outsized steps), but WP-O3 must still be checked for their ghosts.

## 5. Not in scope / not touched

- No changes to any existing module. Additions only: `core/sfm_360.py`,
  `tests/test_sfm_360.py`, `_infusion/wpo2_ihfov_sweep.py`, `_infusion/wpo2_evidence.py`,
  `_infusion/wpo2*`, `tools/reference_sources/colmap_panorama/*`, plus doc/ledger edits.
- 5.1 The "fall back to our own crop generator" contingency in the brief was never triggered.
  Note that our implementation is our own code either way — it re-implements upstream's math
  (which had to be vendored, see §4.8) with cv2 remap tables built once instead of per pano,
  plus mask compositing upstream does not have.
- No pip installs, no torch touch, no CUDA build. ALIKED/LightGlue on crops, densified
  re-crops, and global (vs incremental) mapping are all left for later.

## 6. Reproduce

```
# ingest (winner-of-sweep dataset the delivered poses belong to)
python -m core.ingest_360 --capture-type insta360 \
  --input D:/HYPERCLONE/FOOTAGE/FlowerShop/VID_20260721_133413_00_077.insv \
  --out_dir _dataset/360/flowershop_077_ihfov198 --overwrite --ih_fov 198 --iv_fov 198 \
  --contact_sheet 12 --seed 0 --progress 5000

# ARM A (4 h 14 m)                     # ARM B (21 m)
python -m core.sfm_360 --capture-type 360 --arm crop_rig \
  --dataset _dataset/360/flowershop_077_ihfov198 --out_dir _dataset/360/sfm/wpo2_crop_rig \
  --overwrite --matcher sequential --overlap 10 --quadratic_overlap 1 --loop_detection 1 \
  --seed 0 --timeout_s 21600 --heartbeat_s 300 --progress 50
python -m core.sfm_360 --capture-type 360 --arm spherical \
  --dataset _dataset/360/flowershop_077_ihfov198 --out_dir _dataset/360/sfm/wpo2_spherical \
  --overwrite --matcher exhaustive --seed 0 --timeout_s 21600 --heartbeat_s 300

# sweep + noise floor + evidence
python _infusion/wpo2_ihfov_sweep.py --input <clip.insv> --work _dataset/360/_ihfov \
  --out _infusion/wpo2/ihfov_sweep.json
python _infusion/wpo2_ihfov_sweep.py --collect_noise --work _dataset/360/_ihfov \
  --out _infusion/wpo2/ihfov_noise.json
python _infusion/wpo2_evidence.py \
  --arm crop_rig=_dataset/360/sfm/wpo2_crop_rig \
  --arm spherical=_dataset/360/sfm/wpo2_spherical \
  --arm spherical_ihfov200=_dataset/360/sfm/wpo2_spherical_ihfov200 \
  --log crop_rig=_dataset/360/sfm_armA.err \
  --log spherical=_dataset/360/log_wpo2_spherical.log \
  --log spherical_ihfov200=_dataset/360/log_wpo2_spherical_ihfov200.log

python -m pytest tests/test_sfm_360.py -q      # 28 passed;  tests/ -> 211 passed
```

## 7. Hand-off to WP-O3

- Use `_dataset/360/sfm/wpo2_crop_rig/sparse/0` + `_dataset/360/sfm/wpo2_crop_rig/images/`
  + `.../masks/` directly: 1 812 pinhole crops at 960×960, f = 480, cx = cy = 480, one rig
  frame per pano, per-crop masks already compositing operator/nadir/seam **and** the
  exclusive tiling (mean 33.1 % of each crop is valid — the trainer's loss mask must honour
  that or it will train on duplicated and masked pixels).
- Held-out splits must withhold **whole rig frames** (all 12 crops of a pano together);
  `frames.bin` in the model gives the grouping, `_infusion/wpo2/rig_crop_rig.json` the
  fixed rig extrinsics if crops need to be re-derived or densified.
- 622 325 SfM points are available for Gaussian initialisation — ~3.7× what the ERP arm gives.
- Expect ghosts from the unmasked bystanders (§4.12) and check the ±90° seam bands: they are
  excluded from FEATURES but the crop images still contain the seam pixels, so the trainer's
  mask plumb-through is what keeps them out of the loss.
