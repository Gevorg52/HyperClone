# Apex Reality — 360° Capture Lane (PLAN_360, authored 2026-08-04)

**Audience: the coding agent executing the 360° work packages (WP-O1…O4, stubs in
EXECUTION_PLAN.md).** Strategy + WHY live here; environment contract and working rules are
unchanged from EXECUTION_PLAN.md §0–§1 and apply in full (default-OFF flags, validation
gates, GATE EVIDENCE RULE, POC license stance). Everything below was verified 2026-08-04:
local facts by direct execution on this machine, external facts by a research agent against
primary sources (COLMAP source @ main, GitHub API, official Insta360 manuals, ffmpeg docs,
arXiv). Tags: [V] verified · [PC] paper/community-claim · [U] unverified.

**Bottom line:** COLMAP 4.1.0 (released 2026-06-26 — exactly what we already run as
pycolmap 4.1.0) shipped native panorama SfM: an `EQUIRECTANGULAR` camera model plus a
rig-constrained perspective-crop pipeline (`panorama_sfm.py`). We validated ERP-direct SfM
end-to-end on the FlowerShop footage locally (40/40 registered, 1.22 px). Training stays on
plain perspective gsplat (crops) — every native-ERP rasterizer out there is a frozen research
artifact with no Blackwell evidence. The whole lane needs ZERO new CUDA builds.

---

## §1 Test footage analysis — FLOWERSHOP.mov [V, all measured locally 2026-08-04]

`D:\HYPERCLONE\FOOTAGE\FlowerShop\FLOWERSHOP.mov`, 422 MB, QuickTime/MOV, created
2026-07-21. **In-camera Insta360 file** (video handler `INS.HVC`) — NOT a Studio export.

- **Two independent HEVC streams, 1920×1920 each** (stream 0 = front lens → scene;
  stream 1 = rear lens → operator) + AAC audio. Dual circular fisheye, image circle
  inscribed (~200°/lens), yuvj420p full-range BT.709. 91.83 s @ 29.97 fps → 2752
  frames/lens (~17.8 + 17.3 Mbps).
- **Resolution is LOW**: 2×1920² ≈ 3840×1920 ERP equivalent. X4/X5 native 8K is
  7680×3840 — 4× the pixels. This file is likely a low-res proxy (LRV-class) or low-res
  record mode. **ASK USER for the original `.insv` / an 8K Studio export**; pilot proceeds
  on this file regardless.
- **Scene**: "Hospital of Emotions" art installation — flower-shop set in a hospital
  corridor. Fluorescent-lit, well exposed. Corridor = dense colorful flower buckets
  (feature-rich; thin structures). Side exam rooms = large textureless white walls +
  inflatable figures. Glossy vinyl floor (specular; WP-G1 phase-1 says expect baked
  reflections — accept for POC).
- **Sharpness sweep** (Laplacian-var, central crop, 0.5 s cadence): min 36 / p10 77 /
  median 357 / p90 831. The low window t≈35–60 s is mostly LOW TEXTURE (white rooms),
  not blur — re-confirms DEVELOPER_NOTES §5: absolute sharpness culls are invalid for
  video; selection must be window-relative (max-sharp per temporal window).
- **Motion**: continuous walking the whole clip (median inter-0.5s abs-diff 31, p90 46);
  no static segments; moderate motion blur + HEVC artifacts expected on this low-res
  source. (Future tie-in: prime BAD-Gaussians/deblur-lever testbed — do NOT couple into v1.)
- **Operator fills the central ~30% of the rear lens for the entire clip** (short stick,
  chest height). Rear hemisphere still carries real backward scene coverage → MASK the
  operator (SAM 2.1 video propagation), don't discard the lens. Feet/hand at nadir
  intermittently.
- No gyro/IMU stream exposed by ffprobe (Insta360 gyro lives in a proprietary trailer;
  treat as unavailable in this file).

## §2 Local capability audit [V, all by direct execution 2026-08-04]

- **pycolmap 4.1.0 has `CameraModelId.EQUIRECTANGULAR`** (params = `w,h` only — fixed
  mapping, nothing to optimize in BA) and `Camera.cam_ray_from_img` full-sphere
  unprojection verified correct (center→+z, edges→−z, quarter→45°). Note: 2D-normalized
  `cam_from_img` returns NaN for rear-hemisphere pixels — expected; spherical paths use rays.
- **Full rig support present**: `Rig`, `RigConfig`, `RigConfigCamera`, `apply_rig_config`,
  `read_rig_config`. Also OPENCV_FISHEYE, EUCM, DIVISION, THIN_PRISM_FISHEYE, etc.
- **ERP-direct SfM PROBE PASSED**: 40 ERP frames (1 fps, our own crude v360 stitch, seams
  and operator UNMASKED), pycolmap SIFT-CPU exhaustive + incremental mapping with
  EQUIRECTANGULAR single-camera → **40/40 registered, 13,408 pts, mean reproj 1.22 px,
  mean track 3.9**; trajectory bbox 11.4×1.5×3.8 (corridor-shaped), smooth steps, path
  length ≈ 10× end-to-end displacement (walk in + rooms + return). Caveats: repeated
  ceres LM warnings "dense Cholesky failure" during BA (map still healthy — watch at
  scale); a few larger steps suggest mild pose jitter/scale drift. **PROMOTED 2026-08-04 by
  WP-O2** — metrics re-derived from the probe's own sparse model with WP-O2's code (NOT
  re-run) and reproducing exactly (40/40, 13 408 pts, 1.2243 px, track 3.944, long axis
  11.41): `_infusion/wpo2/probe_erp_summary.json` + `probe_erp_scripts/` +
  `trajectory_plan360_probe.jpg`. The Cholesky warnings DO reproduce at 151 frames and in
  BOTH WP-O2 arms with no quality cost (§5 WP-O2). Caveat learned at scale: the probe's 40 s
  slice never entered the room loops, so it understated both the non-planarity of the walk
  (plane residual 0.24 → 0.88) and the pano tilt (max 16° → 49°).
- **gsplat 1.5.3 (our sm_120 AOT build) exposes `camera_model: Literal["pinhole","ortho",
  "fisheye","ftheta"]` in `gsplat.rasterization`** — fisheye-native training possible with
  NO rasterizer rebuild. Not plumbed through `core/training/gsplat_trainer.py` (perspective-only).
- **ffmpeg `v360` + `remap` filters present.** Verified on the footage:
  `[0:0][0:1]hstack → v360=dfisheye:e:ih_fov=200:iv_fov=200` → coherent 3840×1920 ERP;
  moderate seams at the ±90° lens boundaries (blend-only stitch). Good enough for SfM;
  seam bands must be feature-masked. **Insta360 Studio 5.9.10 installed 2026-08-04**
  (`C:\Program Files\Insta360 Studio\Insta360 Studio.exe`, silent Inno install, user-approved)
  — the quality-stitch export lane is available; note Studio export is a GUI workflow
  (no documented CLI), so in-pipeline automation stays on the v360 path.
- **Pipeline gaps** (recon 2026-08-04): zero 360 handling in apex-sfm; `camera_model`
  config limited to OPENCV|SIMPLE_RADIAL|PINHOLE|RADIAL (configs/defaults.yaml:34 →
  gui/stages/reconstruction.py:46 → HLoc import); `core/frame_select.py` video sharpness
  auto-cull known-broken (absolute threshold); SAM 2.1 mask stage exists; matching today
  is LightGlue exhaustive (fine ≤~400 imgs; a 360 walkthrough at 2 pano-fps × 12 crops
  ≈ 2.2k images ⇒ needs sequential+retrieval pairing in the crop lane).

## §3 External research findings (condensed; researcher agent, primary-sourced)

- **COLMAP 4.1.0 native panorama pipeline** [V, read from source `python/pycolmap/panorama.py`
  + `python/examples/panorama_sfm.py` @ main]:
  - `PERSPECTIVE_OVERLAPPING` (default): **12 crops/frame** — 4 yaw steps × pitch rows
    (−35°, 0°, +35°), 90°×90° FOV, staggered half-step yaw on non-zero pitch rows.
  - Crops of one pano = one **rig frame** (`RigConfig`, pure-rotation `cam_from_rig`,
    shared optical center), **exclusive per-pixel masks** (nearest-camera-center) so no
    pano pixel yields duplicate keypoints; matching uses `rig_verification=True`,
    `skip_image_pairs_in_same_frame=True`; BA **fixes** rig extrinsics AND virtual
    intrinsics (known by construction).
  - `SPHERICAL` mode = direct EQUIRECTANGULAR mapping (what our probe did); source comment:
    "generally faster but less accurate".
  - `convert_to_equirectangular()` folds the crop reconstruction back to one
    EQUIRECTANGULAR image per pano frame — clean final artifact either way.
  - Load-bearing preconditions flagged IN COLMAP'S OWN SOURCE: panos "approximately
    upright" (→ shoot Horizon Lock ON), and rig poses perfect "except for potential
    stitching artifacts by the spherical image provider" (→ stitch quality matters).
  - **Known bug** [V]: colmap/colmap#3599 — `panorama_sfm.py` silently hangs on some
    setups; unconfirmed fix. Validate on this box early; our own Equilib/ffmpeg crop
    generator + RigConfig is the drop-in fallback.
- **nerfstudio `equirect_utils.py`**: cruder (8 or 14 independent crops, NO rig, no
  exclusive masks, hardcoded cuda device, open crop bug #3445). Do not use. [V]
- **Native-ERP/fisheye 3DGS rasterizers** (ODGS MIT/frozen 2025-05, OmniGS GPL-3/CUDA-11.8
  pinned, 360-GS stale, Seam360GS no code found, Spherical-GOF unreleased, PFGS360 CVPR'26
  pose-free, DirectFisheye-GS single-fisheye): ALL single-paper custom-CUDA artifacts,
  zero sm_120/torch-2.11 evidence. PARK (Annex-B-style watch), do not build. [V/PC]
- **Commercial 360→splat products** (Insta360×Splatica et al.) are necessarily crop-based;
  no published crop-vs-native-ERP quality benchmark exists [U] — open empirical question,
  irrelevant for POC.
- **Reprojection libs**: ffmpeg `v360` (`e`→`flat` with yaw/pitch/h_fov/v_fov; `dfisheye`
  input) [V, official docs]; **Equilib** (`pip install pyequilib`, Apache-2.0, pushed
  2026-07-28, torch-native → GPU-batched Equi2Pers on the PRO 6000) — preferred lib;
  py360convert (MIT, numpy) as simple fallback. [V]
- **Insta360 facts**: X5 8K30/5.7K60, H.264/H.265 ≤180 Mbps; Studio exports H.264/H.265/
  ProRes; stitching modes: Optical Flow / New Optical Flow / Template — Template does NOT
  correct parallax (unusable for close foliage) [V/PC]. Keep subjects **≥0.8 m** (official
  blind-zone guidance) [V]. Balanced Exposure = Auto avoids inter-lens tearing [V].
  FlickerFix for fluorescent interiors [V]. Horizon Lock rotates frames upright via IMU —
  satisfies COLMAP's upright assumption [V]. No onboard GPS; gyro in `.insv` proprietary
  trailer; Gyroflow does not support Insta360 360° [V].
- **SphereSfM** (json87/SphereSfM, BSD-3, COLMAP fork for ERP, pushed 2026-02) exists as
  a reference if upstream EQUIRECTANGULAR misbehaves [PC]. `.insv` → rename to `.mp4`
  exposes the dual-fisheye streams [PC — our .mov already exposes them directly].

## §4 Architecture decision

**Ingest** (two accepted sources, auto-detected):
1. **In-camera dual-fisheye file** (.insv/.mov, 2 video streams): automated ffmpeg v360
   `dfisheye→e` stitch in-pipeline (validated §2). No Studio dependency. Seam-band masks
   mandatory.
2. **Insta360 Studio ERP export** (1 video stream, 2:1): preferred quality path when
   available (optical-flow stitch, FlowState/Horizon Lock, 8K). User exports manually
   (GUI workflow); Studio 5.9.10 installed 2026-08-04.

**SfM**: COLMAP 4.1.0 panorama pipeline, `PERSPECTIVE_OVERLAPPING` rig mode as the
production lane (accuracy + gives us pinhole crops with poses = training-ready), with the
`SPHERICAL`/EQUIRECTANGULAR direct lane (probe-validated) as the fast/fallback arm and as
an A/B check. Features: start with COLMAP SIFT (what the pipeline ships and the probe
used); ALIKED+LightGlue-on-crops via HLoc is a later upgrade, pairing must switch from
exhaustive to sequential+retrieval at crop counts >~800.

**Training**: plain perspective gsplat on the SfM crops (or denser re-crops sharing the
same poses — pure rotation offsets from the pano pose, no new SfM needed). Zero trainer
change beyond dataset packaging + mask plumb-through. Fisheye-native training (gsplat
`camera_model="fisheye"` on per-lens streams) is the quality upgrade A/B AFTER the crop
lane closes — it avoids double resampling (fisheye→ERP→crop) but needs trainer plumbing +
OPENCV_FISHEYE SfM (<180° crop of the 200° circle) + fisheye-aware masks.

**Masking** (all lanes): SAM 2.1 video propagation for the operator (rear-lens region) +
fixed nadir band (stick/feet) + stitch-seam bands (±90° yaw ±few°) + black fisheye corners.
Masks ride along every reprojection (ERP→crop uses the same warp).

**Rejected/parked**: native-ERP rasterizers (§3, no Blackwell evidence, custom CUDA);
nerfstudio equirect path (cruder); OpenSfM/OpenMVG spherical (COLMAP native supersedes);
gyro-based pose priors (proprietary trailer, Gyroflow unsupported — revisit only if SfM
struggles on worse captures).

## §5 Work packages (stubs registered in EXECUTION_PLAN.md; same gate discipline)

### WP-O1 — 360 ingest module — ~2–3 days — ✅ GATE PASSED 2026-08-04
> **Shipped `core/ingest_360.py` + `python -m core.ingest_360 --capture-type insta360`
> (default-OFF: no flag ⇒ exit 2, no production module imports it) + 22 tests
> (`tests/test_ingest_360.py`; suite 151→173 passing). Gate on `_077.insv`: **151 ERP
> frames** @ 3840×1920, 1.998 pano-fps, sharpness p10 454 / median 1691 (selected median
> 1886, +11.6%), **operator SAM-masked on 20/20 random frames** (0 area-guard, 0
> lost-guard), seam bands at x=W/4,3W/4 + 12° nadir present (geometric excluded 10.85%,
> total 15.46%), 46.8 s wall clock. Resolution-agnostic: `_027.insv` 2×3840² → **7680×3840
> ERP, 94 frames, same code/flags**, 105 s. `FLOWERSHOP.mov` auto-detects → 184 frames.
> Pre-stitched-ERP branch validated on a synthesised 2:1 clip (stitch skipped, seam +
> operator masks auto-off). Evidence + honest negatives: `_infusion/wpo1_RECORD.md`
> + `_infusion/wpo1/` (contact sheets, 8K samples incl. 1:1 native crops, reports, argv).
> **Two findings that change downstream plans: (1) §7's label is wrong — `_077` is the
> corridor/drawing rooms, the FLOWER SHOP itself exists only in the low-res `FLOWERSHOP.mov`
> (`_082` proxy) / `FLOWERSHOP2.mov`, so a high-res reshoot of that room is still needed;
> (2) other people in the scene are NOT masked (only the operator) — WP-O3 must be checked
> for their ghosts.** `ih_fov` calibration deferred to WP-O2 (200/200 stitches coherently
> at both resolutions and the seams are masked).**
> **Review fixes 2026-08-04** (follow-up commit; gate NOT re-run — with default flags the
> resolved scan stream and stitch yaw are unchanged): re-run guard (a second run into a
> non-empty `out_dir` without `--overwrite` is refused with exit 2 instead of orphaning the
> first run's frames); mid-pipeline ffmpeg/SAM failures exit 1 with a one-line error, write
> no report and flag the partial dir with an `INCOMPLETE` file; new `--operator_stream {0,1}`
> (default 1) makes the lens/stream order explicit and records it in `ingest_report.json` —
> **it is an assumption verified by eye on the 2026-07-21 clips only, so spot-check one frame
> per stream for any new camera/firmware**. "yaw-normalize" below is NOT implemented and is
> not needed: the dual-lens rig is rigid, so the stitch applies ONE CONSTANT `v360` yaw to
> every frame (`--yaw`; +180° with `--operator_stream 0`) — world orientation is camera pose
> and belongs to WP-O2. Tests 22→32, suite 173→183. Details: `_infusion/wpo1_RECORD.md` §6.
`core/ingest_360.py` + CLI flag (`--capture-type insta360`, default-OFF, GUI untouched).
Auto-detect source kind (2-stream dual-fisheye vs 1-stream ERP). Frame sampling:
window-relative max-sharpness (default ~2 pano-fps for walking captures; NEVER the broken
absolute cull). Stitch: ffmpeg v360 dfisheye→e (ih_fov constant, calibrate ~192–200 on a
checker seam-continuity sweep once), yaw-normalize. Masks: SAM 2.1 operator propagation
(rear hemisphere), nadir band, seam bands, written per-frame in the existing mask
convention. Output: standard `images/` + `masks/` layout the rest of the pipeline already
understands, + `ingest_report.json` (per-frame sharpness, chosen indices, stitch params).
**Gate:** on FLOWERSHOP.mov — ≥150 ERP frames with sane sharpness histogram; visual QA
contact sheet (stitch + mask overlays) persisted; operator fully covered by mask on 20
random frames; report JSON committed. Record negatives honestly.

### WP-O2 — 360 SfM lane — ~2–3 days — ✅ GATE PASSED 2026-08-04
> **Shipped `core/sfm_360.py` + `python -m core.sfm_360 --capture-type 360` (default-OFF:
> no flag ⇒ exit 2; nothing outside tests/`_infusion` imports it) + 28 tests
> (`tests/test_sfm_360.py`; suite 183→211). Both arms run on the 151-pano `_077` dataset,
> seed 0, CPU SIFT (our pycolmap wheel is built **without CUDA** — `has_cuda False`).
> **ARM A crop-rig: 151/151 registered (100%), mean reproj 1.2313 px in the 960² crop model
> / 1.3419 px after `convert_to_equirectangular()`, 622 325 points, 1 reconstruction,
> 4 h 14 m.** ARM B spherical: 151/151, 1.2694 px, 166 670 points, 20.7 m. Trajectory
> (both arms agree to ~2 %): floor plan 11.9 × 5.4 (aspect 0.45 = corridor + 2 side rooms),
> 29× the median step long, **0 steps > 5× median**, path/displacement 23.4 — matches §2's
> 40-frame probe shape at 151 frames. colmap#3599 **did not reproduce** (a heartbeat +
> `--timeout_s` watchdog stays in as insurance), so the upstream logic shipped; the crop
> generator is our own code either way because `panorama.py` is NOT in the 4.1.0 wheel —
> at tag 4.1.0 upstream shipped it as a self-contained `python/examples/panorama_sfm.py`
> (snapshots + the C++ files that pin mask/pairing behaviour:
> `tools/reference_sources/colmap_panorama/`).
> **Honest verdict on the A/B: accuracy is a DEAD HEAT** (both 100 % registered; ARM B is
> nominally *ahead* in the one comparable unit, ERP px) — §3's "SPHERICAL is generally less
> accurate" does not reproduce here. ARM A ships as the production lane for what it
> measurably buys: **3.7× the 3D points and 3.7× the observations** from the same panos, plus
> pinhole crops + poses that WP-O3 can train on directly, at 12.2× the wall clock.
> **ih_fov calibration (deferred from WP-O1) = NEGATIVE RESULT**: on 20 frames 192→1.109,
> 194→1.115, 196→1.084, **198→1.067**, 200→1.079 px, all 20/20 registered — but repeating the
> identical argv moves the number by up to 0.018 px (COLMAP's mapper is multi-threaded and
> **not bit-reproducible at a fixed seed**), and at the full 151 frames the ranking FLIPS
> (198 → 1.2694 vs 200 → 1.2588). Only 192/194 are separable. **`--ih_fov` default stays
> 200**; the delivered poses are on a 198 re-ingest, so WP-O3 must consume
> `_dataset/360/flowershop_077_ihfov198`.
> **Two findings that change the plan: (1) the panos are NOT "approximately upright" —
> measured zenith tilt median 11° / p90 27° / max 49° (both arms agree), i.e. §3's
> load-bearing COLMAP precondition is violated by this clip and it registered anyway; §6's
> "Horizon Lock ON" now has a number behind it and an ingest levelling pass is a cheap future
> lever. (2) §2's ceres "dense Cholesky failure" warnings DO reproduce at scale and in BOTH
> arms (ERP 157 / crop-rig 36 `CHOLMOD warning: Matrix not positive definite` lines) with no
> measurable quality cost — cosmetic, do not chase, re-check only if a scene fragments.**
> Deliverables: `sparse/0` + `sparse_equirectangular/0` under `_dataset/360/sfm/wpo2_*`;
> committed evidence + honest negatives in `_infusion/wpo2_RECORD.md` + `_infusion/wpo2/`
> (A/B scorecard, per-pano pose CSVs, the fixed rig JSON, trajectory plots, sweep + noise
> floor JSON, independent mask-integrity re-check: **0** keypoints on masked-out pixels
> across 1 055 352 checked).**
Vendor/adapt COLMAP's `panorama_sfm.py` logic (RigConfig 12-crop PERSPECTIVE_OVERLAPPING,
exclusive masks composited WITH our WP-O1 masks) into a `reconstruction` path selectable by
the flag; run BOTH arms on FLOWERSHOP: (a) crop-rig, (b) SPHERICAL direct (probe
replication at full frame count). Watch colmap#3599 (silent hang) — if hit, fall back to
our own Equilib crop generator + `apply_rig_config` (same math, our code). **Gate:** ≥95%
pano frames registered in the crop-rig arm; mean reproj ≤1.5 px; trajectory sanity
(corridor-shaped, smooth, loop consistency); A/B record crop-rig vs spherical (reg%, reproj,
wall-clock) persisted per GATE EVIDENCE RULE. Deliver `sparse/0` + the
`convert_to_equirectangular()` artifact.

### WP-O3 — 360 training lane — ~2–3 days — ✅ GATE PASSED 2026-08-05 (ARM 2)
> **Shipped `core/train_360_prep.py` (`--capture-type 360`, default-OFF) + two default-OFF
> trainer flags (`--holdout_list`, `--loss_mask`) + 18 tests (suite 211→**235**). Packaged
> WP-O2 ARM A into the trainer layout — names flattened `pano_cameraN/pano_XXXXX.jpg` →
> `pano_XXXXX_camNN.jpg` in BOTH the file tree and the COLMAP model, masks renamed to the
> trainer's `<stem>.png` convention, pixels untouched — giving **1 584 train / 228 test
> crops (19 withheld panos × 12, zero intra-pano leakage)**.
> **THE HEADLINE IS A NEGATIVE RESULT: the production trainer config destroys this capture,
> and the single cause is `--opacity_reg 0.01`.** ARM 1 ran 30 k steps to completion with a
> plausible loss and a full 3 M cap and rendered **flat colour**: held-out masked LPIPS
> **0.6949** / PSNR **13.86**, mean opacity **0.00025**, **99.87 % dead**, means at ±2.1 e6
> (camera cloud radius = 1). An 8-arm × 5 k-step sweep isolates it — arm G (`opacity_reg 0`,
> `scale_reg 0.01`) healthy vs arm H (`opacity_reg 0.01`, `scale_reg 0`) 99.9 % dead. Why:
> 1 584 views ⇒ any one Gaussian is rendered rarely, so the every-step opacity penalty beats
> the photometric term, and gsplat's MCMC injects **maximum** noise at near-zero opacity by
> design (`op_sigmoid(1-opacity)`) → positional runaway. `--rgb_object_masked` (WP-D1's
> object-refit lever) amplifies it on these 33 %-valid crops — ÷3 loss magnitude plus black
> mask-boundary edges — into a rasterizer memory blow-up; `--loss_mask` is the
> correct-semantics replacement. **Not 360-specific: expect this on any large many-view
> walkthrough.**
> **ARM 2** (`--opacity_reg 0 --loss_mask --max_steps 200000`, else identical): **74 min
> 32 s**, 3 M Gaussians, mean opacity 0.555, dead 21.3 %, ‖mean‖ p99 4.20. Held-out
> (228 crops): **masked LPIPS 0.5222** tiled / 0.5572 bbox, masked **PSNR 17.483**, SSIM
> 0.4334. **Ablation inside ARM 2: at the same 30 k budget the fix alone is worth +3.34 dB
> and −20.5 % LPIPS; the next 170 k steps add only −5.5 % LPIPS / +0.29 dB.** Visual gate:
> **no operator ghost, no seam-band phantom geometry** (crops 5/7 point straight down the
> v360 seams), corridor walkable (`walkthrough.mp4`, 151 SfM poses). The splat is
> **structurally correct but soft** — capture-limited by the 1 920²-per-lens proxy, which
> re-arms §7's 8K request; `_027`/`_028` are already WP-O1-proven at 7680×3840.
> Honest negatives: the skin-tone bystander detector FAILS here (cream paper ⇒ 228/228 false
> positives, so the bystander check is visual over ~25 crops, not exhaustive — masking ALL
> people is the WP-O1 follow-up); the held-out split is **interpolation**, so the numbers are
> optimistic for true novel views; ARM 2 moves three levers at once (the sweep isolates two);
> `pose_opt` is OFF in both arms on a reasoned-but-unmeasured argument (the 12 crops share an
> optical centre by construction) and is the cleanest next A/B.
> Evidence: `_infusion/wpo3_RECORD.md` + `_infusion/wpo3/`; PLY
> `_dataset/360/train/wpo3_run_arm2/ply/point_cloud_199999.ply`.
Dataset writer: rig crops → the trainer's expected COLMAP layout (pinhole cameras, poses
from rig frames; optionally densified re-crops [more yaw steps / 2 extra pitch rows] sharing
pano poses). Per-crop masks → existing loss-mask plumbing. Train baseline config (3M cap,
current production levers) on FLOWERSHOP. Held-out: withhold every Nth pano frame's crops
(all 12 together — no intra-pano leakage). **Gate:** trains to completion; held-out
masked-LPIPS recorded (baseline number — no pass bar yet, this is the first 360 splat);
visual checks persisted: no operator ghosts, no seam-band phantom geometry, corridor
walkable in viewer. Negative results (e.g., low-res source too soft) recorded honestly —
they justify requesting the 8K original.

### WP-O5 — 8K validation + free-viewpoint quality — ✅ GATE PASSED 2026-08-05
> **Commissioned by the user's Postshot verdict on the WP-O3 `_077` splat — UNUSABLE in free
> view ("hot mess… only from key angles everything lines up") — against an interpolative
> holdout that said masked LPIPS 0.5222. §5's WP-O3 entry had already flagged that split as
> the wrong instrument; this WP built the right ones and re-ran the lane on real 8K.**
> **SfM on `_027` (94 panos @ 7680×3840 → 1 128 crops of 1920², f=960; `core/sfm_360.py`
> derives crop geometry from pano height, so 8K needed no code change): 94/94 registered
> (100 %, all 12 crops each), mean reproj 1.2096 px crop / 1.3308 px ERP, ONE reconstruction,
> 63 min 46 s — 4.0× faster than WP-O2's 4 h 14 m at 4× the pixels per crop.** Stage split:
> render 9.9 s · SIFT extract 198.7 s · match 3 470.6 s · map 132.4 s. §4's plan said "ALIKED
> +LightGlue on crops is a later upgrade"; it was built (`--features aliked`, default-OFF) and
> **it failed the parity guard**: at 11 476 provably identical pairs it is +0.166 px reproj
> (~20× the mapper's noise floor), worse in every track-length stratum, worse on a matched
> track population, unchanged at native resolution, and 2.3× slower. It does yield 4.0× the
> points and agrees with SIFT on poses to 0.08 of an inter-panorama step — so the regression is
> narrow, and the parity criterion may be the wrong one here (see the follow-up below).
> **Training, 4 arms × 30 k** (baseline / +pose_opt / +opacity_reg 1e-3 / +erank 0.01@7k).
> **`--opacity_reg 0.001` destroys the model exactly like 0.01 did, only ~10× later** — 99.9 %
> dead by step 10 000, then a 4.6 → 61.7 GB rasterizer blow-up and a CUDA illegal memory access
> at ~11 500. **The safe value is 0.** This run also had the GPU to itself, which removes
> WP-O3's doubt about whether that crash was contention.
> **`--pose_opt` is a measured NO-OP** (view-stability 3/8 sweeps, +0.00001; −0.10 dB held-out)
> — WP-O3's ranked-first open question, closed with a negative.
> **Held-out metrics cannot rank the arms** (masked PSNR 21.469 / 21.366 / 21.456, LPIPS
> 0.5505 / 0.5499 / 0.5517) even though one arm has 4.5× fewer needles — which is the whole
> point. A GT-free **viewpoint-stability** metric (LPIPS between consecutive renders of a small
> lateral sweep, valid only at equal on-path fidelity) ranks **arm D (erank) first, winning
> 8/8 sweeps by 3.2 %**. 8K vs the proxy: masked **PSNR 17.483 → 21.469**, SSIM
> **0.4334 → 0.7140** — §7's 8K request is vindicated.
> **Free-view verdict, bounded honestly: usable to ≈ ±1–1.5 m off the capture path** (rooms
> readable, walls planar at 1.67 inter-panorama steps, where `_077` smeared) **but a genuinely
> unvisited viewpoint 4.62 m off-path is incoherent in every arm.**
> **Two transferable negatives.** (1) **Post-hoc pruning cannot fix free view** — deleting all
> sub-0.05-opacity Gaussians (18–33 % of the model) is visually a no-op at every viewpoint on
> four independent models, and deleting needles is catastrophic even on-path because they ARE
> the surface. (2) **The `_027` theatre yields a 10.1× sparser cloud per panorama than `_077`
> (per-pano normalized; corrected per review 2026-08-05)
> — detection is identical (~4 000 kpts/crop) but MATCHING collapses, 39.2 → 4.7 inliers per
> verified pair** (texture-poor repetitive tiling and/or SIFT at 1920²; not isolated).
> **Highest-value follow-up: the ALIKED-init A/B** (≈5× denser init on this capture).
> **ADDENDUM (same day, after the user retested in Postshot):** `tools/align_ply.py` +
> `_infusion/wpo5_guide.py`. The delivered PLY was in raw SfM coordinates — tilted floor,
> arbitrary origin and scale — which is most of why navigation felt disorienting. One lossless
> similarity transform now bakes in gravity alignment (floor at Y=0), recentring and metric
> scale (4.2286 m/unit under a stated 0.65 m-per-station assumption), with **up = −Y**, the
> convention Postshot itself exports. Means, scales, quats AND spherical harmonics are rotated
> (SH by per-band least squares against gsplat's own basis); verified lossless at
> **min PSNR 74.2 dB**. §6's "Horizon Lock ON" advice now has a fallback: gravity IS
> recoverable post-hoc, but NOT from the mean camera up — on a hand-held stick that is
> systematically **8.3°** off, so the walking-plane normal + an iterated floor RANSAC are what
> actually work. A `GUIDE.jpg` (metric floor plan + 8 best-supported stops with renders) and
> `README_POSTSHOT.md` ship alongside. Suite 263 → **277**.
> New: `core/sfm_360_hloc.py`, `--features/--aliked_*/--retrieval_num`, `tools/prune_ply.py`
> (+ the erank/needle metric), `_infusion/wpo5_{sfm_ab,reproj_split,traj_agree,train,eval,
> needle_probe,viewstab,pipeline}.py`, 26 tests (suite 235 → 263).
> Evidence: `_infusion/wpo5_RECORD.md` + `_infusion/wpo5/`.

### WP-O4 (optional, after O3 closes) — fisheye-native A/B — ~2–4 days
Plumb `camera_model` through gsplat_trainer (flag, default "pinhole"); OPENCV_FISHEYE
per-lens SfM (2-camera rig, circle cropped <180°, black-corner+operator masks); train on
raw fisheye frames. **Gate:** A/B vs WP-O3 on the same held-out pano views (render crops
from the fisheye model for comparison) — masked-LPIPS + visual; adopt only if clearly
better. If not run by 2026-09, park.

## §6 Capture protocol — future 360 shoots (give to whoever operates the camera)

- X4/X5 in **8K30 360 mode** (5.7K60 only if motion blur proves worse than resolution
  loss). H.265 max bitrate.
- **FlowState + Horizon Lock ON** (upright panos = COLMAP precondition). Measured cost of
  ignoring it, WP-O2 on the un-locked `_077` clip: pano zenith tilt **median 11° / p90 27° /
  max 48.6°** off the mean zenith (both arms agree to 0.3°) — yet COLMAP still registered
  **151/151** at 1.23 px, so the "approximately upright" precondition is softer than its
  wording. Shoot with Horizon Lock anyway (free); an ingest levelling pass is the cheap
  retro-fit for footage that lacks it. **Balanced
  Exposure = Auto**; **FlickerFix ON** indoors under fluorescent/LED.
- Walk SLOWLY (half normal pace), stick overhead or chest-forward but CONSISTENT; keep
  **≥0.8 m from everything** (official blind-zone; close foliage = seam ghosts); loop the
  space and re-close the loop (revisit start); enter each room in a small arc, not a spin.
- Prefer keeping the original **`.insv`** files; if exporting: Insta360 Studio,
  full-res ERP, **Optical Flow stitching** (never Template), H.265 high bitrate, no
  aggressive denoise ("Remove Grain" OFF for texture fidelity — verify visually once).
- One operator, no crew following; wristbands/phones in pockets (they become floating
  ghosts otherwise).

## §7 Open questions for the user
1. ~~Original `.insv`?~~ PARTIALLY RESOLVED 2026-08-04 — user supplied originals.
   **Footage inventory** (all same venue/shoot, 2026-07-21):
   - `VID_..._133413_00_077.insv` — 2×1920² @30, 75 s, ~46 Mbps — **PILOT PRIMARY**
     (user-designated). CONTENT CORRECTED by WP-O1 (2026-08-04): corridor +
     charcoal-drawing rooms, NOT the flower shop.
   - `VID_..._125954_00_049.insv` — 2×1920² @30, 93 s — post-it room (WP-O1 correction)
   - `FLOWERSHOP.mov` (≙ proxy of missing `_082`; matching `LRV_..._082.lrv` present) +
     `FLOWERSHOP2.mov` (45 s) — in-camera low-res files
   - `VID_..._123244_00_027.insv` — **2×3840² 8K**, 47 s, ~128 Mbps — operating-theater
     room (IV-bag art) — 8K SCALING TEST ASSET
   - `VID_..._123352_00_028.insv` — **2×3840² 8K**, 75 s, ~134 Mbps — CRT security room
     (dark, LED-lit, monitors showing operator — hard-mode scene: screens + low light)
   The flower-shop scene itself exists ONLY in the low-res `FLOWERSHOP.mov` /
   `FLOWERSHOP2.mov` (+ the `_082` `.lrv` proxy) — WP-O1 correction; the camera was switched
   to 8K for the other rooms. A high-res re-shoot of the flower shop per §6 remains
   desirable.
2. ~~OK to install Insta360 Studio?~~ RESOLVED 2026-08-04: user approved; 5.9.10 installed.
3. Is 360 the primary capture modality going forward (affects how much GUI wiring WP-O1
   eventually gets), or one input among many?

## §8 THIRD_PARTY ledger additions (when vendored — POC stance: log-only)
COLMAP `panorama_sfm.py`/`panorama.py` (BSD-3, colmap/colmap @ main) · Equilib
(Apache-2.0) · py360convert (MIT, fallback) · SphereSfM (BSD-3, reference-only) ·
ffmpeg v360 (LGPL/GPL, system binary).
