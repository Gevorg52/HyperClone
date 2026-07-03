# WP-F2 — EDGS-style dense-matcher triangulation init — RECORD (run 2026-07-02)

**Verdict: GATE PASSED.** Dense RoMa-triangulation init raises deck SfM-point density
~4.7x, and a matched 15k A/B (dense-init ON vs OFF) IMPROVES both deck-region LPIPS
(0.3006 -> 0.2936, -2.3%) and global held-out masked-LPIPS (0.3549 -> 0.3425, -3.5%)
with NO floater regression (floaters get *cleaner* across every indicator). This
beats WP-F1's null appearance result — the principled dense cloud (not random
injection, cf. CoMapGS) helps rather than hurts.

## What was built
- `core/dense_init.py` — pairs_from_covisibility -> match_dense(RoMa) -> hloc
  triangulation (Python API; poses preserved) -> voxel-dedup merge with the original
  sparse -> augmented COLMAP model. Selectivity flag `low_density_only` (CoMapGS).
- `Hierarchical-Localization/hloc/matchers/roma.py` (new) + `roma` entry in
  `hloc/match_dense.confs` (`max_error 2, cell_size 4`, RGB). Patches:
  `tools/patches/hloc_roma_matcher.patch`, `tools/patches/hloc_match_dense_roma_conf.patch`.
- `core/training/gsplat_backend.py` — `--dense_init` opt-in in `prepare_data` +
  `get_default_config` (default OFF = production unchanged). Augmented model drops
  into the existing `init_gaussians_from_sfm` (color from COLMAP track mean -> SH DC,
  KNN scales) — no trainer change.
- `romatch` (MIT, v0.1.2) installed with the torch constraints file; torch
  2.11.0+cu128 / torchvision 0.26 / numpy 2.4.3 verified untouched. THIRD_PARTY.md
  updated (romatch, EDGS param provenance, cv2 side-effect).

## Dense-init argv + intermediate stats (Laptop deck testbed)
```
python -m core.dense_init \
  --reference_model _dataset/Laptop/apex_images/training/gsplat_prod/data/sparse/0 \
  --image_dir      _dataset/Laptop/apex_images/training/gsplat_prod/data/images \
  --out_model_dir  _dataset/Laptop/apex_images/training/wpf2_ab/dense_model \
  --work_dir       _dataset/Laptop/apex_images/training/wpf2_ab/dense_work \
  --num_matched 5 --num_workers 0
```
(matcher: roma_outdoor, coarse 560x560 fp16 upsample_preds=False, sample 15000/pair,
certainty >= 0.05; max_kps=None so the hloc 8192 default does NOT cap keypoints.)

| quantity | value |
|---|---|
| covisible pairs (num_matched=5) | 1520 (1109 unique matched) |
| RoMa keypoints/image (avg) | 91,189 (27.7M total) |
| triangulated points | 1,222,914 |
| **new-point reproj error (mean / median)** | **1.42 / 1.38 px** |
| original sparse points | 93,126 |
| dense points added (after voxel dedup) | **614,142** (dropped 608,772 as voxel-dup) |
| **final points** | **707,268 (7.59x sparse)** |
| voxel size (= median-NN of sparse) | 0.01531 (world units) |
| database.db | 425 MB |
| runtime | 15.9 min (10.5 match + 5.1 triangulate) |
Stats JSON: `_dataset/.../wpf2_ab/dense_work/dense_init_stats.json`.
New points comfortably exceed the gate's >=100k threshold (+614k), with confirmed
deck coverage (see density below).

## Deck-region SfM density (before/after) — `_infusion/wpf2_density.json`
Method: project every SfM point into the 38 name-sorted every-8 held-out views; count
points in the fractional DECK box `(0.30,0.72,0.55,0.95)` (identical to
wpf1_deck_eval.py / render_compare.py) vs a KEYBOARD box `(0.36,0.80,0.50,0.68)`
(textured keys — DEFINED + RECORDED here; the original 0.28x profiling's keyboard
region was never committed to a box). Density = points / box-kpx, mean over views.

| region | OFF (orig sparse) | ON (dense init) | rise |
|---|---|---|---|
| **DECK density (pts/kpx)** | 16.86 | **79.59** | **4.72x** |
| deck points / view (mean) | 23,499 | 110,900 | 4.72x |
| keyboard density (pts/kpx) | 14.29 | 85.35 | 5.97x |
| whole-frame density (pts/kpx) | 5.84 | 34.31 | 5.87x |

**Deck density rises materially (4.72x) — gate satisfied.** HONEST CAVEAT: the fixed
fractional box averaged over an *orbit* (the laptop moves through the frame) does NOT
reproduce the original single-view "deck = 0.28x keyboard" contrast — my deck/keyboard
ratio reads ~1.18 BEFORE (the boxes catch table/screen/background at many orbit
angles). The robust, box-placement-independent signal is the SAME-box before/after
rise (4.72x on the deck), which both arms share identically. The trained-quality
payoff is confirmed directly by the deck-region LPIPS A/B below.

## A/B: two 15k runs, dense-init ON vs OFF
Shared argv (both arms; SEQUENTIAL on an otherwise-idle RTX PRO 6000; the ONLY
difference is `sparse/0` = original vs augmented model):
```
python -m core.training.gsplat_trainer \
  --data_dir _dataset/Laptop/apex_images/training/wpf2_ab/<ARM>/data \
  --data_factor 2 --max_steps 15000 --test_every 8 \
  --strategy mcmc --cap_max 1000000 --opacity_reg 0.01 --scale_reg 0.01 \
  --antialiased --seed 0 --save_steps 15000 \
  --result_dir _dataset/Laptop/apex_images/training/wpf2_ab/<ARM>/output
```
- `off/` — data/sparse/0 = ORIGINAL ALIKED sparse (93,126 pts).
- `on/`  — data/sparse/0 = dense-init AUGMENTED model (707,268 pts).
- image dirs (`images`, `images_2`) are directory JUNCTIONS to the shared
  `gsplat_prod/data` copies -> byte-identical pixels/order/split in both arms.
- **Recipe caveat (same as WP-F1/WP-Q2 A/B):** `pose_opt=false`, no coarse-to-fine,
  normalize_world_space=true, cap_max 1M. Differs from full production
  (`pose_opt=True, coarse_factor 4`) — a clean isolation of the dense-init lever,
  matched to the WP-F1 deck-eval lineage. OFF reproduces the WP-F1 OFF baseline
  (masked LPIPS 0.3549 vs WP-F1 0.3546; deck 0.3006 vs 0.3013) — confirms the setup.

### Gate table (all numbers persisted; see artifacts)
| metric | OFF | ON (dense init) | delta | verdict |
|---|---|---|---|---|
| **masked LPIPS** (qa_report, 38 views) | 0.3549 | **0.3425** | **-0.0124 (-3.5%)** | **improved (not-worse gate ✓)** |
| masked PSNR | 23.213 | 23.336 | +0.123 dB | improved |
| masked SSIM | 0.8417 | 0.8431 | +0.0014 | improved |
| **deck-region LPIPS** (crop 0.30-0.72x,0.55-0.95y) | 0.3006 | **0.2936** | **-0.0070 (-2.3%)** | **improved ✓ (PRIMARY)** |
| full-frame LPIPS (VGG, unmasked) | 0.3549 | 0.3425 | -0.0124 (-3.5%) | improved |
| full-frame eval PSNR / SSIM (trainer) | 23.21 / 0.8416 | 23.34 / 0.8431 | +0.13 / +0.0015 | improved |
| TSED ratio (consistency) | 0.9697 | 0.9697 | 0.000 | equal (no regression) |
| TSED median epipolar (px) | 0.265 | 0.319 | +0.054 | both << 2px thresh |
| RAFT warp-err (mean) | 0.04209 | 0.04133 | -0.00076 | slightly better |
| flicker score | 0.9938 | 0.9937 | -0.0001 | equal |
| needle (erank<2) % | 91.12 | **82.72** | -8.40 pp | cleaner ✓ |
| SOR-outlier % | 0.03 | 0.01 | -0.02 pp | cleaner |
| giant-blob % | 10.93 | 9.89 | -1.04 pp | cleaner |
| low-opacity % | 10.3 | 7.1 | -3.2 pp | cleaner |
| mean effective-rank | 1.474 | 1.606 | +0.132 | cleaner ✓ |
| final Gaussians | 1,000,000 | 1,000,000 | (both at cap) | — |

**GATE: PASSED.**
- deck-region LPIPS improves (0.3006 -> 0.2936) ✓
- global held-out masked-LPIPS NOT worse — in fact improved (-3.5%) ✓
- no floater regression — floaters get cleaner on EVERY indicator ✓
- consistency (TSED/RAFT/flicker) unchanged-to-better ✓

**TSED per-pair composition note (from the persisted per-pair lists in
`qa/wpf2_ab_{off,on}/scorecard.json`):** the aggregate ratio is equal (32/33 in
both arms) and the single failing pair is the SAME pair (#16; its neighbor #15 is
also match-starved in both arms — a hard low-texture view pair), but the failure
MODE differs: OFF fails it as `few_matches` (9 SIFT matches < 10 -> unjudgeable,
counted inconsistent by policy), while ON crosses the judgeability threshold
(12 matches) and is judged a GENUINE epipolar failure (median 11.12px >> the
2.81px pair threshold). So the equal aggregate hides one genuinely-failed pair in
ON vs one unjudgeable pair in OFF. All 32 other pairs pass in both arms and ON's
judged medians are otherwise comparable (median-of-medians 0.319 vs 0.265px, both
far under threshold). Recorded as a wash, not a win.

## Determinism (added at review closeout, 2026-07-02)

**Dense-init matching (RoMa sampling):** romatch's `sample()` draws matches with
an (otherwise unseeded) `torch.multinomial`. `core.dense_init` now seeds torch
(CPU+CUDA) with `--seed` (default 0) at the start of the matching phase.
**Verified:** two seeded back-to-back runs on 3 Laptop pairs produced **bitwise
identical** features AND matches h5 content (all 6 feature + 15 match datasets,
max float diff 0.0e+00; RTX PRO 6000, torch 2.11.0+cu128, within one process).
Cross-process/cross-hardware bitwise equality is expected (inference-only forward,
cudnn.benchmark off) but was not separately verified. **Honest provenance: the
RECORDED production dense-init run above predates this fix and was UNSEEDED**
(noted in its `dense_init_stats.json`: `"seed": null`); its outputs remain valid
gate evidence (the A/B consumed the persisted augmented model, not a re-draw),
but an exact re-run from scratch would re-sample different match subsets.

**Trainer runs:** the gsplat trainer with `--seed 0` pins view order and init but
is NOT bit-deterministic (CUDA atomics in rasterization/reductions;
`torch.use_deterministic_algorithms` not set). This explains the small
OFF-arm deltas vs WP-F1's OFF arm under identical argv+seed: masked LPIPS 0.3549
here vs 0.3546 there; deck 0.3006 vs 0.3013. Same-seed trainer runs are
statistically matched, not bitwise reproducible — consistent with the WP-Q2
`--seed` note in `wpq2_ab/AB_RECORD.md`.

## Poses-preserved check (persisted per §0; added at review closeout)

Computed by `core.dense_init._pose_delta` (now asserted INSIDE
`build_dense_init` on every run and written to `dense_init_stats.json`), and
recomputed for the recorded artifacts -> `_infusion/wpf2_pose_check.json`:

| comparison (304 images) | max \|Δt\| | max ‖ΔR‖_F | angle readout |
|---|---|---|---|
| reference vs triangulated | **0.0** (exact) | 1.17e-15 | 1.7e-6 deg |
| reference vs written augmented | **0.0** (exact) | **0.0** (exact) | 2.4e-6 deg |

Translations are bitwise identical; the Frobenius deltas are at/below float64
quaternion-round-trip epsilon. (The "angle" column is arccos-based and
ill-conditioned near identity — pure float64 matmul noise reads ~1e-6 deg, which
is why the code asserts on `max_dt`/`max_dR_fro` < 1e-9, not the angle.)
Poses are preserved exactly; triangulation added points only.

## Interpretation
The dense RoMa-triangulated cloud is geometrically valid (mean reproj 1.42px,
epipolar-verified vs the fixed reference poses) and multi-view-consistent, so unlike
CoMapGS's random injection it does NOT dilute quality — it gives MCMC a better-
conditioned, denser start on textureless regions (deck 4.7x denser), yielding fewer
degenerate needles (91%->83%) and a cleaner, marginally sharper splat globally and on
the deck. Global (non-selective) injection was sufficient here; `low_density_only`
(CoMapGS selectivity) remains available but was NOT needed (global injection helped,
so the pre-approved selectivity fallback stays unused).

## Deviations from the plan
- **Matcher route:** hloc `match_dense` + RoMa (per plan). RoMa returns warp with a
  leading batch dim in romatch 0.1.2 (handled transparently by `sample`).
- **DataLoader workers:** hloc hardcodes 16 workers; on Windows with 4K images those
  spawn-workers crash. Overridden to 0 at call time from `core/dense_init.py` (RoMa
  GPU matching dominates, so serial image loading costs ~nothing) — keeps hloc edits
  to exactly {roma.py, conf entry}, no match_dense patch.
- **Coverage guard:** `import_features` iterates ALL reference images; we stub empty
  keypoints for any image absent from the covis pairs so triangulation can't KeyError
  (all 304 Laptop images had covisibility, so 0 stubs were needed at run time).
- **cv2 side-effect:** romatch pulled `opencv-python-headless 5.0.0` alongside the
  existing `opencv-python 4.13`; the trainer/hloc undistort APIs verified fine under
  5.0.0. The outcome was install-order-dependent, so it is now PINNED
  (`opencv-python-headless==5.0.0.93` + `romatch==0.1.2` in requirements.txt;
  recorded in THIRD_PARTY.md).
- **Deck/keyboard density ratio** does not reproduce the original 0.28x (fixed-box-
  over-orbit proxy limitation, documented above); the absolute deck-density rise and
  the deck-LPIPS A/B are the load-bearing evidence.

## Post-review fixes applied (2026-07-02, review verdict "gate stands with nits")

1. **`hloc/matchers/roma.py` certainty-filter bug (HIGH):** the filter was wrapped
   in `if keep.any():` — when NO match cleared `min_certainty` it returned ALL
   matches (the entire low-certainty tail) instead of an empty result. Filter now
   applies UNCONDITIONALLY (empty result is valid). Patch re-exported to
   `tools/patches/hloc_roma_matcher.patch`. Impact on the recorded run: none in
   practice (RoMa certainties on these pairs were overwhelmingly above 0.05 —
   97% of pixels on the smoke pair — and downstream epipolar verification at
   max_error 4px catches outliers), but the edge case was real. Regression test:
   `tests/test_dense_init.py::TestRomaCertaintyFilter::test_all_below_threshold_returns_empty`
   (fails against the old code).
2. **Seeded RoMa sampling (HIGH):** `--seed` (default 0) on `core.dense_init` +
   `dense_init_seed` in the backend config; determinism verified + recorded above.
3. **Pose check persisted (HIGH, §0):** `_pose_delta` computed + asserted inside
   `build_dense_init`, written to `dense_init_stats.json`; recorded-run values
   recomputed and persisted (`_infusion/wpf2_pose_check.json`, table above).
4. **cv2 pin (MEDIUM):** `opencv-python-headless==5.0.0.93` + `romatch==0.1.2`
   pinned in requirements.txt; THIRD_PARTY.md updated.
5. **RECORD accuracy (MEDIUM):** TSED per-pair composition note + trainer
   nondeterminism note added (see gate-table footnote + Determinism section).
6. **Backend defaults dedup (MEDIUM):** `prepare_data` now merges
   `{**get_default_config(), **config.extra}` exactly like `train()` — one source
   of truth, no hardcoded fallback duplicates.
7. **min_certainty vs romatch `sample_thresh` (LOW):** documented in roma.py —
   the two are DELIBERATELY equal (romatch remaps cert>0.05 to 1.0 before
   multinomial; our floor removes the sub-threshold tail sampling lets through).
   Unit tests added: `tests/test_dense_init.py` (5 tests, all pass; pytest 9.1.1
   installed into the main env with the torch constraints file — torch/numpy
   verified untouched).

## Persisted artifacts (committable)
- `_infusion/wpf2_RECORD.md` (this file)
- `_infusion/wpf2_density.json` + `_infusion/wpf2_density_eval.py` (deck/keyboard density)
- `_infusion/wpf2_ab_deck.json` (deck + full-frame VGG-LPIPS, exact crop box in JSON)
- `_infusion/wpf2_ab_floaters.txt` (floater_report both arms)
- `_infusion/wpf2_pose_check.json` (poses-preserved numbers, recorded artifacts)
- `qa/wpf2_ab_off/scorecard.json`, `qa/wpf2_ab_on/scorecard.json` (masked PSNR/SSIM/
  LPIPS per view + aggregate + tsed/raft_warp/flicker incl. per-pair TSED lists;
  deterministic)
- `tests/test_dense_init.py` (certainty-filter regression + merge-helper tests)
- `_dataset/.../wpf2_ab/dense_work/dense_init_stats.json` (dense-init counts/timings
  + pose checks + seed provenance; `_dataset/` is gitignored — recomputable from the
  argv above)
- `tools/patches/hloc_roma_matcher.patch`, `tools/patches/hloc_match_dense_roma_conf.patch`
- `requirements.txt` (cv2-headless + romatch pins)
- THIRD_PARTY.md WP-F2 section (romatch MIT, EDGS params, CoMapGS, cv2 pin note)
