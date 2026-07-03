# GENUINE-HOLE testbed — standing completion benchmark (WP-C1 Part 1, 2026-07-02)

Purpose: a REAL geometric void with **observed% < 100** (measurable by
`core/qa/visibility.py`), so the completion lane can finally be judged on an
actual coverage hole instead of the June wedge (which the re-judgment proved was
**100% observed** — an angular-extrapolation test, not a hole). This doc is the
standing benchmark; the gate loop that consumes it is `wpc1_RECORD.md`.

## TL;DR
- **Scene:** `GPU_V4` (ZOTAC RTX card on polka-dot cloth, full-scene room dome;
  569 imgs, factor-2 production ckpt `…/gsplat_prod/output/ckpts/ckpt_29999_rank0.pt`,
  2.0M Gaussians).
- **Void:** footprint-aware **depth-column deletion** — a coherent central hole.
- **Holed ckpt:** `_infusion/hole/holed_gpuv4.pt` (1,235,399 Gaussians; 764,601 =
  38.2% deleted). Carve metadata stored in the ckpt under key `carve`.
- **Hole-crossing arc:** name-sorted views **293–315** (23 views), `wedge:293:316`.
- **Observed% on the arc:** mean **80.6%**, per-view range **77.1–84.8%** — squarely in the
  40–85% target band. Away from the arc most views are **100%**, BUT grazing views on the
  void-facing side still see **6–9% of the void** (v0060 → 93.9%, v0540 → 90.8% observed): the
  void is localized to one excludable arc but is NOT hard-zero outside it. That residual grazing
  coverage + MCMC regrowth is exactly what the no-fill CONTROL exploits — see the four-way
  attribution in `wpc1_RECORD.md` §CLOSEOUT (H2). (Earlier "away from the arc: 100%, verified"
  was an overstatement — corrected 2026-07-02 closeout.)
- **Baseline banked (full QA suite):** `qa/hole_baseline/` — obs 80.6%, masked
  PSNR 15.23 / SSIM 0.7512 / LPIPS 0.5518; crossscore 0.6052, nova 0.3664,
  tsed 1.000, raft_warp 0.01486, nvs_sqa −2.358, flicker 0.9910.

## Why NOT pure view-withholding (the task's recommended first route)
Both production scenes are **dense, ~complete domes** (measured with
`core/enhancement/coverage.py`): Laptop = **100%** upper-hemisphere coverage (304
cams, full 360° azimuth), GPU_V4 = **95%** (569 cams). Consequences, all measured:
1. **Foreground deletion does not make a void.** Deleting the whole depth column of
   a central image box (26–38% of all Gaussians) left observed% at **~100%** — the
   scenes are full of pervasive low-opacity HAZE (only 230k of GPU_V4's 2M
   Gaussians have opacity>0.1); large background Gaussians centred outside the box
   still splat into it (a "present-but-smudged" smear, alpha stays >0.3). A clean
   `alpha<0.3` void needs FOOTPRINT-aware deletion (remove any Gaussian whose
   3σ projected footprint overlaps the column), which we implemented.
2. **The deleted content is seen from ~half the orbit** (broad-scan: the void is
   visible across ~views 170–375, the whole side facing the deleted card region).
   So "withhold ALL views that see it" ≈ withhold ~half the dataset — destructive,
   and MCMC would still partially heal from grazing views. A stable-under-retrain
   void is not achievable on a dense dome without wrecking the reconstruction.
This is a real extension of the June re-judgment finding: **dense captures resist
coverage holes.** The honest, controllable route is geometry DELETION.

## Protocol (reproducible)
Tool: `_infusion/hole_carve.py` (measures observed% with the SAME
`core/qa/visibility.py` the gate uses — ground truth).
```
python _infusion/hole_carve.py \
  --ckpt _dataset/GPU_V4/apex_images/training/gsplat_prod/output/ckpts/ckpt_29999_rank0.pt \
  --data _dataset/GPU_V4/apex_images/training/gsplat_prod/data --factor 2 \
  --view 300 --box 0.5,0.42,0.52,0.58 --depth 0,3 --span 15 --step 1 \
  --out _infusion/hole/holed_gpuv4.pt --emit_split _infusion/hole/split.json
```
- **Mechanism:** delete every Gaussian whose projected centre (grown by 0.5×its 3σ
  screen radius = footprint overlap) lands in image box (cx,cy,w,h)=(0.5,0.42,0.52,
  0.58) of view 300, at view-space depth in [0, 3×median-object-depth ≈ 8.0].
- **Depth-bounding (d1 = 3× median) is load-bearing:** the whole column (0,∞)
  deleted a cone to infinity that ALSO voided the opposite arc (views ~92–108 →
  65–75%). Bounding to 3× median removes only the far-INFINITY leakage: the void
  stays clean on 293–315 while the opposite arc returns to **100%** — the void is
  now confined to one excludable arc (needed for the completion distill).
- **Export:** `_infusion/hole_export.py` dumps, per arc view, the holed render, the
  OGGSplat unobserved mask (white=fill), the alpha, the real GT photo, and the
  camera (K, c2w, normalized frame) → `_infusion/hole/arc/` (fill + distill inputs).

## Validation
- `qa_report.py … --split wedge:293:316 --scorers all` on the holed ckpt →
  `qa/hole_baseline/scorecard.json`: observed **80.6%**, `render_sanity ok`,
  full 6-scorer bank. The void renders near-empty (low crossscore/high nova) — the
  clean "generate-into-emptiness" substrate the program's house rules require.
- Localization confirmed by a whole-trajectory broad scan: only ~views 170–375
  show any void; the primary in-band arc is 293–315.

## Held-out real photos (reference-only)
The 23 arc GTs (`arc/vXXXX_gt.png`) are the real photos of the void region, used
ONLY for a "plausible-vs-actual" comparison (NOT a gate criterion — unobserved
regions have no GT claim). NOTE: these were TRAINING views of the production splat;
the void exists because the GEOMETRY was deleted, independent of which split the
photo belonged to. The distill (`hole_distill.py`) holds this arc out of REAL
supervision (APEX_HOLDOUT) so the void is filled ONLY by the generator.

## Files
- `_infusion/hole_carve.py` — carver + observed% measurement (footprint-aware).
- `_infusion/hole_export.py` — per-view fill/distill inputs.
- `_infusion/hole/holed_gpuv4.pt` — the holed baseline splat.
- `_infusion/hole/split.json` — hole arc + carve params.
- `_infusion/hole/arc/` — render/mask/alpha/gt/meta per arc view.
- `qa/hole_baseline/` — banked baseline scorecard (the gate reference).
