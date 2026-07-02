# ArtiFixer wedge RE-JUDGMENT under the WP-Q1 product-metric suite (2026-07-02)

First scientific use of the visibility-split QA harness (`core/qa/`) to re-judge a
June verdict. Re-scores the June ArtiFixer3D artifacts on the Laptop **wedge**
(40-frame contiguous held-out arc, frames 0131–0170) and, for contrast, the
**easy** every-8th case — under masked metrics + the 6 product scorers instead of
bare held-out-GT LPIPS. Driver: `_infusion/wedge_rejudge.py` (a normalize-
configurable consumer of `core/qa/`; nothing in `core/qa/` modified).

## Evidence (persisted — GATE EVIDENCE RULE)
Per-scene scorecards (JSON + HTML + per-view thumbnails):
`qa/rejudge_wedge_base/`, `qa/rejudge_wedge_af3d/`,
`qa/rejudge_easy_base/`, `qa/rejudge_easy_af3d/`.
PLYs re-exported from the surviving WSL 3DGRUT checkpoints:
`artifixer-checkpoints/ply/laptop_base_3dgrut_wedge.ply`,
`…/laptop_artifixer3d_wedge.ply` (+ the pre-existing easy pair). COLMAP scene
staged at `artifixer-checkpoints/rejudge/scene_laptop720/` (305 imgs, 304
registered, 720p). Driver argv (both arms identical except PLY + split + out):
`python _infusion/wedge_rejudge.py <ply> <scene> --split {wedge:130:170|every8}
--scene <name> --out qa/<name> --scorers all` (factor=1, normalize=False,
sh_degree=3, seed=0).

## Pipeline validation (the load-bearing check)
The June eval scored FRAME DUMPS; the PLY→project-renderer path was never
validated. Two independent checks confirm it here:
1. **Coordinate frame.** 3DGRUT PLYs live in the RAW COLMAP world frame →
   `normalize=False`. Probe (`align_probe.py`): render-vs-GT reproduces June's own
   base-vs-GT to 0.06 dB (mine 22.98 vs June 22.92); `normalize=True` = garbage
   (7.8 dB). Decisive 15 dB separation.
2. **Deltas reproduce June.** Masked-LPIPS Δ(base→af3d): **easy −10.6%** (June
   −10.8%), **wedge −3.6%** (June +4.0% i.e. −4.0% LPIPS). Absolute LPIPS differs
   from June (bbox-crop LPIPS + undistortion vs June full-frame `get_lpips`) but
   the deltas match → alignment + format (SH deg 3, logit-opacity, log-scale,
   SH-DC) all correct.

## Full re-judgment table
Both cases: 40 (wedge) / 38 (easy) held-out views, **100.0% observed area**
(see finding #1). Arrows: ↑=higher better, ↓=lower better.

| metric (dir) | WEDGE base | WEDGE af3d | Δ | | EASY base | EASY af3d | Δ |
|---|---|---|---|---|---|---|---|
| masked LPIPS ↓ | 0.3626 | **0.3496** | −3.6% ✓ | | 0.3420 | **0.3059** | −10.6% ✓ |
| masked PSNR ↑ | 21.224 | **21.522** | +0.30 ✓ | | 22.401 | **22.563** | +0.16 ✓ |
| masked SSIM ↑ | 0.8226 | **0.8262** | +0.004 ✓ | | 0.8207 | **0.8249** | +0.004 ✓ |
| CrossScore ↑ | 0.8010 | **0.8244** | +0.023 ✓ | | 0.7943 | **0.8336** | +0.039 ✓ |
| NOVA ↓ | 0.1743 | **0.1611** | −0.013 ✓ | | 0.1611 | **0.1408** | −0.020 ✓ |
| TSED ↑ | **0.9091** | 0.8788 | −0.030 ✗ | | 0.9091 | **0.9394** | +0.030 ✓ |
| RAFT-warp ↓ | **0.0474** | 0.0527 | +0.005 ✗ | | 0.0626 | **0.0525** | −0.010 ✓ |
| NVS-SQA ↑ | −2.9400 | **−2.9281** | +0.012 ✓ | | −2.9464 | **−2.9307** | +0.016 ✓ |
| flicker ↑ | 0.9948 | 0.9950 | +0.0002 ≈ | | 0.9945 | **0.9951** | +0.0006 ✓ |
| **tally (af3d better)** | | | **6 / 2✗ / 1≈** | | | | **9 / 0** |

(TSED/RAFT computed on the harness's centred 34-frame moving clip = name-sorted
views 146–157, which for this scene lands INSIDE the wedge → a wedge-interior
trajectory. flicker on the near-static micro-jitter clip: both ≈0.995 = low
discrimination here. n_pairs=33; low-N, deltas modest.)

## Verdict

**Does ArtiFixer3D's wedge fill beat base on the product axes with masked metrics
not-worse? PARTIALLY — a qualified win.**
- Masked metrics: **not-worse (better)** ✓.
- Cleanliness / plausibility (CrossScore, NOVA, NVS-SQA): **all better** ✓.
- Multi-view consistency (TSED, RAFT): **both slightly worse** ✗.

So on "clean" the wedge fill genuinely improves; on "consistent" it slightly
regresses. Against the product bar ("clean **AND** consistent") it is a **marginal
net-positive, not a clean win** — appearance up, consistency fractionally down.

### Was the June verdict right/wrong/partially right? — PARTIALLY RIGHT, with one premise corrected.
1. **Mechanism CONFIRMED and sharpened.** June: "bidirectional is weak exactly at
   gaps (+4.0% wedge vs −10.8% easy)." The product suite shows the SAME collapse
   across every axis: **easy = 9/9 metrics improve; wedge = 6/9, and the two that
   regress are precisely the multi-view-consistency ones (TSED/RAFT).** That is the
   textbook bidirectional-vs-causal signature — per-frame enhancement injects
   view-inconsistent detail that 3D distillation only partly fuses out. Where the
   base is well-observed (easy), the same enhancer IMPROVES consistency (TSED
   +0.030, RAFT −0.010); in the under-observed wedge it trades a little
   consistency for a little cleanliness. The investigation's core conclusion —
   "ArtiFixer3D sharpens well-but-imperfectly-observed views; it is NOT a
   coverage-gap filler in bidirectional mode" — is upheld by the product metrics.
2. **June's "failure" framing was TOO HARSH.** Under product metrics the wedge is
   not a wash: masked LPIPS −3.6%, CrossScore +0.023, NOVA −0.013, NVS-SQA +0.012
   are all real cleanliness/plausibility gains. "Plausible-not-faithful failure"
   overstates it — it's a modest product improvement offset by a small consistency
   cost.
3. **The re-judgment's OWN premise was partly wrong (biggest correction).** The
   program doc (§0 Reframe-1) assumed the wedge "+4.0% only" was scored "against
   photons we never captured → GT metrics invalid." **The visibility split refutes
   that for this experiment: the wedge is 100.0% observed.** It is an *angular
   extrapolation* gap, not a coverage HOLE — the laptop + scene geometry is
   globally reconstructed from other angles, so nothing renders empty (alpha≥0.3
   everywhere) and masked metrics ≈ full-frame. **The June held-out-GT LPIPS was
   therefore a LEGITIMATE measurement**, and the product metrics broadly agree with
   it rather than overturning it. The wedge was mislabelled as a "coverage gap";
   it never tested generative hole-filling at all.

## Implication for WP-C1 (completion lane)
**Bidirectional-ArtiFixer3D is NOT shippable as a gap/hole filler as-is; the
causal/generative completion path remains necessary.**
- This experiment does not even demonstrate hole-filling: the "gap" is 100%
  observed. For a TRUE hole (unobserved back / genuine void, observed%<100) there
  is no present geometry to sharpen, so the enhancer has nothing to work with —
  the causal ArtiFixer (Tier-1 H100 probe), GENA3D, and G4Splat-pattern
  visibility-gated inpainting (§4) stay on the critical path.
- Even on this mild angular gap the bidirectional path REGRESSES multi-view
  consistency (TSED/RAFT). This is the concrete argument for the universal
  opacity-mask gate (§2, GaMO): gate enhancement to unobserved pixels so it can
  never inject the consistency regression onto observed evidence.
- **Adopt** ArtiFixer3D-bidirectional for its validated job — **sharpen
  well-observed views** (easy = 9/9 win) — and NOT as the completion solution.
- **Harness gap:** to actually judge hole-filling, WP-C1 needs a test with
  observed%<100 in the held-out region (back-of-object holdout / slab-deletion
  void). The Laptop wedge cannot provide it.

## Caveats / honesty
- 100% observed ⇒ masked metrics ≈ full-frame; this is NOT a generative-hole
  stress test.
- Temporal clip is wedge-interior (not boundary-crossing); flicker near-static
  (both ≈0.995). TSED/RAFT are low-N (33 pairs) with small absolute deltas — the
  consistency regression is real but modest.
- CrossScore reference pool = evenly-spaced real train photos from the every-8th
  `train_indices` (load_colmap_data's default split, not the 3DGRUT 264-anchor
  training set); a few wedge-region reals may appear as references. This only
  strengthens the cross-view evidence near the wedge; it does not invalidate the
  render-vs-real comparison.
- Within-scene-only metrics (NVS-SQA JOD, flicker) used only for base↔af3d deltas
  within each case (valid).

## core/qa findings (CONSUMER report — no code changed)
- **No bug in `core/qa/` scoring logic.** Ran cleanly end-to-end as a library on
  foreign (3DGRUT) PLYs; all 6 scorers produced results first try.
- **Footgun in the `qa_report.py` DRIVER (not core/qa):** it hardcodes
  `load_colmap_data(normalize=True)`. On a foreign PLY in the raw COLMAP frame that
  silently renders garbage (7.8 dB) yet still reports plausible-looking 100%-
  observed metrics — a SILENT wrong-answer, no error. Worked around by a dedicated
  driver exposing `--normalize`. Recommend the harness owner surface a
  `--normalize` flag (or auto-detect frame) if qa_report is ever pointed at
  non-project checkpoints.
