# WP-Q1 phase-1 gate record (2026-07-02)

Produced by `qa_report.py` (core/qa/ harness, components 1+8). All runs factor-2,
split every8 unless noted. Determinism verified: identical `scorecard.json`
sha256 across repeat runs.

| run | corruption | obs% | mPSNR | mSSIM | mLPIPS |
|---|---|---|---|---|---|
| GPU_V4 (ckpt_29999, 2M GS) | none | 100.0 | 25.278 | 0.8514 | 0.4125 |
| GPU_V4_floaters | 2000 floaters @ scale 0.01 | 100.0 | 12.487 | 0.6209 | 0.7573 |
| GPU_V4_floaters_mild | 2000 floaters @ scale 0.003 | 100.0 | 20.309 | 0.7874 | 0.4851 |
| GPU_V4_blur | render blur sigma=2 | 100.0 | 25.724 | 0.8559 | 0.4319 |
| Laptop (ckpt_29999, 3M GS) | none | 100.0 | 22.319 | 0.8120 | 0.3540 |
| Laptop_wedge (views 131–169) | none | 100.0 | 23.618 | 0.8427 | 0.3491 |

**Blur-corruption note (expected, verified — not a harness failure):** for
blur-type corruption, **LPIPS is the judging axis**. PSNR/SSIM can move slightly
POSITIVE under blur by construction (blur suppresses high-frequency residual
error), which is exactly why the plan mandates judging perceptual work by
LPIPS/region metrics, never PSNR alone.

100% observed on both production scenes is correct (dense captures). Mask
flagging validated by slab-deletion: removing the z>=p55 Gaussian slab drops a
mid-trajectory view to 45.2% observed with the removed region tinted red
(`_flagdemo_hole.png`), and the full masked-metrics path runs on that partial
mask (LPIPS bbox 0.4802 / tiled 0.2970, PSNR 23.242, SSIM 0.8308).
Sliver-mask robustness (thin bboxes, sub-64px blobs, frames smaller than the
LPIPS crop) covered by expansion-within-bounds + replicate padding.

---

# WP-Q1 phase-2 gate record — external scorers (2026-07-02)

Six plug-in scorers now fill the `scorers` slot of every scorecard. Enable with
`qa_report.py ... --scorers all` (default OFF → phase-1 behavior unchanged). A
targeted re-run (e.g. `--scorers nvs_sqa`) merges into the existing scorecard
without wiping the others — but ONLY if the prior scorecard's `corruption` dict
matches the current run's exactly; on mismatch the merge is refused (fresh
scorecard, reason printed).

> **Review-hardened 2026-07-02** (all fixes re-validated, see "review fixes"
> section below): staged renders are ALWAYS regenerated (the out_dir-keyed disk
> cache that could serve pristine renders to a corrupted re-run is gone);
> corruption-checked merge; percentile-robust flicker guard; `--floater_jitter`
> temporal positive control added and banked (`qa/GPU_V4_jitter/`).

**Pose frame (`--normalize {auto,true,false}`, default true):** our own trainer
checkpoints live in the NORMALIZED frame (default = unchanged behavior — all
banked scorecards); foreign PLYs (e.g. 3DGRUT exports) live in the raw COLMAP
frame and need `--normalize false` (`auto` probes the first held-out view under
both frames and keeps the higher-PSNR one). An ALWAYS-ON degenerate-render check
self-flags wrong-frame runs: first-view render-vs-GT PSNR < 9 dB or near-black →
loud warning + `render_sanity: "suspect"` in the scorecard (thresholds
EMPIRICAL: the 3DGRUT wedge PLY under the wrong frame renders a BRIGHT garbage
blob at 7.7–7.8 dB / lum 0.45 — a black-frame check alone misses it — while the
heaviest banked intentional corruption, floaters@0.01, stays at 12.5 dB).
Verified 2026-07-02: GPU_V4 re-run byte-identical to the bank apart from the two
new fields (`qa/_t_normcheck/`); foreign 3DGRUT PLY sane at 22.76 dB under
`false` / flagged suspect at 7.75 dB under `true`; `auto` resolves to `false`
(`qa/_t_foreignply_{false,true,auto}/`).

## scorer → environment mapping
| scorer | env | how | model | direction |
|---|---|---|---|---|
| tsed | **main** (torch 2.11) | in-process, vendored SIFT+epipolar | opencv SIFT | ratio ↑ = better |
| raft_warp | **main** | in-process | torchvision `raft_large` | warp-err ↓ = better |
| flicker | **main** | in-process, vendored VBench | none (numpy) | score ↑ = less flicker |
| crossscore | **qa venv** | subprocess → Hydra CLI | CrossScore + DINOv2 | pred-SSIM ↑ = better |
| nova | **qa venv** | subprocess → driver | NOVA (timm DINOv2) | cos-dist ↓ = better |
| nvs_sqa | **qa venv** | subprocess → driver | NVS-SQA seq encoder | JOD ↑ = better (within-scene) |

qa venv = `D:\REPOS\HYPERclone\venvs\qa` (`--system-site-packages`; inherits the
working torch 2.11/cu128). Subprocess scorers shell out to it; the whole scorecard
still runs from one `qa_report.py` call. tsed/raft/flicker consume a rendered
SMOOTH MOVING trajectory (SLERP through a contiguous run of real cameras); flicker
consumes a NEAR-STATIC micro-jitter clip and refuses genuinely-moving clips (guard:
max inter-frame center motion / **percentile-robust scene scale** ≥ 0.02; robust =
1–99 pct bbox diagonal of camera centers, same pattern as floater placement — a
raw max radius would let one outlier camera quietly loosen the guard). Renders come
from the (possibly-corrupted) splats; blur applies to scorer renders too. External
clone commits + the two CrossScore patch files are pinned in `THIRD_PARTY.md` /
`tools/patches/`.

## gate table — baseline bank (all 6 scorers × 5 runs)
| run | corruption | crossscore↑ | nova↓ | tsed↑ | raft_warp↓ | nvs_sqa↑ | flicker↑ | mPSNR | mSSIM | mLPIPS |
|---|---|---|---|---|---|---|---|---|---|---|
| GPU_V4 | none | **0.9116** | **0.0184** | **1.000** | **0.0146** | **−2.370** | **0.9887** | 25.278 | 0.8514 | 0.4125 |
| Laptop | none | **0.8250** | **0.1321** | **0.9091** | **0.0417** | **−2.867** | **0.9942** | 22.319 | 0.8120 | 0.3540 |
| GPU_V4_floaters | 2000 @0.01 static | 0.8253 | 0.3129 | 0.9697 | 0.0120 | −3.061 | 0.9965 | 12.487 | 0.6209 | 0.7573 |
| GPU_V4_blur | σ=2 | 0.9121 | 0.0330 | 1.000 | 0.0120 | −2.459 | 0.9921 | 25.724 | 0.8559 | 0.4319 |
| GPU_V4_jitter | 2000 @0.01 **jittered/frame** | 0.8253 | 0.3129 | **0.6970** | **0.1625** | −3.043 | **0.9399** | 12.487 | 0.6209 | 0.7573 |

Bold = the two production baselines (bank these; all gates are per-scene deltas vs
this bank) and the jitter row's temporal-metric hits. NVS-SQA JOD + flicker are
WITHIN-scene only — do NOT compare GPU_V4 vs Laptop on those two. The baseline
values were re-verified byte-identical AFTER the review fixes (the fixes changed
no clean-run numerics). Jitter's crossscore/nova/phase-1 equal the static-floaters
row by design: held-out view renders keep the STATIC injection; only the temporal
clips jitter.

## runtime (s per scorer per scene, RTX PRO 6000, post-review always-fresh staging)
| scene | crossscore | nova | tsed | raft_warp | nvs_sqa | flicker |
|---|---|---|---|---|---|---|
| GPU_V4 (72 views) | 147.6 | 50.5 | 3.1 | 2.2 | 5.5 | 0.5 |
| Laptop (38 views) | 69.9 | 9.2 | 1.6 | 1.0 | 3.7 | 0.2 |
| GPU_V4_jitter | 156.5 | 55.8 | 37.3 | 2.4 | 6.4 | 16.5 |

CrossScore dominates (72 full-res fresh renders + DINOv2 + Lightning model load
per call; the first scorer that stages renders pays that cost, later scorers in
the same run reuse them via the in-memory memo). Always-fresh staging costs
~+60 s/scene vs the old (unsound) disk cache — the price of correctness. Jitter
adds ~1 s/clip-frame (per-frame floater re-injection does a CPU quantile over 2M
means). A full 6-scorer GPU_V4 scorecard ≈ 3.5 min end-to-end.

## determinism — PERFECT (GPU_V4 run twice → identical), re-verified post-fixes
All six scorers reproduce **identically as FULL result dicts** (every per-view /
per-pair value, not just headline scalars; only `runtime_s` and embedded output
paths are excluded), and phase-1 masked metrics are float-equal. No irreducible
nondeterminism to document: gsplat renders are deterministic, SIFT/epipolar and
the VBench MAE are pure numpy, and the model forwards (CrossScore
`precision=32-true` + seed 1 + `neighbour_config.deterministic`, NOVA/NVS-SQA
fixed-weight eval) are stable on this stack.

## corruption direction (baseline → corrupted; delta interpretation)
**Floaters (2000 @0.01)** — worsen the appearance/geometry metrics correctly:
crossscore 0.912→0.825 (−0.086), nova 0.018→0.313 (+0.29, strong), tsed 1.00→0.970
(one pair flips inconsistent), nvs_sqa −2.37→−3.06 (−0.69). **RAFT (0.0146→0.0120)
and flicker (0.989→0.997) do NOT worsen** — and that is correct behavior, not a
miss: the injected floaters are large *soft, semi-transparent* Gaussians that veil
the render with LOW-frequency haze rather than high-frequency popping; being rigid
3D they also warp smoothly across the moving clip. RAFT/flicker are tuned to
HIGH-frequency temporal popping (view-dependent needles/aliasing — WP-Q2's 89%
needle finding), which soft floaters don't produce. So the gate's "floaters →
consistency/NR worsen" is met by TSED (geometry-consistency) + crossscore/nova/
nvs_sqa (NR); RAFT/flicker stay flat by design.

**Blur (σ=2)** — worsens the perceptual/NR metrics that should catch it: nova
0.018→0.033 (+0.015), nvs_sqa −2.37→−2.46 (−0.09), and phase-1 masked LPIPS
0.4125→0.4319 (the designated blur axis). crossscore is ~flat (+0.0005): it resizes
the short side to 518, so a σ=2 blur at factor-2 res is largely washed out by that
downsample. TSED/RAFT/flicker are (correctly) blur-invariant — blur moves no
geometry and reduces temporal residual (RAFT 0.0146→0.0120, flicker 0.989→0.992),
consistent with the phase-1 rule that blur is judged by LPIPS/perceptual, never by
geometry/temporal or PSNR.

**Jitter (2000 floaters @0.01, positions re-randomized every clip frame,
`--floater_jitter`) — the TEMPORAL POSITIVE CONTROL.** Static floaters and blur
are both LOW-PASS corruptions (soft haze / smoothing) that a temporal-popping
detector *should* ignore; jitter creates genuine frame-to-frame popping and both
temporal metrics fire hard: **raft_warp 0.0146→0.1625 (11× worse**, valid-pixel
fraction collapses 0.98→0.59 as forward-backward flow consistency breaks) and
**flicker 0.9887→0.9399** (mean consecutive-frame MAE 2.9→15.3) — while the
flicker static-camera guard still passes (camera motion unchanged at
max_motion_frac 0.0007; only splat CONTENT jitters). TSED also craters
1.00→0.697 with SIFT matches 451→87/pair (popping blobs violate epipolar
geometry outright). So every scorer now has a demonstrated true positive:
crossscore/nova/nvs_sqa/tsed (static floaters), nova/nvs_sqa/LPIPS (blur),
raft_warp/flicker/tsed (jitter). Banked at `qa/GPU_V4_jitter/`.

## review fixes (2026-07-02, all re-validated)
1. **HIGH — stale disk-render cache REMOVED** (`core/qa/scorers/common.py`): the
   `if not p.exists()` guards keyed staged renders only by out_dir+frame — a
   corrupted re-run of an out_dir could consume PRISTINE renders
   (reviewer-reproduced: nova returned 0.004719 instead of 0.280123). Staging
   dirs are now cleared and re-rendered EVERY run; scorers within one run share
   work via an in-memory (per-process) memo that cannot go stale.
   *Re-validated:* same-out_dir corrupted re-run now yields nova 0.280123 ==
   fresh-dir control (scorecards: `qa/_t_stale/`, `qa/_t_stale_fresh/`).
2. **HIGH — corruption-checked merge** (`qa_report.py`): prior scorer values are
   merged only when the prior scorecard's `corruption` dict equals the current
   run's; otherwise the merge is refused with a printed reason.
3. **MEDIUM — clone reproducibility**: commit hashes of the three qa_repos
   clones pinned in THIRD_PARTY.md; the two CrossScore working-tree fixes
   exported as `tools/patches/crossscore_mkdir.patch` +
   `crossscore_winpath.patch` (apply line in THIRD_PARTY.md); the extracted
   NVS-SQA linear head vendored INTO apex-sfm
   (`core/qa/vendor/nvs_sqa_linear_head.npz`).
4. **MEDIUM — robust flicker guard**: max_motion_frac now normalizes by the 1–99
   percentile bbox diagonal of camera centers, not the outlier-sensitive max
   radius.
5. **MEDIUM — `--floater_jitter` positive control** (row + analysis above).
6. **LOW**: `run_scorers` in scorecard.py is now THE scorer loop (qa_report calls
   it; no diverging inline copy); blur/floater math deduped into
   `core/qa/corruptions.py` (one copy, two callers); CrossScore CSV reader now
   errors informatively if the `pred_*` column is missing.

**Verdict: gate PASSED (and re-validated post-fixes)** — every scorer runs
end-to-end on all 5 runs, all six reproduce identically as full dicts, and each
corruption is caught in the correct direction by the metric family that should
catch it (floaters → consistency+NR; blur → perceptual/NR; temporal popping →
raft_warp/flicker/tsed).
