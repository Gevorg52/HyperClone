# WP-F4 — Planar-prior losses (PlanarGS port) RECORD

Status: **GATE RAN 2026-07-04 (poller auto-fired at 12:37 when WP-S1 freed the GPU;
pipeline exit 0 at 14:39). VERDICT: FAILED as-specced (PlanarGS lambdas) — geometry
FLATNESS improves substantially but appearance degrades badly; DIAGNOSED as L_rn
over-weighting (lambda does not transfer to our normalization). Recalibration A/B is
the recommended next step (not run — GPU handed to WP-S2 per coordination).** Mechanism
checks (GPU, one loss at a time) PASSED. Full argv/evals in `_infusion/wpf4_gpu_all.sh`;
raw evidence in `qa/wpf4_ab_{off,on}/scorecard.json`, `_infusion/wpf4_ab_flatness.json`,
`_infusion/wpf4_ab_floaters.txt`, `_dataset/.../wpf4_ab/{off,on}.log`.

## 1. Mask-source decision (RECON-driven)

Plan's primary = SAM 2.1 plane instances; documented fallback = MoGe-2 normal cache.
**Chose the geometric MoGe route.** Reasons (recon 2026-07-03):
- Production SAM *plane* instances were never preserved (DEVELOPER_NOTES); the only
  SAM assets are the foreground object-vs-bg masks (not plane instances), and even
  those aren't on disk for Laptop/GPU_V4.
- The sam2 clone + `sam2.1_hiera_large.pt` (857 MB) IS runnable (`gui/stages/masking.py`),
  but running SAM-2.1 AMG fresh on 304+569 full-res views is GPU-heavy and would
  contend with WP-S1's training; its class-agnostic segments would STILL need the
  same geometric plane inspection.
- The geometric route reads only the already-cached MoGe-2 `normal/`+`depth/` maps,
  is CPU-only (sklearn K-means + scipy CC + numpy plane fit), deterministic, and
  produces plane instances directly. Simpler-but-slightly-slower, per §0.

`core/plane_masks.py` recipe (PlanarGS geometric inspection): reconstruct RAW MoGe
depth (undo per-view scale+shift so the shift can't bend a plane) → K-means k=5 on
normals (fit on 20k subsample, assign all) → 8-conn connected components → per CC
regularized LS plane fit `(PPᵀ+0.01I)⁻¹P·1`, relative-residual prune <0.05, refit,
largest-CC, min 2000 px AND ≥0.5% frame → unique uint16 instance labels (0=bg).

Laptop build (`priors/plane_mask/`, 304 views): **304/304 covered, 8.57 planes/view,
median 58.7% of frame labeled, 501 s CPU.** Visual check (frame_0001 overlay): the
wooden tabletop (the "deck"/flatness target) = one large plane, the textured laptop
keyboard/screen correctly EXCLUDED, background ground/plants = benign planar regions.
Shared priors dir → `wps1/{stack,ctrl}/data/priors/plane_mask` are byte-identical.

## 2. Loss diffs (what shipped, all default OFF)

`core/training/gsplat_trainer.py`:
- `co_planar_depth(depth, plane_mask, inv_K)` + `normal_from_depth(depth, inv_K)` —
  ported from PlanarGS `planar/co_planar.py` + `common_utils/graphics_utils.py`.
- **L_p** (`--planar_lambda`, `--planar_start` 14k, `--planar_detach` default off):
  per-instance plane fit on rendered depth, masked-L1 pull toward the plane,
  median-normalized. **Detach decision:** PlanarGS does NOT detach (`co_planar` runs
  on `render_depth` with grad; no `.detach()` in `train.py`) — we match that
  (attached) by default; `--planar_detach` exposes the alternative.
- **L_rd** (`--prior_depth_lambda`, `--prior_depth_start` 7k): rendered depth vs the
  aligned MoGe cache, per-pixel weight = validity × conf × inverse-Canny edge, whole
  term × per-view depth_weight; relative-squared (median-normalized L2, per PlanarGS
  l2_loss). Complementary to WP-F1 `--mono_depth_lambda` (adds the conf/edge gating).
- **L_rn** (`--prior_normal_lambda`, `--prior_normal_start` 20k): depth-derived normal
  (no Gaussian-normal rasterization — L_dn skipped) vs MoGe cache normal, L1 + (1−cos),
  masked to planar ∩ conf ∩ edge. MoGe normals point toward camera → match pcd2normal
  → no sign flip.
- Per-loss EMA logging (`Lp=`, `Lrd=`, `Lrn=` in the step line); setup block hard-
  errors on missing cache / zero plane-mask coverage (mirrors mono-depth guard);
  losses are fine-phase-only (start raised to coarse_until+1).

`core/training/gsplat_backend.py`: the 6 flags + `--planar_detach` added to
`get_default_config()` (defaults 0/off) and the subprocess argv.

SKIPPED per plan: **L_dn** (needs a rasterized Gaussian-normal map — our RGB+ED
renderer has none) and PlanarGS **`lambda_scale=100`** min-scale flatten (erank hinge
covers anti-needle; aggressive flatten biases MCMC relocation).

## 3. Mechanism checks

**CPU math check — PASSED** (`_infusion/wpf4_mechanism_cpu.{py,txt}`, GPU-free):
- T1 perfect-plane L_p rel-residual = 1.09e-07 (plane fit exact).
- T2 bumped-depth L_p = 0.0725 > 0; gradients finite + nonzero.
- T3 depth-normal vs true-plane |cos| = 1.0000; 100% point toward camera (matches MoGe).
- T4 L_rn = 0.1007 > 0; finite, grads flow, no NaN. ALL OUTPUTS FINITE.

**GPU per-loss checks (one loss at a time, 600 steps) — PASSED** (`wpf4_gpu_all.sh`
§1; MECHANISM GATE PASSED, all fired, finite, nonzero, no nan/inf):
- L_rd: last EMA `Lrd=0.2160`
- L_p:  last EMA `Lp=0.0236`
- L_rn: last EMA `Lrn=0.5731`

## 3b. Post-review fixes (reviewer pass, 2026-07-03 — applied before the A/B)

Static review reproduced the CPU mechanism math independently and found:
1. **(HIGH, fixed)** `co_planar_depth` degenerate-denominator guard used a fixed
   `+1e-6` → grazing rays (paramᵀr→0) mapped to plane_depth ~1e6, a *finite*
   spurious target the `isfinite` guard misses. Now **sign-preserving** (magnitude
   ~1, matches `plane_masks._plane_depth_along`). New regression test
   `tests/test_planar_losses.py::test_coplanar_grazing_denominator_bounded`.
2. **(MED, fixed)** Border asymmetry — PlanarGS zeros the 8-px border in-place so
   its loss mask drops it; our clone kept those pixels in the L_p/L_rn denominator
   (slight under-weight). Now we zero the border of the loss mask too (consistent
   numerator/denominator).
3. **(LOW-MED, fixed)** Added a conf/edge cache hard-error for the L_rd/L_rn path
   (previously only plane_mask coverage was guarded → a missing conf/edge cache
   would silently no-op L_rd).
4. **(LOW, doc)** `--mono_depth_use_weight` reuse for L_rd now commented as intentional.
5. **(LOW, doc)** Mask-sum (coverage-invariant) vs PlanarGS full-frame normalization
   documented in THIRD_PARTY.md (lambda values are re-interpreted, not transferred).
6. **(LOW, fixed)** Added `tests/test_planar_losses.py` (7 CPU tests) — the math is
   now in the pytest suite (**146 passed** after the fixes; was 139).

CPU mechanism check re-run after fixes: still **OVERALL PASS** (T1 1.09e-07, T3
cos=1.0, T4 L_rn=0.1007, no NaN).

## 4. A/B (15k, seed 0, cap 1M, factor 2) — RAN 2026-07-04

Base (both arms): dense_init (707k) + mono_depth — `wps1/stack/data`.
- OFF argv: `--mono_depth_lambda 0.05 --mono_depth_start 2000` (+ COMMON below).
- ON argv:  OFF + `--prior_depth_lambda 0.05 --prior_depth_start 3500
  --planar_lambda 0.5 --planar_start 7000 --prior_normal_lambda 0.2
  --prior_normal_start 10000` (PlanarGS 7k/14k/20k scaled to the 15k budget).
- COMMON: `--data_dir wps1/stack/data --data_factor 2 --max_steps 15000 --test_every 8
  --strategy mcmc --cap_max 1000000 --opacity_reg 0.01 --scale_reg 0.01 --antialiased
  --seed 0 --save_steps 15000`.

38 held-out (every-8) views. qa_report LPIPS = tiled (256/0.8). Flatness/region LPIPS
= VGG (torchmetrics). deck box (0.30–0.72 x, 0.55–0.95 y); tex box (0.30–0.72 x,
0.18–0.50 y).

| metric | OFF (base) | ON (planar pkg) | delta | gate leg |
|---|---|---|---|---|
| **deck plane RMS (flatness, ↓ better)** | 0.02068 | **0.01678** | **−18.9%** | improves ✓ **PASS** |
| all-labeled-planes RMS (↓ better) | 0.01951 | **0.01130** | **−42.1%** | (geometry) ✓ |
| **deck-region LPIPS** (VGG, ↓) | 0.2948 | 0.3280 | **+11.3%** | not worse ✗ **FAIL** |
| textured-box LPIPS (VGG, ↓) | 0.2821 | 0.3316 | **+17.6%** | untouched ✗ **FAIL** |
| full-frame LPIPS (VGG, ↓) | 0.3419 | 0.3859 | +12.9% | (context) |
| **global masked-LPIPS** (tiled, ↓) | 0.2572 | 0.2966 | **+15.3%** | not worse ✗ **FAIL** |
| masked PSNR (↑) | 23.231 | 21.865 | −1.37 dB | worse |
| masked SSIM (↑) | 0.8428 | 0.8284 | −0.014 | worse |
| floater needle % (↓) | 82.11 | 79.84 | −2.3pp | cleaner ✓ |
| floater giant % (↓) | 9.84 | 11.48 | +1.6pp | worse ✗ |
| floater low-opa % (↓) | 7.0 | 10.9 | +3.9pp | worse ✗ |
| mean-erank (↑=less needly) | 1.615 | 1.672 | +0.06 | cleaner ✓ |

**GATE VERDICT: FAILED.** Gate required deck flatness improves AND deck LPIPS not
worse AND global masked-LPIPS not worse AND textured untouched. Only the flatness leg
passed; every appearance leg regressed and MCMC gained giants/low-opacity floaters.

### Root-cause diagnosis (loss curves, `wpf4_ab/on.log`) — NOT a fundamental failure
The lever moves geometry exactly as intended (deck RMS −18.9%, all-planes −42.1%, and
the erank/needle indicators even improve), but the **PlanarGS lambda VALUES do not
transfer to our mask-sum + median normalization** (the review's finding #5, now
confirmed empirically). The dominant culprit is **L_rn**: its EMA term is 0.34–0.72,
so at λ0.2 it contributes ~0.07–0.14 to the loss — **larger than the entire
photometric loss (~0.09)**. The ON-arm total loss **triples (0.097→0.261) and PSNR
crashes (24.2→21.7) the instant L_rn activates at step 10k**; the OFF arm is stable
throughout (loss ~0.06, eval PSNR 23.23). L_rd (EMA ~0.4 × λ0.05 ≈ 0.02) and L_p
(~0.035 × λ0.5 ≈ 0.017) are moderate and activated (3.5k/7k) without a catastrophic
jump. So the normal prior over-drives the optimization toward the MoGe normal field
at a large appearance cost — the classic geometry↔appearance trade (DEVELOPER_NOTES
§6) amplified by a ~5–10× too-strong effective λ_rn.

### Recommended next step (NOT run — GPU handed to WP-S2 per coordination)
Recalibrate for our normalization and re-run the same A/B: **λ_rn ≈ 0.02 (÷10),
λ_rd ≈ 0.02 (÷2.5), keep λ_p ≈ 0.5** — target each per-loss contribution ≲0.02
(≈20–30% of the base loss), then confirm flatness still improves at appearance-neutral
LPIPS. Also worth: gate the plane masks to foreground/object planes only (background
plants currently get labeled), and/or a shorter L_rn warm-in. The lever is mechanically
sound and the flatness payload is real; it needs λ tuning before it is usable.

## 5. Persisted artifacts (committable)
- `core/plane_masks.py`, `core/training/gsplat_trainer.py`, `core/training/gsplat_backend.py`.
- `_infusion/wpf4_mechanism_cpu.{py,txt}` (CPU mechanism evidence).
- `_infusion/wpf4_gpu_all.sh` (mechanism + A/B + eval runner, exact argv).
- `_infusion/wpf4_flatness_eval.py` (deck-flatness + region-LPIPS eval).
- `qa/wpf4_ab_{off,on}/scorecard.json` (global masked metrics, tiled LPIPS, 38 views).
- `_infusion/wpf4_ab_flatness.json` (deck/all-plane RMS + region VGG-LPIPS + deltas).
- `_infusion/wpf4_ab_floaters.txt` (floater_report both arms).
- `_infusion/wpf4_gpu_all.log` (full pipeline: mechanism gate + both arms + evals).
- `_dataset/.../wpf4_ab/{off,on}.log` (per-step loss curves — the diagnosis source).

Note: `_dataset/` is gitignored — the checkpoints are recomputable from the argv
above; all gate NUMBERS land in the committable artifacts listed here.
