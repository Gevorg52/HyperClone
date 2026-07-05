# WP-N1 — Native-resolution isolation experiment — RECORD (run 2026-07-04/05)

**Scene: GPU_V4 (24 MP source, the June native-res FAILURE scene). Verdict: GATE NOT
PASSED — but the June failure is DEFINITIVELY root-caused and half-solved.** The naive
June native run REGRESSED ~4 dB with a ~6 dB train/test gap; BLACKWELL_FRONTIER §1.1
hypothesised pose_opt overfitting train poses at 4× sharper photometric gradient. This
experiment isolates it: **N-b (pose_opt ON) reproduces the June failure exactly
(test 20.76 dB, 6.18 dB train/test gap); N-a (pose_opt OFF) eliminates the gap
(0.05 dB) and turns the −4 dB regression into a +3.5 dB PSNR / +0.04 SSIM WIN over the
factor-2 F2 stack, and wins ALL native-res microstructure crops.** BUT N-a LOSES the
gate's decisive metric — common-res (factor-2) masked-LPIPS (0.432 vs F2 0.410) — because
the perceptual advantage is **resolution-matched**: N-a is best viewed at native res,
F2 at factor-2. The gate (beat F2 on common-res LPIPS **AND** microstructure) fails on
the AND. **factor-2 stays the production recipe for factor-2 perceptual delivery;
native-res + pose_opt-OFF (N-a) is validated for NATIVE 24 MP detail delivery only.**

---

## Arms (GPU_V4, 72 held-out every-8 views, 3 M cap, seed 0)
| arm | resolution | pose_opt | source |
|---|---|---|---|
| **F2** (baseline) | factor-2 (6 MP) | ON (lr 1e-5) | banked WP-S1 GPU **stack** ckpt — REUSED, not retrained |
| **N-a** | factor-1 (24 MP), c2f 2→1 @15k | **OFF** | trained this WP |
| **N-b** | factor-1 (24 MP), c2f 2→1 @15k | ON (lr 1e-5) | trained this WP |

Both N arms = the WP-S1 stacked recipe (dense-init 882,902 + mono-depth 0.05 +
erank 0.01@7k + gated-LPIPS 0.2) at `--data_factor 1 --coarse_factor 2 --coarse_until
15000`, 30k, seed 0. Exact argv: `_infusion/wpn1_train.sh`. Startup both arms:
`Mono-depth loss ON (start=15001, mode=per-view, 569/569 reliable, 497/497 covered)`.

### Prior-cache decision (the gate that gated everything)
The trainer HARD-ERRORS when `cache_factor != data_factor` (gsplat_trainer.py:1209-1215).
WP-S1 priors are factor-2 → a factor-1 run needs a factor-1 cache. A **full** factor-1
MoGe cache is infeasible here: ~256 GB disk (depth+normal+conf) and the builder retains
per-view normal+image tensors → ~400 GB RAM OOM (system has 203 GB). WP-N1 uses only
`--mono_depth_lambda` (no planar losses), and the mono-depth loss reads ONLY `depth/` +
`align.json` (load path guards `load_planar_priors`, which is off). So I added an
additive, default-OFF **`--depth_only`** flag to `core/priors.py` (drops the per-view
normal/img retention + skips the normal/conf/edge_mask writes). Factor-1 depth-only
cache: **51 GB disk, ~68 GB RAM peak, 569/569 views reliable, mode=per-view,
global_scale 1.3095** (≈ the factor-2 cache's 1.3142 — the only difference is
resolution). Built against the ORIGINAL ALIKED sparse (123,289 pts) exactly as the
factor-2 WP-S1 priors were (normalize is pose-based → point-independent, and the loss is
scale-invariant, so this mirrors the F2 baseline's prior provenance). Coverage verified:
497/497 train views, depth shape (3999,5999) == factor-1 undistorted frame.
Cache: `_dataset/GPU_V4/.../wps1/n1/priorsrc/priors` (junctioned into `n1/data/priors`).

### VRAM / runtime (RTX PRO 6000, 96 GB)
| arm | peak VRAM | runtime | note |
|---|---|---|---|
| N-a | **27,803 MB** | ~11,988 s (~3.3 h) | fine phase (24 MP) did NOT exceed the coarse-phase MCMC-densification peak |
| N-b | **27,803 MB** | ~11,518 s (~3.2 h) | same peak |

Both heavily contended by two other agents' bursty jobs (core.delivery renders +
Laptop training) throughout — wall-clock is contention-inflated, not a native-res cost.
**Native 24 MP @ 3 M is only ~28 GB — huge 96 GB headroom; VRAM is NOT the limiter.**

---

## GATE TABLE 1 — COMMON-RES masked (factor-2), the decisive gate metric
Two independent common-res methods; they agree tightly.
- **A** = `qa_report --factor 2` (direct factor-2 render; the exact validated harness that
  produced the banked F2 stack 0.4071). Scorecards: `qa/wpn1_gpu_{f2,na,nb}_f2/`.
- **B** = render@factor-1 → area-downsample to factor-2 GT shape → masked (the June
  "downsample everything" protocol; `_infusion/wpn1_gpu_masked.json`, `wpn1_eval.py`).

| arm | **masked-LPIPS (A)** | masked-LPIPS (B) | masked-PSNR (A) | masked-SSIM (A) |
|---|---|---|---|---|
| **F2** | **0.4071** | 0.4097 | 22.93 | 0.8249 |
| **N-a** | 0.4308 | 0.4322 | **26.40** | **0.8667** |
| **N-b** | 0.4621 | 0.4642 | 20.87 | 0.8068 |

→ On the **gate metric (common-res masked-LPIPS), N-a LOSES to F2** (+5.8% A / +5.5% B).
  N-a WINS common-res PSNR (**+3.47 dB**) and SSIM. N-b is worst on everything
  (reproduces June). F2's common-A (0.4071) reproduces the bank exactly → harness/data
  consistency confirmed.

## GATE TABLE 2 — NATIVE-RES (factor-1) masked + full-frame LPIPS (for the record)
`_infusion/wpn1_gpu_masked.json` (PSNR/SSIM) + `_infusion/wpn1_gpu_shiny_region.json` (full LPIPS).

| arm | native full-LPIPS | native PSNR | native SSIM |
|---|---|---|---|
| F2 | 0.5039 | 22.77 | 0.8177 |
| **N-a** | **0.4898** | **26.08** | **0.8451** |
| N-b | 0.5133 | 20.76 | 0.8047 |

→ At NATIVE res, **N-a WINS all three** (incl. full-frame LPIPS −2.8% vs F2). The LPIPS
  verdict FLIPS with eval resolution: N-a best at native, F2 best at factor-2 — each is
  best at its own training resolution.

## GATE TABLE 3 — TRAIN/TEST GAP (the June ~6 dB overfitting signature) — MECHANISM PROOF
`_infusion/wpn1_gpu_gap.json`. train = trainer training-loop PSNR (uses pose_opt-REFINED
train poses for pose_opt-ON arms), mean of steps ≥29000; test = trainer [Eval] held-out
72 views at ORIGINAL poses (factor 1).

| arm | pose_opt | train-loop PSNR (refined) | held-out test PSNR | **gap** |
|---|---|---|---|---|
| **N-a** | OFF | 26.13 | 26.08 | **0.05 dB** |
| **N-b** | ON | 26.94 | 20.76 | **6.18 dB** ← June signature reproduced |

→ **DEFINITIVE: pose_opt is the June mechanism.** N-b (pose_opt ON) fits the refined
  train poses to 26.94 dB but held-out test collapses to 20.76 dB = 6.18 dB gap =
  train-pose overfitting at 4× sharper 24 MP photometric gradient. N-a (pose_opt OFF)
  has ~0 gap → test 26.08 dB. **The stack alone did NOT fix the interaction (N-b still
  fails); pose_opt-OFF is required.** NOTE: `train_psnr_native` in wpn1_gpu_masked.json
  is CONFOUNDED for pose_opt-ON arms (it renders train views at ORIGINAL, un-refined
  poses → artificially low) — use this table's trainer-loop PSNR for the gap.

## GATE TABLE 4 — MICROSTRUCTURE region-LPIPS (native factor-1) + strips
`_infusion/wpn1_gpu_{shiny,fins48,fins24}_region.json`; strips
`qa/wpn1_gpu_strips_{shiny,fins48,fins24}/` (GT|F2|N-a|N-b full + 2× region crop).

| region (native f1) | F2 | **N-a** | N-b | N-a vs F2 |
|---|---|---|---|---|
| shiny (etched traces + ZOTAC logo + "LIVE TO GAME" microtext) | 0.4816 | **0.4580** | 0.5057 | **−4.9%** |
| fins48 (backplate fin grille, view48) | 0.4952 | **0.4814** | 0.4982 | **−2.8%** |
| fins24 (heatsink fin stack + trace corner, view24) | 0.4948 | **0.4770** | 0.5305 | **−3.6%** |

→ **N-a WINS all three microstructure regions** at native res. Visual (view24 strip):
  N-a's heatsink fin lines + etched backplate traces are **visibly crisper than F2**;
  N-b softest/misaligned. Modest, not dramatic — the finest fin mesh is still not fully
  resolved by any arm (partly a genuine capture/specular limit, not just resolution;
  cf. WP-S1's specular-residual diagnosis).

---

## GATE VERDICT
Gate: *an N arm beats F2 on common-res masked-LPIPS **AND** the microstructure crops*.
- Common-res masked-LPIPS: **N-a LOSES** (0.432 vs 0.410). ✗
- Microstructure crops: **N-a WINS** all three (native region-LPIPS −2.8…−4.9%). ✓
→ **GATE NOT PASSED** (the AND fails on common-res LPIPS).

## Which mechanism persists (per the task's decision rule)
The task: *"if both N arms underperform but train PSNR is high → poses; if train PSNR is
also low → under-convergence, consider the 60k extension."* Here **train-loop PSNR is
HIGH** (N-a 26.13, N-b 26.94) → it is **POSES, not under-convergence.** N-a already
converges to test 26.08 dB in 30k. **The 60k extension is NOT warranted** (under-
convergence ruled out) — recorded as an unnecessary follow-up; do not run it.

## PRODUCTION RECOMMENDATION
1. **factor-2 remains the production recipe for factor-2-resolution perceptual delivery**
   — F2 wins common-res masked-LPIPS (0.407 vs 0.432). Native-res is NOT a blanket win.
2. **pose_opt MUST be OFF for any native-res training** — it is the confirmed June failure
   mechanism (6.18 dB overfitting at 24 MP). The stacked levers do NOT compensate for it.
3. **For NATIVE 24 MP detail delivery, N-a's recipe is validated** (native factor-1,
   coarse-to-fine 2→1, dense-init + mono-depth + erank + gated-LPIPS, **pose_opt OFF**):
   it beats F2 on native PSNR/SSIM/LPIPS and ALL microstructure regions, with a clean
   0.05 dB train/test gap and only ~28 GB VRAM. The +3.5 dB common-res PSNR/SSIM win
   means N-a is also geometrically superior even at factor-2 — only the perceptual-LPIPS
   texture match at factor-2 favours the factor-2-native F2.
4. The BLACKWELL_FRONTIER §1.1 isolation item (a) *native without pose_opt* is now DONE
   and POSITIVE; the June regression was pose_opt, not resolution. Native-res is unlocked
   for detail work (pose_opt-OFF), not adopted as the factor-2 production default.

---

## Persisted artifacts (committable)
- `_infusion/wpn1_RECORD.md` (this file), `_infusion/wpn1_train.sh` (exact argv + seed),
  `_infusion/wpn1_eval.sh` / `wpn1_eval_stage23.sh` (eval drivers), `_infusion/wpn1_eval.py`
- `_infusion/wpn1_gpu_masked.json` (common-A-equiv B + native masked + gap-native[confounded note])
- `_infusion/wpn1_gpu_gap.json` (the CORRECT trainer-loop train/test gap = mechanism proof)
- `_infusion/wpn1_gpu_{shiny,fins48,fins24}_region.json` (native microstructure region-LPIPS + boxes)
- `qa/wpn1_gpu_{f2,na,nb}_f2/scorecard.json` (common-res-A masked scorecards; F2=0.4071 reproduces bank)
- `qa/wpn1_gpu_strips_{shiny,fins48,fins24}/` (GT|F2|N-a|N-b zoom strips, native factor-1)
- code: `core/priors.py` `--depth_only` flag (additive, default-OFF; used to build the factor-1 cache)
- gitignored but recomputable: ckpts `_dataset/GPU_V4/.../wps1/n1/{na,nb}/output/ckpts/ckpt_29999_rank0.pt`,
  factor-1 depth cache `.../wps1/n1/priorsrc/priors`, train logs `.../wps1/n1/logs/{na,nb}.log`

## Deviations / operational notes
- **`--depth_only` prior build** (new flag) — necessary to fit factor-1 priors in disk/RAM;
  builds ONLY what the mono-depth loss reads. No new third-party component (MoGe-2 already
  logged in THIRD_PARTY.md WP-F1).
- **Common-res eval done two ways** (qa_report direct-f2 A; render@f1→downsample B) — they
  agree to <0.003 LPIPS, so the verdict is robust to method.
- **GPU coordination:** two other agents ran bursty jobs (core.delivery, Laptop training)
  concurrently; I owned the long runs, ran the two arms sequentially, never killed others'
  processes. My own orphaned/duplicate eval chain (harness severed one python but left the
  bash) was cleaned up (killed the stale tree) before the final clean eval.
- **Native full-frame LPIPS at 24 MP** is neither cheap (tiled = ~5800 VGG calls/view) nor
  well-defined; native perceptual quality is measured by the microstructure region-LPIPS.
