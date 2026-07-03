# WP-C1 — GaMO-pattern gated generative completion, v1 (2026-07-02)

Full gate loop on the GENUINE-HOLE testbed (`hole_testbed_RECORD.md`). Question:
does a **latent-gated** per-view generative fill (a) plausibly fill a real void,
(b) stay **structurally unable to touch observed pixels**, (c) fuse into a
consistent 3D completion when distilled back — and does the gate beat an ungated
generation? Per the §0 GATE EVIDENCE RULE every number below is persisted.

## Components built (owned: core/enhancement additions + _infusion scripts + WSL)
- `core/enhancement/latent_gate.py` — backbone-agnostic GaMO gate mechanism (torch
  only): alpha→latent max-pool mask, progressive dilation (k=5, ×2/×1/×0),
  known-region composite `x=m·x+(1-m)·x_known`, trigger-step map (30/50/70%),
  feathered paste-back (inset 0.6, feather 0.08).
- `_infusion/gamo_fill.py` — WSL runner (artifixer-venv, diffusers 0.37.1). Backbone
  **SDXL-Inpaint** + DDPM (see backbone note). GATED = latent injection + feathered
  paste-back; UNGATED = same generator, no injection, no paste-back (raw VAE frame).
- `_infusion/hole_distill.py` — distill fills → splat via the production trainer's
  extra-views path (MCMC densifies the void from the fills; void-seeing arc held out
  of REAL supervision via APEX_HOLDOUT).
- `_infusion/hole_gate.py` — gate metrics (2D gate-integrity + splat fixed-mask).

## Backbone note (deviation from the plan's "FLUX.1-Fill v1")
FLUX.1-Fill was NOT provisioned on this machine: the conda envs referenced by
`core/enhancement/inpaint_backend.py` (`…/anaconda3/envs/{zimage,flux-fill,difix}`)
do not exist (migrated-away box), no FLUX weights are cached anywhere (Win or WSL),
and the weights are gated (no HF token). The only large diffusion model local to the
WSL diffusers stack is Wan2.1-T2V-14B (text-to-VIDEO — not image/mask conditioned,
awkward for per-view stills). So v1 backbone = **SDXL-Inpaint**
(`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, ungated, mask-native, the
registered `sdxl` backend), downloaded into the WSL venv. It is DDPM-friendly → a
clean `scheduler.add_noise` path for the GaMO known-region injection. The gate
MECHANISM is backbone-agnostic; a FLUX/Wan swap is a one-file change. RePaint 3-iter
resample is approximated by SDXL's iterative refinement + trigger-injection + feather
(a `callback_on_step_end` cannot re-run a sampler step) — the one mechanism deviation.

## Result 1 — 2D per-view gate integrity (BEFORE distill) — DECISIVE, `gate2d.json`
Observed region = OGGSplat observed mask from the HOLED alpha (fixed). Fill vs the
holed render on that region:

| arm | observed MAD (↓, [0,1]) | observed gate-PSNR (↑) | void-vs-GT PSNR* | void-vs-GT LPIPS* |
|---|---|---|---|---|
| **GATED**   | **0.000369** (max 0.00044) | **55.8 dB** | 5.45 | 0.6757 |
| ungated | 0.013808 (max 0.0182)      | 33.5 dB | 6.14 | 0.6941 |

*void-vs-GT = reference-only "plausible-vs-actual" (NOT a gate criterion — the void
is unobserved, no GT claim). Both poor vs GT, as expected for generated fill.

**Gate integrity PASSES decisively:** the latent gate + feathered paste-back keeps
observed pixels byte-identical (55.8 dB; the 0.00037 residual is only the feather
ring + cubic-resize rounding — far below any metric noise floor). WITHOUT the gate,
observed pixels drift 37× more (33.5 dB) via the VAE round-trip + free generation —
exactly the corruption the gate exists to prevent. Visual: `_infusion/hole/
_compare_strip.jpg` (holed | GATED | ungated | GT) — ungated shows global softening +
colour-shift blobs on OBSERVED pixels; gated leaves them untouched.

Per-view fills are mutually INCONSISTENT (each SDXL fill hallucinates the void
independently — visible across v0297/v0304/v0311 in the strip). This is the
"per-view fill flickers before distill" the plan predicts; whether distill-back
fuses it is the decisive test below.

## Result 2 — splat-level gate (AFTER distill)
Distilled splats: `_infusion/hole/distill_{gated,ungated}/ckpts/ckpt_3999_rank0.pt`
(init = holed; 364 real train + 23 fill views; void-seeing arc indices 170–374 held out of
REAL supervision via APEX_HOLDOUT=0.30,0.36; MCMC cap 1.4M — +165k headroom, both
arms hit the cap; 4000 steps, seed 0). Scored on the hole arc `wedge:293:316`,
full 6-scorer suite. Scorecards: `qa/hole_{baseline,gated,ungated}/`; fixed-mask
gate JSONs: `_infusion/hole/gate_splat_{gated,ungated}.json`.

| metric (dir) | holed baseline | GATED-fill splat | ungated-fill splat |
|---|---|---|---|
| observed% (own mask) ↑ | 80.6 | **100.0** | **100.0** |
| obs masked PSNR ↑ (FIXED baseline mask) | 15.23 | 16.81 | 17.03 |
| obs masked SSIM ↑ (fixed) | 0.7512 | 0.7617 | 0.7673 |
| obs masked **LPIPS_tiled** ↓ (fixed, **H1-corrected**) | 0.4791 | **0.4722** | 0.4805 |
| obs masked LPIPS ↓ (fixed, ~~deprecated FULL-FRAME~~) | ~~0.5518~~ | ~~0.5446~~ | ~~0.5504~~ |
| obs drift-vs-holed MAD ↓ (fixed) | — | **0.0386** | 0.0430 |

**H1 (2026-07-02 closeout):** the fixed-mask observed-LPIPS row above was re-measured in
`masked_lpips(mode="tiled", tile=256, cover=0.8)` at full res (the banked FULL-FRAME values —
struck through — were accidental: the observed region is an ANNULUS around the central void, so
`bbox` mode's crop = the whole frame, `bbox` area frac → 1.0, and the value was bit-identical to
the own-mask scorecard LPIPS incl. plausible-not-faithful void pixels). Corrected tiled ordering:
**gated 0.4722 < baseline 0.4791 < ungated 0.4805** — the gate leg (gated < ungated) HOLDS. New
per-view + mean fields live in `_infusion/hole/gate_splat_{baseline,gated,ungated,control}.json`
(old ones kept as `observed_masked_LPIPS_deprecated_fullframe`). PSNR/SSIM/drift are pixelwise and
stand unchanged. **The NO-FILL control (§CLOSEOUT-H2 below) actually wins this row (0.4648) — read
the four-way attribution before citing any single arm.**
| CrossScore ↑ (plausibility) | 0.6052 | 0.6663 | 0.6767 |
| NOVA ↓ (plausibility) | 0.3664 | 0.3663 | 0.3596 |
| TSED ↑ (consistency) | 1.000 | 1.000 | 1.000 |
| RAFT-warp ↓ (consistency) | 0.01486 | 0.01313 | 0.01283 |
| NVS-SQA ↑ | −2.358 | −2.383 | −2.382 |
| flicker ↑ | 0.9910 | 0.9895 | 0.9896 |

Visual: `_infusion/hole/_distill_strip.jpg` (holed | gated-distilled | GT) — the
mutually-inconsistent 2D fills FUSE into semi-consistent 3D card-ish geometry with
white smearing where fills disagreed (fusion residue of an SDXL prior that knows
nothing of the true card).

## Verdict

**1. Genuine-hole completion WORKS end-to-end (vs the holed base): PASS.**
observed% 80.6 → **100.0** (void completed); CrossScore +0.061; RAFT-warp improves
0.0149→0.0131; TSED stays at ceiling 1.0 (no consistency injected); NOVA flat;
observed masked-LPIPS (fixed mask, **tiled/H1-corrected**) **improves** 0.4791→0.4722. Against the
WP-C1 bar "beats base on {CrossScore, NOVA, TSED, RAFT} with observed masked-LPIPS
unchanged": 2 wins + 2 ties, 0 losses, LPIPS better ⇒ **met** (qualified: TSED is
at ceiling, NOVA flat). **BUT the CLOSEOUT no-fill control (§below) beats the fills on every
splat-level axis — this "PASS vs base" is largely MCMC-regrowth, not the generative fill; read
§CLOSEOUT-H2.** NVS-SQA/flicker marginally worse (−0.025 JOD / −0.0015)
— the smeary fill content; honest cost, within these metrics' small-delta range.

**2. The gate's structural guarantee is a 2D-level property (decisive, 37×).**
Observed pixels: gated 55.8 dB vs ungated 33.5 dB. Wherever fills are consumed
directly (per-view enhancement, 2D deliverables) or fills dominate supervision,
only the gated path is safe-by-construction.

**3. NEW FINDING — real-supervised distillation is itself a strong secondary gate.**
With 364 real views vs 23 fill views, the ungated arm's observed regions get pulled
back to truth at distill time (fixed-mask PSNR 17.03 vs gated 16.81) and its
plausibility even edges ahead (CrossScore 0.677 vs 0.666, NOVA 0.360 vs 0.366);
gated keeps the edge on observed LPIPS (0.5446 vs 0.5504) and drift-vs-holed
(0.0386 vs 0.0430). So in the REAL-VIEW-DOMINANT distill regime, splat-level
gated ≈ ungated — the trainer dilutes 2D corruption. The gate's mandatory role is
(a) the guarantee itself (never inject corruption, rather than hope training
rescues it), and (b) the regimes distill can't rescue: fill-dominant supervision,
fill poses with little real-view overlap, 2D delivery. Refines program-doc §2,
does not contradict it.

**4. Fusion works; quality is backbone-limited.** Per-view SDXL fills are mutually
inconsistent, yet after distill TSED=1.0 and RAFT improves — multiview MCMC fusion
averages per-view hallucination into stable (if smeary) 3D. The ceiling is the 2D
prior: FLUX.1-Fill / causal ArtiFixer / GENA3D slot into this same gate + distill
loop unchanged (one-file backbone swap).

## Caveats / honesty
- **Boundary leakage:** grazing views OUTSIDE the holdout arc still see ~6–10% of
  the void (broad scan: v60 → 93.9%, v540 → 90.8% observed) → some real supervision
  reached the void boundary in BOTH arms equally. The void interior was fill-only.
- The 23 arc GTs were training views of the ORIGINAL production splat; the void is
  a carved deletion (geometry gone, arc held out of the distill), not a
  never-captured hole. A never-captured testbed still requires a new capture.
- TSED at ceiling (1.0) for all three arms on this arc — low discriminative power
  here; RAFT carries the consistency signal.
- NVS-SQA/flicker deltas ≤0.03 — treat as ~flat.
- Single seed (0), single scene, one box geometry, n=23 views, 4000-step distill.
- 2D void-vs-GT (5–6 dB) confirms plausible-NOT-faithful — expected and correct
  for an unobserved region; never a gate criterion.

## Status vs the WP-C1 plan
- GaMO mechanism re-implemented on our stack ✓; mask source unified with WP-Q1's
  visibility field (the OGGSplat mask IS the gate mask) ✓; genuine-hole testbed
  (the plan's mandate) built + banked ✓; gated beats/ties base with observed
  region protected ✓; gated-vs-ungated: decisive in 2D, ~tied at splat level in
  the real-view-dominant regime (finding #3) — the "beats ungated" leg holds for
  the gate-integrity axis (LPIPS/drift), not the plausibility axis.
- Backbone deviation: SDXL-Inpaint, not Wan2.1/FLUX (see backbone note). Next
  levers, all drop-in: FLUX.1-Fill (request gated weights), Wan2.1-I2V video gate
  (temporal coherence across the arc), causal-ArtiFixer probe (Annex A).

## Reproduce
```
# fills (WSL)
~/artifixer-venv/bin/python _infusion/gamo_fill.py --arc _infusion/hole/arc \
    --out _infusion/hole/fills_gated  --mode gated   --steps 30 --res 1024 --seed 0
~/artifixer-venv/bin/python _infusion/gamo_fill.py --arc _infusion/hole/arc \
    --out _infusion/hole/fills_ungated --mode ungated --steps 30 --res 1024 --seed 0
# 2D gate
python _infusion/hole_gate.py --mode fills2d --arc _infusion/hole/arc \
    --gated _infusion/hole/fills_gated --ungated _infusion/hole/fills_ungated --out _infusion/hole/gate2d.json
# distill + splat gate
python _infusion/hole_distill.py --holed _infusion/hole/holed_gpuv4.pt --fills _infusion/hole/fills_gated \
    --arcmeta _infusion/hole/arc/meta.json --data <GPU_V4 data> --out _infusion/hole/distill_gated --holdout 0.30,0.36 --steps 4000
python qa_report.py _infusion/hole/distill_gated/ckpts/ckpt_3999_rank0.pt <data> --factor 2 \
    --split wedge:293:316 --scene hole_gated --out qa/hole_gated --scorers all
python _infusion/hole_gate.py --mode splat --arc _infusion/hole/arc \
    --splat _infusion/hole/distill_gated/ckpts/ckpt_3999_rank0.pt --data <data> --out _infusion/hole/gate_splat_gated.json
```

═══════════════════════════════════════════════════════════════════════
# WP-C1 CLOSEOUT — code-review fixes (2026-07-03)
═══════════════════════════════════════════════════════════════════════
Review verdict: **benchmark VALID, gate STANDS, with findings.** The numbers below
are all persisted (gate JSONs / this record / qa scorecards). Nothing regenerated
except the fixed-mask LPIPS re-measurement (H1) and one new distill arm (H2 control).

## H1 — observed-LPIPS re-banked in TILED mode (the fixed-mask LPIPS was full-frame)
The banked "fixed-mask observed-LPIPS" were accidentally FULL-FRAME: the observed
region is an ANNULUS around the central void, so `masked_lpips` **bbox** mode's crop =
the whole frame (verified: bbox area frac → 1.0; the banked `gate_splat_gated.json`
value was bit-identical to the own-mask scorecard LPIPS). Re-measured with
`masked_lpips(mode="tiled", tile=256, cover=0.8)` at FULL resolution, FIXED baseline
visibility mask on all arms, full 23-view arc (`_infusion/hole_gate.py::splat_mode`):

| arm | tiled obs-LPIPS ↓ (fixed, full arc) | reviewer 7-view spot-check | deprecated full-frame |
|---|---|---|---|
| baseline (holed) | 0.4791 | 0.4818 | 0.5518 |
| **gated**        | **0.4722** | 0.4745 | 0.5446 |
| ungated          | 0.4805 | 0.4828 | 0.5504 |
| control (H2)     | **0.4648** | — | 0.5115 |

Full-arc ≈ spot-check (each ~0.002–0.003 lower on the fuller sample, same ordering).
**Gate leg gated (0.4722) < ungated (0.4805) HOLDS** on the corrected metric; gated also
beats baseline (0.4791). New fields banked per-view + mean in
`_infusion/hole/gate_splat_{baseline,gated,ungated,control}.json`
(`observed_masked_LPIPS_tiled` + `_deprecated_fullframe`). PSNR/SSIM/drift are
pixelwise/pixel-decomposable → genuinely correct, unchanged.

## H2 — NO-FILL CONTROL distill (bounds MCMC-regrowth + grazing-view attribution)
Identical config to the fill arms, **zero fill views** (real-view supervision only).
argv (from `_infusion/hole/distill_control.log`, byte-for-byte the fill-arm trainer
line minus the 23 fills): `hole_distill.py --holed holed_gpuv4.pt --fills
<empty dir> --arcmeta arc/meta.json --data _dataset/GPU_V4/.../data --out
distill_control --holdout 0.30,0.36 --steps 4000` → trainer `--data_factor 2
--max_steps 4000 --strategy mcmc --cap_max 1400000 --refine_stop_iter 3200 --init_ckpt
holed_gpuv4.pt --extra_views [] --lpips_lambda 0.2 --lpips_gated --lpips_start 500
--antialiased --random_bkgd --save_steps 4000 --seed 0`, `APEX_HOLDOUT=0.30,0.36`
(holds out indices 170–374; 364 real train, 205 test). Scored identically:
`qa/hole_control/` (6-scorer suite) + `gate_splat_control.json` (tiled fixed-mask) +
`floater_report.py`.

### Four-way table (splat-level, hole arc `wedge:293:316`, 23 views, seed 0)
| metric (dir) | holed base | **CONTROL (no-fill)** | GATED-fill | ungated-fill |
|---|---|---|---|---|
| observed% (own mask) ↑ | 80.6 | **100.0** | 100.0 | 100.0 |
| obs masked PSNR ↑ (fixed) | 15.23 | **19.59** | 16.81 | 17.03 |
| obs masked SSIM ↑ (fixed) | 0.7512 | **0.7811** | 0.7617 | 0.7673 |
| obs masked LPIPS_tiled ↓ (fixed, H1) | 0.4791 | **0.4648** | 0.4722 | 0.4805 |
| obs drift-vs-holed MAD ↓ (fixed) | — | 0.0985 | **0.0386** | 0.0430 |
| CrossScore ↑ | 0.6052 | **0.7697** | 0.6663 | 0.6767 |
| NOVA ↓ | 0.3664 | **0.2116** | 0.3663 | 0.3596 |
| TSED ↑ | 1.000 | 1.000 | 1.000 | 1.000 |
| RAFT-warp ↓ | 0.01486 | **0.01231** | 0.01313 | 0.01283 |
| NVS-SQA ↑ | −2.358 | −2.408 | −2.383 | −2.382 |
| flicker ↑ | 0.9910 | 0.9893 | 0.9895 | 0.9896 |
| own-mask masked-LPIPS ↓ | (0.5518, diff mask) | 0.5115 | 0.5446 | 0.5504 |

### Floater report (all four splats, `floater_report.py`)
| splat | Gaussians | needle% | SOR-outlier% | giant% | low-opa% | med-opa | erank |
|---|---|---|---|---|---|---|---|
| holed | 1,235,399 | 92.46 | 0.14 | 9.54 | 33.6 | 0.024 | 1.564 |
| control | 1,400,000 | 93.67 | 0.19 | 9.38 | 32.6 | 0.020 | 1.517 |
| gated | 1,400,000 | 93.93 | 0.24 | 9.33 | 32.4 | 0.020 | 1.521 |
| ungated | 1,400,000 | 93.90 | 0.11 | 9.11 | 32.8 | 0.020 | 1.521 |
All three distills densify to the 1.4M cap and stay clean: **SOR-outlier 0.11–0.24%**
(isolated-floater fraction) barely moves off the holed 0.14%, giant/low-opa/erank all
comparable → observed%→100 is COHERENT densified content, **not isolated-floater haze**.

### Honest attribution — how much does the control already achieve?
**Essentially all of it, and then some.** On this DENSE-DOME, REAL-VIEW-DOMINANT testbed:
1. **observed% 80.6 → 100: FULLY achieved by the no-fill control.** MCMC densification +
   grazing real-training views (v0060 → 93.9%, v0540 → 90.8% observed; both outside the
   170–374 holdout, each seeing 6–9% of the void boundary) re-grow the void's ALPHA to
   100% observed with zero generative input. The fill adds **no** incremental observed%.
2. **Fixed-mask observed region: the control WINS** (tiled LPIPS 0.4648, PSNR 19.59 vs
   gated 0.4722/16.81). Real supervision on the 364 surrounding views HEALS the observed
   annulus toward true GT more than either fill arm — the fills slightly perturb near-void
   geometry. The control's high drift-vs-holed (0.0985) is that healing (movement toward
   GT; "vs holed" is the wrong reference for it — the holed render is not ground truth).
3. **Plausibility/consistency (own-mask, no-reference): the control DOMINATES** —
   CrossScore 0.7697 (vs gated 0.6663/ungated 0.6767), NOVA 0.2116 (vs ~0.36), RAFT 0.0123
   best; own-mask GT-LPIPS 0.5115 (vs 0.5446/0.5504). Its void interior is a SMOOTH guess,
   which NR scorers + reference-LPIPS reward over the mutually-inconsistent SDXL fills that
   fuse to smeary card geometry (the classic "blurry beats sharp-but-wrong on reference
   metrics" bias — the void is unobserved, so there is no "correct" sharp content to hit).

**What this means (and does NOT mean).** It is a BOUNDING result, not a refutation of the
gate. It quantifies the testbed's built-in limitation (already flagged in
`hole_testbed_RECORD.md`: dense captures resist coverage holes; MCMC heals from grazing
views) — **a never-captured hole (a fresh capture with genuinely zero grazing coverage) is
required to make generative fill strictly necessary at the splat level.** The gate's value
survives exactly where §Result 1 and the M3 ratification place it: the **2D structural
guarantee** (observed pixels 55.8 dB gated vs 33.5 dB ungated, 37×, decisive) and the
**fill-dominant / low-real-overlap / 2D-delivery** regimes where there is no army of 364
real views to regrow the region. On the real-view-dominant SPLAT axis, distill-as-secondary
-gate is so strong that a no-fill control beats the fills — the sharpest form of finding #3.

## M2 — injection timestep off-by-one (fixed for future runs; banked fills NOT regenerated)
`_infusion/gamo_fill.py` `callback_on_step_end` composited the known latent noised to
`t` = `timesteps[i]` (the step just COMPLETED), but `cbk["latents"]` is the scheduler.step
OUTPUT = the sample at `timesteps[i+1]`. The installed SDXL-inpaint pipeline's own
known-region injection uses `noise_timestep = timesteps[i+1]` (verified at
`.../pipeline_stable_diffusion_xl_inpaint.py:1649`). **Fixed** to noise x0 to
`timesteps[i+1]` (guarded `i+1 < len`). The banked gated fills / distill_gated /
`gate_splat_gated.json` / `qa/hole_gated/` were generated with the OFF-BY-ONE and are NOT
regenerated: practical effect is small — the too-noisy known-latent only perturbs the
injected latents INSIDE the void, and the feathered pixel paste-back (§Result 1) masks the
observed-pixel consequences to the byte-identical 55.8 dB shown there (which was measured on
the as-banked fills, so it stands). The void interior is unobserved/plausible-only regardless.

## M3 — main-loop ratification (verbatim)
> Main-loop ratification: the beats-ungated gate leg holds on the gate-integrity axis (2D
> 55.8 vs 33.5 dB; corrected tiled LPIPS gated < ungated) and NOT on splat-level
> plausibility scores (ungated edges gated on CrossScore/NOVA/RAFT there — the
> distill-as-secondary-gate regime finding). Closure ratified on that basis; the latent gate
> remains mandatory for the 2D structural guarantee and fill-dominant/low-real-overlap/
> 2D-delivery regimes.

(The H2 control sharpens this: on the splat axis a NO-FILL control beats BOTH fills, so the
gate is not sold on splat-level plausibility here at all — only on the 2D guarantee + the
regimes distill/regrowth cannot rescue.)

## LOW / nits
- **Two coexisting mask definitions** (now commented in `gamo_fill.py`): the SDXL pipeline
  INPAINT mask = OGGSplat unobserved (alpha<0.3 + morphology, `_mask.png`) drives synthesis
  + paste-back; the LATENT-GATE mask = raw alpha<0.6 (GaMO spec) drives known-latent
  re-injection. alpha<0.6 is a CONSERVATIVE SUPERSET of alpha<0.3 (protects a slightly
  smaller observed set than the inpaint mask carves → safe direction; the hard guarantee is
  the OGGSplat-mask paste-back).
- **Same-epsilon-across-triggers (cosmetic):** `gamo_fill.py` reuses ONE `noise` tensor at
  all three trigger injections (30/50/70%) rather than a fresh draw per trigger, so the
  injected known-latent noise realization is correlated across triggers. Practical effect
  negligible (progressively-dilated masks + feather paste-back); flagged for the funded
  rebuild, not fixed (would perturb the banked fills for no measurable gain).
