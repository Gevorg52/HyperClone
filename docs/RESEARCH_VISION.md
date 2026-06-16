# Apex Reality — Research Vision: Toward Hyper-Realistic Splats (June 2026)

Forward-looking research direction beyond incremental tuning. Grounded in this
project's empirical findings + a multi-agent, primary-source-verified literature
sweep. Tags: **[V]** verified vs primary source, **[I]** inferred/our analysis.

## The core reframe — why we plateau

Today's 3DGS (ours included) is **memoryless per-scene curve-fitting**: optimize raw
primitives to match captured pixels, with **no learned prior** about how the world
looks. So in any region the photons don't constrain — sub-Nyquist detail, dark mush,
occlusions, the unseen back — it falls back to L1/SSIM and produces the *blurry mean
of plausible answers*. That's the ceiling. Diffusion models don't have this problem
because they carry a massive learned prior; human vision completes the unseen from
material/shape priors.

**Thesis:** give reconstruction a *memory* (a learned prior over 3D appearance/
material/geometry) and **fuse it with captured data by confidence** — authentic where
photons constrain, generative where they don't, never in conflict.

## The make-or-break finding (3/3 research agents, verified)

**No 2D image editor produces geometrically consistent novel views** — not Nano
Banana Pro (= Gemini 3 Pro Image, API-only), GPT Image 2 (API-only), Qwen-Image-Edit,
FLUX Kontext/2, Z-Image. They are *perceptually* consistent and *identity*-preserving,
but "rotate the object / see the back" = sampled plausible pixels, **not
triangulatable geometry**. Feeding their alternate angles into COLMAP/3DGS injects
false correspondences and **degrades** reconstruction. [V] The 2026 corrective
research line (MEt3R, Edit360, Tinker, HAD) exists *because* of this. "Consistency
LoRAs" give **identity** consistency, not geometric. [V]

**Corollary:** recreating the unseen back is **plausible, not faithful** — by
construction (those photons were never captured). Fine as *artistic* fill if labeled
and confidence-gated so it never touches captured regions. [V]

## Empirical ground truth from THIS project (what we proved)

- Generative enhancement (Difix v1/v2/v3) on a well-captured DENSE scene plateaus;
  helps only soft/under-observed regions. More Gaussians hurt (3M optimal). [project]
- WD-R perceptual loss: adds texture-grain but trades fidelity; **floaters in mush**
  regions (knn-spread 197 vs 103) — the perception-distortion tradeoff as 3D artifacts.
- The deck is capture-limited (defocus+codec, NOT motion blur; SfM not starved).
- Bilateral grid net-negative on manual-exposure clips.
- Lesson: every win came from **gating** a prior to where it helps. The next step is
  to make that gating principled + back it with a real learned prior.

## Strategy: two tracks, never mixed

- **Track A — AUTHENTIC** (faithful): maximize real signal; fuse any prior *by
  confidence* so it never overrides data.
- **Track B — ARTISTIC** (plausible): generative completion ONLY for genuinely-unseen
  regions, confidence-gated and labeled.

## Role of the named models (verified access matrix)

| Model | Access | Role for us |
|---|---|---|
| Qwen-Image-Edit-2511 | open (Apache, quant on 24GB), **fine-tunable** | best open editor; candidate 2D fixer in distill loop + per-object LoRA; 2D cleanup/inpaint only, NOT viewpoint |
| Z-Image (6B, Apache) | open, 4090-easy | best small open generator to LoRA; **Z-Image-Edit not released yet** — watch |
| FLUX.1 Kontext / FLUX.2 | open (non-commercial), tight 24GB | local edits/inpaint; backup editor |
| Nano Banana Pro / GPT Image 2 | **API-only, no fine-tune** | quality-ceiling ref / one-off artistic fill; not in core local pipeline |
| TRELLIS / VGGT / MapAnything | open, 4090 | the actual 3D-structured tools for viewpoint/geometry |

## Action plan (phased; feasibility honest)

**Phase 1 — Confidence-gated prior fusion** (near-term, highest leverage). Upgrade the
distill loop into the rigorous version of our hand-rolled gating:
- HARD data term (captured views always full weight — never blur real pixels).
- ANNEALED generation loss (GenFusion): real-dominated early, prior ramps in.
- REAL confidence gate: **WarpRF (multi-view warp disagreement) + ray-termination/
  transmittance** — NOT per-Gaussian scale (validated weakest). [V]
- VISIBILITY loss (Nerfbusters): density→0 in never-observed space → kills
  prior-introduced floaters (the WD-grain/Difix-floater failure mode). [V]
- Reliability masking (HAD / 3DGS-Enhancer dual-confidence): reject inconsistent
  generated pixels (+0.78 dB over plain Difix). [V]
- Optional: swap 2D fixer to Qwen-Image-Edit + per-object LoRA, A/B vs Difix.
- NOTE: true null-space projection (DDNM) does NOT transfer to 3DGS (non-linear
  renderer) → use **soft uncertainty-weighted gating** (OracleGS is closest). [V]

**Phase 2 — Authentic detail recovery** (near-term, no AI): burst/multi-frame
sub-pixel fusion of redundant video frames into the splat — recover REAL detail
(laptop is capture-limited). Authentic-first.

**Phase 3 — Artistic completion of unseen regions** (research bet): 3D-STRUCTURED
generators (TRELLIS image→Gaussians / ReconViaGen / Amodal3R — condition on real
views, generate only unseen) + optional light per-object LoRA (iFusion/DisCo3D recipe:
rank 12–64, ~200 steps, minutes, +3.3 dB [V]); graft real front + generated back via
the reliability mask (graft-and-splice is DIY). Labeled synthetic.

**Phase 4 — The flywheel (moonshot)**: reusable cross-scene splat prior from our own
scans. Open object-level bases: **DiffGS** (native partial→complete, weights out),
**GaussianCube** (3D-U-Net diffusion). Scene-scale is the real research cost. The
compounding moat: each scan → corpus → better prior → better/faster scans.

## Validate-before-build: the proof experiment (doing now)

Hold out a contiguous wedge of views from a scene, train baseline on the rest, then
compare on the held-out (simulated-unseen) region: baseline vs generative-completion +
confidence-gated distill, scored vs held-out GT (PSNR/LPIPS + visual). Confirms
whether the prior adds *faithful* value in the under-observed regime *on our data*
before investing in the flywheel.

## Key sources
WarpRF 2506.22433 · FisherRF 2311.17874 · Nerfbusters 2304.10532 · OracleGS
2509.23258 · GenFusion 2503.21219 · Difix3D+ 2503.01774 · HAD 2605.16873 ·
3DGS-Enhancer 2410.16266 · iFusion 2312.17250 · DisCo3D 2508.01684 · ReconViaGen
2510.23306 · Amodal3R 2503.13439 · TRELLIS 2412.01506 · DiffGS 2410.19657 ·
GaussianCube 2403.19655 · Bolt3D 2503.14445 · CAT3D 2405.10314 · Stable Virtual
Camera (Seva) 2503.14489 · ViewCrafter 2409.02048 · Gen3C 2503.03751.
