# Flywheel: Splat-Native Generative Prior — Executable R&D Plan (June 2026)

The moonshot (RESEARCH_VISION.md Invention 2): a model trained across many scans
that can **complete** (recreate unseen regions), **super-resolve**, and **clean**
3D Gaussian splats — the genuine "recreate splats" leap + a compounding data moat
(each scan → corpus → better prior → better scans). Per-scene optimization is at
its ceiling; this is the path past it (alongside re-capture).

Each stage has a **validation gate** — advance only if it passes.

## Stage 0 — Canonical encoding ✅ SOLVED (validated)
A generative model needs a fixed-topology, regular tensor; Gaussians are an
unordered variable-size set. **Result:** naive volumetric voxel grid FAILS (surface
manifold → 90% empty / collisions, −8 dB; `splat_codec.py`). **Morton-ordered
"Gaussian-image" WORKS** (`splat_codec2.py`): order Gaussians along a Z-order curve
into an (S,S,F) image — every Gaussian → one pixel (no collisions/waste), spatially
coherent, CNN/diffusion-ready, fixed N=S² across scenes (prune-by-opacity/pad).
Validated **lossless**: 3M splat → (1280,1280,59) → decode → render 24.18 vs 24.18
(dPSNR −0.00). F=59 = means3+scales3+quats4+opacity1+sh0_3+shN_45.

## Stage 1 — Corpus (THE bottleneck) ⬜
A generative prior needs **hundreds–thousands** of splats; we have ~4. Sources:
1. **Our scans** (seed; grows with use — the flywheel). Convert each trained splat
   → Gaussian-image via `splat_codec2.encode`.
2. **Public 3DGS datasets** — pre-trained splats / scenes from MipNeRF360, Tanks&
   Temples, DL3DV, and the training sets used by DiffGS (2410.19657) / GaussianCube
   (2403.19655) / TRELLIS (2412.01506). Object-level sets are the easiest start.
3. **Synthetic** — render public 3D assets (Objaverse-scale) from many views → fit a
   splat per asset with our gsplat trainer (batchable, unlimited supply). This is the
   scalable engine; build a `render_asset → train_splat → encode` pipeline.
   - **Gate:** ≥ a few hundred encoded Gaussian-images, normalized to fixed S, with
     per-channel statistics computed (means/scales/quats/sh have very different
     ranges → must standardize per-channel for diffusion to train).

## Stage 2 — Autoencoder / latent (optional) ⬜
(S,S,59) at S=1280 may be heavy for diffusion. Train a VAE to compress the
Gaussian-image → a smaller latent (e.g. (160,160,C)), OR diffuse directly in image
space at smaller S. **Gate:** VAE round-trip renders within ~0.3 dB of the encoded
splat on held-out scans.

## Stage 3 — Generative model ⬜
2D diffusion (standard diffusers UNet) over the (normalized) Gaussian-images (or VAE
latents). **Gate:** unconditional samples decode to plausible splats; held-out splats
denoise/reconstruct well.

## Stage 4 — Conditioning & use ⬜
- **Completion:** mask the Gaussian-image pixels corresponding to unobserved
  Gaussians (from the Phase-1a coverage/visibility field), inpaint with the diffusion
  prior → recreate the unseen back, conditioned on the observed Gaussians.
- **Super-resolution / cleaning:** prior as a denoiser over the image.
- **Fusion:** graft generated vs real per the confidence field (Phase-1a) — real
  where captured, generated where not. **Gate:** on a held-out-wedge splat, completion
  beats the Phase-1a Difix baseline (22.59 on the GPU wedge) on LPIPS + visual.

## Open research questions (flagged)
- Morton has Z-order discontinuities; Hilbert ordering may give better 2D locality
  for the CNN — test both.
- Per-channel normalization of the 59-dim features is essential and non-trivial
  (quats on a sphere, scales log-space, opacity logit).
- Fixed N (prune/pad) vs variable scenes; scene-scale vs object-scale (public priors
  are object-level — start object-level, extend to scene).
- Diffusing 59-channel "images" is non-standard; may need a tailored UNet.

## Honest status
Stage 0 solved + validated. Stage 1 (corpus) is a deliberate data effort (public
datasets + a synthetic-splat pipeline), the real gate — not a quick experiment.
Stages 2–4 are standard-ish once the corpus exists. Tools: splat_codec.py (voxel,
negative), splat_codec2.py (Gaussian-image, validated).
