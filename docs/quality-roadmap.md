# Apex Reality — Quality Roadmap (June 2026, rev 3)

**Goal:** general pipeline improvements so reconstructions are crisp on *all*
surface types across *many* clips. **Context: RESEARCH-ONLY** — license is not a
selection filter; we pick the strongest method and reimplement clean-licensed
versions for our own OSS. License noted per method for that future port.

## The headline problem (this clip, and general): mushy flat/textureless surfaces

Confirmed visually (4K render vs GT crop of the laptop's black deck): textured
regions (keys, labels, foliage, brick) are crisp; **flat, dark, semi-gloss plastic
(palm-rest, deck, bezel) is mushy/waxy with lost micro-detail and slight geometric
smear** — worst on held-out (novel) views. Floaters are NOT the issue here (they
were the GPU-scan concern; keep that tooling for other clips).

### Mechanism (verified across primary sources)
1. **Photometric ambiguity (root cause):** on flat-color regions L1+SSIM is
   satisfied by many geometries → depth unconstrained → Gaussians sit at arbitrary
   depths/spread instead of snapping to the surface (PlanarGS, AmbiSuR "photometric
   blending ambiguity", PolarGS).
2. **Starved init:** detector features (ALIKED) don't fire on flat plastic → no SfM
   points there → sparse, coarse Gaussians. **For our MCMC trainer this is the
   dominant half** (MCMC doesn't densify by gradient, so AbsGS/Pixel-GS don't apply
   — the fix must come from init + loss terms, not the densifier).
3. **Dark semi-gloss view-dependence:** SH≤3 can't fit a sharp/moving specular lobe
   → optimizer smears low-opacity Gaussians to absorb the error ("mush").

## Ranked fixes (impact ÷ effort)

| # | Fix | Attacks | Effort | OSS-clean |
|---|---|---|---|---|
| 1 | **Edge-aware geometric reg** (depth-TV + depth→normal consistency, gated by `exp(-|∇I|)`), reimplemented in our trainer | photometric ambiguity (root) | Med | Yes (our code) |
| 2 | **Mono-normal supervision** (Lotus/StableNormal → L1 + TV normal loss, masked to low-texture) | unconstrained flat geometry | Med-High | Yes (Apache estimators) |
| 3 | **Better textureless init** (HLoc Efficient-LoFTR dense branch; + optional mono-depth backprojection seeding) | starved init | Med | Yes (Apache/MIT) |
| 4 | **`app_opt` (+ `sh_degree 4` A/B)** — already in gsplat | dark/exposure mush, view-dep color | Low | Yes (Apache) |
| 5 | **Surface-aligned rep** (gsplat native 2DGS A/B; or RaDe-GS backend) | flat-surface geometry | Low / High | 2DGS/RaDe-GS non-commercial; gsplat harness Apache |
| 6 | **Spec-Gaussian (ASG)** backend | dark semi-gloss specular | High | MIT |

**The unifying principle (do this first):** every SOTA textureless method (AtomGS,
PGSR, DN-Splatter, DET-GS) gates its geometric regularizer by the **inverse image
gradient** — strong smoothing where flat, silent at edges. This is exactly our
"texture-aware weighting" need: fixes the flat plastic without softening crisp keys.

## Concrete integration (for our gsplat MCMC 3DGS trainer)

### Fix 1 — Edge-aware geometric regularization (license-free, our code)
We already can render expected depth (`render_mode="RGB+ED"`). Add, gated by GT
image gradient `w = exp(-|∇I|)`:
- **Edge-aware depth TV:** `mean(|∂xD|·exp(-|∂xI|) + |∂yD|·exp(-|∂yI|))`.
- **Depth→normal consistency:** normals from neighbouring back-projected depth
  pixels `N_d = norm(cross(dD/dx, dD/dy))`; loss `mean((1-|∇I|)²·||N_d − N||₁)`
  (PGSR Eq.6). No external prior needed; uses only the depth we render.
- Start ~iter 3000 (after coarse geometry forms), λ swept ~0.01–0.05.

### Fix 2 — Monocular normal supervision
Precompute per-frame normals with **Lotus** (Apache, lowest error) or
**StableNormal** (Apache, sharpest on blur/low-texture). Add DN-Splatter-style
`L_N = ||N_render − N_mono||₁ + TV(N_render)` + scale-flatten
`Σ‖min-axis(s)‖₁`, masked to low-gradient pixels, λ ramped down after ~7k.
Normal route (quick): depth→normal (Fix 1). Route (faithful): per-Gaussian normals
via gsplat `return_normals=True, with_eval3d=True`.

### Fix 3 — Textureless init
Add a **dense-matching reconstruction branch** to Stage 3 using HLoc's
`match_dense.py` with **Efficient-LoFTR** (Apache) — finds matches on flat plastic
that ALIKED misses; HLoc quantizes to COLMAP keypoints + fuses with ALIKED tracks.
Optionally seed `points3D` by back-projecting mono-depth (aligned per-view to the
sparse points). We already built **RoMa v2 rescue** (Phase 2) — same idea; LoFTR is
the lower-friction default. This directly puts points on the black plastic.

### Fix 4 — app_opt + SH (free, in-stack)
Vendor/enable gsplat's `AppearanceOptModule` (`app_opt`, embed 16, lr 1e-3) — absorbs
residual per-frame photometric drift (the modest ±15% we measured) and helps dark
mush. A/B `sh_degree 4` (low expected impact; quadratic VRAM; revert if it overfits).

### Fix 5/6 — only if 1–4 leave it soft
gsplat native **2DGS** mode (`--normal_loss --dist_loss`) — surface-aligned A/B, but
~1 dB PSNR loss + over-smooths low-texture (tension with our crisp goal). **RaDe-GS**
keeps full-3D quality but is a separate CUDA backend. **Spec-Gaussian** (MIT, ASG
lobes) for the dark glossy highlights — separate backend, Windows CUDA build.

## Recommended first chunk — "Textureless-surface package" (Phase A)
General across clips, highest-confidence, license-clean:
1. Render depth in the trainer; add **edge-aware depth-TV + depth→normal
   consistency** (Fix 1) — the root-cause fix.
2. Add **mono-normal supervision** (Fix 2) with Lotus/StableNormal.
3. Enable **app_opt** (Fix 4) — cheap, in-stack.
4. Add **Efficient-LoFTR dense init branch** (Fix 3).
Then retrain the Laptop (object-only AND full-scene), re-crop the black deck vs GT,
and measure held-out PSNR/SSIM. Layer 2DGS/RaDe-GS/Spec-Gaussian only if needed.

## General toolkit for OTHER clips (not this one's issue, keep available)
- **Floaters:** Clean-GS (Apache, reuses our SAM masks, no retrain) + opacity/scale
  prune + SuperSplat. (GPU-scan-type clips.)
- **Soft under-observed extremes:** Difix3D+ distill-back (ships a gsplat trainer)
  / FreeFix (MIT). Conservative — HAD shows blanket diffusion lowers fidelity.
- **Detail ceiling:** higher MCMC `cap_max`; SeedVR2 on final video; SuperGaussian
  splat-SR.

## Constraints (unchanged)
MCMC → no AbsGS/absgrad. Don't super-resolve inputs. SfM at 1600px, train 4K.
Scope = build masking as a toggle → produce full-scene AND object-only.

## OSS-license map (for our future open-source port)
Clean: our own losses, DN-Splatter, Lotus, StableNormal, Metric3D-v2, Efficient-
LoFTR, GIM, RoMa v2, gsplat app_opt, Spec-Gaussian, Clean-GS, 3D-Gaussian-Flats
(Apache/MIT/BSD). Non-commercial (reimplement, don't ship): 2DGS/GOF/RaDe-GS/
GaussianShader/AtomGS (Inria), PGSR (ZJU), DSINE (Imperial), MASt3R/VGGT-default/
DA3-Giant weights, Difix3D+ research weights, 3DGS-DR (no license).
