# Apex Reality — Quality Roadmap (June 2026, rev 3)

**Goal:** general pipeline improvements so reconstructions are crisp on *all*
surface types across *many* clips. **Context: RESEARCH-ONLY** — license is not a
selection filter; we pick the strongest method and reimplement clean-licensed
versions for our own OSS. License noted per method for that future port.

## FINAL RECOMMENDATION (research-backed, June 2026)

**Root cause, confirmed at gsplat source level:** MCMC relocation samples
`probs = opacities.flatten()` (opacity only) — so flat low-error/low-opacity
regions are drained *by design*. Our 0.16× deck density is the expected
equilibrium, not a bug. **You cannot densify or sharpen where there is no
photometric signal**, and a black semi-gloss textureless surface is signal-
starved (low texture + low albedo/SNR + view-dependent highlight). Much of the
deck's "lost" detail was **never photometrically captured** → a real per-capture
ceiling. The biggest gains are in the photons, not the optimizer.

**Recommendation — for the GENERAL tool (helps every clip), in priority order:**
1. **WD-R perceptual loss** (Apple, *Drop-In Perceptual Optimization for 3DGS*,
   arXiv 2603.23297, Mar 2026, code). The strongest *reproduced* perceived-quality
   lever (39k-pair human study, 2.3× preferred). Drop-in objective swap. Use the
   **regularized** form (unregularized makes web-like artifacts in undersampled
   regions = our deck). Track **LPIPS/DISTS, not PSNR**. → biggest method-side win.
2. **PGSR-style planar/normal reg + lower `opacity_reg` (0.01→0.001)** — the only
   MCMC-compatible way to "rebalance" toward textureless regions: manufactures a
   flat-region gradient so MCMC stops draining the deck. Keep cap_max 3M. Fixes
   coverage/geometry, NOT albedo detail.
3. **Spec-Gaussian (ASG) or 3DGS-DR (deferred reflection)** for glossy/semi-gloss
   surfaces (common: laptops, GPUs, statues) — replaces low-freq SH so view-
   dependent highlights are crisp not foggy. + optional GES (learnable kernel) for
   edges. All stack with our Mip-Splatting.
4. **Sharpest-frame selection** (variance-of-Laplacian per partition) in Stage 1 —
   fixes our video-cull bug AND stops soft frames averaging into mush. Optionally
   WildGaussians per-pixel uncertainty as a **binary mask** (not a soft weight —
   soft weighting destabilizes MCMC).
5. **Capture playbook** for hard surfaces (future shoots): cross-polarized lighting
   (kills specular), matte/scanning spray (adds correspondence texture), RAW +
   expose-for-shadows (beats the 8-bit SNR floor). Highest ROI where applicable.

**Reserve as cosmetic polish (synthesis, not measurement):** SplatSuRe
(arXiv 2512.02172 — most faithful, reuses real close-up detail, not diffusion) >
Difix3D+ (cleaner but can invent). Distilling 2D edits back into 3D bakes
*consistent hallucination* on textureless surfaces — fine for looks, not fidelity.

**DO NOT (evidence says these won't help our case):** more depth/normal priors
expecting RGB detail (geometry-only — DN-Splatter/DNGaussian treat depth↔color as
a trade-off); 2DGS/GOF/RaDe-GS (over-smooth appearance, lose ~1dB); raise cap_max
or swap the MCMC sampler (proven to hurt / would worsen starvation); FreGS /
Sobel/gradient losses (zero signal on textureless by construction).

**For THIS laptop specifically:** it's near its capture ceiling. Levers 1–3 give a
real but modest gain; pixel-perfect black plastic from passive 8-bit gimbal video
is not achievable. Invest in the general levers + capture guidance, not in
squeezing this one surface.

## EMPIRICAL UPDATE 3 (LPIPS perceptual loss — implemented + tested)

Added `--lpips_lambda` (torchmetrics VGG, random-crop, gradients verified).
Result is **resolution-dependent, not a clean universal win**:
- **≤1080p / under-resolved content: clean win** (factor-2 A/B: keyboard LPIPS
  -5.7%, deck -3.8%, both sharper).
- **Full 4K (well-resolved scene): a wash.** LPIPS helps where detail is MISSING
  (starved deck) but OVER-TEXTURES where detail is already present (keyboard adds
  wrong grain → LPIPS worse). λ=0.2 over-cooks (keyboard 0.274→0.327, PSNR -2dB);
  λ=0.1 milder but keyboard still regresses (0.285).
- **Lesson:** LPIPS λ must scale DOWN with resolution+run length; a *global*
  perceptual loss can't distinguish under- vs well-resolved regions. A clean 4K
  gain needs a **spatially-selective** perceptual loss.
- **SOLVED — gated LPIPS (`--lpips_gated`, default on).** Gates the LPIPS term by
  relative per-crop L1 error (EMA-normalized) → acts on under-resolved crops, ~0 on
  well-resolved. 4-way 4K A/B (held-out, deck/keyboard LPIPS):
  orig 0.318/0.274 · global0.1 0.316/0.285 · global0.2 0.309/0.327 ·
  **gated0.2 0.313/0.274**. Gated improves the deck WITHOUT harming the keyboard
  (0.274 = original) — the clean win global LPIPS couldn't achieve. Makes the
  perceptual lever safe at any resolution; now the production default. Deck gain
  modest (capture-limited) but free + safe. (Apple WD-R rejected: unreleased,
  spatially-uniform σ=4 = same blind spot, full-image-4K-VGG hostile to 24GB.)

## EMPIRICAL UPDATE 2 (cap_max experiment — budget is NOT the ceiling)

- **cap_max 8M FAILED (diverged + OOM).** Trajectory: trained fine to ~3M
  (PSNR ~23), then degraded as it added more — 21.6 @ 5.9M, 19.6 @ 8M, collapsed
  to ~10, memory exploded (overlapping giant Gaussians → tile-intersection
  blowup) → OOM at 33 GB. **More Gaussians actively hurt this scene.**
- **Conclusion: 3M is near-optimal; the Gaussian budget is NOT the detail
  ceiling.** "Not much difference between the 3M files" is because 3M already
  saturates the detail this *capture* supports. The ceiling is the **source
  capture + textureless representation**, not primitive count. Naive budget
  raise is OFF the table (and would need scale/opacity-reg retuning + gradual
  cap even to be stable).
- **Revised lever priority (after 2 negatives — geo-reg and cap-raise):**
  1. **Diffusion detail enhancement** (SeedVR2 on renders / Difix3D+ distill-back)
     — the realistic way to add perceived high-frequency detail the photometric
     reconstruction can't recover from this capture. Highest visual payoff.
  2. **Deck-specific dense init + priors** (LoFTR points on the deck + depth/normal
     supervision so MCMC retains them) — principled, modest payoff; targets the
     0.16× starvation without raising global count.
  3. **Better source** (deblur/sharpness-aware frame selection; drop the soft
     frames that average detail away).
- **Honest limit:** flat, dark, low-contrast micro-texture is a known hard case
  for photometric 3DGS; structural fixes will only partially close 184→703.

## EMPIRICAL UPDATE (laptop A/B + density profiling)

- **Fix 1 (edge-aware depth smoothness) did NOT help the mushy deck.** Direct
  deck-region measurement on a held-out view: sharpness 184 (baseline) vs 182
  (geo_reg) vs **703 (GT)**; deck-PSNR unchanged. Verified wired (gradients reach
  all params) — it's just the wrong lever here. Kept in code (off by default;
  may help genuinely-bumpy geometry on other clips), but **de-prioritized**.
- **Profiled the real bottleneck: the flat deck is Gaussian-STARVED.** Projected
  into the held-out view, the deck has **0.16×** the trained-Gaussian density of
  the textured keyboard (914 vs 5852 per kpx) and **0.28×** the SfM-init density.
  MCMC drains Gaussians from low-error flat regions toward textured ones, so the
  deck is *sparser after training than at init*. Too few Gaussians → only
  low-frequency (blurry) content representable → "mush".
- **Reordered fix priority (profile-driven):**
  1. **Fix 3 — denser init on textureless** (LoFTR/RoMa dense matching) → more
     SfM points → more seeded Gaussians on the deck. Attacks the 0.28× starvation.
  2. **Fix 2 — external depth/normal priors** → adds a *supervision signal* on
     flat regions (where photometric loss is ~flat), which both constrains
     geometry AND gives MCMC an error gradient there so it RETAINS Gaussians
     (counters the 0.16× drain).
  3. Fix 1 depth-smoothness: keep off by default.

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
