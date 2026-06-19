# Apex Reality — Quality Roadmap (June 2026, rev 5)

**Goal:** general pipeline improvements so reconstructions are crisp on *all*
surface types across *many* clips. **Context: RESEARCH-ONLY** — license is not a
selection filter; we pick the strongest method and reimplement clean-licensed
versions for our own OSS. License noted per method for that future port.

## SESSION CHAPTER (June 2026, rev 4): completion, generation, and the imperfect-capture frontier

A long build/validate session on *filling holes, adding detail, and generating
missing views*. Everything below is empirically validated on our GPU + Laptop scans.

### Shipped to the Stage-10 GUI (committed)
- **Auto-complete holes** (`core/enhancement/autocomplete.py`): detect (err×WarpRF
  multi-view-inconsistency + occlusion-aware coverage + alpha, edge-gated) → **InFusion
  seed** (cold-start Gaussians where MCMC can't spawn) → **accept/reject gate** (commit
  only if it improves agreement with the REAL photos on held-out neighbour views) →
  **production-trainer reconstruction** (MCMC inject_noise + relocation + gated-LPIPS).
  Validated: a genuine GPU hole 23.5→**32.6 dB** (beats original), floater-free.
- **Pluggable 2D inpainter** (`inpaint_backend.py`/`inpaint_runner.py`): **FLUX.1-Fill
  (FP8 on the 4090) is the default** — won the A/B vs Z-Image decisively (continues real
  material vs Z-Image hallucinating an off-context blob). SDXL/Z-Image as fallbacks.
- **Type-3 flows**: inpaint-fallback (occluded reference) + recreate-the-back (plausible).
- **Detail-refine** (de-mush): gated-LPIPS multi-view + densification recovers detail
  from the raw images for under-resolved fills.
- **Capture Coverage report** (`coverage.py`): map cameras onto the object's viewing
  hemisphere (centre via least-squares camera-ray convergence), report coverage % +
  missing-angle gaps + a "shoot these angles" re-capture report + coverage map.

### Validated learnings (the load-bearing rules)
1. **Mush was the weak REFINER, not the seed.** Our hand-rolled detail-refine left fill
   geometry stuck (no MCMC inject_noise/relocation). The REAL production trainer
   reconstructs faithfully (laptop hole, vs GT photos: holed 0.46 → reconstructed
   **0.325 LPIPS, beats the original 0.338**). FIX: InFusion = cold-start seed only; the
   validated MCMC+gated-LPIPS trainer does the reconstruction.
2. **Cause-A vs Cause-B (the detail dichotomy, information-theoretic).** Detail is
   recoverable faithfully ONLY if it exists in ≥1 view (Cause A = geometry/sampling-
   starved → anchors+re-texture/SR recover it). A textureless/dark/glossy surface
   featureless in ALL views is Cause B = CAPTURE-LIMITED → more Gaussians only smooth,
   any "detail" is hallucination. EMPIRICAL: +1.08M anchors + re-texture on the laptop
   lid → −0.0016 LPIPS (noise) = Cause B confirmed; the splat already matches the photos.
3. **Present-but-smooth vs genuinely-absent (the rule for WHEN to generate).** Generate
   INTO emptiness (missing table/periphery — FLUX fills convincingly) = good; generate
   OVER a present-but-smooth surface (the lid) = WRONG (invents texture that contradicts
   reality). Demo'd both.
4. **Coverage completeness = RE-CAPTURE, guided by Module 1.** Generative angle-fill is
   plausible-only and OOD-limited on real captures (see critical review).

### CRITICAL REVIEW — mistakes / things to re-examine
- **EscherNet "OOD wall" verdict was CONFOUNDED — needs a second look.** The real-object
  held-out-90°-wedge generated mostly BLANK/misframed output (PSNR 9.2). But blank =
  object OUT OF FRAME = a FRAMING/normalization bug (EscherNet needs the object
  normalized to ~unit size at a fixed radius/FOV/elevation convention; we recenter +
  scale-to-radius but DO NOT normalize object SIZE, and only tried a blender-flip). So we
  did NOT cleanly isolate OOD-appearance — the failure was dominated by an unfixed
  framing bug. The OOD concern is real (Objaverse-trained; matches the early TRELLIS/
  DiffGS finding) but plausible object-side generation may be achievable with PROPER
  object normalization (+ maybe a per-object LoRA). RE-TEST before fully writing it off.
- **MOTION BLUR is the biggest UN-EXPLORED lever for the real use case.** We diagnosed the
  LAPTOP softness as defocus+codec (not motion) and ruled out BAD-Gaussians *for that
  clip* — but the user's realistic captures are handheld ORBIT VIDEOS where motion blur
  is inherent. We have NOT pursued deblur-3DGS. KEY INSIGHT: **blur-smeared detail IS in
  the signal (Cause A) → motion-deblur recovers it FAITHFULLY**, unlike the Cause-B lid.
  Imperfect captures have MORE faithful headroom than the near-perfect laptop did.
- **Background-freeze TODO unbuilt.** Auto-complete's reconstruction retrains the WHOLE
  splat (mild global-drift risk on already-good regions). A background-freeze (optimize
  only seed + MCMC-densify the hole) would guarantee no-touch. Build for production.
- **Coverage ↔ Difix-fusion not connected.** Coverage detects UNDER-observed (≠ unseen)
  regions; Phase-1a coverage-gated Difix-fusion (validated +1 dB, in-prod) is the
  faithful tool for exactly those. Wire coverage gaps → Difix-fusion for the
  single-elevation-orbit case (the realistic capture).
- **Frame selection unbuilt for video.** We have the blur diagnostic (var-of-Laplacian,
  blur_report.py) but no sharpest-frame + coverage-balanced selection for video input.

### Forward plan — "near-perfect object splat from an imperfect orbit video"
Realistic input = handheld orbit video with motion blur + single-elevation coverage gaps
+ lower-res frames + textureless regions. Ranked levers (research synthesis pending):
1. **Motion deblur for 3DGS** (BAD-Gaussians / Deblur-3DGS lineage) — faithful recovery
   of blur-lost detail (Cause A). Likely the single biggest lever for orbit video.
2. **Sharpest-frame + coverage-balanced frame selection** from video (cheap, big).
3. **Coverage gaps → Difix-fusion (under-observed, faithful) + re-capture report (unseen).**
4. **Selective super-resolution** for aliased-but-present detail (under-sampled, Cause A).
5. **EscherNet proper-normalization retry** for plausible object-side completion.
6. Background-freeze reconstruction for production safety.

---

## SESSION CHAPTER (June 2026, rev 5): imperfect-capture pipeline — BUILD + VALIDATE + HANDOFF

This chapter is the **handoff record** for the "near-perfect object splat from an
imperfect orbit video" work. It is self-contained: a successor can read this section
alone and continue. Realistic input = handheld ORBIT VIDEO (motion blur + single-
elevation coverage gaps + lower-res frames + textureless regions). The rev-4 forward
plan above was researched (~40 arXiv IDs verified, all real) into a ranked program;
this rev-5 chapter records what got BUILT, TESTED, and what remains.

### The ranked program (research-backed, faithful-vs-plausible tagged)
Order of operations is load-bearing: **sharp-frame select → masked SfM → joint-deblur
train (deblur converges FIRST) → depth-reg → THEN coverage-gated generative fusion.**
NEVER fuse a generative prior onto blurry-overfit Gaussians (a prior mistakes blur for
low-freq texture and SOLIDIFIES it — arXiv 2512.10369).

| # | Lever | Faithful? | Status | Commit |
|---|---|---|---|---|
| 0 | Sharpest-frame selection (var-of-Laplacian + min-gap) | ✅ | **SHIPPED** | `05657e3` |
| 1 | **Joint motion-deblur** (BAD-Gaussians SE(3) B-spline port) — THE biggest lever | ✅ | **NOT STARTED** (big build; needs real blurry video to validate) | — |
| 2 | Object-mask alpha loss (kills background floaters) | ✅ | **SHIPPED** | `99781f5` |
| 3 | Sparse-depth supervision (SfM-point depth) | ✅ | **SHIPPED** | `5692d5b` |
| 4 | SplatSuRe selective super-resolution | ✅(gated) | **RESEARCHED + VALIDATED → no SR headroom for orbits; harvest the GATE** | `2f602b8` |
| 5 | SEVA (Stable Virtual Camera) wedge-fill (NOT EscherNet) + iFusion LoRA | ⚠️ plausible | NOT STARTED (gate to true gaps) | — |
| 6 | GSFixer A/B vs our Difix loop | — | NOT STARTED | — |

### SHIPPED this session — details, validation, how to use

**#0 — Sharpest-frame selection** · `core/frame_select.py` · commit `05657e3`
- `select_sharp_frames(paths, window=8, min_gap=3, keep_frac=1.0)` → picks the sharpest
  frame per non-overlapping window (var-of-Laplacian) while enforcing a min index gap so
  orbit COVERAGE is preserved. `extract_video_frames(video, out_dir, every)` for video in.
- Run BEFORE COLMAP. Validated on the 305 Laptop frames: culls severe blur (min var 8),
  selected-frame median sharpness **+34%** vs all-frame median. (Gain is modest on our
  near-still scans — median var 344, already crisp; the real gain is on genuinely
  motion-blurred handheld video.) CLI: `python core/frame_select.py <dir|video> [window]`.

**#2 — Object-mask alpha loss** · `core/training/gsplat_trainer.py` + `gsplat_backend.py`
· commit `99781f5`
- THE GAP IT FIXES (verified at source): stock masking pre-multiplies images to a black
  background but NEVER supervises rendered opacity — a black-but-opaque Gaussian
  "satisfies" a black bg pixel, so background floaters grow unpenalized.
- ADDS `L_bg = mean(alpha·(1−mask))` → pushes rendered opacity→0 OUTSIDE the object mask
  (deletes floaters). Optional `alpha_obj_lambda` pushes alpha→1 inside (fills see-through
  holes; use sparingly — can thicken thin parts).
- Plumbing: `gsplat_backend._export_masks()` writes `masks/` + `masks_{factor}/` (NEAREST)
  in `prepare_data`; trainer loads stem-matched masks, undistorts + ROI-crops them
  IDENTICALLY to the image (stays pixel-aligned), applies the loss.
- CLI: `--alpha_loss_lambda 0.1 --alpha_obj_lambda 0.0`. Default OFF.
- Validated (`_infusion/val_alpha_loss.py`): loader alignment on real data confirmed; a
  real Adam step drives background alpha **0.701 → 0.001** while object alpha is preserved
  (1.000). NOTE: validated for MECHANISM + integration + no-harm; the QUALITY lift only
  shows on scenes with visible background/floaters (our tightly-framed Laptop object fills
  ~100% of frame → almost no background to test on). Production A/B awaits a scene with
  real background.

**#3 — Sparse-depth supervision** · `core/training/gsplat_trainer.py` + `gsplat_backend.py`
· commit `5692d5b`
- Free + faithful: supervises rendered expected-depth against the SfM points each view
  observes. Per-view point observations are built once from the COLMAP point TRACKS
  (`load_colmap_data` → `data["point_obs"]`, ~1820 pts/view on Laptop). At train time the
  points are RE-PROJECTED with the view's own K+pose (so it matches the rendered grid
  regardless of undistortion), rendered depth is sampled there via `grid_sample`, robust
  L1 normalized by median depth (scale/resolution invariant).
- Grounds geometry where the photometric signal is ambiguous (under-textured / sparse-view
  captures). CLI: `--depth_lambda 0.1 --depth_start 2000`. Default OFF.
- Validated: integrates cleanly (57 it/s, no perf hit), A/B NEUTRAL on the texture-rich
  Laptop scene (22.05 vs 22.31 dB at 2500 steps — within short-run noise, as expected:
  a complete texture-rich scene doesn't exercise it). Its lift is scene-dependent →
  opt-in for the imperfect-capture case. SPARSE only; the mono-depth scale-invariant +
  inconsistency-masked prior (In Depth We Trust, 2604.05715) is a heavier follow-up.

**#4 — SplatSuRe selective super-resolution** · `_infusion/val_splatsure.py` · commit `2f602b8`
- RESEARCH VERDICT: a splat trained on overlapping views IS ALREADY a multi-frame
  super-resolver (sub-pixel fusion). Bolting a 2D single-image SR net onto each rendered
  view CORRUPTS multi-view consistency (per-view-different hallucination → conflicting
  gradients → mush) — now the published consensus. Faithful high-res = TRAIN AT TARGET RES
  + Mip-Splatting (BOTH ALREADY ON in our trainer). SplatSuRe's real contribution is a
  per-Gaussian GATE for where SR is even licensed. Paper code = Inria NON-COMMERCIAL
  (re-implement math; research-only is fine for us anyway).
- METRIC (re-implemented faithfully via gsplat `fully_fused_projection`, `eps2d=0` to drop
  the 0.3 AA dilation exactly as the paper specifies): per-view 3σ screen radius
  `r = 3·√(max eigenvalue of projected 2D cov)`; `ρ = r_max/r_min` over ≥3 observing views
  (else score 0); `score = σ((ρ−1.1)/0.05)`. Low ρ≈1 = undersampled = SR headroom; high ρ
  = some view saw it close = trust the fused splat.
- VALIDATED on GPU_V4 (2M-GS dense orbit): **100% well-sampled, 0% SR headroom; heatmap
  ALL BLUE.** Orbits self-supply multi-scale detail via BOTH distance variation (2.34×)
  AND viewing-ANGLE foreshortening — even a tight ±3% constant-radius band stays 99.5%
  well-sampled (ρ median 3.0). So selective generative SR adds ~nothing faithful for our
  capture style → CONFIRMS "per-scene ceiling reached" (matches the paper's own dense-
  trajectory MipNeRF360 result). The metric DOES discriminate (headroom 0.0→0.2→0.5% as
  the band tightened) — it's just that orbits are intrinsically well-sampled.
- ADOPTABLE WIN (not yet wired): fold the per-Gaussian ρ-score into the existing
  coverage-gated fusion / accept-reject as a SHARPER per-Gaussian "under-constrained"
  gate, replacing the coarse camera-hemisphere coverage gate. The 24.8% of Gaussians seen
  by <3 views = back/occluded = exactly where generative completion (Difix/SEVA/inpaint)
  is licensed. CLI to reproduce: `python _infusion/val_splatsure.py <ckpt.pt> <data_dir>
  [factor] [n_views] [dist_band]`.

### Validated capture insight (NEW, important for the playbook)
An ORBIT inherently provides multi-SCALE sampling of every surface patch — not only via
distance variation but via VIEWING-ANGLE foreshortening (a patch seen frontally projects
large, grazing projects small). This is WHY dense orbits leave almost no faithful SR/
detail headroom (the ρ analysis), and it reframes capture guidance: distance variation
helps but is not strictly required for multi-scale fusion; ANGULAR coverage already
delivers it. The genuine headroom is in regions seen by <3 views or only ever grazing/far
(back, top, occluded) — the coverage-tool / ρ-gate targets.

### Research synthesis (the strategic answer to "how do we drastically improve")
- **Motion deblur (#1) is the single biggest FAITHFUL lever** for orbit video; blur-lost
  detail is Cause-A (in the signal). Port BAD-Gaussians (arXiv 2403.11831, Apache-2.0,
  already gsplat-based): SE(3) cubic B-spline in-exposure camera trajectory + N-subframe
  averaging → synthesize the blurry image, supervise vs blurry input → underlying
  Gaussians stay sharp. Composes with our `pose_opt`. Alts: DeblurGS (2404.11358, robust
  to noisy COLMAP), CoMoGaussian (2503.05332, extreme blur). DON'T 2D-deblur-preprocess
  (multi-view-inconsistent, corrupts depth — 2510.01640).
- **EscherNet failure WAS a framing/normalization bug, confirmed (not a clean OOD wall).**
  But the better tool is **SEVA / Stable Virtual Camera (2503.14489, open, 576p, fits the
  4090)** — consumes COLMAP poses directly, no fragile object-crop. + iFusion (2312.17250)
  per-object LoRA for glossy OOD. PLAUSIBLE → gate to true gaps only.
- **GSFixer (2508.09667, open, DINOv2+VGGT geometry tokens)** benchmarks beat Difix3D+/
  GenFusion on sparse-view → A/B vs our Difix loop. Watch ArtiFixer (2603.00492, NVIDIA
  SIGGRAPH26, the real Difix successor, +1–3 dB, not-yet-open).

### What a successor should do next (priority order)
1. **Port BAD-Gaussians (#1)** — the big faithful lever. Needs: SE(3) B-spline trajectory
   module + N-subframe render+average in the train loop + REAL motion-blurred orbit video
   to validate (our existing scans are near-still and CANNOT prove a deblur win — capture
   a blurry handheld clip first). Multi-day build; deserves its own focused session.
2. **Wire the ρ-score gate** (#4 harvest) into `core/enhancement/autocomplete.py` /
   coverage-gated fusion — replace the hemisphere coverage gate with the per-Gaussian
   `ρ`/`<3-view` signal from `val_splatsure.py` (lift the metric into a reusable module).
3. **Production A/B for #2 and #3** on a scene with real background + under-textured
   regions (the Laptop/GPU scans don't exercise them) to measure the actual quality lift.
4. **#5 SEVA** for true unseen gaps (gated by the accept/reject machinery), **#6 GSFixer**
   A/B. Mono-depth prior as the heavier #3 follow-up.

### Reproduce / key paths
- Trainer: `core/training/gsplat_trainer.py` (CLI flags above) via `gsplat_backend.py`
  (subprocess; `get_default_config()` exposes `alpha_loss_lambda`, `alpha_obj_lambda`,
  `depth_lambda`, `depth_start`). Both default 0.0 → current production behaviour
  UNCHANGED until opted in.
- Validation scripts (in `_infusion/`, committed; binaries gitignored):
  `val_alpha_loss.py`, `val_splatsure.py` (+ heatmap `splatsure_heat.png`).
- Memory note: `imperfect-capture-pipeline.md` mirrors this chapter (build status inline).
- Datasets: GPU_V4 (`_dataset/GPU_V4/.../gsplat_prod/`, 2M-GS factor-2 30k ckpt),
  Laptop (`_dataset/Laptop/apex_images/training/gsplat_prod/`). NOTE: Laptop ckpts dir was
  cleared; only PLY remains. Real per-image masks are NOT preserved on disk (images are
  pre-multiplied black-bg) — `val_alpha_loss.py` derives silhouettes for testing only;
  production masks come from the SAM 2.1 masking stage.

### Honest limitations (do not over-claim)
- #2/#3 are validated for mechanism + integration + no-harm, NOT for production quality
  lift (our test scenes don't exercise their target regime). #4 is a "confirm + harvest the
  metric" result, not a new capability. #0's gain is modest on near-still input. The one
  genuinely transformative un-built lever is **#1 motion-deblur**, gated on capturing real
  blurry orbit footage.

---

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
