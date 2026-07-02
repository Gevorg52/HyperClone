# Apex Reality — The Clean-Splat Program (July 2026, rev 2)

**PRODUCT GOAL (user, 2026-07-01):** deliver **clean, polished, usable splats even when the capture
is imperfect** — lean on ArtiFixer-class fixers and augment/generate images, videos, or the Gaussians
themselves to fill holes and make flat surfaces polished instead of blocky. Plausible-but-consistent
fill is a FEATURE in under-observed regions, as long as gating guarantees it never corrupts real
observations. This supersedes rev-1's capture-first ordering (rev-1's evidence is retained below —
it's the context that makes the product bet sound).

Synthesis of a 5-agent primary-source research sweep (Meta Hyperscape · frontier recipe · init
evidence · video-diffusion rescue + QA · splat-space repair) × this project's validated learnings.
Tags: **[V]** verified vs primary source · **[PC]** paper-claim · **[U]** unconfirmed. RESEARCH-ONLY:
license noted per method, not a filter.

---

## 0. The two reframes that drive everything

**Reframe 1 — the metric.** We have judged every experiment by held-out-GT fidelity. The product
metric is *perceptual cleanliness + multi-view consistency*, where plausible fill is desired. This
invalidates at least one of our own verdicts: the ArtiFixer wedge "+4.0% only, plausible-not-faithful"
failure was scored against photons we never captured — we never measured whether the filled wedge
*looks clean and consistent*. The field's answer is a **visibility-split evaluation**: GT metrics only
where the capture observed; no-reference/plausibility + consistency metrics elsewhere (§6). Re-judge
past completion outputs under it.

**Reframe 2 — the moat.** Verified triple convergence (rev-1): Meta Hyperscape has **no secret
reconstruction algorithm** [V — absence of paper/patent/talk; ~80% guided capture + unconstrained
compute, quality visibly dropped when rendering moved on-device Nov'25]; the per-scene optimization
frontier is saturated (3DGS-MCMC — our base — still tops the only apples-to-apples board [V]); and
NTIRE-2026 RealX3D (33 teams) confirms the whole field rescues imperfect captures exactly the way we
do: **2D-restore → re-optimize, one winner on 3DGS-MCMC** [V]. So our stack is not behind — the
differentiator is **gating discipline + a defect-routed rescue pipeline**, not a missing algorithm.
"Clean" = real where observed, generated where not, seamless at the boundary; anyone can run a fixer
over everything, and it flickers and lies.

**Load-bearing house rules that survive the reframe (all validated here):**
- *Present-but-smooth vs genuinely-absent*: generate INTO emptiness; never over a present-but-smooth
  surface (invents contradicting texture → view-flicker → looks dirty, not polished). For observed
  flats the polished look comes from faithful smoothness/planarity, not generative texture.
- *Deblur converges BEFORE generative fusion* (CoherentGS premature-injection).
- *Never uniform per-frame 2D restoration before 3DGS* — now benchmark-documented, not just our
  finding (SplatSuRe premise; Deblur4DGS in-loop 27.16 vs restore-then-recon 26.06 [V]).

---

## 1. Pipeline v2 — the defect-routed rescue stack

**ingest → TRIAGE → robust base recon → faithful polish → gated completion → delivery + QA scorecard**

Triage auto-diagnoses each capture (we already own most diagnostics: `blur_report`, `coverage.py`,
ρ-gate, sharpness maps, exposure drift) and routes which rescues fire:

| Capture defect | Faithful rescue (first) | Generative rescue (gated, second) | Status / adoption |
|---|---|---|---|
| Motion blur / defocus | **in-loop deblur** — BAD-Gaussians port, upgrade path CoMoGaussian (2503.05332, neural-ODE) [V]; GeMS recipe (VGGSfM poses + MCMC init) when blur kills COLMAP [V, no code] | ArtiFixer3D sharpen after | ❌ biggest unbuilt; needs blurry clip |
| Exposure / WB drift | **PPISP** (NVIDIA 2601.18336, **Apache, merged gsplat PR #858**): per-frame exposure/WB/vignetting/CRF + novel-view ISP controller; 24.12 vs bilateral-grid 21.41 [V] | — | ❌ adopt; sm_120 rebuild of its kernel |
| Noise / low-light | pre-restore + cross-view harmonization (RealX3D winner pattern [V]); Luminance-GS as in-loop net (code [V]) | — | ⚠️ |
| Low-res frames | **SplatSuRe selective SR** (2512.02172, code [V]; foundation of NTIRE'26 SR winner) — we already validated the ρ-gate | — | ✅ gate built; SR injection unbuilt |
| Flat/textureless "blocky" | EDGS dense triangulation + masked mono-depth (MoGe-2) + **PlanarGS losses** (2510.23930, **MIT**, code [V]) or 3DGS-Flats MCMC-relocation-to-planes (2509.16423, Inria-NC → reimplement; built ON MCMC [V]) + eff-rank reg (2406.11672) | gated re-texture only if truly absent | ⚠️ half-built (§3) |
| Under-observed mush | EDGS points + depth prior + distill | ArtiFixer3D (−10.8%, adopted) / Difix-v2 (prod) | ✅ strong |
| Holes / unseen back | — (nothing faithful exists) | **completion lane §4** (GENA3D / causal ArtiFixer / G4Splat pattern / autocomplete) | ⚠️ core product gap |
| Floaters / needles | prune.py + alpha-mask loss + eff-rank; **SplatTransform** filters (MIT [V]) | — | ✅ strong |
| Aliasing / popping | Mip-Splatting ✅ + SA-GS test-time (2403.19615, code [V]) | — | ✅ good |
| Glossy smear | Ref-DGS (2603.07664) class backend; cross-pol capture guidance when user can re-shoot | — | ❌ subject-gated |

**Delivery pass (product-shaped, mostly off-the-shelf):** SplatTransform (`--filter-floaters`,
cluster/island removal, training-free merge-decimation, LOD, **SOG** export; MIT, v2.7.1 [V]) →
2–5k-step self-refit on original views (PUP/LightGaussian recovery pattern) → SPZ (MIT) / SOG.

---

## 2. The universal gate (the moat, made concrete)

Port one pattern everywhere: **GaMO's opacity-mask latent blending** (2512.25073, code [V]) — the
splat's own rendered opacity marks unobserved areas; during diffusion denoising, real-content latents
are re-injected at observed pixels so generation is *structurally* unable to touch evidence. This is
ArtiFixer's opacity-mixing, GENA3D's geometry-gated attention, RnG's reconstruction-guided causal
attention, and VidSplat's confidence-weighted loss — the 2026 winners all converged on our
RESEARCH_VISION thesis [V]. Our implementation assets: rendered-alpha, coverage hemisphere, ρ<3-view
gate, WarpRF, accept/reject-vs-real-photos. Unify as ONE confidence field consumed by every
generative stage, and label fill in the deliverable.

---

## 3. "Polished, not blocky" — the flat-surface package (faithful, local, buildable now)

The mush/blocky look on flat/dark surfaces decomposes into **starved seeds** + **ill-conditioned
photometric objective** + **needle/blob artifacts**; fixes respectively:
1. **EDGS-style dense-matcher triangulation** (2504.13204, CVPR'26 [V]: MipNeRF360 LPIPS 0.215→0.141)
   into our EXISTING COLMAP model — HLoc native (`match_dense` LoFTR/RoMa → `triangulation` with
   reference_model; poses untouched) [V]. Seed only starved regions, certainty-thresholded (random
   injection HURTS, −0.2..−0.7 dB [V]). NOTE [V]: wholesale feed-forward init REFUTED (VGGT-X −1.5 dB
   vs COLMAP on dense captures) — augment, don't replace.
2. **Masked dense mono-depth (+normal) supervision** — In-Depth-We-Trust recipe (2604.05715 [V],
   +0.3–0.66 dB, can't corrupt textured regions) with **MoGe-2** (MIT) prior; extends our shipped
   sparse `--depth_lambda`. Normals: StableNormal (Apache) / Metric3D-v2 (BSD-2); DN-Splatter (Apache)
   as reference implementation.
3. **Planar prior** — port PlanarGS's two losses (MIT, code) into our trainer; or reimplement
   3DGS-Flats' plane-snapping relocation (it literally modifies MCMC relocation [V]). Honest negative
   [V]: no published post-hoc plane-fit-and-resplat exists; in-training is the proven route.
4. **Effective-rank reg** (one loss term; kills needles; also directly attacks our giant-blob 8M
   divergence mechanism → re-opens the >3M "master splat" test with retuned regs).

---

## 4. The completion lane (holes, backs, wedges) — ranked adoptions

1. **Causal ArtiFixer unlock** — our n=6 causal data (−12.9% class) + the paper's headline say causal
   AR is the real gap-filler; bidirectional (our sm_120 workaround) is weak exactly there (+4.0%
   wedge). **De-risk on an H100 spot for <$1/run** (FA2/3 just works on sm_90) BEFORE deciding on the
   local cuDNN-SDPA build (feasibility already proven: causal+window correct on sm_120; only the
   attention-sink remains — two-pass LSE merge / 9.23 block-mask / window-only test).
2. **GENA3D** (né AmodalGen3D, ECCV'26, 2511.21945 — **inference code + weights landed 2026-06-27**
   [V], license unstated): object back-completion from 1–4 unposed occluded views, geometry-gated
   (π³-MVS point cloud modulates attention). Successor of our Amodal3R/TRELLIS lane; FID 32.12 GSO@2v
   beats TRELLIS/Amodal3R [V]. Fits 96 GB. → Install, run on our GPU/Laptop backs, judge under §6.
3. **RnG** (CVPR'26 Highlight, code+3.5 GB weights [V]): reconstruction-guided causal attention —
   cleanest published "generation cannot touch observed evidence"; <1 s/object; RGBD out → distill.
4. **Scene-level holes:** **G4Splat** (ICLR'26, **Apache**, code [V] — unposed video, plane-prior
   depth, visibility-mask-gated inpainting, reports unobserved-region metrics separately) ·
   **S2C-3D** (SIGGRAPH'26, code [V] — scene-specific diffusion finetuned on inputs+degraded renders).
   Pattern to port regardless: GSCompleter's "insert only NEW primitives into voids" [V abs].
5. Our existing `autocomplete.py` (InFusion seed + accept/reject + production-trainer reconstruction,
   validated 23.5→32.6 dB on a real hole) remains the faithful-reconstruction arm; background-freeze
   TODO becomes production-relevant.
6. **Virtual re-capture (strategic bet, not a quick adopt):** **FlowR** (ICCV'25 Highlight,
   **Apache**, 2504.01647 [V]) is our product loop as one model — render flawed views → flow-match
   directly to "what a dense capture would have seen" → retrain on real+corrected. Caveats: public
   release is an SD3 retrain, pretrained weights unconfirmed → budget training [V]. Cheaper prototype
   first: **VidSplat recipe** (SIGGRAPH'26, 2605.11424 [V]) is *training-free on stock Wan2.1 I2V* +
   per-pixel point-fusion confidence gating — and our WSL ArtiFixer env already runs the Wan stack.
   Negative [V]: no CAT4D/SEVA/Gen3C successors exist; ReCamMaster open ckpt verifiably weaker than
   its paper; skip FlashVSR on sm_120 (sparse-attention kernel class).

---

## 5. Splat-space ("generate the Gaussians themselves") — honest state + flywheel v2

- **No released system repairs a real captured splat natively in 3D** [V — 3 agents independently].
  First movers: **GaussianGPT** (ECCV'26, TUM, **MIT, code+ckpts** [V]) — AR transformer over VQ
  Gaussian tokens with prefix-conditioned completion (`complete_chunks.py`); synthetic-indoor domain →
  fine-tune territory. **CompleteSplat** (Niantic, 2508.21542, real-capture splat diffusion) has no
  code yet — watch. For objects TODAY: **TRELLIS v1 SLAT inpainting** (MIT, RePaint-style region
  inpaint conditioned on observed regions [V]) + Amodal3R → GENA3D (§4).
- **Flywheel verdict — core bet as-designed is REFUTED at our scale** [V]: nobody diffuses raw
  Morton-serialized attribute grids; Gaussian Atlas needed **205,737 fitted objects**; L3DG ablation
  shows near-raw-parameter diffusion collapses (FID 197 vs 8.5 with learned latent); a 59-channel
  1280² tensor has no pretrained prior to transfer. (The 1280²=1.64M-slot vs 3M-Gaussian accounting
  is fine — ~58% of a 3M MCMC model is dead/low-opacity, matching prune-then-encode.)
- **Flywheel v2 (proven-scale path):** (a) keep the Morton codec for storage/compression
  (Self-Organizing-Gaussians-class, 17–42× [V]); (b) **~100–300 scans is enough to fine-tune our OWN
  render-space fixer** — Difix3D+ itself trained on 112 DL3DV scenes [V]; a fixer tuned on OUR defect
  distribution is the realistic data moat; (c) bootstrap latent/splat-native work from public corpora
  — **SceneSplat-7K (7,916 pre-fitted real splat scenes, downloadable [V])** + DL3DV-10K; (d)
  object-category fine-tunes of TRELLIS-class generators are proven at 300–2k curated assets [V].

---

## 6. QA harness — the operational definition of "clean and usable" (all released code)

Visibility-split scoring replaces bare held-out LPIPS:

| Role | Metric | Code/License |
|---|---|---|
| Observed-region fidelity | PSNR/SSIM/LPIPS **masked by our coverage/visibility field** | ours |
| GT-free per-pixel quality map | CrossScore | BSD-3 [V] |
| Unobserved-region plausibility | NOVA (WACV'26, non-aligned-reference) + FID vs scene frames | MIT [V] |
| Cross-view consistency | **MEt3R** (de-facto 2025-26 standard) | MIT [V] |
| Cheap geometric CI gate | TSED | MIT [V] |
| Temporal flicker | VBench `temporal_flickering` + RAFT warp-error | Apache [V] |
| Whole-scene no-reference score | NVS-SQA (TPAMI) | CC0 [V] |
| Artifact heatmaps (9-type taxonomy) | **ArtifactWorld inference — now partially RELEASED** (2026-06-24, LTX-Video two-stage + LoRA weights) | LTX license [V] |

Cautions [V]: MLLM judges near-random at artifact localization (Artifact-Bench) — coarse ranking
only; pyiqa is PolyForm-NC. First action: **re-judge the ArtiFixer wedge + autocomplete outputs**
under this suite — some recorded "failures" are likely product wins.

---

## 7. THE PROGRAM (product-first priority)

**Tier 0 — QA harness + re-judgment.** Build §6 (days; all code released). Re-score wedge/completion
experiments. Recalibrates every decision after it. Also wire the scorecard into Stage-10.

**Tier 1 — Causal unlock + universal gate.** H100 spot causal ArtiFixer (<$1) → settles the
gap-filling ceiling and the authors' headline on OUR data; decide local cuDNN build after. Port the
GaMO opacity-mask gate as the single confidence field all generative stages consume (§2).

**Tier 2 — Flat-surface polish package (§3).** EDGS triangulation → masked MoGe-2 depth → PlanarGS
losses → eff-rank. Validates on EXISTING scans (Laptop deck 0.16× starvation is the testbed). This is
the literal "polished vs blocky" deliverable, all faithful, no new data needed.

**Tier 3 — Robust ingestion.** PPISP into our trainer (gsplat PR #858 exists — port + sm_120 rebuild)
· in-loop deblur port (BAD-Gaussians → CoMoGaussian; GeMS recipe for blur-killed SfM) — **still needs
one real blurry orbit clip** · SplatSuRe SR injection behind our validated ρ-gate.

**Tier 4 — Completion lane (§4).** GENA3D + RnG installs for object backs; G4Splat-pattern
visibility-gated inpainting for scene holes; background-freeze in autocomplete; judge everything
under Tier-0 metrics, label fill in deliverables.

**Tier 5 — Delivery pass.** SplatTransform + self-refit + SA-GS + SOG/SPZ export; define the
per-deliverable QA scorecard as the product's acceptance test.

**Tier 6 — Strategic bets (gated on Tiers 0–4 landing).** VidSplat-recipe prototype on our existing
Wan runtime → FlowR trained properly if the prototype pays · flywheel v2: start banking every scan
now (corpus meter to 100–300) for the fixer fine-tune; GaussianGPT/SceneSplat-7K research track.

**Kept from rev-1 (demoted, not dead):** coverage → "shoot these angles" report ships as user
guidance when re-capture IS possible; capture-protocol doc + casual-vs-protocol decomposition
experiment when convenient; glossy backend subject-gated; frontier trainer A/Bs (Opt3DGS 2511.13571
[PC, test skeptically], NASGabor 2606.09794, SA-GS) as cheap background experiments.

**Skip list (verified dead ends):** wholesale feed-forward init · uniform per-frame restoration/SR ·
Morton-image diffusion at our corpus scale · waiting for CAT4D/SEVA/Gen3C successors (none exist) ·
FlashVSR + stock flash-attn-4 on sm_120 · WildGaussians-class appearance for mild video drift ·
MLLM artifact judges · bare post-hoc RANSAC plane-snap (measured inaccurate).

---

## 8. Key sources (new in rev 2)
Rescue/QA: FlowR 2504.01647 · VidSplat 2605.11424 · PR-IQA 2604.04576 · OracleGS 2509.23258 ·
ExploreGS (code) · GENA3D 2511.21945 · RnG 2603.01194 · G4Splat 2510.12099 · S2C-3D · GaMO 2512.25073 ·
GSCompleter 2604.20155 · GaussFusion 2603.25053 · Deblur4DGS · CoMoGaussian 2503.05332 · GeMS
2508.14682 · SplatSuRe 2512.02172 · Sequence-Matters 2412.11525 · RealX3D 2604.04135 · SeedVR2
(ByteDance-Seed/SeedVR) · PPISP 2601.18336 (gsplat PR #858) · fused-bilagrid (PR #706) · CHROMA
2507.15748 · CrossScore · NOVA · MEt3R · TSED · VBench · NVS-SQA · ArtifactWorld 2604.12251 (inference
released) · Artifact-Bench 2605.18984. Splat-space: GaussianGPT 2603.26661 · CompleteSplat 2508.21542 ·
TRELLIS 2412.01506 · Amodal3R 2503.13439 · DiffGS 2410.19657 · ReSplat 2510.08575 · AnySplat ·
PlanarGS 2510.23930 · 3DGS-Flats 2509.16423 · SplatTransform (PlayCanvas) · PUP · NanoGS · SPZ ·
Gaussian Atlas 2503.15877 · L3DG 2410.13530 · Can3Tok 2508.01464 · SceneSplat-7K (HF) · Z-Order
Transformer 2605.13465 · Hunyuan3D-Omni · ShapeR 2601.11514. (Rev-1 sources retained in git history;
init/trainer/capture citations unchanged: EDGS 2504.13204 · VGGT-X 2509.25191 · In-Depth-We-Trust
2604.05715 · MoGe-2 2507.02546 · eff-rank 2406.11672 · Opt3DGS 2511.13571 · NASGabor 2606.09794 ·
SA-GS 2403.19615 · Robust-GS 2404.04211 · Hyperscape sources · RawNeRF/HDRSplat/OLATverse/Lepidoptera.)
