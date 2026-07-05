# Apex Reality — Blackwell-Era Frontier & Plan (June 2026)

**Trigger:** hardware upgraded **RTX 4090 (24 GB) → Blackwell RTX PRO 6000 (96 GB)** — the long-
planned "production" GPU. Goal: push toward *perfect* object Gaussian splats. This doc is the
research synthesis (cutting-edge since the June rev-5 roadmap) + the prioritized plan. Research
was a multi-agent, primary-source-verified sweep (arXiv/GitHub/HF, adversarial cross-checks).
Tags: **[V]** verified vs primary source · **[PC]** paper-claim (authors' own number) · **[U]** unconfirmed.

RESEARCH-ONLY: license is NOT a selection filter; noted per method for our future clean-licensed OSS port.

---

## 0. HEADLINE — ArtiFixer (NVIDIA) is REAL, OPEN NOW, and is the centerpiece adoption

**ArtiFixer: "Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models."**
arXiv **2603.00492** [V], NVIDIA Spatial Intelligence Lab + Toronto AI Lab (de Lutio, Fischer, Ren,
Gojcic, Turki, … — **same group as Difix3D+**), **SIGGRAPH 2026**. It is the genuine Difix3D+ successor.

- **What it is [V]:** an **autoregressive VIDEO-diffusion** scene enhancer/extender built on **Wan2.1-T2V-14B**
  (also a 1.3B variant). Two phases: (I) finetune a bidirectional teacher with **"opacity mixing"** —
  start denoising from the existing *render* where rendered **opacity** says geometry exists, and from
  pure **noise** where it's empty → fidelity where observed, free generation where under-observed.
  (II) distill the teacher into a **causal 4-step AR generator** (Self-Forcing DMD) → hundreds of
  trajectory frames in one pass. Targets the exact extrapolation failure modes we fight: floaters,
  holes, under-observed/extrapolated views, blur, ghosting.
- **Three variants [V]:** `ArtiFixer` (2D AR views, sharpest) · **`ArtiFixer3D`** (distills generated
  views back into a fresh **3DGRUT** optimization on real anchor + generated views — *a real 3D distill
  loop, exactly what we hand-rolled*) · `ArtiFixer3D+` (re-apply AR as post-process; the Difix3D **+** analogue).
- **Benchmarks [PC]** (numbers cross-checked: GSFixer/Difix3D+ baseline rows match those papers' own results):
  Nerfbusters ArtiFixer3D **20.24 dB / FID 39.67** vs Difix3D+ 18.51 / 41.77 → **+1.73 dB**;
  DL3DV **20.14 dB / FID 24.27** vs Difix3D+ 17.99 / 40.86 → **+2.15 dB, FID ~halved**;
  beats CAT3D / GSFixer / GenFusion / Gen3C on sparse Mip-NeRF360. The "+1–3 dB over SOTA" claim holds.
- **Release [V]:** code **github.com/nv-tlabs/artifixer (Apache-2.0)**; weights **HF nvidia/ArtiFixer**
  `artifixer-14b.pt` (67.6 GB ckpt, posted 2026-06-04), **NVIDIA Non-Commercial weights license** (fine for
  research; reimplement clean for OSS). Base Wan2.1 = Apache.
- **Inputs [V]:** ingests a **COLMAP scene** `<scene>/images + sparse/0/` (cameras/images/points) — **exactly
  our Stage-7 export** — or trajectory JSONs (intrinsics + 4×4 c2w). Docker envs (`Dockerfile.cuda12/cuda13`).
- **Throughput [PC]:** 14B 8.36 FPS · 1.3B 34.38 FPS · ArtiFixer3D 268 FPS (pure 3DGRUT render after distill).
  Trained on 128×H100. Inference VRAM not stated [U]; 14B bf16 fits comfortably in 96 GB. **The 14B model is
  the payoff of the hardware upgrade — it was out of reach on the 24 GB 4090.**
- **Why it matters for us:** it is the validated, far-stronger version of TWO things we built by hand — the
  Difix **distill-back loop** AND the **auto-complete / coverage-gated generative fusion** (opacity mixing =
  principled "real where photons constrain, generative where they don't, never in conflict" — our exact thesis,
  RESEARCH_VISION.md). **Adopt + benchmark head-to-head vs our distill-v2** rather than keep reinventing.
  Most portable single idea even if we don't adopt wholesale: **opacity-mixing initialization** drops into any
  distill-back loop.

**Alternatives / complements to ArtiFixer (all verified, for our distill-back stack):**
| Method | arXiv | vs Difix3D+ [PC] | Release | License | Note |
|---|---|---|---|---|---|
| **HAD** | 2605.16873 | +0.54 dB post; learned per-pixel **hallucination mask** | code+ckpt [V] | CC-BY-NC-ND | Cheapest high-leverage add-on: masks hallucinated px out of the 3DGS loss; layers on ANY fixer incl. ArtiFixer |
| **FreeFix** | 2601.20857 | wins LLFF/Mip360 | code [V] | **MIT** | **Fine-tuning-free** (frozen Flux/SDXL) + per-pixel confidence — cleanest OSS-port baseline |
| **GSFixer** | 2508.09667 | +2.58 dB (DL3DV-Res) | code+weights [V] | none (ARR) | DINOv2+VGGT geometry-token conditioning — mine the idea; bad license |
| **Difix3D+** | 2503.01774 | (baseline) | code+weights [V] | NVIDIA NC | our current native backend — keep as the reference everyone is measured against |
| **NuRec Fixer (Cosmos 0.6B)** | — | fastest (~27 ms) | HF nvidia/Fixer | **NVIDIA Open (commercial OK)** | keep as fast sharpness-gated first pass |
| **ArtifactWorld** | 2604.12251 | dataset + 9-type taxonomy + golden bench | **NOT yet open** [V] | — | grab the eval harness the moment it drops |

---

## 1. What 96 GB actually unlocks (ranked by *fidelity* gain, not speed)

1. **Native full-resolution training — HYPOTHESIS FALSIFIED (naive config). ⚠️ See empirical update below.**
   The theory: `factor 2` was a 24 GB compromise; on 96 GB train native (`--data_factor 1`; our loader has
   NO 1600px cap — that stock-3DGS issue doesn't apply to us). **Judge by LPIPS at a FIXED eval res** — PSNR
   isn't comparable across resolutions.

   > **⚠️ EMPIRICAL (2026-06-29, GPU_V4, fair same-split eval):** a naive native-24 MP run (`--data_factor 1
   > --coarse_factor 2 --coarse_until 15000 --cap_max 3M --pose_opt --lpips_gated 0.2`, 30k) **REGRESSED ~4 dB
   > vs the validated factor-2 recipe at BOTH eval resolutions** — at 24 MP: factor2 **25.0 dB / 0.495 LPIPS**
   > vs native 21.0 / 0.513; at 6 MP: 25.3 / 0.413 vs 21.1 / 0.460. The factor-2 (6 MP-trained) model wins
   > even when both render at 24 MP. Native train≈27 / test≈21 = **~6 dB train/test gap = heavy held-out
   > overfitting** (the flat ~21 dB at both res + only mild LPIPS gap = GEOMETRIC/pose misalignment, not an
   > appearance deficit). Leading cause: **pose_opt overfits train poses harder at 24 MP** (4× sharper
   > photometric gradient → train-pose drift → unoptimized test poses render misaligned); secondary:
   > under-convergence at 4× pixels in 30k steps. **Native-res is NOT a free win** — it must be paired with a
   > fix for the pose/convergence interaction. Isolating experiments to run before adopting native-res:
   > (a) native **without** pose_opt; (b) native + test-time pose alignment in eval; (c) native at 60–100k
   > steps; (d) match cap (3M vs 2M) to remove that confound. Until then **factor-2 remains the validated
   > production recipe.** Vindicates the research caveats ("capture/optimizer, not res, is the ceiling").
   >
   > **✅ RESOLVED (2026-07-05, WP-N1 isolation, GPU_V4, `_infusion/wpn1_RECORD.md`):** ran isolation (a)
   > native **without** pose_opt vs native **with** pose_opt, both = the WP-S1 stacked recipe at factor-1
   > (c2f 2→1, cap 3M, seed 0, dense-init + mono-depth[factor-1 cache] + erank + gated-LPIPS). **Root cause
   > CONFIRMED = pose_opt, NOT under-convergence NOR resolution.** N-b (pose_opt ON) reproduced June exactly:
   > held-out 20.76 dB, **train/test gap 6.18 dB** (train-loop-refined 26.94 / test 20.76). N-a (pose_opt
   > OFF) eliminated the gap (**0.05 dB**) → test 26.08 dB, i.e. **+3.47 dB common-res PSNR and +0.04 SSIM
   > OVER the factor-2 F2 stack**, and won ALL native-res microstructure crops (region-LPIPS −2.8…−4.9%;
   > fins/etched-traces visibly crisper). The stack alone did NOT fix it — **pose_opt-OFF is required at
   > native res.** Train PSNR was HIGH (26–27) both arms ⇒ NOT under-convergence ⇒ the 60k extension (c) is
   > unwarranted. **BUT the gate is NOT passed:** the perceptual win is **resolution-matched** — N-a LOSES
   > common-res (factor-2) masked-LPIPS to F2 (0.432 vs 0.410, both A=direct-f2 and B=render@f1→downsample
   > agree), because F2 is gated-LPIPS-tuned at factor-2. Net: **native-res + pose_opt-OFF is validated for
   > NATIVE 24 MP detail delivery, NOT as the factor-2 production default** (F2 still best at factor-2
   > perceptual). VRAM: native 24 MP @ 3 M = only ~28 GB (96 GB → not the limiter). factor-2 stays prod.
2. **The "3M ceiling" was a 24 GB + untuned-reg + Gaussian-EXTENT artifact, NOT a hard quality ceiling.**
   Mechanisms of our 8M divergence/OOM, all confirmed: (a) **giant/elongated Gaussians blow up tile-
   intersection buffers** → the 33 GB OOM (bound *scale*, not just count); (b) **MCMC SGLD noise ∝ covariance**
   violently displaces big low-opacity Gaussians each step (`noise_lr` default 5e5) → high-count instability;
   (c) gsplat MCMC **relocates rather than prunes** dead Gaussians → ~75% dead at cap 5M (issue #822). We
   never retuned `scale_reg`/`opacity_reg` (both 0.01, tuned for rooms not a multi-M single object).
   **→ re-test >3 M as a CONTROLLED experiment** (raise scale_reg + max-scale/anisotropy penalty, raise
   opacity_reg, lower noise_lr, DashGaussian resolution-synced budget). **Honest expectation: modest gain
   on a single orbit object** — literature shows single objects saturate below millions; native-res is the
   bigger lever. Judge by LPIPS at fixed eval res.
3. **Bigger generative models resident locally** — the 14B ArtiFixer, large fixers, hi-res SEVA/Gen3C,
   ReconViaGen + TRELLIS.2. Out of reach at 24 GB; routine at 96 GB.
4. **Deblur subframe count** — BAD-Gaussians' N-subframe blur model scales VRAM ~linearly in N; push
   N=10→20–40 for finer trajectory integration = more faithful blur recovery, alongside a larger budget.
5. **Glossy/specular pipelines** (compute/memory-heavy, big per-content gains): **Ref-Gaussian** (2412.19282,
   ~+4–5 dB on Shiny Blender [PC]), **Spec-Gaussian** (2402.15870, ASG+tiny MLP, +3.88 dB anisotropic [PC]),
   **3DGS-DR** (2404.18454). Our subjects (GPUs, laptops, statues) ARE glossy/semi-gloss → relevant; only for glossy.

**Do NOT spend VRAM on:** higher SH degree (diminishing; field moved to ASG/neural appearance); Scaffold/Octree/GES
(these *save* memory); per-frame super-resolution of inputs (view-inconsistent — we own real 24 MP pixels, native
training strictly dominates).

---

## 2. Blackwell BUILD requirements (BLOCKS all GPU work — do first)

Our current AOT build targets **sm_89 / CUDA 12.1 / torch 2.3** — that **will not run on the new card.**
The Blackwell RTX PRO 6000 is **GB202, compute capability 12.0 / sm_120** (same arch as RTX 5090, **not** the
old sm_89 "RTX 6000 Ada"). All [V] vs NVIDIA CUDA GPU table + gsplat issues #855/#809 + PyTorch maintainers:
- `TORCH_CUDA_ARCH_LIST="12.0"` (or `12.0+PTX`). **CUDA Toolkit ≥ 12.8** (first to emit native sm_120 cubins;
  12.6 fails → silent fallback to sm_90). **PyTorch ≥ 2.7 cu128** (`--index-url .../whl/cu128`); prefer current
  2.8/2.9 cu128. Windows cu128 wheels exist (the old "use WSL2" advice is outdated).
- Rebuild **gsplat AOT** from source in a VS2022 `vcvars64.bat` dev shell (same recipe as DEVELOPER_NOTES §2,
  swap arch/CUDA/torch). gsplat rasterizer is **pure FP32** → Blackwell FP4/FP8/Tensor-Cores are irrelevant
  to our path; what matters is the 96 GB + high FP32 throughput.
- Re-verify the whole stack on the card: pycolmap 4.0.4, SAM 2.1, BiRefNet, the difix isolated env, FLUX.1-Fill.

---

## 3. Faithful detail recovery (the biggest UNBUILT lever, carried over from rev-5)

**Motion-deblur for 3DGS — port BAD-Gaussians (2403.11831, Apache, gsplat/nerfstudio-native, ECCV'24) [V].**
Faithful (blur-lost detail is Cause-A, in the signal). Lowest-friction port: its in-exposure **SE(3) cubic
B-spline trajectory + N-subframe averaging** is a *superset of our existing `pose_opt`* — extend each camera's
single pose into a per-exposure spline of poses, render N subframes through our MCMC Gaussians, average,
supervise on the blurry frame. Composes with MCMC + Mip-Splatting (already on). **Reconcile with pose_opt** so
the two BA loops don't fight (spline owns intra-exposure motion; pose_opt owns inter-frame pose).
- Upgrade path if B-spline underfits handheld motion: **CoMoGaussian** (2503.05332, MIT, ICCV'25, neural-ODE
  continuous trajectory; INRIA backend = harder port). DeblurGS (2404.11358) for very noisy COLMAP init.
- **Process order, CONFIRMED [V]:** do NOT 2D-deblur-preprocess (multi-view-inconsistent, corrupts depth —
  2510.01640 / 2512.10369); deblur **jointly** in the train loop; deblur must **converge BEFORE** any
  generative fusion (CoherentGS 2512.10369 "Premature Injection" ablation: an early prior mistakes blur for
  low-freq texture and **solidifies** it — its 1,500-iter deblur warm-up before injecting the prior is the
  strongest published statement of our ordering rule).
- **Needs real blurry orbit footage to validate** — our existing scans are near-still and CANNOT prove a deblur
  win. Capture a handheld clip first.

**Frame selection (fixes the known Stage-1 video-cull bug):** windowed **variance-of-Laplacian computed INSIDE
the foreground mask** + dense-optical-flow motion gating, **relative** (per-window max), never an absolute
threshold (VoL isn't comparable across content/noise — Pertuz 2013) [V]. Learned NR-IQA has no proven
downstream-3DGS advantage over masked windowed VoL — don't bother yet.

---

## 4. Unseen-region completion (the back/top of a single-elevation orbit)

**Honest ceiling:** no released method *faithfully* reconstructs a never-photographed surface — it's all
plausible-only; the engineering goal is to **gate** it so it never overwrites real observations.
- **ArtiFixer's opacity-mixing already does this** (generate from noise only where opacity≈0) → it is our
  primary unseen-region tool now, subsuming much of auto-complete + coverage-gated fusion.
- Structured-generator alternative (faithful-where-seen, mesh out): **ReconViaGen** (2510.23306, **MIT**, ≤24 GB)
  — only released structured generator that ingests real per-view extrinsics+intrinsics and is reconstruction-
  anchored. **Amodal3R** (2503.13439) for heavy occlusion.
- Pose-conditioned view-synth alternative: **SEVA / Stable Virtual Camera** (2503.14489, non-commercial) ingests
  our `transforms.json`; **OracleGS** (2509.23258) is a published SEVA+VGGT-uncertainty-gated-into-3DGS loop.
  **Gen3C** (NVIDIA, commercial-OK, TSED-proven 3D consistency) but scene-scale.
- Gating machinery (real views ALWAYS full unmodulated loss): **GenFusion** (2503.21219, MIT — hard coverage
  mask: opacity `T<τ` OR depth-mismatch) most deployable; **HAD** learned hallucination mask the cleanest 3DGS
  gate; **WarpRF** (2506.22433) training-free reliability. Wire these (+ our coverage tool + the SplatSuRe
  ρ<3-view gate from rev-5) as the confidence field.

---

## 5. Backend: adopt 3DGRUT / 3DGUT

- **3DGUT** (2412.12507, CVPR'25, **already upstreamed into gsplat, Apache**) — PSNR-neutral vs 3DGS, better
  SSIM/LPIPS, handles distorted/rolling-shutter cameras + secondary rays at **zero quality cost**. Adopt via gsplat.
- **3DGRUT** (`github.com/nv-tlabs/3dgrut`, **Apache-2.0**, Windows + Blackwell/sm_120 via CUDA 12.8) bundles
  ray-tracing (3DGRT) + 3DGUT. **ArtiFixer3D distills into 3DGRUT** → adopting it aligns our backend with the
  ArtiFixer adoption. 3DGRT (RT-core, slower) only if the subject is reflective.
- Targeted popping/AA (if needed, verify Windows/Blackwell per-repo): StopThePop, EVER (2410.01804),
  AAA-Gaussians; 3DGEER (2505.24053, gsplat-integrated) for fisheye/distorted.

---

## 6. PRIORITIZED PLAN (smart-scientist ordering)

**Tier 0 — Foundation (blocks all GPU work):**
- [ ] Rebuild env for Blackwell: gsplat AOT `arch=12.0`, CUDA ≥12.8, torch ≥2.7 cu128; verify whole stack on card (§2).

**Tier 1 — Faithful wins:**
- [x] ~~Native 24 MP training (the "certain win")~~ **DONE + ISOLATED (WP-N1, 2026-07-05).** Naive config
      regressed ~4 dB; **root cause = pose_opt overfitting train poses at 24 MP (6.18 dB train/test gap),
      NOT resolution/convergence.** With **pose_opt OFF** (N-a), native beats the factor-2 stack by +3.5 dB
      PSNR/+0.04 SSIM and wins all native microstructure crops — but LOSES factor-2 masked-LPIPS (resolution-
      matched perceptual). **factor-2 stays prod; native+pose_opt-OFF validated for NATIVE 24 MP detail
      delivery only.** Never use pose_opt with native-res training. (§1.1, `_infusion/wpn1_RECORD.md`.)
- [ ] **Fix frame-selection / video-cull bug** (masked windowed VoL + motion gate, relative) (§3).

**Tier 2 — Headline adoption (highest strategic leverage):**
- [ ] **Adopt ArtiFixer** (3DGRUT backend + ArtiFixer3D / 3D+). Ingests our COLMAP export directly. Benchmark
      head-to-head vs our distill-v2 on the held-out wedge. Steal opacity-mixing for our loop regardless (§0, §5).
- [ ] Add **HAD** hallucination-mask as a cheap reliability gate on top of whatever fixer we run (§0).

**Tier 3 — Biggest unbuilt faithful lever (needs real data):**
- [ ] **Port BAD-Gaussians** motion-deblur into the gsplat loop; reconcile with pose_opt; push N-subframes on
      96 GB. **First capture a real handheld blurry orbit clip to validate** (§3).

**Tier 4 — Controlled experiments / conditional wins:**
- [ ] **Re-test >3 M Gaussians** as a controlled experiment (retune scale/opacity-reg + anti-blob + noise_lr +
      DashGaussian schedule). Expect modest single-object gain (§1.2).
- [ ] **Glossy/specular backend** (Ref-Gaussian / Spec-Gaussian / 3DGS-DR) — only for glossy subjects (§1.5).
- [ ] **Unseen-region completion**: ReconViaGen / SEVA + GenFusion/WarpRF gating, if ArtiFixer's coverage extension
      proves insufficient on a true single-elevation orbit (§4).

**Strategic note:** ArtiFixer significantly overlaps/supersedes our home-grown distill-back loop, auto-complete-
holes, and coverage-gated fusion. This is an **adopt-and-benchmark, don't-keep-reinventing** moment. Our prior
work is NOT wasted — the gating philosophy, coverage analysis, SAM-2 masking, faithful losses (alpha-mask,
sparse-depth), frame-selection, and ρ-gate all *complement* ArtiFixer. The faithful/plausible two-track thesis
(RESEARCH_VISION.md) is now validated by ArtiFixer's opacity-mixing design.

---

## Key verified sources
ArtiFixer 2603.00492 (github.com/nv-tlabs/artifixer, HF nvidia/ArtiFixer) · Difix3D+ 2503.01774 · GSFixer
2508.09667 · GSFix3D 2508.14717 · FreeFix 2601.20857 · HAD 2605.16873 · ArtifactWorld 2604.12251 ·
BAD-Gaussians 2403.11831 · CoMoGaussian 2503.05332 · DeblurGS 2404.11358 · CoherentGS 2512.10369 ·
3DGUT 2412.12507 · 3DGRUT (nv-tlabs/3dgrut) · Ref-Gaussian 2412.19282 · Spec-Gaussian 2402.15870 ·
3DGS-DR 2404.18454 · DashGaussian 2503.18402 · Grendel-GS 2406.18533 · MCMC-3DGS 2404.09591 (gsplat #822) ·
ReconViaGen 2510.23306 · Amodal3R 2503.13439 · SEVA 2503.14489 · Gen3C 2503.03751 · OracleGS 2509.23258 ·
GenFusion 2503.21219 · WarpRF 2506.22433. Blackwell build: CUDA 12.8 / sm_120 / torch cu128 (NVIDIA CUDA GPU
table, gsplat #855/#809).
