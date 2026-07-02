# ArtiFixer on Blackwell (RTX PRO 6000 / sm_120) — Investigation & Learnings (June 2026)

Full record of the ArtiFixer feasibility + evaluation work on the new 96 GB Blackwell box.
Companion: `BLACKWELL_FRONTIER.md` (the broader frontier + native-res finding), memory
`artifixer-adoption.md` (build state). Tags: results are our own measurements unless noted.

## TL;DR
- **ArtiFixer3D runs LOCALLY on the RTX PRO 6000 and beats our Difix-v2** on well-observed views
  (Laptop, 38 held-out @720p: base 0.3531 LPIPS → **ArtiFixer3D 0.3150, −10.8%**, PSNR up; Difix-v2 = −8.7%).
- We got there by **sidestepping a hardware wall**: ArtiFixer's intended *causal AR* inference needs
  **block-sparse flash-attention, which does NOT exist for sm_120** (workstation/consumer Blackwell) →
  ~96 min/frame (unusable). The **`--inference_pipeline bidirectional`** path uses plain SDPA (fast on
  sm_120) → full 38-frame run in ~12 min.
- **The bidirectional workaround is weak exactly where the causal model shines**: coverage gaps
  (+4.0% only), 3D+ (regresses), 2D-consistency (hurts PSNR). Every divergence from the paper traces
  to bidirectional-vs-causal. Full quality / gap-filling / 3D+ need the **causal model → cloud H100/B200**.

## 1. Environment (see memory `blackwell-build`)
- sm_120 / cap (12,0), CUDA 12.8/12.9, torch 2.11.0+cu128. WSL2-native venv `~/artifixer-venv`, repo `~/artifixer`.
- gsplat AOT rebuilt for sm_120 (our trainer). ArtiFixer stack: diffusers 0.37.1, transformers 5.5.0, 3DGRUT compiled.
- ArtiFixer inputs = a COLMAP scene (our Stage-7 export). Weights `nvidia/ArtiFixer` artifixer-14b.pt (67 GB, Apache code / NVIDIA-NC weights).

## 2. The sm_120 block-sparse wall (why causal is slow) — VERIFIED
- ArtiFixer's transformer uses `flex_attention` with a **block-causal + attention-sink + local-window**
  mask ONLY in the causal AR / diffusion-forcing path (`frames_per_block` set); otherwise plain SDPA.
- On sm_120: stock flash-attn-4 **asserts** block-sparse off (`"Block sparsity not supported on SM 12.0"`)
  AND its SDPA override errors (`"FA4 requires compute capability 9.0 or 10.0"`). Installing FA4 is
  actively harmful on this card.
- Fallback = Triton flex → **~96 min/frame (~48,000× slower than the paper's 8 FPS)**. NOT a recompile
  trap: patched `dynamic=True` + recompile_limit=64 → 0 recompiles, STILL ~90 min/frame → genuine kernel slowness.
- **Dense-SDPA rewrite is infeasible** (local-window+sink over long seqs → ~20 GB/layer mask). cuDNN can't
  natively express the sink. So the pattern genuinely needs a block-sparse kernel.
- **FA4 sm_120 community fork** (`SecondNatureComputing/flash-attn-4-sm120`) HAS the code (pure CuTeDSL,
  block-sparse fwd, head_dim≤128 = Wan fits) but is locked to a **nightly cutlass-dsl**: hand-porting hit a
  cascade of API mismatches (`cute.core.ThrMma`→`cute.ThrMma`, then `cute.make_fragment` missing, then
  a tile-size mismatch `sparse_block_size[1]=64`…). Open-ended. Also `pip install flash-attn-4` DOWNGRADES
  torch 2.11→2.9 (use `--no-deps`); the `kernels` lib breaks transformers + has an httpx UA bug.

## 3. The bidirectional unlock (the working local path)
`run_inference --inference_pipeline bidirectional --bidirectional_chunk_size 8 --max_neighbors_per_encode 1`
→ SDPA (all 4 backends verified on sm_120), no block-sparse. Full pipeline on the card in ~25 min:
`prepare` (3DGRUT recon) → bidirectional `run_inference` (2D pred) → `run_artifixer3d` (distill into fresh 3DGRUT).
Resolution: downscale scenes to ~720–960p first (model built for DL3DV-960P; 4K OOMs). prepare asserts
uniform aspect → use exact 16:9 multiples (1280×720, 1712×963).

## 4. Empirical results (Laptop, our own eval, LPIPS via project get_lpips)

### Easy case — 266 anchors, 38 every-8th held-out (interpolation), @720p
| variant | LPIPS↓ | PSNR | SSIM | vs base |
|---|---|---|---|---|
| base 3DGRUT | 0.3531 | 22.94 | 0.806 | — |
| ArtiFixer 2D (bidir) | 0.3368 | 20.49 | 0.759 | −4.6% (hurts PSNR = inconsistent) |
| **ArtiFixer3D** | **0.3150** | **23.27** | **0.810** | **−10.8% (best; PSNR recovered)** |
| ArtiFixer3D+ | 0.3307 | 21.06 | 0.767 | −6.3% (regresses; bidir re-fix re-adds inconsistency) |
Ref: our Difix-v2 distill = −8.7% (same 38 views, factor2). **ArtiFixer3D beats it.**

### Hard case — 40-frame CONTIGUOUS WEDGE (coverage gap), @720p
base 0.3742 → 2D 0.3991 (−6.6% WORSE) → **ArtiFixer3D 0.3591 (only +4.0%)**. Lift SHRANK vs easy case.
Cause: (1) bidirectional is off-distribution + weak at EXTRAPOLATION (the causal AR rollout it was distilled
for is what propagates anchors into gaps); (2) wedge = dark/simple laptop rear = Cause-B capture-limited
(no faithful detail; generative fill is plausible-not-faithful — matches quality-roadmap rev-5 + RESEARCH_VISION).

### Resolution — 960p (1712×963), matched budget, COMMON-RES eval (all→720p vs same GT)
ArtiFixer3D@720p **0.3150** vs ArtiFixer3D@960p→720 **0.3248** → 960p slightly WORSE. Cause in data:
base@960p→720 (0.3651) worse than base@720p (0.3531) → the 3DGRUT recon (10k iters, 1M Gaussians, SAME
budget) is UNDER-CONVERGED at 1.77× pixels. **Resolution alone isn't a lever — need resolution + scaled
recon budget.** (Careful: absolute LPIPS not comparable across res; the common-res eval is what settled it.)

## 5. Mechanistic insights (the load-bearing conclusions)
- **ArtiFixer3D's real strength = SHARPENING well-but-imperfectly-observed views** (faithful detail that
  IS in the signal, Cause-A). It is NOT a coverage-gap filler in bidirectional mode.
- **The 3D distillation is what makes it work**: it fuses out the 2D per-frame inconsistency (recovers PSNR).
  2D alone hurts PSNR; 3D recovers it; 3D+ (bidir re-fix) re-breaks it.
- **The base recon quality is a binding limiter** (960p finding). Better base → better ArtiFixer3D.
- **Every divergence from the paper = bidirectional vs causal.** Where the causal mechanism isn't essential
  (sharpen observed), we match/beat; where it is (gaps, 3D+), the workaround underperforms.

## 6. Consistency with the authors
- ✅ Matches: the 2D(sharp,inconsistent)→3D(consistent) pattern; ArtiFixer>Difix direction.
- ⚠️ Can't confirm: their HEADLINE (transformative on sparse/under-observed) — that needs the causal model,
  which we can't run on sm_120. Our 3D+ regressing (vs their 3D+ helping) is the same bidirectional gap.
- We are NOT reproducing the paper (bidirectional workaround, easy dense case, our scene, homemade Difix,
  720p, under-converged base). Our numbers are a LOCAL FEASIBILITY STUDY.

## 7. Open questions (→ improvement directions)
1. Does the CAUSAL model actually fill gaps + make 3D+ win? (cloud H100 test — settles the authors' claim.)
2. Can we get causal-quality attention on sm_120 WITHOUT the fork? (cuDNN sliding-window + explicit sink?)
3. Does a BETTER base recon (our gsplat MCMC trainer w/ gated-LPIPS/alpha/depth, vs vanilla 3DGRUT) lift ArtiFixer3D?
4. 960p + scaled recon budget (25k iters, 2–3M cap) — does higher res then help?
5. Should ArtiFixer + motion-deblur (the rev-5 faithful lever) be combined for real orbit video?
6. Two-track: ArtiFixer3D (sharpen observed) + completion track (ReconViaGen/SEVA + WarpRF/HAD gating) for gaps?

## 8. Research refinement (2026-06-30, primary-source verified) — corrects earlier assumptions
- **cuDNN SDPA on sm_120 natively does what we need — no FA4 fork.** cuDNN (≥9.15, cc-120-optimized;
  ~97% speed-of-light on a 5090) supports a fused **128×128 block-mask** ("1=compute tile, 0=skip") +
  sliding-window band API. Encode **block-causal ∧ local-window ∧ first-K-sink-columns** as ONE static
  block-mask → fused, fast, no fork. Caveats: call the **cuDNN-frontend bindings directly** (torch's public
  SDPA exposes no block-mask), use **cuDNN ≥ 9.23** (sm_120 SDPA correctness fixed there), head_dim ≤ 128
  on sm_120 (Wan DiT = 128 ✓), numerically validate vs the bidirectional reference. THIS is the cheapest
  LOCAL causal path.
- **Triton flex on sm_120 is BROKEN, not just slow** (PyTorch #176426: Triton kernels w/ ≥2 `tl.load`
  segfault/mis-codegen on sm_120; sm_100 unaffected). Confirms our 96 min/frame is a known sm_120 codegen
  bug — we won't fix it. torch flex's fast FA4 backend is datacenter-only (H200/GB200).
- **CORRECTION: ArtiFixer's base is ALREADY 3DGUT + MCMC densification (10k iters)** — NOT "vanilla
  3DGRUT." So "swap base for MCMC" is a near no-op. The real untested deltas: more iters / bigger Gaussian
  budget / perceptual+depth losses on the final recon, and especially **restoring the PROGRESSIVE distill
  loop** — ArtiFixer drops it for speed, but Difix3D+ shows progressive > single-pass (Nerfbusters LPIPS
  0.342 single → 0.328 incremental). Our `core/enhancement/distill.py` already IS a progressive loop.
- **Cheapest causal test = H100 spot (sm_90), ~$0.03–0.55 per ~40-frame run** — stock flash-attn "just
  works" on sm_90 (FA2+FA3); B200 (sm_100) uses FA4 and costs 2–4× more for no needed benefit; FA3 does NOT
  run on any Blackwell. So DE-RISK on a cheap H100 spot BEFORE any kernel work.
- **Faithful levers for orbit video (confirmed priority):** deblur (BAD-Gaussians / CoMoGaussian /
  CoherentGS for sparse+blur) + feed-forward geometry anchor (VGGT / MASt3R-SfM / VGGS) + confidence-gated
  fills (FlowObject 2606.19019, entropy-gate 2503.15742, CoR-GS/PUP/FisherRF uncertainty) + object symmetry
  (AutoSplat) + OrbitForge-style "fill only detected gaps." These attack our true defects more directly than
  the causal-vs-bidirectional attention question.
- Weights license flag: HF card shows "NVIDIA License" on weights (code Apache-2.0) — review before OSS use.

## Reproduce
Scripts in session scratchpad: `wsl_bidir_infer.sh`, `wsl_artifixer3d.sh`, `wsl_3dplus.sh`,
`wsl_wedge_pipeline.sh`, `wsl_960_pipeline.sh`; eval `lpips_3way_arg.py`, `common_res_eval.py`;
PLY export `export_ply.py`. Samples/PLYs under `D:\REPOS\HYPERclone\artifixer-checkpoints\`.
