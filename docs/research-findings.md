# Apex Reality — Research Findings & Recommended Changes

*Generated: 2026-02-25 — June 2026 update appended at top*

---

# June 2026 Research Update

*Adversarially verified multi-agent research pass, 2026-06-11. 107 claims
extracted from 23 primary sources, 25 top claims verified by 3-vote panels,
plus three targeted follow-up investigations (enhancement, trainers, SfM).*

## Headline verdicts

1. **COLMAP-based SfM remains the right backbone.** Sattler et al. benchmark
   (arXiv 2603.20428, Mar 2026): feed-forward transformer SfM (VGGT lineage)
   is "significantly less accurate" than COLMAP/GLOMAP; best trade-off is
   feed-forward init + classical BA. VGGT-X and VGG-T3 (CVPR 2026) both
   concede the gap in their own papers.
2. **COLMAP 4.0 released Mar 2026** (4.0.4 current): GLOMAP merged as the
   "global" mapper, native ALIKED+LightGlue via ONNX, ~10-15% faster BA,
   deterministic with random_seed. Upgrade pycolmap 3.13 → 4.0.4.
3. **3DGS-MCMC densification is the proven training-quality lever**
   (NeurIPS 2024 spotlight; independently reproduced in gsplat eval:
   29.65 vs 29.00 PSNR vs DefaultStrategy at 3M Gaussians). Adopted as our
   default (gsplat MCMCStrategy, cap_max budget).
4. **Mip-Splatting anti-aliasing is the accepted production default**
   (gsplat rasterize_mode="antialiased"). Adopted.
5. **Our Cosmos 0.6B Fixer is NVIDIA's current commercial SOTA fixer** —
   released publicly as "Omniverse NuRec Fixer" (NeurIPS Dec 2025). Code
   Apache-2.0, weights NVIDIA Open Model License. **V2 weights (Nov 2025) on
   HF nvidia/Fixer — re-pull.** Difix3D+ itself is non-commercial; skip.
6. **The big artifact-removal win is the distill-back loop** (Difix3D+
   progressive update): render pseudo-views on interpolated trajectories →
   Fixer → append to train set → resume training. NVIDIA doesn't ship the
   loop; we implement it around Fixer + gsplat (commercial-clean).
7. **Faster-GS (CVPR Jun 2026, Apache-2.0)**: integrated quality stack
   (MCMC + Mip-AA + Speedy-Splat pruning + PPISP + mask-based carving init)
   inside the NeRFICG framework (MIT, Windows/CUDA 12.8). Young but the most
   interesting external trainer — add as experimental backend, A/B it.
8. **IMC 2025 winners still use ALIKED+LightGlue+COLMAP**; frontier moved to
   pair selection (MASt3R-ASMK retrieval). Keep our matcher. RoMa v2 (MIT,
   DINOv3 license caveat) as rescue matcher for weak pairs only.
9. **Masked-object SfM consensus unchanged**: run SfM unmasked, mask only for
   training. Our flow already correct.
10. **High-res practice**: 24MP native training infeasible on 24GB; train at
    factor 2 with coarse-to-fine (densify at factor 4, finish at 2). Adopted.
    Factor 1-2 at 6K realistic on Blackwell 96GB.

## Adopted June 2026 (v1.2)

- gsplat 1.4.0 → 1.5.3 (AOT-compiled; PLY export via export_splats now works —
  it silently failed on 1.4.0 which lacks the exporter)
- MCMCStrategy default (cap_max 1M objects / 3M scenes), opacity/scale reg
- rasterize_mode="antialiased" default
- SE(3) pose refinement (pose_opt, lr 1e-5) default on
- Coarse-to-fine resolution schedule (factor 2x final until step 15000)
- data_factor default 4 → 2
- Windows JIT shim: core/training/gsplat_jit_shim.py (gsplat passes GCC flags
  to MSVC; build needs vcvars -vcvars_ver=14.34 — MSVC 14.44 STL requires
  CUDA >= 12.4 while we're on 12.1)

## Watch list (no code or license-blocked as of June 2026 — recheck quarterly)

| Item | What | Status |
|---|---|---|
| GloSplat (2603.04847) | SfM tracks as optimizable anchors during 3DGS training; GloSplat-A beats all COLMAP baselines (28.86 PSNR MipNeRF360) | paper only, no code (verified via GitHub API) |
| SSR-GS (2603.05152) | direct+indirect specular decomposition (glossy objects) | paper only |
| 3DGRUT v1.1 (nv-tlabs) | 3DGUT popping-free rendering, Apache-2.0, Windows | candidate Blackwell-era backend |
| PPISP (nv-tlabs/ppisp) | learned ISP fixes photometric floater causes; also in gsplat main + Faster-GS | code out; needs newer gsplat than 1.5.3 for built-in path |
| CLM-GS (ASPLOS 2026) | CPU offload breaks single-GPU memory wall for full-res | for Blackwell-era full-res |
| MapAnything (Meta) | feed-forward SfM init, Apache weights variant, COLMAP export | optional rescue initializer |
| GSFix3D / GSFixer / FreeFix | repair+distill loops, stronger than Fixer on holes | license-encumbered (Inria/RAIL/CogVideoX) |
| ArtifactWorld (2604.12251) | 107.5K-clip artifact restoration dataset | unreleased |
| CHROMA (2507.15748) | feed-forward bilateral grids for appearance | availability unresolved |

---

## Current Pipeline (MVP)
- SuperPoint + LightGlue + pycolmap (COLMAP incremental SfM)
- 8192 keypoints, full resolution (resize_max=99999), exhaustive matching
- RTX 4090 (24GB VRAM)

---

## Recommended Config Changes (Immediate)

### 1. Switch to ALIKED-N16 (from SuperPoint)

**Why:**
- Sub-pixel keypoint detection via SoftArgMax (no 8x8 grid quantization like SuperPoint)
- At 6000px width, SuperPoint has up to 5.6px quantization error; ALIKED has none
- HPatches MMA@3: ALIKED 74.43% vs SuperPoint 65.37% (+9 points)
- Multiview mAA@5deg: ALIKED 61.44% vs SuperPoint 44.35% (+17 points)
- IMC 2024: 4 of 5 gold medalists used ALIKED
- License: BSD-3 (permissive) vs SuperPoint's MagicLeap non-commercial
- Already supported in our pipeline (feature_configs.py has ALIKED config)

**With LightGlue (near-equal):**
- MegaDepth AUC@5: SuperPoint 51.0, ALIKED 52.3 — essentially tied
- ScanNet AUC@5: SuperPoint 21.9, ALIKED 19.6 — SuperPoint slightly better indoors

**Verdict:** ALIKED is genuinely better for high-res SfM. The advantage is NOT just licensing.

### 2. Use resize_max=1600 (from 99999/full resolution)

**Why:**
- COLMAP has hardcoded pixel-space thresholds calibrated for ~1080p
- At 6000px, these thresholds become overly tight, rejecting valid matches
- HLoc's "high quality" config uses resize_max=1600
- 3DGS training typically uses 1600px anyway
- HLoc automatically scales keypoint coordinates back to original resolution
- Dramatically faster extraction and matching (fewer pixels to process)

**Resolution sweet spots:**
| Resolution | Quality | Notes |
|---|---|---|
| 512px | Risky | 1px error = ~12px at full res |
| 1024px | Minimum viable | HLoc default |
| **1600px** | **Recommended** | HLoc "high quality", matches 3DGS default |
| 2000-3200px | Diminishing returns | More compute, marginal quality gain |
| 6000px | **Counterproductive** | COLMAP thresholds too tight |

### 3. Keep 8192 keypoints

At 1600px, 8192 keypoints is proportionally denser than at 6000px. Good balance.

### 4. Keep exhaustive matching + adaptive disabled

Correct for ~520 images with quality priority.

---

## Feature Extractor Comparison

| Extractor | Keypoint Type | License | HPatches MMA@3 | IMC 2024 | Notes |
|---|---|---|---|---|---|
| **ALIKED-N16** | Sub-pixel (SoftArgMax) | BSD-3 | 74.43% | 4/5 gold | Best for high-res |
| SuperPoint | Grid-quantized (8x8) | Non-commercial | 65.37% | Used by some | Industry standard |
| DISK | Learned | Apache 2.0 | ~65% | Rarely used | RGB input |
| XFeat | Fast learned | Apache 2.0 | ~60% | Not competitive | Speed-focused |
| DeDoDe v2 | Dense detect+describe | MIT | ~70% | Some use | More compute |

## Matcher Comparison

| Matcher | Type | License | MegaDepth AUC@5 | Speed | Notes |
|---|---|---|---|---|---|
| LightGlue | Sparse | Apache 2.0 | 51.0 (SP), 52.3 (ALIKED) | ~10 pairs/sec | Current choice |
| **RoMa v2** | Dense | MIT | **62.8** | ~15-25 pairs/sec | Best quality, MIT |
| LoFTR | Semi-dense | Apache 2.0 | ~48 | Slower | Older |
| OmniGlue | Sparse + foundation | Apache 2.0 | ~53 | Slower | DINOv2 guided |
| GlueStick | Line + point | — | Similar to LG | Similar | Adds line features |

---

## Alternative SfM Methods (Transformer-Based)

### Key Finding: All process at ~512px internally

Every transformer-based method downsamples from 24MP to ~512x512 internally (12x downscale).
This means they lose the sub-pixel precision that matters for 3DGS quality.

| Method | Internal Res | Max Images (24GB) | License | Status |
|---|---|---|---|---|
| DUSt3R | 512x384 | ~20 pairs | CC BY-NC-SA 4.0 | Baseline, superseded |
| MASt3R | 512x512 | ~30 pairs | CC BY-NC-SA 4.0 | Better matching, still O(N^2) |
| MUSt3R | 512x512 | ~100 views | Non-commercial | Linear scaling, CVPR 2025 |
| VGGT | 518x518 | ~60-80 images | Apache 2.0 | CVPR 2025 Best Paper |
| Spann3R | 512x384 | Real-time | Apache 2.0 | Sequential, drift issues |

### VGGT Limitations
- 1.2B parameters
- OOMs at ~300 images even on large GPUs (A100 80GB)
- RTX 4090: OOM at just ~10 images (no native bfloat16)
- Multiple workaround papers exist: InfiniteVGGT, XStreamVGGT, IncVGGT, VGGT-Long
- NOT viable for 520 images on current hardware

### MUSt3R Limitations
- Processes sequentially via memory mechanism (not "simultaneously")
- Maximum tested: ~100 views
- Non-commercial license
- Still 512px max dimension

### Bottom Line
SuperPoint/ALIKED + LightGlue + COLMAP remains the best option for 520+ high-res images
on RTX 4090. The transformer methods are promising but not ready for production 24MP workflows.

---

## Training Custom High-Res Models

| Target | Cost | Feasibility |
|---|---|---|
| 1024px fine-tune MASt3R | $100-400 (rented A100s) | Feasible |
| 1024px from scratch | $4,000-16,000 | Possible but expensive |
| 2048px anything | $100K+ | Impractical (O(N^2) attention) |
| RTX 4090 training | — | Not possible even at 1024px |

Nobody has publicly trained DUSt3R/MASt3R/MUSt3R above 512px.
RoPE interpolation trick enables inference at 1024-1536px without retraining.

---

## Low-Res SfM + High-Res 3DGS (Decoupled Resolution)

This is standard practice in the community:
1. Extract features at resize_max=1600
2. HLoc scales keypoints back to original coordinates automatically
3. COLMAP runs SfM at the scaled keypoint positions
4. 3DGS training uses full-resolution images with the SfM poses

**Critical finding:** Running SfM at full 6000px is actually WORSE than 1600px because
COLMAP's fixed thresholds are calibrated for ~1080p imagery.

---

## Future Pipeline Considerations (Post-MVP)

### Phase 1: Drop-in Improvements (Low effort)
- Switch SuperPoint -> ALIKED-N16 (already in feature_configs.py)
- Set resize_max=1600 (one line change)
- Consider RoMa v2 as optional matcher for difficult scenes

### Phase 2: Advanced Matching (Medium effort)
- Integrate RoMa v2 as alternative to LightGlue for challenging scenes
- RoMa v2 outputs per-pixel 2x2 precision matrices for uncertainty-aware matching
- MIT licensed, ~15-25 pairs/sec on RTX 4090, 4.8GB VRAM

### Phase 3: Next-Gen SfM (High effort, future hardware)
- MUSt3R/VGGT for initial pose estimation (when hardware allows)
- Joint pose+3DGS optimization (3R-GS, JOGS, Nope-RoomGS)
- 2DGS + GlossGau for glossy/flat surface rendering
- Grendel-GS for distributed training on Blackwell clusters

### Phase 4: Glossy Surface Rendering (Future)
- 2D Gaussian Splatting (2DGS) for flat surfaces (SIGGRAPH 2024)
- Gaussian Opacity Fields (GOF) for surface extraction (SIGGRAPH Asia 2024)
- GlossGau for inverse rendering / BRDF decomposition
- Polarization-aided capture for specular surface separation

---

## Flash Attention Status (Windows)

PyTorch Windows builds do NOT include Flash Attention kernel (open issue since Aug 2023).
Memory Efficient Attention is used as fallback — identical quality, ~1.5-2x slower.
No fix available. FlashAttention-3 exists for Hopper/Blackwell but only on Linux.

---

## References

### Extractors & Matchers
- ALIKED: arXiv 2304.03608 (ICCV 2023)
- RoMa v2: arXiv 2511.15706 (Nov 2025)
- LightGlue: arXiv 2306.13643

### Foundation Models
- DUSt3R: arXiv 2312.14132 (CVPR 2024)
- MASt3R: arXiv 2406.09756
- MUSt3R: arXiv 2503.01661 (CVPR 2025)
- VGGT: arXiv 2503.11651 (CVPR 2025 Best Paper)
- DINOv3: arXiv 2508.10104 (Aug 2025)

### Joint Pose + 3DGS
- 3R-GS: arXiv 2504.04294 (Apr 2025)
- JOGS: arXiv 2510.26117 (Oct 2025)
- Nope-RoomGS: NeurIPS 2025

### Rendering
- 2DGS: arXiv 2403.17888 (SIGGRAPH 2024)
- GOF: arXiv 2404.10772 (SIGGRAPH Asia 2024)
- GlossGau: arXiv 2502.14129 (Feb 2025)

### Distributed Training
- Grendel-GS: arXiv 2406.18533 (ICLR 2025 Oral)
- LiteGS: arXiv 2503.01199 (single-GPU optimization, NOT distributed)
- FastGS: arXiv 2511.04283 (Nov 2025)

---

## Fact-Check: "Next-Gen Camera Alignment" Recommendations

An external recommendation document proposed a multi-phase pipeline replacing COLMAP entirely
with MUSt3R/VGGT/RoMa v2/3R-GS/2DGS/GlossGau/Grendel-GS. Below is our fact-check.

### Confirmed (Real and Accurately Described)

| Technology | Paper | Status | Description Accuracy |
|---|---|---|---|
| 3R-GS | arXiv 2504.04294 | Real | MCMC + MLP pose refinement + epipolar loss. Accurate. |
| JOGS | arXiv 2510.26117 | Real | Alternating pose/Gaussian optimization. Accurate. |
| Nope-RoomGS | NeurIPS 2025 | Real | Piecewise planarity for indoor scenes. Accurate. |
| GlossGau | arXiv 2502.14129 | Real | Inverse rendering with ASG-based BRDF. Accurate. |
| 2DGS | arXiv 2403.17888, SIGGRAPH 2024 | Real | 2D oriented disks for flat surfaces. Accurate. |
| GOF | arXiv 2404.10772, SIGGRAPH Asia 2024 | Real | Level-set surface extraction. Accurate. |
| Grendel-GS | arXiv 2406.18533, ICLR 2025 Oral | Real | Sparse all-to-all, batched multi-view training. Accurate. |
| FastGS | arXiv 2511.04283 | Real | Multi-view consistency densification. Accurate. |
| RoMa v2 covariance | arXiv 2511.15706 | Real | Per-pixel 2x2 precision matrix (Sigma^-1). Accurate. |
| RoMa v2 uses DINOv3 | Confirmed | Real | DINOv3 released Aug 2025 by Meta FAIR. Correct. |
| MUSt3R linear scaling | arXiv 2503.01661, CVPR 2025 | Real | Memory grows linearly via memory mechanism. Correct. |
| Spann3R | arXiv 2408.16061, 3DV 2025 | Real | Incremental ViT with spatial memory. Accurate. |
| LONG3R | arXiv 2507.18255, ICCV 2025 | Real | Dynamic spatial pointer memory. Accurate. |

### Critical Errors

#### 1. VGGT CANNOT process 600 images in a single pass (even with 96GB)

**Claim:** "With 96GB Blackwell, VGGT can ingest and optimize the entire 600-image dataset
in a single, unfragmented pass."

**Reality:**
- VGGT OOMs at ~300 images on A100 80GB
- On RTX 4090: OOMs at just ~10 images (no native bfloat16, falls back to float32)
- Attention scales O((NL)^2) where N=views, L=tokens/view
- Multiple papers exist to work around this: InfiniteVGGT, XStreamVGGT, IncVGGT, VGGT-Long
- 600 images at 518x518 in one pass is NOT feasible even at 96GB
- Would require chunked/streaming approaches (IncVGGT handles 10K frames under 80GB)

#### 2. MUSt3R does NOT process images "simultaneously"

**Claim:** "MUSt3R processes the entire 600-image sequence simultaneously."

**Reality:**
- MUSt3R processes images sequentially via a multi-layer memory mechanism
- New frames cross-attend with cached tokens from previously processed frames
- Maximum tested in experiments: ~100 views (ScanNet++)
- Training was done on 10-image sequences
- Linear scaling is correct, but "simultaneous" is wrong — it's streaming/incremental
- Still limited to max dimension 512px

#### 3. LiteGS is NOT a distributed framework

**Claim:** Document groups LiteGS alongside Grendel-GS as a distributed training system.

**Reality:**
- LiteGS is a single-GPU high-performance framework (by MooreThreads)
- Morton-coded spatial sorting claim is correct
- Achieves 3.4x speedup and ~30% memory reduction on a single GPU
- NOT multi-GPU distributed like Grendel-GS

#### 4. FlashAttention does NOT enable native 4K/24MP through transformers

**Claim:** "FlashAttention v2 allows processing massive image tiles natively at 4K/24MP."

**Reality:**
- Should reference FlashAttention-3 (not v2) for Blackwell/Hopper
- A 24MP image at patch_size=14 produces ~122,000 tokens per image
- 600 images = ~73 million tokens — intractable even with FlashAttention
- All current systems downsample to ~512px regardless of hardware
- FlashAttention helps with moderate sequences, NOT with 24MP-scale token counts
- The "coarse-to-fine tiling" approach is aspirational, not demonstrated in any system

#### 5. All foundation models process at ~512px (glossed over)

The document acknowledges this briefly but then proceeds as if 96GB VRAM solves it.
It doesn't — the resolution limit is architectural (quadratic attention), not just memory.
Even with infinite memory, processing 24MP through self-attention is computationally intractable.

### Summary Verdict

The document is ~70% accurate on individual technologies but ~30% wrong on how they
compose together, particularly around memory/scaling claims. The proposed "synthesized
architecture" would fail at Step 1 (VGGT with 600 images) on any current hardware.

---

## Our Grounded Approach

### Current Phase: RTX 4090 (24GB)

```
ALIKED-N16 @ 1600px -> LightGlue (exhaustive, adaptive off)
  -> COLMAP incremental SfM -> standard 3DGS with full-res images
```

Changes from current MVP:
1. ALIKED-N16 instead of SuperPoint (dropdown change, already supported)
2. resize_max=1600 instead of 99999 (config change)
3. Keep 8192 keypoints, exhaustive matching, adaptive disabled

Why this works:
- ALIKED sub-pixel accuracy eliminates SuperPoint's 5.6px grid quantization
- 1600px matches COLMAP's threshold calibration sweet spot
- HLoc scales keypoints back to original coordinates for 3DGS
- Proven at scale (IMC 2024 competition winners)

### Near-Term Additions (weeks, RTX 4090)

4. **RoMa v2 as optional matcher** for difficult/glossy scenes
   - MIT license, 4.8GB VRAM, ~15-25 pairs/sec
   - Per-pixel uncertainty weighting via covariance matrices
   - Use as verification layer or for scenes where LightGlue fails

5. **VGGT for small subsets** (~60-80 images max on 4090)
   - Use on hardest images (glossy/textureless) for initial pose estimation
   - Fold those poses into main COLMAP reconstruction
   - NOT as full pipeline replacement

### Future Phase: Blackwell Pro 6000 (96GB)

```
MUSt3R @ 512px (streaming, initial poses, ~100-view chunks)
  -> RoMa v2 @ 864px (sub-pixel refinement with covariance)
  -> COLMAP (final geometric verification + BA)
  -> 3R-GS (joint pose + 3DGS optimization)
  -> 2DGS + GlossGau (flat surfaces + glossy BRDF)
  -> Grendel-GS (distributed multi-GPU training at 4K)
```

Key differences from the external recommendation:
- MUSt3R in streaming chunks (~100 views), NOT "simultaneous 600"
- COLMAP still used for final geometric verification (not eliminated)
- IncVGGT or VGGT-Long instead of raw VGGT (chunked, not single pass)
- No assumption of native 24MP through transformers (still ~512-864px)
- LiteGS for single-GPU optimization, Grendel-GS for multi-GPU

### What to Skip Entirely

- **Replacing COLMAP entirely** — transformer SfM cannot match COLMAP's precision
  at scale, even on Blackwell. Use them for initialization, not replacement.
- **Native 24MP through any transformer** — architecturally impossible with
  current attention mechanisms. Always extract at 1600px, train 3DGS at full res.
- **Training custom high-res foundation models** — $100K+ for 2048px, nobody has
  done it above 512px. Use RoPE interpolation for 1024-1536px inference instead.
- **Polarization capture** — interesting research, impractical for production
  (requires hardware modification, calibration, specialized processing pipeline).

### Glossy Surface Strategy (Pragmatic)

For glossy electronics and flat surfaces specifically:

1. **Short term:** More images from varied angles, avoid direct specular reflections
   in capture. Standard 3DGS with good poses handles mild glossiness fine.

2. **Medium term:** 2DGS for flat surfaces (walls, screens) — drop-in replacement
   for standard 3DGS, forces Gaussians into thin oriented disks. Already available
   (SIGGRAPH 2024, open source).

3. **Long term:** GlossGau inverse rendering for explicit BRDF decomposition.
   Separates diffuse albedo from specular reflections. Requires 2DGS as base.

4. **Joint optimization:** 3R-GS for correcting residual pose errors during
   training. MCMC noise prevents Gaussians from getting stuck on featureless
   surfaces. MLP pose refinement maintains global trajectory coherence.
