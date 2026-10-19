# Research Refresh — 2026-10-17

Status check + external sweep after ~10 weeks idle (last work 2026-08-05, HEAD `b225811`).
Sources: 1 scout (repo state) + 5 researcher agents (training quality, poses/SfM, generative
rescue, SH/formats/Unreal, industry landscape). Tags: [V] primary source fetched, [PC] paper/
secondary claim, [U] unverified. Nothing here has been run on our machine yet — every item
below still needs its own default-OFF flag + validation gate.

---

## 0. Repo status (where we stopped)

- Closed & gated: WP-Q1, Q2, F1, F2, S1, F4 (conditional, default-OFF), C1 v1, D1, X1, S2,
  G1-phase1 (3DGS-DR failed), P1 (all generative polish failed), O1, O2, O3, O5; off-plan
  T1/T2 (Sony shoe), UI-1..5; **2026-10-17/18:** STEP 0 (`opacity_reg` default → 0),
  WP-SH4 Phase A & **Phase B** (`_infusion/wpsh4/phaseA_gate.json`, `_infusion/wpsh4_phaseB_RECORD.md`),
  WP-D2 (SPZ v4 writer), WP-B2 (`gsplat main` 1.6.0 @ `512d366` in `venvs/gsplat_main`),
  **WP-F5 (PPISP `v1.3.0` @ `a247368` across Laptop, GPU_V4, 360 — `_infusion/wpf5_RECORD.md`)**,
  **WP-C3 (TIDI-GS cross-view evidence, importance & isolation pruner — `_infusion/wpc3_tidi_RECORD.md`)**,
  **WP-SfM2 (COLMAP / pycolmap 4.2.1 + native GPU `LOMA_B` — `_infusion/wpsfm2_RECORD.md`)**,
  **WP-R1 (observed-preservation 3-zone gate for generative supervision — `_infusion/wpr1_RECORD.md`)**,
  **WP-B1 / Spirula recon (`_infusion/wpb1_spirula_recon.md`)**.
- Not started / parked: F3 DIM depth, C2 GENA3D, G1-phase2 (Spec-Gaussian ASG), O4 fisheye A/B
  (past its park date), WP-M1 (needs capture with transient people), WP-C4 (orbit-fill bake-off).
- 360 lane: `tools/prune_tidi.py` (`WP-C3`) removes `67.1 %` of `prune_op05` splats (`2.41M → 793k`,
  `569.7 → 187.3 MB`) and eliminates `100 %` of mid-opacity unverified fog and `>10×` bbox outliers
  with `0.0000` mLPIPS delta; `COLMAP 4.2.1` + `LOMA_B` (`WP-SfM2`) increases rig-verified inliers
  by `1.86×` over ALIKED+LG (`7.0×` over SIFT).
- Loose ends resolved: `DEVELOPER_NOTES.md` §8 updated to current Oct-2026 Blackwell environment.

### ⚠ Defect found
`opacity_reg` defaults to **0.01** in `core/training/gsplat_trainer.py:1945`,
`core/training/gsplat_backend.py:250` and the `run_*` scripts. WP-O3/O5 showed 0.01 and 0.001
destroy many-view walkthroughs and DEVELOPER_NOTES §3 says it "must be exactly 0", but the code
default was never changed and neither GUI nor 360 path overrides it. Needs a decision + a
re-check of which production runs were affected.

**RESOLVED 2026-10-17 (STEP 0):** default changed to 0 everywhere (still an opt-in flag); Laptop
production-stack regression 0 vs 0.01 = masked LPIPS 0.3098 -> 0.2794, +1.08 dB, fog 71 % -> 27 %.
Affected-run inventory + evidence: `_infusion/opreg0_RECORD.md`.

### Versions
- gsplat installed 1.5.3 (prod venv) + `1.6.0` @ `512d366` + `ppisp 1.3.0` @ `a247368` (`venvs/gsplat_main`, WP-B2/WP-F5). pycolmap 4.1.0 (prod venv) + `pycolmap 4.2.1` + standalone CUDA/ONNX `colmap.exe 4.2.1` @ `bd1fcf6` (`venvs/colmap421`, WP-SfM2).

---

## 1. Training-core findings

| Item | Evidence | Why it matters to us |
|---|---|---|
| **PPISP** (NVIDIA, 2601.18336, CVPR'26 oral, Apache-2.0, nv-tlabs/ppisp) | [V] T&T LPIPS 0.303→0.281, PSNR 23.08→24.36; Mip360 ≈flat. Integrated in gsplat `main` (Jan'26; MCMC strategy Jul'26). Post-render module → **PLY stays standard** | This is our unstarted WP-F5. Industry converged on it: Postshot 1.1 "Photometric Compensation", LichtFeld 0.5, 3DGRUT 1.1. Exposure inconsistency → fog is our #1 imperfect-capture failure |
| **gsplat `main` rebuild** | [V] README: native CUDA MCMC `inject_noise`, configurable relocation, 3DGUT per-ray/ortho, new fisheye/FTheta models, AccuTile, SH ops split w/ faster backward, CUDA 12.8 build compat, torch ≥2.7 | One AOT build unlocks PPISP + speed + camera models. sm_120 build untested [U] |
| Confidence-aware densification (2610.13541, 2026-10-09) | [V abstract] beats MCMC at all budgets 100k–2M; no code | Targets MCMC background starvation → off-path holes/fog. Small sampler change; reimplement |
| TIDI-GS floater pruning (2601.09291) | [V abstract] cross-view evidence + isolation + importance; no code | Directly on the fog/floater bar; cheap pre-export prune |
| "In Depth We Trust" (2604.05715) | [V abstract] mono-depth only on ill-posed regions; no code | Fixes the WP-F1 pattern (prior helps geometry, not LPIPS) |
| Polarized opacity prior (2608.22344) | [V abstract]; no code | Cheap single loss for fog. Thin evidence |
| ImprovedGS / ImprovedGS+ (2508.12313, 2603.08661) | [V] Mip360 +0.15 dB vs MCMC-1M, 27% faster; no LPIPS | Marginal; port only if MCMC is the bottleneck |
| Faster-GS (2602.09999) | [V] 4× faster, quality flat | Speed only; ablations: don't skip invisible updates, don't drop SH |
| UBS-6D Beta splatting (2510.03312) | [V] Mip360 LPIPS 0.218→0.184 | Watch only: no code, **non-standard PLY** |
| LichtFeld Studio 0.5.4 (2026-10-08) | [V secondary] normal-guided training, MRNF strategy, spherical COLMAP loader, PLY/SOG/SPZ | Use as cross-check trainer / reference implementation |

## 2. Poses / SfM findings

| Item | Evidence | Fit |
|---|---|---|
| **COLMAP/pycolmap 4.2.1** (2026-09-29) | [V] GLOMAP merged (`--mapper GLOBAL`, multi-model in 4.2.0); Caspar GPU BA (4.1.0, claim 10–100×); spherical/ERP cameras improved in 4.2.0 (reproj error + guided matching); **LoMa learned matcher built-in via ONNX** | Low-effort bump; re-run the ERP probe — our 1.22 px predates the spherical fixes. Use 4.2.1 not 4.2.0 (BA regression). Windows wheel / Caspar-on-sm_120 [U] |
| LoMa (2604.04931) | [V] LightGlue-sized, beats RoMa v2 on WxBS/IMC22 (README) | Cheap matcher A/B vs our ALIKED/LightGlue/RoMa dense init |
| MapAnything | [V] `demo_colmap.py` COLMAP export; ≤2000 views on 140 GB | Feed-forward fallback for blurry/low-parallax captures |
| VGGT-Long (ICRA'26) | [V] chunk + Sim3 + loop closure; backbone VGGT/π³/MapAnything; Ubuntu-only | WSL fallback for 2000+ frame video / 360 orbits |
| VGGT-Ω (2605.15195, CVPR'26 oral) | [V] 43 GB @ 500 frames; gated weights; **use `*_reproduce.pt`** (original checkpoint flagged) | Replace VGGT in fallback tier after SDPA check |
| VGGT-X | [V] COLMAP output, PINHOLE only | Pose-init → our pose_opt polish (NB WP-T2: pose_opt OFF at native res) |
| SAM 3.1 (2026-03-27) | [V] text-prompted video seg ("person"), gated HF, Linux | Transient masks → matcher + loss (`loss_mask` already exists) |
| Evidence caveat | [PC] GASE (2606.17520): VGGT/DA3/π³ inits blurrier than COLMAP | Keep COLMAP+BA as precision path; feed-forward = fallback/init only |
| No code yet | VGG-T³, PanoVGGT (ERP feed-forward), VGTW, GloSplat | Watch |

## 3. Generative rescue findings

Context: WP-P1 (all polish failed) and WP-C1 (no-fill control beat both fills) mean every
generative item must pass WP-Q1 visibility-split gates. ArtiFixer remains the centerpiece; nothing
newer clearly beats it.

| Item | Evidence | Fit |
|---|---|---|
| **ArtiFixer** full release | [V] nv-tlabs/artifixer (Apache code), HF `nvidia/ArtiFixer`: distilled 14B/1.3B + **new bidirectional S1 teachers**; 1.3B fits 80 GB; card: "FA4 on Blackwell" | **Prerequisite: force SDPA** (FA4 is banned on sm_120 by our rules). Dispatch location [U] |
| **Observed-preservation gating** (GSComplete preservation loss, Filling-the-Unseen QA-Mask, ArtiFixer opacity mixing) | [V abstracts] | Same idea three times: weight generative supervision by render confidence/opacity; observed content cannot change. Cheapest, highest-value fix for why P1/C1 failed |
| RealVDeblur (2607.20628) | [V abstract] one-step video deblur trained on 3DGS-synthesized blur; repo/HF linked, status [U] | Best match for the motion-deblur lever (pre-SfM). Verify repo first |
| ArtifactWorld (2604.12251, ACM MM'26) | [V] inference + LoRAs on LTX-Video 13B; artifact heatmap | 2nd repair model; heatmap = learned "broken here" mask for region gates (was WP-Q1 phase-3 item) |
| SeedVR2 | [V] SDPA fallback in ComfyUI wrapper | Safest SR on sm_120; source frames only |
| FlashVSR v1.x | [V] needs Block-Sparse-Attention | **Blocked on sm_120** (no block-sparse). FlashVSR+ (2610.17552) unverified |
| GSFixer (ICML'26), NVIDIA Fixer (Cosmos 0.6B) | [V] code + weights | Comparators / cheap first-pass |
| HY-World 2.0 WorldStereo-2 / WorldMirror-2 | [V] weights out; 17B / 1.2B | Completion of unobserved regions; WP-Q1 only |
| Lyra 2.0 | [V] single-image→world, research-only | Borrow ideas only |
| MEt3R | [V] 2605.18754 shows it fails robustness (noise scored as consistent) | Demote to diagnostic; prefer SfM registration-rate / reprojection gate (W-GPC style) |

## 4. SH degree 4 / formats / Unreal

**"4SH" = SH degree 4** (user, 2026-10-17: "4 spherical harmonics"; supporting degree 3 would
not be news): 25 coeffs/ch, 72 `f_rest`. The plugin source is not in this workspace — confirm the
importer reads `f_rest_0..71` in channel-major order on the first round-trip test.

Degree-4 path:
- **gsplat supports it** [V main]: CUDA fwd/bwd dispatch 0..4; but the fused K16 fast path is
  degree-3 only → K=25 uses the generic (slower) kernel. Torch reference `_eval_sh_bases_fast`
  is the unit-test oracle (returns garbage >25 — never use degree 5). `export_splats` PLY is
  degree-agnostic; verify channel-major `f_rest` ordering.
- **Our code:** trainer flag `--sh_degree` (`gsplat_trainer.py:1932`) already generic; blockers
  are `tools/align_ply.py:61` (`SH_DEGREE = 3` hard-coded), vendored SuperSplat viewer
  (`gui2/viewer/index.js` caps 3 / `new Array(45)`), QA defaults (`qa_report.py:222`,
  `gui2/app.py:318`), delivery untested at 72 coeffs.
- **Formats:** SPZ v4 supports degree 0–4 [V] (only compact format that does). SOG v2 and
  KHR_gaussian_splatting (ratified) cap at degree 3 [V]. Naive PLY loaders break on 72 `f_rest`
  (Octane 2026.4 bugfix precedent) — Postshot behaviour unknown.
- **Evidence of gain:** **none published** for deg3→deg4 [U]. Prior internal notes
  (BLACKWELL_FRONTIER.md:125) were skeptical. Overfitting risk on interpolative capture paths.
- Delivery recipe: deg-4 master PLY + SPZ v4 for our plugin; deg-3 twin (exact truncation) for
  Postshot; deg-2/3 SOG for web.
- Ecosystem: no first-party Epic splats; NanoGS (MIT, UE 5.6+, Nanite-style LOD) is the most
  interesting engine-side reference; no plugin states deg-4 support — ours may be first.
  PBR/relightable splat export is immature — don't plan around it.

## 5. Industry convergence (what a clean-splat pipeline must have)

1. Per-image photometric compensation (PPISP-style) — **we lack it (F5)**.
2. Appearance/artifact sink discarded at render (NTIRE'26 IDEAL) — we have bilagrid.
3. MCMC + fixed budget + anti-aliasing — have.
4. Pose filtering/refinement — partial (pose_opt, flawed at native res).
5. Depth/normal priors — have MoGe-2 (F1); normals not yet in loss.
6. Non-pinhole cameras (3DGUT/spherical) — 360 via crop-rig; native path not done.
7. Delivery tooling: SH capping, SPZ v4, LOD/streamed SOG — have SOG; **no SPZ writer**.
8. NTIRE'26 winners: restore-2D-first → photometrically tolerant 3DGS → ensemble.

Gaps where we can beat products: off-path/fog metrics (nobody publishes them), integrity-gated
generative fill, 360 raw ingest with a quality gate, engine-ready bounded/pruned deliverables.

---

## 5b. Late additions (user leads, 2026-10-17)

### Spirula Studio (ex-spirulae-splat)
[V] github.com/harry7557558/spirula-studio. One maintainer, GPL-3.0, C++, about 1.7k stars, weekly
releases (latest v2026.10.16). One binary covers frame extraction, SfM, masking (SAM 2.1 text,
BiRefNet), training and meshing. No Python or COLMAP. Vulkan backend plus a legacy CUDA backend;
Windows binaries. Strategy is a mix of MCMC, IGS+ and MRNF. Has PPISP, bilateral grid, quantized
training, RoMa v2 dense init, LoMa, partitioned training, and warm start from PLY. **Native
fisheye, ERP and 360 input** (.insp, GoPro .360), with rig constraints. **SH3 maximum, no SH4.**
The optional spherical beta lobes are not SH. No benchmarks published. sm_120 untested.
→ Best candidate **cross-check trainer**, especially for the 360 lane (`360-camera` preset, beta
lobes OFF), judged with our off-path and fog suite. Run it from its own empty directory (untrusted
binary).

### Orbit-video hole fill (LTX-2.5, MiniMax H3, and others)
- The mechanism is real: video diffusion conditioned on an anchor render plus a mask along a
  trajectory. It is exactly what ArtiFixer already does. The question is backbone choice, not
  concept.
- **LTX-2.5** [V]: 22B, open (gated), roughly Aug 2026. **No camera or pose control.** It does
  have an **In/Outpainting IC-LoRA**: a mask video, with unmasked pixels kept by compositing and
  Laplacian blend. That LoRA is v0.9, trained on 2.3, and aimed at object replacement. Needs
  Python 3.12 in its own WSL venv, SDPA path (FA4 is auto-selected only on datacenter Blackwell).
  Layout-To-Render IC-LoRA restyles rather than preserves.
- **MiniMax H3** [V]: open 33B; ref2va and fl2va; no documented camera control. The "360 orbit" is
  a community fl2va LoRA that rotates **a single photo** (drifts mid-orbit). API about $0.08/s at
  768P. **The Aug-2026 verdict stands.** At most a $1 hosted probe after the other arms.
- Render-conditioned peers: GEN3C (render plus disocclusion mask, Cosmos-7B variant), Wan2.1-VACE
  MV2V (same family as ArtiFixer), TrajectoryCrafter. Not suited: ReCamMaster, Voyager, Lyra 2.0.
- Distillation evidence: VidSplat, G4Splat, Bi-FlowGS, Filling-the-Unseen. Gains appear only with
  geometric visibility masks, per-pixel confidence, masked Laplacian or perceptual loss on
  generated pixels, a capped generated-loss weight, tail-frame discard, single-round or
  re-initialised loops, and per-frame photometric normalisation. Failure modes: geometry cheating,
  mean-colour blur, tail hallucination, exposure drift becoming floaters. **No paper evaluates
  truly unobserved regions the way WP-Q1 does.**
- → **WP-C4 "orbit-fill backbone bake-off"**, after WP-R1. Use the C1 footprint-deletion holes and
  2 to 3 look-at orbits starting from observed poses (8n+1 frames). Geometric visibility masks.
  Arms in order: ArtiFixer (control), LTX-2.5 inpaint IC-LoRA, Wan2.1-VACE MV2V, then an H3 API
  probe. Distillation is identical across arms. Gate: beat the **no-fill control** on the
  unobserved split, with no observed-region regression. Also report off-path renders, fog, a
  geometry-cheating check (observed depth and normal drift, reprojection error on generated
  frames) and flicker.

## 6. Proposed work packages (ranked; each default-OFF + gated)

| # | Proposed WP | Effort | Gate idea |
|---|---|---|---|
| 0 | ✅ **Fix `opacity_reg` default → 0** (`_infusion/opreg0_RECORD.md`) | trivial | CLOSED 2026-10-17: Laptop mLPIPS `0.3098 → 0.2794`, +1.08 dB, `op<0.05` `71% → 27%` |
| 1 | ✅ **WP-B2: gsplat `main` AOT rebuild** (`_infusion/wpb2_RECORD.md`) | 0.5–1 d | CLOSED 2026-10-17: `1.6.0 @ 512d366` in `venvs/gsplat_main`; Laptop gate within 0.0004 mLPIPS |
| 2 | ✅ **WP-F5: PPISP** (`_infusion/wpf5_RECORD.md`) | 1–2 d | CLOSED 2026-10-18: beats bilagrid by 6.6–17.2 % mLPIPS (+3.4 to +5.9 dB) across Laptop, GPU_V4, 360; default-OFF |
| 3 | ✅ **WP-SfM2: COLMAP 4.2.1 + native GPU `LOMA_B`** (`_infusion/wpsfm2_RECORD.md`) | 1–2 d | CLOSED 2026-10-18: `venvs/colmap421`; ERP +7.6..8.2 % verified inliers, +6.8..7.3 % 3D pts, 1.26× faster; `LOMA_B` on 360 rig gives 1.86× inliers & 1.87× 3D pts vs ALIKED+LG (`7.0×` vs SIFT); `guided_matching` on ERP confirmed crash (`#4054`, keep OFF) |
| 4 | ✅ **WP-SH4: degree-4 SH** (`_infusion/wpsh4/phaseA_gate.json`, `_infusion/wpsh4_phaseB_RECORD.md`) | 2–3 d | CLOSED 2026-10-18: Phase A + trainer warm-start/schedule PASS; Phase B warm-start SH4 gains `+0.203 dB` (`GPU_V4`) / `+0.089 dB` (`Laptop`) on unseen 38°-arc specular PSNR with lossless SH3 twin (`+0.058 dB`), but overall mLPIPS flat (`-0.0001`), `+46%` PLY size, and from-scratch SH4 co-adapts bands (`-0.500 dB` on truncated SH3 twin). Keep `--sh_degree 3` default; opt-in `--sh_degree 4` via warm-start only |
| 5 | ✅ **WP-R1: observed-preservation gate** (`_infusion/wpr1_RECORD.md`) | 2–3 d | CLOSED 2026-10-18 (`HOLD_DEFAULT_OFF`): 3-zone target blending (`--extra_unobserved_only --extra_anchor_observed 1.0 --extra_loss_weight 0.25`) beats `distill_control` on observed mLPIPS (`0.4603` vs `0.4632`) & solid-obs PSNR (`20.50` vs `20.19 dB`) and cuts NoVA by `-18%`, but 3D frustum bleed from dark SDXL inpaint still degrades fringe (`11.56` vs `16.51 dB`) |
| 6 | ✅ **WP-C3: TIDI-GS cross-view evidence & isolation pruner** (`_infusion/wpc3_tidi_RECORD.md`) | 2 d | CLOSED 2026-10-18 (`PASS`): `tools/prune_tidi.py` + `core/delivery.py --prune_tidi` removes Mip-Splatting `radii==0` zombie dust + unverified fog: `-29.5%` (`Laptop`), `-20.7%` (`GPU_V4`), **`-67.1%` (`360`: `569.7 → 187.3 MB`)** vs `prune_op05` at `0.0000..0.0003` mLPIPS delta and `0.0000` residual mid-opacity fog |
| 7 | **WP-M1: SAM 3.1 transient masks** → matcher + `loss_mask` | 1–2 d | Needs capture with transient people |
| 8 | ✅ **WP-D2: SPZ v4 writer** (`core/spz_io.py`, `_infusion/wpsh4/phaseA_gate.json`) | 0.5 d | CLOSED 2026-10-17: 6.4–7.4× smaller at SH bits 8/8, exact round-trip |
| 9 | ✅ **WP-B1 revisit: RealVDeblur** (`_infusion/wpb1_spirula_recon.md`) | 0.5 d | RECON CLOSED 2026-10-18: code (`OpenImagingLab/RealVDeblur`) + HF weights (`RBJin/RealVDeblur` on `Wan2.1-T2V-1.3B`) now public (Apache-2.0); parked until a motion-blurred capture arrives |
| 5b | **WP-C4: orbit-fill backbone bake-off** (ArtiFixer / LTX-2.5 inpaint / VACE / H3 probe), §5b | 3–4 d | Beat no-fill control on unobserved split; observed no-regress; cheating + flicker checks |
| 5c | ✅ **Spirula Studio recon** (`_infusion/wpb1_spirula_recon.md`) | 0.5 d | RECON CLOSED 2026-10-18: headless CLI (`spirula train`) confirmed on Vulkan/CUDA (GPL-3.0, SH3 max) |
| 10 | Fallback pose tier: MapAnything / VGGT-Ω-reproduce / VGGT-Long (WSL) | 2–3 d | Only for captures COLMAP fails |
| 11 | ✅ **WP-Perf1 + 4K Object Pipeline (`BOWL_C085`)** (`deliverables/bowl_c085_4k_summary.json`) | 1 d | CLOSED 2026-10-18 (`PASS`): GPU `uint8` VRAM preload + auto coarse-free, stateless `bfloat16` LPIPS, `fused=True` CUDA Adam (`2.23×`), separable 1D SSIM, zero per-step `.item()` syncs (`1.2 → 35.6 it/s` at 1080p, `20.7 it/s` at 4K), cosine MCMC noise decay, 2-stage horizon visual-hull carving (`--vote_horizon_frac 0.85`), pre+post-refit TIDI (`+0.25 dB` PSNR, `-12.7%` LPIPS, `2.4×` lower bg bleed vs 1080p) |

Deprioritized: G1-phase2 ASG (can't export to standard PLY without baking — revisit only if SH4
fails on glossy), UBS/new kernels, FlashVSR, Lyra, HY-Pano, PBR export, 3DGS-LM.
