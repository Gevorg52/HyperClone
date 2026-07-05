# WP-P1 — Generative-polish bake-off — RECORD (run 2026-07-04)

**Question:** which released enhancer best polishes our current-best model (the WP-S1 Laptop
STACK), judged so **frame-level gains only count if they survive into a final splat**? The
residuals being attacked (WP-S1 diagnosis): view-dependent specular smear on the glossy lid +
fine-microstructure softness (the on-screen checkerboard + keyboard/logo microtext).

**Candidates:** (1) no-op control (stack as-is = the bar); (2) ArtiFixer3D-bidir (incumbent);
(3) GeoQuery (2605.12399, github.com/Xiaoc7/GeoQuery); (4) SwiftVR (2606.09516, Wan2.2-TI2V-5B).
FlashVSR excluded up-front (hard block-sparse-attention dep = the known sm_120 killer).

---

## 0. No-op control bar (the WP-S1 Laptop STACK ckpt, rendered as-is)

Ckpt `_dataset/Laptop/apex_images/training/wps1/stack/output/ckpts/ckpt_29999_rank0.pt`
(708 MB, 3M Gaussians). Data `_dataset/Laptop/.../gsplat_prod/data`, factor 2, every-8 held-out
(38 views). Masked 6-scorer scorecard reused from WP-S1 (`qa/wps1_laptop_stack/scorecard.json`,
byte-identical protocol). Region boxes: **deck** (0.30,0.72,0.55,0.95) [WP-F1/F2 lineage],
**lid/screen** (0.30,0.72,0.18,0.58) [NEW — defined this WP: same horizontal extent as the deck,
vertical band over the glossy AORUS screen/bezel; the screen displays a high-freq orange
checkerboard in centered framings, the microstructure target]. Region JSONs
`_infusion/wpp1_noop_{lid,deck}_region.json` (deck reproduces WP-S1's 0.2739 EXACTLY → protocol
parity confirmed). Strips `qa/wpp1_noop_{lid,deck}_strips/`.

| metric | no-op (stack) | source |
|---|---|---|
| **masked-LPIPS** ↓ | **0.3096** | qa/wps1_laptop_stack/scorecard.json |
| masked-PSNR ↑ | 21.957 | " |
| masked-SSIM ↑ | 0.8001 | " |
| **deck region-LPIPS** ↓ | **0.2739** | wpp1_noop_deck_region.json |
| **lid region-LPIPS** ↓ | **0.2472** | wpp1_noop_lid_region.json |
| crossscore ↑ | 0.8636 | scorecard |
| tsed (consistency) ↑ | 0.939 | scorecard |
| raft-warp (consistency) ↓ | 0.0393 | scorecard |
| nvs-sqa (jod) ↑ | −2.868 | scorecard |

**GATE:** a candidate must beat this bar on masked-LPIPS AND the region crops at the SPLAT level,
with NO consistency regression (tsed/raft), to be adopted as the polish stage.

---

## 1. Feasibility outcomes per candidate (sm_120 / RTX PRO 6000 / torch 2.11)

Primary-source verified (researcher) + smoke-tested on the box where noted.

- **GeoQuery — FEASIBLE (smoke-tested on the box).** MIT, gsplat-integrated. Attention on the
  gsplat/inference path is hand-rolled local PyTorch (nn.Linear+matmul+softmax over a window),
  NO flash-attn/xformers/flex — zero sm_120 attention risk. Weights `DIG-UESTC/GeoQuery`
  `geoquery_window_size3_down1.pkl` (5.23 GB, ungated) + a hidden `stabilityai/sd-turbo` pull
  (~5 GB). Dedicated venv `venvs/geoquery` (--system-site-packages; diffusers==0.25.1 [pinned:
  UNet structure must match the .pkl], transformers==4.38.0, peft==0.9.0, hf-hub==0.25.1, cupy-
  cuda12x==13.4.1 for its softsplat forward-warp; xformers stripped [unused on this path]).
  **Smoke: model + sd-turbo load in 7s (cached), sample() forward 1.9s/view @512×904, numpy 2.2.6
  in-venv, torch 2.11 untouched.** cupy trap logged: cupy 14.x needs the `cuda` namespace pkg which
  a stray `C:\...\Python311\cuda.py` shadows → pin cupy 13.4.1 (old CUDA discovery, no cuda-pathfinder).
- **SwiftVR — FEASIBLE (deps only; not yet run).** Apache-2.0, Wan2.2-TI2V-5B video restorer.
  torch==2.10.0 pin in requirements → MUST `pip install --no-deps` + constraints (never clobber
  2.11). Multi-backend attention dispatcher falls through to plain SDPA (stock install = SDPA-only,
  first-class). Weights `H-oliday/SwiftVR` = **20.4 GB self-contained** (under the 30 GB flag; single
  download, no separate Wan2.2 base pull). 2D/video enhancer → distill-back is on us.
- **ArtiFixer3D-bidir — incumbent, already characterized (June/July).** Native flow: COLMAP →
  3DGRUT recon (720p) → bidirectional 2D inference (plain SDPA on sm_120) → run_artifixer3d distill
  into a fresh 3DGRUT. **Architecturally REPLACES the splat (builds its own 720p 3DGRUT); it cannot
  ingest our gsplat stack ckpt to "polish" it.** Its causal AR path needs block-sparse flash-attn =
  the sm_120 wall (ARTIFIXER_INVESTIGATION §2) → forbidden; bidirectional is the only local path.
- **FlashVSR — EXCLUDED.** Hard block-sparse "locality-constrained sparse attention" dep = the
  sm_120 killer. Not attempted.

(Results tables appended below as each arm is measured.)

---

## 2. GeoQuery — measured (frame-level [PC] + splat-level GATE)

**Protocol.** GeoQuery is a 2D geometry-guided single-step diffusion refiner (timestep 199) whose
gsplat example *resumes from a gsplat ckpt* and refines novel views in-loop as pseudo-GT. To keep
the A/B clean (identical base ckpt + trainer + eval, GeoQuery isolated) and normalization exact,
we drove it through OUR machinery: render targets from the WP-S1 stack ckpt, feed GeoQuery each
(target render, nearest-train reference GT image, reference MoGe metric depth, K, poses) → refined
image. Unit-consistency: GeoQuery's warp is similarity-invariant, so we pass RAW-COLMAP poses + RAW
MoGe depth for geometry while rendering from the normalized-frame ckpt (verified: pose convention
matches — `geometry_utils.colmap_pose_to_c2w` = our OpenCV c2w; warp = standard back-proj/reproj).
Drivers: `_infusion/wpp1_geoquery_{prep,refine,frame_eval}.py`.

### 2a. Frame-level [PC — frame-level, NOT the gate] — held-out renders refined 2D, LPIPS vs GT
`_infusion/wpp1_geoquery_frame.json` (factor-2 native) + `_infusion/wpp1_geoquery_frame_960.json`.

| box | no-op | GeoQuery @f2 | Δ | GeoQuery @960p | Δ |
|---|---|---|---|---|---|
| full | 0.3070 | 0.3387 | **+10.3%** | 0.3441 | +12.1% |
| deck | 0.2723 | 0.3112 | **+14.3%** | 0.3273 | +20.2% |
| lid  | 0.2455 | 0.2916 | **+18.8%** | 0.3047 | +24.1% |

**GeoQuery makes our frames WORSE at both resolutions** (960p even worse → not an off-distribution-
resolution artifact; robust). Visual: it re-synthesizes crisper/higher-contrast high-freq detail
(the on-screen checker looks sharper) but NOT GT-faithful → a sharp-but-misregistered pattern scores
worse on LPIPS than the soft-but-aligned no-op. This is the **"plausible-not-faithful"** signature:
GeoQuery is a SPARSE-view (3/6/9-view) artifact remover; our DENSE 266-view stack renders are already
near-artifact-free, so it has no genuine artifact to remove and its generative re-synthesis is
net-negative on fidelity. (The paper's +1.09 dB is on DL3DV *sparse-view* renders = heavy artifacts,
a different regime — [PC], researcher-verified.)

### 2b. Splat-level GATE — distill 60 refined novel views into the stack, score the splat
Distill: 60 midpoint-interpolated novel views (no leakage — held-out never enhanced/fed back),
refined by GeoQuery @f2, appended as extra_views (weight 1.0 uniform — NOT coverage-gated, since the
coverage gate would zero out exactly the well-observed views we're testing), resume-train stack ckpt
5k steps, no densification (`_infusion/wpp1_gq_refine_train.log`; ckpt
`qa/wpp1_gq_novel/refine_out/ckpts/ckpt_4999_rank0.pt`). Scored on the 38 held-out (masked 6-scorer
`qa/wpp1_geoquery/scorecard.json` + region `_infusion/wpp1_geoquery_{deck,lid}_region.json` + strips).

| metric | no-op | **GeoQuery-distill** | Δ | gate read |
|---|---|---|---|---|
| **masked-LPIPS** ↓ | **0.3096** | **0.3245** | **+4.8% WORSE** | **FAIL (gate metric)** |
| **deck region-LPIPS** ↓ | 0.2739 | 0.2776 | +1.3% worse | FAIL |
| **lid region-LPIPS** ↓ | 0.2472 | 0.2609 | +5.5% worse | FAIL |
| masked-PSNR ↑ | 21.957 | 23.376 | +1.42 dB | (smoother, not perceptual) |
| masked-SSIM ↑ | 0.8001 | 0.8428 | +0.043 | (smoother) |
| deck region-PSNR ↑ | 22.448 | 24.810 | +2.36 dB | (smoother) |
| crossscore ↑ | 0.8636 | 0.8455 | −0.018 worse | perceptual down |
| nova ↓ | 0.1221 | 0.1275 | +0.005 worse | perceptual down |
| **tsed** ↑ | 0.939 | 0.970 | +0.031 BETTER | consistency ↑ (no regression) |
| raft-warp ↓ | 0.0393 | 0.0401 | +0.0008 | ~equal |
| nvs-sqa ↑ | −2.868 | −2.870 | ~equal | — |

**GeoQuery GATE = FAIL.** The distill made the splat SMOOTHER (masked-PSNR +1.4 dB, SSIM +0.043,
region-PSNR +2.4 dB, tsed +0.031 = more L2-accurate AND more multi-view-consistent) but PERCEPTUALLY
WORSE (masked-LPIPS +4.8%, both region crops worse, crossscore/nova down). Judged by LPIPS (never
PSNR alone, §0) it does NOT polish our splat — multi-view fusion of GeoQuery's non-GT-faithful
re-synthesis averages into a slight blur. Visual (`qa/wpp1_geoquery_lid_strips/view0`): the distilled
checker is marginally softer than no-op. There is NO consistency regression (tsed↑, raft flat) — the
failure is purely perceptual-fidelity. **This is (to our knowledge) the first splat-level evidence
for GeoQuery, and it is a clean negative on our dense well-observed capture.**

Deviations/notes: refined at factor-2 native (best microstructure ceiling; 960p was worse); uniform
extra-view weight 1.0 (deviation from the coverage-gated distill default, justified above); 5k resume
steps / single scene (Laptop) / seed 0. cupy pinned 13.4.1 (see §1 trap).

---

## 3. SwiftVR — measured (frame-level [PC]); splat-level NOT run (justified)

**Feasibility EMPIRICALLY confirmed on sm_120** (nobody had run it on Blackwell): venv `venvs/swiftvr`
(--system-site-packages; diffusers 0.36.0, decord 0.6.0, accelerate/safetensors/einops/imageio; SwiftVR
installed `--no-deps` so its torch==2.10.0 pin never clobbers 2.11). **NEW sm_120 trap logged + fixed:**
diffusers 0.36's `models/attention_dispatch.py` EAGERLY imports a BROKEN `flash_attn_3` that sits in the
Windows *system* site-packages (leftover flash-attn-4 attempt — the sm_120 landmine); `find_spec` sees
it so `_CAN_USE_FLASH_ATTN_3=True` and the import crashes. Fix = venv-local try/except guard around the
flash-attn imports (falls back to SDPA). Then `SwiftVRPipeline.from_pretrained` + `restore_video`
(`--attention_backend sdpa --dtype bfloat16`) runs end-to-end: **304-frame orbit restored in 43s
(6.94 fps @1904×1072, GPU-free window).** Weights `H-oliday/SwiftVR` 20 GB (self-contained).

**Protocol.** SwiftVR is a temporal video super-res/restorer — it needs a contiguous sequence, so we
rendered the FULL orbit (304 frames, capture order) from the stack ckpt (`_infusion/wpp1_render_orbit.py`),
restored it (`_infusion/wpp1_swiftvr_restore.log`, out 301 frames, offset k=0 verified by global MSE
search: k=0 MSE 281 vs k≥1 4000+), and scored the 38 held-out restored frames vs GT
(`_infusion/wpp1_swiftvr_frame_eval.py`, restored 1904×1072 resized to GT for a fair A/B).

| box | no-op | SwiftVR | Δ | [PC frame-level] |
|---|---|---|---|---|
| full | 0.3070 | 0.4054 | **+32.0%** | strongly WORSE |
| deck | 0.2723 | 0.3944 | **+44.9%** | strongly WORSE |
| lid  | 0.2455 | 0.3899 | **+58.8%** | strongly WORSE |

**SwiftVR is strongly frame-level-negative — far worse than GeoQuery.** Visual
(`qa/wpp1_sv/diag_sv_view0_lid2x.jpg`): it re-synthesizes the on-screen checker at a DENSER, distorted
frequency (a resize cannot add checker squares → this is generative hallucination, not a resolution
artifact) — an aggressive 4x-trained SR model inventing non-GT-faithful detail on clean input. **Splat-
level distill NOT run:** a +32–59%-worse 2D signal is a foregone, even-larger splat regression than
GeoQuery's (which already FAILED the gate at only +5–19% frame-level), and the GPU is shared with a
long training run — spending GPU to confirm a certain larger regression is not warranted (§0: don't add
complexity for a foregone negative; record the honest result). Caveat: native-res restoration is
slightly off SwiftVR's designed 4x-SR regime, but the denser-checker hallucination shows the failure is
generative re-synthesis of already-clean content, not merely a resolution mismatch.

---

## 4. ArtiFixer3D-bidir (incumbent) — characterized from existing evidence (not re-run)

**Architectural verdict: ArtiFixer CANNOT polish our current-best stack.** Its native flow builds its
OWN 720p 3DGRUT from the COLMAP scene and distills into a fresh 3DGRUT — it does not ingest a gsplat
ckpt, so "polish the WP-S1 stack" is not a mode it supports; it would REPLACE our 2MP stack with a 720p
3DGRUT. Existing splat-level evidence (WP-Q1 rejudge, `qa/rejudge_easy_af3d`, same Laptop every-8 split,
720p): ArtiFixer3D masked-LPIPS **0.3059** vs its own base-3DGRUT **0.342** (−10.6%, clean 9/9 sweep) —
a real sharpen-observed lift *over its own weak base*, but its absolute masked-LPIPS (~0.306) only lands
≈ our no-op stack (0.3096) DESPITE being at HALF the resolution and a different trainer → NOT a valid
cross-comparison and NOT a splat-level win over the current-best. On well-observed views ArtiFixer3D-bidir
shows the same plausible-not-faithful sharpening pattern (ARTIFIXER_INVESTIGATION §5, WP-Q1 rejudge).
Re-running its full WSL pipeline on the stack scene was NOT done: foregone (replaces-not-polishes; weaker
720p base; the incumbent is already well-characterized). Causal ArtiFixer stays blocked on sm_120
(block-sparse flash-attn); only the bidirectional local path exists.

---

## 5. GATE VERDICT + adoption

| candidate | feasible sm_120 | frame-level [PC] | splat-level GATE | adopt as polish stage? |
|---|---|---|---|---|
| no-op (stack) | — | — | bar: mLPIPS 0.3096 / deck 0.2739 / lid 0.2472 | — (the bar) |
| **GeoQuery** | YES (smoke) | −(+10–19% worse) | **FAIL** (mLPIPS +4.8%, deck +1.3%, lid +5.5%; PSNR/tsed up but perceptually worse) | **NO** |
| **SwiftVR** | YES (empirical) | −(+32–59% worse) | not run (foregone larger regression) | **NO** |
| ArtiFixer3D-bidir | YES (bidir only) | +over its own weak base | replaces-not-polishes; ~= no-op at ½ res | **NO** |
| FlashVSR | NO (block-sparse) | — | — | excluded |

**VERDICT: NO released generative enhancer polishes our current-best stack → the honest negative.**
No candidate beats the no-op control on masked-LPIPS + the region crops at the splat level. The root
cause is a REGIME MISMATCH: every one of these enhancers (GeoQuery, SwiftVR, ArtiFixer) targets
sparse-view / degraded / compressed inputs with HEAVY artifacts; our WP-S1 stack is a DENSE 266-view,
well-observed, near-artifact-free capture, so the enhancers have little genuine artifact to remove and
their generative re-synthesis TRADES perceptual fidelity (LPIPS↓) for smoothness/PSNR (GeoQuery) or
outright hallucinates non-GT detail (SwiftVR). The residuals we set out to fix (specular-lid smear +
microstructure softness) are NOT artifacts these tools can faithfully recover — they are (a) a
fundamental view-dependent-radiance limit and (b) a factor-2 resolution/texture limit, neither of which
generative sharpening resolves without inventing detail.

**First splat-level evidence.** The researcher confirmed NOBODY has published splat-level (vs
frame-level) evidence for these enhancers. This WP produces the first: **GeoQuery, distilled into a
real production splat, REGRESSES perceptual quality on a dense well-observed capture** (mLPIPS +4.8%),
even though it raises PSNR/consistency. Frame-level gains (where they exist) do NOT survive as
perceptual-fidelity gains into the splat; where frame-level is negative (here), the splat inherits the
regression as a smoothing.

**Recommendation:** do NOT adopt a generative polish stage for this (dense, well-observed) capture
regime. Generative enhancement remains reserved for the COMPLETION lane (genuine holes / sparse /
under-observed, obs%<100 — WP-C1/C2, where GT metrics are invalid and plausible-fill is the goal), NOT
as a polish for already-well-observed splats. The WP-S1 residuals point instead to the in-stack
specular-color lever (Spec-Gaussian ASG, per WP-G1 phase-2) and higher training resolution for
microstructure — faithful levers, not generative re-synthesis.

---

## 6. Persisted artifacts (committable)
- `_infusion/wpp1_RECORD.md` (this file)
- No-op: `qa/wpp1_noop_{lid,deck}_strips/`, `_infusion/wpp1_noop_{lid,deck}_region.json`
  (deck reproduces WP-S1 0.2739 exactly), no-op scorecard reused `qa/wps1_laptop_stack/scorecard.json`
- GeoQuery: `_infusion/wpp1_geoquery_{prep,refine,frame_eval}.py`, `_infusion/wpp1_geoquery_smoke.py`,
  `_infusion/wpp1_geoquery_frame{,_960}.json` (frame-level), `_infusion/wpp1_geoquery_{deck,lid}_region.json`,
  `qa/wpp1_geoquery/scorecard.json` (splat 6-scorer), `qa/wpp1_geoquery_{deck,lid}_strips/`,
  `qa/wpp1_gq_frame/` (+ diag strips), `qa/wpp1_gq_novel/{manifest,extra_views}.json`,
  `_infusion/wpp1_gq_refine_train.log`; ckpt `qa/wpp1_gq_novel/refine_out/ckpts/ckpt_4999_rank0.pt`
  (gitignored? — under qa/, committable-sized 708 MB; recomputable from argv)
- SwiftVR: `_infusion/wpp1_render_orbit.py`, `_infusion/wpp1_swiftvr_frame_eval.py`,
  `_infusion/wpp1_swiftvr_frame.json`, `_infusion/wpp1_swiftvr_restore.log`,
  `qa/wpp1_sv/{orbit_manifest.json,diag_sv_*.jpg}` (in/out_full PNG dirs are bulky — recomputable)
- Clones/weights (gitignored, recomputable): `GeoQuery/` + `GeoQuery/weights/geoquery_window_size3_down1.pkl`
  (5.2 GB), `SwiftVR/` + `SwiftVR_weights/` (20 GB); venvs `venvs/{geoquery,swiftvr}`
  (venv-local patch: `venvs/swiftvr/.../diffusers/models/attention_dispatch.py` flash-attn guard)
- THIRD_PARTY.md: GeoQuery + SwiftVR entries added.
