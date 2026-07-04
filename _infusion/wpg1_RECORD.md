# WP-G1 PHASE 1 — 3DGS-DR specular-ceiling feasibility A/B — RECORD (run 2026-07-04)

**Question this WP answers (from WP-S1 residual diagnosis):** the view-dependent specular
reflections on the glossy laptop lid + GPU backplate render as a *mushy smear* in BOTH WP-S1
arms (SH-3 cannot track a moving mirror reflection). Does 3DGS-DR (deferred reflection: per-pixel
normal propagation + learned cubemap env map + deferred specular composite) render those moving
reflections materially better (region-LPIPS on the specular crops) than the current-best WP-S1
STACK? This is a **ceiling probe** run on the UPSTREAM 3DGS-DR code (no gsplat port yet); the
phase-2 port decision hinges on the result.

**Verdict: GATE FAILS on BOTH scenes → do NOT port 3DGS-DR (phase-2 negative).** On the specular
crops, 3DGS-DR's region-LPIPS is NOT materially better than the WP-S1 stack — it is **worse**:
Laptop deck +34.2%, GPU_V4 shiny backplate +3.1%. 3DGS-DR does render *more L2-accurate* backplate
shading (GPU_V4 region-PSNR +4.55 dB) but that does not become a perceptual (LPIPS) win — it comes
with softer etched microtext + bloomed metal highlights, and on the unbounded outdoor Laptop scene
it badly breaks the background (out of its bounded/object-centric domain). It does NOT deliver the
hoped-for sharp, view-consistent moving reflection. **Fallback per the research ranking: Spec-Gaussian
ASG (MIT, color-function-only swap into our gsplat MCMC) — see phase-2 recommendation.**

**⚠️ IMPORTANT DATA CORRECTION (reshapes WP-S2):** the `gsplat_prod/data/images_2` frames these
models train/eval on are **FULL SCENES, not pre-multiplied black-bg** (Laptop = bright outdoor patio;
GPU_V4 = warm indoor backdrop; measured: Laptop ~0% near-black px / corner-lum 126, GPU_V4 16.8% /
corner-lum 58). The WP-S1 stack ckpts render full backgrounds too (see strips). This contradicts the
DEVELOPER_NOTES / memory claim "current training data is pre-multiplied black-bg = object mode." The
queued **WP-S2 background study assumes production = object-only black-bg** — it must verify the actual
production data mode first, because its (b) "object-only pre-multiplied = current prod" arm may not
describe reality.

---

## 1. Build (sm_120, torch 2.11.0+cu128 untouched) — SUCCESS

Cloned `gapszju/3DGS-DR` @ `4077f1d4b45891885e6ebd1089da7bc51bee8afc` to `D:\REPOS\HYPERclone\3DGS-DR`.
Built its 4 CUDA submodules AOT for sm_120 via the house recipe (`_infusion/wpg1_build.bat`:
VS2022 vcvars64, `CUDA_HOME=...v12.9`, `TORCH_CUDA_ARCH_LIST=12.0`, `DISTUTILS_USE_SDK=1`,
`--no-build-isolation`, `-c tools/blackwell_session_2026-06/constraints.txt`).
`simple-knn`, `diff-gaussian-rasterization_c3` (vanilla base pass), `diff-gaussian-rasterization_c7`
(7-channel deferred rasterizer), `cubemapencoder` (env map) — all built + import OK on the RTX PRO 6000.

**sm_120 / torch-2.11 fixes applied to the clone (the known traps + one new one):**
1. `.data<T>()` → `.data_ptr<T>()` across all 4 submodules (`.data<T>()` removed in torch 2.11).
2. **NEW trap:** the `.cu`-reachable headers included the full `<torch/extension.h>` (and
   `<torch/torch.h>` in cubemapencoder), which under **nvcc + MSVC 14.44 + CUDA 12.9** pulls
   `torch/csrc/dynamo/compiled_autograd.h` and fails with `error C2872: 'std': ambiguous symbol`
   (at `::std::is_same_v<T, ::std::string>`). gsplat's own source avoids this by keeping
   `torch/extension.h` out of `.cu` files (uses pure ATen). Mirrored that: replaced the heavy
   include in the **6 nvcc-compiled files** (`simple-knn/spatial.h`, both `rasterize_points.{cu,h}`,
   `cubemapencoder.{cu,h}`) with a shim `#include <ATen/ATen.h>` + `<ATen/Dispatch.h>` +
   `namespace torch { using namespace at; }`; the cl-compiled pybind `.cpp` files keep
   `<torch/extension.h>`.
3. `torch::kFloat32`/`torch::kInt32` (torch-C++-API aliases, absent from ATen) → `torch::kFloat`/`kInt`
   (identical values).
4. `import open3d` made optional in `utils/general_utils.py` (used only by an unused PLY-writer).

Smoke (Laptop, 3500 iters): clean c3→c7 deferred transition, densify, and save of BOTH
`point_cloud.ply` + `point_cloud.map` (cubemap env-map sidecar) — env-map persistence confirmed.
c7-phase rate ~21.5 it/s at 2 MP.

## 2. Data export (GT-identical to WP-S1) — `_infusion/wpg1_export_colmap.py`

3DGS-DR is an Inria fork: its loader **asserts on the OPENCV camera model** and **cannot
undistort**, but our capture cameras ARE OPENCV (k1,k2≈0.1) and our trainer trains/evaluates on
**undistorted, ROI-cropped** frames. To (a) give 3DGS-DR a fair shot and (b) keep held-out GT
byte-identical to the WP-S1 stack, the export reuses OUR OWN `load_colmap_data`+`load_image` to
emit a classic PINHOLE COLMAP text model + the exact undistorted factor-2 images:

| scene | images (test) | init points (sparse ALIKED) | frame HxW | pose round-trip err |
|---|---|---|---|---|
| Laptop | 304 (38) | 93,126 | 1079×1919 | 9.06e-08 |
| GPU_V4 | 569 (72) | 123,289 | 1999×2999 | 8.22e-08 |

Split parity: 3DGS-DR sorts by image_name + holds out `idx%8==0`; our loader does the same on the
same registered set → **identical held-out views** (Laptop 38, GPU_V4 72 — matches WP-S1). Images
named by stem (`.png`) to preserve sort order. Init from the **sparse ALIKED** model (3DGS-DR's
designed regime; the dense-init advantage the WP-S1 stack had is geometry-axis, not specular-axis).

## 3. Training recipe (ceiling probe — documented deltas vs WP-S1) — `_infusion/wpg1_train.sh`

```
cd D:\REPOS\HYPERclone\3DGS-DR
python -u train.py -s <scene>/wpg1/3dgsdr_data -m <scene>/wpg1/3dgsdr_out --eval \
  --iterations 30000 --longer_prop_iter 24000 \
  --test_iterations 30000 54000 --save_iterations 54000
```
(black bg = no `--white_background`, matches our premultiplied black-bg frames; `-i images` and
`-r -1` defaults → full factor-2 res; TOT = 54001, normal-propagation window iter 3000→48000,
densify_until 54000.)

**Deltas vs WP-S1 stack (this is a CEILING probe, NOT a controlled A/B):**
- rasterizer: 3DGS-DR c7 **deferred reflection + cubemap env map** vs gsplat MCMC antialiased.
- densifier: Inria adaptive clone/split/prune vs gsplat MCMC (cap 3M).
- levers: 3DGS-DR has NONE of the WP-S1 stack levers (dense-init, mono-depth, erank, gated-LPIPS);
  WP-S1 stack has no specular modeling. Different axes.
- iterations: 54k (30k base == WP-S1 for comparability + 24k longer_prop). Reduced from the
  authors' real-scene 61k/36k for GPU coordination (WP-S2 queued); core geometry converges by 30k
  (`position_lr_max_steps=30000`), the tail is refl/normal refinement.
- no `--use_env_scope`: our bg is masked black → little background to suppress; a mis-sized sphere
  would wrongly kill object specular (false-negative risk). Env-scope suggestion stored in each
  scene's `wpg1_export_meta.json` in case a re-run is warranted.

## 4. Evaluation protocol — `_infusion/wpg1_render_dr.py` (stage 1) + `_infusion/wpg1_eval.py` (stage 2)

Deferred-reflection splats carry extra env/normal/refl params + a cubemap that do NOT round-trip to
standard PLY, so we evaluate via **3DGS-DR's own renders** of the held-out views (stage 1: load the
trained model incl. the `.map` env-map, render every test cam with the c7 deferred path, save
`<stem>.png`). Stage 2 scores 3DGS-DR renders **and** the WP-S1 stack (rendered live from its gsplat
ckpt) against the **same canonical undistorted `load_image` GT**, with the **same LPIPS**
(torchmetrics VGG normalize=True) and the **same region boxes** as WP-S1:
Laptop **deck** (0.30,0.72,0.55,0.95); GPU_V4 **shiny backplate** (0.28,0.72,0.30,0.55).
Gate metric = **region-LPIPS** on the specular crop; global sanity = full-frame LPIPS/PSNR + obj-PSNR.

## 5. RESULTS

**Important data note discovered at eval:** neither scene's `gsplat_prod/data/images_2` frames are
black-bg — BOTH are **full scenes** (Laptop = bright outdoor patio: house wall, garden, palm, chair;
GPU_V4 = darker/closer indoor-ish backdrop, more object-centric). So the "pre-multiplied black-bg"
note does not describe these training frames. The WP-S1 stack was trained on the SAME frames (shared
junctions), so the A/B is still fair; but it means 3DGS-DR must reconstruct the whole scene, not just
a masked object. My eval reproduces the WP-S1 stack region-LPIPS **exactly** (Laptop deck 0.2739) —
strong confirmation the GT/LPIPS/box/render match WP-S1's own methodology.

### Laptop — 3DGS-DR LOSES (region-LPIPS worse on every region; global much worse)
Model: 54k iters, 1,325,062 Gaussians, **351,328 (26%) with refl>0.1** (reflections DID develop).
No env_scope (full outdoor scene; ALIKED sparse points too spread to define a clean object sphere).
Scorecards `qa/wpg1_laptop/scorecard.json` (deck) + `scorecard_lid.json`; strips `qa/wpg1_laptop_strips/`,
`qa/wpg1_laptop_lid_strips/`.

| region | metric | wps1-stack | 3dgs-dr | Δ (dr−stack) | verdict |
|---|---|---|---|---|---|
| **deck** (0.30,0.72,0.55,0.95) | **region LPIPS** ↓ | **0.2739** | 0.3675 | **+0.0936 (+34.2%)** | **stack better ✓** |
| deck | region PSNR ↑ | 22.448 | 23.613 | +1.17 dB | dr higher (=smoother, not sharper) |
| upper box (0.19,0.53,0.16,0.52)* | region LPIPS ↓ | 0.2530 | 0.3564 | +0.1034 (+40.9%) | stack better ✓ |
| full frame | full LPIPS ↓ | 0.3096 | 0.4095 | +0.0999 (+32.3%) | stack much better ✓ |
| full frame | full PSNR ↑ | 21.958 | 22.170 | +0.21 dB | ~equal |

*(the "upper box" I intended as a lid crop actually lands on near-background — tree/chair/house —
because the laptop lid is small; it still shows the failure clearly.)*

**Visual (view8=frame_0065 strips):** 3DGS-DR **badly breaks the outdoor background** — the house
wall + bushes wash out to a white haze, the palm trunk warps, surroundings smear; the WP-S1 stack
keeps the background coherent (soft but structurally correct). On the laptop deck/keyboard the two
are comparably soft (DR marginally softer → higher PSNR, worse LPIPS). The glossy-lid reflection is
mushy in BOTH — 3DGS-DR does **not** render a sharper/view-consistent lid reflection here; any
specular benefit is swamped by its much worse whole-scene geometry (sparse-init + Inria adaptive
densify + domain mismatch — its real scenes are bounded/object-centric, not an unbounded outdoor
patio; + the reduced 54k budget). **Laptop = clear loss for 3DGS-DR.**

### GPU_V4 — WASH (region-LPIPS marginally worse for DR; big region-PSNR win, no perceptual gain)
Model: 54k iters, **WITH env_scope** center=(0.133,0.37,0.107) r=1.0 (compact object → clean sphere;
authors' real-scene recipe = DR's best shot). 351,581 Gaussians, **17,031 (4.8%) refl>0.1** (env_scope
correctly localized reflectance to the object). Scorecard `qa/wpg1_gpu/scorecard.json`; strips
`qa/wpg1_gpu_strips/` (view48 backplate, view24 fins).

| region | metric | wps1-stack | 3dgs-dr | Δ (dr−stack) | verdict |
|---|---|---|---|---|---|
| **shiny backplate** (0.28,0.72,0.30,0.55) | **region LPIPS** ↓ | **0.3473** | 0.3580 | **+0.0107 (+3.1%)** | **stack better (marginal)** |
| shiny | region PSNR ↑ | 22.480 | 27.033 | **+4.55 dB** | **dr much better (L2 tone)** |
| full frame | full LPIPS ↓ | 0.4072 | 0.4294 | +0.0222 (+5.5%) | stack better |
| full frame | full PSNR ↑ | 22.933 | 26.080 | +3.15 dB | dr much better |
| full frame | obj PSNR ↑ | 22.746 | 25.882 | +3.14 dB | dr much better |

**Visual (view48 backplate / view24 fins):** unlike Laptop, DR reconstructs the WHOLE GPU_V4 scene
coherently (backplate text, ZOTAC logo, polka-dot cloth, warm background all clean — the bounded,
object-centric scene + env_scope suit its domain). On the specular backplate DR matches the overall
dark semi-gloss tone much better (→ the +4.5 dB PSNR) BUT **softens the fine etched circuit traces +
"LIVE TO GAME" microtext and renders the chrome ZOTAC logo as a hazy bloom/glow** rather than a crisp
edge; the stack keeps those high-freq details sharper (→ its better LPIPS). On the fins both are
comparably soft, DR slightly more so with mildly bloomed metal highlights. Net: a **wash** — DR is more
L2-accurate but not perceptually sharper, and shows NO clean "sharp view-consistent moving reflection"
advantage. env_scope worked (localized reflectance) but did not flip the LPIPS result.

---

## 6. GATE VERDICT + PHASE-2 RECOMMENDATION

**GATE (region-LPIPS on the specular crops materially better for 3DGS-DR) = FAILS on both scenes.**

| scene | region-LPIPS stack→dr | Δ | region-PSNR Δ | global (full-LPIPS Δ) | read |
|---|---|---|---|---|---|
| Laptop (deck) | 0.2739 → 0.3675 | **+34.2% worse** | +1.17 dB | +32.3% worse | clear loss (broken bg) |
| GPU_V4 (backplate) | 0.3473 → 0.3580 | **+3.1% worse** | +4.55 dB | +5.5% worse | wash (PSNR↑, LPIPS↓) |

Judged by LPIPS (the metric we trust for perceptual quality — never PSNR alone, per §0), 3DGS-DR does
**not** beat the WP-S1 stack on the specular surfaces on either scene, and it does not render the
moving reflections sharper/more-view-consistent (Laptop lid stays mushy; GPU logo blooms). The GPU_V4
+4.5 dB PSNR is a real signal that deferred shading captures the semi-gloss *tone* better, but it does
not translate to perceptual sharpness and it degrades the fine microstructure.

### Recommendation: DO NOT port 3DGS-DR into gsplat (phase-2 negative). Ranked follow-ups:

1. **Spec-Gaussian ASG** (MIT; anisotropic-spherical-gaussian view-dependent color function) —
   **RECOMMENDED next specular lever.** It is the next-cheapest per the research ranking: a
   *color-function-only swap* that stays inside our gsplat MCMC + dense-init + faithful-levers stack
   (no whole-pipeline deferred-shading/env-map replacement, no Inria-densifier dependency, none of
   3DGS-DR's unbounded-scene fragility). It targets exactly the view-dependent specular color the
   GPU_V4 PSNR signal hints is under-modeled, without abandoning the geometry cleanup the stack won on.
2. **Accept-the-limitation + generative polish** (GeoQuery / SwiftVR bake-off) — the residual is
   partly a *fundamental* view-dependent-radiance limit (a moving mirror reflection a finite splat
   cannot track) AND partly fine-microstructure softness that is a factor-2 resolution/texture limit
   BOTH arms (and 3DGS-DR) share. Generative sharpening is the higher-ceiling, higher-complexity path
   and is worth the bake-off regardless of the specular-color lever (it also addresses the microtext
   softness). Judge on OBSERVED surfaces (GT metrics valid — this is not a fill task).
3. **3DGS-DR full port** — NOT recommended. Ceiling probe is negative on LPIPS; its deferred-shading +
   env-map + Inria-densify pipeline is a poor fit for our unbounded captures and does not clear the gate.

### Deltas / caveats affecting the read (honesty)
- Ceiling probe, NOT a controlled A/B: 3DGS-DR = its own c7 deferred rasterizer + cubemap env map +
  Inria adaptive densify + sparse ALIKED init + NONE of the WP-S1 stack levers, at 54k iters (30k base
  + 24k longer_prop, reduced from the authors' real-scene 61k/36k for GPU coordination). Laptop had no
  env_scope (spread points); GPU_V4 had env_scope (compact object). All documented above.
- The reduced iteration budget could under-develop reflections, but reflections DID develop (Laptop 26%,
  GPU_V4 4.8% refl>0.1) and GPU_V4 got the authors' env_scope recipe — so this is a fair-enough ceiling.
  Even given every reasonable advantage on the cleaner GPU_V4 scene, the LPIPS result did not flip.

## 7. Persisted artifacts (committable)
- `_infusion/wpg1_RECORD.md` (this file)
- `_infusion/wpg1_build.bat` (sm_120 CUDA build), `_infusion/wpg1_build.log` (build log)
- `_infusion/wpg1_export_colmap.py` (GT-identical PINHOLE export), `_infusion/wpg1_train.sh` (exact argv),
  `_infusion/wpg1_render_dr.py` (stage-1), `_infusion/wpg1_eval.py` (stage-2 A/B)
- `_infusion/wpg1_train_laptop.log`, `_infusion/wpg1_train_gpu.log`, `_infusion/wpg1_render_gpu.log`
- `qa/wpg1_laptop/scorecard.json` (+ `scorecard_lid.json`), `qa/wpg1_gpu/scorecard.json`
- `qa/wpg1_laptop_strips/`, `qa/wpg1_laptop_lid_strips/`, `qa/wpg1_gpu_strips/` (GT|stack|dr full + region2x)
- 3DGS-DR clone + sm_120-fixed submodules at `D:\REPOS\HYPERclone\3DGS-DR` (not committed; gitignored
  `_dataset` holds the exported datasets + trained models — recomputable from the argv above)
- THIRD_PARTY.md: 3DGS-DR provenance + build-fix ledger.
