# Apex Reality — Developer Notes & Engineering Log

Canonical "read this first" reference for anyone working on this SfM→3DGS pipeline.
Captures the build environment, the trainer's quality knobs, and — most valuably —
the **empirical learnings and dead-ends** from the June 2026 quality investigation,
so you don't relearn them the hard way.

Companion docs (deeper detail):
- `docs/quality-roadmap.md` — the full quality investigation, ranked levers, and
  every empirical update (cap_max, geo-reg, LPIPS, WD-R), with the reasoning.
- `docs/research-findings.md` — literature/SOTA research (extractors, matchers,
  trainers, enhancement, June 2026).
- `docs/brief.txt`, `docs/rules.txt` — original product brief and working rules.

---

## 1. Pipeline overview

GUI (Gradio) pipeline, JPEG stills (or extracted video frames) → COLMAP-format
sparse reconstruction → 3D Gaussian Splatting → optional diffusion enhancement.
Target: max-quality, artifact-free splats from ordinary footage. RTX 4090 (24 GB)
now; Blackwell Pro 6000 (96 GB) later. **Quality ≫ speed. RESEARCH-ONLY** — license
restrictions are not a selection filter (we reimplement best methods as our own OSS).

10 stages (`gui/stages/`): 1 import/QC · 2 ALIKED extraction · 3 LightGlue matching
(+ optional RoMa v2 dense rescue) · 4-5 COLMAP recon (incremental or GLOMAP global) ·
6 final BA · 7 export (+ gravity alignment) · 8 masking (BiRefNet + SAM 2.1) ·
9 gsplat training · 10 enhancement (NVIDIA Fixer + distill-back loop).
Entry point: `run.py` (launches the Gradio app). Core logic in `core/`; the trainer
is a self-contained subprocess (`core/training/gsplat_trainer.py`).

---

## 2. Environment & build (CRITICAL — read before touching CUDA code)

- **There is NO named conda env.** The whole stack lives in the **anaconda `base`
  env** (`C:\Users\gevor\anaconda3`, **Python 3.11.5**). Activate:
  `& C:\Users\gevor\anaconda3\shell\condabin\conda-hook.ps1; conda activate base`.
- **Torch:** 2.3.0+cu121. **CUDA toolkits installed:** v11.7, v11.8 (default
  `CUDA_PATH`!), v12.1. For any CUDA build, set `CUDA_HOME`/`CUDA_PATH` to v12.1.
- **MSVC pin:** VS 2022's default toolset (14.4x) STL hard-errors with CUDA < 12.4
  (`STL1002`). Use `vcvars64.bat -vcvars_ver=14.34` (toolsets 14.16–14.38 installed).
- **gsplat** is built **AOT from source** (the PyPI wheel is pure-Python = JIT-only,
  which needs `cl.exe` in *every* process → unusable for the GUI). Recipe:
  ```
  set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
  set "CUDA_PATH=%CUDA_HOME%"
  set "PATH=C:\Users\gevor\anaconda3;...\Scripts;...\Library\bin;%CUDA_HOME%\bin;%PATH%"
  set "TORCH_CUDA_ARCH_LIST=8.9+PTX"   (4090 = sm_89)
  set "DISTUTILS_USE_SDK=1"            (required when VC env active, else torch build aborts)
  set "MAX_JOBS=16"
  call "...\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.34
  pip install --force-reinstall --no-deps --no-build-isolation --no-cache-dir \
    "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3"
  ```
  Verify: `python -c "from gsplat import csrc"`. (Current pin: gsplat **1.5.3**.)
  - `core/training/gsplat_jit_shim.py` fixes a gsplat<=1.5.3 bug where it passes
    GCC-style flags to MSVC (D8021) — imported before gsplat everywhere.
- **pycolmap 4.0.4** (COLMAP 4.0: GLOMAP merged, native ALIKED+LightGlue, faster BA,
  deterministic `random_seed`). NOTE: pycolmap 4.x made `Image.cam_from_world` a
  **method** (was a property in 3.x) and moved BA solver opts to
  `BundleAdjustmentOptions.ceres.solver_options` — see `core/alignment._cam_from_world`
  and `gui/stages/export._run_standalone_ba` for the version-robust handling.
- **Gotchas seen:** cv2 "DLL load failed" → `pip install --force-reinstall --no-deps
  opencv-contrib-python==4.11.0.86`. `ninja` must be on PATH for JIT. In `cmd /c`
  chains, `%PATH%` expands at parse time — set PATH BEFORE `call vcvars64.bat`.

### Running it
- GUI: `python run.py`.
- Headless SfM (stages 1-7): see `run_pipeline_laptop.py` for a worked example —
  drives the real stage handlers with a no-op `progress` stub.
- Training is a subprocess; flags below. Tests: `pytest tests` (134 passing;
  `tests/smoke_gsplat.py` needs a VS dev shell for first JIT).

---

## 3. The 3DGS trainer — quality knobs & recommended settings

`core/training/gsplat_trainer.py` (CLI; driven by `GsplatBackend`). Key flags:

| Flag | Default | Notes |
|---|---|---|
| `--data_factor` | 2 | training resolution = source / factor. **Relative to source!** 24MP→factor2=6MP; 4K→factor1=8MP. Pick by target MP, not habit. |
| `--strategy` | mcmc | `mcmc` (MCMC-3DGS, better quality + budget control) or `default`. |
| `--cap_max` | 1e6 | MCMC Gaussian budget. **~3M is near-optimal for a single object+scene; more HURTS** (see §6). |
| `--antialiased` | on | Mip-Splatting (`rasterize_mode`). Keep on. |
| `--pose_opt` | off→on(GUI) | SE(3) per-camera refinement, lr 1e-5. |
| `--coarse_factor`/`--coarse_until` | — | coarse-to-fine: densify at lower res, finish at full. |
| `--opacity_reg`/`--scale_reg` | 0.01 | MCMC regularizers. |
| `--geo_reg` | off | edge-aware curvature depth smoothness. **No effect on textureless mush — leave off** (§6). |
| `--lpips_lambda` | 0 | perceptual loss weight. 0=off. ~0.2 at factor≥2, gated (below). |
| `--lpips_gated` | on | **gate LPIPS toward under-resolved (high-error) crops** so it doesn't over-texture well-resolved regions. Makes LPIPS safe at any resolution. |
| `--lpips_start`/`--lpips_patch` | 3000/512 | start step / random-crop size (bounds 4K VRAM). |

**Recommended by scenario:**
- **Object/scene, 24MP stills:** factor 2, mcmc, cap 1-3M, antialiased, pose_opt,
  coarse-to-fine. (GPU_V4 run: 25.3 PSNR, 6.8 GB VRAM.)
- **4K video:** factor 1 (don't waste the 4K — see §5), cap ~3M, coarse 2→1.
- **Perceptual sharpness:** `--lpips_lambda 0.2 --lpips_gated`; on full-res
  coarse-to-fine the GUI auto-starts LPIPS in the fine phase only.

VRAM at 4K/3M ≈ 6.8 GB (peak set by the KNN init `cdist`, not training) — lots of
headroom on a 4090, but **don't spend it on more Gaussians** (§6).

---

## 4. Key learnings (evidence-based, June 2026)

- **SfM is not the bottleneck.** ALIKED+LightGlue+COLMAP registers ~100% with ~1.5px
  reproj even on hard subjects. IMC-2025 winners use the same stack.
- **MCMC densification + Mip-Splatting are the proven training-quality base.**
- **3M Gaussians is near-optimal**; the detail ceiling is the *capture*, not the
  primitive budget (§6).
- **Decoupled resolution holds:** SfM features at 1600px (COLMAP thresholds are
  ~1080p-calibrated); train 3DGS at full res. **Never super-resolve input frames**
  (per-frame SR → view-inconsistent → 3DGS averages to blur + Gaussian bloat).
- **Perceptual (LPIPS) loss helps perceived sharpness** — clean win at ≤1080p /
  under-resolved content; at full 4K a *global* LPIPS is a wash (over-textures
  well-resolved regions). **Gating it by relative crop error fixes that** → safe
  at any resolution. Judge perceptual changes by **LPIPS/crops, never PSNR**
  (perceptual loss deliberately trades PSNR).
- **Textureless/flat/dark/semi-gloss surfaces are capture-limited.** They're
  Gaussian-starved because MCMC samples relocation by **opacity only**
  (`gsplat/strategy/ops.py`: `probs = opacities.flatten()`) → drains low-error flat
  regions by design. You can't reconstruct detail the photons didn't carry; the
  fixes are capture-side (cross-pol, matte spray, RAW) or generative synthesis.
- **Diffusion enhancement = synthesis, not measurement.** Our NVIDIA Fixer (=Cosmos
  NuRec Fixer) + the Difix3D+-style distill-back loop (`core/enhancement/distill.py`)
  add *plausible* detail; reserve for cosmetic polish.

---

## 5. Video input

Pipeline expects stills. Extract frames with ffmpeg (`-r 1` ≈ 1 fps gave ~300 frames
from a 5-min 4K clip — plenty for a single-object orbit). **Train at factor 1 for 4K
source** (factor 2 throws away 75% of captured pixels). For a continuous orbit,
retrieval (NetVLAD) matching is the designed path for >300 frames.

**KNOWN BUG (TODO):** the Stage-1 sharpness auto-cull default (threshold 200) is
tuned for 24MP DSLR stills and is **wrong for video** — 4K video frames score much
lower (median ~138), so the default would cull ~74% of them, and the metric
conflates "dark/low-texture" with "blurry" (it flagged sharp-but-dark lid close-ups).
Fix: relative/data-driven threshold, or Tenengrad windowed sharpest-per-partition
selection. For now, for video, **don't auto-cull** (let COLMAP's geometric
verification drop unregisterable frames).

---

## 6. What was tried and did NOT work (do not repeat)

- **Raising `cap_max` above ~3M:** 8M *diverged* (PSNR 23→10 as count grew past 3M)
  and OOM'd (overlapping giant Gaussians → tile-intersection blowup, 33 GB). More
  Gaussians actively hurt this scene. 3M is near-optimal; budget is not the ceiling.
- **Edge-aware depth-smoothness (`--geo_reg`):** zero effect on textureless mush —
  it's a *geometry* lever, the mush is lost *appearance* detail. Gradients verified
  to flow (not a no-op); just the wrong lever. Kept off by default.
- **Global LPIPS at full 4K:** over-textures already-well-resolved regions (keyboard
  LPIPS 0.274→0.327 @ λ0.2; PSNR −2 dB). → led to **gated** LPIPS.
- **Apple WD-R (arXiv 2603.23297):** not worth it — code unreleased (project page
  only), ships spatially-uniform (σ=4, same blind spot as global LPIPS), full-image
  4K VGG hostile to 24 GB, gains benchmarked vs blurry baselines not LPIPS-as-loss.
- **Depth/normal priors for *detail*:** they fix geometry, not albedo sharpness
  (DN-Splatter/DNGaussian treat depth↔color as a trade-off).
- **2DGS/GOF/RaDe-GS for sharpness:** over-smooth appearance (~−1 dB).

---

## 7. Open items / future work

- Fix the Stage-1 video sharpness-cull bug (§5).
- ~~Validate gated LPIPS at 4K~~ DONE: 4-way 4K A/B confirms gated λ0.2 improves
  the deck (LPIPS 0.318→0.313, sharper) with the keyboard EXACTLY unharmed (0.274 =
  original), vs global λ0.2 which wrecked the keyboard (0.327). Gated LPIPS is the
  production default for the perceptual lever; deck gain is modest (capture-limited)
  but free and safe at any resolution.

### Artifact correction (June 2026)
- **DONE — faithful floater cleanup** (`core/training/prune.py`, Stage 9 GUI
  "Floater cleanup"): removes near-invisible (opacity<0.01) + spatial-outlier
  Gaussians. Validated lossless (3M→1.34M, PLY 708→316 MB, held-out PSNR/LPIPS
  unchanged). **Do this first** for any clip — faithful, no model, no Docker.
- **SCAFFOLDED — native Difix3D+ backend** (`core/enhancement/difix_backend.py`,
  registered as enhancement backend `difix`): WSL-FREE diffusion artifact cleaner
  (the Docker dependency was the *separate* Cosmos NuRec Fixer, not Difix). Drops
  into the distill-back loop (`core/enhancement/distill.py`). To enable: install in
  an ISOLATED env (`diffusers==0.25.1 transformers==4.38.0 peft==0.9.0 accelerate`);
  weights `nvidia/difix` auto-download. **Use only for clips with real
  floater/ghosting artifacts — NOT for capture-limited textureless surfaces (it
  hallucinates, doesn't recover).** WD-R was researched and rejected (see §6).
- Enhancement backends now: `fixer` (Cosmos, Docker — avoid) and `difix` (native).
- Re-pull `nvidia/Fixer` V2 weights; run the distill-back loop (needs Docker up).
- Optional installs (research-only, not yet wired): RoMa v2 dense rescue
  (`core/roma_rescue.py`), Faster-GS backend (`docs/research-findings.md` has setup).
- Capture playbook for hard surfaces: cross-polarization, matte/scanning spray, RAW
  expose-for-shadows — highest ROI for textureless/glossy subjects.
- Watch list (no code yet): GloSplat, SSR-GS, 3DGRUT/3DGUT (Blackwell-era).

---

## 8. Versions (June 2026)
gsplat 1.5.3 (AOT) · pycolmap 4.0.4 · torch 2.3.0+cu121 · HLoc (ALIKED-n16 +
LightGlue) · BiRefNet + SAM 2.1 · Gradio 5.x · Python 3.11.5 (anaconda base).
