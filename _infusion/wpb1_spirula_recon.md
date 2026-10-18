# WP-B1 Revisit (`RealVDeblur`) & Spirula Studio Recon — Gate Record (2026-10-18)

## 1. WP-B1 Revisit: RealVDeblur (`arXiv:2607.20628`, NeurIPS 2025)

### Question (`docs/RESEARCH_REFRESH_2026-10.md` §1.2 & §6 #9)
Did `RealVDeblur` publish inference code and pretrained weights since the April 2026 check?

### Verified Upstream Status (2026-10-18)
- **GitHub Repository:** `https://github.com/OpenImagingLab/RealVDeblur` (redirects to `https://github.com/RBJin/RealVDeblur`, commit `63a2eb0` — *"First release of our code"*).
- **License:** Apache-2.0.
- **Public Inference Code:** **YES** — `inference.py`, `model/src/`, `utils/`, `requirements.txt`, and `assets/` are now published in the repository.
- **Public Pretrained Weights:** **YES** — hosted on HuggingFace at `https://huggingface.co/RBJin/RealVDeblur`:
  - `realvdeblur_dmd.safetensors` (one-step Distribution Matching Distillation generator + blur-aware conditioning extractor + lightweight video decoder).
  - Base model dependency: `Wan-AI/Wan2.1-T2V-1.3B` (`Wan2.1_VAE.pth` + `models_t5_umt5-xxl-enc-bf16.pth` + `Wan2.1-T2V-1.3B` diffusion backbone).
- **Inference Command:**
  ```bash
  python inference.py \
      --input_path ./assets \
      --output_path ./output \
      --transformer_path ckpts/RealVDeblur/realvdeblur_dmd.safetensors \
      --base_model_path ckpts/Wan2.1-T2V-1.3B
  ```
- **Architecture & Compute Footprint:**
  - Built on **Wan2.1-T2V-1.3B** (video diffusion backbone distilled via DMD to single-step inference), requiring ~10–14 GB download (`Wan2.1-T2V-1.3B` + `umt5-xxl` + `realvdeblur_dmd.safetensors`) and `diffusers` / `transformers` / `safetensors`.

### Gate Status & Blocker
- **Code & Weights Blocker:** **CLEARED** (both inference code and `realvdeblur_dmd.safetensors` are now publicly available under Apache-2.0).
- **Dataset Blocker:** **STILL BLOCKED ON BLURRY CAPTURE** — none of our three production benchmark datasets (`Laptop`, `GPU_V4`, `360` FlowerShop) suffer from severe camera-shake motion blur (`Laptop` and `GPU_V4` are sharp studio turntable/orbital captures; `360` is a tripod walkthrough). Per `docs/EXECUTION_PLAN.md` §WP-B1, deblur preprocessing is only triggered on frames in the bottom 15% of Tenengrad sharpness (`--deblur_preprocess`).
- **Recommendation:** Keep `WP-B1` parked as an opt-in pre-SfM preprocessor (`--deblur_preprocess realvdeblur`) until a handheld motion-blurred walkthrough video arrives in `_dataset/`. When needed, download `RBJin/RealVDeblur` + `Wan-AI/Wan2.1-T2V-1.3B` into an isolated venv (`venvs/realvdeblur`).

---

## 2. Spirula Studio Recon (`docs/RESEARCH_REFRESH_2026-10.md` §5c)

### Question (`docs/RESEARCH_REFRESH_2026-10.md` §5c)
Does `harry7557558/spirula-studio` expose a headless CLI for batch benchmarking alongside its GUI, and what are its capabilities relative to `gsplat` + `Splatbus`?

### Verified Upstream Status (2026-10-18)
- **GitHub Repository:** `https://github.com/harry7557558/spirula-studio` (commit `74a92ef`, GPL-3.0 license).
- **Headless CLI Support:** **YES** — Spirula Studio exposes a full CLI interface in addition to its native C++ GUI:
  - `spirula --help`
  - `spirula train --help` (supports headless training from COLMAP scenes, YAML config export/import from GUI, checkpoint export, and dense point cloud export).
- **Compute Backend:**
  - Primarily **Vulkan** (custom hand-written Cross-Vendor Vulkan compute shaders via Kompute/glslang; recommended by upstream over CUDA for stability and speed), with an optional CUDA backend.
  - Prebuilt binaries are published on GitHub Releases for Windows, Linux, and macOS.
- **Supported Features:**
  - Standard 3DGS + 3DGUT, **360° equirectangular camera**, fisheye/distorted cameras, **LoMa** (local feature matching integration), bilateral grid / PPISP-style appearance modeling, and background/mask training.
  - **SH Degree Limit:** Supports up to **SH degree 3** (`sh_degree <= 3`); does **not** currently support SH degree 4 (`25` coefficients) or our custom gates (`--erank_lambda`, `--lpips_gated`, `--prune_tidi`, `--extra_unobserved_only`).
- **License Note:** Released under **GPL-3.0**, so it cannot be linked into `apex-sfm` core code, though it can be invoked as an external standalone tool if desired.
- **Verdict:** No action required for our production pipeline (`gsplat 1.5.3` / `1.6.0` + `ppisp` + `prune_tidi` + `SH4` covers all required capabilities with permissive Apache-2.0 / MIT licensing), but `spirula train` is confirmed viable as an external cross-vendor Vulkan reference tool.
