# PPISP

Code for the work "PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction".

[Project website](https://research.nvidia.com/labs/sil/projects/ppisp/) | [Paper](https://arxiv.org/abs/2601.18336)

## Overview

PPISP is a learned post-processing module for radiance field rendering that models common photometric variations found in real-world multi-camera video captures:

- **Exposure compensation** (per-frame): Corrects brightness variations caused by auto-exposure or changing lighting conditions
- **Vignetting** (per-camera): Models radial light falloff from lens optics
- **Color correction** (per-frame): Handles white balance drift and color cast variations via chromaticity homography
- **Camera Response Function (CRF)** (per-camera): Learns the nonlinear tone mapping applied by each camera's ISP

The module also includes a **controller** that predicts per-frame exposure and color corrections from rendered radiance images. This enables consistent novel view synthesis without requiring per-frame parameters at inference time.

All processing is implemented as a differentiable CUDA kernel for efficient integration into training pipelines.

## Adding PPISP as a Dependency

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use. See [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for a list of dependencies and their licenses.

### Build System

PPISP uses a two-file build configuration:

- **`pyproject.toml`** declares build-time requirements (`setuptools`, `wheel`) and runtime dependencies (`torch`)
- **`setup.py`** compiles the CUDA extension using `torch.utils.cpp_extension`, which requires PyTorch to be available during the build process

To compile the CUDA extension against the exact PyTorch version in your environment for ABI compatibility, we recommend installing with `--no-build-isolation`.

### Installation

**Local development install:**

```bash
pip install . --no-build-isolation
```

**As a dependency in `requirements.txt`:**

Add the following to your `requirements.txt`, optionally pinning a specific version:

```
ppisp @ git+https://github.com/nv-tlabs/ppisp.git@v1.0.0
```

Then install with the `--no-build-isolation` flag:

```bash
pip install -r requirements.txt --no-build-isolation
```

## Integration into Existing Radiance Field Reconstruction Pipelines

### Basic Usage

```python
from ppisp import PPISP, PPISPConfig

# 1. Initialize
ppisp = PPISP(num_cameras=3, num_frames=500)

# 2. Create optimizers and scheduler
ppisp_optimizers = ppisp.create_optimizers()
ppisp_schedulers = ppisp.create_schedulers(ppisp_optimizers, max_optimization_iters)

# 3. Training loop
for step in range(max_optimization_iters):
    # ...

    # Render raw RGB from your radiance field
    rgb_raw = renderer(camera_idx, frame_idx)  # [..., 3]

    # Apply PPISP post-processing
    rgb_out = ppisp(
        rgb_raw,
        pixel_coords,           # [..., 2]
        resolution=(W, H),
        camera_idx=camera_idx,
        frame_idx=frame_idx,
    )

    # Add PPISP regularization loss to other losses
    loss = reconstruction_loss(rgb_out, rgb_gt) + ppisp.get_regularization_loss()
    loss.backward()

    # Step optimizers and scheduler
    for opt in ppisp_optimizers:
        opt.step()
        opt.zero_grad(set_to_none=True)
    for sched in ppisp_schedulers:
        sched.step()

# 4. Novel view rendering: pass camera_idx as usual, frame_idx=-1
rgb_out = ppisp(rgb_raw, pixel_coords, resolution=(W, H), camera_idx=camera_idx, frame_idx=-1)
```

Use `PPISPConfig` to customize regularization weights, learning rates, and controller activation timing.
PPISP regularization loss uses the fused CUDA implementation.

### Controller Distillation Mode

Controller distillation is enabled by default. When the controller activates, the scene representation (e.g., Gaussians, NeRF) should be frozen. This allows the controller to learn from fixed PPISP parameters without interference from scene updates:

```python
from ppisp import PPISP

ppisp = PPISP(num_cameras=3, num_frames=500)

# Training loop: freeze scene parameters when controller activates (default: 80% of training)
controller_activation_step = int(0.8 * max_optimization_iters)

for step in range(max_optimization_iters):
    if step >= controller_activation_step:
        freeze_scene_parameters()  # Your function to freeze Gaussians/NeRF params
    # ... rest of training loop ...
```

To disable distillation, set `controller_distillation=False` in the config.

When distillation is enabled and the controller activates:
- PPISP automatically freezes its internal parameters (exposure, vignetting, color, CRF)
- PPISP detaches the input RGB to prevent gradients flowing back to the scene
- Only the controller network receives gradients

### Exporting PDF Reports (Optional)

After training, you can export a PDF summary of the learned PPISP parameters:

```python
from pathlib import Path
from ppisp.report import export_ppisp_report

# Export PDF reports (one per camera) and a JSON with all parameters
pdf_paths = export_ppisp_report(
    ppisp,
    frames_per_camera=[100, 100, 100],  # frames per camera
    output_dir=Path("./ppisp_reports"),
    camera_names=["cam_front", "cam_left", "cam_right"],  # optional
)
```

The report includes visualizations for exposure, vignetting, color correction, and CRF per camera.

## BibTeX

```bibtex
@misc{deutsch2026ppispphysicallyplausiblecompensationcontrol,
      title={PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction}, 
      author={Isaac Deutsch and Nicolas Moënne-Loccoz and Gavriel State and Zan Gojcic},
      year={2026},
      eprint={2601.18336},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.18336},
}
```
