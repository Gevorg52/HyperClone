"""Pluggable 2D-inpainter backends for the Type-3 (unseen-region) fill.

Each backend = (isolated env python, model id, sampler defaults). They all drive the
same inpaint_runner.py (which auto-detects the diffusers inpaint pipeline class), so
adding FLUX.1-Fill or Qwen-Image-Inpaint later is just a new spec — no new code.

Default: Z-Image-Turbo (Apache-2.0, ~6B, ungated, fits the 4090). FLUX.1-Fill [dev]
is the documented quality upgrade (gated weights + FP8 on the 4090); SDXL-Inpaint is
the lightweight baseline. Chosen via the research pass (see memory; arXiv 2511.22699
Z-Image, bfl.ai FLUX.1-Fill).
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_RUNNER = str(Path(__file__).resolve().parent / "inpaint_runner.py")
_ENVS = r"C:\Users\gevor\anaconda3\envs"


@dataclass
class InpaintBackendSpec:
    name: str
    env_python: str           # python.exe of the env with a modern diffusers
    model_id: str             # HF model id (or local path)
    steps: int = 8
    guidance: float = 1.0
    note: str = ""


INPAINT_BACKENDS: dict[str, InpaintBackendSpec] = {
    # Default: ungated, Apache-2.0, fits 24GB with no quantization
    "zimage": InpaintBackendSpec(
        "zimage", os.path.join(_ENVS, "zimage", "python.exe"),
        "Tongyi-MAI/Z-Image-Turbo", steps=8, guidance=1.0,
        note="Apache-2.0, 6B, ungated — default; base-latent-blend inpaint"),
    # Quality upgrade (opt-in): gated weights, needs FP8 on the 4090
    "flux-fill": InpaintBackendSpec(
        "flux-fill", os.path.join(_ENVS, "flux", "python.exe"),
        "black-forest-labs/FLUX.1-Fill-dev", steps=28, guidance=30.0,
        note="dedicated trained Fill model, best quality; gated + needs FP8 (not set up)"),
    # Lightweight baseline (InFusion's default), fits 24GB FP16
    "sdxl": InpaintBackendSpec(
        "sdxl", os.path.join(_ENVS, "zimage", "python.exe"),
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", steps=30, guidance=8.0,
        note="OpenRAIL++, fits 24GB; baseline quality"),
}

DEFAULT_BACKEND = "zimage"


def get_spec(name: str) -> InpaintBackendSpec:
    if name not in INPAINT_BACKENDS:
        raise KeyError(f"Unknown inpaint backend '{name}'. "
                       f"Available: {', '.join(INPAINT_BACKENDS)}")
    return INPAINT_BACKENDS[name]


def validate_backend(name: str) -> tuple[bool, str]:
    spec = get_spec(name)
    if not os.path.exists(spec.env_python):
        return False, (f"env python not found for '{name}': {spec.env_python} "
                       f"(create the env / install diffusers).")
    if not os.path.exists(_RUNNER):
        return False, f"inpaint_runner.py missing: {_RUNNER}"
    return True, f"Inpaint backend '{name}' ready ({spec.model_id}). {spec.note}"


def inpaint(name: str, image_path: str, mask_path: str, out_path: str,
            prompt: Optional[str] = None, seed: int = 0) -> str:
    """Inpaint the WHITE region of mask_path in image_path -> out_path. Returns out_path."""
    spec = get_spec(name)
    cmd = [spec.env_python, _RUNNER, "--image", image_path, "--mask", mask_path,
           "--out", out_path, "--model", spec.model_id, "--steps", str(spec.steps),
           "--guidance", str(spec.guidance), "--seed", str(seed)]
    if prompt:
        cmd += ["--prompt", prompt]
    subprocess.run(cmd, check=True)
    return out_path
