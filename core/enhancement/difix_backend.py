"""NVIDIA Difix3D+ enhancement backend — NATIVE Windows (no Docker / no WSL).

Difix3D+ (arXiv 2503.01774, weights hf.co/nvidia/difix) is a single-step
SD-Turbo diffusion model that removes 3DGS artifacts (floaters, ghosting, blur)
from rendered novel views. Unlike the Cosmos "NuRec Fixer" (Docker-only), Difix
runs natively via `diffusers` — pure pip, no flash-attn/triton/bitsandbytes.

This backend implements the standard EnhancementBackend interface, so it drops
straight into the Difix3D+-style distill-back loop in `core/enhancement/distill.py`
(render pseudo-views → enhance → append → resume), giving a WSL-free diffusion
artifact-correction path.

SCAFFOLD STATUS: structurally complete and registered. Running it requires the
Difix deps + weights, which conflict with newer diffusers used elsewhere — install
in an ISOLATED env (per the README): diffusers==0.25.1, transformers==4.38.0,
peft==0.9.0, accelerate, plus torch/torchvision. Weights auto-download on first
use (~SD-Turbo size; research/non-commercial license — fine for R&D). License is
NOT a blocker for this research project. Validate the exact pipeline call against
the current nv-tlabs/Difix3D repo before relying on it in production.

Use it for clips that genuinely have floater/ghosting/extrapolation artifacts —
NOT for capture-limited textureless surfaces (it would hallucinate, not recover).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

from core.enhancement import (
    EnhancementBackend,
    EnhancementConfig,
    EnhancementProgressCallback,
)

DEFAULT_MODEL = "nvidia/difix"          # or "nvidia/difix_ref" (reference-guided)
DEFAULT_TIMESTEP = 199                   # SD-Turbo single-step timestep
_pipe = None


def _load_pipeline(model: str, device: str = "cuda"):
    """Lazily build the Difix diffusers pipeline (cached at module level)."""
    global _pipe
    if _pipe is None:
        import torch
        # The repo ships a custom DifixPipeline; fall back to a generic
        # DiffusionPipeline with trust_remote_code if the package isn't present.
        try:
            from difix3d import DifixPipeline  # type: ignore
            _pipe = DifixPipeline.from_pretrained(model, trust_remote_code=True)
        except Exception:
            from diffusers import DiffusionPipeline
            _pipe = DiffusionPipeline.from_pretrained(
                model, trust_remote_code=True, torch_dtype=torch.float16,
            )
        _pipe = _pipe.to(device)
    return _pipe


class DifixBackend(EnhancementBackend):
    """Native (diffusers) NVIDIA Difix3D+ artifact-removal backend."""

    name = "difix"

    def validate_environment(self) -> tuple[bool, str]:
        try:
            import diffusers  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False, (
                "Difix needs `diffusers` (native, no Docker). Recommended in an "
                "ISOLATED env to avoid version clashes:\n"
                "  pip install diffusers==0.25.1 transformers==4.38.0 peft==0.9.0 "
                "accelerate\n"
                "Weights (nvidia/difix) auto-download on first use."
            )
        return True, (
            "Difix (native diffusers) ready. Weights 'nvidia/difix' download on "
            "first use. NOTE: pinned diffusers/transformers recommended in an "
            "isolated env; validate the pipeline call vs the current Difix3D repo."
        )

    def get_default_config(self) -> dict:
        return {
            "model": DEFAULT_MODEL,
            "timestep": DEFAULT_TIMESTEP,
            "num_inference_steps": 1,
            "prompt": "remove degradation",  # Difix's training prompt
        }

    def enhance_images(
        self,
        image_paths: list[str],
        output_dir: str,
        config: EnhancementConfig,
        progress_cb: Optional[EnhancementProgressCallback] = None,
    ) -> list[str]:
        if not image_paths:
            return []
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cfg = {**self.get_default_config(), **(config.extra or {})}

        pipe = _load_pipeline(cfg["model"])
        enhanced: list[str] = []
        n = len(image_paths)
        for i, src in enumerate(image_paths):
            if progress_cb and i % 5 == 0:
                progress_cb(i, n, f"Difix enhancing {i + 1}/{n}")
            img = Image.open(src).convert("RGB")
            try:
                result = pipe(
                    cfg["prompt"], image=img,
                    num_inference_steps=cfg["num_inference_steps"],
                    timesteps=[cfg["timestep"]],
                )
                fixed = result.images[0]
            except Exception as e:  # API mismatch → fail loudly with guidance
                raise RuntimeError(
                    f"Difix pipeline call failed ({e}). Verify the inference API "
                    "against the current nv-tlabs/Difix3D repo and update "
                    "core/enhancement/difix_backend._load_pipeline / enhance_images."
                )
            dst = out / (Path(src).stem + ".png")
            fixed.save(str(dst))
            enhanced.append(str(dst))

        if progress_cb:
            progress_cb(n, n, f"Difix enhanced {len(enhanced)} images")
        return enhanced
