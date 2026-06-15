"""NVIDIA Difix3D+ enhancement backend — NATIVE Windows (no Docker / no WSL).

Difix3D+ (arXiv 2503.01774, weights hf.co/nvidia/difix) is a single-step
SD-Turbo diffusion model that removes 3DGS artifacts (floaters, ghosting, blur)
from rendered novel views. Unlike the Cosmos "NuRec Fixer" (Docker-only), Difix
runs natively — but its pinned deps (diffusers==0.25.1, transformers==4.38.0,
numpy<2) clash with the main env, so we run it as a SUBPROCESS in an isolated
conda env via `conda run -n difix` (no Docker, no WSL).

Drops into the distill-back loop (core/enhancement/distill.py).

SETUP (one-time, native):
    git clone https://github.com/nv-tlabs/Difix3D   (-> DIFIX_REPO)
    conda create -n difix python=3.10 -y
    conda run -n difix pip install torch==2.3.0 torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cu121
    conda run -n difix pip install diffusers==0.25.1 transformers==4.38.0 \
        peft==0.9.0 huggingface-hub==0.25.1 accelerate "imageio[ffmpeg]" \
        einops lpips torchmetrics pillow "numpy<2"
    # weights (nvidia/difix) auto-download on first use. Research/non-commercial.

EMPIRICAL (laptop A/B, June 2026): full-image LPIPS improved (0.366->0.350 over 8
views; biggest gains on under-observed orbit views) — Difix genuinely cleans soft
/ under-observed regions. BUT on the well-resolved SUBJECT it over-sharpens /
HALLUCINATES (deck LPIPS 0.256->0.279, keyboard 0.194->0.234 — sharper but less
faithful). Use it for clips with real floater/ghosting/under-observed artifacts;
do NOT apply globally to a well-reconstructed subject if fidelity matters.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from core.enhancement import (
    EnhancementBackend,
    EnhancementConfig,
    EnhancementProgressCallback,
)

DIFIX_ENV = "difix"
DIFIX_REPO = r"F:\HYPERclone\Difix3D"
_RUNNER = str(Path(__file__).resolve().parent / "difix_runner.py")


def _conda() -> Optional[str]:
    return shutil.which("conda") or (
        r"C:\Users\gevor\anaconda3\Scripts\conda.exe"
        if os.path.exists(r"C:\Users\gevor\anaconda3\Scripts\conda.exe") else None
    )


class DifixBackend(EnhancementBackend):
    """Native (isolated-conda-env subprocess) NVIDIA Difix3D+ artifact cleaner."""

    name = "difix"

    def validate_environment(self) -> tuple[bool, str]:
        conda = _conda()
        if conda is None:
            return False, "conda not found on PATH (needed to run the isolated difix env)."
        if not Path(DIFIX_REPO).is_dir():
            return False, (
                f"Difix3D repo not found at {DIFIX_REPO}. Clone it:\n"
                "  git clone https://github.com/nv-tlabs/Difix3D"
            )
        try:
            envs = subprocess.run([conda, "env", "list"], capture_output=True,
                                  text=True, timeout=30).stdout
        except Exception as e:
            return False, f"Could not list conda envs: {e}"
        if DIFIX_ENV not in envs:
            return False, (
                f"Isolated conda env '{DIFIX_ENV}' not found. See setup in "
                "core/enhancement/difix_backend.py docstring (one-time)."
            )
        return True, (
            f"Difix ready (native subprocess: conda env '{DIFIX_ENV}', repo "
            f"{DIFIX_REPO}). Weights nvidia/difix auto-download on first use."
        )

    def get_default_config(self) -> dict:
        return {"env": DIFIX_ENV, "repo": DIFIX_REPO,
                "model": "nvidia/difix", "timestep": 199,
                "prompt": "remove degradation"}

    def enhance_images(
        self,
        image_paths: list[str],
        output_dir: str,
        config: EnhancementConfig,
        progress_cb: Optional[EnhancementProgressCallback] = None,
    ) -> list[str]:
        if not image_paths:
            return []
        cfg = {**self.get_default_config(), **(config.extra or {})}
        conda = _conda()
        if conda is None:
            raise RuntimeError("conda not found; cannot run the difix env.")

        out = Path(output_dir)
        staging = out / "difix_staging"
        enhanced = out / "enhanced"
        if staging.is_dir():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        enhanced.mkdir(parents=True, exist_ok=True)
        for src in image_paths:
            shutil.copy2(src, staging / Path(src).name)

        if progress_cb:
            progress_cb(0, len(image_paths), "Running Difix (isolated env)...")
        cmd = [
            conda, "run", "-n", cfg["env"], "python", _RUNNER,
            "--input", str(staging), "--output", str(enhanced),
            "--repo", cfg["repo"], "--model", cfg["model"],
            "--prompt", cfg["prompt"], "--timestep", str(cfg["timestep"]),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if "DIFIX_RUNNER_DONE" not in result.stdout:
            raise RuntimeError(
                f"Difix runner failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
            )

        outs = sorted(str(p) for p in enhanced.iterdir()
                      if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        shutil.rmtree(staging, ignore_errors=True)
        if progress_cb:
            progress_cb(len(outs), len(image_paths), f"Difix enhanced {len(outs)} images")
        return outs
