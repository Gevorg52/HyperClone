"""NVIDIA Fixer enhancement backend.

Uses Docker to run the Cosmos-based Fixer model for artifact removal
on rendered 3DGS views.  Requires Docker Desktop with GPU passthrough
(WSL2 on Windows) and the fixer-cosmos-env Docker image built from
https://github.com/nv-tlabs/Fixer.

Fixer is a single-step diffusion model (0.6B params) that removes
typical 3DGS artifacts (floaters, blurriness, view-inconsistencies).
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCKER_IMAGE = "fixer-cosmos-env"
DEFAULT_TIMESTEP = 250
MODEL_PATH_IN_CONTAINER = "/work/models/pretrained/pretrained_fixer.pkl"
INFERENCE_SCRIPT = "/work/src/inference_pretrained_model.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _docker_available() -> tuple[bool, str]:
    """Check if Docker is installed and the daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return True, "Docker is available"
        return False, f"Docker daemon not running: {result.stderr.strip()}"
    except FileNotFoundError:
        return False, "Docker is not installed"
    except subprocess.TimeoutExpired:
        return False, "Docker timed out — is the daemon running?"


def _docker_image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _gpu_runtime_available() -> bool:
    """Check if Docker GPU runtime (--gpus) is available."""
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus=all",
             "nvidia/cuda:12.1.0-base-ubuntu22.04", "nvidia-smi"],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# FixerBackend
# ---------------------------------------------------------------------------

class FixerBackend(EnhancementBackend):
    """Post-training image enhancement using NVIDIA Fixer (Docker)."""

    name = "fixer"

    def validate_environment(self) -> tuple[bool, str]:
        """Check Docker + GPU runtime + Fixer image are available."""
        ok, msg = _docker_available()
        if not ok:
            return False, (
                f"Docker is required for Fixer: {msg}\n\n"
                "Install Docker Desktop and enable WSL2 backend with GPU support:\n"
                "https://docs.docker.com/desktop/wsl/"
            )

        if not _docker_image_exists(DOCKER_IMAGE):
            return False, (
                f"Fixer Docker image '{DOCKER_IMAGE}' not found.\n\n"
                "Build it from the Fixer repo:\n"
                "  git clone https://github.com/nv-tlabs/Fixer.git\n"
                "  cd Fixer\n"
                "  docker build -t fixer-cosmos-env -f Dockerfile.cosmos ."
            )

        return True, f"Fixer ready (Docker image: {DOCKER_IMAGE})"

    def get_default_config(self) -> dict:
        return {
            "timestep": DEFAULT_TIMESTEP,
            "docker_image": DOCKER_IMAGE,
        }

    def enhance_images(
        self,
        image_paths: list[str],
        output_dir: str,
        config: EnhancementConfig,
        progress_cb: Optional[EnhancementProgressCallback] = None,
    ) -> list[str]:
        """Enhance rendered images via Fixer Docker container.

        Copies images to a staging directory, mounts it into Docker,
        runs inference, and collects enhanced outputs.
        """
        if not image_paths:
            return []

        out = Path(output_dir)
        staging = out / "staging_input"
        enhanced_dir = out / "enhanced"

        # Clean and create staging directories
        if staging.is_dir():
            shutil.rmtree(str(staging))
        staging.mkdir(parents=True, exist_ok=True)
        enhanced_dir.mkdir(parents=True, exist_ok=True)

        # Copy input images to staging
        if progress_cb:
            progress_cb(0, len(image_paths), "Copying images to staging...")

        for i, src in enumerate(image_paths):
            dst = staging / Path(src).name
            shutil.copy2(src, str(dst))

        if progress_cb:
            progress_cb(0, len(image_paths), "Running Fixer enhancement...")

        # Build Docker command
        timestep = config.extra.get("timestep", DEFAULT_TIMESTEP)
        docker_image = config.extra.get("docker_image", DOCKER_IMAGE)

        # Convert Windows paths to Docker-compatible mount paths
        staging_mount = str(staging).replace("\\", "/")
        enhanced_mount = str(enhanced_dir).replace("\\", "/")

        cmd = [
            "docker", "run", "--rm", "--gpus=all",
            "--ipc=host",
            "-v", f"{staging_mount}:/input",
            "-v", f"{enhanced_mount}:/output",
            docker_image,
            "python", INFERENCE_SCRIPT,
            "--model", MODEL_PATH_IN_CONTAINER,
            "--input", "/input",
            "--output", "/output",
            "--timestep", str(timestep),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=3600,  # 1 hour max
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Fixer Docker container failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        # Collect enhanced images
        enhanced_images = sorted(
            str(p) for p in enhanced_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if progress_cb:
            progress_cb(
                len(enhanced_images), len(image_paths),
                f"Enhanced {len(enhanced_images)} images",
            )

        # Clean up staging
        shutil.rmtree(str(staging), ignore_errors=True)

        return enhanced_images
