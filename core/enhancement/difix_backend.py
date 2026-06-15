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

EMPIRICAL (laptop A/B, June 2026): ungated Difix improves full-image LPIPS
(0.366->0.350 over 8 views; biggest gains on under-observed orbit views) by
cleaning soft / under-observed regions, BUT on the well-resolved SUBJECT it over-
sharpens / HALLUCINATES (deck LPIPS 0.256->0.279, keyboard 0.194->0.234 — sharper
but less faithful). FIX: a per-pixel SHARPNESS GATE (`gated`, on by default,
see `_gate_blend`) keeps the render where it is already locally sharp and only
lets Difix act where the render is soft. Gated result beats BOTH render and
ungated Difix: full-image LPIPS 0.343 (vs 0.366 / 0.350); deck 0.254 & keyboard
0.199 back to ~render fidelity; background gain (0.350->0.318) retained. So Difix
is now safe to run by default; set `gated=False` for the raw (ungated) output.
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


def _gate_blend(render_bgr, difix_bgr, wmax: float = 0.8):
    """Per-pixel blend: keep the render where it is already locally SHARP,
    let Difix take over where the render is SOFT. final = (1-w)*render + w*difix.

    `w` is high (->wmax) in low-sharpness regions, ~0 in high-sharpness regions,
    with the pivot at the image's MEDIAN local sharpness (resolution-robust — no
    absolute threshold). This protects the well-resolved subject from Difix's
    over-sharpening / hallucination while keeping its gains on soft / under-
    observed regions. Validated on the laptop clip (8 views): full-image LPIPS
    render 0.366 / ungated difix 0.350 / GATED 0.343; deck & keyboard returned to
    ~render fidelity, background gain (0.350->0.318) kept. FreeFix-style 2D gate.
    """
    import cv2
    import numpy as np

    if difix_bgr.shape != render_bgr.shape:
        difix_bgr = cv2.resize(difix_bgr, (render_bgr.shape[1], render_bgr.shape[0]),
                               interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    sharp = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 7)  # local sharpness
    med = float(np.median(sharp)) + 1e-6
    z = np.clip((med - sharp) / (0.5 * med), -30, 30)               # clip => no exp overflow
    w = np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, wmax)[..., None]     # ~wmax soft, ~0 sharp
    out = render_bgr.astype(np.float32) * (1 - w) + difix_bgr.astype(np.float32) * w
    return out.astype(np.uint8)


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
                "prompt": "remove degradation",
                "gated": True,        # blend toward render in already-sharp regions
                "gate_wmax": 0.8}

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
        raw = out / "difix_raw"          # ungated Difix output (runner writes here)
        enhanced = out / "enhanced"      # final (gated) output
        for d in (staging, raw):
            if d.is_dir():
                shutil.rmtree(d)
        staging.mkdir(parents=True, exist_ok=True)
        raw.mkdir(parents=True, exist_ok=True)
        enhanced.mkdir(parents=True, exist_ok=True)
        # map source -> staged name; remember the runner's output name (stem + .png)
        staged = {}
        for src in image_paths:
            name = Path(src).name
            shutil.copy2(src, staging / name)
            staged[name] = (src, Path(name).stem + ".png")

        if progress_cb:
            progress_cb(0, len(image_paths), "Running Difix (isolated env)...")
        cmd = [
            conda, "run", "-n", cfg["env"], "python", _RUNNER,
            "--input", str(staging), "--output", str(raw),
            "--repo", cfg["repo"], "--model", cfg["model"],
            "--prompt", cfg["prompt"], "--timestep", str(cfg["timestep"]),
        ]
        # expandable_segments cuts allocator fragmentation on the 1080p VAE
        # decode (keeps Difix well under the 24 GiB ceiling on a 4090).
        sub_env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=7200, env=sub_env)
        if "DIFIX_RUNNER_DONE" not in result.stdout:
            raise RuntimeError(
                f"Difix runner failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
            )

        gated = bool(cfg.get("gated", True))
        wmax = float(cfg.get("gate_wmax", 0.8))
        outs = []
        if gated:
            import cv2
            if progress_cb:
                progress_cb(0, len(staged), "Gating Difix (protect sharp regions)...")
            for i, (name, (src, raw_name)) in enumerate(staged.items()):
                raw_path = raw / raw_name
                if not raw_path.exists():
                    continue
                render_img = cv2.imread(src)
                difix_img = cv2.imread(str(raw_path))
                blended = _gate_blend(render_img, difix_img, wmax=wmax)
                dst = enhanced / (Path(name).stem + ".png")
                cv2.imwrite(str(dst), blended)
                outs.append(str(dst))
                if progress_cb:
                    progress_cb(i + 1, len(staged), f"Gated {i + 1}/{len(staged)}")
        else:
            for p in raw.iterdir():
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    dst = enhanced / p.name
                    shutil.copy2(p, dst)
                    outs.append(str(dst))

        outs = sorted(outs)
        shutil.rmtree(staging, ignore_errors=True)
        if progress_cb:
            progress_cb(len(outs), len(image_paths),
                        f"Difix enhanced {len(outs)} images" + (" (gated)" if gated else ""))
        return outs
