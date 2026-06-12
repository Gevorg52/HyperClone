"""Training backend registry and base class for 3DGS training.

Provides a pluggable backend system so different training engines
(gsplat, 2DGS, future tools) can be swapped in without changing the GUI.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import gsplat_jit_shim  # noqa: F401  (must precede gsplat import)


# ---------------------------------------------------------------------------
# Training config (backend-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Common training hyperparameters passed to any backend."""

    backend: str = "gsplat"
    max_steps: int = 30_000
    data_factor: int = 4           # Image downsampling: 1 / 2 / 4 / 8
    test_every: int = 8            # Hold out every Nth image for eval
    save_steps: list[int] = field(default_factory=lambda: [7_000, 30_000])
    extra: dict = field(default_factory=dict)  # Backend-specific overrides


# Type alias for progress callbacks:  (step, max_steps, loss, psnr) -> None
ProgressCallback = Callable[[int, int, float, float], None]


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Result returned after training completes (or is stopped early)."""

    checkpoint_dir: Optional[str] = None   # Directory with saved checkpoint
    ply_path: Optional[str] = None         # Exported point cloud (.ply)
    psnr: float = 0.0
    ssim: float = 0.0
    steps_completed: int = 0
    final_loss: float = 0.0


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TrainingBackend(abc.ABC):
    """Abstract base class for 3DGS training backends."""

    name: str = ""  # Display name ("gsplat", "2dgs", etc.)

    @abc.abstractmethod
    def validate_environment(self) -> tuple[bool, str]:
        """Check if this backend is installed and ready.

        Returns (ok, message).  If ok is False, message explains what's missing.
        """

    @abc.abstractmethod
    def get_default_config(self) -> dict:
        """Return default training hyperparameters as a flat dict."""

    @abc.abstractmethod
    def prepare_data(
        self,
        export_dir: str,
        mask_dir: Optional[str],
        output_dir: str,
        config: TrainingConfig,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> str:
        """Prepare data directory in the backend's expected format.

        Returns the path to the prepared data directory.
        """

    @abc.abstractmethod
    def train(
        self,
        data_dir: str,
        output_dir: str,
        config: TrainingConfig,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        """Run training.  Must respect stop signals from the progress callback.

        The progress callback may raise ``TrainingInterrupted`` to request
        early stop (the backend should catch it and save the latest checkpoint).
        """

    @abc.abstractmethod
    def render_views(
        self,
        checkpoint_dir: str,
        data_dir: str,
        output_dir: str,
        max_images: int = 0,
    ) -> list[str]:
        """Render views from a trained checkpoint.

        If *max_images* is 0, render all test-set views.
        Returns list of rendered image paths.
        """


# ---------------------------------------------------------------------------
# Interruption signal
# ---------------------------------------------------------------------------

class TrainingInterrupted(Exception):
    """Raised inside a progress callback to request early stop."""


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

TRAINING_BACKENDS: dict[str, TrainingBackend] = {}


def register_backend(backend: TrainingBackend) -> None:
    """Register a training backend instance by its name."""
    TRAINING_BACKENDS[backend.name] = backend


def get_backend(name: str) -> TrainingBackend:
    """Look up a registered training backend by name."""
    if name not in TRAINING_BACKENDS:
        available = ", ".join(TRAINING_BACKENDS) or "(none)"
        raise KeyError(
            f"Unknown training backend '{name}'. Available: {available}"
        )
    return TRAINING_BACKENDS[name]


def available_backends() -> list[str]:
    """Return names of all registered backends."""
    return list(TRAINING_BACKENDS.keys())


# ---------------------------------------------------------------------------
# Auto-register bundled backends (import triggers registration)
# ---------------------------------------------------------------------------

def _auto_register():
    """Import bundled backends so they self-register."""
    try:
        from core.training.gsplat_backend import GsplatBackend  # noqa: F401
        register_backend(GsplatBackend())
    except Exception:
        pass  # gsplat not installed — backend simply unavailable


_auto_register()
