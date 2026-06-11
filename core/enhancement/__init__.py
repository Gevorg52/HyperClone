"""Enhancement backend registry and base class for post-training enhancement.

Provides a pluggable system for image-level enhancement of rendered 3DGS views.
Each enhancer (NVIDIA Fixer, future DiFix3D+, etc.) implements a common interface
that takes rendered images and produces cleaned/enhanced versions.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Enhancement config (backend-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class EnhancementConfig:
    """Common enhancement parameters."""

    backend: str = "fixer"
    num_views: int = 100              # How many novel views to render + enhance
    extra: dict = field(default_factory=dict)  # Backend-specific overrides


# Progress callback: (current, total, description) -> None
EnhancementProgressCallback = Callable[[int, int, str], None]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class EnhancementBackend(abc.ABC):
    """Abstract base class for post-training image enhancers."""

    name: str = ""  # Display name ("fixer", "difix3d+", etc.)

    @abc.abstractmethod
    def validate_environment(self) -> tuple[bool, str]:
        """Check if this enhancer is available.

        Returns (ok, message).
        """

    @abc.abstractmethod
    def get_default_config(self) -> dict:
        """Return default enhancement parameters as a flat dict."""

    @abc.abstractmethod
    def enhance_images(
        self,
        image_paths: list[str],
        output_dir: str,
        config: EnhancementConfig,
        progress_cb: Optional[EnhancementProgressCallback] = None,
    ) -> list[str]:
        """Enhance a batch of rendered images.

        Returns paths to the enhanced images in *output_dir*.
        """


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

ENHANCEMENT_BACKENDS: dict[str, EnhancementBackend] = {}


def register_backend(backend: EnhancementBackend) -> None:
    """Register an enhancement backend instance by its name."""
    ENHANCEMENT_BACKENDS[backend.name] = backend


def get_backend(name: str) -> EnhancementBackend:
    """Look up a registered enhancement backend by name."""
    if name not in ENHANCEMENT_BACKENDS:
        available = ", ".join(ENHANCEMENT_BACKENDS) or "(none)"
        raise KeyError(
            f"Unknown enhancement backend '{name}'. Available: {available}"
        )
    return ENHANCEMENT_BACKENDS[name]


def available_backends() -> list[str]:
    """Return names of all registered backends."""
    return list(ENHANCEMENT_BACKENDS.keys())


# ---------------------------------------------------------------------------
# Auto-register bundled backends
# ---------------------------------------------------------------------------

def _auto_register():
    """Import bundled backends so they self-register."""
    try:
        from core.enhancement.fixer_backend import FixerBackend  # noqa: F401
        register_backend(FixerBackend())
    except Exception:
        pass


_auto_register()
