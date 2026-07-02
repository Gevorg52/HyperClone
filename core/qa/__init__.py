"""Apex Reality QA harness (WP-Q1).

Visibility-split quality measurement for trained splats. Phase 1 ships the core:
  * visibility.py       — observed/unobserved mask builder (OGGSplat protocol)
  * masked_metrics.py   — masked PSNR / SSIM / LPIPS on the OBSERVED region
  * scorecard.py        — per-scene scorecard dataclass (JSON + HTML) + a scorer
                          registry/interface so phase-2 scorers plug in unchanged.

Later phases (CrossScore, NOVA, TSED, RAFT warp-error, NVS-SQA, flicker) register
themselves as `Scorer` objects; the report driver calls each with a `SceneContext`
and stores the result under `Scorecard.scorers[name]`.  Everything is default-OFF
and deterministic (fixed seeds + sorted iteration) — running twice reproduces the
JSON exactly, which is a WP gate requirement.
"""

from .visibility import render_alpha, build_visibility_mask, VisibilityConfig
from .masked_metrics import masked_psnr, masked_ssim, masked_lpips
from .scorecard import (
    Scorecard,
    ViewScore,
    SceneContext,
    Scorer,
    SCORER_REGISTRY,
    write_html,
)

__all__ = [
    "render_alpha",
    "build_visibility_mask",
    "VisibilityConfig",
    "masked_psnr",
    "masked_ssim",
    "masked_lpips",
    "Scorecard",
    "ViewScore",
    "SceneContext",
    "Scorer",
    "SCORER_REGISTRY",
    "write_html",
]
