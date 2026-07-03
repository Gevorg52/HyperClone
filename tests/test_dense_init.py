"""Tests for WP-F2 dense init: the RoMa matcher's certainty filter (hloc) and
the pure merge helpers in core.dense_init.

Pure logic only — no RoMa weights, no GPU, no COLMAP models. The matcher test
drives the REAL Roma._forward with a stubbed `net`, which is exactly where the
review-found bug lived (`if keep.any():` returned the whole low-certainty tail
whenever NO match cleared the floor; the filter must apply unconditionally and
an empty result is valid).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.dense_init import _median_nn_distance, _voxel_keys  # noqa: E402

# hloc is an editable install (../Hierarchical-Localization); the roma matcher
# module imports romatch only lazily inside _init, so this import is light.
roma_mod = pytest.importorskip("hloc.matchers.roma")


class _StubNet:
    """Stands in for the romatch model: returns canned matches/certainties."""

    def __init__(self, certs):
        self._certs = torch.as_tensor(certs, dtype=torch.float32)

    def match(self, im_A, im_B, device=None):
        return "warp", "certainty"  # opaque; only sample() consumes them

    def sample(self, warp, certainty, num):
        n = len(self._certs)
        return torch.zeros(n, 4), self._certs

    def to_pixel_coordinates(self, matches, H0, W0, H1, W1):
        n = matches.shape[0]
        return torch.zeros(n, 2), torch.zeros(n, 2)


def _run_forward(certs, min_certainty=0.05):
    """Call the real Roma._forward with a stub self (conf + net + _to_pil)."""
    fake = SimpleNamespace(
        conf={**roma_mod.Roma.default_conf, "min_certainty": min_certainty},
        net=_StubNet(certs),
        _to_pil=roma_mod.Roma._to_pil,
    )
    data = {
        "image0": torch.rand(1, 3, 32, 32),
        "image1": torch.rand(1, 3, 32, 32),
    }
    return roma_mod.Roma._forward(fake, data)


class TestRomaCertaintyFilter:
    def test_all_below_threshold_returns_empty(self):
        """The review-found edge case: NO match clears min_certainty -> the
        result must be EMPTY, not the whole low-certainty tail."""
        pred = _run_forward([0.01, 0.02, 0.049])
        assert pred["scores"].shape[0] == 0
        assert pred["keypoints0"].shape[0] == 0
        assert pred["keypoints1"].shape[0] == 0

    def test_mixed_keeps_only_above_threshold(self):
        pred = _run_forward([0.01, 0.05, 0.5, 0.04, 1.0])
        assert pred["scores"].shape[0] == 3  # 0.05 (>=), 0.5, 1.0
        assert bool((pred["scores"] >= 0.05).all())
        assert pred["keypoints0"].shape[0] == pred["scores"].shape[0]
        assert pred["keypoints1"].shape[0] == pred["scores"].shape[0]

    def test_all_above_threshold_keeps_everything(self):
        pred = _run_forward([0.1, 0.9, 0.6])
        assert pred["scores"].shape[0] == 3


class TestMergeHelpers:
    def test_median_nn_distance_grid(self):
        """Unit grid -> median nearest-neighbour distance is exactly 1."""
        xs, ys = np.meshgrid(np.arange(4.0), np.arange(4.0))
        pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(16)], axis=1)
        assert _median_nn_distance(pts) == pytest.approx(1.0)

    def test_voxel_keys_dedup_semantics(self):
        """Points inside the same voxel share a key; across boundary differ."""
        origin = np.zeros(3)
        pts = np.array([[0.1, 0.1, 0.1], [0.4, 0.4, 0.4], [1.1, 0.1, 0.1]])
        keys = _voxel_keys(pts, voxel=1.0, origin=origin)
        assert tuple(keys[0]) == tuple(keys[1])
        assert tuple(keys[0]) != tuple(keys[2])
