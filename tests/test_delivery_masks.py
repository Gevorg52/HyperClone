"""Tests for core.delivery.ensure_masks mask staging (UI-3 closeout finding #1).

Guards the AUTHORITATIVE --mask_dir path: completeness must be judged by the masks
actually copied FROM mask_dir, so a pre-existing STALE dst file can never silently
fill a gap the mask_dir does not cover (the reviewer's reproduced HIGH finding). Also
pins the None-mask_dir path as unchanged. No GPU / no BiRefNet — every case uses
gen_masks=False, so the model is never loaded.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.delivery import ensure_masks


def _png(path: Path, val: int = 255, h: int = 8, w: int = 8):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w), val, np.uint8))


def _setup(tmp_path: Path, image_names):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # image_paths are never read while gen_masks=False, but keep them stem-aligned.
    image_paths = [str(tmp_path / "imgs" / n) for n in image_names]
    return str(data_dir), image_paths


class TestExplicitMaskDirAuthoritative:
    """An explicit --mask_dir judges completeness by copies FROM mask_dir."""

    def test_partial_mask_dir_with_stale_dst_raises(self, tmp_path):
        """REVIEWER REPRO: mask_dir has 1/2 stems and a STALE dst file exists for the
        other stem. Pre-fix this 'succeeded' with mixed fresh+stale content; post-fix
        it MUST raise, naming the missing stem."""
        image_names = ["A.jpg", "B.jpg"]
        data_dir, image_paths = _setup(tmp_path, image_names)
        factor = 2
        mask_dir = tmp_path / "reviewed"
        _png(mask_dir / "A.png", val=255)                    # mask_dir covers only A
        _png(Path(data_dir) / f"masks_{factor}" / "B.png", val=17)  # STALE dst for B
        with pytest.raises(RuntimeError) as e:
            ensure_masks(data_dir, factor, str(mask_dir), False,
                         image_paths, image_names, "cpu")
        msg = str(e.value)
        assert "partial" in msg.lower()
        assert "B" in msg                                    # names the missing stem

    def test_complete_mask_dir_overwrites_stale(self, tmp_path):
        """A COMPLETE mask_dir returns dst and its content is the mask_dir's (fresh),
        overwriting any stale pre-existing dst byte."""
        image_names = ["A.jpg", "B.jpg"]
        data_dir, image_paths = _setup(tmp_path, image_names)
        factor = 2
        mask_dir = tmp_path / "reviewed"
        _png(mask_dir / "A.png", val=201)
        _png(mask_dir / "B.png", val=202)
        _png(Path(data_dir) / f"masks_{factor}" / "B.png", val=17)  # stale B
        dst = ensure_masks(data_dir, factor, str(mask_dir), False,
                           image_paths, image_names, "cpu")
        b = cv2.imread(os.path.join(dst, "B.png"), cv2.IMREAD_GRAYSCALE)
        assert int(round(b.mean())) == 202                   # from mask_dir, not 17

    def test_partial_mask_dir_no_stale_still_raises(self, tmp_path):
        """Partial mask_dir with NO stale dst also raises (count, not presence)."""
        image_names = ["A.jpg", "B.jpg", "C.jpg"]
        data_dir, image_paths = _setup(tmp_path, image_names)
        mask_dir = tmp_path / "reviewed"
        _png(mask_dir / "A.png")
        with pytest.raises(RuntimeError) as e:
            ensure_masks(data_dir, 2, str(mask_dir), False,
                         image_paths, image_names, "cpu")
        assert "2 missing" in str(e.value)                   # B and C


class TestNoneMaskDirUnchanged:
    """The None-mask_dir path is unchanged by the fix."""

    def test_complete_dst_returns(self, tmp_path):
        """No mask_dir + a COMPLETE masks_{factor} returns it, no error."""
        image_names = ["A.jpg", "B.jpg"]
        data_dir, image_paths = _setup(tmp_path, image_names)
        factor = 2
        _png(Path(data_dir) / f"masks_{factor}" / "A.png")
        _png(Path(data_dir) / f"masks_{factor}" / "B.png")
        dst = ensure_masks(data_dir, factor, None, False,
                           image_paths, image_names, "cpu")
        assert os.path.isdir(dst)
        assert os.path.isfile(os.path.join(dst, "A.png"))
        assert os.path.isfile(os.path.join(dst, "B.png"))

    def test_incomplete_dst_no_gen_raises(self, tmp_path):
        """No mask_dir + incomplete dst + no --gen_masks raises the no-masks error."""
        image_names = ["A.jpg", "B.jpg"]
        data_dir, image_paths = _setup(tmp_path, image_names)
        _png(Path(data_dir) / "masks_2" / "A.png")           # only 1 of 2
        with pytest.raises(RuntimeError) as e:
            ensure_masks(data_dir, 2, None, False,
                         image_paths, image_names, "cpu")
        assert "no masks available" in str(e.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
