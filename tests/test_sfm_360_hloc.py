"""Tests for core.sfm_360_hloc — WP-O5 ALIKED+LightGlue feature backend.

Covers the parts that are ours and that a silent regression would corrupt: the mask filter
(HLoc has no mask support, so this is what stands in for COLMAP's MaskFeatures), the pair
bookkeeping, and the CLI contract that keeps `--features sift` the untouched default.
No network, no GPU, no COLMAP reconstruction.
"""

import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.sfm_360 import build_parser, main  # noqa: E402
from core.sfm_360_hloc import (  # noqa: E402
    aliked_conf,
    filter_features_by_mask,
    lightglue_conf,
    union_pairs,
    verify_mask_integrity,
    write_pairs,
)


def write_features(path: Path, name: str, kpts: np.ndarray, dim: int = 128):
    """One HLoc-shaped feature group: keypoints (N,2), descriptors (D,N), scores (N,)."""
    n = kpts.shape[0]
    with h5py.File(str(path), "a", libver="latest") as fd:
        grp = fd.create_group(name)
        grp.create_dataset("keypoints", data=kpts.astype(np.float32))
        grp.create_dataset("descriptors",
                           data=np.tile(np.arange(n, dtype=np.float32), (dim, 1)))
        grp.create_dataset("keypoint_scores",
                           data=np.arange(n, dtype=np.float32))
        grp.create_dataset("image_size", data=np.array([64, 64], np.int32))
        grp["keypoints"].attrs["uncertainty"] = 1.5


# --- configs ---------------------------------------------------------------

def test_aliked_conf_uses_production_field_names():
    conf = aliked_conf(8192, 1600)
    assert conf["model"] == {"name": "aliked", "max_num_keypoints": 8192}
    assert conf["preprocessing"]["resize_max"] == 1600
    assert conf["preprocessing"]["grayscale"] is False
    assert conf["output"] == "feats-aliked-n8192-r1600"


def test_aliked_conf_native_resolution_sentinel():
    for value in (0, -1, None):
        assert aliked_conf(4096, value)["preprocessing"]["resize_max"] == 99999


def test_lightglue_conf_is_aliked_flavoured():
    assert lightglue_conf()["model"] == {"name": "lightglue", "features": "aliked"}


# --- mask filter -----------------------------------------------------------

def test_filter_drops_exactly_the_masked_keypoints(tmp_path):
    masks = tmp_path / "masks" / "pano_camera0"
    masks.mkdir(parents=True)
    mask = np.zeros((64, 64), np.uint8)
    mask[:, 32:] = 255                       # right half VALID
    cv2.imwrite(str(masks / "pano_00000.jpg.png"), mask)
    kpts = np.array([[5.0, 10.0], [40.0, 10.0], [50.0, 60.0], [1.0, 1.0]])
    feats = tmp_path / "feats.h5"
    write_features(feats, "pano_camera0/pano_00000.jpg", kpts)

    stats = filter_features_by_mask(feats, tmp_path / "masks",
                                    ["pano_camera0/pano_00000.jpg"], progress=0)
    assert stats["kept"] == 2 and stats["dropped"] == 2
    assert stats["dropped_frac"] == pytest.approx(0.5)
    assert stats["empty_images"] == [] and stats["missing_masks"] == 0

    with h5py.File(str(feats), "r") as fd:
        grp = fd["pano_camera0/pano_00000.jpg"]
        kept = grp["keypoints"][...]
        # descriptors are (D, N) in hloc's ALIKED layout — the N axis must be filtered
        assert grp["descriptors"].shape == (128, 2)
        assert grp["keypoint_scores"].shape == (2,)
        assert np.allclose(kept, kpts[[1, 2]])
        # the surviving descriptors/scores still line up with their keypoints
        assert np.allclose(grp["keypoint_scores"][...], [1.0, 2.0])
        assert np.allclose(grp["descriptors"][0], [1.0, 2.0])
        assert grp["keypoints"].attrs["uncertainty"] == pytest.approx(1.5)
        assert list(grp["image_size"][...]) == [64, 64]


def test_filter_is_a_noop_when_everything_is_valid(tmp_path):
    masks = tmp_path / "masks" / "pano_camera0"
    masks.mkdir(parents=True)
    cv2.imwrite(str(masks / "pano_00000.jpg.png"), np.full((64, 64), 255, np.uint8))
    kpts = np.array([[5.0, 10.0], [40.0, 10.0]])
    feats = tmp_path / "feats.h5"
    write_features(feats, "pano_camera0/pano_00000.jpg", kpts)
    stats = filter_features_by_mask(feats, tmp_path / "masks",
                                    ["pano_camera0/pano_00000.jpg"], progress=0)
    assert stats["dropped"] == 0 and stats["kept"] == 2
    with h5py.File(str(feats), "r") as fd:
        assert np.allclose(fd["pano_camera0/pano_00000.jpg"]["keypoints"][...], kpts)


def test_filter_reports_fully_masked_images(tmp_path):
    masks = tmp_path / "masks" / "pano_camera0"
    masks.mkdir(parents=True)
    cv2.imwrite(str(masks / "pano_00000.jpg.png"), np.zeros((64, 64), np.uint8))
    feats = tmp_path / "feats.h5"
    write_features(feats, "pano_camera0/pano_00000.jpg", np.array([[5.0, 10.0]]))
    stats = filter_features_by_mask(feats, tmp_path / "masks",
                                    ["pano_camera0/pano_00000.jpg"], progress=0)
    assert stats["kept"] == 0
    assert stats["empty_images"] == ["pano_camera0/pano_00000.jpg"]


def test_missing_mask_is_counted_and_leaves_features_alone(tmp_path):
    (tmp_path / "masks").mkdir()
    feats = tmp_path / "feats.h5"
    write_features(feats, "pano_camera0/pano_00000.jpg", np.array([[5.0, 10.0]]))
    stats = filter_features_by_mask(feats, tmp_path / "masks",
                                    ["pano_camera0/pano_00000.jpg"], progress=0)
    assert stats["missing_masks"] == 1 and stats["images"] == 0


def test_verify_mask_integrity_is_clean_after_filtering(tmp_path):
    masks = tmp_path / "masks" / "pano_camera0"
    masks.mkdir(parents=True)
    mask = np.zeros((64, 64), np.uint8)
    mask[:, 32:] = 255
    cv2.imwrite(str(masks / "pano_00000.jpg.png"), mask)
    feats = tmp_path / "feats.h5"
    names = ["pano_camera0/pano_00000.jpg"]
    write_features(feats, names[0], np.array([[5.0, 10.0], [40.0, 10.0]]))
    assert verify_mask_integrity(feats, tmp_path / "masks",
                                 names)["keypoints_on_masked_pixels"] == 1
    filter_features_by_mask(feats, tmp_path / "masks", names, progress=0)
    out = verify_mask_integrity(feats, tmp_path / "masks", names)
    assert out["keypoints_on_masked_pixels"] == 0 and out["keypoints_checked"] == 1


# --- pairs -----------------------------------------------------------------

def test_union_pairs_dedupes_regardless_of_order():
    a = [("x.jpg", "y.jpg"), ("y.jpg", "z.jpg")]
    b = [("y.jpg", "x.jpg"), ("z.jpg", "w.jpg"), ("q.jpg", "q.jpg")]
    out = union_pairs(a, b)
    assert out == [("x.jpg", "y.jpg"), ("y.jpg", "z.jpg"), ("w.jpg", "z.jpg")]


def test_write_pairs_is_one_space_separated_pair_per_line(tmp_path):
    p = write_pairs([("a/1.jpg", "b/2.jpg"), ("a/1.jpg", "b/3.jpg")],
                    tmp_path / "sub" / "pairs.txt")
    assert p.read_text(encoding="utf8").splitlines() == ["a/1.jpg b/2.jpg",
                                                         "a/1.jpg b/3.jpg"]


# --- CLI contract ----------------------------------------------------------

def test_features_defaults_to_sift_so_production_is_unchanged():
    args = build_parser().parse_args(["--capture-type", "360", "--dataset", "d",
                                      "--out_dir", "o"])
    assert args.features == "sift"
    assert args.retrieval_num == 0
    assert args.aliked_max_keypoints == 8192 and args.aliked_resize_max == 1600


def test_aliked_is_refused_on_the_spherical_arm(tmp_path, capsys):
    rc = main(["--capture-type", "360", "--arm", "spherical", "--features", "aliked",
               "--dataset", str(tmp_path), "--out_dir", str(tmp_path / "out")])
    assert rc == 2
    assert "crop_rig" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_sequential_pairing_options_are_shared_with_the_sift_matcher():
    from core.sfm_360 import sequential_pairing_options

    args = build_parser().parse_args(["--capture-type", "360", "--dataset", "d",
                                      "--out_dir", "o", "--overlap", "7",
                                      "--quadratic_overlap", "0", "--loop_detection", "1"])
    opts = sequential_pairing_options(args)
    assert opts.overlap == 7
    assert opts.quadratic_overlap is False
    assert opts.loop_detection is True
    assert opts.expand_rig_images is True


def test_image_size_survives_a_two_keypoint_image(tmp_path):
    """Regression: `image_size` has shape (2,). A filter that subsets any 1-D array as long
    as the keypoint array would corrupt it on an image with exactly 2 keypoints."""
    masks = tmp_path / "masks" / "pano_camera0"
    masks.mkdir(parents=True)
    mask = np.zeros((64, 64), np.uint8)
    mask[:, 32:] = 255
    cv2.imwrite(str(masks / "pano_00000.jpg.png"), mask)
    feats = tmp_path / "feats.h5"
    write_features(feats, "pano_camera0/pano_00000.jpg",
                   np.array([[5.0, 10.0], [40.0, 10.0]]))   # 1 dropped, 1 kept
    filter_features_by_mask(feats, tmp_path / "masks",
                            ["pano_camera0/pano_00000.jpg"], progress=0)
    with h5py.File(str(feats), "r") as fd:
        grp = fd["pano_camera0/pano_00000.jpg"]
        assert grp["keypoints"].shape == (1, 2)
        assert list(grp["image_size"][...]) == [64, 64]
