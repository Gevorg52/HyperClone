"""Tests for core.sfm_360 — WP-O2 360 SfM lane (rig geometry, masks, CLI gate).

Pure-geometry + IO coverage on synthetic panoramas. No COLMAP reconstruction is run here
(that is what the WP-O2 gate runs do); pycolmap is only used for the camera/rig objects.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.sfm_360 import (  # noqa: E402
    RENDER_PRESETS,
    PanoProcessor,
    build_parser,
    create_pano_rig_config,
    create_virtual_camera,
    get_virtual_camera_rays,
    get_virtual_rotations,
    list_panos,
    main,
    prior_output,
    spherical_img_from_cam,
    trajectory_stats,
)

PANO_W, PANO_H = 256, 128


def make_pano_dataset(root: Path, n: int = 3, mask_band=None):
    """n synthetic ERP frames + WP-O1-style masks (255 = VALID, 0 = EXCLUDED)."""
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, (PANO_H, PANO_W, 3), dtype=np.uint8)
        cv2.imwrite(str(images / f"pano_{i:05d}.jpg"), img)
        m = np.full((PANO_H, PANO_W), 255, np.uint8)
        if mask_band is not None:
            y0, y1 = mask_band
            m[y0:y1] = 0
        cv2.imwrite(str(masks / f"pano_{i:05d}.png"), m)
    return images, masks


# --------------------------------------------------------------------------- CLI gate

class TestCliGate:
    def test_off_by_default(self, capsys):
        assert main([]) == 2
        assert "OFF (default)" in capsys.readouterr().out

    def test_missing_dataset(self, capsys):
        assert main(["--capture-type", "360", "--out_dir", "x"]) == 2
        assert "--dataset" in capsys.readouterr().out

    def test_missing_out_dir(self, capsys):
        assert main(["--capture-type", "360", "--dataset", "x"]) == 2
        assert "--out_dir" in capsys.readouterr().out

    def test_defaults(self):
        a = build_parser().parse_args([])
        assert a.capture_type == "off"
        assert a.arm == "crop_rig"
        assert a.use_masks == 1
        assert a.render_type == "perspective_overlapping"


class TestRerunGuard:
    def test_prior_output_enumerates(self, tmp_path):
        assert prior_output(tmp_path) == []
        (tmp_path / "sparse").mkdir()
        (tmp_path / "sfm_report.json").write_text("{}")
        assert set(prior_output(tmp_path)) == {"sparse", "sfm_report.json"}

    def test_guard_refuses_and_writes_nothing(self, tmp_path, capsys):
        out = tmp_path / "out"
        out.mkdir()
        (out / "database.db").write_text("x")
        before = sorted(p.name for p in out.iterdir())
        rc = main(["--capture-type", "360", "--dataset", str(tmp_path / "nope"),
                   "--out_dir", str(out)])
        assert rc == 2
        assert "prior output" in capsys.readouterr().out
        assert sorted(p.name for p in out.iterdir()) == before

    def test_missing_image_dir_is_clean_error(self, tmp_path, capsys):
        rc = main(["--capture-type", "360", "--dataset", str(tmp_path / "nope"),
                   "--out_dir", str(tmp_path / "out")])
        assert rc == 1
        assert "ERROR:" in capsys.readouterr().out


# --------------------------------------------------------------------------- geometry

class TestVirtualRig:
    def test_overlapping_preset_is_12_crops(self):
        ro = RENDER_PRESETS["perspective_overlapping"]
        rots = get_virtual_rotations(ro.num_steps_yaw, ro.pitches_deg)
        assert len(rots) == 12
        assert len(get_virtual_rotations(4, (0.0,))) == 4

    def test_rotations_are_proper(self):
        ro = RENDER_PRESETS["perspective_overlapping"]
        for r in get_virtual_rotations(ro.num_steps_yaw, ro.pitches_deg):
            assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
            assert np.isclose(np.linalg.det(r), 1.0)

    def test_positive_pitch_row_is_yaw_staggered(self):
        """COLMAP staggers the +pitch row by half a yaw step; the -pitch row is not."""
        rots = get_virtual_rotations(4, (-35.0, 0.0, 35.0))
        centers = np.einsum("nij,i->nj", rots, [0, 0, 1])
        yaw = np.degrees(np.arctan2(centers[:, 0], centers[:, 2]))
        assert np.allclose(np.sort(yaw[0:4] % 360), [0, 90, 180, 270], atol=1e-6)
        assert np.allclose(np.sort(yaw[8:12] % 360), [45, 135, 225, 315], atol=1e-6)

    def test_virtual_camera_matches_pano_angular_resolution(self):
        cam = create_virtual_camera(3840, 1920, 90.0, 90.0)
        assert (cam.width, cam.height) == (960, 960)
        assert np.isclose(cam.params[0], 480.0)          # f = w / (2 tan(hfov/2))
        assert cam.model_name == "SIMPLE_PINHOLE"
        assert cam.has_prior_focal_length

    def test_rig_config_is_pure_rotation_with_one_reference(self):
        rots = get_virtual_rotations(4, (-35.0, 0.0, 35.0))
        rig = create_pano_rig_config(rots)
        assert len(rig.cameras) == 12
        assert [c.ref_sensor for c in rig.cameras].count(True) == 1
        assert rig.cameras[0].ref_sensor and rig.cameras[0].cam_from_rig is None
        assert rig.cameras[3].image_prefix == "pano_camera3/"
        for c in rig.cameras[1:]:
            assert np.allclose(c.cam_from_rig.translation, 0.0)   # shared optical centre

    def test_spherical_projection_landmarks(self):
        w, h = 3840, 1920
        xy = spherical_img_from_cam((w, h), np.array(
            [[0, 0, 1.0],        # forward -> ERP centre
             [1, 0, 0.0],        # +x      -> 3/4 width
             [-1, 0, 0.0],       # -x      -> 1/4 width
             [0, -1, 1e-9]]))    # up      -> top row
        assert np.allclose(xy[0], [w / 2, h / 2])
        assert np.allclose(xy[1], [3 * w / 4, h / 2])
        assert np.allclose(xy[2], [w / 4, h / 2])
        assert np.isclose(xy[3][1], 0.0, atol=1e-3)

    def test_spherical_projection_rejects_non_2to1(self):
        with pytest.raises(ValueError):
            spherical_img_from_cam((100, 100), np.array([[0, 0, 1.0]]))

    def test_solid_angle_of_one_crop_is_analytic(self):
        cam = create_virtual_camera(3840, 1920, 90.0, 90.0)
        rays = get_virtual_camera_rays(cam)
        sr = float((rays[:, 2] ** 3 / float(cam.params[0]) ** 2).sum())
        assert np.isclose(sr, 4 * np.arcsin(np.sin(np.pi / 4) ** 2), rtol=1e-5)


def _claim_counts(processor, n=4000, seed=1):
    """How many crops claim each of n random directions (0, 1 or - at boundaries - 2)."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    cam = processor.camera
    f, cx, cy = cam.params
    claims = np.zeros(len(dirs), int)
    for cam_idx, rot in enumerate(processor.cams_from_pano_rotation):
        in_cam = dirs @ rot.T                       # pano -> camera frame
        ok = in_cam[:, 2] > 1e-9
        u = f * in_cam[:, 0] / np.where(ok, in_cam[:, 2], 1) + cx
        v = f * in_cam[:, 1] / np.where(ok, in_cam[:, 2], 1) + cy
        inside = ok & (u >= 0) & (v >= 0) & (u < cam.width) & (v < cam.height)
        iu = np.clip(u.astype(int), 0, cam.width - 1)
        iv = np.clip(v.astype(int), 0, cam.height - 1)
        claims += inside & (processor.excl_masks[cam_idx][iv, iu] > 0)
    return claims


def _prepared(tmp_path, pano_w):
    p = PanoProcessor(tmp_path, None, tmp_path / "img", tmp_path / "msk",
                      RENDER_PRESETS["perspective_overlapping"])
    p.prepare(pano_w, pano_w // 2)
    return p


class TestExclusiveMasks:
    def test_nearly_every_direction_is_claimed_exactly_once(self, tmp_path):
        """The point of the exclusive masks: no pano pixel yields features twice."""
        claims = _claim_counts(_prepared(tmp_path, PANO_W))
        assert claims.max() <= 2          # only the 1-px-wide Voronoi boundary can double up
        assert (claims == 1).mean() > 0.95
        assert (claims >= 2).mean() < 0.02

    def test_double_claims_are_a_boundary_effect_that_shrinks_with_resolution(self, tmp_path):
        coarse = _claim_counts(_prepared(tmp_path / "a", 256))
        fine = _claim_counts(_prepared(tmp_path / "b", 1024))
        assert (fine >= 2).mean() < (coarse >= 2).mean()

    def test_reported_sphere_coverage(self, tmp_path):
        """12 crops + exclusive masks see ~98% of the sphere; the gap is the polar caps."""
        p = _prepared(tmp_path, PANO_W)
        assert 0.95 < p.sphere_coverage <= 1.0
        unclaimed = (_claim_counts(p) == 0).mean()
        assert abs(unclaimed - (1 - p.sphere_coverage)) < 0.02


class TestRenderAndMaskCompositing:
    def test_crops_and_masks_are_written(self, tmp_path):
        images, masks = make_pano_dataset(tmp_path, n=2)
        out_i, out_m = tmp_path / "ci", tmp_path / "cm"
        p = PanoProcessor(images, masks, out_i, out_m,
                          RENDER_PRESETS["perspective_overlapping"])
        p.prepare(PANO_W, PANO_H)
        p.process("pano_00000.jpg")
        for k in range(12):
            img = cv2.imread(str(out_i / f"pano_camera{k}" / "pano_00000.jpg"))
            msk = cv2.imread(str(out_m / f"pano_camera{k}" / "pano_00000.jpg.png"),
                             cv2.IMREAD_GRAYSCALE)
            assert img.shape[:2] == (p.camera.height, p.camera.width)
            assert msk.shape == (p.camera.height, p.camera.width)
            assert set(np.unique(msk)).issubset({0, 255})

    def test_wpo1_mask_shrinks_every_crop_mask(self, tmp_path):
        """A masked ERP band must remove crop pixels, never add them (AND, not OR)."""
        images, masks = make_pano_dataset(tmp_path, n=1, mask_band=(PANO_H // 2, PANO_H))
        p_ref = PanoProcessor(images, None, tmp_path / "i0", tmp_path / "m0",
                              RENDER_PRESETS["perspective_overlapping"])
        p_ref.prepare(PANO_W, PANO_H)
        p_ref.process("pano_00000.jpg")
        p = PanoProcessor(images, masks, tmp_path / "i1", tmp_path / "m1",
                          RENDER_PRESETS["perspective_overlapping"])
        p.prepare(PANO_W, PANO_H)
        p.process("pano_00000.jpg")
        shrunk = 0
        for k in range(12):
            a = cv2.imread(str(tmp_path / "m0" / f"pano_camera{k}" / "pano_00000.jpg.png"), 0)
            b = cv2.imread(str(tmp_path / "m1" / f"pano_camera{k}" / "pano_00000.jpg.png"), 0)
            assert np.all(b[a == 0] == 0)          # never re-enables an excluded pixel
            assert b.sum() <= a.sum()
            shrunk += int(b.sum() < a.sum())
        assert shrunk >= 4                         # the bottom rows lose real area

    def test_bottom_pitch_row_loses_most_to_a_nadir_band(self, tmp_path):
        images, masks = make_pano_dataset(tmp_path, n=1,
                                          mask_band=(int(PANO_H * 0.9), PANO_H))
        p = PanoProcessor(images, masks, tmp_path / "i", tmp_path / "m",
                          RENDER_PRESETS["perspective_overlapping"])
        p.prepare(PANO_W, PANO_H)
        stats = p.process("pano_00000.jpg")
        valid = np.array(stats["valid_frac"])
        excl = np.array(p.excl_frac)
        loss = 1 - valid / excl
        assert loss[0:4].mean() > loss[8:12].mean()   # -35 deg row looks down, +35 does not

    def test_rejects_non_2to1_panorama(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        cv2.imwrite(str(images / "bad.jpg"), np.zeros((100, 100, 3), np.uint8))
        p = PanoProcessor(images, None, tmp_path / "i", tmp_path / "m",
                          RENDER_PRESETS["perspective_overlapping"])
        p.prepare(PANO_W, PANO_H)
        with pytest.raises(ValueError):
            p.process("bad.jpg")

    def test_split_image_name_round_trip(self, tmp_path):
        p = PanoProcessor(tmp_path, None, tmp_path, tmp_path,
                          RENDER_PRESETS["perspective_overlapping"])
        assert p.split_image_name("pano_camera7/pano_00042.jpg") == (7, "pano_00042.jpg")
        with pytest.raises(ValueError):
            p.split_image_name("pano_00042.jpg")


# --------------------------------------------------------------------------- misc

class TestHelpers:
    def test_list_panos_sorted_and_capped(self, tmp_path):
        images, _ = make_pano_dataset(tmp_path, n=5)
        assert list_panos(images) == [f"pano_{i:05d}.jpg" for i in range(5)]
        assert list_panos(images, 2) == ["pano_00000.jpg", "pano_00001.jpg"]

    def test_list_panos_empty_dir(self, tmp_path):
        (tmp_path / "e").mkdir()
        with pytest.raises(FileNotFoundError):
            list_panos(tmp_path / "e")

    def test_trajectory_stats_on_a_straight_walk(self):
        centers = np.stack([np.arange(10.0), np.zeros(10), np.zeros(10)], 1)
        s = trajectory_stats(centers)
        assert s["n"] == 10
        assert np.isclose(s["step_median"], 1.0)
        assert np.isclose(s["path_over_displacement"], 1.0)
        assert s["num_steps_gt_5x_median"] == 0
        assert s["bbox_axes_sorted"][0] == 0

    def test_trajectory_stats_flags_a_jump(self):
        centers = np.stack([np.array([0, 1, 2, 3, 40.0]), np.zeros(5), np.zeros(5)], 1)
        assert trajectory_stats(centers)["num_steps_gt_5x_median"] == 1

    def test_report_schema_keys_are_json_serialisable(self):
        s = trajectory_stats(np.random.default_rng(0).normal(size=(20, 3)))
        json.dumps(s)          # numpy scalars would blow up here


# --------------------------------------------------------------------------- review fixes

class _FakeImage:
    def __init__(self, name):
        self.name = name


class _FakeRec:
    """Enough of a pycolmap.Reconstruction for reconstruction_metrics()."""

    def __init__(self, names):
        self.images = {i: _FakeImage(n) for i, n in enumerate(names)}

    def reg_image_ids(self):
        return list(self.images)

    def num_images(self):
        return len(self.images)

    num_reg_images = num_images

    def num_frames(self):
        return len(self.images)

    num_reg_frames = num_frames

    def num_points3D(self):
        return 0

    def compute_num_observations(self):
        return 0

    def compute_mean_reprojection_error(self):
        return 0.0

    def compute_mean_track_length(self):
        return 0.0

    def compute_mean_observations_per_reg_image(self):
        return 0.0


class TestStrictPanoRegistration:
    """A held-out split needs ALL 12 crops of a pano, not 'any crop registered'."""

    def test_partial_pano_counts_as_registered_but_not_fully(self):
        from core.sfm_360 import reconstruction_metrics

        names = [f"pano_camera{c}/pano_00000.jpg" for c in range(12)]      # complete
        names += [f"pano_camera{c}/pano_00001.jpg" for c in range(9)]      # 3 crops missing
        m = reconstruction_metrics(_FakeRec(names), 2,
                                   lambda n: n.split("/", 1)[1], images_per_pano=12)
        assert m["num_registered_panos"] == 2
        assert m["panos_fully_registered"] == 1
        assert m["images_per_pano"] == 12
        assert m["panos_fully_registered_pct"] == 50.0

    def test_defaults_to_one_image_per_pano(self):
        from core.sfm_360 import reconstruction_metrics

        m = reconstruction_metrics(_FakeRec(["pano_00000.jpg", "pano_00001.jpg"]), 2)
        assert m["num_registered_panos"] == m["panos_fully_registered"] == 2


class TestCaptureTypeChoices:
    def test_insta360_is_not_a_choice_of_this_module(self, capsys):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--capture-type", "insta360"])
        assert "invalid choice" in capsys.readouterr().err
