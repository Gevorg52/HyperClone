"""Tests for core.train_360_prep — WP-O3 dataset packaging (naming, split, layout).

Builds a tiny synthetic crop-rig reconstruction with pycolmap (no SfM is run) and packages
it, then checks the artifact against what core/training/gsplat_trainer.py actually reads.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.train_360_prep import (  # noqa: E402
    build_parser,
    flat_name,
    main,
    pano_split,
    split_rig_name,
)

CROP = 16


def make_crop_rig_dataset(root: Path, n_panos: int = 24, n_cams: int = 12) -> Path:
    """A WP-O2-shaped run dir: sparse/0 + images/pano_cameraN/ + masks/<name>.png."""
    import pycolmap

    images = root / "images"
    masks = root / "masks"
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera.create_from_model_id(
        camera_id=1, model=pycolmap.CameraModelId.SIMPLE_PINHOLE,
        focal_length=CROP / 2, width=CROP, height=CROP)
    rec.add_camera_with_trivial_rig(cam)

    rng = np.random.default_rng(0)
    image_id = 0
    for p in range(n_panos):
        for c in range(n_cams):
            image_id += 1
            name = f"pano_camera{c}/pano_{p:05d}.jpg"
            (images / f"pano_camera{c}").mkdir(parents=True, exist_ok=True)
            (masks / f"pano_camera{c}").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(images / name),
                        rng.integers(0, 255, (CROP, CROP, 3), dtype=np.uint8))
            m = np.full((CROP, CROP), 255, np.uint8)
            m[:2] = 0                                    # a masked-out band
            cv2.imwrite(str(masks / f"{name}.png"), m)
            # Cameras on a line, all looking down +z; only the poses' existence matters.
            cam_from_world = pycolmap.Rigid3d(
                pycolmap.Rotation3d(np.eye(3)),
                np.array([[float(p) * 0.1 + 0.01 * c], [0.0], [0.0]]))
            rec.add_image_with_trivial_frame(
                pycolmap.Image(name=name, camera_id=1, image_id=image_id),
                cam_from_world)
    for i, xyz in enumerate([[0.0, 0.0, 3.0], [1.0, 0.5, 4.0], [-1.0, 0.2, 5.0]]):
        rec.add_point3D_with_id(i + 1, pycolmap.Point3D(
            xyz=np.array(xyz), color=np.array([10, 20, 30], np.uint8),
            track=pycolmap.Track()))
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    rec.write(str(sparse))
    return root


# --------------------------------------------------------------------------- CLI gate

class TestCliGate:
    def test_off_by_default(self, capsys):
        assert main([]) == 2
        assert "OFF (default)" in capsys.readouterr().out

    def test_requires_sfm_and_out_dir(self):
        assert main(["--capture-type", "360"]) == 2
        assert main(["--capture-type", "360", "--sfm", "x"]) == 2

    def test_default_capture_type_is_off(self):
        assert build_parser().parse_args([]).capture_type == "off"

    def test_missing_input_exits_1(self, tmp_path, capsys):
        rc = main(["--capture-type", "360", "--sfm", str(tmp_path / "nope"),
                   "--out_dir", str(tmp_path / "out")])
        assert rc == 1 and "ERROR" in capsys.readouterr().out


# --------------------------------------------------------------------------- naming

class TestNaming:
    def test_split_rig_name(self):
        assert split_rig_name("pano_camera7/pano_00042.jpg") == (7, "pano_00042.jpg")
        assert split_rig_name("pano_camera0\\pano_00000.jpg") == (0, "pano_00000.jpg")
        for bad in ("pano_00042.jpg", "images/pano_00042.jpg", "pano_cameraX/a.jpg"):
            with pytest.raises(ValueError):
                split_rig_name(bad)

    def test_flat_name_is_pano_major_and_collision_free(self):
        assert flat_name(7, "pano_00042.jpg") == "pano_00042_cam07.jpg"
        names = [flat_name(c, f"pano_{p:05d}.jpg") for p in range(3) for c in range(12)]
        assert len(set(names)) == len(names)
        # sorted order must group all crops of a pano together (the split relies on it)
        panos = [n.split("_cam")[0] for n in sorted(names)]
        assert panos == sorted(panos)


# --------------------------------------------------------------------------- split

class TestPanoSplit:
    def test_every_eighth_pano(self):
        panos = [f"pano_{i:05d}.jpg" for i in range(24)]
        train, test = pano_split(panos, 8)
        assert len(test) == 3 and len(train) == 21
        assert test == ["pano_00000.jpg", "pano_00008.jpg", "pano_00016.jpg"]
        assert not set(train) & set(test)

    def test_offset_and_disable(self):
        panos = [f"p{i}" for i in range(10)]
        assert pano_split(panos, 4, holdout_offset=1)[1] == ["p1", "p5", "p9"]
        assert pano_split(panos, 0) == (panos, [])

    def test_rank_based_not_index_based(self):
        """Pano numbers are sampled from video and are NOT contiguous."""
        panos = ["pano_00001.jpg", "pano_00003.jpg", "pano_00007.jpg", "pano_00011.jpg"]
        assert pano_split(panos, 2)[1] == ["pano_00001.jpg", "pano_00007.jpg"]


# --------------------------------------------------------------------------- packaging

@pytest.fixture(scope="module")
def packaged(tmp_path_factory):
    root = tmp_path_factory.mktemp("wpo3")
    sfm = make_crop_rig_dataset(root / "sfm", n_panos=24, n_cams=12)
    out = root / "data"
    assert main(["--capture-type", "360", "--sfm", str(sfm), "--out_dir", str(out)]) == 0
    return out


class TestPackaging:
    def test_layout_matches_what_the_trainer_reads(self, packaged):
        assert (packaged / "sparse" / "0" / "cameras.bin").is_file()
        assert (packaged / "images" / "pano_00000_cam00.jpg").is_file()
        # trainer mask convention: os.path.splitext(image_name)[0] + ".png"
        assert (packaged / "masks" / "pano_00000_cam00.png").is_file()
        assert not (packaged / "masks" / "pano_00000_cam00.jpg.png").exists()
        assert len(list((packaged / "images").iterdir())) == 24 * 12
        assert len(list((packaged / "masks").iterdir())) == 24 * 12

    def test_model_names_were_rewritten(self, packaged):
        import pycolmap
        rec = pycolmap.Reconstruction(str(packaged / "sparse" / "0"))
        names = sorted(im.name for im in rec.images.values())
        assert names[0] == "pano_00000_cam00.jpg"
        assert all("/" not in n for n in names)
        assert rec.num_images() == 24 * 12

    def test_pixels_are_untouched(self, packaged, tmp_path_factory):
        a = cv2.imread(str(packaged / "images" / "pano_00003_cam05.jpg"))
        assert a is not None and a.shape == (CROP, CROP, 3)
        m = cv2.imread(str(packaged / "masks" / "pano_00003_cam05.png"),
                       cv2.IMREAD_GRAYSCALE)
        assert m[0, 0] == 0 and m[-1, -1] == 255      # the band survived, 255 = VALID

    def test_split_lists_hold_out_whole_panos(self, packaged):
        test = [ln.strip() for ln in (packaged / "holdout_test.txt").read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
        train = [ln.strip() for ln in (packaged / "holdout_train.txt").read_text().splitlines()
                 if ln.strip()]
        assert len(test) == 3 * 12 and len(train) == 21 * 12
        assert not set(test) & set(train)
        assert set(test) | set(train) == {f"pano_{p:05d}_cam{c:02d}.jpg"
                                          for p in range(24) for c in range(12)}
        te_panos = {n.split("_cam")[0] for n in test}
        tr_panos = {n.split("_cam")[0] for n in train}
        assert not te_panos & tr_panos                       # no intra-pano leakage
        for p in te_panos:                                   # all 12 crops together
            assert sum(1 for n in test if n.startswith(p + "_cam")) == 12

    def test_report_records_the_transformations(self, packaged):
        rep = json.loads((packaged / "prep_report.json").read_text())
        assert rep["wp"] == "WP-O3"
        assert rep["counts"]["crops_per_pano"] == 12
        assert rep["counts"]["panos_fully_registered"] == 24
        assert rep["counts"]["images_test"] == 36
        assert rep["counts"]["images_train"] == 252
        assert any("flattened" in t for t in rep["transformations"])
        assert rep["argv"]

    def test_rerun_guard(self, packaged, capsys):
        rc = main(["--capture-type", "360", "--sfm", str(packaged.parent / "sfm"),
                   "--out_dir", str(packaged)])
        assert rc == 2 and "prior output" in capsys.readouterr().out


class TestTrainerConsumesIt:
    def test_load_colmap_data_finds_masks_and_honours_holdout_list(self, tmp_path):
        from core.training.gsplat_trainer import load_colmap_data

        sfm = make_crop_rig_dataset(tmp_path / "sfm", n_panos=16, n_cams=12)
        out = tmp_path / "data"
        assert main(["--capture-type", "360", "--sfm", str(sfm), "--out_dir", str(out)]) == 0

        data = load_colmap_data(str(out), factor=1, test_every=8, normalize=False,
                                holdout_list=str(out / "holdout_test.txt"))
        assert len(data["image_paths"]) == 16 * 12
        assert data["mask_paths"] is not None
        assert all(Path(p).is_file() for p in data["mask_paths"])
        assert len(data["test_indices"]) == 2 * 12          # panos 0 and 8
        assert len(data["train_indices"]) == 14 * 12
        names = [Path(data["image_paths"][i]).name for i in data["test_indices"]]
        assert {n.split("_cam")[0] for n in names} == {"pano_00000", "pano_00008"}

    def test_holdout_list_rejects_unknown_names(self, tmp_path):
        from core.training.gsplat_trainer import load_colmap_data

        sfm = make_crop_rig_dataset(tmp_path / "sfm", n_panos=4, n_cams=2)
        out = tmp_path / "data"
        assert main(["--capture-type", "360", "--sfm", str(sfm), "--out_dir", str(out),
                     "--holdout_every", "2"]) == 0
        bad = tmp_path / "bad.txt"
        bad.write_text("pano_99999_cam00.jpg\n", encoding="utf8")
        with pytest.raises(ValueError, match="not in the reconstruction"):
            load_colmap_data(str(out), factor=1, normalize=False, holdout_list=str(bad))

    def test_without_holdout_list_the_every_nth_rule_is_unchanged(self, tmp_path):
        from core.training.gsplat_trainer import load_colmap_data

        sfm = make_crop_rig_dataset(tmp_path / "sfm", n_panos=4, n_cams=2)
        out = tmp_path / "data"
        assert main(["--capture-type", "360", "--sfm", str(sfm), "--out_dir", str(out)]) == 0
        data = load_colmap_data(str(out), factor=1, test_every=4, normalize=False)
        assert list(data["test_indices"]) == [0, 4]


class TestLossMaskPrimitives:
    """The pieces --loss_mask adds to gsplat_trainer (default OFF)."""

    def test_ssim_return_map_mean_matches_scalar_ssim(self):
        import torch
        from core.training.gsplat_trainer import ssim

        g = torch.Generator().manual_seed(0)
        a = torch.rand(1, 3, 48, 48, generator=g)
        b = (a + 0.1 * torch.rand(1, 3, 48, 48, generator=g)).clamp(0, 1)
        assert torch.allclose(ssim(a, b, return_map=True).mean(), ssim(a, b), atol=1e-6)
        assert ssim(a, b).ndim == 0                      # default return unchanged
        assert ssim(a, b, return_map=True).shape == a.shape

    def test_all_ones_mask_reproduces_the_unmasked_terms(self):
        """With mask=1 everywhere the masked averages must equal the plain means."""
        import torch
        from core.training.gsplat_trainer import ssim

        g = torch.Generator().manual_seed(1)
        r = torch.rand(24, 24, 3, generator=g)
        t = torch.rand(24, 24, 3, generator=g)
        m = torch.ones(24, 24)
        w = m.unsqueeze(-1)
        l1_masked = ((r - t).abs() * w).sum() / (w.sum() * 3)
        assert torch.allclose(l1_masked, (r - t).abs().mean(), atol=1e-6)
        smap = ssim(r.permute(2, 0, 1)[None], t.permute(2, 0, 1)[None], return_map=True)[0]
        ssim_masked = (smap * m).sum() / (m.sum() * smap.shape[0])
        assert torch.allclose(ssim_masked, smap.mean(), atol=1e-6)

    def test_flags_are_off_by_default(self):
        import subprocess
        import sys
        out = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "core" / "training" / "gsplat_trainer.py"),
             "--help"], capture_output=True, text=True).stdout
        assert "--loss_mask" in out and "--no-loss_mask" in out
        assert "--holdout_list" in out
