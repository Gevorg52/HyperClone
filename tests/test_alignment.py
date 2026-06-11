"""Tests for core.alignment — gravity alignment and centering.

Tests pure math (rotation, SVD), sidecar I/O, and mocked pycolmap interactions.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.alignment import (
    _rotation_between_vectors,
    read_sidecar_gravity,
    estimate_gravity_from_cameras,
    collect_gravity_vectors,
)


# ============================================================================
# TestRotationBetweenVectors — pure math, no pycolmap
# ============================================================================

class TestRotationBetweenVectors:
    """Tests for Rodrigues rotation between two unit vectors."""

    def test_identity_same_vector(self):
        """Same vector -> identity rotation."""
        v = np.array([0.0, 1.0, 0.0])
        R = _rotation_between_vectors(v, v)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90_degree_y_to_z(self):
        """Y-up -> Z-up is a 90-degree rotation."""
        v_from = np.array([0.0, 1.0, 0.0])
        v_to = np.array([0.0, 0.0, 1.0])
        R = _rotation_between_vectors(v_from, v_to)

        result = R @ v_from
        np.testing.assert_allclose(result, v_to, atol=1e-10)

    def test_180_degree_anti_parallel(self):
        """Y -> -Y is a 180-degree rotation."""
        v_from = np.array([0.0, 1.0, 0.0])
        v_to = np.array([0.0, -1.0, 0.0])
        R = _rotation_between_vectors(v_from, v_to)

        result = R @ v_from
        np.testing.assert_allclose(result, v_to, atol=1e-10)

    def test_preserves_norm(self):
        """Rotation should preserve vector magnitude."""
        v_from = np.array([0.3, 0.5, 0.8])
        v_from = v_from / np.linalg.norm(v_from)
        v_to = np.array([-0.2, 0.9, -0.4])
        v_to = v_to / np.linalg.norm(v_to)

        R = _rotation_between_vectors(v_from, v_to)
        test_vec = np.array([1.0, 2.0, 3.0])
        result = R @ test_vec
        np.testing.assert_allclose(np.linalg.norm(result), np.linalg.norm(test_vec), atol=1e-10)

    def test_result_is_orthogonal(self):
        """R @ R^T should equal I, and det(R) should be 1."""
        v_from = np.array([0.3, 0.5, 0.8])
        v_from = v_from / np.linalg.norm(v_from)
        v_to = np.array([-0.7, 0.1, 0.6])
        v_to = v_to / np.linalg.norm(v_to)

        R = _rotation_between_vectors(v_from, v_to)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_arbitrary_vectors(self):
        """General case: rotated v_from should equal v_to."""
        v_from = np.array([1.0, 0.0, 0.0])
        v_to = np.array([0.0, 0.0, 1.0])
        R = _rotation_between_vectors(v_from, v_to)

        result = R @ v_from
        np.testing.assert_allclose(result, v_to, atol=1e-10)

    def test_anti_parallel_x(self):
        """X -> -X should work (exercises the ortho vector selection)."""
        v_from = np.array([1.0, 0.0, 0.0])
        v_to = np.array([-1.0, 0.0, 0.0])
        R = _rotation_between_vectors(v_from, v_to)

        result = R @ v_from
        np.testing.assert_allclose(result, v_to, atol=1e-10)


# ============================================================================
# TestSidecarIO — filesystem, no pycolmap
# ============================================================================

class TestSidecarIO:
    """Tests for sidecar JSON gravity reader."""

    def test_read_from_image_dir(self, tmp_path):
        """Sidecar in same directory as image."""
        sidecar = {
            "version": 1,
            "gravity": [0.0, -1.0, 0.0],
            "timestamp": 1709234567.123,
            "source": "test",
        }
        (tmp_path / "photo001.json").write_text(json.dumps(sidecar))

        result = read_sidecar_gravity("photo001", tmp_path, tmp_path)
        assert result is not None
        np.testing.assert_allclose(result, [0.0, -1.0, 0.0], atol=1e-10)

    def test_read_from_imu_subfolder(self, tmp_path):
        """Sidecar in imu/ subfolder of image dir."""
        imu_dir = tmp_path / "imu"
        imu_dir.mkdir()
        sidecar = {"version": 1, "gravity": [0.0, 0.0, -1.0]}
        (imu_dir / "photo001.json").write_text(json.dumps(sidecar))

        result = read_sidecar_gravity("photo001", tmp_path, tmp_path)
        assert result is not None
        np.testing.assert_allclose(result, [0.0, 0.0, -1.0], atol=1e-10)

    def test_read_from_project_imu_dir(self, tmp_path):
        """Sidecar in project_dir/imu/."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        imu_dir = project_dir / "imu"
        imu_dir.mkdir()

        sidecar = {"version": 1, "gravity": [0.1, -0.9, 0.0]}
        (imu_dir / "photo001.json").write_text(json.dumps(sidecar))

        result = read_sidecar_gravity("photo001", image_dir, project_dir)
        assert result is not None
        # Should be normalized
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-10)

    def test_missing_sidecar_returns_none(self, tmp_path):
        result = read_sidecar_gravity("nonexistent", tmp_path, tmp_path)
        assert result is None

    def test_invalid_json_returns_none(self, tmp_path):
        (tmp_path / "bad.json").write_text("not valid json {{{")
        result = read_sidecar_gravity("bad", tmp_path, tmp_path)
        assert result is None

    def test_zero_gravity_returns_none(self, tmp_path):
        sidecar = {"version": 1, "gravity": [0.0, 0.0, 0.0]}
        (tmp_path / "zero.json").write_text(json.dumps(sidecar))
        result = read_sidecar_gravity("zero", tmp_path, tmp_path)
        assert result is None

    def test_non_unit_vector_normalized(self, tmp_path):
        """Non-unit gravity vector should be normalized."""
        sidecar = {"version": 1, "gravity": [0.0, -5.0, 0.0]}
        (tmp_path / "big.json").write_text(json.dumps(sidecar))

        result = read_sidecar_gravity("big", tmp_path, tmp_path)
        assert result is not None
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-10)
        np.testing.assert_allclose(result, [0.0, -1.0, 0.0], atol=1e-10)

    def test_missing_gravity_field_returns_none(self, tmp_path):
        sidecar = {"version": 1, "timestamp": 123}
        (tmp_path / "nofield.json").write_text(json.dumps(sidecar))
        result = read_sidecar_gravity("nofield", tmp_path, tmp_path)
        assert result is None

    def test_wrong_gravity_length_returns_none(self, tmp_path):
        sidecar = {"version": 1, "gravity": [0.0, -1.0]}
        (tmp_path / "short.json").write_text(json.dumps(sidecar))
        result = read_sidecar_gravity("short", tmp_path, tmp_path)
        assert result is None


# ============================================================================
# TestSVDGravityEstimation — mocked pycolmap
# ============================================================================

def _make_mock_image(rotation_matrix):
    """Create a mock pycolmap image with given rotation matrix."""
    image = MagicMock()
    rot = MagicMock()
    rot.matrix.return_value = rotation_matrix
    image.cam_from_world.rotation = rot
    image.cam_from_world.translation = np.zeros(3)
    return image


class TestSVDGravityEstimation:
    """Tests for SVD-based gravity estimation from camera poses."""

    def test_horizontal_cameras_give_y_up(self):
        """4 cameras on a horizontal plane with X-axes in XZ plane -> Y-up."""
        # Cameras at 0, 90, 180, 270 degrees around Y-axis
        # Camera looking along -Z: R = I (X=[1,0,0], Y=[0,1,0], Z=[0,0,1])
        R0 = np.eye(3)
        # Camera rotated 90 degrees around Y
        R90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
        # Camera rotated 180 degrees around Y
        R180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float)
        # Camera rotated 270 degrees around Y
        R270 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=float)

        rec = MagicMock()
        rec.images = {
            1: _make_mock_image(R0),
            2: _make_mock_image(R90),
            3: _make_mock_image(R180),
            4: _make_mock_image(R270),
        }

        up = estimate_gravity_from_cameras(rec)
        # Should be close to [0, 1, 0]
        assert abs(up[1]) > 0.9, f"Expected Y-up, got {up}"
        assert up[1] > 0, f"Expected positive Y, got {up}"
        np.testing.assert_allclose(np.linalg.norm(up), 1.0, atol=1e-10)

    def test_returns_unit_vector(self):
        """Result should always be a unit vector."""
        R0 = np.eye(3)
        R90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)

        rec = MagicMock()
        rec.images = {
            1: _make_mock_image(R0),
            2: _make_mock_image(R90),
        }

        up = estimate_gravity_from_cameras(rec)
        np.testing.assert_allclose(np.linalg.norm(up), 1.0, atol=1e-10)

    def test_single_image_returns_y_axis(self):
        """Single image -> falls back to camera Y-axis."""
        R = np.eye(3)
        rec = MagicMock()
        rec.images = {1: _make_mock_image(R)}

        up = estimate_gravity_from_cameras(rec)
        # Camera Y-axis when R=I is [0, 1, 0]
        np.testing.assert_allclose(up, [0.0, 1.0, 0.0], atol=1e-10)

    def test_no_images_returns_default(self):
        """Empty reconstruction -> default Y-up."""
        rec = MagicMock()
        rec.images = {}

        up = estimate_gravity_from_cameras(rec)
        np.testing.assert_allclose(up, [0.0, 1.0, 0.0], atol=1e-10)

    def test_degenerate_same_direction_falls_back(self):
        """All cameras facing the same direction -> degenerate, falls back to Y-axis average."""
        R = np.eye(3)
        rec = MagicMock()
        # All cameras with identical rotation
        rec.images = {i: _make_mock_image(R.copy()) for i in range(10)}

        up = estimate_gravity_from_cameras(rec)
        # Should still return a valid unit vector (Y-axis average)
        np.testing.assert_allclose(np.linalg.norm(up), 1.0, atol=1e-10)
        # With R=I for all, Y-axis is [0,1,0], should get Y-up
        assert abs(up[1]) > 0.9

    def test_z_up_cameras(self):
        """Cameras on horizontal plane with Z as up -> should detect Z-up."""
        # Cameras rotated so Z is the vertical axis
        # Looking along -Y with Z up: X=[1,0,0], Y=[0,0,1], Z=[0,-1,0]
        R0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
        # Rotated 90 degrees around Z
        R90 = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float)
        # Rotated 180 degrees around Z
        R180 = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        # Rotated 270 degrees around Z
        R270 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)

        rec = MagicMock()
        rec.images = {
            1: _make_mock_image(R0),
            2: _make_mock_image(R90),
            3: _make_mock_image(R180),
            4: _make_mock_image(R270),
        }

        up = estimate_gravity_from_cameras(rec)
        # Should detect Z-up (aligned with camera Y average)
        np.testing.assert_allclose(np.linalg.norm(up), 1.0, atol=1e-10)
        # The dominant component should be Z
        assert abs(up[2]) > 0.9, f"Expected Z-up, got {up}"


# ============================================================================
# TestGravityCollector — mocked I/O
# ============================================================================

class TestGravityCollector:
    """Tests for collect_gravity_vectors."""

    def test_collects_from_sidecar(self, tmp_path):
        """Should collect gravity from sidecar JSONs."""
        sidecar = {"version": 1, "gravity": [0.0, -1.0, 0.0]}
        (tmp_path / "img001.json").write_text(json.dumps(sidecar))

        image = MagicMock()
        image.name = "img001.jpg"

        rec = MagicMock()
        rec.images = {1: image}

        result = collect_gravity_vectors(rec, tmp_path, tmp_path)
        assert 1 in result
        np.testing.assert_allclose(result[1], [0.0, -1.0, 0.0], atol=1e-10)

    def test_empty_when_no_sidecars(self, tmp_path):
        """No sidecar files -> empty dict."""
        image = MagicMock()
        image.name = "img001.jpg"

        rec = MagicMock()
        rec.images = {1: image}

        result = collect_gravity_vectors(rec, tmp_path, tmp_path)
        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
