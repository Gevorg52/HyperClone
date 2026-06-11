"""Tests for object masking: mask I/O, overlay generation, RGBA compositing.

Tests pure logic only — no model loading, no GPU, no heavy dependencies.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.masking import (
    BIREFNET_VARIANTS,
    create_rgba_image,
    generate_overlay,
    load_mask,
    postprocess_mask,
    save_mask,
)


# ============================================================================
# Mask I/O tests
# ============================================================================

class TestMaskIO:
    """Tests for save_mask and load_mask round-trips."""

    def test_save_and_load_soft_mask(self, tmp_path):
        """Soft mask values should survive save/load round-trip."""
        mask = np.array([[0, 64, 128, 192, 255]], dtype=np.uint8)
        path = tmp_path / "mask.png"
        save_mask(mask, str(path), soft=True)

        loaded = load_mask(str(path))
        np.testing.assert_array_equal(loaded, mask)

    def test_save_and_load_binary_mask(self, tmp_path):
        """Binary mask should threshold at 128: below→0, above→255."""
        mask = np.array([[50, 100, 128, 200, 255]], dtype=np.uint8)
        path = tmp_path / "mask.png"
        save_mask(mask, str(path), soft=False)

        loaded = load_mask(str(path))
        expected = np.array([[0, 0, 0, 255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(loaded, expected)

    def test_save_float_mask_normalized(self, tmp_path):
        """Float [0, 1] mask should be scaled to [0, 255]."""
        mask = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        path = tmp_path / "mask.png"
        save_mask(mask, str(path), soft=True)

        loaded = load_mask(str(path))
        assert loaded[0, 0] == 0
        assert loaded[0, 2] == 255
        assert 100 <= loaded[0, 1] <= 140  # ~127.5

    def test_mask_is_grayscale(self, tmp_path):
        """Saved mask should be single-channel grayscale."""
        mask = np.ones((100, 100), dtype=np.uint8) * 200
        path = tmp_path / "mask.png"
        save_mask(mask, str(path))

        loaded = load_mask(str(path))
        assert loaded.ndim == 2, "Mask should be 2D (grayscale)"

    def test_mask_values_in_range(self, tmp_path):
        """All mask values should be in [0, 255]."""
        mask = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        path = tmp_path / "mask.png"
        save_mask(mask, str(path))

        loaded = load_mask(str(path))
        assert loaded.min() >= 0
        assert loaded.max() <= 255

    def test_load_nonexistent_raises(self):
        """Loading a nonexistent mask file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_mask("/nonexistent/path/mask.png")

    def test_save_creates_parent_dirs(self, tmp_path):
        """save_mask should create parent directories if needed."""
        path = tmp_path / "sub" / "dir" / "mask.png"
        mask = np.zeros((10, 10), dtype=np.uint8)
        save_mask(mask, str(path))
        assert path.exists()

    def test_roundtrip_2d_shape(self, tmp_path):
        """Mask shape should be preserved through save/load."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        path = tmp_path / "mask.png"
        save_mask(mask, str(path))
        loaded = load_mask(str(path))
        assert loaded.shape == (480, 640)


# ============================================================================
# Overlay generation tests
# ============================================================================

class TestOverlayGeneration:
    """Tests for generate_overlay visualization."""

    def test_output_shape_matches_input(self):
        """Overlay should have same shape as input image."""
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        mask = np.ones((100, 200), dtype=np.uint8) * 255
        overlay = generate_overlay(image, mask)
        assert overlay.shape == (100, 200, 3)

    def test_output_dtype_uint8(self):
        """Overlay should be uint8."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = np.ones((50, 50), dtype=np.uint8) * 128
        overlay = generate_overlay(image, mask)
        assert overlay.dtype == np.uint8

    def test_green_tint_in_masked_region(self):
        """Default green tint should increase green channel in foreground."""
        image = np.full((50, 50, 3), 100, dtype=np.uint8)
        mask = np.full((50, 50), 255, dtype=np.uint8)
        overlay = generate_overlay(image, mask, alpha=0.5, color=(0, 255, 0))

        # Green channel should be higher than red and blue
        assert overlay[25, 25, 1] > overlay[25, 25, 0]
        assert overlay[25, 25, 1] > overlay[25, 25, 2]

    def test_no_tint_in_background(self):
        """Background (mask=0) should be unchanged."""
        image = np.full((50, 50, 3), 100, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        overlay = generate_overlay(image, mask, alpha=0.5, color=(0, 255, 0))

        np.testing.assert_array_equal(overlay, image)

    def test_does_not_modify_original(self):
        """generate_overlay should not modify the original image."""
        image = np.full((50, 50, 3), 100, dtype=np.uint8)
        original_copy = image.copy()
        mask = np.full((50, 50), 255, dtype=np.uint8)
        generate_overlay(image, mask)
        np.testing.assert_array_equal(image, original_copy)


# ============================================================================
# RGBA composite tests
# ============================================================================

class TestRGBAComposite:
    """Tests for create_rgba_image compositing."""

    def _create_test_image(self, path: Path, w: int = 100, h: int = 80):
        """Create a simple test JPEG image."""
        img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        Image.fromarray(img).save(str(path))

    def _create_test_mask(self, path: Path, w: int = 100, h: int = 80, value: int = 200):
        """Create a simple test grayscale mask PNG."""
        mask = np.full((h, w), value, dtype=np.uint8)
        cv2.imwrite(str(path), mask)

    def test_output_has_4_channels(self, tmp_path):
        """RGBA output should have 4 channels."""
        img_path = tmp_path / "test.png"
        mask_path = tmp_path / "mask.png"
        out_path = tmp_path / "rgba.png"

        self._create_test_image(img_path)
        self._create_test_mask(mask_path)
        create_rgba_image(str(img_path), str(mask_path), str(out_path))

        rgba = Image.open(str(out_path))
        assert rgba.mode == "RGBA"

    def test_alpha_channel_matches_mask(self, tmp_path):
        """Alpha channel should match the mask values."""
        img_path = tmp_path / "test.png"
        mask_path = tmp_path / "mask.png"
        out_path = tmp_path / "rgba.png"

        self._create_test_image(img_path, 50, 50)
        self._create_test_mask(mask_path, 50, 50, value=128)
        create_rgba_image(str(img_path), str(mask_path), str(out_path))

        rgba = np.array(Image.open(str(out_path)))
        assert rgba.shape[2] == 4
        # Alpha channel should be 128 everywhere
        np.testing.assert_array_equal(rgba[:, :, 3], 128)

    def test_rgb_channels_match_source(self, tmp_path):
        """RGB channels should match the source image."""
        img_path = tmp_path / "test.png"
        mask_path = tmp_path / "mask.png"
        out_path = tmp_path / "rgba.png"

        # Create a known-color image
        img = np.full((50, 50, 3), [100, 150, 200], dtype=np.uint8)
        Image.fromarray(img).save(str(img_path))
        self._create_test_mask(mask_path, 50, 50, value=255)

        create_rgba_image(str(img_path), str(mask_path), str(out_path))

        rgba = np.array(Image.open(str(out_path)))
        np.testing.assert_array_equal(rgba[:, :, 0], 100)  # R
        np.testing.assert_array_equal(rgba[:, :, 1], 150)  # G
        np.testing.assert_array_equal(rgba[:, :, 2], 200)  # B

    def test_creates_output_directory(self, tmp_path):
        """create_rgba_image should create parent dirs if needed."""
        img_path = tmp_path / "test.png"
        mask_path = tmp_path / "mask.png"
        out_path = tmp_path / "sub" / "dir" / "rgba.png"

        self._create_test_image(img_path)
        self._create_test_mask(mask_path)
        create_rgba_image(str(img_path), str(mask_path), str(out_path))
        assert out_path.exists()

    def test_mask_resized_to_match_image(self, tmp_path):
        """If mask and image have different sizes, mask should be resized."""
        img_path = tmp_path / "test.png"
        mask_path = tmp_path / "mask.png"
        out_path = tmp_path / "rgba.png"

        self._create_test_image(img_path, 200, 150)
        self._create_test_mask(mask_path, 100, 75)  # half size
        create_rgba_image(str(img_path), str(mask_path), str(out_path))

        rgba = Image.open(str(out_path))
        assert rgba.size == (200, 150)


# ============================================================================
# Post-processing tests
# ============================================================================

class TestPostprocessMask:
    """Tests for postprocess_mask threshold and erosion."""

    def test_no_postprocessing(self):
        """With threshold=0 and erosion=0, output matches input scaled to uint8."""
        mask = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        result = postprocess_mask(mask, threshold=0.0, erosion_px=0)
        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_threshold_zeros_low_confidence(self):
        """Pixels below threshold should become 0."""
        mask = np.array([[0.1, 0.3, 0.5, 0.7, 0.9]], dtype=np.float32)
        result = postprocess_mask(mask, threshold=0.5, erosion_px=0)
        assert result[0, 0] == 0   # 0.1 < 0.5
        assert result[0, 1] == 0   # 0.3 < 0.5
        assert result[0, 2] > 0    # 0.5 >= 0.5, preserved
        assert result[0, 3] > 0    # 0.7 >= 0.5, preserved
        assert result[0, 4] > 0    # 0.9 >= 0.5, preserved

    def test_threshold_preserves_values_above(self):
        """Values above threshold should keep their proportional intensity."""
        mask = np.array([[0.8]], dtype=np.float32)
        result = postprocess_mask(mask, threshold=0.5, erosion_px=0)
        # 0.8 * 255 ≈ 204
        assert 200 <= result[0, 0] <= 210

    def test_erosion_shrinks_mask(self):
        """Erosion should remove thin features."""
        # Create a mask with a thin 1px line
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[25, :] = 255  # horizontal line
        result = postprocess_mask(mask, threshold=0.0, erosion_px=2)
        # The 1px line should be completely eroded away
        assert result[25, 25] == 0

    def test_erosion_preserves_bulk(self):
        """Erosion should preserve the interior of large regions."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255  # 60x60 filled square
        result = postprocess_mask(mask, threshold=0.0, erosion_px=3)
        # Center should still be foreground
        assert result[50, 50] == 255
        # Edge should be eroded
        assert result[20, 20] == 0

    def test_accepts_uint8_input(self):
        """Should handle uint8 input as well as float."""
        mask = np.array([[0, 100, 200, 255]], dtype=np.uint8)
        result = postprocess_mask(mask, threshold=0.5, erosion_px=0)
        assert result[0, 0] == 0    # 0/255 = 0.0 < 0.5
        assert result[0, 1] == 0    # 100/255 ≈ 0.39 < 0.5
        assert result[0, 2] > 0     # 200/255 ≈ 0.78 >= 0.5
        assert result[0, 3] == 255  # 255/255 = 1.0 >= 0.5


class TestVariantRegistry:
    """Tests for BiRefNet variant registry."""

    def test_default_variant_exists(self):
        from core.masking import BIREFNET_DEFAULT
        assert BIREFNET_DEFAULT in BIREFNET_VARIANTS

    def test_all_variants_have_model_id_and_size(self):
        for name, (hf_id, size) in BIREFNET_VARIANTS.items():
            assert isinstance(hf_id, str) and hf_id.startswith("ZhengPeng7/")
            assert isinstance(size, tuple) and len(size) == 2

    def test_general_is_1024(self):
        _, size = BIREFNET_VARIANTS["general"]
        assert size == (1024, 1024)

    def test_hr_matting_is_2048(self):
        _, size = BIREFNET_VARIANTS["HR matting (2048)"]
        assert size == (2048, 2048)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
