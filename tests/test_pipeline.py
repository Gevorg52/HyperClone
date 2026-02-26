"""Tests for Apex Reality SfM Pipeline.

Tests pure logic, config builders, state management, and bugfix verification.
Heavy dependencies (hloc, pycolmap) are mocked where needed.
"""

import sys
from pathlib import Path
from dataclasses import fields

import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Source file paths for source-level verification tests
SRC_RECONSTRUCTION = PROJECT_ROOT / "gui" / "stages" / "reconstruction.py"
SRC_IMPORT_QC = PROJECT_ROOT / "gui" / "stages" / "import_qc.py"
SRC_EXPORT = PROJECT_ROOT / "gui" / "stages" / "export.py"
SRC_EXTRACTION = PROJECT_ROOT / "gui" / "stages" / "extraction.py"

from core.pipeline_state import PipelineState
from core.feature_configs import build_extraction_config
from core.matcher_configs import build_matching_config

# Check optional heavy deps
try:
    import pycolmap
    HAS_PYCOLMAP = True
except ImportError:
    HAS_PYCOLMAP = False


# ============================================================================
# PipelineState tests
# ============================================================================

class TestPipelineState:
    """Tests for core.pipeline_state.PipelineState."""

    def test_default_construction(self):
        state = PipelineState()
        assert state.project_dir is None
        assert state.image_dir is None
        assert state.image_list == []
        assert state.excluded_images == []
        assert state.num_images == 0
        assert state.feature_path is None
        assert state.match_path is None
        assert state.sfm_dir is None
        assert state.refined is False
        assert state.export_dir is None

    def test_all_fields_serializable(self):
        """PipelineState must only contain serializable types for gr.State."""
        state = PipelineState()
        for f in fields(state):
            val = getattr(state, f.name)
            assert isinstance(val, (type(None), str, int, float, bool, list, dict)), (
                f"Field '{f.name}' has non-serializable type: {type(val)}"
            )

    def test_mutable_defaults_independent(self):
        """Each PipelineState instance must have independent mutable fields."""
        s1 = PipelineState()
        s2 = PipelineState()
        s1.image_list.append("test.jpg")
        assert s2.image_list == [], "Mutable default leaked between instances"

    def test_field_assignment(self):
        state = PipelineState()
        state.project_dir = "/tmp/test"
        state.num_images = 42
        state.refined = True
        assert state.project_dir == "/tmp/test"
        assert state.num_images == 42
        assert state.refined is True

    def test_save_and_load(self, tmp_path):
        """State should persist to disk and restore correctly."""
        state = PipelineState()
        state.project_dir = str(tmp_path)
        state.image_dir = "/fake/images"
        state.feature_path = "/fake/features.h5"
        state.num_images = 42
        state.refined = True
        state.image_list = ["a.jpg", "b.jpg"]
        state.save()

        restored = PipelineState.load(str(tmp_path))
        assert restored.image_dir == "/fake/images"
        assert restored.feature_path == "/fake/features.h5"
        assert restored.num_images == 42
        assert restored.refined is True
        assert restored.image_list == ["a.jpg", "b.jpg"]

    def test_load_missing_file(self, tmp_path):
        """Loading from dir without state file returns fresh state."""
        state = PipelineState.load(str(tmp_path))
        assert state.project_dir is None
        assert state.num_images == 0

    def test_load_corrupt_file(self, tmp_path):
        """Loading corrupt JSON returns fresh state."""
        (tmp_path / "pipeline_state.json").write_text("not json{{{")
        state = PipelineState.load(str(tmp_path))
        assert state.project_dir is None


# ============================================================================
# Feature config tests
# ============================================================================

class TestFeatureConfigs:
    """Tests for core.feature_configs.build_extraction_config."""

    def test_superpoint_default(self):
        conf = build_extraction_config("superpoint")
        assert conf["model"]["name"] == "superpoint"
        assert conf["model"]["max_keypoints"] == 8192
        assert conf["model"]["nms_radius"] == 3
        assert conf["preprocessing"]["grayscale"] is True
        assert "output" in conf

    def test_superpoint_custom_params(self):
        conf = build_extraction_config(
            "superpoint", max_keypoints=4096, nms_radius=2, resize_max=1600
        )
        assert conf["model"]["max_keypoints"] == 4096
        assert conf["model"]["nms_radius"] == 2
        assert conf["preprocessing"]["resize_max"] == 1600

    def test_disk_config(self):
        conf = build_extraction_config("disk")
        assert conf["model"]["name"] == "disk"
        assert conf["model"]["max_keypoints"] == 8192
        assert conf["preprocessing"]["grayscale"] is False

    def test_aliked_uses_max_num_keypoints(self):
        """ALIKED uses 'max_num_keypoints' (not 'max_keypoints')."""
        conf = build_extraction_config("aliked", max_keypoints=2048)
        assert conf["model"]["max_num_keypoints"] == 2048
        assert "max_keypoints" not in conf["model"]

    def test_full_res_uses_sentinel(self):
        """When resize_max=None (full res), HLoc gets a large sentinel value."""
        conf = build_extraction_config("superpoint", resize_max=None)
        assert conf["preprocessing"]["resize_max"] == 99999

    def test_as_half_propagated(self):
        conf = build_extraction_config("superpoint", as_half=True)
        assert conf["preprocessing"]["as_half"] is True

    def test_unknown_extractor_raises(self):
        with pytest.raises(ValueError, match="Unknown extractor"):
            build_extraction_config("nonexistent")

    def test_output_contains_keypoints_and_resize(self):
        conf = build_extraction_config("superpoint", max_keypoints=4096, resize_max=1600)
        assert "4096" in conf["output"]
        assert "1600" in conf["output"]


# ============================================================================
# Matcher config tests
# ============================================================================

class TestMatcherConfigs:
    """Tests for core.matcher_configs.build_matching_config."""

    def test_lightglue_default(self):
        conf = build_matching_config()
        assert conf["model"]["name"] == "lightglue"
        assert conf["model"]["features"] == "aliked"
        assert conf["model"]["depth_confidence"] == -1
        assert conf["model"]["width_confidence"] == -1

    def test_lightglue_custom(self):
        conf = build_matching_config(
            features="disk", depth_confidence=0.0, width_confidence=0.0, filter_threshold=0.2
        )
        assert conf["model"]["features"] == "disk"
        assert conf["model"]["depth_confidence"] == 0.0
        assert conf["model"]["filter_threshold"] == 0.2

    def test_output_includes_feature_type(self):
        conf = build_matching_config(features="aliked")
        assert "aliked" in conf["output"]

    def test_unknown_matcher_raises(self):
        with pytest.raises(ValueError, match="Unknown matcher"):
            build_matching_config(matcher="brute_force")


# ============================================================================
# Bugfix verification tests (source-level checks — read .py files directly)
# ============================================================================

class TestBug1_ReconstructionCameraModel:
    """Bug 1: camera_model must be passed via image_options, not as kwarg."""

    def test_reconstruction_passes_image_options(self):
        source = SRC_RECONSTRUCTION.read_text()
        assert "image_options=" in source, "image_options not found in reconstruction call"
        assert "camera_model=camera_model" not in source, (
            "camera_model=camera_model should not be a direct kwarg"
        )
        assert '"camera_model"' in source or "'camera_model'" in source


class TestBug2_ExifFallbackTypes:
    """Bug 2: EXIF fallback values must match Dataframe column datatypes."""

    def test_width_height_fallback_numeric(self):
        source = SRC_IMPORT_QC.read_text()
        assert 'exif.get("width") or 0' in source or "exif.get('width') or 0" in source
        assert 'exif.get("height") or 0' in source or "exif.get('height') or 0" in source

    def test_camera_fallback_is_string(self):
        source = SRC_IMPORT_QC.read_text()
        assert '"unknown"' in source or "'unknown'" in source

    def test_no_question_mark_for_numeric_columns(self):
        """Width and Height columns should not fall back to '?'."""
        source = SRC_IMPORT_QC.read_text()
        # Find the load_images table_rows section — check that "?" only appears
        # for non-numeric fallbacks
        lines = source.split("\n")
        for line in lines:
            if 'exif.get("width")' in line or "exif.get('width')" in line:
                assert '"?"' not in line, f"Width fallback should not be '?': {line.strip()}"
            if 'exif.get("height")' in line or "exif.get('height')" in line:
                assert '"?"' not in line, f"Height fallback should not be '?': {line.strip()}"


class TestBug3_ApplyTableChangesReturns:
    """Bug 3: apply_table_changes must return (status_msg, state)."""

    def test_apply_returns_tuple(self):
        from gui.stages.import_qc import apply_table_changes

        state = PipelineState()
        state.image_dir = "/fake/dir"

        table_data = [
            ["img1.jpg", 200.0, 6000, 4000, "24.0", "Sony", True],
            ["img2.jpg", 50.0, 6000, 4000, "24.0", "Sony", False],
        ]
        result = apply_table_changes(table_data, state)
        assert isinstance(result, tuple), "apply_table_changes should return a tuple"
        assert len(result) == 2, "Should return (status_msg, state)"
        status_msg, returned_state = result
        assert isinstance(status_msg, str)
        assert "1 images included" in status_msg
        assert "1 excluded" in status_msg

    def test_apply_empty_table_returns_tuple(self):
        from gui.stages.import_qc import apply_table_changes

        state = PipelineState()
        result = apply_table_changes([], state)
        assert isinstance(result, tuple), "Early return should also be a tuple"
        assert len(result) == 2

    def test_apply_wiring_targets_cull_info(self):
        source = SRC_IMPORT_QC.read_text()
        assert 'c["cull_info"]' in source


class TestBug4_ColmapBooleanFormat:
    """Bug 4: COLMAP CLI booleans must be 'true'/'false', not '1'/'0'."""

    def test_no_int_conversion_for_booleans(self):
        source = SRC_EXPORT.read_text()
        assert "str(int(refine_focal))" not in source
        assert "str(int(refine_principal))" not in source
        assert "str(int(refine_extra))" not in source

    def test_uses_lower_conversion(self):
        source = SRC_EXPORT.read_text()
        assert "str(refine_focal).lower()" in source
        assert "str(refine_principal).lower()" in source
        assert "str(refine_extra).lower()" in source

    def test_boolean_conversion_produces_correct_strings(self):
        assert str(True).lower() == "true"
        assert str(False).lower() == "false"


class TestBug5_BAOptionsSafety:
    """Bug 5: pycolmap BA options should have a safety check."""

    def test_ba_options_has_safety_check(self):
        source = SRC_EXPORT.read_text()
        assert "AttributeError" in source
        assert "pycolmap BA API mismatch" in source


class TestBug6_NoDeadImageListWrite:
    """Bug 6: extraction.py should NOT write image_list.txt to disk."""

    def test_no_image_list_file_write(self):
        source = SRC_EXTRACTION.read_text()
        assert "image_list.txt" not in source, "Dead code writing image_list.txt should be removed"
        assert "image_list_path" not in source


# ============================================================================
# Utility function tests
# ============================================================================

class TestUtils:
    """Tests for core.utils functions."""

    def test_get_image_list_filters_extensions(self, tmp_path):
        (tmp_path / "photo.jpg").write_bytes(b"fake")
        (tmp_path / "photo.jpeg").write_bytes(b"fake")
        (tmp_path / "photo.png").write_bytes(b"fake")
        (tmp_path / "photo.tif").write_bytes(b"fake")
        (tmp_path / "photo.bmp").write_bytes(b"fake")  # unsupported
        (tmp_path / "readme.txt").write_bytes(b"text")  # unsupported

        from core.utils import get_image_list

        result = get_image_list(str(tmp_path))
        names = {p.name for p in result}
        assert "photo.jpg" in names
        assert "photo.jpeg" in names
        assert "photo.png" in names
        assert "photo.tif" in names
        assert "photo.bmp" not in names
        assert "readme.txt" not in names

    def test_get_image_list_sorted(self, tmp_path):
        (tmp_path / "c.jpg").write_bytes(b"fake")
        (tmp_path / "a.jpg").write_bytes(b"fake")
        (tmp_path / "b.jpg").write_bytes(b"fake")

        from core.utils import get_image_list

        result = get_image_list(str(tmp_path))
        names = [p.name for p in result]
        assert names == ["a.jpg", "b.jpg", "c.jpg"]

    def test_get_image_list_nonexistent_raises(self):
        from core.utils import get_image_list

        with pytest.raises(FileNotFoundError):
            get_image_list("/nonexistent/path/that/should/not/exist")

    def test_get_image_list_empty_folder(self, tmp_path):
        from core.utils import get_image_list

        result = get_image_list(str(tmp_path))
        assert result == []

    def test_create_project_dir(self, tmp_path):
        from core.utils import create_project_dir

        result = create_project_dir(str(tmp_path), "my_photos")
        assert result.exists()
        assert result.is_dir()
        assert result.name == "apex_my_photos"

    def test_create_project_dir_sanitizes_name(self, tmp_path):
        from core.utils import create_project_dir

        result = create_project_dir(str(tmp_path), "my photos (2024)")
        assert result.exists()
        assert " " not in result.name
        assert "(" not in result.name

    def test_validate_exif_consistency_empty(self):
        from core.utils import validate_exif_consistency

        assert validate_exif_consistency([]) == []

    def test_validate_exif_consistency_all_same(self):
        from core.utils import validate_exif_consistency

        exifs = [
            {"width": 6000, "height": 4000, "focal_length_mm": 24.0},
            {"width": 6000, "height": 4000, "focal_length_mm": 24.0},
        ]
        warnings = validate_exif_consistency(exifs)
        assert warnings == []

    def test_validate_exif_consistency_mixed_resolution(self):
        from core.utils import validate_exif_consistency

        exifs = [
            {"width": 6000, "height": 4000, "focal_length_mm": 24.0},
            {"width": 4000, "height": 3000, "focal_length_mm": 24.0},
        ]
        warnings = validate_exif_consistency(exifs)
        assert any("resolution" in w.lower() for w in warnings)

    def test_validate_exif_consistency_mixed_focal(self):
        from core.utils import validate_exif_consistency

        exifs = [
            {"width": 6000, "height": 4000, "focal_length_mm": 24.0},
            {"width": 6000, "height": 4000, "focal_length_mm": 35.0},
        ]
        warnings = validate_exif_consistency(exifs)
        assert any("focal" in w.lower() for w in warnings)

    def test_validate_exif_consistency_no_focal(self):
        from core.utils import validate_exif_consistency

        exifs = [
            {"width": 6000, "height": 4000, "focal_length_mm": None},
        ]
        warnings = validate_exif_consistency(exifs)
        assert any("focal" in w.lower() for w in warnings)

    def test_create_3dgs_directory_layout(self, tmp_path):
        from core.utils import create_3dgs_directory

        sfm_dir = tmp_path / "sfm"
        sfm_dir.mkdir()
        (sfm_dir / "cameras.bin").write_bytes(b"fake")
        (sfm_dir / "images.bin").write_bytes(b"fake")
        (sfm_dir / "points3D.bin").write_bytes(b"fake")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "photo1.jpg").write_bytes(b"fake")
        (img_dir / "photo2.jpg").write_bytes(b"fake")

        export_dir = tmp_path / "export"
        result = create_3dgs_directory(str(sfm_dir), str(img_dir), str(export_dir))

        assert (result / "sparse" / "0" / "cameras.bin").exists()
        assert (result / "sparse" / "0" / "images.bin").exists()
        assert (result / "sparse" / "0" / "points3D.bin").exists()
        assert (result / "images" / "photo1.jpg").exists()
        assert (result / "images" / "photo2.jpg").exists()


# ============================================================================
# Import & QC logic tests (no real images needed)
# ============================================================================

class TestImportQCLogic:
    """Tests for import_qc.py logic functions."""

    def test_auto_cull_excludes_below_threshold(self):
        from gui.stages.import_qc import auto_cull

        state = PipelineState()
        state.image_dir = "/fake/dir"

        table = [
            ["sharp.jpg", 200.0, 6000, 4000, "24.0", "Sony", True],
            ["blurry.jpg", 50.0, 6000, 4000, "24.0", "Sony", True],
            ["ok.jpg", 100.0, 6000, 4000, "24.0", "Sony", True],
        ]
        updated_rows, cull_msg, new_state = auto_cull(table, 100.0, state)

        assert updated_rows[0][6] is True   # sharp
        assert updated_rows[1][6] is False  # blurry
        assert updated_rows[2][6] is True   # ok (equal to threshold)
        assert "1 images below threshold" in cull_msg
        assert new_state.num_images == 2

    def test_auto_cull_preserves_row_data(self):
        from gui.stages.import_qc import auto_cull

        state = PipelineState()
        state.image_dir = "/fake/dir"

        table = [["img.jpg", 150.0, 6000, 4000, "24.0", "Sony", True]]
        updated_rows, _, _ = auto_cull(table, 100.0, state)

        assert updated_rows[0][0] == "img.jpg"
        assert updated_rows[0][1] == 150.0
        assert updated_rows[0][2] == 6000

    def test_apply_table_changes_updates_state(self):
        from gui.stages.import_qc import apply_table_changes

        state = PipelineState()
        state.image_dir = "/fake/dir"

        table = [
            ["a.jpg", 200.0, 6000, 4000, "24.0", "Sony", True],
            ["b.jpg", 50.0, 6000, 4000, "24.0", "Sony", False],
            ["c.jpg", 150.0, 6000, 4000, "24.0", "Sony", True],
        ]
        status, new_state = apply_table_changes(table, state)

        assert len(new_state.image_list) == 2
        assert len(new_state.excluded_images) == 1
        assert new_state.num_images == 2
        assert "2 images included" in status
        assert "1 excluded" in status

    def test_auto_cull_with_dataframe(self):
        """Gradio passes pandas DataFrame, not list — verify it works."""
        from gui.stages.import_qc import auto_cull

        state = PipelineState()
        state.image_dir = "/fake/dir"

        df = pd.DataFrame([
            ["sharp.jpg", 200.0, 6000, 4000, "24.0", "Sony", True],
            ["blurry.jpg", 50.0, 6000, 4000, "24.0", "Sony", True],
        ], columns=["Filename", "Sharpness", "Width", "Height", "Focal (mm)", "Camera", "Include"])

        updated_rows, cull_msg, new_state = auto_cull(df, 100.0, state)
        assert updated_rows[0][6] is True
        assert updated_rows[1][6] is False
        assert new_state.num_images == 1

    def test_apply_table_changes_with_dataframe(self):
        """Gradio passes pandas DataFrame — verify apply_table_changes handles it."""
        from gui.stages.import_qc import apply_table_changes

        state = PipelineState()
        state.image_dir = "/fake/dir"

        df = pd.DataFrame([
            ["a.jpg", 200.0, 6000, 4000, "24.0", "Sony", True],
            ["b.jpg", 50.0, 6000, 4000, "24.0", "Sony", False],
        ], columns=["Filename", "Sharpness", "Width", "Height", "Focal (mm)", "Camera", "Include"])

        status, new_state = apply_table_changes(df, state)
        assert len(new_state.image_list) == 1
        assert len(new_state.excluded_images) == 1
        assert "1 images included" in status


# ============================================================================
# Cache tests
# ============================================================================

class TestImageCache:
    """Tests for import_qc disk caching."""

    def test_cache_save_and_load(self, tmp_path):
        from gui.stages.import_qc import _save_cache, _load_cache, CACHE_VERSION

        cache_data = {
            "img1.jpg": {
                "mtime": 1234567890.0,
                "size": 5000000,
                "sharpness": 234.5,
                "exif": {"width": 6000, "height": 4000, "focal_length_mm": 24.0},
            }
        }
        _save_cache(tmp_path, cache_data)
        loaded = _load_cache(tmp_path)
        assert loaded["img1.jpg"]["sharpness"] == 234.5
        assert loaded["img1.jpg"]["exif"]["width"] == 6000

    def test_cache_empty_on_missing_file(self, tmp_path):
        from gui.stages.import_qc import _load_cache

        assert _load_cache(tmp_path) == {}

    def test_cache_invalidated_on_version_mismatch(self, tmp_path):
        import json
        from gui.stages.import_qc import _load_cache

        cache_path = tmp_path / "image_cache.json"
        cache_path.write_text(json.dumps({"version": -1, "images": {"a": 1}}))
        assert _load_cache(tmp_path) == {}

    def test_cache_key_uses_mtime_and_size(self, tmp_path):
        from gui.stages.import_qc import _cache_key

        test_file = tmp_path / "test.jpg"
        test_file.write_bytes(b"fake image data")
        key = _cache_key(test_file)
        assert "mtime" in key
        assert "size" in key
        assert key["size"] == len(b"fake image data")


# ============================================================================
# Browse button test
# ============================================================================

# ============================================================================
# File discovery tests
# ============================================================================

class TestDiscoverExistingOutputs:
    """Tests for PipelineState.discover_outputs()."""

    def test_discovers_feature_file_and_extractor_name(self, tmp_path):
        (tmp_path / "feats-superpoint-n16384-r99999.h5").write_bytes(b"fake")
        state = PipelineState()
        state.project_dir = str(tmp_path)
        state.discover_outputs()

        assert state.feature_path == str(tmp_path / "feats-superpoint-n16384-r99999.h5")
        assert state.extractor_name == "superpoint"

    def test_discovers_pairs_file(self, tmp_path):
        (tmp_path / "pairs-exhaustive.txt").write_bytes(b"img1.jpg img2.jpg\n")
        state = PipelineState()
        state.project_dir = str(tmp_path)
        state.discover_outputs()

        assert state.pairs_path == str(tmp_path / "pairs-exhaustive.txt")

    def test_discovers_match_file(self, tmp_path):
        # Create both feats and matches — only matches should be picked up
        (tmp_path / "feats-superpoint-n16384-r99999.h5").write_bytes(b"fake")
        (tmp_path / "feats-superpoint-n16384-r99999_matches-lightglue.h5").write_bytes(b"fake")
        state = PipelineState()
        state.project_dir = str(tmp_path)
        state.discover_outputs()

        assert state.match_path is not None
        assert "matches" in state.match_path

    def test_feats_file_not_mistaken_for_matches(self, tmp_path):
        # Only a feats file, no matches file — match_path should stay None
        (tmp_path / "feats-superpoint-n16384-r99999.h5").write_bytes(b"fake")
        state = PipelineState()
        state.project_dir = str(tmp_path)
        state.discover_outputs()

        assert state.match_path is None

    def test_discovers_sfm_directory(self, tmp_path):
        sfm_dir = tmp_path / "sfm"
        sfm_dir.mkdir()
        (sfm_dir / "cameras.bin").write_bytes(b"fake")
        (sfm_dir / "images.bin").write_bytes(b"fake")
        (sfm_dir / "points3D.bin").write_bytes(b"fake")

        state = PipelineState()
        state.project_dir = str(tmp_path)
        state.discover_outputs()

        assert state.sfm_dir == str(sfm_dir)

    def test_sfm_dir_ignored_if_no_bin_files(self, tmp_path):
        sfm_dir = tmp_path / "sfm"
        sfm_dir.mkdir()
        # Empty sfm dir — should not be discovered

        state = PipelineState()
        state.project_dir = str(tmp_path)
        state.discover_outputs()

        assert state.sfm_dir is None

    def test_does_not_overwrite_existing_fields(self, tmp_path):
        (tmp_path / "feats-superpoint-n16384-r99999.h5").write_bytes(b"fake")
        (tmp_path / "pairs-exhaustive.txt").write_bytes(b"fake")

        state = PipelineState()
        state.project_dir = str(tmp_path)
        state.feature_path = "/existing/features.h5"
        state.extractor_name = "disk"
        state.pairs_path = "/existing/pairs.txt"

        state.discover_outputs()

        assert state.feature_path == "/existing/features.h5"
        assert state.extractor_name == "disk"
        assert state.pairs_path == "/existing/pairs.txt"

    def test_picks_last_sorted_when_multiple_files(self, tmp_path):
        (tmp_path / "feats-aliked-n4096-r1600.h5").write_bytes(b"fake")
        (tmp_path / "feats-superpoint-n16384-r99999.h5").write_bytes(b"fake")

        state = PipelineState()
        state.project_dir = str(tmp_path)
        state.discover_outputs()

        # sorted() puts superpoint after aliked alphabetically
        assert "superpoint" in state.feature_path
        assert state.extractor_name == "superpoint"

    def test_no_project_dir_changes_nothing(self):
        state = PipelineState()
        state.discover_outputs()

        assert state.feature_path is None
        assert state.pairs_path is None
        assert state.match_path is None
        assert state.sfm_dir is None

    def test_empty_project_dir_changes_nothing(self, tmp_path):
        state = PipelineState()
        state.project_dir = str(tmp_path)
        state.discover_outputs()

        assert state.feature_path is None
        assert state.pairs_path is None
        assert state.match_path is None
        assert state.sfm_dir is None


class TestBrowseButton:
    """Verify browse button is wired in UI."""

    def test_browse_btn_in_source(self):
        source = SRC_IMPORT_QC.read_text()
        assert "browse_btn" in source
        assert "_browse_folder" in source
        assert "filedialog" in source


# ============================================================================
# Import smoke tests (all modules importable)
# ============================================================================

class TestImports:
    """Verify all project modules can be imported without error."""

    def test_import_pipeline_state(self):
        from core.pipeline_state import PipelineState
        assert PipelineState is not None

    def test_import_utils(self):
        from core.utils import (
            get_image_list,
            compute_sharpness,
            read_exif,
            validate_exif_consistency,
            create_project_dir,
            create_3dgs_directory,
        )

    def test_import_feature_configs(self):
        from core.feature_configs import build_extraction_config

    def test_import_matcher_configs(self):
        from core.matcher_configs import build_matching_config

    def test_import_import_qc(self):
        from gui.stages.import_qc import (
            build_stage1_tab, wire_stage1, load_images, auto_cull, apply_table_changes,
        )

    def test_import_extraction(self):
        from gui.stages.extraction import build_stage2_tab, wire_stage2, run_extraction

    def test_import_matching(self):
        from gui.stages.matching import build_stage3_tab, wire_stage3, run_matching

    @pytest.mark.skipif(not HAS_PYCOLMAP, reason="pycolmap not installed")
    def test_import_reconstruction(self):
        from gui.stages.reconstruction import (
            build_stage45_tab, wire_stage45, run_reconstruction,
        )

    @pytest.mark.skipif(not HAS_PYCOLMAP, reason="pycolmap not installed")
    def test_import_export(self):
        from gui.stages.export import (
            build_stage67_tab, wire_stage67, run_refinement, run_export,
        )

    @pytest.mark.skipif(not HAS_PYCOLMAP, reason="pycolmap not installed")
    def test_import_app(self):
        from gui.app import create_app, launch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
