"""Tests for core.ingest_360 — WP-O1 360 ingest (layout probe, sampling, masks, CLI gate).

Pure-function coverage plus a real ffmpeg round-trip on a tiny synthetic clip (skipped when
ffmpeg is not on PATH). No SAM 2.1 / GPU is exercised here.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.ingest_360 import (  # noqa: E402
    EXCLUDED,
    VALID,
    SourceInfo,
    box_mask,
    build_filter_graph,
    build_parser,
    geometric_mask,
    main,
    operator_region,
    probe_source,
    window_size,
    _parse_points,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
ffmpeg_only = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")


def _info(layout="dual_fisheye", w=1920, h=1920):
    return SourceInfo(path="x", layout=layout, width=w, height=h, fps=30.0,
                      fps_frac="30/1", nb_frames=100, duration=3.3)


# ============================================================================
# CLI gate — default-OFF
# ============================================================================

class TestDefaultOff:
    def test_capture_type_defaults_to_off(self):
        assert build_parser().parse_args([]).capture_type == "off"

    def test_main_is_a_noop_without_the_flag(self, capsys):
        assert main([]) == 2
        assert "OFF (default)" in capsys.readouterr().out

    def test_main_requires_input_when_enabled(self):
        assert main(["--capture-type", "insta360"]) == 2

    def test_missing_file_is_a_clean_error_not_a_traceback(self, capsys, tmp_path):
        rc = main(["--capture-type", "insta360", "--input", str(tmp_path / "nope.insv"),
                   "--out_dir", str(tmp_path / "out")])
        assert rc == 1 and "ERROR" in capsys.readouterr().out
        assert not (tmp_path / "out").exists()      # no debris on a bad input


# ============================================================================
# Sampling — window-relative, never absolute
# ============================================================================

class TestSampling:
    def test_window_size_from_target_fps(self):
        assert window_size(29.97, 2.0) == 15
        assert window_size(29.97, 1.0) == 30
        assert window_size(29.97, 60.0) == 1        # never 0

    def test_selection_is_window_relative_not_absolute(self):
        """A globally DARK/low-texture stretch must still contribute frames
        (DEVELOPER_NOTES §5: absolute sharpness culls are invalid for video)."""
        from core.frame_select import select_indices
        sharp = np.concatenate([np.full(30, 1000.0), np.full(30, 10.0)])
        sharp[5] = 1500.0
        sharp[40] = 20.0                            # best of the low-texture stretch
        idx = select_indices(sharp, window=15, min_gap=1, keep_frac=1.0)
        assert len(idx) == 4                        # one per window, low window included
        assert 5 in idx and 40 in idx


# ============================================================================
# Masks — 255 VALID / 0 EXCLUDED, angular bands
# ============================================================================

class TestGeometricMask:
    def test_nadir_band_height_matches_degrees(self):
        m = geometric_mask(360, 180, nadir_deg=12.0, zenith_deg=0.0, seam_deg=0.0)
        assert m.dtype == np.uint8
        assert m[0, 0] == VALID and m[-1, 0] == EXCLUDED
        assert (m == EXCLUDED).sum() == pytest.approx(360 * 12, rel=0.02)

    def test_zenith_band(self):
        m = geometric_mask(360, 180, nadir_deg=0.0, zenith_deg=10.0, seam_deg=0.0)
        assert m[0, 0] == EXCLUDED and m[-1, 0] == VALID

    def test_seam_bands_sit_at_plus_minus_90_deg_yaw(self):
        w = 3600
        m = geometric_mask(w, 180, nadir_deg=0.0, zenith_deg=0.0, seam_deg=4.0)
        assert m[90, w // 4] == EXCLUDED and m[90, 3 * w // 4] == EXCLUDED
        assert m[90, w // 2] == VALID                # ERP centre stays valid
        assert m[90, 0] == VALID                     # +-180 (front lens centre) stays valid
        # two bands of 2*4 deg each
        assert (m[90] == EXCLUDED).sum() == pytest.approx(2 * 2 * 4 * w / 360, rel=0.02)

    def test_no_bands_means_all_valid(self):
        m = geometric_mask(64, 32, nadir_deg=0.0, zenith_deg=0.0, seam_deg=0.0)
        assert (m == VALID).all()


class TestOperatorGeometry:
    def test_region_is_the_rear_hemisphere_below_the_pitch_cap(self):
        reg = operator_region(360, 180, max_yaw_deg=90.0, max_pitch_deg=25.0)
        assert reg[90, 180]            # ERP centre, equator
        assert not reg[90, 5]          # front hemisphere
        assert not reg[10, 180]        # high in the ceiling
        assert reg[170, 180]           # near nadir

    def test_box_mask_is_normalised(self):
        b = box_mask(100, 100, (0.4, 0.6, 0.7, 1.0))
        assert b.sum() == 20 * 30
        assert b[80, 50] and not b[50, 50]

    def test_parse_points(self):
        p = _parse_points("0.1,0.2;0.3,0.4")
        assert p.shape == (2, 2) and p[1, 1] == pytest.approx(0.4)
        assert _parse_points("").shape == (0, 2)


# ============================================================================
# Filter graph
# ============================================================================

class TestFilterGraph:
    def test_dual_fisheye_graph_stitches_then_selects_before_v360(self):
        g = build_filter_graph(_info(), [3, 7], 3840, 1920, 200.0, 200.0, 0.0)
        assert "hstack=inputs=2" in g
        assert g.index("select=") < g.index("v360")      # remap only on kept frames
        assert "eq(n,3)+eq(n,7)" in g
        assert "dfisheye:e:ih_fov=200.0:iv_fov=200.0" in g
        assert "w=3840:h=1920" in g

    def test_erp_graph_skips_the_stitch(self):
        info = _info("erp", 3840, 1920)
        g = build_filter_graph(info, [1], 3840, 1920, 200.0, 200.0, 0.0)
        assert "hstack" not in g and "v360" not in g and "null[out]" in g

    def test_erp_graph_scales_when_asked(self):
        g = build_filter_graph(_info("erp", 3840, 1920), [1], 1920, 960, 200.0, 200.0, 0.0)
        assert "scale=1920:960" in g


class TestErpSize:
    def test_dual_fisheye_defaults_to_two_by_input_height(self):
        assert _info(w=1920, h=1920).erp_size == (3840, 1920)
        assert _info(w=3840, h=3840).erp_size == (7680, 3840)   # resolution-agnostic

    def test_erp_keeps_native_size(self):
        assert _info("erp", 7680, 3840).erp_size == (7680, 3840)


# ============================================================================
# ffmpeg round-trip (layout auto-detection + full ingest on a tiny synthetic ERP)
# ============================================================================

@ffmpeg_only
class TestRoundTrip:
    @staticmethod
    def _make(path, size="320x160", dur=1):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", f"testsrc=size={size}:rate=30:duration={dur}",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)

    def test_two_to_one_stream_detects_as_erp(self, tmp_path):
        v = tmp_path / "erp.mp4"
        self._make(v, "320x160")
        assert probe_source(str(v)).layout == "erp"

    def test_non_360_aspect_is_rejected(self, tmp_path):
        v = tmp_path / "bad.mp4"
        self._make(v, "320x180")
        with pytest.raises(ValueError, match="unsupported 360 layout"):
            probe_source(str(v))

    def test_force_layout_overrides(self, tmp_path):
        v = tmp_path / "bad.mp4"
        self._make(v, "320x180")
        assert probe_source(str(v), force_layout="erp").layout == "erp"

    def test_full_erp_ingest_writes_the_expected_layout(self, tmp_path):
        v = tmp_path / "erp.mp4"
        self._make(v, "320x160", dur=1)
        out = tmp_path / "out"
        rc = main(["--capture-type", "insta360", "--input", str(v), "--out_dir", str(out),
                   "--target_fps", "5", "--contact_sheet", "2", "--contact_cols", "2",
                   "--progress", "0"])
        assert rc == 0
        imgs = sorted((out / "images").glob("*.jpg"))
        masks = sorted((out / "masks").glob("*.png"))
        assert len(imgs) == len(masks) == 5
        assert [p.stem for p in imgs] == [p.stem for p in masks]   # <stem>.png convention
        rep = json.loads((out / "ingest_report.json").read_text())
        assert rep["source"]["layout"] == "erp"
        assert rep["sampling"]["absolute_threshold"] is None       # never used
        assert rep["masks"]["seam_deg"] == 0.0                     # auto-off for ERP input
        assert rep["masks"]["operator_mode"] == "none"             # auto-off for ERP input
        assert len(rep["frames"]) == 5
        assert (out / "qa" / "contact_sheet.jpg").is_file()

        import cv2
        m = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
        assert set(np.unique(m)).issubset({EXCLUDED, VALID})
        assert m[0, 0] == VALID and m[-1, 0] == EXCLUDED           # nadir band present
