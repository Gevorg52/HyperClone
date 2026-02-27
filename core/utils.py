"""Utility functions for the SfM pipeline: image listing, EXIF, sharpness, project dirs."""

import shutil
from pathlib import Path
from typing import Optional

import cv2
import exifread


# Supported image extensions (lowercase, with dot)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def get_image_list(folder: str) -> list[Path]:
    """Return sorted list of image paths in *folder* with supported extensions."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    images = [
        p for p in sorted(folder.iterdir())
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()
    ]
    return images


def compute_sharpness(image_path: str | Path) -> float:
    """Compute sharpness score using tile-based Laplacian variance (90th percentile).

    Uses an 8x8 grid of tiles and reports the 90th percentile of their
    Laplacian variances. Robust to depth-of-field (ignores bokeh tiles),
    sensitive to motion blur (all tiles score low).
    Higher = sharper. Typical blurry threshold is ~200.
    """
    import numpy as np

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    tile_grid = 8
    th, tw = img.shape[0] // tile_grid, img.shape[1] // tile_grid
    variances = []
    for r in range(tile_grid):
        for c in range(tile_grid):
            tile = img[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            variances.append(float(cv2.Laplacian(tile, cv2.CV_64F).var()))
    return float(np.percentile(variances, 90))


def read_exif(image_path: str | Path) -> dict:
    """Read EXIF tags from an image file. Returns dict with standardized keys."""
    image_path = Path(image_path)
    result = {
        "filename": image_path.name,
        "width": None,
        "height": None,
        "focal_length_mm": None,
        "camera_make": None,
        "camera_model": None,
        "iso": None,
        "exposure_time": None,
    }
    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)

        # Image dimensions — try EXIF first, fall back to actual pixel read
        if "EXIF ExifImageWidth" in tags:
            result["width"] = int(str(tags["EXIF ExifImageWidth"]))
        if "EXIF ExifImageLength" in tags:
            result["height"] = int(str(tags["EXIF ExifImageLength"]))

        # Fall back to reading actual image dimensions if EXIF didn't have them
        if result["width"] is None or result["height"] is None:
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                result["height"], result["width"] = img.shape[:2]

        # Focal length
        if "EXIF FocalLength" in tags:
            fl = tags["EXIF FocalLength"]
            val = fl.values[0]
            result["focal_length_mm"] = float(val.num) / float(val.den) if val.den else None

        # Camera info
        if "Image Make" in tags:
            result["camera_make"] = str(tags["Image Make"]).strip()
        if "Image Model" in tags:
            result["camera_model"] = str(tags["Image Model"]).strip()

        # ISO
        if "EXIF ISOSpeedRatings" in tags:
            result["iso"] = int(str(tags["EXIF ISOSpeedRatings"]))

        # Exposure time
        if "EXIF ExposureTime" in tags:
            result["exposure_time"] = str(tags["EXIF ExposureTime"])

    except Exception:
        # If EXIF reading fails entirely, still try to get dimensions
        if result["width"] is None or result["height"] is None:
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                result["height"], result["width"] = img.shape[:2]

    return result


def validate_exif_consistency(exif_list: list[dict]) -> list[str]:
    """Check that all images share the same resolution and focal length.

    Returns list of warning strings (empty = all consistent).
    """
    warnings = []
    if not exif_list:
        return warnings

    # Check resolution consistency
    resolutions = set()
    for e in exif_list:
        if e["width"] and e["height"]:
            resolutions.add((e["width"], e["height"]))
    if len(resolutions) > 1:
        warnings.append(
            f"Mixed resolutions detected: {resolutions}. "
            "This may cause issues with single-camera reconstruction."
        )

    # Check focal length consistency
    focal_lengths = set()
    for e in exif_list:
        if e["focal_length_mm"] is not None:
            focal_lengths.add(e["focal_length_mm"])
    if len(focal_lengths) > 1:
        warnings.append(
            f"Mixed focal lengths detected: {focal_lengths} mm. "
            "Consider using per-image camera models instead of single camera."
        )
    if not focal_lengths:
        warnings.append("No focal length found in EXIF data.")

    return warnings


def create_project_dir(base_dir: str, image_dir_name: str) -> Path:
    """Create a project working directory under base_dir.

    Structure: base_dir/apex_<image_dir_name>/
    Returns the project directory path.
    """
    base = Path(base_dir)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in image_dir_name)
    project_dir = base / f"apex_{safe_name}"
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def get_reconstruction_stats(sfm_dir: str) -> dict:
    """Load a COLMAP reconstruction and return summary statistics.

    Returns dict with num_cameras, num_images, num_points3d, mean_reprojection_error.
    """
    import pycolmap

    sfm_path = Path(sfm_dir)
    rec = pycolmap.Reconstruction(str(sfm_path))
    errors = [
        p.error for p in rec.points3D.values()
        if p.error > 0
    ]
    mean_error = sum(errors) / len(errors) if errors else 0.0

    return {
        "num_cameras": len(rec.cameras),
        "num_images": len(rec.images),
        "num_points3d": len(rec.points3D),
        "mean_reprojection_error": round(mean_error, 4),
    }


def create_3dgs_directory(sfm_dir: str, image_dir: str, export_dir: str) -> Path:
    """Create standard Gaussian Splatting input directory layout.

    Layout:
        export_dir/
        ├── images/         (symlinked or copied source images)
        └── sparse/
            └── 0/
                ├── cameras.bin
                ├── images.bin
                └── points3D.bin

    Returns the export_dir path.
    """
    sfm_path = Path(sfm_dir)
    img_path = Path(image_dir)
    out_path = Path(export_dir)

    # Create directory structure
    sparse_dir = out_path / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_out = out_path / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    # Copy COLMAP binary files
    for fname in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = sfm_path / fname
        if src.exists():
            shutil.copy2(str(src), str(sparse_dir / fname))

    # Copy images (use copies, not symlinks, for Windows compatibility)
    for src_img in img_path.iterdir():
        if src_img.suffix.lower() in SUPPORTED_EXTENSIONS and src_img.is_file():
            dst_img = images_out / src_img.name
            if not dst_img.exists():
                shutil.copy2(str(src_img), str(dst_img))

    return out_path
