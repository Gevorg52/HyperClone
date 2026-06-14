"""Gravity alignment and centering for SfM reconstructions.

Aligns reconstructions to gravity before export so models appear upright
in 3DGS/NeRF viewers. Device-agnostic: uses IMU data when available
(sidecar JSON or DJI XMP), falls back to SVD camera-pose estimation.

Applied after bundle adjustment, before writing export files.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _cam_from_world(image):
    """Return the image's cam_from_world Rigid3d.

    pycolmap >= 4.0 exposes ``cam_from_world`` as a method; 3.x as a property.
    Call it only when it's not already a Rigid3d-like object (has ``.rotation``)
    — this handles both pycolmap versions and test mocks.
    """
    cfw = image.cam_from_world
    if callable(cfw) and not hasattr(cfw, "rotation"):
        cfw = cfw()
    return cfw


# ============================================================================
# 2a. Sidecar JSON reader
# ============================================================================

def read_sidecar_gravity(
    image_stem: str,
    image_dir: Path,
    project_dir: Path,
) -> Optional[np.ndarray]:
    """Read gravity vector from sidecar JSON file.

    Searches for <image_stem>.json in:
      1. image_dir (same folder as images)
      2. image_dir/imu/
      3. project_dir/imu/

    Returns normalized gravity-down unit vector (3,) or None if not found.
    """
    search_dirs = [
        image_dir,
        image_dir / "imu",
        project_dir / "imu",
    ]

    for d in search_dirs:
        sidecar_path = d / f"{image_stem}.json"
        if not sidecar_path.exists():
            continue

        try:
            data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.debug("Invalid sidecar JSON: %s", sidecar_path)
            return None

        gravity = data.get("gravity")
        if gravity is None or len(gravity) != 3:
            logger.debug("Missing or invalid gravity field in: %s", sidecar_path)
            return None

        g = np.array(gravity, dtype=np.float64)
        norm = np.linalg.norm(g)
        if norm < 1e-8:
            logger.debug("Zero gravity vector in: %s", sidecar_path)
            return None

        return g / norm

    return None


# ============================================================================
# 2b. DJI XMP fallback
# ============================================================================

def _read_dji_gravity(image_path: Path) -> Optional[np.ndarray]:
    """Best-effort parse of DJI gimbal pitch/roll from XMP metadata.

    Converts gimbal angles to an approximate gravity vector.
    Returns normalized gravity-down vector or None on failure.
    """
    try:
        import exifread
    except ImportError:
        return None

    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except OSError:
        return None

    # DJI stores gimbal data in XMP; exifread may expose it as string tags.
    # Also try reading raw XMP from the file bytes.
    pitch_deg = None
    roll_deg = None

    # Try exifread tags first
    for key, val in tags.items():
        key_lower = key.lower()
        if "gimbalpitch" in key_lower:
            try:
                pitch_deg = float(str(val))
            except (ValueError, TypeError):
                pass
        elif "gimbalroll" in key_lower:
            try:
                roll_deg = float(str(val))
            except (ValueError, TypeError):
                pass

    # If exifread didn't find them, try raw XMP parsing
    if pitch_deg is None:
        pitch_deg, roll_deg = _parse_xmp_gimbal(image_path)

    if pitch_deg is None:
        return None

    if roll_deg is None:
        roll_deg = 0.0

    # Convert gimbal pitch/roll to gravity vector.
    # DJI convention: pitch=0 means camera level, pitch=-90 means nadir.
    # Gravity points down in world frame.
    pitch_rad = np.radians(pitch_deg)
    roll_rad = np.radians(roll_deg)

    # Approximate gravity in camera frame from gimbal orientation
    gx = -np.sin(roll_rad)
    gy = np.sin(pitch_rad) * np.cos(roll_rad)
    gz = -np.cos(pitch_rad) * np.cos(roll_rad)

    g = np.array([gx, gy, gz], dtype=np.float64)
    norm = np.linalg.norm(g)
    if norm < 1e-8:
        return None

    return g / norm


def _parse_xmp_gimbal(image_path: Path) -> tuple[Optional[float], Optional[float]]:
    """Extract GimbalPitchDegree and GimbalRollDegree from raw XMP in JPEG."""
    pitch = None
    roll = None

    try:
        raw = image_path.read_bytes()
    except OSError:
        return None, None

    # Find XMP packet (bounded by <?xpacket begin> ... <?xpacket end>)
    import re

    # Search for gimbal pitch
    match = re.search(rb'GimbalPitchDegree["\s>=]+([+-]?\d+\.?\d*)', raw)
    if match:
        try:
            pitch = float(match.group(1))
        except ValueError:
            pass

    match = re.search(rb'GimbalRollDegree["\s>=]+([+-]?\d+\.?\d*)', raw)
    if match:
        try:
            roll = float(match.group(1))
        except ValueError:
            pass

    return pitch, roll


# ============================================================================
# 2c. Gravity collector
# ============================================================================

def collect_gravity_vectors(rec, image_dir: Path, project_dir: Path) -> dict:
    """Collect gravity vectors for registered images.

    Tries sidecar JSON first, then DJI XMP for each image.
    Returns {image_id: gravity_vector} for images where gravity was found.
    """
    gravity_vectors = {}

    for image_id, image in rec.images.items():
        stem = Path(image.name).stem
        image_path = image_dir / image.name

        # Try sidecar JSON first
        g = read_sidecar_gravity(stem, image_dir, project_dir)

        # Fall back to DJI XMP
        if g is None and image_path.exists():
            g = _read_dji_gravity(image_path)

        if g is not None:
            gravity_vectors[image_id] = g

    return gravity_vectors


# ============================================================================
# 2d. IMU-based alignment
# ============================================================================

def estimate_gravity_from_imu(rec, gravity_vectors: dict) -> np.ndarray:
    """Estimate gravity-down direction in world frame from IMU data.

    For each image with a device-frame gravity vector, transform it to
    world frame using the camera's rotation: g_world = R^T @ g_camera.
    Average all world-frame vectors and normalize.

    Returns gravity-down unit vector in world frame (3,).
    """
    world_gravities = []

    for image_id, g_device in gravity_vectors.items():
        if image_id not in rec.images:
            continue

        image = rec.images[image_id]
        # cam_from_world rotation: R maps world → camera
        R = _cam_from_world(image).rotation.matrix()
        # Transform device gravity to world frame: g_world = R^T @ g_device
        g_world = R.T @ g_device
        world_gravities.append(g_world)

    if not world_gravities:
        raise ValueError("No valid IMU gravity vectors to average")

    avg = np.mean(world_gravities, axis=0)
    norm = np.linalg.norm(avg)
    if norm < 1e-8:
        raise ValueError("Averaged IMU gravity vectors cancel out")

    return avg / norm


# ============================================================================
# 2e. SVD-based estimation (fallback — the common path)
# ============================================================================

def estimate_gravity_from_cameras(rec) -> np.ndarray:
    """Estimate gravity-up direction from camera poses via SVD.

    Algorithm (nerfstudio "vertical" method):
    1. Extract camera X-axes in world frame for all registered images
    2. SVD on the stacked X-axes matrix
    3. Smallest singular vector = direction perpendicular to horizontal plane = gravity
    4. Degeneracy check: if cameras are nearly co-directional, fall back to Y-axis average
    5. Sign disambiguation: ensure result aligns with average camera Y-axis

    Returns gravity-up unit vector in world frame (3,).
    """
    if len(rec.images) == 0:
        logger.warning("No registered images — returning default Y-up")
        return np.array([0.0, 1.0, 0.0])

    x_axes = []
    y_axes = []

    for image in rec.images.values():
        R = _cam_from_world(image).rotation.matrix()
        # Camera X-axis in world frame: first row of cam_from_world rotation
        x_axes.append(R[0, :])
        # Camera Y-axis in world frame: second row
        y_axes.append(R[1, :])

    x_axes = np.array(x_axes)  # (N, 3)
    y_axes = np.array(y_axes)  # (N, 3)
    N = len(x_axes)

    if N == 1:
        # Single image: can't do SVD meaningfully, use camera Y-axis
        logger.warning("Single image — using camera Y-axis as up estimate")
        y = y_axes[0]
        norm = np.linalg.norm(y)
        return y / norm if norm > 1e-8 else np.array([0.0, 1.0, 0.0])

    # SVD: smallest singular vector of X-axes matrix
    _, S, Vt = np.linalg.svd(x_axes, full_matrices=False)

    # Degeneracy check: if S[1] < 0.17 * sqrt(N), cameras are nearly co-directional
    if S[1] < 0.17 * np.sqrt(N):
        logger.warning(
            "Degenerate camera configuration (S[1]=%.3f < %.3f). "
            "Falling back to Y-axis average.",
            S[1], 0.17 * np.sqrt(N),
        )
        avg_y = np.mean(y_axes, axis=0)
        norm = np.linalg.norm(avg_y)
        return avg_y / norm if norm > 1e-8 else np.array([0.0, 1.0, 0.0])

    # Smallest singular vector = last row of Vt
    up = Vt[-1, :]

    # Sign disambiguation: ensure result aligns with average camera Y-axis
    # (most captures have cameras above subject, so camera Y ≈ world up)
    avg_y = np.mean(y_axes, axis=0)
    if np.dot(up, avg_y) < 0:
        up = -up

    norm = np.linalg.norm(up)
    return up / norm if norm > 1e-8 else np.array([0.0, 1.0, 0.0])


# ============================================================================
# 2f. Rotation utility
# ============================================================================

def _rotation_between_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Compute 3x3 rotation matrix that rotates v_from to v_to.

    Uses Rodrigues' formula. Handles identity (parallel) and
    180-degree (anti-parallel) cases.

    Both inputs should be unit vectors.
    """
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)

    dot = np.clip(np.dot(v_from, v_to), -1.0, 1.0)

    # Identity case: vectors are already aligned
    if dot > 1.0 - 1e-8:
        return np.eye(3)

    # Anti-parallel case: 180-degree rotation
    if dot < -1.0 + 1e-8:
        # Find an orthogonal vector to rotate around
        if abs(v_from[0]) < 0.9:
            ortho = np.array([1.0, 0.0, 0.0])
        else:
            ortho = np.array([0.0, 1.0, 0.0])
        axis = np.cross(v_from, ortho)
        axis = axis / np.linalg.norm(axis)
        # 180-degree rotation around axis: R = 2 * outer(axis, axis) - I
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    # General case: Rodrigues' formula
    cross = np.cross(v_from, v_to)
    skew = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0],
    ])
    R = np.eye(3) + skew + skew @ skew / (1.0 + dot)
    return R


# ============================================================================
# 2g. Main entry point
# ============================================================================

def align_reconstruction(
    rec,
    image_dir: str,
    project_dir: str,
    up_axis: str = "y",
    center: bool = True,
    use_imu: bool = True,
) -> dict:
    """Align reconstruction to gravity and optionally center at origin.

    Modifies rec in-place via pycolmap.Sim3d transform.

    Args:
        rec: pycolmap.Reconstruction (loaded, in-place modification)
        image_dir: Path to source images
        project_dir: Path to project directory
        up_axis: Target up direction — "y" for 3DGS/NeRF, "z" for CAD/GIS
        center: Whether to translate camera centroid to origin
        use_imu: Whether to prefer IMU gravity data when available

    Returns:
        Metadata dict with keys: method, imu_coverage, rotation_angle_deg,
        up_axis, centered.
    """
    import pycolmap

    image_dir = Path(image_dir)
    project_dir = Path(project_dir)

    method = "estimated"
    imu_coverage = 0.0
    estimated_up = None

    # Try IMU path first
    if use_imu:
        gravity_vectors = collect_gravity_vectors(rec, image_dir, project_dir)
        num_with_imu = len(gravity_vectors)
        num_registered = len(rec.images)
        imu_coverage = num_with_imu / num_registered if num_registered > 0 else 0.0

        if num_with_imu >= 3:
            try:
                gravity_down = estimate_gravity_from_imu(rec, gravity_vectors)
                estimated_up = -gravity_down  # gravity-down → up
                method = "imu"
                logger.info(
                    "IMU alignment: %d/%d images (%.0f%% coverage)",
                    num_with_imu, num_registered, imu_coverage * 100,
                )
            except ValueError as e:
                logger.warning("IMU alignment failed: %s. Falling back to SVD.", e)

    # SVD fallback (or if IMU wasn't attempted)
    if estimated_up is None:
        estimated_up = estimate_gravity_from_cameras(rec)
        method = "estimated"
        imu_coverage = 0.0
        logger.info("SVD alignment: estimated up = %s", estimated_up)

    # Target up vector
    if up_axis == "z":
        target_up = np.array([0.0, 0.0, 1.0])
    else:
        target_up = np.array([0.0, 1.0, 0.0])

    # Compute rotation from estimated up to target up
    rotation_matrix = _rotation_between_vectors(estimated_up, target_up)
    rotation_angle_deg = float(np.degrees(
        np.arccos(np.clip(np.dot(estimated_up, target_up), -1.0, 1.0))
    ))

    # Compute translation for centering
    translation = np.zeros(3)
    centered = False
    if center and len(rec.images) > 0:
        # Camera centers: c = -R^T @ t
        centers = []
        for image in rec.images.values():
            cfw = _cam_from_world(image)
            R = cfw.rotation.matrix()
            t = cfw.translation
            cam_center = -R.T @ t
            centers.append(cam_center)
        centroid = np.mean(centers, axis=0)
        # After rotation, centroid should map to origin:
        # new_pos = R @ old_pos + t => want R @ centroid + t = 0 => t = -R @ centroid
        translation = -rotation_matrix @ centroid
        centered = True

    # Apply Sim3d transform (scale=1.0, rotation, translation)
    # pycolmap.Sim3d expects a Rotation3d for the rotation parameter
    transform = pycolmap.Sim3d(
        1.0,
        pycolmap.Rotation3d(rotation_matrix),
        translation,
    )
    rec.transform(transform)

    return {
        "method": method,
        "imu_coverage": imu_coverage,
        "rotation_angle_deg": rotation_angle_deg,
        "up_axis": up_axis,
        "centered": centered,
    }
