"""Module 1 of object coverage-completion: analyze a capture's angular coverage on
the viewing hemisphere, detect gaps, and emit (a) target novel poses to GENERATE and
(b) a re-capture report ("shoot these angles").

Framework (research-backed, June 2026): a complete object needs the UPPER HEMISPHERE +
EQUATOR (the bottom doesn't matter for a surface-resting object), sampled with a
Fibonacci/golden-spiral sphere (a raw az/el grid over-samples the poles). Coverage =
for each target direction, is there a real camera within `cover_thresh_deg`? Empty
target directions are gaps -> place fill poses at them (orbit rings filling the missing
azimuth/elevation), oriented at the object centre, at the median capture radius.

No deps beyond numpy/cv2. analyze_coverage() returns a CoverageResult; coverage_report()
is human-readable; render_coverage_map() draws the azimuth x elevation map.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import cv2


@dataclass
class CoverageResult:
    coverage_pct: float            # fraction of hemisphere target bins that are covered
    bin_dirs: np.ndarray           # (B,3) target directions on the hemisphere
    bin_covered: np.ndarray        # (B,) bool
    bin_az: np.ndarray             # (B,) azimuth deg [0,360)
    bin_el: np.ndarray             # (B,) elevation deg [0,90]
    cam_az: np.ndarray             # (N,) captured-camera azimuths
    cam_el: np.ndarray             # (N,) captured-camera elevations
    object_center: np.ndarray
    up_axis: np.ndarray
    ref: np.ndarray
    perp: np.ndarray
    radius: float
    gap_dirs: np.ndarray           # (G,3) uncovered directions


def _basis(up):
    up = up / (np.linalg.norm(up) + 1e-9)
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    ref = a - a.dot(up) * up; ref /= np.linalg.norm(ref) + 1e-9
    perp = np.cross(up, ref)
    return up, ref, perp


def _fib_hemisphere(n, min_elev_rad):
    """Golden-spiral directions on the upper cap (canonical z=up)."""
    ga = math.pi * (3 - 5 ** 0.5)
    zmin = math.sin(min_elev_rad)
    z = zmin + (1 - zmin) * (np.arange(n) + 0.5) / n
    r = np.sqrt(np.clip(1 - z * z, 0, 1))
    phi = np.arange(n) * ga
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def object_center_from_cameras(cam_centers, forwards):
    """Least-squares point closest to all camera viewing rays (the orbited object).
    cam_centers (N,3), forwards (N,3) = camera +z (OpenCV). Robust for object scans."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for C, d in zip(np.asarray(cam_centers, float), np.asarray(forwards, float)):
        d = d / (np.linalg.norm(d) + 1e-9)
        P = np.eye(3) - np.outer(d, d); A += P; b += P @ C
    return np.linalg.solve(A + 1e-6 * np.eye(3), b)


def estimate_up(cam_centers, object_center):
    """Orbit-plane normal (smallest-variance PCA dir), signed toward the cameras."""
    c0 = cam_centers - cam_centers.mean(0)
    _, _, V = np.linalg.svd(c0, full_matrices=False)
    up = V[2]
    if (cam_centers - object_center).dot(up).mean() < 0:
        up = -up
    return up


def analyze_coverage(cam_centers: np.ndarray, object_center: np.ndarray,
                     up_axis=None, n_bins: int = 240, cover_thresh_deg: float = 22.0,
                     min_elev_deg: float = -8.0) -> CoverageResult:
    O = np.asarray(object_center, float)
    cam_centers = np.asarray(cam_centers, float)
    up = estimate_up(cam_centers, O) if up_axis is None else np.asarray(up_axis, float)
    up, ref, perp = _basis(up)

    cam_dirs = cam_centers - O
    radius = float(np.linalg.norm(cam_dirs, axis=1).mean())
    cam_dirs = cam_dirs / (np.linalg.norm(cam_dirs, axis=1, keepdims=True) + 1e-9)

    bins_canon = _fib_hemisphere(n_bins, math.radians(min_elev_deg))
    bin_dirs = bins_canon[:, 0:1] * ref + bins_canon[:, 1:2] * perp + bins_canon[:, 2:3] * up

    cos_t = math.cos(math.radians(cover_thresh_deg))
    sim = bin_dirs @ cam_dirs.T                                  # (B,N)
    bin_covered = sim.max(1) > cos_t

    def az_el(d):
        el = np.degrees(np.arcsin(np.clip(d.dot(up), -1, 1)))
        az = np.degrees(np.arctan2(d.dot(perp), d.dot(ref))) % 360
        return az, el

    bin_az, bin_el = az_el(bin_dirs)
    cam_az, cam_el = az_el(cam_dirs)
    return CoverageResult(
        coverage_pct=float(bin_covered.mean()), bin_dirs=bin_dirs, bin_covered=bin_covered,
        bin_az=bin_az, bin_el=bin_el, cam_az=cam_az, cam_el=cam_el,
        object_center=O, up_axis=up, ref=ref, perp=perp, radius=radius,
        gap_dirs=bin_dirs[~bin_covered])


def target_poses(res: CoverageResult, max_poses: int = 60) -> np.ndarray:
    """c2w (M,4,4) looking at the object from each gap direction (OpenCV convention)."""
    gaps = res.gap_dirs
    if len(gaps) > max_poses:
        gaps = gaps[np.linspace(0, len(gaps) - 1, max_poses).astype(int)]
    O, up = res.object_center, res.up_axis
    out = []
    for d in gaps:
        eye = O + res.radius * d
        f = O - eye; f /= np.linalg.norm(f) + 1e-9
        r = np.cross(f, up); r /= np.linalg.norm(r) + 1e-9
        u2 = np.cross(r, f)
        c = np.eye(4); c[:3, 0] = r; c[:3, 1] = -u2; c[:3, 2] = f; c[:3, 3] = eye
        out.append(c)
    return np.stack(out) if out else np.zeros((0, 4, 4))


def coverage_report(res: CoverageResult) -> str:
    lines = [f"Angular coverage: {100*res.coverage_pct:.0f}% of the upper hemisphere "
             f"({int(res.bin_covered.sum())}/{len(res.bin_covered)} target views)."]
    # captured elevation span
    lines.append(f"Captured elevations: {res.cam_el.min():.0f}deg..{res.cam_el.max():.0f}deg "
                 f"(median {np.median(res.cam_el):.0f}deg); azimuth span "
                 f"{res.cam_az.max()-res.cam_az.min():.0f}deg over {len(res.cam_az)} cameras.")
    if res.coverage_pct > 0.97:
        lines.append("-> Coverage is essentially complete; no significant gaps.")
        return "\n".join(lines)
    g_el = res.bin_el[~res.bin_covered]; g_az = res.bin_az[~res.bin_covered]
    # summarize gaps by elevation band
    for lo, hi, name in [(-90, 20, "low / eye-level"), (20, 55, "mid / 45deg"),
                         (55, 91, "high / top-down")]:
        m = (g_el >= lo) & (g_el < hi)
        if m.sum() > 0.02 * len(res.bin_covered):
            az = np.sort(g_az[m])
            lines.append(f"  GAP @ {name} elevation: {int(m.sum())} missing views, "
                         f"azimuths ~{az.min():.0f}-{az.max():.0f}deg")
    lines.append("RE-CAPTURE: shoot the gap elevation bands above (orbit the object at "
                 "those heights). Or GENERATE them (Module 2) — plausible, not faithful.")
    return "\n".join(lines)


def render_coverage_map(res: CoverageResult, W=720, H=300) -> np.ndarray:
    """azimuth(x 0-360) x elevation(y 0-90) map: green=covered target, red=gap, white=real cam."""
    img = np.full((H, W, 3), 28, np.uint8)
    for y in range(0, 91, 15):
        yy = int(H - y / 90 * (H - 1)); cv2.line(img, (0, yy), (W, yy), (50, 50, 50), 1)
        cv2.putText(img, f"{y}", (2, yy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    def xy(az, el):
        return int(az / 360 * (W - 1)), int(H - np.clip(el, 0, 90) / 90 * (H - 1))
    for az, el, cov in zip(res.bin_az, res.bin_el, res.bin_covered):
        x, y = xy(az, el); cv2.circle(img, (x, y), 5, (90, 200, 90) if cov else (60, 60, 220), -1)
    for az, el in zip(res.cam_az, res.cam_el):
        x, y = xy(az, el); cv2.circle(img, (x, y), 1, (255, 255, 255), -1)
    cv2.putText(img, "az->  green=covered red=GAP white=cameras  ^el", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return img


if __name__ == "__main__":
    import sys, os
    from core.training.gsplat_trainer import load_colmap_data
    DATA = sys.argv[1] if len(sys.argv) > 1 else r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
    data = load_colmap_data(DATA, factor=4, test_every=8, normalize=True)
    cams = np.stack([data["camtoworlds"][i][:3, 3] for i in range(len(data["camtoworlds"]))])
    O = np.median(data["points"], 0) if "points" in data and len(data["points"]) else cams.mean(0)
    res = analyze_coverage(cams, O)
    print(coverage_report(res))
    print(f"target fill poses available: {len(res.gap_dirs)}")
    out = r"F:\HYPERclone\apex-sfm\_infusion\coverage"
    os.makedirs(out, exist_ok=True)
    cv2.imwrite(os.path.join(out, "coverage_map.png"), render_coverage_map(res))
    print("viz:", os.path.join(out, "coverage_map.png"))
