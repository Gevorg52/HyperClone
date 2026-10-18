"""Gravity-align, recentre and metrically scale a 3DGS PLY / checkpoint — losslessly.

WP-O5 addendum. The trainer works in a NORMALISED SfM frame: arbitrary origin, arbitrary
scale (camera cloud radius ~1) and, on a 360 walkthrough, an arbitrary tilt — WP-O2 measured
zenith tilt median 11 deg / max 49 deg on these panoramas. In a viewer that means the floor is
not level and navigation is disorienting, which is the user's Postshot complaint. This applies
one similarity transform

    x' = s * R * (x - c)

to the FULL Gaussian model, and it is **lossless**: the transformed model renders
pixel-identically from correspondingly transformed cameras (`--verify` proves it per run).

What has to move, and why each one is easy to get wrong:
  means       s * R * (x - c)
  scales      log-space, so `+= log(s)`; rotation does NOT change them (they are the axis
              lengths of the ellipsoid, and R only reorients it)
  quats       q' = q_R (x) q, Hamilton product with the GLOBAL rotation on the LEFT, because
              R_new = R @ R_gauss. gsplat stores (w, x, y, z) — see `_quat_to_rotmat`.
  **SH**      the load-bearing one. Spherical harmonics are a function of the WORLD-space view
              direction, so rotating the scene rotates every Gaussian's view-dependent lobe.
              Leaving them alone silently breaks all view-dependent shading (specular
              highlights and lighting slide across surfaces as you orbit). Band 0 is
              rotation-invariant; bands 1..L must each be multiplied by a (2l+1)x(2l+1)
              rotation matrix. L (0..4) is read from the model itself (f_rest count) —
              WP-SH4: it used to be hard-coded to 3.
  opacity     unchanged (scalar).

SH ROTATION — how, and why this way. Each band l spans a (2l+1)-dimensional space that is
CLOSED under rotation, so the band matrix D_l is obtained by solving a small linear system
rather than by hand-implementing Wigner-D / Ivanic-Ruedenberg recursions:

    want   sum_n c'_n Y_n(u) = sum_m c_m Y_m(R^T u)   for all u
    write  Y_m(R^T u) = sum_n D[n, m] Y_n(u)          =>   c' = D c
    solve  A D = B   with   A[i, n] = Y_n(u_i),  B[i, m] = Y_m(R^T u_i)

over many well-spread sample directions u_i (Fibonacci sphere), in float64. This IS the
real-SH Wigner-D matrix (the band representation of R), obtained numerically; it is generic in
the degree. The basis Y is gsplat's own SH basis: its float64 torch oracle
`gsplat.cuda._torch_impl._eval_sh_bases_fast` (WP-SH4: `tests/test_sh_degree4.py` proves the
CUDA kernel equals it to fp32 tolerance at degrees 0..4, and that the band-4 basis is
orthonormal), so the result is in the rasterizer's exact SH convention by construction. The
oracle runs on CPU in float64, so D is exact to ~1e-12 and needs no GPU. D comes out
block-diagonal per band and orthogonal per band, and D(R1 R2) = D(R1) D(R2) — all three are
checked in the tests.

  python tools/align_ply.py --ply run/ply/point_cloud_29999.ply \
      --data_dir _dataset/360/train/wpo5_theater_027_8k \
      --out run/ply/point_cloud_29999_aligned.ply --up_axis y_down --verify 3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.alignment import _rotation_between_vectors  # noqa: E402  (pure math, reused)
from core.sh_utils import sh_degree_of  # noqa: E402
from tools.prune_ply import read_ckpt, read_gsplat_ply, splat_stats  # noqa: E402

# Default degree for the helper functions' signatures only. The transform itself derives the
# degree from the model (`sh_degree_of`), so a degree-4 (72 f_rest) PLY rotates bands 1..4.
SH_DEGREE = 3


# ---------------------------------------------------------------------------
# 1. spherical-harmonic rotation
# ---------------------------------------------------------------------------

def fibonacci_sphere(n: int, device, dtype=torch.float32) -> torch.Tensor:
    i = torch.arange(n, dtype=torch.float64) + 0.5
    phi = torch.acos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return torch.stack([torch.cos(theta) * torch.sin(phi),
                        torch.sin(theta) * torch.sin(phi),
                        torch.cos(phi)], dim=-1).to(device=device, dtype=dtype)


def sh_basis(dirs: torch.Tensor, degree: int = SH_DEGREE) -> torch.Tensor:
    """(M, K) matrix of gsplat's real-SH basis evaluated at `dirs` (any dtype/device).

    gsplat's torch oracle (`_eval_sh_bases_fast`, the same Sloan closed form the CUDA kernel
    implements; equality is a unit test). Directions are normalised first, as the kernel does.
    """
    from gsplat.cuda._torch_impl import _eval_sh_bases_fast

    d = dirs / dirs.norm(dim=-1, keepdim=True)
    return _eval_sh_bases_fast((degree + 1) ** 2, d)


def sh_basis_cuda(dirs: torch.Tensor, degree: int = SH_DEGREE) -> torch.Tensor:
    """Same basis through the CUDA kernel `gsplat.spherical_harmonics` (one-hot coeffs, K
    calls). Used by the tests as the cross-check of `sh_basis`."""
    import gsplat

    k = (degree + 1) ** 2
    cols = []
    for j in range(k):
        coeffs = torch.zeros(dirs.shape[0], k, 3, device=dirs.device, dtype=dirs.dtype)
        coeffs[:, j, 0] = 1.0
        cols.append(gsplat.spherical_harmonics(degree, dirs, coeffs)[:, 0])
    return torch.stack(cols, dim=-1)


def sh_rotation_matrix(R: np.ndarray, degree: int = SH_DEGREE, samples: int = 512,
                       device=None) -> torch.Tensor:
    """(K, K) block-diagonal matrix D with c' = D @ c for a world rotation R (float64).

    `device` is accepted for backward compatibility; the solve runs on CPU in float64.
    """
    k = (degree + 1) ** 2
    u = fibonacci_sphere(samples, "cpu", torch.float64)
    Rt = torch.as_tensor(R, dtype=torch.float64)
    # rows of (u @ R) are R^T u  ->  (R^T u)_i = sum_j R_ji u_j
    u_rot = u @ Rt
    A = sh_basis(u, degree)
    B = sh_basis(u_rot, degree)
    D = torch.zeros(k, k, dtype=torch.float64, device=device)
    for band in range(degree + 1):
        s, e = band * band, (band + 1) ** 2
        D[s:e, s:e] = torch.linalg.lstsq(A[:, s:e], B[:, s:e]).solution
    return D


def rotate_sh(sh: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """sh: (N, K, 3) -> (N, K, 3), applying c' = D c independently per colour channel."""
    return torch.einsum("ij,njc->nic", D.to(device=sh.device, dtype=sh.dtype), sh)


# ---------------------------------------------------------------------------
# 2. quaternions (gsplat order: w, x, y, z)
# ---------------------------------------------------------------------------

def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s,
             (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s,
             (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
             0.25 * s]
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_left_multiply(q_global: np.ndarray, quats: torch.Tensor) -> torch.Tensor:
    """q' = q_global (x) q  for a batch of (N, 4) wxyz quaternions (Hamilton product)."""
    w1, x1, y1, z1 = [float(v) for v in q_global]
    w2, x2, y2, z2 = quats.unbind(dim=-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


# ---------------------------------------------------------------------------
# 3. gravity
# ---------------------------------------------------------------------------

def camera_up_prior(c2w: np.ndarray) -> np.ndarray:
    """Mean world-space camera up (-Y column) over the given poses. OpenCV convention."""
    up = -c2w[:, :3, 1].mean(axis=0)
    return up / max(np.linalg.norm(up), 1e-9)


def distinct_centres(centres: np.ndarray, rel_tol: float = 1e-5) -> np.ndarray:
    """Collapse runs of co-located cameras to one position each, order preserved.

    Load-bearing for the metric scale on a RIG capture: WP-O2's 12 virtual crops of one
    panorama share an optical centre exactly by construction, and the trainer's images sort
    pano-major, so a naive `diff(centres)` is mostly zeros and its median step is 0. That
    would silently collapse the scale to 1.0 (caught by this being the first thing the
    transform report prints).
    """
    extent = float(np.linalg.norm(centres.max(axis=0) - centres.min(axis=0)))
    tol = max(extent * rel_tol, 1e-12)
    keep = [0]
    for i in range(1, centres.shape[0]):
        if np.linalg.norm(centres[i] - centres[keep[-1]]) > tol:
            keep.append(i)
    return centres[keep]


def fit_floor_plane(points: np.ndarray, up_prior: np.ndarray, cam_heights: np.ndarray,
                    max_tilt_deg: float = 35.0, iters: int = 2000,
                    inlier_frac: float = 0.02, seed: int = 0) -> dict:
    """RANSAC the dominant near-horizontal plane BELOW the cameras — the floor.

    The camera-derived up (mean panorama up) is only a prior here: it selects which points
    are floor candidates and disambiguates the normal's sign, but the returned normal comes
    from the plane fit, which is what actually makes the floor level in a viewer.
    """
    rng = np.random.default_rng(seed)
    h = points @ up_prior
    cam_lo = float(np.percentile(cam_heights, 5))
    cand = points[h < cam_lo]                      # everything below the lowest cameras
    if cand.shape[0] < 100:
        cand = points[h < np.percentile(h, 30)]
    if cand.shape[0] < 50:
        return {"normal": up_prior, "offset": float(np.median(h)), "method": "prior-only",
                "inliers": 0, "candidates": int(cand.shape[0])}
    span = float(np.percentile(cand @ up_prior, 95) - np.percentile(cand @ up_prior, 5))
    thresh = max(span * inlier_frac, 1e-6)
    cos_max = np.cos(np.deg2rad(max_tilt_deg))
    best = (-1, up_prior, float(np.median(cand @ up_prior)))
    n = cand.shape[0]
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = cand[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        if ln < 1e-12:
            continue
        nrm = nrm / ln
        if np.dot(nrm, up_prior) < 0:
            nrm = -nrm
        if np.dot(nrm, up_prior) < cos_max:        # not a floor-ish plane
            continue
        d = float(nrm @ p0)
        n_in = int((np.abs(cand @ nrm - d) < thresh).sum())
        if n_in > best[0]:
            best = (n_in, nrm, d)
    n_in, nrm, d = best
    # least-squares refit on the inliers
    inl = cand[np.abs(cand @ nrm - d) < thresh]
    if inl.shape[0] >= 3:
        centroid = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - centroid)
        refined = vt[-1]
        if np.dot(refined, up_prior) < 0:
            refined = -refined
        if np.dot(refined, up_prior) >= cos_max:
            nrm, d = refined, float(refined @ centroid)
            n_in = int((np.abs(cand @ nrm - d) < thresh).sum())
    return {"normal": nrm, "offset": d, "method": "ransac+refit", "inliers": int(n_in),
            "candidates": int(cand.shape[0]), "threshold": float(thresh),
            "tilt_vs_prior_deg": float(np.degrees(np.arccos(
                np.clip(np.dot(nrm, up_prior), -1, 1))))}


# ---------------------------------------------------------------------------
# 4. the transform
# ---------------------------------------------------------------------------

# name -> the world-space vector that "up" points along in the OUTPUT file.
# `y_down` means the file uses Y-increases-downward, i.e. up = -Y: the standard
# INRIA/gsplat/COLMAP 3DGS convention, which is what Postshot itself EXPORTS and what
# SuperSplat/PlayCanvas assume (they apply a fixed 180-deg Z rotation to match it).
AXES = {"y_down": np.array([0.0, -1.0, 0.0]), "y_up": np.array([0.0, 1.0, 0.0]),
        "z_up": np.array([0.0, 0.0, 1.0]), "z_down": np.array([0.0, 0.0, -1.0])}
AXIS_INDEX = {"y_down": 1, "y_up": 1, "z_up": 2, "z_down": 2}


def trajectory_plane_normal(stations: np.ndarray, up_prior: np.ndarray) -> dict:
    """Normal of the best-fit plane through the capture stations = the walking plane.

    A far better gravity estimate than the mean camera up on a 360 stick capture: the mean
    camera up is the mean STICK direction, and an operator who consistently leans the stick
    biases it systematically (measured here: 8.3 deg off). The operator's feet, by contrast,
    are on the floor. Only used when the track is convincingly planar (S[1] >> S[2]).
    """
    if stations.shape[0] < 3:
        return {"normal": up_prior, "singular_values": [], "planarity_ratio": 0.0,
                "angle_vs_prior_deg": 0.0,
                "note": "fewer than 3 distinct stations — fell back to the camera prior"}
    centred = stations - stations.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]
    if normal @ up_prior < 0:
        normal = -normal
    planarity = float(s[1] / max(s[2], 1e-12)) if s.shape[0] >= 3 else 0.0
    return {"normal": normal, "singular_values": s.tolist(), "planarity_ratio": planarity,
            "angle_vs_prior_deg": float(np.degrees(np.arccos(
                np.clip(normal @ up_prior, -1, 1))))}


def estimate_up(c2w: np.ndarray, points: np.ndarray, stations: np.ndarray,
                seed: int = 0, iters: int = 3, min_planarity: float = 3.0) -> dict:
    """Gravity-up, in three escalating steps, each recorded.

    prior  mean camera up (the stick direction) — biased, but fixes the SIGN
    track   normal of the plane through the capture stations (the walking plane)
    floor   RANSAC floor plane, re-seeded from `track` and iterated; the best-inlier
            iteration wins (the iteration oscillates by a couple of degrees rather than
            converging monotonically, so "last" would be arbitrary)
    """
    prior = camera_up_prior(c2w)
    track = trajectory_plane_normal(stations, prior)
    up = track["normal"] if track["planarity_ratio"] >= min_planarity else prior
    used = "trajectory-plane" if track["planarity_ratio"] >= min_planarity else "camera-prior"
    best, history = None, []
    for _ in range(iters):
        f = fit_floor_plane(points, up, stations @ up, max_tilt_deg=20.0, iters=4000,
                            seed=seed)
        history.append({"inliers": f["inliers"], "candidates": f["candidates"],
                        "moved_deg": float(np.degrees(np.arccos(np.clip(
                            f["normal"] @ up, -1, 1))))})
        if best is None or f["inliers"] > best["inliers"]:
            best = f
        up = f["normal"]
    return {"up": best["normal"], "floor": best, "prior": prior, "track": track,
            "seed_method": used, "ransac_history": history}


def build_transform(c2w: np.ndarray, points: np.ndarray, up_axis: str,
                    step_m: float, scale_mode: str, seed: int = 0) -> dict:
    centres = c2w[:, :3, 3]
    stations = distinct_centres(centres)
    est = estimate_up(c2w, points, stations, seed=seed)
    up_prior, up, floor = est["prior"], est["up"], est["floor"]
    R = _rotation_between_vectors(up, AXES[up_axis])

    steps = np.linalg.norm(np.diff(stations, axis=0), axis=1)
    step_median = float(np.median(steps)) if steps.size else 0.0
    if scale_mode == "baseline" and step_median <= 1e-9:
        raise ValueError("median inter-station step is 0 — cannot derive a metric scale; "
                         "pass --scale_mode none")
    scale = (step_m / step_median) if scale_mode == "baseline" else 1.0

    # centre: horizontal centroid of the trajectory, dropped onto the floor plane
    rc = centres @ R.T                                  # rotated camera centres
    axis_i = AXIS_INDEX[up_axis]
    axis_sign = float(AXES[up_axis][axis_i])
    horiz = [i for i in range(3) if i != axis_i]
    centre_rot = np.zeros(3)
    centre_rot[horiz[0]] = float(np.median(rc[:, horiz[0]]))
    centre_rot[horiz[1]] = float(np.median(rc[:, horiz[1]]))
    centre_rot[axis_i] = float((R @ (floor["offset"] * up))[axis_i])   # floor height
    centre = R.T @ centre_rot                           # back to source frame

    return {"R": R, "centre": centre, "scale": scale, "up_axis": up_axis,
            "up_prior": up_prior, "up_estimated": up, "floor": floor,
            "up_estimate": est, "axis_index": axis_i, "axis_sign": axis_sign,
            "step_median_units": step_median, "assumed_step_m": step_m,
            "num_stations": int(stations.shape[0]), "num_cameras": int(centres.shape[0]),
            "scale_mode": scale_mode,
            "rotation_angle_deg": float(np.degrees(np.arccos(
                np.clip((np.trace(R) - 1) / 2, -1, 1))))}


def apply_to_splats(splats: dict, T: dict, device=None) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R, c, s = T["R"], T["centre"], T["scale"]
    Rt = torch.as_tensor(R, dtype=torch.float32)
    ct = torch.as_tensor(c, dtype=torch.float32)
    out = {}
    out["means"] = (torch.as_tensor(s, dtype=torch.float32)
                    * ((splats["means"] - ct) @ Rt.T))
    out["scales"] = splats["scales"] + float(np.log(s))
    out["quats"] = quat_left_multiply(rotmat_to_quat_wxyz(R), splats["quats"])
    out["opacities"] = splats["opacities"].clone()
    degree = sh_degree_of(splats)                                  # raises on a bad count
    D = sh_rotation_matrix(R, degree)
    sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)          # (N, K, 3)
    sh = rotate_sh(sh, D)
    out["sh0"], out["shN"] = sh[:, :1], sh[:, 1:]
    return out


def transform_camera(c2w: np.ndarray, T: dict) -> np.ndarray:
    """The camera pose that sees the transformed model exactly as `c2w` saw the original."""
    R, c, s = T["R"], T["centre"], T["scale"]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R @ c2w[:3, :3]
    out[:3, 3] = s * (R @ (c2w[:3, 3] - c))
    return out


# ---------------------------------------------------------------------------
# 5. verification — the transform must be invisible
# ---------------------------------------------------------------------------

def verify(splats: dict, aligned: dict, data: dict, T: dict, n_views: int,
           device, sh_degree: int = None) -> dict:
    from core.training.gsplat_trainer import load_image, render_splats

    if sh_degree is None:
        sh_degree = sh_degree_of(splats)                           # render ALL bands

    src = {k: v.to(device) for k, v in splats.items()}
    dst = {k: v.to(device) for k, v in aligned.items()}
    idxs = np.unique(np.linspace(0, len(data["image_paths"]) - 1, n_views).astype(int))
    rows = []
    for i in idxs:
        batch = load_image(data, int(i))
        gt = batch["image"]
        H, W = int(gt.shape[0]), int(gt.shape[1])
        K = batch["K"].to(device)
        c2w = batch["c2w"].numpy().astype(np.float64)
        bg = torch.zeros(3, device=device)
        with torch.no_grad():
            a, _, _ = render_splats(src, torch.linalg.inv(
                torch.as_tensor(c2w, dtype=torch.float32, device=device)),
                K, W, H, sh_degree, bg, "antialiased")
            b, _, _ = render_splats(dst, torch.linalg.inv(
                torch.as_tensor(transform_camera(c2w, T), dtype=torch.float32,
                                device=device)),
                K, W, H, sh_degree, bg, "antialiased")
        d = (a[0][..., :3] - b[0][..., :3]).abs()
        mse = float((d ** 2).mean())
        rows.append({"index": int(i), "name": Path(data["image_paths"][int(i)]).name,
                     "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                     "psnr_db": (float("inf") if mse == 0
                                 else float(10 * np.log10(1.0 / max(mse, 1e-20))))})
        print(f"[verify] {rows[-1]['name']}: max|diff| {rows[-1]['max_abs_diff']:.3e}  "
              f"mean {rows[-1]['mean_abs_diff']:.3e}  PSNR {rows[-1]['psnr_db']:.1f} dB",
              flush=True)
    return {"views": rows,
            "max_abs_diff": max(r["max_abs_diff"] for r in rows),
            "min_psnr_db": min(r["psnr_db"] for r in rows)}


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python tools/align_ply.py",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ply")
    src.add_argument("--ckpt")
    p.add_argument("--data_dir", required=True,
                   help="packaged trainer dataset (supplies cameras + points for gravity, "
                        "recentring and scale)")
    p.add_argument("--out", required=True)
    p.add_argument("--up_axis", default="y_down", choices=sorted(AXES),
                   help="world up in the OUTPUT file. DEFAULT 'y_down' = the standard "
                        "INRIA/gsplat/COLMAP 3DGS convention (Y increases DOWNWARD), which "
                        "is what Postshot itself exports and what SuperSplat/PlayCanvas "
                        "assume - see README_POSTSHOT.md")
    p.add_argument("--step_m", type=float, default=0.65,
                   help="assumed metres per inter-panorama step (2 pano/s => 0.5 s of "
                        "walking). ASSUMPTION — recorded in the transform JSON")
    p.add_argument("--scale_mode", default="baseline", choices=["baseline", "none"])
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--holdout_list", default=None)
    p.add_argument("--transform_json", default=None)
    p.add_argument("--verify", type=int, default=3, help="views to render pre/post (0 = skip)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import gsplat
    from core.training.gsplat_trainer import load_colmap_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    source = Path(args.ply or args.ckpt)
    splats = read_gsplat_ply(source) if args.ply else read_ckpt(source)
    holdout = args.holdout_list or str(Path(args.data_dir) / "holdout_test.txt")
    data = load_colmap_data(args.data_dir, factor=args.factor, normalize=True,
                            holdout_list=holdout)
    c2w = np.asarray(data["camtoworlds"], dtype=np.float64)
    points = np.asarray(data["points"], dtype=np.float64)

    T = build_transform(c2w, points, args.up_axis, args.step_m, args.scale_mode, args.seed)
    print(f"[align] up prior {np.round(T['up_prior'], 4)} -> floor normal "
          f"{np.round(T['up_estimated'], 4)} "
          f"(tilt vs prior {T['floor'].get('tilt_vs_prior_deg', 0):.2f} deg, "
          f"{T['floor']['inliers']}/{T['floor']['candidates']} inliers, "
          f"{T['floor']['method']})", flush=True)
    print(f"[align] {T['num_stations']} distinct stations from {T['num_cameras']} cameras; "
          f"median step {T['step_median_units']:.4f} u = {args.step_m} m", flush=True)
    print(f"[align] rotation {T['rotation_angle_deg']:.2f} deg to +{args.up_axis.upper()} up * "
          f"scale {T['scale']:.4f} m/unit * centre {np.round(T['centre'], 4)}", flush=True)

    aligned = apply_to_splats(splats, T, device)

    ver = verify(splats, aligned, data, T, args.verify, device) if args.verify else None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gsplat.export_splats(means=aligned["means"], scales=aligned["scales"],
                         quats=aligned["quats"], opacities=aligned["opacities"],
                         sh0=aligned["sh0"], shN=aligned["shN"],
                         format="ply", save_to=str(out))

    cam_rot = np.stack([transform_camera(m, T) for m in c2w])[:, :3, 3]
    # signed so "height above floor" is positive-up whichever axis convention was chosen
    cam_h = cam_rot[:, AXIS_INDEX[args.up_axis]] * float(AXES[args.up_axis][
        AXIS_INDEX[args.up_axis]])
    payload = {
        "tool": "tools/align_ply.py",
        "argv": [sys.executable, __file__] + list(sys.argv[1:]),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(source.resolve()), "out": str(out.resolve()),
        "out_bytes": out.stat().st_size,
        "up_axis": args.up_axis,
        "rotation_matrix": T["R"].tolist(),
        "centre_source_frame": T["centre"].tolist(),
        "scale_metres_per_unit": T["scale"],
        "rotation_angle_deg": T["rotation_angle_deg"],
        "up_prior_mean_camera": T["up_prior"].tolist(),
        "up_estimated_floor_normal": np.asarray(T["up_estimated"]).tolist(),
        "up_estimate_chain": {
            "seed_method": T["up_estimate"]["seed_method"],
            "prior_mean_camera_up": T["up_estimate"]["prior"].tolist(),
            "trajectory_plane": {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                                 for k, v in T["up_estimate"]["track"].items()},
            "ransac_history": T["up_estimate"]["ransac_history"]},
        "floor_plane": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in T["floor"].items()},
        "step_median_units": T["step_median_units"],
        "num_stations": T["num_stations"], "num_cameras": T["num_cameras"],
        "assumed_step_m": args.step_m,
        "scale_assumption": "2 panoramas/s at ~1.3 m/s walking => one inter-panorama step is "
                            "0.5 s of walking ~ %.2f m. This is an ASSUMPTION; every metre "
                            "figure downstream inherits it." % args.step_m,
        "camera_height_above_floor_m": {
            "median": float(np.median(cam_h)), "p5": float(np.percentile(cam_h, 5)),
            "p95": float(np.percentile(cam_h, 95)), "sd": float(cam_h.std()),
            "note": "positive = above the fitted floor plane; a handheld 360 stick should "
                    "sit ~1.8-2.3 m, and the spread is the operator raising/lowering it"},
        "trajectory_extent_m": (cam_rot.max(axis=0) - cam_rot.min(axis=0)).tolist(),
        "verification": ver,
        "before": splat_stats(splats), "after": splat_stats(aligned),
        "seconds": round(time.time() - t0, 2),
    }
    tj = Path(args.transform_json) if args.transform_json else out.with_suffix(".transform.json")
    tj.parent.mkdir(parents=True, exist_ok=True)
    tj.write_text(json.dumps(payload, indent=1), encoding="utf8")
    print(f"[align] wrote {out} ({out.stat().st_size / 1e6:.1f} MB) + {tj} "
          f"in {time.time() - t0:.0f}s", flush=True)
    if ver:
        print(f"[align] LOSSLESS CHECK: max|diff| {ver['max_abs_diff']:.3e}, "
              f"min PSNR {ver['min_psnr_db']:.1f} dB over {len(ver['views'])} views",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
