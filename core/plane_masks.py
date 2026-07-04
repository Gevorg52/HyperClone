"""core/plane_masks.py — per-view plane-instance label maps (WP-F4).

Builds `priors/plane_mask/{stem}.npy` (uint16, 0=bg, 1..M = plane instances)
for the PlanarGS-style co-planarity / prior-normal losses. These label maps mark
contiguous, verified-planar image regions (the deck, walls, tables) so the
trainer's L_p / L_rn losses only flatten genuinely-planar geometry and leave
textured/curved regions untouched.

Mask source (recon 2026-07-03): production SAM 2.1 *plane* instances were never
preserved and the SAM foreground masks are object-vs-bg, not plane instances.
Running SAM-2.1 AMG fresh on ~300–570 full-res views is GPU-heavy and would
contend with the WP-S1 training jobs, and its class-agnostic segments would STILL
need the geometric plane inspection below. So we take the plan's documented
fallback: derive plane proposals **directly from the MoGe-2 normal+depth cache**
(WP-F1) with the PlanarGS geometric-inspection recipe. This route is CPU-only
(sklearn K-means + scipy connected-components + numpy plane fit), deterministic,
and needs no GPU — it reads only the already-cached `normal/` + `depth/` maps.

Geometric inspection (per view), PlanarGS-informed:
  1. valid = finite depth AND unit-length normal.
  2. Reconstruct the RAW MoGe metric depth from the cached aligned depth via
     align.json (raw = (aligned - shift)/scale) so the per-view depth *shift*
     (which bends a true plane) does not corrupt planarity checks; scale is a
     positive global factor and preserves planarity.
  3. K-means (k=5) split of the valid pixels by normal direction (fit on a
     deterministic subsample, assign all) → coarse orientation clusters.
  4. For each orientation cluster: connected components (8-conn) in image space.
  5. For each component with ≥ min_area px: regularized LS plane fit
     `param = (P Pᵀ + 0.01 I)⁻¹ P·1` on the unprojected 3D points; predict
     `plane_depth = 1/(paramᵀ K⁻¹ ũ)`; keep pixels with relative depth residual
     < dist_thresh (0.05); refit once; take the largest connected component of
     the surviving inliers; if still ≥ min_area, emit a unique instance label.
Clusters are disjoint by construction, so proposals never overlap (no explicit
containment/merge step needed — a documented simplification vs PlanarGS's SAM
route, whose overlapping instance masks require it).

Determinism: K-means `random_state=0`, subsample RNG seed 0, deterministic
connected-component labelling → byte-identical output for a fixed cache.

CLI:  python -m core.plane_masks <data_dir> [--factor 2] [--k_normal 5]
      [--dist_thresh 0.05] [--min_area 2000] [--only stem1,stem2]

Provenance: co_planar plane-fit + geometric-inspection recipe ported from
PlanarGS (github.com/SJTU-ViSYS-team/PlanarGS, MIT) — see THIRD_PARTY.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

try:
    from sklearn.cluster import KMeans
except ImportError as e:  # pragma: no cover
    raise SystemExit("core.plane_masks needs scikit-learn (KMeans). "
                     "pip install scikit-learn (constraints.txt)") from e

try:
    from core.training.gsplat_trainer import load_colmap_data
except ImportError:  # running as a loose script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.training.gsplat_trainer import load_colmap_data


# ---------------------------------------------------------------------------
# Plane fit (numpy port of PlanarGS planar/co_planar.py ProjectDepth)
# ---------------------------------------------------------------------------

def _fit_plane(P: np.ndarray) -> np.ndarray:
    """Regularized least-squares plane through the origin-referenced cloud.

    P: (3, N) camera-frame points. Returns param (3,) s.t. paramᵀ·X ≈ 1 on the
    plane (PlanarGS: param = (P Pᵀ + 0.01 I)⁻¹ P·1). The plane's z-depth along a
    ray r (=K⁻¹ũ) is 1/(paramᵀ r).
    """
    A = P @ P.T + 0.01 * np.eye(3)
    rhs = P.sum(axis=1)
    return np.linalg.solve(A, rhs)


def _plane_depth_along(param: np.ndarray, rays: np.ndarray) -> np.ndarray:
    """z-depth predicted by plane `param` along camera rays (3, N) = K⁻¹ũ."""
    denom = param @ rays                     # (N,)
    # planar depth = 1/(paramᵀ r); guard degenerate/near-zero denominators
    safe = np.where(np.abs(denom) < 1e-6, np.sign(denom) + (denom == 0), denom)
    return 1.0 / safe


# ---------------------------------------------------------------------------
# Per-view builder
# ---------------------------------------------------------------------------

def build_view_mask(normal: np.ndarray, depth_raw: np.ndarray, inv_K: np.ndarray,
                    k_normal: int = 5, dist_thresh: float = 0.05,
                    min_area: int = 2000, min_frac: float = 0.005,
                    subsample: int = 20000,
                    rng: np.random.Generator | None = None) -> tuple[np.ndarray, int]:
    """Return (label_map uint16 (H,W), n_instances). 0 = background.

    A plane region must exceed `min_area` px AND `min_frac` of the frame — the
    fractional floor is what keeps the mask to a handful of substantial surfaces
    (deck/wall/table) instead of shattering high-res frames into specks (the raw
    2000-px PlanarGS floor was calibrated for ~512-px DUSt3R inputs).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    H, W = depth_raw.shape
    min_area = max(int(min_area), int(min_frac * H * W))
    nrm = np.linalg.norm(normal, axis=-1)
    valid = np.isfinite(depth_raw) & (depth_raw > 1e-6) & (nrm > 0.5)
    labels = np.zeros((H, W), dtype=np.uint16)
    if valid.sum() < min_area:
        return labels, 0

    # homogeneous pixel coords + rays (K⁻¹ũ) for the whole grid (cheap)
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64),
                         np.arange(W, dtype=np.float64), indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)], axis=0)   # (3,HW)
    rays_full = inv_K @ coords                                            # (3,HW)

    # ---- K-means normal clustering (fit on subsample, assign all) ----
    vidx = np.flatnonzero(valid.ravel())
    N = normal.reshape(-1, 3)[vidx]                                       # (V,3)
    N = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
    kk = int(min(k_normal, max(1, len(vidx) // max(min_area, 1))))
    if kk < 1:
        kk = 1
    if len(vidx) > subsample:
        sub = rng.choice(len(vidx), size=subsample, replace=False)
        km = KMeans(n_clusters=kk, random_state=0, n_init=5).fit(N[sub])
    else:
        km = KMeans(n_clusters=kk, random_state=0, n_init=5).fit(N)
    centers = km.cluster_centers_                                         # (kk,3)
    # assign every valid pixel to nearest center (vectorized argmin)
    d2 = ((N[:, None, :] - centers[None, :, :]) ** 2).sum(-1)            # (V,kk)
    cluster_of = np.argmin(d2, axis=1)                                   # (V,)

    cluster_map = -np.ones(H * W, dtype=np.int32)
    cluster_map[vidx] = cluster_of
    cluster_map = cluster_map.reshape(H, W)

    depth_flat = depth_raw.ravel()
    next_id = 1
    struct = np.ones((3, 3), dtype=bool)                                 # 8-conn

    for c in range(kk):
        cmask = (cluster_map == c)
        if cmask.sum() < min_area:
            continue
        cc, n_cc = ndimage.label(cmask, structure=struct)
        for lab in range(1, n_cc + 1):
            comp = (cc == lab)
            if comp.sum() < min_area:
                continue
            idx = np.flatnonzero(comp.ravel())
            rays = rays_full[:, idx]                                     # (3,n)
            dvals = depth_flat[idx]                                      # (n,)
            pts = rays * dvals[None, :]                                  # (3,n)
            param = _fit_plane(pts)
            pred = _plane_depth_along(param, rays)
            resid = np.abs(pred - dvals) / np.maximum(dvals, 1e-6)
            inl = resid < dist_thresh
            if inl.sum() < min_area:
                continue
            # refit on inliers once, re-evaluate
            param = _fit_plane(pts[:, inl])
            pred = _plane_depth_along(param, rays)
            inl = (np.abs(pred - dvals) / np.maximum(dvals, 1e-6)) < dist_thresh
            if inl.sum() < min_area:
                continue
            # largest connected component of the surviving inliers
            inl_map = np.zeros(H * W, dtype=bool)
            inl_map[idx[inl]] = True
            inl_map = inl_map.reshape(H, W)
            cc2, n2 = ndimage.label(inl_map, structure=struct)
            if n2 == 0:
                continue
            sizes = ndimage.sum(np.ones_like(cc2), cc2, index=np.arange(1, n2 + 1))
            big = int(np.argmax(sizes)) + 1
            if sizes[big - 1] < min_area:
                continue
            labels[cc2 == big] = next_id
            next_id += 1

    return labels, next_id - 1


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_cache(data_dir: str, factor: int = 2, k_normal: int = 5,
                dist_thresh: float = 0.05, min_area: int = 2000,
                min_frac: float = 0.005, only: list[str] | None = None) -> dict:
    """Build `<data_dir>/priors/plane_mask/{stem}.npy` from the MoGe cache."""
    t0 = time.time()
    prior_dir = Path(data_dir) / "priors"
    normal_dir = prior_dir / "normal"
    depth_dir = prior_dir / "depth"
    out_dir = prior_dir / "plane_mask"
    if not normal_dir.is_dir() or not depth_dir.is_dir():
        raise SystemExit(f"[plane_masks] no MoGe cache under {prior_dir} "
                         f"(need normal/ and depth/). Build it with "
                         f"`python -m core.priors {data_dir} --factor {factor}`.")
    out_dir.mkdir(parents=True, exist_ok=True)

    align = json.loads((prior_dir / "align.json").read_text(encoding="utf-8"))
    imgs = align["images"]

    # stem -> undistorted K (training frame) via load_colmap_data
    data = load_colmap_data(data_dir, factor=factor, test_every=8, normalize=True)
    stem_to_K = {}
    for name, cid in zip(data["image_names"], data["camera_ids"]):
        stem = os.path.splitext(os.path.basename(name))[0]
        stem_to_K[stem] = np.asarray(data["Ks_dict"][cid], dtype=np.float64)

    stems = sorted(p.stem for p in normal_dir.glob("*.npy"))
    if only:
        want = set(only)
        stems = [s for s in stems if s in want]

    rng = np.random.default_rng(0)
    n_written = 0
    total_planes = 0
    covered = 0                # views with >=1 plane
    plane_px_frac = []
    for i, stem in enumerate(stems):
        Kd = stem_to_K.get(stem)
        if Kd is None:
            print(f"[plane_masks] WARN {stem}: no camera in COLMAP model; skipped")
            continue
        inv_K = np.linalg.inv(Kd)
        depth_al = np.load(depth_dir / f"{stem}.npy").astype(np.float64)   # aligned, inf=invalid
        normal = np.load(normal_dir / f"{stem}.npy").astype(np.float64)    # cam-frame unit
        # reconstruct RAW metric depth (undo per-view scale+shift) for planarity
        meta = imgs.get(stem, {})
        s = float(meta.get("scale", 1.0)) or 1.0
        b = float(meta.get("shift", 0.0))
        with np.errstate(invalid="ignore"):
            depth_raw = (depth_al - b) / s          # inf stays inf
        labels, n_pl = build_view_mask(
            normal, depth_raw, inv_K, k_normal=k_normal,
            dist_thresh=dist_thresh, min_area=min_area, min_frac=min_frac, rng=rng)
        np.save(out_dir / f"{stem}.npy", labels.astype(np.uint16))
        n_written += 1
        total_planes += n_pl
        if n_pl > 0:
            covered += 1
            plane_px_frac.append(float((labels > 0).mean()))
        if (i + 1) % 50 == 0 or i == len(stems) - 1:
            print(f"[plane_masks]  {i+1}/{len(stems)} | planes so far {total_planes} "
                  f"| covered {covered} | {time.time()-t0:.0f}s", flush=True)

    stats = dict(
        data_dir=str(data_dir), factor=factor, n_views=n_written,
        n_covered=covered, total_planes=total_planes,
        mean_planes_per_covered=(total_planes / covered) if covered else 0.0,
        median_plane_px_frac=(float(np.median(plane_px_frac)) if plane_px_frac else 0.0),
        k_normal=k_normal, dist_thresh=dist_thresh, min_area=min_area, min_frac=min_frac,
        runtime_s=time.time() - t0,
    )
    print(f"[plane_masks] DONE {n_written} views | {covered} covered | "
          f"{total_planes} planes ({stats['mean_planes_per_covered']:.2f}/covered) | "
          f"median plane-px frac {stats['median_plane_px_frac']:.3f} | "
          f"{stats['runtime_s']:.0f}s", flush=True)
    (out_dir / "_build_stats.json").write_text(json.dumps(stats, indent=2),
                                               encoding="utf-8")
    return stats


def main():
    ap = argparse.ArgumentParser(description="Plane-instance mask builder (WP-F4)")
    ap.add_argument("data_dir", help="prepared COLMAP data dir (has priors/, sparse/)")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--k_normal", type=int, default=5)
    ap.add_argument("--dist_thresh", type=float, default=0.05)
    ap.add_argument("--min_area", type=int, default=2000)
    ap.add_argument("--min_frac", type=float, default=0.005,
                    help="fractional area floor (of H*W) for a plane region")
    ap.add_argument("--only", default=None,
                    help="comma-separated stems to (re)build (debug)")
    args = ap.parse_args()
    only = args.only.split(",") if args.only else None
    build_cache(args.data_dir, factor=args.factor, k_normal=args.k_normal,
                dist_thresh=args.dist_thresh, min_area=args.min_area,
                min_frac=args.min_frac, only=only)


if __name__ == "__main__":
    main()
