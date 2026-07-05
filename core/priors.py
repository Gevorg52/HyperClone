"""core/priors.py — MoGe-2 metric depth + normal prior cache builder (WP-F1).

Builds a per-view geometric-prior cache from Microsoft MoGe-2
(`Ruicheng/moge-2-vitl-normal`, MIT) that downstream faithful-package losses
(WP-F1 mono-depth, WP-F3 DIM, WP-F4 planar) consume.

Design (scale-first metric alignment):
  MoGe-2 predicts METRIC depth (real metres) + camera-frame unit normals. The
  trainer renders in a NORMALIZED world space (`_similarity_from_cameras`), so
  the prior must be brought into that same depth scale. We do this per view with
  a robust scale(+optional shift) fit against the COLMAP sparse points the view
  observes: the point's normalized-world camera depth `z` vs MoGe's metric depth
  `d` sampled at the point's *undistorted* image projection. Because MoGe-2 is
  metric the per-view scales should CLUSTER; if they don't, we fall back to one
  GLOBAL pooled scale (reported in align.json meta).

Coordinate frames (IMPORTANT — kept pixel-identical to what the trainer renders):
  MoGe runs on the SAME undistorted+cropped image the trainer trains on
  (`load_image`), so depth/normal/conf/edge maps are byte-aligned with the
  rendered `RGB+ED` depth — no load-time remap needed. Sparse points are
  projected through the UNDISTORTED intrinsics (`K_undist`) + normalized pose,
  minus the ROI crop offset, so their sample location matches the map.

Cache layout (`<data_dir>/priors/`, stem-keyed, at training-factor resolution):
  depth/{stem}.npy      float32 (H,W)   aligned METRIC depth, +inf = invalid
  normal/{stem}.npy     float32 (H,W,3) camera-frame unit normal, 0 = invalid
  conf/{stem}.npy       float32 (H,W)   confidence in [0,1], 0 = invalid
  edge_mask/{stem}.png  uint8   (H,W)   inverse-Canny(50/150)+5px dilate;
                                        255 = away-from-edge (supervise here)
  align.json            per-image {scale, shift, n_inliers,
                                   median_rel_residual, reliable, depth_weight}
                        + meta {mode, factor, global_scale, dispersion, ...}

CLI:  python -m core.priors <data_dir> [--factor 2]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Trainer helpers give us the EXACT training frame + normalized geometry.
# (Importing the trainer also pulls gsplat; fine on the build machine.)
try:
    from core.training.gsplat_trainer import load_colmap_data, load_image
except ImportError:  # running from inside core/ as a loose script
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.training.gsplat_trainer import load_colmap_data, load_image


DEFAULT_MODEL = "Ruicheng/moge-2-vitl-normal"


# ---------------------------------------------------------------------------
# Sampling / map helpers
# ---------------------------------------------------------------------------

def _bilinear_sample(dmap: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear-sample (H,W) `dmap` at float pixel coords. Returns NaN out of
    bounds and +inf/NaN-propagated where any of the 4 corners is non-finite
    (invalid) — callers drop non-finite samples, so invalid depth never leaks."""
    H, W = dmap.shape
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    out = np.full(u.shape, np.nan, dtype=np.float64)
    inb = (x0 >= 0) & (y0 >= 0) & (x1 < W) & (y1 < H)
    if not inb.any():
        return out
    x0i, y0i, x1i, y1i = x0[inb], y0[inb], x1[inb], y1[inb]
    fu, fv = u[inb] - x0i, v[inb] - y0i
    Ia = dmap[y0i, x0i]; Ib = dmap[y0i, x1i]
    Ic = dmap[y1i, x0i]; Id = dmap[y1i, x1i]
    val = (Ia * (1 - fu) * (1 - fv) + Ib * fu * (1 - fv)
           + Ic * (1 - fu) * fv + Id * fu * fv)
    out[inb] = val
    return out


def _edge_mask(image_uint8_gray: np.ndarray) -> np.ndarray:
    """Inverse-Canny(50/150) + 5px dilate. 255 = away from edges (supervise),
    0 = on/near an edge (depth/normal unreliable there)."""
    edges = cv2.Canny(image_uint8_gray, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, k, iterations=1)
    return np.where(edges > 0, np.uint8(0), np.uint8(255))


def _conf_map(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Heuristic confidence in [0,1] (MoGe-2 emits no calibrated confidence in
    this snapshot): high on locally-smooth depth, low near discontinuities,
    0 where invalid. Uses relative log-depth gradient normalized by its median."""
    logd = np.log(np.clip(np.where(valid, depth, np.nan), 1e-6, None))
    gx = np.zeros_like(logd); gy = np.zeros_like(logd)
    gx[:, 1:] = np.abs(np.diff(logd, axis=1))
    gy[1:, :] = np.abs(np.diff(logd, axis=0))
    g = np.maximum(np.nan_to_num(gx, nan=0.0), np.nan_to_num(gy, nan=0.0))
    finite = valid & np.isfinite(g) & (g > 0)
    g0 = float(np.median(g[finite])) if finite.any() else 1.0
    conf = 1.0 / (1.0 + g / (g0 + 1e-6))
    return np.where(valid, conf, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-view scale(+shift) alignment
# ---------------------------------------------------------------------------

def _align_view(d: np.ndarray, z: np.ndarray, w: np.ndarray,
                ransac_rounds: int = 100, tau: float = 0.05,
                rng: np.random.Generator = None) -> dict | None:
    """Robust metric scale-first alignment for ONE view.

    d: MoGe metric depth samples · z: normalized-world COLMAP depth (target) ·
    w: per-sample weights (1/reproj_err). Returns None if <10 usable samples
    (view excluded). scale s maps MoGe->normalized: aligned = s*d + b.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if len(d) < 10:
        return None
    # percentile-prune the ratio to kill gross outliers before RANSAC
    r = z / d
    lo, hi = np.percentile(r, [0.1, 99.9])
    keep = (r >= lo) & (r <= hi)
    d, z, w = d[keep], z[keep], w[keep]
    if len(d) < 10:
        return None

    # ~100-round 1-point RANSAC on the scale (relative inlier test)
    n = len(d)
    cand = rng.integers(0, n, size=min(ransac_rounds, n))
    best_inl, best_count = None, -1
    for j in cand:
        s_c = z[j] / d[j]
        if s_c <= 0:
            continue
        inl = np.abs(s_c * d - z) / np.abs(z) < tau
        c = int(inl.sum())
        if c > best_count:
            best_count, best_inl = c, inl
    if best_inl is None or best_inl.sum() < 10:
        # fall back to plain median scale over all pruned samples
        best_inl = np.ones(n, dtype=bool)
    di, zi, wi = d[best_inl], z[best_inl], w[best_inl]

    # weighted LS refit, scale-only (through origin)
    denom = float((wi * di * di).sum())
    s = float((wi * zi * di).sum() / denom) if denom > 1e-12 else float(np.median(zi / di))
    med_s = float(np.median(np.abs(s * di - zi) / np.abs(zi)))

    # optional shift: accept only if it cuts median rel-residual >20% and is small
    b = 0.0
    A = np.array([[float((wi * di * di).sum()), float((wi * di).sum())],
                  [float((wi * di).sum()), float(wi.sum())]])
    rhs = np.array([float((wi * zi * di).sum()), float((wi * zi).sum())])
    try:
        s2, b2 = np.linalg.solve(A, rhs)
        med_sb = float(np.median(np.abs(s2 * di + b2 - zi) / np.abs(zi)))
        med_z = float(np.median(zi))
        if s2 > 0 and med_sb < 0.8 * med_s and abs(b2) < 0.2 * med_z:
            s, b, med_s = float(s2), float(b2), med_sb
    except np.linalg.LinAlgError:
        pass

    return dict(scale=s, shift=b, n_inliers=int(best_inl.sum()),
                median_rel_residual=med_s,
                inl_d=di, inl_z=zi, inl_w=wi)


def _depth_weight(reliable: bool, median_rel_residual: float) -> float:
    """Per-view soft weight from alignment residual (consumers may multiply the
    dense-depth loss by this). 0 for unreliable views."""
    if not reliable:
        return 0.0
    return float(np.clip(1.0 - median_rel_residual / 0.10, 0.2, 1.0))


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_cache(data_dir: str, factor: int = 2, model_name: str = DEFAULT_MODEL,
                resolution_level: int = 9, device: str = "cuda",
                reproj_max: float = 2.5, track_min: int = 5,
                depth_min: float = 0.001, depth_max: float = 1000.0,
                disp_thresh: float = 0.30,
                global_rel_thresh: float = 0.30,
                min_inliers_pv: int = 50,
                depth_only: bool = False) -> dict:
    """Build the MoGe-2 prior cache under `<data_dir>/priors/`. Returns a stats
    dict (also printed).

    depth_only (WP-N1): build ONLY the aligned `depth/` maps + `align.json` and
    skip the normal/conf/edge_mask subdirs. This is for runs that use the
    mono-depth loss but NO planar losses (which alone read normal/conf/edge). At
    native factor-1 (24 MP) the full cache is ~256 GB on disk and retaining the
    per-view normal + image tensors OOMs system RAM (~400 GB); depth-only keeps
    disk ~55 GB and RAM ~68 GB. Default OFF = existing full-cache behaviour."""
    from moge.model.v2 import MoGeModel
    import pycolmap

    t_start = time.time()
    out_root = Path(data_dir) / "priors"
    subs = ("depth",) if depth_only else ("depth", "normal", "conf", "edge_mask")
    for sub in subs:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    # Normalized geometry + the exact training frame (undistort maps, ROI, K_undist)
    data = load_colmap_data(data_dir, factor=factor, test_every=8, normalize=True)
    image_names = data["image_names"]
    points = data["points"]              # (P,3) normalized world
    point_obs = data["point_obs"]        # per-image row indices into `points`
    camtoworlds = data["camtoworlds"]    # normalized
    camera_ids = data["camera_ids"]
    roi_dict = data.get("roi_dict", {})
    n_views = len(image_names)

    # pid -> (reproj error, track length). pid_sorted matches load_colmap_data's
    # `points` row order (both sorted(rec.points3D.keys())).
    sparse_dir = os.path.join(data_dir, "sparse", "0")
    if not os.path.exists(sparse_dir):
        sparse_dir = os.path.join(data_dir, "sparse")
    rec = pycolmap.Reconstruction(sparse_dir)
    pid_sorted = sorted(rec.points3D.keys())
    pt_err = np.array([rec.points3D[p].error for p in pid_sorted], dtype=np.float64)
    pt_trk = np.array([rec.points3D[p].track.length() for p in pid_sorted], dtype=np.int64)

    print(f"[priors] {data_dir} | factor={factor} | {n_views} views | "
          f"{len(points)} sparse points | model={model_name}", flush=True)

    t0 = time.time()
    model = MoGeModel.from_pretrained(model_name).to(device).eval()
    print(f"[priors] model loaded in {time.time()-t0:.1f}s", flush=True)

    rng = np.random.default_rng(0)
    per_view = {}          # stem -> align dict (pre global-decision)
    view_meta = {}         # stem -> {H,W, valid, roi_off, cid}
    peak_vram = 0.0
    n_infer_samples = []

    for i in range(n_views):
        name = image_names[i]
        stem = os.path.splitext(os.path.basename(name))[0]
        cid = camera_ids[i]
        batch = load_image(data, i)
        img = batch["image"]                     # (H,W,3) float [0,1], training frame
        H, W = img.shape[:2]
        Ku = np.asarray(data["Ks_dict"][cid], dtype=np.float64)
        fx = float(Ku[0, 0])
        fov_x = 2.0 * math.degrees(math.atan(W / (2.0 * fx)))

        # ---- MoGe-2 inference on the undistorted training frame ----
        im3 = img.permute(2, 0, 1).contiguous().to(device)
        torch.cuda.reset_peak_memory_stats()
        out = model.infer(im3, fov_x=fov_x, resolution_level=resolution_level,
                          use_fp16=True, apply_mask=True)
        peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1024**2)
        depth_m = out["depth"].float().cpu().numpy()          # (H,W) metric, inf=invalid
        normal = out["normal"].float().cpu().numpy()          # (H,W,3) cam-frame unit
        valid = np.isfinite(depth_m) & (out["mask"].cpu().numpy())

        # ---- gather this view's sparse correspondences (project 3D -> undistorted px) ----
        rows = np.asarray(point_obs[i], dtype=np.int64) if point_obs is not None else np.empty(0, np.int64)
        d_s = z_s = w_s = np.empty(0)
        if rows.size:
            X = points[rows]                                   # (M,3) normalized world
            w2c = np.linalg.inv(camtoworlds[i]).astype(np.float64)
            Xc = X @ w2c[:3, :3].T + w2c[:3, 3]                 # (M,3) camera frame
            z = Xc[:, 2]
            uv = Xc @ Ku.T
            front = z > 1e-6
            u = np.full(len(rows), -1.0); v = np.full(len(rows), -1.0)
            u[front] = uv[front, 0] / uv[front, 2]
            v[front] = uv[front, 1] / uv[front, 2]
            roi = roi_dict.get(cid)
            if roi is not None and roi[2] > 0 and roi[3] > 0:
                u = u - roi[0]; v = v - roi[1]
            err = pt_err[rows]; trk = pt_trk[rows]
            good = (front & (z >= depth_min) & (z <= depth_max)
                    & (err <= reproj_max) & (trk >= track_min)
                    & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1))
            if good.any():
                d_samp = _bilinear_sample(depth_m, u[good], v[good])
                zz = z[good]; ww = 1.0 / np.clip(err[good], 1e-3, None)
                fin = np.isfinite(d_samp) & (d_samp > 1e-6)
                d_s, z_s, w_s = d_samp[fin], zz[fin], ww[fin]

        n_infer_samples.append(len(d_s))
        al = _align_view(d_s, z_s, w_s, tau=0.05, rng=rng) if len(d_s) >= 10 else None
        per_view[stem] = al
        # depth_only: drop the per-view normal + image tensors (~576 MB/view at
        # 24 MP) — they are only needed for the normal/conf/edge writes we skip.
        view_meta[stem] = dict(H=H, W=W, valid=valid, depth_m=depth_m,
                               normal=(None if depth_only else normal),
                               img=(None if depth_only else img))

        if (i + 1) % 25 == 0 or i == n_views - 1:
            done = sum(1 for a in per_view.values() if a is not None)
            print(f"[priors]  {i+1}/{n_views} views | aligned {done} | "
                  f"peakVRAM {peak_vram:.0f}MB | {(time.time()-t0):.0f}s", flush=True)

    # ---- global vs per-view decision ----------------------------------------
    # MoGe-2 is metric, so a single true meters->normalized scale exists. Two
    # regimes produce high per-view scale dispersion: (a) REAL per-image metric
    # bias that per-view alignment can absorb WHEN each view has enough SfM
    # points to pin its scale, or (b) NOISE from data-starved views where per-
    # view scales are unreliable. We pick per-view when it is well-determined
    # (ample inliers) AND actually lowers the residual, else the robust global
    # pooled scale (the plan's fallback). Dispersion only *gates* the check —
    # when scales already cluster, per-view is trivially fine.
    aligned_list = [a for a in per_view.values() if a is not None]
    scales = np.array([a["scale"] for a in aligned_list])
    med_s = float(np.median(scales)) if scales.size else 1.0
    logs = np.log(scales[scales > 0]) if scales.size else np.array([0.0])
    dispersion = float(1.4826 * np.median(np.abs(logs - np.median(logs)))) if logs.size else 0.0
    n_aligned = int(scales.size)

    global_scale = 1.0
    if scales.size:
        pd = np.concatenate([a["inl_d"] for a in aligned_list])
        pz = np.concatenate([a["inl_z"] for a in aligned_list])
        pw = np.concatenate([a["inl_w"] for a in aligned_list])
        gden = float((pw * pd * pd).sum())
        global_scale = float((pw * pz * pd).sum() / gden) if gden > 1e-12 else med_s

    # per-view vs global residual + inlier support (over aligned views)
    pv_res = np.array([a["median_rel_residual"] for a in aligned_list])
    gl_res = np.array([float(np.median(np.abs(global_scale * a["inl_d"] - a["inl_z"])
                                       / np.abs(a["inl_z"]))) for a in aligned_list])
    med_pv_res = float(np.median(pv_res)) if pv_res.size else float("nan")
    med_gl_res = float(np.median(gl_res)) if gl_res.size else float("nan")
    med_inliers = float(np.median([a["n_inliers"] for a in aligned_list])) if aligned_list else 0.0

    if n_aligned < 3:
        use_global, reason = True, "too few aligned views"
    elif dispersion <= disp_thresh:
        use_global, reason = False, "scales cluster (low dispersion)"
    else:
        well_det = (med_inliers >= min_inliers_pv) and (med_pv_res <= med_gl_res)
        use_global = not well_det
        reason = ("high dispersion but per-view well-determined "
                  "(ample inliers, lower residual)" if well_det
                  else "high dispersion + per-view data-starved / no residual gain")
    mode = "global" if use_global else "per-view"
    print(f"[priors] alignment: {n_aligned}/{n_views} views | median scale {med_s:.4f} "
          f"| log-MAD dispersion {dispersion:.3f} | median inliers {int(med_inliers)} "
          f"| residual per-view {med_pv_res:.4f} vs global {med_gl_res:.4f} "
          f"| mode={mode} ({reason}) | global_scale={global_scale:.4f}", flush=True)

    # ---- finalize per-view (s,b), reliability, then write maps ----
    align_json = {"meta": {
        "model": model_name, "factor": factor, "mode": mode, "mode_reason": reason,
        "n_views": n_views, "n_aligned": n_aligned,
        "median_scale": med_s, "dispersion": dispersion,
        "median_perview_residual": med_pv_res, "median_global_residual": med_gl_res,
        "median_inliers": med_inliers,
        "global_scale": global_scale, "global_rel_thresh": global_rel_thresh,
        "reproj_max": reproj_max,
        "track_min": track_min, "resolution_level": resolution_level,
        "depth_only": bool(depth_only),
    }, "images": {}}

    n_reliable = 0
    for i in range(n_views):
        name = image_names[i]
        stem = os.path.splitext(os.path.basename(name))[0]
        vm = view_meta[stem]
        al = per_view[stem]
        H, W = vm["H"], vm["W"]
        valid = vm["valid"]

        if use_global:
            s, b = global_scale, 0.0
            if al is None:
                n_inl, med_res, reliable = 0, float("nan"), False
            else:
                n_inl = al["n_inliers"]
                # residual of THIS view's inliers under the GLOBAL scale
                med_res = float(np.median(np.abs(s * al["inl_d"] - al["inl_z"])
                                          / np.abs(al["inl_z"])))
                # A view must actually AGREE with the global scale to be
                # supervised: its own fitted scale inside the sanity window
                # around the global scale AND its median residual under the
                # global scale below global_rel_thresh. "Had >=10
                # correspondences" alone is NOT enough — a badly-disagreeing
                # view would otherwise be merely down-weighted to the 0.2
                # depth_weight floor instead of excluded.
                scale_ok = 0.2 * global_scale <= al["scale"] <= 5.0 * global_scale
                reliable = (n_inl >= 10) and scale_ok and (med_res <= global_rel_thresh)
        else:
            if al is None:
                s, b, n_inl, med_res, reliable = global_scale, 0.0, 0, float("nan"), False
            else:
                s, b, n_inl, med_res = al["scale"], al["shift"], al["n_inliers"], al["median_rel_residual"]
                # per-view scale sanity vs the cohort median
                reliable = (n_inl >= 10) and (0.2 * med_s <= s <= 5.0 * med_s)

        dw = _depth_weight(reliable, med_res if math.isfinite(med_res) else 1.0)
        if reliable:
            n_reliable += 1

        # aligned metric depth (normalized units), inf=invalid
        aligned = np.where(valid, s * vm["depth_m"] + b, np.inf).astype(np.float32)
        np.save(out_root / "depth" / f"{stem}.npy", aligned)
        if not depth_only:
            np.save(out_root / "normal" / f"{stem}.npy", vm["normal"].astype(np.float32))
            np.save(out_root / "conf" / f"{stem}.npy", _conf_map(vm["depth_m"], valid))
            gray = (np.clip(vm["img"].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(str(out_root / "edge_mask" / f"{stem}.png"), _edge_mask(gray))

        align_json["images"][stem] = dict(
            scale=float(s), shift=float(b), n_inliers=int(n_inl),
            median_rel_residual=(float(med_res) if math.isfinite(med_res) else None),
            reliable=bool(reliable), depth_weight=float(dw))

    align_json["meta"]["n_reliable"] = n_reliable
    (out_root / "align.json").write_text(json.dumps(align_json, indent=2), encoding="utf-8")

    runtime = time.time() - t_start
    reliable_res = [align_json["images"][s]["median_rel_residual"]
                    for s in align_json["images"]
                    if align_json["images"][s]["reliable"]
                    and align_json["images"][s]["median_rel_residual"] is not None]
    stats = dict(
        data_dir=str(data_dir), factor=factor, mode=mode, n_views=n_views,
        n_aligned=n_aligned, n_reliable=n_reliable,
        median_scale=med_s, global_scale=global_scale, dispersion=dispersion,
        scale_min=float(scales.min()) if scales.size else None,
        scale_max=float(scales.max()) if scales.size else None,
        median_inliers=float(np.median(n_infer_samples)) if n_infer_samples else 0,
        median_reliable_residual=float(np.median(reliable_res)) if reliable_res else None,
        peak_vram_mb=peak_vram, runtime_s=runtime,
    )
    print(f"[priors] DONE {mode} | reliable {n_reliable}/{n_views} | "
          f"median reliable rel-residual "
          f"{stats['median_reliable_residual']} | scale span "
          f"[{stats['scale_min']}, {stats['scale_max']}] | "
          f"peakVRAM {peak_vram:.0f}MB | {runtime:.0f}s", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser(description="MoGe-2 prior cache builder (WP-F1)")
    ap.add_argument("data_dir", help="prepared COLMAP data dir (has sparse/, images_<factor>/)")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--resolution_level", type=int, default=9)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--depth_only", action="store_true",
                    help="Build only the aligned depth/ maps + align.json (skip "
                         "normal/conf/edge_mask). For mono-depth-only runs; needed "
                         "at native factor-1 to keep disk/RAM bounded (WP-N1).")
    args = ap.parse_args()
    build_cache(args.data_dir, factor=args.factor, model_name=args.model,
                resolution_level=args.resolution_level, device=args.device,
                depth_only=args.depth_only)


if __name__ == "__main__":
    main()
