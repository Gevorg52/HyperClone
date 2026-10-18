"""WP-S2 interaction + floater analysis (read-only):
  (A) dense-init points OUTSIDE the object mask  (the black-supervised hazard count)
  (B) per-arm object-hull classification: background Gaussians + BLACK-OPAQUE floaters
  (C) per-arm pose drift (CameraOptModule embeds) — the pose-stability hypothesis

Frames: with normalize=True, load_colmap_data returns dense-init points AND cameras in
the SAME normalized frame the trained splats live in, so the object hull (built from the
object-classified init points) and the Gaussian inside/outside test are all pure-3D in
one consistent frame (no projection of trained Gaussians needed).

(A) uses the exact stored COLMAP 2D observations (distorted full-res) vs the distorted
factor-2 BiRefNet masks (xy/factor) — frame-consistent, no undistortion.

Color: rgb = C0*sh0 + 0.5 (C0=0.28209...). near-black = max(rgb)<--rgb_thr; opaque =
sigmoid(opacity)>--opa_thr. BLACK-OPAQUE floater = outside object hull AND near-black
AND opaque.

Usage:
  python _infusion/wps2_analysis.py --eval_dir <eval/data> \
    --sparse <eval/data/sparse/0> \
    --masks <wps2/masks_2> \
    --arms a_full=<ckpt> b_obj=<ckpt> c_obj_alpha=<ckpt> \
    --out _infusion/wps2_floater_analysis.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.gsplat_trainer import load_colmap_data, _rotation_6d_to_matrix  # noqa: E402

C0 = 0.28209479177387814


def init_points_outside_mask(sparse_dir, masks_dir, factor):
    """Return (object_bool over pid_sorted order, stats dict). Classify every dense-init
    point OBJECT vs BACKGROUND by GEOMETRIC PROJECTION into ALL views (the 759k RoMa
    dense points carry NO COLMAP tracks, so track-lookup misses them — projection covers
    both). cv2.projectPoints applies the OPENCV distortion so the projected full-res px /
    factor indexes the distorted factor-2 BiRefNet mask exactly. A point is OBJECT if the
    majority of the views where it lands in-front + in-bounds put it inside the mask.
    Aligned to load_colmap_data's pid_sorted (= sorted point3D ids)."""
    import pycolmap
    import cv2
    rec = pycolmap.Reconstruction(sparse_dir)
    pid_sorted = sorted(rec.points3D.keys())
    pts = np.array([rec.points3D[p].xyz for p in pid_sorted], dtype=np.float64)  # raw world
    n = len(pid_sorted)
    inside = np.zeros(n, dtype=np.int32)
    total = np.zeros(n, dtype=np.int32)
    for im in rec.images.values():
        stem = os.path.splitext(im.name)[0]
        m = cv2.imread(os.path.join(masks_dir, stem + ".png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        H, W = m.shape
        cam = rec.cameras[im.camera_id]
        p = np.array(cam.params)
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], dtype=np.float64)
        dist = np.array([p[4], p[5], p[6], p[7]], dtype=np.float64)  # OPENCV k1,k2,p1,p2
        cfw = im.cam_from_world()
        R = cfw.rotation.matrix()
        t = cfw.translation.reshape(3)
        z = (pts @ R.T + t)[:, 2]                      # cam-frame depth
        infront = z > 1e-6
        rvec = cv2.Rodrigues(R)[0]
        proj, _ = cv2.projectPoints(pts, rvec, t, K, dist)
        px = np.nan_to_num(proj.reshape(-1, 2) / factor, nan=-1e9, posinf=-1e9, neginf=-1e9)
        xi = np.round(px[:, 0]).astype(np.int64)
        yi = np.round(px[:, 1]).astype(np.int64)
        inb = infront & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        idx = np.where(inb)[0]
        total[idx] += 1
        hit = m[yi[idx], xi[idx]] > 127
        inside[idx[hit]] += 1
    obj = (total > 0) & (inside * 2 >= total)     # majority (>=50%) of in-view frames inside
    stats = dict(
        n_init_points=n, n_object_points=int(obj.sum()),
        n_background_points=int((~obj).sum()),
        pct_background=float((~obj).mean() * 100.0),
        n_points_never_in_view=int((total == 0).sum()),
        mean_views_per_point=float(total.mean()))
    return obj, stats, pid_sorted


def build_hull(obj_points, margin=0.10, keep_pct=98.0):
    """Convex hull of object init points (normalized frame). First trims the farthest
    `100-keep_pct`% of points from the object centroid (robust against a few background
    points that leak past the majority-vote when they sit behind the object silhouette),
    then expands the hull `margin` from its centroid; returns a scipy Delaunay for fast
    batched inside/outside tests."""
    from scipy.spatial import ConvexHull, Delaunay
    c0 = obj_points.mean(0)
    d = np.linalg.norm(obj_points - c0, axis=1)
    keep = d <= np.percentile(d, keep_pct)
    trimmed = obj_points[keep]
    hull = ConvexHull(trimmed)
    verts = trimmed[hull.vertices]
    c = verts.mean(0)
    verts_exp = c + (verts - c) * (1.0 + margin)
    return Delaunay(verts_exp), c, verts_exp, int(keep.sum())


def load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def gaussian_stats(ck, dela, rgb_thr, opa_thr):
    s = ck["splats"]
    means = s["means"].float().numpy()
    opa = torch.sigmoid(s["opacities"].float()).numpy()
    sh0 = s["sh0"].float().numpy()[:, 0, :]             # (N,3)
    rgb = C0 * sh0 + 0.5
    maxrgb = rgb.max(axis=1)
    n = means.shape[0]
    inside = dela.find_simplex(means) >= 0
    outside = ~inside
    near_black = maxrgb < rgb_thr
    opaque = opa > opa_thr
    out_op = outside & opaque
    out_black = outside & near_black
    black_opaque_out = outside & near_black & opaque
    tot_mass = float(opa.sum())
    out_mass = float(opa[outside].sum())
    return dict(
        n_gaussians=int(n),
        n_outside=int(outside.sum()), pct_outside=float(outside.mean() * 100),
        n_outside_opaque=int(out_op.sum()), pct_outside_opaque=float(out_op.mean() * 100),
        n_outside_black=int(out_black.sum()),
        n_BLACK_OPAQUE_outside=int(black_opaque_out.sum()),
        pct_BLACK_OPAQUE_outside=float(black_opaque_out.mean() * 100),
        opacity_mass_outside=out_mass,
        opacity_mass_outside_frac=float(out_mass / max(1e-9, tot_mass)),
        mean_opacity_outside=float(opa[outside].mean()) if outside.any() else 0.0,
        mean_maxrgb_outside=float(maxrgb[outside].mean()) if outside.any() else 0.0)


def pose_drift(ck, scene_scale):
    if "pose_embeds" not in ck:
        return dict(note="no pose_embeds (pose_opt off?)")
    e = ck["pose_embeds"].float()             # (N,9): [0:3]=trans, [3:9]=6D rot delta
    trans = e[:, :3]
    tnorm = trans.norm(dim=1).numpy()
    id6 = torch.tensor([1.0, 0, 0, 0, 1.0, 0])
    R = _rotation_6d_to_matrix(e[:, 3:] + id6)          # (N,3,3)
    # angle from identity: theta = arccos((trace-1)/2)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos = ((tr - 1) / 2).clamp(-1, 1)
    ang = torch.rad2deg(torch.arccos(cos)).numpy()
    return dict(
        n_views=int(e.shape[0]),
        trans_norm_mean=float(tnorm.mean()), trans_norm_p95=float(np.percentile(tnorm, 95)),
        trans_norm_max=float(tnorm.max()),
        trans_norm_mean_pct_scene=float(tnorm.mean() / scene_scale * 100),
        rot_deg_mean=float(ang.mean()), rot_deg_p95=float(np.percentile(ang, 95)),
        rot_deg_max=float(ang.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--rgb_thr", type=float, default=0.15)
    ap.add_argument("--opa_thr", type=float, default=0.5)
    ap.add_argument("--hull_margin", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("[A] init points outside object mask ...", flush=True)
    obj_bool, initstats, pid_sorted = init_points_outside_mask(args.sparse, args.masks, args.factor)
    print("   ", json.dumps(initstats), flush=True)

    # normalized-frame points (same order as pid_sorted) for the hull
    data = load_colmap_data(args.eval_dir, factor=args.factor, test_every=8, normalize=True)
    pts = data["points"]
    scene_scale = float(data["scene_scale"])
    assert pts.shape[0] == len(pid_sorted), \
        f"point count mismatch {pts.shape[0]} vs {len(pid_sorted)}"
    obj_pts = pts[obj_bool]
    print(f"[B] object hull from {obj_pts.shape[0]:,} object init points "
          f"(margin {args.hull_margin}) ...", flush=True)
    dela, cen, _, n_hull_kept = build_hull(obj_pts, margin=args.hull_margin)
    initstats["n_object_points_after_trim"] = n_hull_kept

    arms = {}
    for spec in args.arms:
        lbl, ck_path = spec.split("=", 1)
        ck = load_ckpt(ck_path)
        g = gaussian_stats(ck, dela, args.rgb_thr, args.opa_thr)
        pd = pose_drift(ck, scene_scale)
        arms[lbl] = dict(ckpt=ck_path, floaters=g, pose_drift=pd)
        print(f"  {lbl:12s} nGS={g['n_gaussians']:,} outside={g['pct_outside']:.1f}% "
              f"outside-opaque={g['n_outside_opaque']:,} "
              f"BLACK-OPAQUE-out={g['n_BLACK_OPAQUE_outside']:,} "
              f"opa-mass-out={g['opacity_mass_outside_frac']*100:.2f}% | "
              f"pose trans%scene={pd.get('trans_norm_mean_pct_scene', float('nan')):.3f} "
              f"rot_deg={pd.get('rot_deg_mean', float('nan')):.4f}", flush=True)

    result = dict(
        note=("(A) dense-init pts outside object mask = black-supervised hazard in b/c; "
              "(B) black-opaque floater = outside object hull(margin) AND max(rgb)<rgb_thr "
              "AND sigmoid(opa)>opa_thr; (C) pose drift = CameraOptModule embeds vs identity"),
        rgb_thr=args.rgb_thr, opa_thr=args.opa_thr, hull_margin=args.hull_margin,
        scene_scale=scene_scale, init=initstats, arms=arms)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
