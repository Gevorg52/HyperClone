"""WP-F1 gate evidence: held-out depth-residual eval (persisted, reproducible).

For each held-out (every-8) view: reproject the COLMAP sparse points the view
observes (filters: reproj error <= 2.5 px, track >= 5, depth in [0.001, 1000] —
identical to the sparse --depth_lambda loss / core.priors alignment filters),
bilinear-sample the rendered expected-depth (render_mode="RGB+ED") at those
pixels, and report per-view median |d_render - z| / z. Lower = the rendered
geometry agrees better with the SfM sparse depths.

INDEPENDENCE CAVEAT (state honestly in any writeup): the mono-depth loss
consumed MoGe depth on TRAIN views; this eval measures rendered depth against
COLMAP sparse points on HELD-OUT camera poses. The eval target is a different
modality from the supervision signal (triangulated SfM vs monocular prior), and
the evaluated *renders* are from views never supervised by the loss — but the
sparse 3D points themselves are triangulated from tracks that include train
views, so the point cloud is not fully independent of train-view geometry.

Usage (repo root):
  python _infusion/wpf1_depth_eval.py [--out _infusion/wpf1_ab_depth.json]
Defaults evaluate the WP-F1 A/B 15k checkpoints (off vs mono_depth_lambda=0.05).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.gsplat_trainer import (  # noqa: E402
    load_colmap_data, load_image, render_splats,
)
import pycolmap  # noqa: E402

AB = "_dataset/Laptop/apex_images/training/wpf1_ab"
DEFAULT_ARMS = {
    "off": f"{AB}/off/output/ckpts/ckpt_14999_rank0.pt",
    "on":  f"{AB}/on/output/ckpts/ckpt_14999_rank0.pt",
}
DEFAULT_DATA = "_dataset/Laptop/apex_images/training/gsplat_prod/data"
FILTERS = dict(reproj_max=2.5, track_min=5, depth_min=0.001, depth_max=1000.0)


def load_splats(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    return {k: v.to(device) for k, v in ck["splats"].items()}


def eval_ckpt(ckpt, data, rec_stats, sh_degree, device):
    pt_err, pt_trk = rec_stats
    points = data["points"]; point_obs = data["point_obs"]
    c2w = data["camtoworlds"]; cids = data["camera_ids"]
    roi_dict = data.get("roi_dict", {})
    splats = load_splats(ckpt, device)
    per_view = {}
    for i in sorted(int(x) for x in data["test_indices"]):
        cid = cids[i]; Ku = np.asarray(data["Ks_dict"][cid], dtype=np.float64)
        b = load_image(data, i); H, W = b["image"].shape[:2]
        w2c = np.linalg.inv(c2w[i]).astype(np.float64)
        rows = np.asarray(point_obs[i])
        if rows.size == 0:
            continue
        X = points[rows]; Xc = X @ w2c[:3, :3].T + w2c[:3, 3]
        z = Xc[:, 2]; uv = Xc @ Ku.T
        front = z > 1e-6
        u = np.full(len(rows), -1.0); v = np.full(len(rows), -1.0)
        u[front] = uv[front, 0] / uv[front, 2]
        v[front] = uv[front, 1] / uv[front, 2]
        roi = roi_dict.get(cid)
        if roi is not None and roi[2] > 0 and roi[3] > 0:
            u -= roi[0]; v -= roi[1]
        good = (front & (z >= FILTERS["depth_min"]) & (z <= FILTERS["depth_max"])
                & (pt_err[rows] <= FILTERS["reproj_max"])
                & (pt_trk[rows] >= FILTERS["track_min"])
                & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1))
        if good.sum() < 10:
            continue
        with torch.no_grad():
            wt = torch.from_numpy(w2c).float().to(device)
            Kt = torch.from_numpy(Ku).float().to(device)
            renders, _, _ = render_splats(splats, wt, Kt, W, H, sh_degree,
                                          torch.zeros(3, device=device),
                                          "antialiased", render_mode="RGB+ED")
            depth_map = renders[0][..., 3]
            gx = torch.from_numpy((u[good] / (W - 1)) * 2 - 1).float().to(device)
            gy = torch.from_numpy((v[good] / (H - 1)) * 2 - 1).float().to(device)
            grid = torch.stack([gx, gy], -1)[None, None]
            d_r = torch.nn.functional.grid_sample(
                depth_map[None, None], grid, mode="bilinear",
                align_corners=True)[0, 0, 0].cpu().numpy()
        rel = np.abs(d_r - z[good]) / np.abs(z[good])
        per_view[int(i)] = dict(n_points=int(good.sum()),
                                median_rel=float(np.median(rel)))
    med = np.array([v["median_rel"] for v in per_view.values()])
    return dict(ckpt=str(ckpt), n_views=len(per_view),
                median=float(np.median(med)), mean=float(np.mean(med)),
                p90=float(np.percentile(med, 90)), per_view=per_view)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument("--arms", nargs="*", default=[f"{k}={v}" for k, v in DEFAULT_ARMS.items()],
                    help="label=ckpt pairs")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--out", default="_infusion/wpf1_ab_depth.json")
    args = ap.parse_args()
    device = "cuda"

    data = load_colmap_data(args.data_dir, factor=args.factor, test_every=8,
                            normalize=True)
    sparse_dir = os.path.join(args.data_dir, "sparse", "0")
    rec = pycolmap.Reconstruction(sparse_dir)
    pid_sorted = sorted(rec.points3D.keys())
    rec_stats = (np.array([rec.points3D[p].error for p in pid_sorted]),
                 np.array([rec.points3D[p].track.length() for p in pid_sorted]))

    result = dict(metric="held-out per-view median |d_render - z|/z vs COLMAP "
                         "sparse (lower=better)",
                  data_dir=str(args.data_dir), factor=args.factor,
                  split="every8 held-out", filters=FILTERS, arms={})
    for spec in args.arms:
        label, ckpt = spec.split("=", 1)
        r = eval_ckpt(ckpt, data, rec_stats, args.sh_degree, device)
        result["arms"][label] = r
        print(f"{label:6s} views={r['n_views']}  median={r['median']:.4f}  "
              f"mean={r['mean']:.4f}  p90={r['p90']:.4f}")
    labels = list(result["arms"])
    if len(labels) == 2:
        a, b = (result["arms"][l] for l in labels)
        result["delta"] = {k: dict(
            abs=b[k] - a[k], pct=(b[k] - a[k]) / a[k] * 100.0)
            for k in ("median", "mean", "p90")}
        print(f"delta {labels[1]} vs {labels[0]}: " + "  ".join(
            f"{k} {result['delta'][k]['abs']:+.4f} ({result['delta'][k]['pct']:+.1f}%)"
            for k in ("median", "mean", "p90")))
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
