"""WP-F4 gate evidence: deck FLATNESS (plane RMS) + region LPIPS on held-out views.

The headline "polished vs blocky" number: fit a single plane to the RENDERED depth
inside the fixed deck box on each held-out view and report the RMS *relative* plane
residual  rms(|plane_depth - render_depth| / render_depth).  Lower = flatter/more
polished. Also reports an all-labeled-planes flatness aggregate (using each test
view's plane_mask), plus deck-box / textured-box / full-frame VGG-LPIPS so the gate
can check "deck LPIPS not worse" and "textured region untouched".

Boxes are fractional (x0, x1, y0, y1):
  DECK = (0.30, 0.72, 0.55, 0.95)  — identical to wpf1_deck_eval / render_compare
  TEX  = (0.30, 0.72, 0.18, 0.50)  — screen/upper-keyboard (textured control region)

Usage (repo root):
  python _infusion/wpf4_flatness_eval.py --data_dir <base/data> \
      --arms off=<ckpt> on=<ckpt> --out _infusion/wpf4_ab_flatness.json
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
    load_colmap_data, load_image, render_splats, get_lpips,
)
from core.plane_masks import _fit_plane, _plane_depth_along  # noqa: E402

DECK = (0.30, 0.72, 0.55, 0.95)
TEX = (0.30, 0.72, 0.18, 0.50)


def load_splats(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    return {k: v.to(device) for k, v in ck["splats"].items()}


def _box_px(box, H, W):
    x0, x1, y0, y1 = box
    return int(x0 * W), int(x1 * W), int(y0 * H), int(y1 * H)


def _plane_rms(depth_np, inv_K, ys, xs, dist_thresh=0.05):
    """RMS relative plane residual for pixels (ys, xs) of a rendered depth map.
    Robust: fit on all, then on inliers (<dist_thresh) once, report inlier RMS +
    inlier fraction. Returns (rms, inlier_frac, n) or None if too few pixels."""
    d = depth_np[ys, xs].astype(np.float64)
    good = np.isfinite(d) & (d > 1e-6)
    if good.sum() < 200:
        return None
    ys, xs, d = ys[good], xs[good], d[good]
    coords = np.stack([xs.astype(np.float64), ys.astype(np.float64),
                       np.ones(len(xs))], axis=0)        # (3,N)
    rays = inv_K @ coords                                 # (3,N)
    pts = rays * d[None, :]
    param = _fit_plane(pts)
    pred = _plane_depth_along(param, rays)
    rel = np.abs(pred - d) / np.maximum(d, 1e-6)
    inl = rel < dist_thresh
    if inl.sum() >= 200:
        param = _fit_plane(pts[:, inl])
        pred = _plane_depth_along(param, rays)
        rel = np.abs(pred - d) / np.maximum(d, 1e-6)
        rms = float(np.sqrt(np.mean((rel[inl]) ** 2)))
        return rms, float(inl.mean()), int(len(d))
    rms = float(np.sqrt(np.mean(rel ** 2)))
    return rms, float(inl.mean()), int(len(d))


def eval_ckpt(ckpt, data, lp, sh_degree, device, prior_dir):
    splats = load_splats(ckpt, device)
    deck_rms, allplane_rms, deck_lp, tex_lp, full_lp = [], [], [], [], []
    with torch.no_grad():
        for idx in sorted(int(x) for x in data["test_indices"]):
            b = load_image(data, idx)
            gt = b["image"].to(device); K = b["K"].to(device)
            c2w = b["c2w"].to(device)
            H, W = gt.shape[:2]; w2c = torch.linalg.inv(c2w)
            r, _, _ = render_splats(splats, w2c, K, W, H, sh_degree,
                                    torch.zeros(3, device=device), "antialiased",
                                    render_mode="RGB+ED")
            rgb = r[0][..., :3].clamp(0, 1)
            depth = r[0][..., 3]
            depth_np = depth.detach().cpu().numpy()
            inv_K = np.linalg.inv(K.detach().cpu().numpy().astype(np.float64))

            # deck-box flatness (single plane fit to rendered depth in the box)
            x0, x1, y0, y1 = _box_px(DECK, H, W)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            fr = _plane_rms(depth_np, inv_K, yy.ravel(), xx.ravel())
            if fr is not None:
                deck_rms.append(fr[0])

            # all-labeled-planes flatness (per-instance, pixel-weighted) via the
            # test view's plane_mask (built for ALL cache views)
            stem = os.path.splitext(os.path.basename(data["image_paths"][idx]))[0]
            mpath = os.path.join(prior_dir, "plane_mask", stem + ".npy")
            if os.path.isfile(mpath):
                pm = np.load(mpath)
                if pm.shape[:2] == depth_np.shape[:2]:
                    tot_w, tot_e = 0.0, 0.0
                    for lab in np.unique(pm[pm > 0]):
                        ys, xs = np.where(pm == lab)
                        fr = _plane_rms(depth_np, inv_K, ys, xs)
                        if fr is not None:
                            tot_e += fr[0] * fr[2]; tot_w += fr[2]
                    if tot_w > 0:
                        allplane_rms.append(tot_e / tot_w)

            # region + full-frame LPIPS
            rb = rgb.permute(2, 0, 1).unsqueeze(0)
            gb = gt.permute(2, 0, 1).unsqueeze(0)
            full_lp.append(float(lp(rb, gb)))
            for box, acc in ((DECK, deck_lp), (TEX, tex_lp)):
                bx0, bx1, by0, by1 = _box_px(box, H, W)
                acc.append(float(lp(rb[:, :, by0:by1, bx0:bx1],
                                    gb[:, :, by0:by1, bx0:bx1])))
    return dict(
        ckpt=str(ckpt), n_views=len(full_lp),
        deck_plane_rms=float(np.mean(deck_rms)) if deck_rms else None,
        allplane_rms=float(np.mean(allplane_rms)) if allplane_rms else None,
        deck_lpips=float(np.mean(deck_lp)),
        tex_lpips=float(np.mean(tex_lp)),
        full_lpips=float(np.mean(full_lp)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="label=ckpt ...")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--out", default="_infusion/wpf4_ab_flatness.json")
    args = ap.parse_args()
    device = "cuda"

    data = load_colmap_data(args.data_dir, factor=args.factor, test_every=8,
                            normalize=True)
    prior_dir = os.path.join(args.data_dir, "priors")
    lp = get_lpips(device)
    result = dict(
        metric="deck plane RMS (rel residual, lower=flatter) + region VGG-LPIPS "
               "on every8 held-out views",
        deck_box_frac=dict(zip(("x0", "x1", "y0", "y1"), DECK)),
        tex_box_frac=dict(zip(("x0", "x1", "y0", "y1"), TEX)),
        data_dir=str(args.data_dir), factor=args.factor, arms={})
    for spec in args.arms:
        label, ckpt = spec.split("=", 1)
        r = eval_ckpt(ckpt, data, lp, args.sh_degree, device, prior_dir)
        result["arms"][label] = r
        print(f"{label:6s} n={r['n_views']} deck_rms={r['deck_plane_rms']} "
              f"allplane_rms={r['allplane_rms']} deck_lpips={r['deck_lpips']:.4f} "
              f"tex_lpips={r['tex_lpips']:.4f} full_lpips={r['full_lpips']:.4f}")
    labels = list(result["arms"])
    if len(labels) == 2:
        a, b = (result["arms"][l] for l in labels)
        keys = ("deck_plane_rms", "allplane_rms", "deck_lpips", "tex_lpips", "full_lpips")
        result["delta"] = {}
        for k in keys:
            if a[k] is None or b[k] is None:
                continue
            result["delta"][k] = dict(abs=b[k] - a[k],
                                      pct=(b[k] - a[k]) / a[k] * 100.0)
        print(f"delta {labels[1]} vs {labels[0]}:")
        for k, dv in result["delta"].items():
            print(f"  {k}: {dv['abs']:+.5f} ({dv['pct']:+.1f}%)")
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
