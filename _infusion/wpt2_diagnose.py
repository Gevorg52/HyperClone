"""WP-T2 diagnosis: WHERE is the mush, and WHY is the 400-512 view cluster weak?

Runs on the existing factor-3 production ckpt. For each held-out (every-8) view:
  - masked PSNR + masked LPIPS (visibility-masked, OGGSplat protocol)
  - under-observed fraction: mean over OBSERVED pixels of the accumulated splat
    weight coming from Gaussians seen by < k training views (SplatSuRe machinery)
  - camera geometry: distance to object centroid + "below-ness" (normalized
    y-component of the object->camera direction; the capture was exported up=+y,
    so below-ness < 0 ~ sole-from-below cameras looking up).

Then splits the test views into the weak [lo,hi] cluster vs the rest and reports:
  UNDER-OBSERVATION signature  -> cluster under-frac >> rest (Gaussians there are
                                  seen by few views; geometry is starved)
  SPECULAR/appearance signature-> cluster under-frac ~= rest (well-observed) but
                                  cluster masked-LPIPS/PSNR worse (view-dependent
                                  reflection the faithful levers can't model)

Also dumps a GT|render|under-heatmap diagnostic panel for one cluster view.

Usage:
  python _infusion/wpt2_diagnose.py <ckpt> <data_dir> --factor 3 \
      --cluster 400 512 --out _infusion/wpt2_diagnosis.json \
      --panel_out qa/wpt2_diag
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats  # noqa: E402
from core.qa.visibility import (  # noqa: E402
    VisibilityConfig, build_visibility_mask,
    per_gaussian_obs_counts, render_undersampled_map,
)
from core.qa.masked_metrics import masked_psnr, masked_lpips  # noqa: E402


def load_splats(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    return {k: v.to(device).float() for k, v in ck["splats"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("data_dir")
    ap.add_argument("--factor", type=int, default=3)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--k", type=int, default=3, help="a Gaussian is under-observed if seen by <k views")
    ap.add_argument("--cluster", nargs=2, type=int, default=[400, 512],
                    help="raw view-index range [lo,hi] of the weak cluster")
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel_out", default=None)
    args = ap.parse_args()
    device = "cuda"

    splats = load_splats(args.ckpt, device)
    N = int(splats["means"].shape[0])
    print(f"[diag] {N:,} Gaussians | ckpt {args.ckpt}", flush=True)

    data = load_colmap_data(args.data_dir, factor=args.factor, test_every=8, normalize=True)
    test = sorted(int(x) for x in data["test_indices"])
    names = data["image_names"]

    # object centroid + per-view geometry (normalized world)
    obj = torch.from_numpy(data["points"]).float().mean(0).numpy()
    c2ws = data["camtoworlds"]
    cam_pos = c2ws[:, :3, 3]
    d_all = np.linalg.norm(cam_pos - obj, axis=1)
    print(f"[diag] camera->object distance: min {d_all.min():.3f} median "
          f"{np.median(d_all):.3f} max {d_all.max():.3f} spread "
          f"{d_all.max()/max(d_all.min(),1e-6):.2f}x", flush=True)

    # per-Gaussian training-view observation counts (heavy: 1 projection/train view)
    print(f"[diag] computing per-Gaussian obs counts over "
          f"{len(data['train_indices'])} train views ...", flush=True)
    n_obs = per_gaussian_obs_counts(splats, data, device)
    no = n_obs.cpu().numpy()
    opac = torch.sigmoid(splats["opacities"].float()).cpu().numpy()
    # opacity-weighted global observation health
    w = np.clip(opac, 0, 1)
    seen1 = (no >= 1); seen3 = (no >= args.k)
    frac_under_g = float((w * (no < args.k)).sum() / w.sum())
    print(f"[diag] Gaussians seen by >=1 view {100*seen1.mean():.1f}% | "
          f">={args.k} views {100*seen3.mean():.1f}% | opacity-weighted "
          f"under-observed (<{args.k} views) {100*frac_under_g:.1f}%", flush=True)
    print(f"[diag] n_obs percentiles: p05 {np.percentile(no,5):.0f} p25 "
          f"{np.percentile(no,25):.0f} p50 {np.percentile(no,50):.0f} p75 "
          f"{np.percentile(no,75):.0f} p95 {np.percentile(no,95):.0f}", flush=True)

    vcfg = VisibilityConfig()
    per_view = []
    lo, hi = args.cluster
    with torch.no_grad():
        for idx in test:
            b = load_image(data, idx)
            gt = b["image"].to(device)
            K = b["K"].to(device); c2w = b["c2w"].to(device)
            H, W = gt.shape[:2]
            w2c = torch.linalg.inv(c2w)
            r, a, _ = render_splats(splats, w2c, K, W, H, args.sh_degree,
                                    torch.zeros(3, device=device), "antialiased")
            render = r[0].clamp(0, 1); alpha = a[0, ..., 0].clamp(0, 1)
            mask = build_visibility_mask(alpha, vcfg)["observed"]
            m = torch.from_numpy(mask).to(device)
            p = masked_psnr(render, gt, m)
            l = masked_lpips(render, gt, m, device, mode="bbox")
            # under-observed fraction within the OBSERVED region
            umap = render_undersampled_map(splats, n_obs, w2c, K, W, H, device, k=args.k)
            mt = m.bool()
            ufrac = float(umap[mt].mean().item()) if mt.any() else float("nan")
            dvec = cam_pos[idx] - obj
            belowness = float(dvec[1] / (np.linalg.norm(dvec) + 1e-9))
            per_view.append(dict(
                idx=idx, name=names[idx], dist=float(d_all[idx]),
                belowness=belowness, masked_psnr=float(p),
                masked_lpips=float(l), under_frac=ufrac,
                in_cluster=bool(lo <= idx <= hi)))
            del render, alpha, umap, r, a, gt
            if len(per_view) % 10 == 0:
                print(f"[diag]  {len(per_view)}/{len(test)} views", flush=True)
    torch.cuda.empty_cache()

    def agg(rows, key):
        v = np.array([r[key] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    clus = [r for r in per_view if r["in_cluster"]]
    rest = [r for r in per_view if not r["in_cluster"]]
    summary = {}
    for lbl, rows in (("cluster_%d_%d" % (lo, hi), clus), ("rest", rest)):
        summary[lbl] = {
            "n_views": len(rows),
            "mean_masked_psnr": agg(rows, "masked_psnr"),
            "mean_masked_lpips": agg(rows, "masked_lpips"),
            "mean_under_frac": agg(rows, "under_frac"),
            "mean_dist": agg(rows, "dist"),
            "mean_belowness": agg(rows, "belowness"),
        }
    # decisive ratio
    cu = summary["cluster_%d_%d" % (lo, hi)]["mean_under_frac"]
    ru = summary["rest"]["mean_under_frac"]
    under_ratio = cu / ru if ru else float("nan")

    result = dict(
        ckpt=str(args.ckpt), data_dir=str(args.data_dir), factor=args.factor,
        n_gaussians=N, k=args.k, cluster=[lo, hi],
        global_obs={
            "seen_ge1_pct": 100 * float(seen1.mean()),
            "seen_gek_pct": 100 * float(seen3.mean()),
            "opacity_weighted_under_pct": 100 * frac_under_g,
            "n_obs_p05": float(np.percentile(no, 5)),
            "n_obs_p50": float(np.percentile(no, 50)),
            "n_obs_p95": float(np.percentile(no, 95)),
        },
        distance={"min": float(d_all.min()), "median": float(np.median(d_all)),
                  "max": float(d_all.max()),
                  "spread": float(d_all.max() / max(d_all.min(), 1e-6))},
        summary=summary,
        cluster_vs_rest_under_ratio=under_ratio,
        per_view=per_view,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n[diag] ===== CLUSTER vs REST =====", flush=True)
    for lbl in summary:
        s = summary[lbl]
        print(f"  {lbl:16s} n={s['n_views']:3d}  PSNR {s['mean_masked_psnr']:.2f}  "
              f"LPIPS {s['mean_masked_lpips']:.4f}  under-frac {s['mean_under_frac']:.4f}  "
              f"dist {s['mean_dist']:.3f}  belowness {s['mean_belowness']:+.3f}", flush=True)
    print(f"  under-frac ratio (cluster/rest) = {under_ratio:.2f}x  -> "
          f"{'UNDER-OBSERVATION' if under_ratio > 1.5 else 'NOT under-obs (specular/appearance)'}",
          flush=True)
    print(f"[diag] wrote {args.out}", flush=True)

    # diagnostic panel for the worst cluster view (lowest PSNR)
    if args.panel_out and clus:
        import cv2
        worst = min(clus, key=lambda r: r["masked_psnr"])
        idx = worst["idx"]
        b = load_image(data, idx)
        gt = b["image"].to(device); K = b["K"].to(device); c2w = b["c2w"].to(device)
        H, W = gt.shape[:2]; w2c = torch.linalg.inv(c2w)
        with torch.no_grad():
            r, a, _ = render_splats(splats, w2c, K, W, H, args.sh_degree,
                                    torch.zeros(3, device=device), "antialiased")
            umap = render_undersampled_map(splats, n_obs, w2c, K, W, H, device, k=args.k)
        gtb = (gt.cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
        rb = (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
        u = umap.cpu().numpy()
        heat = cv2.applyColorMap((np.clip(u, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        panel = np.concatenate([gtb, rb, heat], axis=1)
        outp = Path(args.panel_out); outp.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outp / f"clusterworst_view{idx}_{worst['name']}_panel.jpg"),
                    panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[diag] panel (GT|render|under-heat) for worst cluster view {idx} "
              f"(PSNR {worst['masked_psnr']:.2f}) -> {outp}", flush=True)


if __name__ == "__main__":
    main()
