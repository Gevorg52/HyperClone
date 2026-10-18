"""Exploration for the GENUINE-HOLE testbed (WP-C1 prerequisite).

Loads a trained production splat, reports Gaussian-cloud geometry (opacity-weighted
bbox), and for each every-8 held-out view reports the object-area fraction. Also
renders a few held-out views + the opacity-weighted centroid so we can pick a
coherent surface region to carve into a void.

Usage: python _infusion/hole_explore.py <ckpt> <data_dir> [factor]
"""
import os, sys
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("APEX_HOLDOUT", None)
from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats

SH = 3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hole_explore")
os.makedirs(OUT, exist_ok=True)


def main(ckpt, data_dir, factor=2):
    dev = "cuda"
    c = torch.load(ckpt, map_location=dev, weights_only=False)
    sp = c["splats"] if isinstance(c, dict) and "splats" in c else c
    sp = {k: v.to(dev).float() for k, v in sp.items()}
    means = sp["means"]
    opac = torch.sigmoid(sp["opacities"].float())
    N = means.shape[0]
    print(f"[splat] {N:,} Gaussians")

    # opacity-weighted stats on 'solid' gaussians
    solid = opac > 0.1
    m = means[solid].cpu().numpy()
    w = opac[solid].cpu().numpy()
    print(f"[splat] solid (op>0.1): {solid.sum().item():,}")
    ctr = (m * w[:, None]).sum(0) / w.sum()
    print(f"[splat] opacity-wt centroid: {np.round(ctr,3)}")
    for ax, nm in enumerate("xyz"):
        p = np.percentile(m[:, ax], [1, 10, 25, 50, 75, 90, 99])
        print(f"  {nm}: p1..p99 = {np.round(p,3)}")

    data = load_colmap_data(data_dir, factor=factor, test_every=8, normalize=True)
    scene = float(data["scene_scale"])
    print(f"[data] scene_scale={scene:.3f}  n_imgs={len(data['image_names'])}")
    test_idx = sorted(int(i) for i in data["test_indices"])

    # object area per held-out view (alpha>0.5)
    rows = []
    for k, idx in enumerate(test_idx):
        b = load_image(data, idx)
        gt = b["image"].to(dev); K = b["K"].to(dev); c2w = b["c2w"].to(dev)
        H, W = gt.shape[:2]
        with torch.no_grad():
            r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH,
                                    torch.zeros(3, device=dev), "antialiased")
        alpha = a[0, ..., 0]
        objfrac = float((alpha > 0.5).float().mean())
        rows.append((idx, data["image_names"][idx], objfrac))
        # save a handful of renders side by side with GT
        if k % 6 == 0:
            rr = (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            gg = (gt.cpu().numpy() * 255).astype(np.uint8)
            cat = np.concatenate([rr, gg], 1)
            cv2.imwrite(os.path.join(OUT, f"v{idx:04d}.jpg"),
                        cv2.cvtColor(cat, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    fr = np.array([r[2] for r in rows])
    print(f"[obj-area] over {len(rows)} held-out views: "
          f"min {fr.min():.2f} median {np.median(fr):.2f} max {fr.max():.2f}")
    # camera azimuth of held-out views (relative to centroid, using est up)
    return ctr, scene, data


if __name__ == "__main__":
    ck = sys.argv[1] if len(sys.argv) > 1 else \
        r"_dataset/Laptop/apex_images/training/gsplat_prod/output_4k/ckpts/ckpt_29999_rank0.pt"
    dd = sys.argv[2] if len(sys.argv) > 2 else \
        r"_dataset/Laptop/apex_images/training/gsplat_prod/data"
    fac = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    main(ck, dd, fac)
