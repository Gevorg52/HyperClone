"""Fair base-vs-distilled comparison at the SAME resolution (factor 2).

The base run's test PSNR (21.45) was measured at factor 1 and the distill loop
evals at factor 2 — not comparable. This re-evals BOTH checkpoints with identical
code at factor 2 on the held-out test split, and adds LPIPS (the perceptual
metric that actually tracks the gated-Difix improvement; the trainer eval only
does PSNR/SSIM).

Usage:  python eval_compare.py <base_ckpt.pt> <distilled_ckpt.pt>
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.training.gsplat_trainer import (
    load_colmap_data, load_image, render_splats, ssim, get_lpips,
)

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\train\data"
SH = 3
FACTOR = 2
device = "cuda"


def load_splats(path):
    ck = torch.load(path, map_location=device, weights_only=True)
    return {k: v.to(device) for k, v in ck["splats"].items()}


def evaluate(splats, data):
    lp = get_lpips(device)
    ps, ss, ls = [], [], []
    with torch.no_grad():
        for idx in data["test_indices"]:
            b = load_image(data, idx)
            gt = b["image"].to(device)
            K = b["K"].to(device)
            c2w = b["c2w"].to(device)
            H, W = gt.shape[:2]
            w2c = torch.linalg.inv(c2w)
            bk = torch.zeros(3, device=device)
            r, _, _ = render_splats(splats, w2c, K, W, H, SH, bk, "antialiased")
            r = r[0].clamp(0, 1)
            mse = F.mse_loss(r, gt)
            ps.append(10.0 * math.log10(1.0 / (mse.item() + 1e-10)))
            rb = r.permute(2, 0, 1).unsqueeze(0)
            gb = gt.permute(2, 0, 1).unsqueeze(0)
            ss.append(ssim(rb, gb).item())
            ls.append(float(lp(rb, gb)))
    return float(np.mean(ps)), float(np.mean(ss)), float(np.mean(ls))


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: python eval_compare.py <base_ckpt> <distilled_ckpt>")
    base, dist = sys.argv[1], sys.argv[2]
    data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)
    print(f"\nEval @ factor {FACTOR} (1080p) on {len(data['test_indices'])} "
          f"held-out views   (LPIPS lower=better)\n")
    print(f"{'model':12s} {'Gaussians':>10s}  {'PSNR':>7s}  {'SSIM':>7s}  {'LPIPS':>7s}")
    rows = {}
    for name, ck in [("BASE", base), ("DISTILLED", dist)]:
        s = load_splats(ck)
        p, q, l = evaluate(s, data)
        rows[name] = (p, q, l)
        print(f"{name:12s} {s['means'].shape[0]:>10,}  {p:7.3f}  {q:7.4f}  {l:7.4f}")
    if "BASE" in rows and "DISTILLED" in rows:
        bp, bq, bl = rows["BASE"]
        dp, dq, dl = rows["DISTILLED"]
        print(f"\n{'delta':12s} {'':>10s}  {dp - bp:+7.3f}  {dq - bq:+7.4f}  "
              f"{dl - bl:+7.4f}   (LPIPS {-(dl - bl) / bl * 100:+.1f}%)")


if __name__ == "__main__":
    main()
