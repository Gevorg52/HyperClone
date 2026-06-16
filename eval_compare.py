"""Compare N trained checkpoints at the SAME resolution on the held-out split.

Reports PSNR / SSIM / LPIPS / DISTS (DISTS is texture-stat-tolerant — a fairer
judge for a texture-statistic loss like WD-R; LPIPS alone is mildly adversarial to
WD-R). DISTS is optional (needs pyiqa); degrades to N/A if unavailable.

Usage:  python eval_compare.py <label=ckpt> [<label=ckpt> ...]
   e.g. python eval_compare.py base=...\ckpt_29999_rank0.pt wd=...\ckpt_29999_rank0.pt
   (back-compat: two bare paths are treated as base=.. cand=..)
"""
import math
import sys

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

_dists = None
try:
    import pyiqa
    _dists = pyiqa.create_metric("dists", device=device)
except Exception:
    _dists = None


def load_splats(path):
    ck = torch.load(path, map_location=device, weights_only=True)
    return {k: v.to(device) for k, v in ck["splats"].items()}


def evaluate(splats, data):
    lp = get_lpips(device)
    ps, ss, ls, ds = [], [], [], []
    with torch.no_grad():
        for idx in data["test_indices"]:
            b = load_image(data, idx)
            gt = b["image"].to(device)
            K = b["K"].to(device)
            c2w = b["c2w"].to(device)
            H, W = gt.shape[:2]
            w2c = torch.linalg.inv(c2w)
            r, _, _ = render_splats(splats, w2c, K, W, H, SH,
                                    torch.zeros(3, device=device), "antialiased")
            r = r[0].clamp(0, 1)
            ps.append(10.0 * math.log10(1.0 / (F.mse_loss(r, gt).item() + 1e-10)))
            rb = r.permute(2, 0, 1).unsqueeze(0)
            gb = gt.permute(2, 0, 1).unsqueeze(0)
            ss.append(ssim(rb, gb).item())
            ls.append(float(lp(rb, gb)))
            if _dists is not None:
                ds.append(float(_dists(rb, gb)))
    dmean = float(np.mean(ds)) if ds else float("nan")
    return float(np.mean(ps)), float(np.mean(ss)), float(np.mean(ls)), dmean


def main():
    global DATA
    args = sys.argv[1:]
    # optional data=<dir> override (else default Laptop)
    args = [a for a in args if not (a.startswith("data=") and (DATA := a.split("=", 1)[1]))]
    if not args:
        raise SystemExit("usage: python eval_compare.py [data=<dir>] <label=ckpt> ...")
    if "=" not in args[0]:                       # back-compat: bare paths
        args = [f"base={args[0]}"] + [f"cand{i}={p}" for i, p in enumerate(args[1:])]
    items = [a.split("=", 1) for a in args]

    data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)
    print(f"\nEval @ factor {FACTOR} on {len(data['test_indices'])} held-out views   "
          f"(LPIPS/DISTS lower=better; DISTS={'on' if _dists else 'N/A — pip pyiqa'})\n")
    print(f"{'model':12s} {'Gaussians':>10s}  {'PSNR':>7s}  {'SSIM':>7s}  "
          f"{'LPIPS':>7s}  {'DISTS':>7s}")
    rows = {}
    for label, ck in items:
        s = load_splats(ck)
        p, q, l, d = evaluate(s, data)
        rows[label] = (p, q, l, d)
        print(f"{label:12s} {s['means'].shape[0]:>10,}  {p:7.3f}  {q:7.4f}  "
              f"{l:7.4f}  {d:7.4f}")
    # deltas vs the first (baseline) model
    base_label = items[0][0]
    bp, bq, bl, bd = rows[base_label]
    print(f"\ndeltas vs {base_label} (LPIPS/DISTS: negative = better):")
    for label, ck in items[1:]:
        p, q, l, d = rows[label]
        print(f"  {label:12s} PSNR {p-bp:+.3f}  SSIM {q-bq:+.4f}  "
              f"LPIPS {l-bl:+.4f} ({-(l-bl)/bl*100:+.1f}%)  "
              f"DISTS {d-bd:+.4f} ({-(d-bd)/bd*100:+.1f}%)")


if __name__ == "__main__":
    main()
