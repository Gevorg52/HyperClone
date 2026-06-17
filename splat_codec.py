"""Flywheel step 1: canonical splat <-> voxel-grid codec + reconstruction validation.

A splat-native generative prior needs a CANONICAL, fixed-topology representation
(Gaussians are an unordered, variable-size set). This builds the foundational
encoding: normalize a Gaussian model into a unit cube, voxelize to a GxGxG grid
keeping the highest-opacity Gaussian per cell (exact params stored, so decode is
lossless per surviving Gaussian -> the ONLY loss is cell collisions). Decode back
to a splat. VALIDATE by rendering decoded vs original at held-out views (PSNR/SSIM
vs GT). The grid resolution where reconstruction ~= original is the viable
canonical size for the autoencoder/diffusion stage. Corpus-free de-risk.

Usage: python splat_codec.py <ckpt> [grid1 grid2 ...]
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from core.training.gsplat_trainer import (
    load_colmap_data, load_image, render_splats, ssim,
)

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
SH, dev = 3, "cuda"


def encode(splats, G):
    """splat dict -> (grid params per occupied cell, cell index, normalize transform)."""
    means = splats["means"]
    c = means.median(0).values                        # robust center
    # robust half-extent: 99th pct of per-Gaussian max-coord deviation (ignore far
    # floaters that would otherwise collapse everything into a few central cells)
    scale = torch.quantile((means - c).abs().max(dim=1).values, 0.99) * 1.05
    nm = (means - c) / (2 * scale)                    # normalized means ~[-0.5,0.5]
    cell = ((nm + 0.5) * G).floor().clamp(0, G - 1).long()
    lin = (cell[:, 0] * G + cell[:, 1]) * G + cell[:, 2]   # G^3 linear index
    opa = splats["opacities"].squeeze(-1)
    # keep highest-opacity Gaussian per cell: sort by opacity desc, take first per lin
    order = torch.argsort(opa, descending=True)
    lin_s = lin[order]
    seen = torch.zeros(G ** 3, dtype=torch.bool, device=means.device)
    keep_local = []
    # first occurrence per cell in opacity order
    uniq, first = np.unique(lin_s.cpu().numpy(), return_index=True)
    keep = order[torch.from_numpy(first).to(order.device)]
    return keep, dict(c=c, scale=scale)


def decode(splats, keep):
    return {k: v[keep].clone() for k, v in splats.items()}


def render_psnr_ssim(splats, data, idxs):
    ps, ss = [], []
    with torch.no_grad():
        for idx in idxs:
            b = load_image(data, idx); gt = b["image"].to(dev)
            K = b["K"].to(dev); c2w = b["c2w"].to(dev); H, W = gt.shape[:2]
            r, _, _ = render_splats(splats, torch.linalg.inv(c2w), K, W, H, SH,
                                    torch.zeros(3, device=dev), "antialiased")
            r = r[0].clamp(0, 1)
            ps.append(10 * math.log10(1 / (F.mse_loss(r, gt).item() + 1e-9)))
            ss.append(ssim(r.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).item())
    return float(np.mean(ps)), float(np.mean(ss))


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else \
        r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\output\ckpts\ckpt_29999_rank0.pt"
    grids = [int(x) for x in sys.argv[2:]] or [96, 128, 160, 200]
    s = torch.load(ck, weights_only=True, map_location=dev)["splats"]
    splats = {k: v.to(dev) for k, v in s.items()}
    N = splats["means"].shape[0]
    data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True)
    idxs = list(data["test_indices"])[:20]

    p0, s0 = render_psnr_ssim(splats, data, idxs)
    print(f"\noriginal splat: {N:,} Gaussians  PSNR {p0:.2f}  SSIM {s0:.4f}\n")
    print(f"{'grid':>6} {'cells':>12} {'kept':>10} {'kept%':>6}  {'PSNR':>7} {'SSIM':>7}  vs-orig")
    for G in grids:
        keep, _ = encode(splats, G)
        dec = decode(splats, keep)
        p, q = render_psnr_ssim(dec, data, idxs)
        print(f"{G:>6} {G**3:>12,} {len(keep):>10,} {100*len(keep)/N:>5.0f}%  "
              f"{p:7.2f} {q:7.4f}  dPSNR {p-p0:+.2f}")


if __name__ == "__main__":
    main()
