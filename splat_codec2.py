"""Flywheel step 2: canonical 'Gaussian-image' encoding (Morton-ordered) + validate.

The naive volumetric voxel grid failed (surface manifold -> 90% empty / collisions).
Fix: order the Gaussians along a Morton (Z-order) space-filling curve and reshape
to an HxWxF image -- every Gaussian gets a pixel (NO collisions, NO wasted volume),
spatially coherent (nearby-in-3D -> nearby-in-2D), a regular CNN/diffusion-ready
tensor, and LOSSLESS (a reorder of exact params). Normalize to a fixed count
N=H*W (prune-by-opacity / pad) so every scene maps to the same canonical shape --
the prerequisite for a cross-scene generative prior. Validate: render decoded vs
original (should ~match, modulo the prune-to-N which we showed is near-lossless).

Usage: python splat_codec2.py [ckpt] [side]   (N = side^2; default 1280 -> 1.64M)
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats, ssim

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
SH, dev = 3, "cuda"


def morton_key(nm, bits=10):
    """Morton (Z-order) key from normalized coords in [0,1)^3, `bits` per axis."""
    q = (nm.clamp(0, 1) * (2 ** bits - 1)).long()
    key = torch.zeros(nm.shape[0], dtype=torch.int64, device=nm.device)
    for i in range(bits):
        for a in range(3):
            key |= ((q[:, a] >> i) & 1) << (3 * i + a)
    return key


def encode(splats, side):
    """-> canonical (side,side,F) image (Morton-ordered) + the kept indices."""
    N = side * side
    opa = splats["opacities"].squeeze(-1)
    keep = torch.argsort(opa, descending=True)[:N]            # top-N by opacity
    if keep.numel() < N:                                      # pad by repeating last
        keep = torch.cat([keep, keep[-1].repeat(N - keep.numel())])
    means = splats["means"][keep]
    lo, hi = means.min(0).values, means.max(0).values
    nm = (means - lo) / (hi - lo + 1e-9)
    order = torch.argsort(morton_key(nm))                     # space-filling order
    keep = keep[order]
    feat = torch.cat([splats[k][keep].reshape(N, -1) for k in
                      ("means", "scales", "quats", "opacities", "sh0", "shN")], dim=1)
    return feat.reshape(side, side, -1), keep


def decode(splats_like, img, splats_ref):
    """image (side,side,F) -> splat dict (inverse reshape; params are exact)."""
    side = img.shape[0]; N = side * side
    flat = img.reshape(N, -1)
    dims = {k: splats_ref[k].shape[1:] for k in
            ("means", "scales", "quats", "opacities", "sh0", "shN")}
    sizes = {k: int(np.prod(v)) for k, v in dims.items()}
    out, o = {}, 0
    for k in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        out[k] = flat[:, o:o + sizes[k]].reshape(N, *dims[k]).contiguous(); o += sizes[k]
    return out


def render_psnr(splats, data, idxs):
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
    side = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
    s = torch.load(ck, weights_only=True, map_location=dev)["splats"]
    splats = {k: v.to(dev) for k, v in s.items()}
    data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True)
    idxs = list(data["test_indices"])[:20]
    p0, q0 = render_psnr(splats, data, idxs)
    print(f"\noriginal: {splats['means'].shape[0]:,} Gaussians  PSNR {p0:.2f} SSIM {q0:.4f}")
    img, keep = encode(splats, side)
    print(f"canonical Gaussian-image: {tuple(img.shape)}  ({side*side:,} Gaussians, "
          f"feat-dim {img.shape[-1]}) -> CNN/diffusion-ready, lossless reorder")
    dec = decode(None, img, splats)
    p, q = render_psnr(dec, data, idxs)
    print(f"decoded:  PSNR {p:.2f} SSIM {q:.4f}  (dPSNR vs original {p-p0:+.2f})")


if __name__ == "__main__":
    main()
