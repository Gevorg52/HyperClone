"""Masked image metrics on the OBSERVED region only (WP-Q1 comp.1).

PSNR/SSIM are pixelwise-decomposable, so they are averaged over the observed mask
directly.  LPIPS is NOT pixelwise-decomposable (deep VGG features have large
receptive fields), so we crop the tight bounding box of the observed region (or,
in `tiled` mode, tile into >=64px crops with >=80% observed coverage and average).

Reuses the project's LPIPS (`get_lpips` in the trainer module, VGG, normalize=True)
read-only; the SSIM Gaussian-window math is duplicated here (small, self-contained)
because the trainer's `ssim` returns a scalar mean and we need the per-pixel map to
average over the mask.

All inputs: render/gt as (H, W, 3) float tensors in [0, 1]; mask as (H, W) bool
tensor.  All outputs deterministic (no random crops).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SSIM map (duplicated small helper — matches core.training.gsplat_trainer.ssim)
# ---------------------------------------------------------------------------

def _gauss_window(size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    return g / g.sum()


def _ssim_map(img1: torch.Tensor, img2: torch.Tensor,
              window_size: int = 11, C1: float = 0.01 ** 2,
              C2: float = 0.03 ** 2) -> torch.Tensor:
    """Per-pixel SSIM. Inputs (1, 3, H, W) in [0, 1]; returns (1, 3, H, W)."""
    window = _gauss_window(window_size, 1.5, img1.device)
    window = window.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=3)
    mu2 = F.conv2d(img2, window, padding=pad, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=3) - mu1_mu2
    return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def masked_psnr(render: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """PSNR over observed pixels only. NaN if the mask is empty."""
    if mask.sum() == 0:
        return float("nan")
    m = mask.unsqueeze(-1).expand_as(render)
    diff2 = ((render - gt) ** 2)[m]
    mse = diff2.mean().item()
    return 10.0 * math.log10(1.0 / (mse + 1e-10))


def masked_ssim(render: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean SSIM over observed pixels (per-pixel SSIM map averaged over the mask)."""
    if mask.sum() == 0:
        return float("nan")
    r = render.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    g = gt.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    smap = _ssim_map(r, g).mean(dim=1)[0]     # (H, W) averaged over channels
    return float(smap[mask].mean())


def _expand_span(a0: int, a1: int, size: int, limit: int) -> tuple:
    """Grow [a0, a1) to at least `size` px, centered, clamped to [0, limit)."""
    need = size - (a1 - a0)
    if need > 0:
        a0 = max(0, a0 - need // 2)
        a1 = min(limit, a0 + size)
        a0 = max(0, a1 - size)
    return a0, a1


def _lpips_window(lp, render: torch.Tensor, gt: torch.Tensor,
                  y0: int, y1: int, x0: int, x1: int, min_size: int) -> float:
    """LPIPS on window [y0:y1, x0:x1] of the FULL frames, robust to sliver
    windows: the window is first EXPANDED to min_size within image bounds (real
    context pixels, no synthetic ones); only if the frame itself is smaller than
    min_size is the remainder replicate-padded. replicate — unlike reflect — has
    no pad<dim constraint, so arbitrarily thin inputs survive."""
    H, W = render.shape[:2]
    y0, y1 = _expand_span(int(y0), int(y1), min_size, H)
    x0, x1 = _expand_span(int(x0), int(x1), min_size, W)
    r = render[y0:y1, x0:x1].permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    g = gt[y0:y1, x0:x1].permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    ph = max(0, min_size - r.shape[2])
    pw = max(0, min_size - r.shape[3])
    if ph or pw:
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)  # l, r, t, b
        r = F.pad(r, pad, mode="replicate")
        g = F.pad(g, pad, mode="replicate")
    with torch.no_grad():
        val = float(lp(r, g))
        lp.reset()
        return val


def masked_lpips(render: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
                 device, mode: str = "bbox", min_size: int = 64,
                 tile: int = 64, cover: float = 0.8) -> float:
    """Masked LPIPS (lower = better).

    mode="bbox"  : crop the tight bounding box of the observed region (simple,
                   default; fine when the observed region is roughly contiguous).
    mode="tiled" : average LPIPS over `tile`px crops with >=`cover` observed
                   coverage (better for annular / holey observed regions).
    Both modes are robust to arbitrary sliver masks: any window thinner than
    `min_size` is expanded within image bounds first, then replicate-padded if
    the frame itself is smaller than `min_size`. Returns NaN if nothing
    qualifies.
    """
    from core.training.gsplat_trainer import get_lpips
    lp = get_lpips(device)

    if mask.sum() == 0:
        return float("nan")

    if mode == "tiled":
        H, W = mask.shape
        vals = []
        for ty in range(0, max(1, H - tile + 1), tile):
            for tx in range(0, max(1, W - tile + 1), tile):
                m = mask[ty:ty + tile, tx:tx + tile]
                if float(m.float().mean()) < cover:
                    continue
                vals.append(_lpips_window(
                    lp, render, gt,
                    ty, min(ty + tile, H), tx, min(tx + tile, W), min_size))
        return float(np.mean(vals)) if vals else float("nan")

    # bbox mode
    ys, xs = torch.where(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return _lpips_window(lp, render, gt, y0, y1, x0, x1, min_size)
