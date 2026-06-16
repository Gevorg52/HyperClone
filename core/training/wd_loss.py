"""Wasserstein Distortion (WD-R) perceptual loss — texture-STATISTIC VGG loss.

Thin wrapper around the CANONICAL balle-lab implementation (Apache-2.0, vendored as
_wd_canonical.py with a py3.11 `typing.override` shim). WD matches the LOCAL mean+std
of VGG16 features across a binomial pyramid, with a triangular weight selecting the
pyramid level near a target sigma (constant log2_sigma=4, per Apple WD-R 2603.23297 —
the spatially-varying map didn't help for 3DGS). Unlike LPIPS (pointwise feature L2),
this compares feature *distributions* -> gradient on texture where LPIPS goes flat.

WD-R is a TEXTURE lever (fine/stochastic detail per splat); it is NOT expected to fix
textureless / Gaussian-starved regions.
"""
from __future__ import annotations

import torch

_WD = None
LOG2_SIGMA = 4.0   # paper-recommended constant sigma (passed as a uniform map)


def _get_wd(device):
    global _WD
    if _WD is None:
        try:
            from core.training._wd_canonical import VGG16WassersteinDistortion
        except ModuleNotFoundError:           # trainer runs as a script
            from _wd_canonical import VGG16WassersteinDistortion
        _WD = VGG16WassersteinDistortion(
            feature_net="vgg16", num_levels=5,
            grayscale=False, normalize_center_to_zero=False,
        ).to(device).eval()
        for p in _WD.parameters():
            p.requires_grad_(False)
    return _WD


def wasserstein_distortion(pred, gt, device, num_scales: int = 3):
    """pred, gt: (B,3,H,W) in [0,1]. Scalar WD distance (canonical balle-lab)."""
    wd = _get_wd(device)
    ls = torch.full((pred.shape[0], 1, pred.shape[-2], pred.shape[-1]),
                    LOG2_SIGMA, device=device, dtype=pred.dtype)
    return wd(pred, gt, ls, num_scales=num_scales)
