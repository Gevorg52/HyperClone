"""Corruption knobs for the WP-Q1 sensitivity gates (default off).

Shared by qa_report.py (phase-1 metrics on held-out views) and the phase-2
scorer clip renderers (core/qa/scorers/common.py) so there is exactly ONE copy
of the floater-injection and blur math (review item: no diverging duplicates).
"""
import math

import cv2
import numpy as np
import torch


def inject_floaters(splats: dict, n: int, seed: int, opacity: float = 0.9,
                    scale_frac: float = 0.01) -> dict:
    """Add `n` random floater Gaussians at random positions within the scene bbox
    (opacity ~`opacity`), returning a NEW splats dict. Deterministic (seeded CPU
    RNG). Simulates cloud/floater corruption for the sensitivity gate.

    Placement uses the ROBUST 1-99 percentile bbox of the Gaussian centers (the
    full min/max bbox is blown up by a few far SfM outliers, so raw-bbox floaters
    land in empty far space). Floater size = `scale_frac` x robust bbox diagonal,
    so each is a visible blob (>>median splat) rather than an invisible speck."""
    dev = splats["means"].device
    g = torch.Generator().manual_seed(seed)                 # CPU generator (deterministic)
    m = splats["means"].detach().cpu()
    lo = torch.quantile(m, 0.01, dim=0)
    hi = torch.quantile(m, 0.99, dim=0)
    diag = float((hi - lo).norm())
    pos = lo + (hi - lo) * torch.rand(n, 3, generator=g)
    lin_scale = max(scale_frac * diag, 1e-6)                # visible blob size
    scales = torch.full((n, 3), math.log(lin_scale))
    quats = torch.zeros(n, 4); quats[:, 0] = 1.0            # identity rotation
    opac = torch.full((n,), math.log(opacity / (1.0 - opacity)))
    sh0 = (torch.rand(n, 1, 3, generator=g) - 0.5) * 0.5    # random-ish gray-ish color
    Kc = splats["shN"].shape[1]
    shN = torch.zeros(n, Kc, 3)
    add = {"means": pos, "opacities": opac, "quats": quats, "scales": scales,
           "sh0": sh0, "shN": shN}
    return {k: torch.cat([splats[k], add[k].to(dev)], dim=0) for k in splats}


def gaussian_blur(render: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian-blur an (H,W,3)[0,1] render (simulates a blurry/soft render)."""
    a = render.detach().cpu().numpy()
    b = cv2.GaussianBlur(a, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    return torch.from_numpy(b).to(render.device)
