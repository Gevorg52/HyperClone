"""GaMO-pattern opacity-mask LATENT GATE for gated generative completion (WP-C1).

Clean-room re-implementation of the ~50-line mechanism verified from GaMO's
pipeline source (2512.25073): the splat's own rendered opacity marks the
unobserved region; during diffusion denoising, real-content latents are
re-injected at OBSERVED pixels so generation is *structurally unable* to touch
evidence, and a feathered pixel paste-back guarantees observed pixels stay
byte-identical to the render.

This module is the backbone-AGNOSTIC, torch-only mechanism (importable on Windows
or WSL). The diffusers pipeline glue that consumes it lives in the WSL runner
`_infusion/gamo_fill.py` (SDXL-Inpaint v1) — that host has diffusers; the main
Windows env deliberately does not.

Mechanism (EXECUTION_PLAN WP-C1, verified against GaMO pipeline source):
  render alpha -> mask = (alpha < 0.6)
    |> max-pool (NOT bilinear) to the latent grid (8x for SD/SDXL)
    |> binarize >= 0.5
    |> progressive dilation (F.max_pool2d k=5, x2 / x1 / x0)   [widest first]
  VAE-encode the render -> x0_known
  at denoise trigger points 30/50/70%:
    x_known = scheduler.add_noise(x0_known, noise, t)      # flow-match: (1-s)x0+s*eps
    x = m*x + (1-m)*x_known                                # widest mask first
    (RePaint 3-iter resample harmonization -- see note in gamo_fill.py)
  feathered pixel paste-back (INSET 0.6, FEATHER 0.08)      # hard gate integrity

m == 1 => GENERATE (unobserved void); m == 0 => KEEP (observed evidence).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Latent-space mask (max-pool, NOT bilinear -- GaMO-specified)
# ---------------------------------------------------------------------------

def build_latent_mask(alpha, lat_h: int, lat_w: int, alpha_thresh: float = 0.6,
                      device=None) -> torch.Tensor:
    """alpha (H,W) in [0,1] -> latent-grid generate-mask (1,1,lat_h,lat_w) in {0,1}.

    unobserved = (alpha < alpha_thresh); MAX-pool to the latent grid so a latent
    cell is 'generate' iff ANY of its pixels is unobserved (conservative — the
    generation region fully covers the void); binarize >= 0.5.
    """
    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(np.asarray(alpha), dtype=torch.float32)
    alpha = alpha.float()
    if device is not None:
        alpha = alpha.to(device)
    unobs = (alpha < alpha_thresh).float()[None, None]              # (1,1,H,W)
    m = F.adaptive_max_pool2d(unobs, (lat_h, lat_w))                # -> latent grid
    return (m >= 0.5).float()


def progressive_masks(m: torch.Tensor, k: int = 5, iters=(2, 1, 0)) -> list:
    """Return a list of progressively-dilated masks (widest first).

    Dilation via F.max_pool2d (k, stride 1, pad k//2), applied `n` times per
    entry. iters=(2,1,0) -> [x2 dilated, x1 dilated, undilated] to consume at the
    30/50/70% denoise trigger points.
    """
    pad = k // 2
    out = []
    for n in iters:
        mm = m
        for _ in range(int(n)):
            mm = F.max_pool2d(mm, kernel_size=k, stride=1, padding=pad)
        out.append(mm.clamp(0, 1))
    return out


def composite(x: torch.Tensor, x_known: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """GaMO composite: x = m*x + (1-m)*x_known  (keep generation in the void,
    inject known/observed content elsewhere). m broadcasts over channels."""
    return m * x + (1.0 - m) * x_known


def trigger_steps(num_steps: int, fracs=(0.30, 0.50, 0.70)) -> dict:
    """Map denoise fractions -> step indices (0-based). Returns {step_index: which}
    where `which` indexes into progressive_masks (0=widest ... )."""
    d = {}
    for wi, fr in enumerate(fracs):
        si = int(round(fr * num_steps))
        si = max(0, min(num_steps - 1, si))
        d[si] = wi
    return d


# ---------------------------------------------------------------------------
# Feathered pixel paste-back (hard gate integrity)
# ---------------------------------------------------------------------------

def feather_paste_back(orig_rgb: np.ndarray, gen_rgb: np.ndarray,
                       unobs_pixmask: np.ndarray, inset: float = 0.6,
                       feather: float = 0.08) -> np.ndarray:
    """Composite the generated fill back over the ORIGINAL render so ONLY the void
    changes and the boundary is feathered.

    orig_rgb, gen_rgb : (H,W,3) uint8 or float. unobs_pixmask : (H,W) bool (True =
    void = take generation). inset : erode the void by inset*radius before feather
    (keep the generation slightly INSIDE the void so the observed boundary is
    untouched). feather : gaussian blur sigma as a fraction of the void's bbox
    diagonal. Interior void pixels are byte-identical to the generation; observed
    pixels far from the boundary are byte-identical to the original.
    """
    import cv2
    H, W = unobs_pixmask.shape
    orig = orig_rgb.astype(np.float32)
    gen = gen_rgb.astype(np.float32)
    m = unobs_pixmask.astype(np.uint8)
    if m.sum() == 0:
        return orig_rgb.astype(np.uint8)
    ys, xs = np.nonzero(m)
    diag = float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    # inset: erode the void so generation stays inside it (observed boundary safe)
    er = max(1, int(round(inset * feather * diag)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er + 1, 2 * er + 1))
    m_inset = cv2.erode(m, ker)
    # feathered alpha: blur the inset mask
    sig = max(1.0, feather * diag)
    ksz = int(sig * 4) | 1
    a = cv2.GaussianBlur(m_inset.astype(np.float32), (ksz, ksz), sig)[..., None]
    comp = a * gen + (1.0 - a) * orig
    return np.clip(comp, 0, 255).astype(np.uint8)
