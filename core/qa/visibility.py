"""Visibility masks — which pixels of a held-out view did the reconstruction
actually OBSERVE, vs. which are unobserved / never-seen (behind the object, in
gaps the cameras never covered).

Primary signal = accumulated render alpha (OGGSplat protocol [V], EXECUTION_PLAN
WP-Q1 comp.1): a pixel is UNOBSERVED where the splat renders little/no opacity.
    unobserved = (alpha < 0.3)
                 |> cv2.MORPH_CLOSE  25x25 rect   (fill small holes in the
                                                    unobserved region)
                 |> cv2.dilate       5x5 rect x3  (grow it — be conservative:
                                                    only score confidently-seen px)
    observed   = ~unobserved

The morphology GROWS the unobserved region, so masked metrics are computed only
on pixels the model is confident it saw — never on hallucinated fill.

Optional secondary signal (default OFF, `--vis_view_count`): flag pixels whose
dominant Gaussians were seen by fewer than k training views (undersampled), lifted
from the SplatSuRe projection machinery in `_infusion/val_splatsure.py`.  It is
combined by UNION with the alpha signal (more conservative).  Heavy (one full
projection per training view) — hence off by default.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass
class VisibilityConfig:
    """OGGSplat protocol parameters (defaults ARE the protocol)."""
    alpha_thresh: float = 0.3
    close_ksize: int = 25
    dilate_ksize: int = 5
    dilate_iters: int = 3
    rasterize_mode: str = "antialiased"
    # secondary per-Gaussian view-count signal (experimental, default off)
    use_view_count: bool = False
    view_count_k: int = 3          # a Gaussian is "well-observed" iff seen by >= k views
    view_count_frac: float = 0.5   # pixel undersampled iff >this accumulated weight
    #                                from under-observed Gaussians


def render_alpha(splats, viewmat, K, width, height, sh_degree, device,
                 rasterize_mode="antialiased"):
    """Render accumulated alpha for one view. Returns (H, W) float tensor in [0,1].

    Uses the project renderer read-only. gsplat `rasterization` returns
    render_alphas of shape (C, H, W, 1); we take view 0, channel 0.
    """
    from core.training.gsplat_trainer import render_splats
    bg = torch.zeros(3, device=device)
    _, alphas, _ = render_splats(
        splats, viewmat, K, int(width), int(height), sh_degree, bg, rasterize_mode,
    )
    return alphas[0, ..., 0]


def build_visibility_mask(alpha, cfg: VisibilityConfig = None,
                          secondary_unobserved=None):
    """Build the observed/unobserved masks from a rendered alpha map.

    Parameters
    ----------
    alpha : (H, W) torch.Tensor or np.ndarray in [0, 1]
    cfg   : VisibilityConfig (uses protocol defaults if None)
    secondary_unobserved : optional (H, W) bool array from the per-Gaussian
        view-count signal; UNION'd into the unobserved region.

    Returns
    -------
    dict: observed (H,W bool), unobserved (H,W bool), alpha (H,W float32),
          observed_pct (float, 0..100)
    """
    if cfg is None:
        cfg = VisibilityConfig()

    a = alpha.detach().cpu().numpy() if torch.is_tensor(alpha) else np.asarray(alpha)
    a = a.astype(np.float32)

    unobs = (a < cfg.alpha_thresh).astype(np.uint8)

    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.close_ksize, cfg.close_ksize))
    unobs = cv2.morphologyEx(unobs, cv2.MORPH_CLOSE, kc)

    kd = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.dilate_ksize, cfg.dilate_ksize))
    # dilate 5x5 x3 == one dilate call with iterations=3 (associative)
    unobs = cv2.dilate(unobs, kd, iterations=cfg.dilate_iters)

    unobs_bool = unobs.astype(bool)
    if secondary_unobserved is not None:
        unobs_bool = unobs_bool | np.asarray(secondary_unobserved, dtype=bool)

    observed = ~unobs_bool
    observed_pct = 100.0 * float(observed.mean())
    return {
        "observed": observed,
        "unobserved": unobs_bool,
        "alpha": a,
        "observed_pct": observed_pct,
    }


# ---------------------------------------------------------------------------
# Optional secondary signal — per-Gaussian training-view counts (SplatSuRe-style)
# Lifted from _infusion/val_splatsure.py (fully_fused_projection based). Default off.
# ---------------------------------------------------------------------------

def per_gaussian_obs_counts(splats, data, device):
    """Number of TRAINING views that observe each Gaussian (screen radius > 0).

    Mirrors the projection loop in val_splatsure.py (eps2d=0 to exclude the 0.3 AA
    dilation, exactly as the SplatSuRe protocol specifies). Returns int32 (N,).
    Heavy: one full projection of all Gaussians per training view.
    """
    from gsplat.cuda._wrapper import fully_fused_projection

    means = splats["means"].float()
    quats = splats["quats"].float()
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-12)
    scales = torch.exp(splats["scales"].float())
    opac = torch.sigmoid(splats["opacities"].float())
    N = means.shape[0]
    n_obs = torch.zeros(N, dtype=torch.int32, device=device)

    with torch.no_grad():
        for j in sorted(int(x) for x in data["train_indices"]):
            cid = data["camera_ids"][j]
            K = torch.from_numpy(np.asarray(data["Ks_dict"][cid])).float().to(device)[None]
            W, H = data["imsize_dict"][cid]
            vm = torch.linalg.inv(
                torch.from_numpy(data["camtoworlds"][j]).float().to(device))[None]
            out = fully_fused_projection(
                means, None, quats, scales, vm, K, int(W), int(H),
                eps2d=0.0, near_plane=0.01, far_plane=1e10, packed=False, opacities=opac,
            )
            radii = out[0][0]
            if radii.ndim == 2:
                radii = radii.max(dim=-1).values
            n_obs += (radii > 0).int()
    return n_obs


def render_undersampled_map(splats, n_obs, viewmat, K, width, height, device, k=3):
    """Rasterize an "under-observed" map for one view: per pixel, the accumulated
    fraction of splat weight coming from Gaussians seen by < k training views.
    Returns (H, W) float tensor in [0, 1]. Uses gsplat with per-Gaussian colors.
    """
    from gsplat import rasterization

    means = splats["means"].float()
    quats = splats["quats"].float()
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-12)
    scales = torch.exp(splats["scales"].float())
    opac = torch.sigmoid(splats["opacities"].float())
    under = (n_obs < k).float().to(device)                 # 1 = under-observed
    colors = under[:, None].expand(-1, 3).contiguous()     # (N,3)
    with torch.no_grad():
        img, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac,
            colors=colors[None], viewmats=viewmat[None], Ks=K[None],
            width=int(width), height=int(height), sh_degree=None, render_mode="RGB",
        )
    return img[0, ..., 0]
