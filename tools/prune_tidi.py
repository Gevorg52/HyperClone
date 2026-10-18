"""TIDI-GS style Cross-View Evidence, Blending Importance & 3D Isolation Floater Pruner (WP-C3 Part 1).

Why `tools/prune_ply.py` (`sigmoid(opacity) < 0.05`) is not enough:
1. Off-path floaters sitting in front of surfaces or outside the main camera frustum often
   have moderate opacity (`0.05 .. 0.35`), so they survive raw opacity thresholding (`op < 0.05`)
   and create translucent veils/floaters when a viewer leaves the capture trajectory.
2. Under `--antialiased` (Mip-Splatting) + `--scale_reg 0.01` + `--opacity_reg 0`, hundreds of
   thousands of sub-pixel Gaussians shrink until their 2D AA-compensated opacity falls below
   `1/255` (`radii == 0` in `gsplat`), receiving zero gradient while retaining raw
   `sigmoid(opacity) >= 0.05`. They bloat PLY files by 25-65% and deposit off-path haze.

This tool computes three complementary per-Gaussian evidence signals across the training cameras:
1. **Cross-view visibility, depth consistency & exact blending importance (`view_count`, $M_i$, $I_i$):**
   - For each training view, runs `gsplat.rasterization(..., render_mode="RGB+ED", packed=False)`
     with a 1-channel unit dummy color `c = ones((N, 1), requires_grad=True)`.
   - Because $R_{p,0}(c, o) = \\sum_i T_{p,i}(o) \\alpha_{p,i}(o) c_{i,0}$ equals the accumulated
     alpha $A_p(o)$ at $c_{i,0} = 1$, a single backward pass on `renders[..., 0].sum()` yields:
     * `colors.grad[:, 0]` = $\\sum_{p} T_{p,i} \\alpha_{p,i}$ (the EXACT forward alpha-blending
       weight of Gaussian $i$ across all pixels in the view), and
     * `opacities * opacities.grad` = $\\sum_{p} \\alpha_{p,i} T_{p,i} \\prod_{j > i}(1 - \\alpha_{p,j})$
       (the exact accumulated-alpha sensitivity to Gaussian $i$).
   - Simultaneously, `renders[..., 1]` provides the rendered expected depth $D_{\\text{exp}}(u, v)$,
     and `info` provides per-Gaussian 2D radii, projected centers $(u_i, v_i)$, and camera depths $z_i$.
   - A Gaussian with $z_i < (1 - \\delta) D_{\\text{exp}}(u_i, v_i)$ in front of an opaque surface
     ($A(u_i, v_i) \\ge \\alpha_{\\min}$) is flagged as an **in-front depth-inconsistent** observation,
     whereas $|z_i - D_{\\text{exp}}(u_i, v_i)| \\le \\tau_z D_{\\text{exp}}(u_i, v_i)$ counts as genuine
     **multi-view surface support**.
2. **3D Spatial Isolation (k-NN distance / SOR in 3D & robust bbox extent):**
   - Computes mean distance to the $k=8$ nearest neighbors via `scipy.spatial.cKDTree` (or GPU
     chunked `torch.cdist` for small clouds), flagging spatial outliers (`knn_dist > mean + s * std`
     and `knn_dist > 0.02 * scene_scale`) as well as `> 3x` robust `[p1, p99]` bbox outliers.
3. **Combined TIDI Prune Rule:**
   - Drops `opacity < min_opacity` (default `0.05`).
   - Drops `> bbox_mult` (default `3.0`) outliers (protecting only high-importance distant sky-dome
     splats up to `hard_bbox_mult=10.0` with `importance_total >= bg_protect_importance`).
   - Drops low-evidence Gaussians with fewer than `min_views` active training views
     ($I_{v,i} > \\epsilon$) or `importance_total < min_importance`.
   - Drops in-front depth-inconsistent floaters (`infront_views > surface_views`, low surface
     support, and low blending importance / moderate opacity).
   - Drops 3D spatially isolated floaters (`knn_dist` above SOR / `0.02 * scene_scale` threshold
     without strong multi-view importance).

CLI usage:
  python tools/prune_tidi.py --ckpt <ckpt.pt> --data_dir <data_dir> --factor 2 \\
      --out <pruned.ply> [--out_ckpt <pruned.pt>] [--stats_json <stats.json>]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.prune_ply import read_ckpt, read_gsplat_ply, splat_stats  # noqa: E402


@dataclass
class TidiConfig:
    """Configuration for TIDI cross-view evidence, importance, and 3D isolation pruning."""
    min_opacity: float = 0.05
    bbox_mult: float = 3.0
    hard_bbox_mult: float = 10.0
    bg_protect_importance: float = 100.0
    min_views: int = 2
    active_eps: float = 0.02
    min_importance: float = 0.10
    delta_infront: float = 0.10
    tau_surface: float = 0.08
    alpha_surface_min: float = 0.5
    max_floater_surface_views: int = 2
    infront_max_importance: float = 15.0
    infront_max_opacity: float = 0.35
    sor_k: int = 8
    sor_std: float = 2.0
    sor_scale_frac: float = 0.02
    sor_max_importance: float = 4.0
    sor_hard_max_importance: float = 20.0
    eval_scale: float = 1.0
    rasterize_mode: str = "antialiased"

    @classmethod
    def from_preset(cls, preset: str = "moderate", **overrides) -> "TidiConfig":
        if preset == "conservative":
            cfg = cls(
                min_opacity=0.05,
                bbox_mult=3.0,
                hard_bbox_mult=10.0,
                bg_protect_importance=50.0,
                min_views=2,
                active_eps=0.01,
                min_importance=0.05,
                infront_max_importance=10.0,
                sor_max_importance=2.0,
                sor_hard_max_importance=12.0,
            )
        elif preset == "moderate":
            cfg = cls()
        elif preset == "aggressive":
            cfg = cls(
                min_opacity=0.05,
                bbox_mult=3.0,
                hard_bbox_mult=10.0,
                bg_protect_importance=200.0,
                min_views=2,
                active_eps=0.05,
                min_importance=0.25,
                infront_max_importance=25.0,
                sor_max_importance=15.0,
                sor_hard_max_importance=50.0,
            )
        else:
            raise ValueError(f"Unknown TIDI preset: {preset}")
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


def compute_view_blending_weights(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    rasterize_mode: str = "antialiased",
    compute_alpha_grad: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, dict]:
    """Compute exact per-Gaussian blending weights and expected depth for a single camera view.

    Mathematical identity:
    Let $R_{p,0}(c, o) = \\sum_{i} T_{p,i}(o) \\alpha_{p,i}(o) c_{i,0}$ be the rendered scalar
    channel at pixel $p$ with unit per-Gaussian color $c_{i,0} = 1$.
    1. By linearity in $c$, $\\frac{\\partial \\sum_p R_{p,0}}{\\partial c_{i,0}} = \\sum_p T_{p,i} \\alpha_{p,i}$
       is the exact total blending weight of Gaussian $i$ across all pixels in the view.
    2. Because $R_{p,0}(1, o) = A_p(o)$, differentiating the same scalar wrt activated opacity $o_i$
       yields $o_i \\frac{\\partial \\sum_p A_p}{\\partial o_i} = \\sum_p \\alpha_{p,i} T_{p,i} \\prod_{j > i} (1 - \\alpha_{p,j})$.
    3. Setting `render_mode="RGB+ED"` appends the expected depth $D_{\\text{exp}}(u, v)$ as channel 1
       in the same forward pass.

    Args:
        means: (N, 3) float32 world-space Gaussian centers.
        quats: (N, 4) float32 normalized quaternions.
        scales: (N, 3) float32 activated (linear, exp) 3D scales.
        opacities: (N,) float32 activated (sigmoid) opacities in (0, 1).
        viewmat: (4, 4) or (1, 4, 4) world-to-camera matrix.
        K: (3, 3) or (1, 3, 3) camera intrinsics matrix.
        width: image width in pixels.
        height: image height in pixels.
        rasterize_mode: "antialiased" or "classic".
        compute_alpha_grad: if True, also compute `opacities * opacities.grad`.

    Returns:
        blending_weights: (N,) float32 exact $\\sum_{u,v} T_{p,i} \\alpha_{p,i} \\ge 0$.
        alpha_grad_weights: (N,) float32 $o_i \\frac{\\partial \\sum A}{\\partial o_i}$ if requested, else None.
        expected_depth: (H, W) float32 rendered expected depth $D_{\\text{exp}}(u, v)$.
        alpha: (H, W) float32 rendered accumulated opacity $A(u, v)$.
        info: dict from `gsplat.rasterization` (`radii`, `means2d`, `depths`, ...).
    """
    import gsplat

    N = means.shape[0]
    device = means.device
    vm = viewmat[None] if viewmat.ndim == 2 else viewmat
    Ks = K[None] if K.ndim == 2 else K

    means_d = means.detach()
    quats_d = quats.detach()
    scales_d = scales.detach()
    opac_d = opacities.detach().requires_grad_(bool(compute_alpha_grad))
    colors = torch.ones((N, 1), dtype=torch.float32, device=device, requires_grad=True)

    with torch.enable_grad():
        renders, alphas, info = gsplat.rasterization(
            means=means_d,
            quats=quats_d,
            scales=scales_d,
            opacities=opac_d,
            colors=colors,
            viewmats=vm,
            Ks=Ks,
            width=int(width),
            height=int(height),
            render_mode="RGB+ED",
            rasterize_mode=rasterize_mode,
            near_plane=0.01,
            far_plane=1e10,
            packed=False,
        )
        loss = renders[..., 0].sum()
        loss.backward()

    blending_weights = colors.grad.squeeze(-1).detach().clamp_min(0.0)
    alpha_grad_weights = None
    if compute_alpha_grad and opac_d.grad is not None:
        alpha_grad_weights = (opac_d.detach() * opac_d.grad.detach()).clamp_min(0.0)

    expected_depth = renders[0, ..., 1].detach()
    alpha_map = alphas[0, ..., 0].detach()
    return blending_weights, alpha_grad_weights, expected_depth, alpha_map, info


def compute_knn_distances(
    means: np.ndarray | torch.Tensor,
    k: int = 8,
    subset_mask: Optional[np.ndarray | torch.Tensor] = None,
) -> tuple[np.ndarray, float, float, float, float]:
    """Compute mean distance to the k nearest 3D neighbors using cKDTree.

    If `subset_mask` is provided (e.g. cross-view-supported Gaussians), the KD-tree is built on
    the active subset so that dead/unseen dust does not artificially shield isolated floaters,
    and distances are queried for all points against that tree.
    """
    pts = means.detach().cpu().numpy() if torch.is_tensor(means) else np.asarray(means, dtype=np.float32)
    N = pts.shape[0]
    if N <= k + 1:
        zeros = np.zeros(N, dtype=np.float32)
        return zeros, 0.0, 0.0, 0.0, 0.0

    if subset_mask is not None:
        sm = subset_mask.detach().cpu().numpy().astype(bool) if torch.is_tensor(subset_mask) else np.asarray(subset_mask, dtype=bool)
        ref_pts = pts[sm] if int(sm.sum()) > k + 1 else pts
    else:
        sm = None
        ref_pts = pts

    tree = cKDTree(ref_pts)
    if sm is not None and int(sm.sum()) > k + 1:
        # For points in the reference subset, query k+1 and drop self (col 0).
        d_sub, _ = tree.query(ref_pts, k=k + 1, workers=-1)
        knn_sub = d_sub[:, 1:].mean(axis=1).astype(np.float32)
        knn_all = np.zeros(N, dtype=np.float32)
        knn_all[sm] = knn_sub
        if int((~sm).sum()) > 0:
            d_other, _ = tree.query(pts[~sm], k=k, workers=-1)
            knn_all[~sm] = (d_other.mean(axis=1) if d_other.ndim == 2 else d_other).astype(np.float32)
        ref_vals = knn_sub
    else:
        d_all, _ = tree.query(pts, k=k + 1, workers=-1)
        knn_all = d_all[:, 1:].mean(axis=1).astype(np.float32)
        ref_vals = knn_all

    mean_d = float(ref_vals.mean())
    std_d = float(ref_vals.std())
    med_d = float(np.median(ref_vals))
    mad_d = float(np.median(np.abs(ref_vals - med_d)) * 1.4826)
    return knn_all, mean_d, std_d, med_d, mad_d


def compute_tidi_scores(
    splats: dict,
    data: dict,
    cfg: Optional[TidiConfig] = None,
    device: str | torch.device = "cuda:0",
    pose_embeds: Optional[torch.Tensor] = None,
) -> dict:
    """Compute per-Gaussian TIDI cross-view evidence, importance, depth consistency, and 3D isolation.

    Args:
        splats: trainer splats dict (`means`, `quats`, `scales`, `opacities`, `sh0`, `shN`).
        data: COLMAP dataset dict from `load_colmap_data`.
        cfg: `TidiConfig` thresholds and parameters.
        device: CUDA device for rasterization.
        pose_embeds: optional `(num_images, 9)` tensor from `ckpt["pose_embeds"]`.

    Returns:
        dict of per-Gaussian score tensors (on CPU) and summary scalars.
    """
    if cfg is None:
        cfg = TidiConfig()
    dev = torch.device(device)

    means = splats["means"].detach().float().to(dev)
    quats = splats["quats"].detach().float().to(dev)
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-12)
    scales = torch.exp(splats["scales"].detach().float().to(dev))
    opac = torch.sigmoid(splats["opacities"].detach().float().reshape(-1).to(dev))
    N = means.shape[0]

    # Optional learned training camera pose offsets
    pose_mod = None
    if pose_embeds is not None:
        from core.training.gsplat_trainer import CameraOptModule
        num_imgs = len(data["image_names"])
        if pose_embeds.shape[0] == num_imgs:
            pose_mod = CameraOptModule(num_imgs).to(dev)
            pose_mod.embeds.weight.data.copy_(pose_embeds.detach().float().to(dev))

    proj_views = torch.zeros(N, dtype=torch.int32, device=dev)
    active_views = torch.zeros(N, dtype=torch.int32, device=dev)
    surface_views = torch.zeros(N, dtype=torch.int32, device=dev)
    infront_views = torch.zeros(N, dtype=torch.int32, device=dev)
    importance_total = torch.zeros(N, dtype=torch.float32, device=dev)
    importance_max = torch.zeros(N, dtype=torch.float32, device=dev)

    train_indices = sorted(int(i) for i in data["train_indices"])
    ds = float(cfg.eval_scale)
    area_scale = 1.0 / max(ds * ds, 1e-6)

    for idx in train_indices:
        cid = data["camera_ids"][idx]
        W0, H0 = data["imsize_dict"][cid]
        W = max(1, int(round(W0 * ds)))
        H = max(1, int(round(H0 * ds)))
        K = data["Ks_dict"][cid].astype(np.float32).copy()
        if ds != 1.0:
            K[0, :] *= (W / W0)
            K[1, :] *= (H / H0)
        K_t = torch.from_numpy(K).to(dev)
        c2w = torch.from_numpy(data["camtoworlds"][idx]).float().to(dev)
        if pose_mod is not None:
            with torch.no_grad():
                c2w = pose_mod(c2w, idx)
        vm = torch.linalg.inv(c2w)

        w, _, ed, al, info = compute_view_blending_weights(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opac,
            viewmat=vm,
            K=K_t,
            width=W,
            height=H,
            rasterize_mode=cfg.rasterize_mode,
            compute_alpha_grad=False,
        )

        with torch.no_grad():
            w_scaled = w * area_scale
            importance_total += w_scaled
            importance_max = torch.maximum(importance_max, w_scaled)
            active_views += (w_scaled > cfg.active_eps).int()

            rad = info["radii"][0]
            if rad.ndim == 2:
                rad = rad.max(dim=-1).values
            m2d = info["means2d"][0]
            z = info["depths"][0]

            # A Gaussian has footprint support in this view if radii > 0
            has_rad = (rad > 0) & (z > 0.01)
            proj_views += has_rad.int()

            u = m2d[:, 0].long()
            v = m2d[:, 1].long()
            in_bounds = has_rad & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u_c = u.clamp(0, W - 1)
            v_c = v.clamp(0, H - 1)
            ed_at = ed[v_c, u_c]
            al_at = al[v_c, u_c]

            valid_surf = in_bounds & (al_at >= cfg.alpha_surface_min) & (ed_at > 0.01)
            is_infront = valid_surf & (z < ed_at * (1.0 - cfg.delta_infront))
            is_surface = valid_surf & ((z - ed_at).abs() <= ed_at * cfg.tau_surface)
            infront_views += is_infront.int()
            surface_views += is_surface.int()

    # 3D Spatial Isolation (k-NN & robust bbox)
    # Build KD-tree on the cross-view-supported subset so dead/unseen dust cannot shield isolated floaters
    means_np = means.cpu().numpy()
    lo, hi = np.percentile(means_np, [1, 99], axis=0)
    centre = (lo + hi) / 2.0
    half = np.maximum((hi - lo) / 2.0, 1e-9)
    rel_bbox_radius = torch.from_numpy((np.abs(means_np - centre) / half).max(axis=1)).float()

    supported_mask = (
        (opac >= cfg.min_opacity)
        & (active_views >= cfg.min_views)
        & (importance_total >= cfg.min_importance)
    ).cpu().numpy()
    knn_np, knn_mean, knn_std, knn_med, knn_mad = compute_knn_distances(
        means_np, k=cfg.sor_k, subset_mask=supported_mask,
    )
    scene_scale = float(data.get("scene_scale", 1.0))
    sor_threshold = float(knn_mean + cfg.sor_std * knn_std)
    sor_002_threshold = float(cfg.sor_scale_frac * scene_scale)

    return {
        "opacities": opac.cpu(),
        "proj_views": proj_views.cpu(),
        "active_views": active_views.cpu(),
        "surface_views": surface_views.cpu(),
        "infront_views": infront_views.cpu(),
        "importance_total": importance_total.cpu(),
        "importance_max": importance_max.cpu(),
        "knn_dist": torch.from_numpy(knn_np),
        "rel_bbox_radius": rel_bbox_radius,
        "knn_mean": knn_mean,
        "knn_std": knn_std,
        "knn_median": knn_med,
        "knn_mad": knn_mad,
        "sor_threshold": sor_threshold,
        "sor_002_threshold": sor_002_threshold,
        "scene_scale": scene_scale,
        "num_train_views": len(train_indices),
    }


def tidi_keep_mask(scores: dict, cfg: Optional[TidiConfig] = None) -> tuple[torch.Tensor, dict]:
    """Apply the combined TIDI prune rule to per-Gaussian evidence scores.

    Returns:
        keep: (N,) bool tensor of kept Gaussians.
        reasons: dict with per-criterion drop counts and summary fractions.
    """
    if cfg is None:
        cfg = TidiConfig()

    opac = scores["opacities"]
    active_views = scores["active_views"]
    surface_views = scores["surface_views"]
    infront_views = scores["infront_views"]
    imp_total = scores["importance_total"]
    knn_dist = scores["knn_dist"]
    rel_radius = scores["rel_bbox_radius"]
    sor_thr = float(scores["sor_threshold"])
    sor_002_thr = float(scores["sor_002_threshold"])

    N = int(opac.shape[0])
    keep = torch.ones(N, dtype=torch.bool)
    reasons = {}

    # 1. Raw opacity floor
    if cfg.min_opacity > 0:
        pass_op = opac >= cfg.min_opacity
        reasons["dropped_low_opacity"] = int((~pass_op).sum())
        keep &= pass_op
    else:
        reasons["dropped_low_opacity"] = 0

    # 2. Bounding-box extent outlier (with optional sky-dome protection for high-evidence background splats)
    if cfg.bbox_mult > 0:
        if cfg.bg_protect_importance > 0 and cfg.hard_bbox_mult > cfg.bbox_mult:
            pass_bbox = (rel_radius <= cfg.bbox_mult) | (
                (rel_radius <= cfg.hard_bbox_mult)
                & (active_views >= max(cfg.min_views, 5))
                & (imp_total >= cfg.bg_protect_importance)
            )
        else:
            pass_bbox = rel_radius <= cfg.bbox_mult
        reasons["dropped_bbox_outlier"] = int((keep & (~pass_bbox)).sum())
        keep &= pass_bbox
    else:
        reasons["dropped_bbox_outlier"] = 0

    # 3. Cross-view visibility & blending importance support
    pass_views = (active_views >= cfg.min_views) & (imp_total >= cfg.min_importance)
    reasons["dropped_low_crossview_evidence"] = int((keep & (~pass_views)).sum())
    keep &= pass_views

    # 4. In-front depth-inconsistent floaters (translucent AND low total contribution)
    is_infront_floater = (
        (infront_views > surface_views)
        & (surface_views <= cfg.max_floater_surface_views)
        & (imp_total < cfg.infront_max_importance)
        & (opac < cfg.infront_max_opacity)
    )
    reasons["dropped_infront_depth_floater"] = int((keep & is_infront_floater).sum())
    keep &= ~is_infront_floater

    # 5. 3D spatial isolation (k-NN / SOR outliers lacking strong multi-view importance)
    is_spatial_isolated = (
        ((knn_dist > sor_002_thr) & (imp_total < cfg.sor_max_importance))
        | ((knn_dist > sor_thr) & (imp_total < cfg.sor_hard_max_importance))
    )
    reasons["dropped_3d_isolated_floater"] = int((keep & is_spatial_isolated).sum())
    keep &= ~is_spatial_isolated

    reasons["kept"] = int(keep.sum())
    reasons["dropped_total"] = int(N - keep.sum())
    reasons["dropped_beyond_op05"] = int(
        reasons["dropped_bbox_outlier"]
        + reasons["dropped_low_crossview_evidence"]
        + reasons["dropped_infront_depth_floater"]
        + reasons["dropped_3d_isolated_floater"]
    )
    reasons["kept_frac"] = round(float(keep.float().mean()), 6) if N > 0 else 0.0
    return keep, reasons


def prune_ckpt_tidi(
    source_path: str | Path,
    data_dir: str | Path,
    factor: int = 2,
    holdout_list: Optional[str | Path] = None,
    test_every: int = 8,
    normalize: bool = True,
    cfg: Optional[TidiConfig] = None,
    out_ply: Optional[str | Path] = None,
    out_ckpt: Optional[str | Path] = None,
    stats_json: Optional[str | Path] = None,
    device: str = "cuda:0",
) -> tuple[dict, dict, dict]:
    """Load a checkpoint or PLY, compute TIDI scores, prune floaters, and optionally export PLY/ckpt.

    Returns:
        kept_splats: pruned splats dict (CPU tensors).
        reasons: drop breakdown dict.
        payload: full stats payload (also written to `stats_json` if specified).
    """
    import gsplat
    from core.training.gsplat_trainer import load_colmap_data

    if cfg is None:
        cfg = TidiConfig()

    t0 = time.time()
    src = Path(source_path)
    pose_embeds = None
    raw_ckpt = None
    if src.suffix.lower() == ".ply":
        splats = read_gsplat_ply(src)
    else:
        raw_ckpt = torch.load(str(src), map_location="cpu", weights_only=True)
        s = {k: v.float() for k, v in raw_ckpt["splats"].items()}
        n = s["means"].shape[0]
        splats = {
            "means": s["means"],
            "scales": s["scales"],
            "quats": s["quats"],
            "opacities": s["opacities"].reshape(n),
            "sh0": s["sh0"].reshape(n, 1, 3),
            "shN": s["shN"].reshape(n, -1, 3),
        }
        if "pose_embeds" in raw_ckpt:
            pose_embeds = raw_ckpt["pose_embeds"].float()

    before = splat_stats(splats)
    data = load_colmap_data(
        str(data_dir),
        factor=int(factor),
        test_every=int(test_every),
        normalize=bool(normalize),
        holdout_list=str(holdout_list) if holdout_list else None,
    )

    t_score = time.time()
    scores = compute_tidi_scores(splats, data, cfg=cfg, device=device, pose_embeds=pose_embeds)
    score_sec = time.time() - t_score

    keep, reasons = tidi_keep_mask(scores, cfg=cfg)
    kept_splats = {k: v[keep] for k, v in splats.items()}
    after = splat_stats(kept_splats)

    # Fixed-threshold 3D isolation diagnostics before vs after (using the same sor_002 & op05 SOR thresholds)
    alive_before = scores["opacities"] >= cfg.min_opacity
    sor_002_thr = scores["sor_002_threshold"]
    sor_thr = scores["sor_threshold"]
    iso_stats = {
        "sor_002_threshold": round(sor_002_thr, 6),
        "sor_op05_threshold": round(sor_thr, 6),
        "unpruned_knn_gt_sor002_count": int((scores["knn_dist"] > sor_002_thr).sum()),
        "op05_knn_gt_sor002_count": int((scores["knn_dist"][alive_before] > sor_002_thr).sum()),
        "tidi_knn_gt_sor002_count": int((scores["knn_dist"][keep] > sor_002_thr).sum()),
        "op05_knn_gt_sor_thr_count": int((scores["knn_dist"][alive_before] > sor_thr).sum()),
        "tidi_knn_gt_sor_thr_count": int((scores["knn_dist"][keep] > sor_thr).sum()),
    }

    payload = {
        "tool": "tools/prune_tidi.py",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(src.resolve()),
        "data_dir": str(Path(data_dir).resolve()),
        "factor": int(factor),
        "config": asdict(cfg),
        "score_runtime_s": round(score_sec, 3),
        "before": before,
        "pruned": reasons,
        "isolation_diagnostics": iso_stats,
        "after": after,
    }

    if out_ply:
        out_p = Path(out_ply)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        gsplat.export_splats(
            means=kept_splats["means"],
            scales=kept_splats["scales"],
            quats=kept_splats["quats"],
            opacities=kept_splats["opacities"],
            sh0=kept_splats["sh0"],
            shN=kept_splats["shN"],
            format="ply",
            save_to=str(out_p),
        )
        payload["out_ply"] = str(out_p.resolve())
        payload["out_ply_bytes"] = out_p.stat().st_size

    if out_ckpt:
        out_c = Path(out_ckpt)
        out_c.parent.mkdir(parents=True, exist_ok=True)
        ckpt_out = {
            "step": raw_ckpt.get("step", 30000) if raw_ckpt is not None else 30000,
            "splats": {
                "means": kept_splats["means"],
                "scales": kept_splats["scales"],
                "quats": kept_splats["quats"],
                "opacities": kept_splats["opacities"],
                "sh0": kept_splats["sh0"],
                "shN": kept_splats["shN"],
            },
        }
        if pose_embeds is not None:
            ckpt_out["pose_embeds"] = pose_embeds
        torch.save(ckpt_out, str(out_c))
        payload["out_ckpt"] = str(out_c.resolve())

    payload["total_runtime_s"] = round(time.time() - t0, 3)

    if stats_json:
        st_p = Path(stats_json)
        st_p.parent.mkdir(parents=True, exist_ok=True)
        st_p.write_text(json.dumps(payload, indent=2), encoding="utf8")

    return kept_splats, reasons, payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python tools/prune_tidi.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="TIDI-GS style Cross-View Evidence, Importance & 3D Isolation Floater Pruner",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="gsplat_trainer checkpoint (.pt) to prune")
    src.add_argument("--ply", help="gsplat PLY (.ply) to prune")
    p.add_argument("--data_dir", required=True, help="COLMAP dataset directory for training camera evidence")
    p.add_argument("--factor", type=int, default=2, help="image downsample factor matching training")
    p.add_argument("--holdout_list", default=None, help="optional holdout image list (for 360 rigs)")
    p.add_argument("--test_every", type=int, default=8, help="every-Nth test split when holdout_list is not set")
    p.add_argument("--out", help="output pruned PLY path")
    p.add_argument("--out_ckpt", help="optional output pruned .pt checkpoint path")
    p.add_argument("--stats_json", help="write detailed before/after JSON stats here")
    p.add_argument("--stats_only", action="store_true", help="compute scores and stats without writing PLY")
    p.add_argument("--preset", choices=["conservative", "moderate", "aggressive"], default="moderate")
    p.add_argument("--min_opacity", type=float, default=None, help="raw sigmoid(opacity) floor (default 0.05)")
    p.add_argument("--bbox_mult", type=float, default=None, help="robust [p1, p99] bbox half-extent multiplier (default 3.0)")
    p.add_argument("--min_views", type=int, default=None, help="minimum active training views (default 2)")
    p.add_argument("--active_eps", type=float, default=None, help="per-view blending weight threshold (default 0.02)")
    p.add_argument("--min_importance", type=float, default=None, help="minimum total blending weight across train views")
    p.add_argument("--eval_scale", type=float, default=None, help="resolution scale for rasterization pass (default 1.0)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)

    if not args.stats_only and not args.out and not args.out_ckpt:
        p.error("--out or --out_ckpt is required unless --stats_only")

    cfg = TidiConfig.from_preset(
        args.preset,
        min_opacity=args.min_opacity,
        bbox_mult=args.bbox_mult,
        min_views=args.min_views,
        active_eps=args.active_eps,
        min_importance=args.min_importance,
        eval_scale=args.eval_scale,
    )

    source = args.ckpt or args.ply
    _, reasons, payload = prune_ckpt_tidi(
        source_path=source,
        data_dir=args.data_dir,
        factor=args.factor,
        holdout_list=args.holdout_list,
        test_every=args.test_every,
        cfg=cfg,
        out_ply=None if args.stats_only else args.out,
        out_ckpt=None if args.stats_only else args.out_ckpt,
        stats_json=args.stats_json,
        device=args.device,
    )

    b_n = payload["before"]["num_gaussians"]
    print(
        f"[prune_tidi] {source}: {b_n:,} -> {reasons['kept']:,} kept ({reasons['kept_frac']*100:.1f}%), "
        f"dropped {reasons['dropped_total']:,} "
        f"[low_op={reasons['dropped_low_opacity']:,}, bbox={reasons['dropped_bbox_outlier']:,}, "
        f"low_ev={reasons['dropped_low_crossview_evidence']:,}, "
        f"infront={reasons['dropped_infront_depth_floater']:,}, "
        f"iso3d={reasons['dropped_3d_isolated_floater']:,}] "
        f"in {payload['total_runtime_s']:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
