"""WP-D1 delivery pass (+ WP-S2 synthesis: delivery-time object extraction).

Turns a trained master splat (ckpt or PLY) into a clean, decimated, viewer-ready
deliverable (SOG + compressed PLY), with an optional object-extraction mode that
carves the scanned object out of a full-scene model at delivery time.

Two modes (default = full-scene; `--extract_object` opts in):

  FULL-SCENE (production default, behavior unchanged when the flag is off):
    prune (opacity + statistical-outlier)                       [core.training.prune]
      -> splat-transform  --filter-nan --filter-floaters --filter-cluster
                          --decimate P%                          [PlayCanvas, GPU]
      -> self-refit R steps on the original views (standard)     [gsplat_trainer]
      -> splat-transform  --filter-harmonics --morton-order -> SOG + compressed PLY
      -> QA scorecard: master vs reduced vs delivered (masked visibility metrics)

  EXTRACT-OBJECT (--extract_object, WP-S2 synthesis):
    multi-view BiRefNet mask VOTING on projected Gaussian centers -> object subset
      -> spatial cleanup: prune + splat-transform --filter-cluster (--seed-pos obj)
      -> self-refit R steps MASKED to the object (`--rgb_object_masked` + alpha_loss)
      -> SOG + compressed PLY
      -> QA object-region: master's object region vs delivered object region,
         plus a background-bleed check (out-of-mask rendered alpha on held-out views)

The ratio/projection machinery is lifted from `_infusion/val_splatsure.py` and
`core/qa/visibility.py`; scene scale from `core/qa/scorers/common.py`; object
center from `core/enhancement/coverage.py`. splat-transform CLI is verified in
EXECUTION_PLAN WP-D1 (v2.7.1: --decimate keeps N%; --filter-floaters/-cluster are
GPU-only; -g cpu disables them).

Usage:
  python -m core.delivery --ckpt <ckpt|ply> --data_dir <colmap dir> --name gpu \
        --factor 2 [--extract_object --mask_dir <masks> [--gen_masks]] \
        [--decimate 60%|--no_decimate] [--refit_steps 3000|--no_refit] \
        [--filter_harmonics 2] [--gpu 0] [--out deliverables]
        [--spz [--spz_sh_degree 4] [--spz_sh1_bits 8 --spz_sh_rest_bits 8]]   # WP-D2

SH degree 4 (WP-SH4): splat-transform v2.7.1 assumes the degree-3 f_rest layout — on a
72-f_rest PLY `--filter-harmonics 3` is a no-op (keeps 72), 2/0 give invalid property counts,
and SOG keeps 3 bands under the degree-3 layout assumption. So a degree>3 master is
truncated EXACTLY to 3 (core.sh_utils) before any splat-transform stage; the full-degree model
is kept and delivered as `<name>_<mode>_sh<d>.ply` (pruned, not decimated) and, with --spz, as
SPZ v4 (the only compact format that carries degree 4). Degree<=3 masters: unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# gsplat JIT shim must precede gsplat import (Windows MSVC flag fix)
from core.training import gsplat_jit_shim  # noqa: F401,E402
from core.training.gsplat_trainer import (  # noqa: E402
    load_colmap_data, render_splats,
)
from core.enhancement.coverage import object_center_from_cameras  # noqa: E402
from core.qa.masked_metrics import masked_psnr, masked_ssim, masked_lpips  # noqa: E402
from core.qa.visibility import VisibilityConfig, build_visibility_mask  # noqa: E402
import imageio.v2 as imageio  # noqa: E402


# ---------------------------------------------------------------------------
# Splat I/O — load ckpt/PLY into the trainer's raw-parameter splats dict, and
# write it back out as a ckpt (for --init_ckpt refit) or a standard 3DGS PLY.
# ---------------------------------------------------------------------------

def load_splats(path: str, device: str) -> dict:
    """ckpt(.pt) or .ply -> {means,opacities,quats,scales,sh0,shN} (raw params)."""
    import qa_report  # top-level script; _REPO is on sys.path
    return qa_report.load_splats(path, device)


def sh_degree_of(splats: dict) -> int:
    """SH degree implied by shN band count: (deg+1)^2 - 1 rest coeffs. Exact for 0..4;
    raises on a count that is not a valid band layout (WP-SH4: used to round silently)."""
    from core.sh_utils import sh_degree_of as _exact
    return _exact(splats)


# splat-transform v2.7.1 (and SOG v2) handle SH degree <= 3 only — see module docstring.
SPLAT_TRANSFORM_MAX_SH = 3


def save_ckpt(splats: dict, path: str, step: int = 0):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step,
                "splats": {k: v.detach().cpu() for k, v in splats.items()}}, str(path))


def export_ply(splats: dict, path: str):
    from gsplat import export_splats
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    export_splats(
        means=splats["means"].detach(),
        scales=splats["scales"].detach(),        # raw log
        quats=splats["quats"].detach(),
        opacities=splats["opacities"].detach(),  # raw logit
        sh0=splats["sh0"].detach(),
        shN=splats["shN"].detach(),
        format="ply",
        save_to=str(path),
    )


def apply_keep(splats: dict, keep: torch.Tensor) -> dict:
    return {k: v[keep].contiguous() for k, v in splats.items()}


# ---------------------------------------------------------------------------
# Scene geometry helpers (reused math, no ctx object needed)
# ---------------------------------------------------------------------------

def robust_scene_scale(data: dict) -> float:
    """1-99 percentile bbox diagonal of the camera centers (world units).
    Mirrors core/qa/scorers/common.py::robust_scene_scale."""
    centers = np.stack([np.asarray(c, np.float64)[:3, 3]
                        for c in data["camtoworlds"]], axis=0)
    lo = np.percentile(centers, 1.0, axis=0)
    hi = np.percentile(centers, 99.0, axis=0)
    d = float(np.linalg.norm(hi - lo))
    if d <= 1e-9:
        d = max(float(data.get("scene_scale", 1.0)), 1e-9)
    return d


def object_center(data: dict) -> np.ndarray:
    """World-space object center = LS point closest to all training camera rays."""
    tr = sorted(int(i) for i in data["train_indices"])
    c2w = data["camtoworlds"][tr]
    centers = c2w[:, :3, 3]
    forwards = c2w[:, :3, 2]                  # camera +z (OpenCV)
    return object_center_from_cameras(centers, forwards)


def spatial_floater_stats(splats: dict, center: np.ndarray, scale: float) -> dict:
    """Mask- and topology-INDEPENDENT floater evidence for an extracted object:
    the radial distribution of the delivered Gaussians about the object center,
    in units of the robust scene diagonal. A clean extraction is spatially compact
    (~all mass near the center); surviving background floaters show up as a tail
    of Gaussians far from the center."""
    m = splats["means"].detach().cpu().numpy()
    r = np.linalg.norm(m - np.asarray(center)[None], axis=1) / max(scale, 1e-9)
    N = len(m)
    return {
        "n": N,
        "radial_over_scale_p50": float(np.percentile(r, 50)),
        "radial_over_scale_p90": float(np.percentile(r, 90)),
        "radial_over_scale_p99": float(np.percentile(r, 99)),
        "radial_over_scale_max": float(r.max()),
        "within_0.3_scale_pct": round(100.0 * float((r < 0.3).mean()), 3),
        "within_0.5_scale_pct": round(100.0 * float((r < 0.5).mean()), 3),
        "beyond_0.5_scale_count": int((r >= 0.5).sum()),
        "beyond_1.0_scale_count": int((r >= 1.0).sum()),
    }


# ---------------------------------------------------------------------------
# Object masks in the exact rendered (undistorted + ROI-cropped) frame
# ---------------------------------------------------------------------------

def mask_in_render_frame(data: dict, idx: int):
    """Load the object mask for view idx and undistort/crop it identically to the
    rendered frame (mirrors gsplat_trainer.load_image's mask branch), so it is
    pixel-aligned with fully_fused_projection / render_splats output. Returns
    (H, W) float32 in [0,1] or None."""
    mpaths = data.get("mask_paths")
    if not mpaths or not mpaths[idx] or not os.path.isfile(mpaths[idx]):
        return None
    cid = data["camera_ids"][idx]
    m = imageio.imread(mpaths[idx])
    if m.ndim == 3:
        m = m[..., 0]
    if cid in data["mapx_dict"]:
        H, W = data["mapx_dict"][cid].shape          # pre-ROI undistort map size
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m = cv2.remap(m, data["mapx_dict"][cid], data["mapy_dict"][cid], cv2.INTER_NEAREST)
        x, y, w, h = data["roi_dict"][cid]
        if w > 0 and h > 0:
            m = m[y:y + h, x:x + w]
    else:
        W, H = data["imsize_dict"][cid]
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    return (m.astype(np.float32) / 255.0)


# ---------------------------------------------------------------------------
# Multi-view mask voting — which Gaussians belong to the object
# ---------------------------------------------------------------------------

def vote_object(splats: dict, data: dict, device: str, *, mask_thresh: float,
                vote_frac: float, min_views: int, stride: int = 1,
                near_plane: float = 0.01, horizon_vote_frac: float = 0.0,
                horizon_quantile: float = 0.10) -> tuple[torch.Tensor, dict]:
    """Project every Gaussian center into the training views that observe it and
    keep those landing inside the object mask in >= vote_frac of those views.

    When horizon_vote_frac > 0, also enforces a second visual-hull pass over the
    lowest-elevation / smallest-apparent-area side views (bottom horizon_quantile
    of valid mask coverage), where any pedestal/book/table beneath a concave object
    separates vertically from the object in 2D.
    Returns (keep bool (N,), stats dict)."""
    from gsplat.cuda._wrapper import fully_fused_projection

    means = splats["means"].float().to(device)
    quats = splats["quats"].float().to(device)
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-12)
    scales = torch.exp(splats["scales"].float().to(device))
    opac = torch.sigmoid(splats["opacities"].float().to(device))
    N = means.shape[0]

    inside = torch.zeros(N, device=device)
    observed = torch.zeros(N, device=device)

    tr = sorted(int(i) for i in data["train_indices"])[::max(1, stride)]
    used = 0
    view_areas: list[tuple[float, int]] = []
    for n, idx in enumerate(tr):
        m = mask_in_render_frame(data, idx)
        if m is None:
            continue
        area_frac = float((m > mask_thresh).mean())
        if area_frac >= 0.02:
            view_areas.append((area_frac, idx))
        cid = data["camera_ids"][idx]
        K = torch.from_numpy(np.asarray(data["Ks_dict"][cid])).float().to(device)[None]
        W, H = data["imsize_dict"][cid]
        vm = torch.linalg.inv(
            torch.from_numpy(data["camtoworlds"][idx]).float().to(device))[None]
        with torch.no_grad():
            out = fully_fused_projection(
                means, None, quats, scales, vm, K, int(W), int(H),
                eps2d=0.0, near_plane=near_plane, far_plane=1e10, packed=False,
                opacities=opac)
            radii = out[0][0]
            means2d = out[1][0]                       # (N, 2) pixel coords
            if radii.ndim == 2:
                radii = radii.max(dim=-1).values
            vis = radii > 0
            u = means2d[:, 0].round().long().clamp(0, int(W) - 1)
            v = means2d[:, 1].round().long().clamp(0, int(H) - 1)
            mt = torch.from_numpy(m).to(device)
            mval = mt[v, u]                           # (N,)
            observed += vis.float()
            inside += (vis & (mval > mask_thresh)).float()
        used += 1
        if n == 0 or (n + 1) % 100 == 0:
            fv = (vis).float().mean().item()
            fin = ((vis & (mval > mask_thresh)).float().sum()
                   / vis.float().sum().clamp(min=1)).item()
            print(f"  [vote] view {n + 1}/{len(tr)} ({data['image_names'][idx]}): "
                  f"visible {100 * fv:.1f}%, of-those-inside-mask {100 * fin:.1f}%",
                  flush=True)

    frac = inside / observed.clamp(min=1.0)
    keep = (observed >= min_views) & (frac >= vote_frac)
    n_kept_all = int(keep.sum())
    horizon_used = 0
    if horizon_vote_frac > 0.0 and len(view_areas) >= 20:
        view_areas.sort(key=lambda t: t[0])
        n_hor = max(10, int(round(len(view_areas) * horizon_quantile)))
        hor_indices = [idx for _, idx in view_areas[:n_hor]]
        inside_h = torch.zeros(N, device=device)
        observed_h = torch.zeros(N, device=device)
        for idx in hor_indices:
            m = mask_in_render_frame(data, idx)
            if m is None:
                continue
            cid = data["camera_ids"][idx]
            K = torch.from_numpy(np.asarray(data["Ks_dict"][cid])).float().to(device)[None]
            W, H = data["imsize_dict"][cid]
            vm = torch.linalg.inv(
                torch.from_numpy(data["camtoworlds"][idx]).float().to(device))[None]
            with torch.no_grad():
                out = fully_fused_projection(
                    means, None, quats, scales, vm, K, int(W), int(H),
                    eps2d=0.0, near_plane=near_plane, far_plane=1e10, packed=False,
                    opacities=opac)
                radii = out[0][0]
                means2d = out[1][0]
                if radii.ndim == 2:
                    radii = radii.max(dim=-1).values
                vis = radii > 0
                u = means2d[:, 0].round().long().clamp(0, int(W) - 1)
                v = means2d[:, 1].round().long().clamp(0, int(H) - 1)
                mt = torch.from_numpy(m).to(device)
                mval = mt[v, u]
                observed_h += vis.float()
                inside_h += (vis & (mval > mask_thresh)).float()
            horizon_used += 1
        min_h_views = min(5, max(2, horizon_used // 3))
        frac_h = inside_h / observed_h.clamp(min=1.0)
        keep_h = (observed_h >= min_h_views) & (frac_h >= horizon_vote_frac)
        keep = keep & keep_h
        print(f"  [vote-horizon] {horizon_used} lowest-elevation side views "
              f"(vote_frac>={horizon_vote_frac}): {n_kept_all:,} -> {int(keep.sum()):,} "
              f"(carved {n_kept_all - int(keep.sum()):,} pedestal/under-object Gaussians)",
              flush=True)

    stats = {
        "n_total": N,
        "n_kept_all_pass": n_kept_all,
        "n_kept": int(keep.sum()),
        "kept_pct": round(100.0 * int(keep.sum()) / max(N, 1), 2),
        "views_used": used,
        "horizon_views_used": horizon_used,
        "horizon_vote_frac": horizon_vote_frac,
        "mask_thresh": mask_thresh,
        "vote_frac": vote_frac,
        "min_views": min_views,
        "stride": stride,
        "median_observed_views": float(observed.median().item()),
        "n_never_observed": int((observed < 1).sum()),
    }
    return keep.cpu(), stats


# ---------------------------------------------------------------------------
# In-memory prune (opacity floor + statistical outlier removal) — prune.py logic
# ---------------------------------------------------------------------------

def prune_dict(splats: dict, *, opacity_min: float, sor_k: int, sor_std: float,
               device: str) -> tuple[dict, dict]:
    s = splats
    N0 = int(s["means"].shape[0])
    keep = torch.ones(N0, dtype=torch.bool, device=device)
    n_low = 0
    if opacity_min > 0:
        low = torch.sigmoid(s["opacities"].float().to(device)) < opacity_min
        n_low = int(low.sum())
        keep &= ~low
    n_out = 0
    if sor_k > 0 and N0 > sor_k + 1:
        from scipy.spatial import cKDTree
        means = s["means"].float().cpu().numpy()
        tree = cKDTree(means)
        d, _ = tree.query(means, k=sor_k + 1, workers=-1)
        knn = d[:, 1:].mean(axis=1)
        thr = float(knn.mean() + sor_std * knn.std())
        outlier = torch.tensor(knn > thr, device=device)
        n_out = int(outlier.sum())
        keep &= ~outlier
    out = apply_keep({k: v.to(device) for k, v in s.items()}, keep)
    stats = {"n_before": N0, "n_after": int(keep.sum()),
             "removed_low_opacity": n_low, "removed_outliers": n_out,
             "opacity_min": opacity_min, "sor_std": sor_std}
    return out, stats


# ---------------------------------------------------------------------------
# splat-transform CLI wrapper
# ---------------------------------------------------------------------------

def run_splat_transform(in_ply: str, out_path: str, actions: list[str],
                        gpu: str = "0", iterations: int | None = None,
                        summary: bool = False) -> str:
    """Invoke splat-transform. Returns stdout. Raises on non-zero exit."""
    exe = shutil.which("splat-transform")
    if exe is None:
        raise RuntimeError("splat-transform not on PATH "
                           "(npm install -g @playcanvas/splat-transform)")
    cmd = [exe, "-w"]
    if gpu is not None:
        cmd += ["-g", str(gpu)]
    if iterations is not None:
        cmd += ["-i", str(iterations)]
    cmd += [str(in_ply)] + list(actions)
    if summary:
        cmd += ["-m"]
    cmd += [str(out_path)]
    print("  [splat-transform] " + " ".join(cmd[1:]), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"splat-transform failed ({r.returncode}):\n"
                           f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    return r.stdout


# ---------------------------------------------------------------------------
# Self-refit (subprocess into the trainer with --init_ckpt)
# ---------------------------------------------------------------------------

def self_refit(reduced_ckpt: str, data_dir: str, result_dir: str, *, factor: int,
               steps: int, sh_degree: int, cap_max: int, python: str,
               object_masked: bool, alpha_loss_lambda: float,
               lpips_lambda: float = 0.2, erank_lambda: float = 0.01,
               scale_reg: float = 0.01, seed: int = 0) -> str:
    """Run a short self-refit; return the refit PLY path (point_cloud_{steps-1}.ply)."""
    trainer = os.path.join(_REPO, "core", "training", "gsplat_trainer.py")
    cmd = [python, trainer,
           "--data_dir", data_dir,
           "--result_dir", result_dir,
           "--data_factor", str(factor),
           "--max_steps", str(steps),
           "--save_steps", str(steps),
           "--test_every", "8",
           "--sh_degree", str(sh_degree),
           "--strategy", "mcmc",
           "--cap_max", str(cap_max),
           "--coarse_factor", "0",
           "--opacity_reg", "0.0",
           "--init_ckpt", reduced_ckpt,
           "--seed", str(seed)]
    if scale_reg > 0:
        cmd += ["--scale_reg", str(scale_reg)]
    if erank_lambda > 0:
        cmd += ["--erank_lambda", str(erank_lambda), "--erank_start", "1"]
    if lpips_lambda > 0:
        lp_patch = 768 if int(factor) == 1 else 512
        cmd += ["--lpips_lambda", str(lpips_lambda), "--lpips_gated",
                "--lpips_start", "1", "--lpips_patch", str(lp_patch), "--lpips_bf16"]
    if object_masked:
        cmd += ["--rgb_object_masked"]
        if alpha_loss_lambda > 0:
            cmd += ["--alpha_loss_lambda", str(alpha_loss_lambda)]
    print("  [refit] " + " ".join(cmd[1:]), flush=True)
    env = dict(os.environ)
    env.pop("APEX_HOLDOUT", None)
    r = subprocess.run(cmd, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"self-refit trainer failed (exit {r.returncode})")
    ply = os.path.join(result_dir, "ply", f"point_cloud_{steps - 1}.ply")
    if not os.path.isfile(ply):
        raise RuntimeError(f"refit produced no PLY at {ply}")
    return ply


# ---------------------------------------------------------------------------
# QA — object-region masked metrics + background-bleed check
# ---------------------------------------------------------------------------

def qa_object_region(splats: dict, data: dict, test_indices, *, sh_degree: int,
                     device: str, mask_thresh: float, out_strip: str | None = None,
                     n_strip: int = 6, min_mask_frac: float = 0.02) -> dict:
    """For each held-out view: render (object on black), build the OGGSplat
    visibility mask, intersect with the BiRefNet object mask, and score masked
    PSNR/SSIM/LPIPS INSIDE that object region only. Also measure background bleed
    = mean rendered alpha OUTSIDE the object mask (should be ~0 for a clean cut).

    Views whose object mask is DEGENERATE (coverage < min_mask_frac, e.g. a
    BiRefNet failure on a hard frame) are SKIPPED: object-region quality and
    background-bleed are both undefined without a valid object mask (scoring them
    would attribute the object's own pixels to 'background'). The count of skipped
    views is reported."""
    from core.training.gsplat_trainer import load_image
    vcfg = VisibilityConfig()
    ps, ss, ls, obj_pct, bleed = [], [], [], [], []
    strip_imgs = []
    n_skipped_mask = 0
    tix = sorted(int(i) for i in test_indices)
    strip_at = (set(np.linspace(0, len(tix) - 1, n_strip).round().astype(int).tolist())
                if (out_strip and len(tix) > 0) else set())

    for vi, idx in enumerate(tix):
        m = mask_in_render_frame(data, idx)
        if m is None:
            continue
        if float((m > mask_thresh).mean()) < min_mask_frac:
            n_skipped_mask += 1                       # degenerate mask (e.g. BiRefNet miss)
            continue
        b = load_image(data, idx)
        gt = b["image"].to(device)
        H, W = gt.shape[:2]
        K = b["K"].to(device)
        viewmat = torch.linalg.inv(b["c2w"].to(device))
        with torch.no_grad():
            renders, alphas, _ = render_splats(
                splats, viewmat, K, W, H, sh_degree,
                torch.zeros(3, device=device), vcfg.rasterize_mode)
            render = renders[0].clamp(0, 1)
            alpha = alphas[0, ..., 0]
        mt = torch.from_numpy(m).to(device)
        if mt.shape != alpha.shape:                       # guard resolution drift
            mt = torch.from_numpy(cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)).to(device)
        obj = mt > mask_thresh
        vis = build_visibility_mask(alpha, vcfg)
        observed = torch.from_numpy(vis["observed"]).to(device)
        region = observed & obj                            # object AND confidently-seen
        if int(region.sum()) < 64:
            continue
        ps.append(masked_psnr(render, gt, region))
        ss.append(masked_ssim(render, gt, region))
        # LPIPS uses spatial context (bbox crop), so composite BOTH render and GT
        # over black by the object mask first — otherwise the delivered object's
        # empty (black) background inside the bbox vs GT's real background inflates
        # LPIPS unfairly. This makes it a fair object-APPEARANCE comparison (both
        # object-on-black) between the full-scene master and the extracted object.
        rf = region.float().unsqueeze(-1)
        ls.append(masked_lpips(render * rf, gt * rf, region, device, mode="bbox"))
        obj_pct.append(100.0 * float(obj.float().mean()))
        # background bleed: rendered alpha where the mask says NOT object. NOTE: this
        # 2D metric is only reliable where the BiRefNet mask is good — on frames where
        # the mask fails (empty/partial), the object's own pixels count as 'bleed'.
        # The mask-INDEPENDENT floater evidence is the 3D spatial compactness of the
        # delivered Gaussians (see spatial_floater_stats), computed once per model.
        outside = ~obj
        bleed.append(float((alpha * outside.float()).sum() / outside.float().sum().clamp(min=1)))

        if vi in strip_at:
            r_bgr = np.ascontiguousarray(
                (render.cpu().numpy()[..., ::-1] * 255).astype(np.uint8))
            # tint bleed pixels (alpha>0.1 outside mask) red for eyeballing
            a_np = alpha.cpu().numpy()
            outside_np = (~obj).cpu().numpy()
            bleedpix = outside_np & (a_np > 0.1)
            r_bgr[bleedpix] = (0.4 * r_bgr[bleedpix]
                               + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
            sc = 512 / max(H, W)
            strip_imgs.append(cv2.resize(r_bgr, (int(W * sc), int(H * sc))))

    if out_strip and strip_imgs:
        hmax = max(s.shape[0] for s in strip_imgs)
        padded = [cv2.copyMakeBorder(s, 0, hmax - s.shape[0], 0, 6,
                                     cv2.BORDER_CONSTANT, value=(30, 30, 30))
                  for s in strip_imgs]
        Path(out_strip).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_strip, np.concatenate(padded, axis=1))
        print(f"  [qa] wrote bleed strip {out_strip}", flush=True)

    def _m(x):
        return float(np.mean(x)) if x else float("nan")
    return {"n_views": len(ps), "n_skipped_degenerate_mask": n_skipped_mask,
            "obj_masked_psnr": _m(ps), "obj_masked_ssim": _m(ss),
            "obj_masked_lpips": _m(ls),
            "mean_object_area_pct": _m(obj_pct),
            "mean_bg_bleed_alpha": _m(bleed),
            "max_bg_bleed_alpha": (float(np.max(bleed)) if bleed else float("nan")),
            "min_mask_frac": min_mask_frac, "strip": out_strip}


def run_qa_report(python: str, ckpt_or_ply: str, data_dir: str, out_dir: str, *,
                  factor: int, sh_degree: int) -> dict:
    """Subprocess qa_report.py (standard visibility-split scorecard); return the JSON."""
    qa = os.path.join(_REPO, "qa_report.py")
    cmd = [python, qa, ckpt_or_ply, data_dir, "--factor", str(factor),
           "--sh_degree", str(sh_degree), "--out", out_dir, "--seed", "0"]
    print("  [qa_report] " + " ".join(cmd[1:]), flush=True)
    r = subprocess.run(cmd, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"qa_report failed (exit {r.returncode})")
    j = os.path.join(out_dir, "scorecard.json")
    return json.loads(Path(j).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Mask preparation
# ---------------------------------------------------------------------------

def ensure_masks(data_dir: str, factor: int, mask_dir: str | None,
                 gen_masks: bool, image_paths, image_names, device: str) -> str:
    """Ensure object masks live in <data_dir>/masks_{factor}/ (COLMAP stems), so
    load_colmap_data auto-attaches them. Copies from mask_dir if given, else runs
    BiRefNet if --gen_masks. Returns the masks_{factor} dir.

    An explicit --mask_dir is AUTHORITATIVE: completeness is judged by the masks
    actually COPIED FROM mask_dir (not by counting whatever files already sit in
    masks_{factor}), so a pre-existing STALE dst file can never silently fill a gap
    the mask_dir does not cover. A partial mask_dir RAISES (or, with --gen_masks, the
    still-missing stems are filled by BiRefNet). Without --mask_dir the behaviour is
    unchanged: reuse a COMPLETE dst, else --gen_masks, else error.
    """
    dst = os.path.join(data_dir, "masks" if int(factor) <= 1 else f"masks_{factor}")
    os.makedirs(dst, exist_ok=True)
    stems = [os.path.splitext(n)[0] for n in image_names]

    def _birefnet_fill(paths, want_stems):
        """Run BiRefNet on `paths`; write dst/<colmapstem>.png for each. Returns count."""
        from core.masking import load_birefnet, run_birefnet_batch, unload_birefnet
        import tempfile
        print(f"[masks] running BiRefNet on {len(paths)} images -> {dst}")
        model, transform = load_birefnet("general", device=device)
        tmp = tempfile.mkdtemp(prefix="birefnet_")
        run_birefnet_batch(model, transform, list(paths), tmp, device=device)
        unload_birefnet()
        # rename tmp/<imgstem>.png -> dst/<colmapstem>.png
        n = 0
        for p, s in zip(paths, want_stems):
            src = os.path.join(tmp, os.path.splitext(os.path.basename(p))[0] + ".png")
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(dst, s + ".png"))
                n += 1
        shutil.rmtree(tmp, ignore_errors=True)
        return n

    # --- Explicit --mask_dir: AUTHORITATIVE, completeness = copied-from-mask_dir ----
    if mask_dir and os.path.isdir(mask_dir):
        copied, missing = 0, []
        for s in stems:
            src = os.path.join(mask_dir, s + ".png")
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(dst, s + ".png"))
                copied += 1
            else:
                missing.append(s)
        print(f"[masks] copied {copied}/{len(stems)} from {mask_dir} -> {dst}")
        if copied >= len(stems):
            return dst
        # mask_dir is PARTIAL. A stale dst file for a `missing` stem must NOT count as
        # coverage (it would deliver mixed fresh+stale masks silently).
        if gen_masks:
            miss = set(missing)
            todo = [(p, s) for p, s in zip(image_paths, stems) if s in miss]
            filled = _birefnet_fill([p for p, _ in todo], [s for _, s in todo])
            print(f"[masks] filled {filled}/{len(missing)} missing masks with "
                  f"BiRefNet -> {dst}")
            still = [s for s in missing
                     if not os.path.isfile(os.path.join(dst, s + ".png"))]
            if still:
                raise RuntimeError(
                    f"[masks] {len(still)}/{len(stems)} masks still missing after "
                    f"mask_dir copy + BiRefNet (e.g. {', '.join(still[:5])})")
            return dst
        raise RuntimeError(
            f"[masks] --mask_dir {mask_dir} is partial: only {copied}/{len(stems)} "
            f"of the required masks are present ({len(missing)} missing, e.g. "
            f"{', '.join(missing[:5])}); pass --gen_masks to fill the rest")

    # --- No --mask_dir: unchanged behaviour (reuse a complete dst, else --gen_masks) -
    have = sum(os.path.isfile(os.path.join(dst, s + ".png")) for s in stems)
    if have >= len(stems):
        print(f"[masks] {have}/{len(stems)} present in {dst}")
        return dst
    if int(factor) == 1:
        alt_dst = os.path.join(data_dir, "masks")
        have_alt = sum(os.path.isfile(os.path.join(alt_dst, s + ".png")) for s in stems)
        if have_alt >= len(stems):
            print(f"[masks] {have_alt}/{len(stems)} present in {alt_dst}")
            return alt_dst
    if gen_masks:
        filled = _birefnet_fill(list(image_paths), stems)
        print(f"[masks] generated {filled}/{len(stems)} BiRefNet masks -> {dst}")
        return dst
    raise RuntimeError("[masks] no masks available; give --mask_dir and/or --gen_masks")


# ---------------------------------------------------------------------------
# File size helper
# ---------------------------------------------------------------------------

def _mb(path: str) -> float:
    try:
        return round(os.path.getsize(path) / 1e6, 2)
    except OSError:
        return float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="WP-D1 delivery pass (+ object extract)")
    ap.add_argument("--ckpt", required=True, help="master ckpt(.pt) or .ply")
    ap.add_argument("--data_dir", required=True, help="COLMAP data dir")
    ap.add_argument("--name", required=True, help="deliverable basename, e.g. 'gpu'")
    ap.add_argument("--scene", default=None, help="QA scene label (default=name)")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--out", default="deliverables", help="deliverables dir")
    ap.add_argument("--work", default=None, help="scratch dir (default <out>/_work/<name>_<mode>)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--python", default=sys.executable, help="python for subprocess trainer/qa")
    # mode
    ap.add_argument("--extract_object", action="store_true",
                    help="carve the scanned object out (default OFF = full-scene)")
    ap.add_argument("--mask_dir", default=None, help="BiRefNet object masks (extract mode)")
    ap.add_argument("--gen_masks", action="store_true",
                    help="run BiRefNet to produce any missing masks (extract mode)")
    # voting
    ap.add_argument("--vote_frac", type=float, default=0.6,
                    help="keep a Gaussian if it lands inside the mask in >= this "
                         "fraction of the views that observe it")
    ap.add_argument("--vote_mask_thresh", type=float, default=0.5,
                    help="soft-mask value above which a projection counts as 'inside'")
    ap.add_argument("--vote_min_views", type=int, default=4,
                    help="require a Gaussian be observed by >= this many views to keep it")
    ap.add_argument("--vote_stride", type=int, default=1, help="subsample voting views")
    ap.add_argument("--vote_horizon_frac", type=float, default=0.85,
                    help="second-stage visual-hull vote threshold across the lowest-elevation "
                         "10%% side views to carve pedestals/tables beneath concave objects "
                         "(0 = off; default 0.85)")
    # prune
    ap.add_argument("--opacity_min", type=float, default=0.01)
    ap.add_argument("--sor_k", type=int, default=8)
    ap.add_argument("--sor_std", type=float, default=2.0)
    ap.add_argument("--prune_tidi", action=argparse.BooleanOptionalAction, default=None,
                    help="run TIDI cross-view evidence + isolation pruner (pre- and post-refit) "
                         "instead of basic opacity+SOR prune (default: ON for --extract_object, "
                         "OFF for full-scene)")
    ap.add_argument("--tidi_data_dir", type=str, default=None,
                    help="COLMAP data dir for TIDI cross-view scoring (default: --data_dir)")
    ap.add_argument("--tidi_data_factor", type=int, default=4,
                    help="image downsample factor for TIDI scoring (default: 4)")
    ap.add_argument("--tidi_mode", choices=["conservative", "moderate"], default="moderate",
                    help="TIDI prune preset ('conservative' or 'moderate', default: 'moderate')")
    # splat-transform filters
    ap.add_argument("--gpu", default="0", help="splat-transform GPU adapter idx or 'cpu'")
    ap.add_argument("--filter_floaters", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--floater_frac", type=float, default=0.02,
                    help="floater voxel edge as fraction of robust scene diagonal "
                         "(world units; splat-transform default 0.05 assumes unit scene)")
    ap.add_argument("--filter_cluster", action=argparse.BooleanOptionalAction, default=None,
                    help="keep the connected cluster at the object center "
                         "(default: ON for --extract_object, OFF for full-scene)")
    ap.add_argument("--cluster_res_frac", type=float, default=0.01,
                    help="cluster voxel resolution as fraction of robust scene diagonal")
    ap.add_argument("--decimate", default=None, help="target count for splat-transform "
                    "--decimate (keeps N or N%%); default: OFF for --extract_object "
                    "(voting+TIDI already reduces count by ~80%%), '60%%' for full-scene")
    ap.add_argument("--no_decimate", action="store_true")
    ap.add_argument("--filter_harmonics", type=int, default=None, choices=[0, 1, 2, 3],
                    help="drop SH bands above this at final export. Default: 3 for "
                         "--extract_object (WP-D1 finding: SH2 truncation is the whole "
                         "LPIPS regression on glossy objects — keep specular), 2 for "
                         "full-scene (halves SH payload, still within the QA bound).")
    # refit
    ap.add_argument("--refit_steps", type=int, default=3000)
    ap.add_argument("--refit", action=argparse.BooleanOptionalAction, default=None,
                    help="short self-refit after decimation. Default: ON for "
                         "--extract_object (recovers detail lost to voting/cluster "
                         "pruning), OFF for full-scene (WP-D1 finding: decimation is "
                         "near-lossless and the L1+SSIM refit trades LPIPS for PSNR on "
                         "the LPIPS-tuned master — unfaithful). Override explicitly.")
    ap.add_argument("--alpha_loss_lambda", type=float, default=0.5,
                    help="object-refit background-alpha killer (extract mode, default 0.5)")
    ap.add_argument("--sog_iterations", type=int, default=15, help="SOG SH compression iters")
    # SPZ v4 (WP-D2) — default OFF
    ap.add_argument("--spz", action="store_true",
                    help="also write <name>_<mode>.spz (Niantic SPZ v4, SH degree 0..4; the "
                         "compact carrier for our degree-4 UE plugin). Default OFF.")
    ap.add_argument("--spz_sh_degree", type=int, default=None, choices=[0, 1, 2, 3, 4],
                    help="SH cap for the SPZ consumer (default: the master's degree). <= the "
                         "delivered degree -> written from the delivered model; higher (e.g. "
                         "4 when the SOG/PLY path is capped at 3) -> from the PRUNED master "
                         "(not decimated), recorded as such.")
    ap.add_argument("--spz_sh1_bits", type=int, default=8,
                    help="SPZ quantization bits for SH band 1 (1-8). Default 8: spz's own "
                         "default 5/4 cost 20+ dB render PSNR vs the PLY (WP-D2)")
    ap.add_argument("--spz_sh_rest_bits", type=int, default=8,
                    help="SPZ quantization bits for SH bands 2-4 (1-8; spz's default is 4)")
    # qa
    ap.add_argument("--no_qa", action="store_true")
    args = ap.parse_args()

    mode = "object" if args.extract_object else "full"
    scene = args.scene or args.name
    device = args.device
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(args.work) if args.work else out_dir / "_work" / f"{args.name}_{mode}"
    work.mkdir(parents=True, exist_ok=True)
    qa_root = Path(_REPO) / "qa"
    rec = {"name": args.name, "scene": scene, "mode": mode, "argv": sys.argv,
           "ckpt": args.ckpt, "data_dir": args.data_dir, "factor": args.factor,
           "started": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}
    t0 = time.time()
    print(f"\n{'='*70}\n[delivery] {args.name} mode={mode} scene={scene}\n{'='*70}", flush=True)

    # ---- masks (extract mode) -------------------------------------------------
    masks_dir = None
    if args.extract_object:
        d_probe = load_colmap_data(args.data_dir, factor=args.factor, normalize=True)
        masks_dir = ensure_masks(args.data_dir, args.factor, args.mask_dir,
                                 args.gen_masks, d_probe["image_paths"],
                                 d_probe["image_names"], device)

    data = load_colmap_data(args.data_dir, factor=args.factor, normalize=True)
    rscale = robust_scene_scale(data)
    ocenter = object_center(data)
    print(f"[scene] robust diagonal = {rscale:.4f} world units; "
          f"object center = ({ocenter[0]:.3f},{ocenter[1]:.3f},{ocenter[2]:.3f})")
    rec["robust_scene_scale"] = rscale
    rec["object_center"] = [float(x) for x in ocenter]

    # ---- load master ----------------------------------------------------------
    splats = load_splats(args.ckpt, device)
    n_master = int(splats["means"].shape[0])
    sh_master = sh_degree_of(splats)
    print(f"[master] {n_master:,} Gaussians, SH degree {sh_master}")
    rec["stages"]["master"] = {"n": n_master, "sh_degree": sh_master}

    # ---- object voting (extract mode) ----------------------------------------
    if args.extract_object:
        print("[extract] multi-view mask voting ...", flush=True)
        keep, vstats = vote_object(
            splats, data, device, mask_thresh=args.vote_mask_thresh,
            vote_frac=args.vote_frac, min_views=args.vote_min_views,
            stride=args.vote_stride, horizon_vote_frac=args.vote_horizon_frac)
        print(f"[extract] voting kept {vstats['n_kept']:,}/{vstats['n_total']:,} "
              f"({vstats['kept_pct']}%)")
        rec["stages"]["vote"] = vstats
        splats = apply_keep({k: v.to(device) for k, v in splats.items()},
                            keep.to(device))

    # ---- prune ---------------------------------------------------------------
    do_prune_tidi = args.prune_tidi if args.prune_tidi is not None else args.extract_object
    tidi_dir = args.tidi_data_dir or args.data_dir
    if do_prune_tidi and tidi_dir:
        from tools.prune_tidi import TidiConfig, prune_ckpt_tidi
        tidi_cfg = TidiConfig.from_preset(args.tidi_mode)
        tidi_src = args.ckpt
        if args.extract_object:
            tidi_src = str(work / "voted_ckpt.pt")
            save_ckpt(splats, tidi_src)
        kept_splats, tidi_reasons, tidi_payload = prune_ckpt_tidi(
            source_path=tidi_src,
            data_dir=tidi_dir,
            factor=args.tidi_data_factor,
            cfg=tidi_cfg,
            device=device,
        )
        splats = {k: v.to(device) for k, v in kept_splats.items()}
        pstats = {
            "mode": f"tidi_{args.tidi_mode}",
            "tidi_mode": args.tidi_mode,
            "tidi_data_dir": str(tidi_dir),
            "tidi_data_factor": int(args.tidi_data_factor),
            "n_before": int(tidi_payload["before"]["num_gaussians"]),
            "n_after": int(tidi_reasons["kept"]),
            **tidi_reasons,
            "isolation_diagnostics": tidi_payload.get("isolation_diagnostics", {}),
        }
        print(f"[prune_tidi] ({args.tidi_mode}) {pstats['n_before']:,} -> {pstats['n_after']:,} "
              f"(dropped {tidi_reasons['dropped_total']:,}: "
              f"low_op={tidi_reasons['dropped_low_opacity']:,}, "
              f"bbox={tidi_reasons['dropped_bbox_outlier']:,}, "
              f"low_ev={tidi_reasons['dropped_low_crossview_evidence']:,}, "
              f"infront={tidi_reasons['dropped_infront_depth_floater']:,}, "
              f"iso3d={tidi_reasons['dropped_3d_isolated_floater']:,})")
        rec["stages"]["prune"] = pstats
        rec["stages"]["prune_tidi"] = pstats
    else:
        splats, pstats = prune_dict(splats, opacity_min=args.opacity_min,
                                    sor_k=args.sor_k, sor_std=args.sor_std, device=device)
        print(f"[prune] {pstats['n_before']:,} -> {pstats['n_after']:,} "
              f"(low-opa {pstats['removed_low_opacity']:,}, outliers {pstats['removed_outliers']:,})")
        rec["stages"]["prune"] = pstats
    pruned_ply = str(work / "pruned.ply")
    pruned_full = splats                     # full SH degree, kept for the sh>3 deliverables
    if sh_master > SPLAT_TRANSFORM_MAX_SH:
        from core.sh_utils import truncate_shN
        splats = dict(splats, shN=truncate_shN(splats["shN"], SPLAT_TRANSFORM_MAX_SH))
        print(f"[sh] master is SH degree {sh_master}: truncated EXACTLY to "
              f"{SPLAT_TRANSFORM_MAX_SH} for the splat-transform/SOG path (it corrupts "
              f"degree>3 PLYs); the degree-{sh_master} model is delivered separately")
        rec["stages"]["sh_truncate_for_splat_transform"] = {
            "from": sh_master, "to": SPLAT_TRANSFORM_MAX_SH}
    export_ply(splats, pruned_ply)

    # ---- splat-transform filters + decimate (keeps SH for refit) -------------
    do_cluster = args.filter_cluster if args.filter_cluster is not None else args.extract_object
    actions = ["--filter-nan"]
    if args.filter_floaters and args.gpu != "cpu":
        fsize = round(args.floater_frac * rscale, 5)
        actions += ["--filter-floaters", f"{fsize},0.1,0.004"]
        rec["stages"].setdefault("splat_transform", {})["floater_voxel"] = fsize
    if do_cluster and args.gpu != "cpu":
        cres = round(args.cluster_res_frac * rscale, 5)
        actions += ["--filter-cluster", f"{cres},0.999,0.1",
                    "--seed-pos", f"{ocenter[0]},{ocenter[1]},{ocenter[2]}"]
        rec["stages"].setdefault("splat_transform", {})["cluster_res"] = cres
    decimate_val = None if args.no_decimate else (
        args.decimate if args.decimate is not None else (None if args.extract_object else "60%")
    )
    if decimate_val is not None:
        actions += ["--decimate", decimate_val]
    reduced_ply = str(work / "reduced.ply")
    # summary of the pruned model first (WP-D1 gotcha: run -m before filtering)
    try:
        smry = run_splat_transform(pruned_ply, "null", ["-m"], gpu=args.gpu)
        (work / "pruned_summary.txt").write_text(smry, encoding="utf-8")
    except Exception as e:
        print(f"  [splat-transform] summary skipped: {e}")
    st_out = run_splat_transform(pruned_ply, reduced_ply, actions, gpu=args.gpu)
    (work / "reduce_log.txt").write_text(st_out, encoding="utf-8")
    red = load_splats(reduced_ply, device)
    n_reduced = int(red["means"].shape[0])
    sh_reduced = sh_degree_of(red)
    print(f"[reduce] -> {n_reduced:,} Gaussians (SH {sh_reduced}) at {reduced_ply}")
    rec["stages"].setdefault("splat_transform", {})
    rec["stages"]["splat_transform"].update(
        {"actions": actions, "n_after": n_reduced, "sh_degree": sh_reduced,
         "decimate": decimate_val})

    # ---- self-refit ----------------------------------------------------------
    do_refit = args.refit if args.refit is not None else args.extract_object
    rec["refit_enabled"] = do_refit
    pre_delivery_ply = reduced_ply
    sh_pre = sh_reduced
    if do_refit:
        reduced_ckpt = str(work / "reduced_ckpt.pt")
        save_ckpt(red, reduced_ckpt)
        refit_dir = str(work / "refit")
        t_ref = time.time()
        refit_ply = self_refit(
            reduced_ckpt, args.data_dir, refit_dir, factor=args.factor,
            steps=args.refit_steps, sh_degree=sh_reduced,
            cap_max=max(n_reduced, 1000), python=args.python,
            object_masked=args.extract_object,
            alpha_loss_lambda=args.alpha_loss_lambda)
        pre_delivery_ply = refit_ply
        refit_splats = load_splats(refit_ply, device)
        sh_pre = sh_degree_of(refit_splats)
        print(f"[refit] {args.refit_steps} steps in {time.time()-t_ref:.0f}s "
              f"-> {int(refit_splats['means'].shape[0]):,} Gaussians (SH {sh_pre})")
        rec["stages"]["refit"] = {"steps": args.refit_steps, "ply": refit_ply,
                                  "object_masked": args.extract_object,
                                  "alpha_loss_lambda": (args.alpha_loss_lambda
                                                        if args.extract_object else 0.0),
                                  "seconds": round(time.time()-t_ref, 1)}
        if do_prune_tidi and tidi_dir:
            from tools.prune_tidi import TidiConfig, prune_ckpt_tidi
            tidi_cfg_post = TidiConfig.from_preset(args.tidi_mode)
            refit_ckpt_pt = os.path.join(refit_dir, "ckpts", f"ckpt_{args.refit_steps - 1}_rank0.pt")
            post_src = refit_ckpt_pt if os.path.isfile(refit_ckpt_pt) else refit_ply
            kept_post, post_reasons, _ = prune_ckpt_tidi(
                source_path=post_src,
                data_dir=tidi_dir,
                factor=args.tidi_data_factor,
                cfg=tidi_cfg_post,
                device=device,
            )
            post_ply = str(work / "refit_tidi.ply")
            export_ply({k: v.to(device) for k, v in kept_post.items()}, post_ply)
            pre_delivery_ply = post_ply
            print(f"[prune_tidi post-refit] ({args.tidi_mode}) "
                  f"{int(refit_splats['means'].shape[0]):,} -> {post_reasons['kept']:,} "
                  f"(dropped {post_reasons['dropped_total']:,}: "
                  f"low_op={post_reasons['dropped_low_opacity']:,}, "
                  f"low_ev={post_reasons['dropped_low_crossview_evidence']:,}, "
                  f"iso3d={post_reasons['dropped_3d_isolated_floater']:,})")
            rec["stages"]["prune_tidi_post_refit"] = post_reasons

    # ---- final export: harmonics + morton -> delivered PLY, then SOG ---------
    # Per-mode default SH: 3 for objects (keep specular — SH2 is the whole LPIPS
    # regression on glossy objects), 2 for full-scene (size, still within bound).
    fh = args.filter_harmonics if args.filter_harmonics is not None else \
        (3 if args.extract_object else 2)
    rec["filter_harmonics"] = fh
    delivered_ply = str(out_dir / f"{args.name}_{mode}.ply")
    final_actions = []
    if fh < sh_pre:
        final_actions += ["--filter-harmonics", str(fh)]
    final_actions += ["--morton-order"]
    run_splat_transform(pre_delivery_ply, delivered_ply, final_actions, gpu=args.gpu)
    delivered_sog = str(out_dir / f"{args.name}_{mode}.sog")
    run_splat_transform(delivered_ply, delivered_sog, [], gpu=args.gpu,
                        iterations=args.sog_iterations)
    deliv = load_splats(delivered_ply, device)
    n_deliv = int(deliv["means"].shape[0])
    sh_deliv = sh_degree_of(deliv)
    print(f"[deliver] {n_deliv:,} Gaussians (SH {sh_deliv}) -> "
          f"{delivered_sog} ({_mb(delivered_sog)} MB), "
          f"{delivered_ply} ({_mb(delivered_ply)} MB)")
    rec["deliverable"] = {
        "sog": delivered_sog, "sog_mb": _mb(delivered_sog),
        "ply": delivered_ply, "ply_mb": _mb(delivered_ply),
        "n_gaussians": n_deliv, "sh_degree": sh_deliv}

    # ---- degree>3 master PLY (WP-SH4) + optional SPZ v4 (WP-D2) --------------
    if sh_master > SPLAT_TRANSFORM_MAX_SH:
        hi_ply = str(out_dir / f"{args.name}_{mode}_sh{sh_master}.ply")
        export_ply(pruned_full, hi_ply)
        rec["deliverable"]["ply_full_sh"] = {
            "ply": hi_ply, "ply_mb": _mb(hi_ply), "sh_degree": sh_master,
            "n_gaussians": int(pruned_full["means"].shape[0]),
            "source": "pruned master (opacity + SOR prune; no splat-transform reduce/refit)"}
        print(f"[deliver] SH{sh_master} master -> {hi_ply} ({_mb(hi_ply)} MB)")
    if args.spz:
        from core.spz_io import write_spz
        spz_deg = sh_master if args.spz_sh_degree is None else args.spz_sh_degree
        if spz_deg > sh_master:
            raise SystemExit(f"--spz_sh_degree {spz_deg} > master SH degree {sh_master}")
        if spz_deg <= sh_deliv:
            spz_src, src_name = deliv, "delivered"
        else:
            spz_src, src_name = pruned_full, "pruned master (delivered path is SH-capped)"
        spz_path = str(out_dir / f"{args.name}_{mode}.spz")
        srec = write_spz(spz_src, spz_path, spz_deg, sh1_bits=args.spz_sh1_bits,
                         sh_rest_bits=args.spz_sh_rest_bits)
        srec.update({"mb": _mb(spz_path), "source": src_name})
        rec["deliverable"]["spz"] = srec
        print(f"[deliver] SPZ v4 SH{spz_deg} ({src_name}) -> {spz_path} ({_mb(spz_path)} MB)")
        cf = srec["clip_frac"]
        print(f"[deliver] SPZ 8-bit clipping: log-scale {100*cf['scales']:.3f}% of values, "
              f"sh0 {100*cf['sh0']:.3f}%, shN {100*cf['shN']:.3f}%; dropped (|pos| >= "
              f"{srec['position_limit']}) {srec['dropped_out_of_range']:,}")
        for w in srec["warnings"]:
            print(f"[deliver] WARNING: {w}")
            rec.setdefault("warnings", []).append(w)

    # ---- QA ------------------------------------------------------------------
    if not args.no_qa:
        print("[qa] scoring ...", flush=True)
        if not args.extract_object:
            # full-scene: standard visibility-split scorecard, master vs reduced vs delivered
            master_card = run_qa_report(args.python, args.ckpt, args.data_dir,
                                        str(qa_root / f"wpd1_{args.name}_master"),
                                        factor=args.factor, sh_degree=sh_master)
            reduced_card = run_qa_report(args.python, reduced_ply, args.data_dir,
                                         str(qa_root / f"wpd1_{args.name}_reduced"),
                                         factor=args.factor, sh_degree=sh_reduced)
            deliv_card = run_qa_report(args.python, delivered_ply, args.data_dir,
                                       str(qa_root / f"wpd1_{args.name}_full"),
                                       factor=args.factor, sh_degree=sh_deliv)

            def pick(c):
                return {"masked_psnr": c["masked_psnr"], "masked_ssim": c["masked_ssim"],
                        "masked_lpips": c["masked_lpips"],
                        "observed_area_pct": c["observed_area_pct"]}
            rec["qa"] = {"master": pick(master_card), "reduced": pick(reduced_card),
                         "delivered": pick(deliv_card)}
            ml0 = master_card["masked_lpips"]
            dl = deliv_card["masked_lpips"] - ml0
            rl = reduced_card["masked_lpips"] - ml0
            rec["qa"]["delta_lpips_delivered_vs_master"] = dl
            rec["qa"]["delta_lpips_delivered_vs_master_pct"] = round(100.0 * dl / ml0, 2)
            rec["qa"]["delta_lpips_reduced_vs_master"] = rl                  # decimation-only loss
            rec["qa"]["delta_lpips_delivered_vs_reduced"] = dl - rl          # refit +/- harmonic drop
            rec["qa"]["refit_enabled"] = do_refit
            if do_refit:
                rec["qa"]["lpips_recovered_by_refit"] = rl - dl
            print(f"[qa] masked-LPIPS  master {ml0:.4f}  reduced {reduced_card['masked_lpips']:.4f} "
                  f" delivered {deliv_card['masked_lpips']:.4f}")
            print(f"[qa] decimation loss (reduced-master) {rl:+.4f}; "
                  f"delivered-vs-master {dl:+.4f} ({100*dl/ml0:+.1f}%); "
                  f"delivered-vs-reduced {dl-rl:+.4f} "
                  f"({'refit+' if do_refit else ''}harmonic/morton)")
        else:
            # object mode: object-region masked metrics + bleed, master vs delivered
            master_splats = load_splats(args.ckpt, device)
            master_obj = qa_object_region(
                master_splats, data, data["test_indices"], sh_degree=sh_master,
                device=device, mask_thresh=args.vote_mask_thresh,
                out_strip=str(qa_root / f"wpd1_{args.name}_object" / "strip_master.png"))
            deliv_obj = qa_object_region(
                deliv, data, data["test_indices"], sh_degree=sh_deliv,
                device=device, mask_thresh=args.vote_mask_thresh,
                out_strip=str(qa_root / f"wpd1_{args.name}_object" / "strip_delivered.png"))
            # visual strips of the delivered object (on black) via qa_report
            deliv_card = run_qa_report(args.python, delivered_ply, args.data_dir,
                                       str(qa_root / f"wpd1_{args.name}_object"),
                                       factor=args.factor, sh_degree=sh_deliv)
            # mask-independent floater evidence (3D compactness of the delivered obj)
            floaters = spatial_floater_stats(deliv, ocenter, rscale)
            rec["qa"] = {"master_object": master_obj, "delivered_object": deliv_obj,
                         "delivered_scorecard_lpips": deliv_card["masked_lpips"],
                         "spatial_floaters": floaters}
            dl = deliv_obj["obj_masked_lpips"] - master_obj["obj_masked_lpips"]
            rec["qa"]["delta_obj_lpips_delivered_vs_master"] = dl
            print(f"[qa] object-region LPIPS  master {master_obj['obj_masked_lpips']:.4f}  "
                  f"delivered {deliv_obj['obj_masked_lpips']:.4f}  (d {dl:+.4f}); "
                  f"skipped {deliv_obj['n_skipped_degenerate_mask']} degenerate-mask views")
            print(f"[qa] background bleed (out-of-mask alpha): master "
                  f"{master_obj['mean_bg_bleed_alpha']:.4f}  delivered "
                  f"{deliv_obj['mean_bg_bleed_alpha']:.4f} "
                  f"(max {deliv_obj['max_bg_bleed_alpha']:.4f})")
            print(f"[qa] spatial floaters: {floaters['within_0.5_scale_pct']}% of "
                  f"delivered Gaussians within 0.5x scene-scale of the object center; "
                  f"{floaters['beyond_0.5_scale_count']} beyond (p99 radial "
                  f"{floaters['radial_over_scale_p99']:.3f}x)")

    rec["seconds_total"] = round(time.time() - t0, 1)
    rec_path = out_dir / f"{args.name}_{mode}_delivery.json"
    rec_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n[delivery] done in {rec['seconds_total']}s. record -> {rec_path}")
    print(f"[delivery] SOG: {delivered_sog}")
    for w in rec.get("warnings", []):
        print(f"[delivery] WARNING: {w}")


if __name__ == "__main__":
    main()
