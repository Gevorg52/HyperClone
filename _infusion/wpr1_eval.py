"""WP-R1 evaluation script: Observed-Preservation Gate for Generative Supervision.

Evaluates baseline holed, distill_control, distill_gated, distill_ungated, and
WP-R1 arms across all 23 hole arc views (v0293..v0315) using the fixed OGGSplat
visibility masks (alpha_thresh=0.30) plus sub-zone diagnostics (solid >= 0.99 vs
fringe [0.30, 0.99) in-box / out-box) and off-path / floater metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = str(Path(__file__).resolve().parents[1])
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.environ.pop("APEX_HOLDOUT", None)

from core.qa.masked_metrics import masked_lpips, masked_psnr, masked_ssim
from core.qa.visibility import VisibilityConfig, build_visibility_mask
from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats
from qa_report import load_splats


def _load_rgb(path: str, dev: str) -> torch.Tensor:
    im = cv2.imread(path)
    return torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).float().to(dev) / 255.0


def eval_arm(
    name: str,
    ckpt_path: str,
    data: dict,
    meta: dict,
    arc_dir: str,
    fill_dir: str,
    dev: str = "cuda",
) -> dict:
    sp = load_splats(ckpt_path, dev)
    vcfg = VisibilityConfig(alpha_thresh=0.3)

    # Splat-level opacity & fog stats
    with torch.no_grad():
        op = torch.sigmoid(sp["opacities"].flatten())
        n_gs = int(op.numel())
        op_lt_005_frac = float((op < 0.05).float().mean().item())
        fog_mask = (op > 0.005) & (op < 0.10)
        fog_layer_frac = float(fog_mask.float().mean().item())
        fog_layer_alpha_mean = float((op * fog_mask.float()).mean().item())

    obs_psnr, obs_ssim, obs_lpips_tiled, obs_lpips_bbox, obs_mad = [], [], [], [], []
    unobs_psnr, unobs_ssim, unobs_lpips_bbox, unobs_lpips_comp = [], [], [], []
    unobs_vs_fill_psnr, unobs_vs_fill_lpips = [], []
    solid_psnr, solid_lpips_tiled = [], []
    fringe_psnr, fringe_in_box_psnr, fringe_out_box_psnr = [], [], []
    offpath_fog_alphas = []

    for stem in sorted(meta.keys()):
        idx = int(meta[stem]["index"])
        b = load_image(data, idx)
        gt = b["image"].to(dev)
        K = b["K"].to(dev)
        c2w = b["c2w"].to(dev)
        H, W = gt.shape[:2]

        with torch.no_grad():
            r, a_ren, _ = render_splats(
                sp, torch.linalg.inv(c2w), K, W, H, 3,
                torch.zeros(3, device=dev), "antialiased",
            )
            render = r[0].clamp(0.0, 1.0)

            # Off-path camera perturbation (shift +0.15 along camera up/right)
            c2w_off = c2w.clone()
            c2w_off[:3, 3] = c2w[:3, 3] + 0.15 * c2w[:3, 0] + 0.10 * c2w[:3, 1]
            _, a_off, _ = render_splats(
                sp, torch.linalg.inv(c2w_off), K, W, H, 3,
                torch.zeros(3, device=dev), "antialiased",
            )
            a_off_2d = a_off[0, ..., 0]
            off_semi = ((a_off_2d > 0.01) & (a_off_2d < 0.25)).float()
            offpath_fog_alphas.append(float((a_off_2d * off_semi).mean().item()))

        alpha_b = torch.from_numpy(
            np.load(os.path.join(arc_dir, f"{stem}_alpha.npy"))
        ).to(dev)
        vis = build_visibility_mask(alpha_b, vcfg)
        obs = torch.from_numpy(vis["observed"]).to(dev)
        unobs = torch.from_numpy(vis["unobserved"]).to(dev)
        solid = alpha_b >= 0.99
        fringe = obs & (~solid)

        yy, xx = torch.meshgrid(
            torch.arange(H, device=dev, dtype=torch.float32) / H,
            torch.arange(W, device=dev, dtype=torch.float32) / W,
            indexing="ij",
        )
        in_box = ((xx - 0.50).abs() <= 0.26) & ((yy - 0.42).abs() <= 0.29)
        fringe_in = fringe & in_box
        fringe_out = fringe & (~in_box)

        holed = _load_rgb(os.path.join(arc_dir, f"{stem}_render.png"), dev)
        fill = _load_rgb(os.path.join(fill_dir, f"{stem}_fill.png"), dev)

        obs_psnr.append(masked_psnr(render, gt, obs))
        obs_ssim.append(masked_ssim(render, gt, obs))
        obs_lpips_tiled.append(
            masked_lpips(render, gt, obs, dev, mode="tiled", tile=256, cover=0.9)
        )
        obs_lpips_bbox.append(masked_lpips(render, gt, obs, dev, mode="bbox"))

        om = obs[..., None].float()
        obs_mad.append(float(((render - holed).abs() * om).sum() / (om.sum().clamp_min(1.0) * 3.0)))

        unobs_psnr.append(masked_psnr(render, gt, unobs))
        unobs_ssim.append(masked_ssim(render, gt, unobs))
        unobs_lpips_bbox.append(masked_lpips(render, gt, unobs, dev, mode="bbox"))
        unobs_lpips_comp.append(masked_lpips(render, gt, unobs, dev, mode="composited"))

        unobs_vs_fill_psnr.append(masked_psnr(render, fill, unobs))
        unobs_vs_fill_lpips.append(masked_lpips(render, fill, unobs, dev, mode="bbox"))

        solid_psnr.append(masked_psnr(render, gt, solid))
        solid_lpips_tiled.append(
            masked_lpips(render, gt, solid, dev, mode="tiled", tile=256, cover=0.9)
        )
        fringe_psnr.append(masked_psnr(render, gt, fringe))
        if fringe_in.any():
            fringe_in_box_psnr.append(masked_psnr(render, gt, fringe_in))
        if fringe_out.any():
            fringe_out_box_psnr.append(masked_psnr(render, gt, fringe_out))

    res = {
        "arm": name,
        "ckpt": ckpt_path,
        "num_gaussians": n_gs,
        "observed_masked_PSNR": float(np.mean(obs_psnr)),
        "observed_masked_SSIM": float(np.mean(obs_ssim)),
        "observed_masked_LPIPS": float(np.mean(obs_lpips_tiled)),
        "observed_masked_LPIPS_bbox": float(np.mean(obs_lpips_bbox)),
        "observed_drift_MAD_vs_holed": float(np.mean(obs_mad)),
        "unobserved_masked_PSNR": float(np.mean(unobs_psnr)),
        "unobserved_masked_SSIM": float(np.mean(unobs_ssim)),
        "unobserved_masked_LPIPS_bbox": float(np.mean(unobs_lpips_bbox)),
        "unobserved_masked_LPIPS_composited": float(np.mean(unobs_lpips_comp)),
        "unobserved_vs_fill_PSNR": float(np.mean(unobs_vs_fill_psnr)),
        "unobserved_vs_fill_LPIPS_bbox": float(np.mean(unobs_vs_fill_lpips)),
        "solid_obs_99_PSNR": float(np.mean(solid_psnr)),
        "solid_obs_99_LPIPS_tiled": float(np.mean(solid_lpips_tiled)),
        "fringe_obs_03_99_PSNR": float(np.mean(fringe_psnr)),
        "fringe_in_box_PSNR": float(np.mean(fringe_in_box_psnr)),
        "fringe_out_box_PSNR": float(np.mean(fringe_out_box_psnr)),
        "fog_layer_alpha_mean": fog_layer_alpha_mean,
        "fog_layer_frac": fog_layer_frac,
        "offpath_semi_alpha_mean": float(np.mean(offpath_fog_alphas)),
        "op_lt_005_frac": op_lt_005_frac,
    }
    print(
        f"[{name:18s}] obs PSNR={res['observed_masked_PSNR']:.2f} "
        f"SSIM={res['observed_masked_SSIM']:.4f} LPIPS={res['observed_masked_LPIPS']:.4f} | "
        f"unobs PSNR={res['unobserved_masked_PSNR']:.2f} LPIPS={res['unobserved_masked_LPIPS_bbox']:.4f} | "
        f"solid99={res['solid_obs_99_PSNR']:.2f} fringe={res['fringe_obs_03_99_PSNR']:.2f} "
        f"(in_box={res['fringe_in_box_PSNR']:.2f} out_box={res['fringe_out_box_PSNR']:.2f}) | "
        f"fog={res['fog_layer_alpha_mean']:.5f}"
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="_dataset/GPU_V4/apex_images/training/gsplat_prod/data")
    ap.add_argument("--arc", default="_infusion/hole/arc")
    ap.add_argument("--fill_dir", default="_infusion/hole/fills_gated")
    ap.add_argument("--out", default="_infusion/wpr1/gate.json")
    ap.add_argument("--arms", nargs="*", default=[])
    args = ap.parse_args()

    meta = json.loads(Path(args.arc, "meta.json").read_text(encoding="utf-8"))
    data = load_colmap_data(args.data, factor=2, test_every=8, normalize=True)

    default_arms = [
        ("holed_baseline", "_infusion/hole/holed_gpuv4.pt"),
        ("distill_control", "_infusion/hole/distill_control/ckpts/ckpt_3999_rank0.pt"),
        ("distill_gated", "_infusion/hole/distill_gated/ckpts/ckpt_3999_rank0.pt"),
        ("distill_ungated", "_infusion/hole/distill_ungated/ckpts/ckpt_3999_rank0.pt"),
    ]
    for spec in args.arms:
        k, v = spec.split("=", 1)
        default_arms.append((k, v))

    results = {}
    for name, ckpt in default_arms:
        if os.path.isfile(ckpt):
            results[name] = eval_arm(name, ckpt, data, meta, args.arc, args.fill_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[WP-R1 eval] Wrote {out_path}")


if __name__ == "__main__":
    main()
