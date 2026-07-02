"""ArtiFixer wedge RE-JUDGMENT driver (WP-Q1 consumer, 2026-07-02).

Scores a 3DGRUT-exported PLY (base or ArtiFixer3D) on a held-out split under the
WP-Q1 visibility-split QA suite: masked PSNR/SSIM/LPIPS (observed region only) +
the 6 phase-2 scorers (crossscore/nova/tsed/raft_warp/nvs_sqa/flicker).

WHY a separate driver (not qa_report.py): qa_report hardcodes
load_colmap_data(normalize=True) — correct for THIS PROJECT's own gsplat
checkpoints (trained in the normalized frame) but WRONG for the NVIDIA-3DGRUT
PLYs, whose Gaussian positions live in the RAW COLMAP world frame. Empirically
(align_probe): normalize=False reproduces the June base render (render-vs-GT
22.98 dB == June's 22.92 dB); normalize=True gives garbage (7.8 dB). So this
driver takes `--normalize` (default False) and otherwise reuses the SAME core/qa
library code + SceneContext + scorers as qa_report — nothing in core/qa is
modified (CONSUMER only).

Usage:
  python _infusion/wedge_rejudge.py <ply> <data_dir> --split wedge:130:170 \
      --scene laptop_wedge_base --out qa/rejudge_wedge_base --scorers all
"""
import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from core.qa.visibility import VisibilityConfig, build_visibility_mask
from core.qa.masked_metrics import masked_psnr, masked_ssim, masked_lpips
from core.qa.scorecard import Scorecard, ViewScore, SceneContext, write_html
# reuse qa_report's validated PLY reader + thumbnail helpers verbatim
from qa_report import load_splats, _thumb, to_bgr


def main():
    ap = argparse.ArgumentParser(description="ArtiFixer wedge re-judgment (normalize-configurable qa_report)")
    ap.add_argument("ckpt", help="PLY (or .pt) to score")
    ap.add_argument("data_dir")
    ap.add_argument("--factor", type=int, default=1)
    ap.add_argument("--split", default="wedge:130:170",
                    help="every8 | wedge:LO:HI (absolute name-sorted indices)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--alpha_thresh", type=float, default=0.3)
    ap.add_argument("--lpips_mode", choices=["bbox", "tiled"], default="bbox")
    ap.add_argument("--normalize", action="store_true",
                    help="apply project scene normalization (default OFF — 3DGRUT "
                         "PLYs are in the raw COLMAP frame)")
    ap.add_argument("--thumb", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scorers", default="")
    ap.add_argument("--qa_python", default=None)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    os.environ.pop("APEX_HOLDOUT", None)

    from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats

    out_dir = Path(args.out)
    views_dir = out_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"[RJ] scene={args.scene} split={args.split} factor={args.factor} "
          f"normalize={args.normalize} out={out_dir}")
    print(f"[RJ] loading splats: {args.ckpt}")
    splats = load_splats(args.ckpt, device)
    print(f"[RJ] {splats['means'].shape[0]:,} Gaussians  shN={tuple(splats['shN'].shape)}")

    data = load_colmap_data(args.data_dir, factor=args.factor, test_every=8,
                            normalize=args.normalize)

    if args.split.startswith("wedge:"):
        _, s, e = args.split.split(":")
        lo, hi = int(s), int(e)
        n = len(data["image_names"])
        lo, hi = max(0, lo), min(n, hi)
        test_indices = list(range(lo, hi))
        print(f"[RJ] wedge split: views [{lo},{hi}) of {n}  "
              f"({data['image_names'][lo]} .. {data['image_names'][hi-1]})")
    else:
        test_indices = sorted(int(i) for i in data["test_indices"])
    print(f"[RJ] scoring {len(test_indices)} held-out views")

    vcfg = VisibilityConfig(alpha_thresh=args.alpha_thresh)

    view_scores = []
    psnrs, ssims, lpipss, obs_pcts = [], [], [], []
    for vi, idx in enumerate(test_indices):
        b = load_image(data, idx)
        gt = b["image"].to(device); K = b["K"].to(device); c2w = b["c2w"].to(device)
        H, W = gt.shape[:2]
        viewmat = torch.linalg.inv(c2w)
        with torch.no_grad():
            renders, alphas, _ = render_splats(
                splats, viewmat, K, W, H, args.sh_degree,
                torch.zeros(3, device=device), vcfg.rasterize_mode)
            render = renders[0].clamp(0, 1)
            alpha = alphas[0, ..., 0]
        vis = build_visibility_mask(alpha, vcfg)
        observed = torch.from_numpy(vis["observed"]).to(device)

        mp = masked_psnr(render, gt, observed)
        ms = masked_ssim(render, gt, observed)
        ml = masked_lpips(render, gt, observed, device, mode=args.lpips_mode)

        r_bgr = to_bgr(render); g_bgr = to_bgr(gt)
        overlay = r_bgr.copy(); um = vis["unobserved"]
        overlay[um] = (0.5 * overlay[um] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        name = data["image_names"][idx]
        stem = f"v{idx:04d}"
        rp = views_dir / f"{stem}_render.jpg"; gp = views_dir / f"{stem}_gt.jpg"
        op = views_dir / f"{stem}_overlay.png"
        cv2.imwrite(str(rp), _thumb(r_bgr, args.thumb), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(gp), _thumb(g_bgr, args.thumb), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(op), _thumb(overlay, args.thumb))
        view_scores.append(ViewScore(
            index=idx, name=name, observed_pct=vis["observed_pct"],
            masked_psnr=mp, masked_ssim=ms, masked_lpips=ml,
            render=f"views/{rp.name}", gt=f"views/{gp.name}", overlay=f"views/{op.name}"))
        psnrs.append(mp); ssims.append(ms); lpipss.append(ml)
        obs_pcts.append(vis["observed_pct"])
        if vi % 10 == 0 or vi == len(test_indices) - 1:
            print(f"  [{vi+1}/{len(test_indices)}] {name} obs {vis['observed_pct']:.1f}% "
                  f"PSNR {mp:.2f} SSIM {ms:.4f} LPIPS {ml:.4f}", flush=True)

    card = Scorecard(
        scene=args.scene, ckpt=str(args.ckpt), data_dir=str(args.data_dir),
        factor=args.factor, split=args.split, n_views=len(test_indices),
        observed_area_pct=float(np.nanmean(obs_pcts)) if obs_pcts else float("nan"),
        masked_psnr=float(np.nanmean(psnrs)) if psnrs else float("nan"),
        masked_ssim=float(np.nanmean(ssims)) if ssims else float("nan"),
        masked_lpips=float(np.nanmean(lpipss)) if lpipss else float("nan"),
        alpha_thresh=args.alpha_thresh, corruption={}, views=view_scores)

    _ctx = SceneContext(
        scene=args.scene, ckpt=str(args.ckpt), data_dir=str(args.data_dir),
        factor=args.factor, sh_degree=args.sh_degree, device=device,
        out_dir=str(out_dir), data=data, splats=splats,
        test_indices=test_indices, views=view_scores, corruption={})

    if args.scorers:
        from core.qa.scorecard import run_scorers
        from core.qa.scorers import build_scorers, ScorerConfig
        names = [s.strip() for s in args.scorers.split(",") if s.strip()]
        cfg = ScorerConfig()
        if args.qa_python:
            cfg.qa_python = args.qa_python
        scorers = build_scorers(names, cfg)
        print(f"[RJ] running scorers: {[s.name for s in scorers]}")
        run_scorers(card, _ctx, scorers)

    json_path = out_dir / "scorecard.json"
    card.write_json(json_path)
    write_html(card, out_dir / "report.html")
    print(f"\n[RJ] === {args.scene} ({args.split}) ===")
    print(f"  observed area {card.observed_area_pct:.1f}%  mPSNR {card.masked_psnr:.3f} "
          f"mSSIM {card.masked_ssim:.4f} mLPIPS {card.masked_lpips:.4f}")
    print(f"[RJ] wrote {json_path}")


if __name__ == "__main__":
    main()
