"""WP-F1 gate evidence: deck-region + full-frame LPIPS on held-out views.

Method: render each every-8 held-out view at factor 2 (black background,
antialiased), clamp to [0,1]; LPIPS = torchmetrics VGG-LPIPS (normalize=True,
same model as the trainer's get_lpips). Deck crop = fixed fractional box
(x0,x1,y0,y1) = (0.30, 0.72, 0.55, 0.95) of the frame — identical to
render_compare.py's DECK box (the capture-starved laptop deck region).
Full-frame LPIPS is reported alongside for reference.

Usage (repo root):
  python _infusion/wpf1_deck_eval.py [--out _infusion/wpf1_ab_deck.json]
Defaults evaluate the WP-F1 A/B 15k checkpoints (off vs mono_depth_lambda=0.05).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.gsplat_trainer import (  # noqa: E402
    load_colmap_data, load_image, render_splats, get_lpips,
)

AB = "_dataset/Laptop/apex_images/training/wpf1_ab"
DEFAULT_ARMS = {
    "off": f"{AB}/off/output/ckpts/ckpt_14999_rank0.pt",
    "on":  f"{AB}/on/output/ckpts/ckpt_14999_rank0.pt",
}
DEFAULT_DATA = "_dataset/Laptop/apex_images/training/gsplat_prod/data"
DECK = (0.30, 0.72, 0.55, 0.95)  # x0, x1, y0, y1 as fractions (render_compare.py)


def load_splats(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    return {k: v.to(device) for k, v in ck["splats"].items()}


def eval_ckpt(ckpt, data, lp, sh_degree, device):
    splats = load_splats(ckpt, device)
    full, deck = [], []
    with torch.no_grad():
        for idx in sorted(int(x) for x in data["test_indices"]):
            b = load_image(data, idx)
            gt = b["image"].to(device); K = b["K"].to(device)
            c2w = b["c2w"].to(device)
            H, W = gt.shape[:2]; w2c = torch.linalg.inv(c2w)
            r, _, _ = render_splats(splats, w2c, K, W, H, sh_degree,
                                    torch.zeros(3, device=device), "antialiased")
            r = r[0].clamp(0, 1)
            rb = r.permute(2, 0, 1).unsqueeze(0)
            gb = gt.permute(2, 0, 1).unsqueeze(0)
            full.append(float(lp(rb, gb)))
            x0, x1 = int(DECK[0] * W), int(DECK[1] * W)
            y0, y1 = int(DECK[2] * H), int(DECK[3] * H)
            deck.append(float(lp(rb[:, :, y0:y1, x0:x1], gb[:, :, y0:y1, x0:x1])))
    return dict(ckpt=str(ckpt), n_views=len(full),
                full_lpips=float(np.mean(full)), deck_lpips=float(np.mean(deck)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA)
    ap.add_argument("--arms", nargs="*",
                    default=[f"{k}={v}" for k, v in DEFAULT_ARMS.items()])
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--out", default="_infusion/wpf1_ab_deck.json")
    args = ap.parse_args()
    device = "cuda"

    data = load_colmap_data(args.data_dir, factor=args.factor, test_every=8,
                            normalize=True)
    lp = get_lpips(device)
    result = dict(metric="mean VGG-LPIPS (torchmetrics, normalize=True) over "
                         "every8 held-out views; deck = fractional crop box",
                  deck_box_frac=dict(x0=DECK[0], x1=DECK[1], y0=DECK[2], y1=DECK[3]),
                  data_dir=str(args.data_dir), factor=args.factor, arms={})
    for spec in args.arms:
        label, ckpt = spec.split("=", 1)
        r = eval_ckpt(ckpt, data, lp, args.sh_degree, device)
        result["arms"][label] = r
        print(f"{label:6s} views={r['n_views']}  full={r['full_lpips']:.4f}  "
              f"deck={r['deck_lpips']:.4f}")
    labels = list(result["arms"])
    if len(labels) == 2:
        a, b = (result["arms"][l] for l in labels)
        result["delta"] = {k: dict(abs=b[k] - a[k], pct=(b[k] - a[k]) / a[k] * 100.0)
                           for k in ("full_lpips", "deck_lpips")}
        print(f"delta {labels[1]} vs {labels[0]}: " + "  ".join(
            f"{k} {result['delta'][k]['abs']:+.4f} ({result['delta'][k]['pct']:+.1f}%)"
            for k in ("full_lpips", "deck_lpips")))
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
