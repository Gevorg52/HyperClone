"""De-risk Phase-1: do cheap confidence signals predict where the baseline is WRONG
in the under-observed wedge? If yes, gating on them is valid.

Signals tested (vs error against held-out GT):
  (1) per-view: distance to nearest TRAINING camera (coverage proxy) -> view error
  (2) per-pixel: rendered alpha-accumulation (1-alpha = geometry uncertainty) -> |err|
A positive correlation = the signal is a usable gate.
"""
import os
os.environ["APEX_HOLDOUT"] = "0.275,0.45"

import numpy as np
import torch

from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats, ssim

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\GPU\proof\baseline\output\ckpts\ckpt_14999_rank0.pt"
SH, dev = 3, "cuda"


def pearson(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True)
    c2ws = data["camtoworlds"]
    train_c = c2ws[data["train_indices"]][:, :3, 3]            # training cam centers
    s = torch.load(CK, weights_only=True, map_location=dev)["splats"]
    splats = {k: v.to(dev) for k, v in s.items()}

    cov, verr = [], []
    pconf, perr = [], []
    with torch.no_grad():
        for idx in data["test_indices"]:
            b = load_image(data, idx)
            gt = b["image"].to(dev); K = b["K"].to(dev); c2w = b["c2w"].to(dev)
            H, W = gt.shape[:2]
            center = c2w[:3, 3].cpu().numpy()
            cov.append(float(np.min(np.linalg.norm(train_c - center, axis=1))))
            r, alpha, _ = render_splats(splats, torch.linalg.inv(c2w), K, W, H, SH,
                                        torch.zeros(3, device=dev), "antialiased")
            rgb = r[0][..., :3].clamp(0, 1)
            verr.append(1.0 - ssim(rgb.permute(2, 0, 1)[None],
                                   gt.permute(2, 0, 1)[None]).item())
            # per-pixel (subsample for speed)
            e = (rgb - gt).abs().mean(-1)                       # (H,W)
            unc = (1.0 - alpha[0, ..., 0]).clamp(0, 1)          # (H,W) geometry uncertainty
            ss = max(1, (H * W) // 20000)
            perr.append(e.flatten()[::ss].cpu().numpy())
            pconf.append(unc.flatten()[::ss].cpu().numpy())

    print(f"\n=== confidence-signal validation (wedge, n={len(cov)} views) ===")
    print(f"(1) per-view  corr( dist-to-nearest-train-cam , view error )  = "
          f"{pearson(cov, verr):+.3f}   (positive => coverage predicts error)")
    pe = np.concatenate(perr); pc = np.concatenate(pconf)
    print(f"(2) per-pixel corr( 1-alpha (geom uncertainty) , |error| )    = "
          f"{pearson(pc, pe):+.3f}   (n={pe.size:,} px; positive => alpha predicts error)")
    # how much worse are the farthest-coverage views?
    cov = np.array(cov); verr = np.array(verr)
    q = np.quantile(cov, [0.25, 0.75])
    near = verr[cov <= q[0]].mean(); far = verr[cov >= q[1]].mean()
    print(f"    view error: nearest-25% coverage {near:.3f}  vs farthest-25% {far:.3f}  "
          f"({far/max(near,1e-6):.2f}x)")


if __name__ == "__main__":
    main()
