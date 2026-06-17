"""Prepare InFusion depth-inpainting inputs from our gsplat checkpoint.

For a chosen view, render the splat (RGB + expected-depth + alpha), build a
disparity map (1/depth) for the *known* geometry, carve a rectangular hole mask,
and emit the five files InFusion's run_inference_inpainting.py consumes:
  rgb.png   - reference RGB (we feed the real GT photo: isolates DEPTH-completion
              quality, the core thing to validate; for a true hole you'd SDXL-inpaint)
  mask.png  - white(255) = region to inpaint, black(0) = known
  disp.npy  - disparity (1/depth) of existing gaussians (hole region is ignored)
  intri.npy - 3x3 K at the RGB resolution (InFusion back-projects at RGB res)
  c2w.npy   - 4x4 camera-to-world (normalized frame, OpenCV convention)

Also saves true_depth.npy + alpha.npy so we can score InFusion's completed depth
against the real surface inside the hole (the region is observed -> we know truth).

Usage: python prep.py [ckpt] [factor] [idx] [out_dir]
"""
import os
import sys

import numpy as np
import torch
import cv2

from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
SH, dev = 3, "cuda"


def main():
    ck = sys.argv[1] if len(sys.argv) > 1 else \
        r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\output\ckpts\ckpt_29999_rank0.pt"
    factor = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    idx = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    out = sys.argv[4] if len(sys.argv) > 4 else r"F:\HYPERclone\apex-sfm\_infusion\io"
    os.makedirs(out, exist_ok=True)

    s = torch.load(ck, weights_only=True, map_location=dev)["splats"]
    splats = {k: v.to(dev) for k, v in s.items()}
    data = load_colmap_data(DATA, factor=factor, test_every=8, normalize=True)
    if idx < 0:
        idx = int(list(data["test_indices"])[len(data["test_indices"]) // 2])
    print(f"view idx={idx}  factor={factor}")

    b = load_image(data, idx)
    gt = b["image"].to(dev)                         # (H,W,3) real photo, [0,1]
    K = b["K"].to(dev); c2w = b["c2w"].to(dev)
    H, W = gt.shape[:2]

    with torch.no_grad():
        r, alphas, _ = render_splats(
            splats, torch.linalg.inv(c2w), K, W, H, SH,
            torch.zeros(3, device=dev), "antialiased", render_mode="RGB+ED")
    depth = r[0, ..., 3]                             # (H,W) expected z-depth
    alpha = alphas[0, ..., 0]                        # (H,W) coverage
    valid = (alpha > 0.5) & (depth > 1e-6)
    disp = torch.where(valid, 1.0 / depth.clamp(min=1e-6),
                       torch.zeros_like(depth))      # invalid -> 0 (infinitely far)

    # Hole: centered box over the object (region IS observed -> we know the truth)
    mask = np.zeros((H, W), np.uint8)
    bw, bh = int(0.40 * W), int(0.40 * H)
    x0, y0 = (W - bw) // 2, (H - bh) // 2
    mask[y0:y0 + bh, x0:x0 + bw] = 255

    # Emit files
    cv2.imwrite(os.path.join(out, "rgb.png"),
                cv2.cvtColor((gt.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out, "mask.png"), mask)
    np.save(os.path.join(out, "disp.npy"), disp.cpu().numpy().astype(np.float32))
    np.save(os.path.join(out, "intri.npy"), K.cpu().numpy().astype(np.float64))
    np.save(os.path.join(out, "c2w.npy"), c2w.cpu().numpy().astype(np.float64))
    np.save(os.path.join(out, "true_depth.npy"), depth.cpu().numpy().astype(np.float32))
    np.save(os.path.join(out, "alpha.npy"), alpha.cpu().numpy().astype(np.float32))

    nhole = int((mask > 0).sum())
    valid_in_hole = int((valid.cpu().numpy()[mask > 0]).sum())
    print(f"size {W}x{H}  hole {bw}x{bh} ({nhole} px, {valid_in_hole} observed)")
    print(f"disp known-region range: "
          f"{disp[valid].min().item():.4f}..{disp[valid].max().item():.4f}")
    print(f"OUT {out}")


if __name__ == "__main__":
    main()
