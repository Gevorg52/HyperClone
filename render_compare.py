"""Side-by-side held-out renders: GT | base | v1 | v2, full frame + deck crop.

Lets you SEE what the distill transferred into each PLY (vs the captured GT).
Writes PNGs to _dataset/Laptop/compare/.

Usage:  python render_compare.py <base_ckpt> <v1_ckpt> <v2_ckpt>
"""
import sys
from pathlib import Path

import numpy as np
import torch
import cv2

from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\train\data"
OUT = Path(r"F:\HYPERclone\apex-sfm\_dataset\Laptop\compare")
SH = 3
FACTOR = 2
device = "cuda"
# held-out views to render (indices into the test split) + a deck crop (x0,x1,y0,y1 frac)
VIEW_POS = [0, 12, 24]
DECK = (0.30, 0.72, 0.55, 0.95)


def load_splats(p):
    ck = torch.load(p, weights_only=True, map_location=device)
    return {k: v.to(device) for k, v in ck["splats"].items()}


def render(splats, b):
    gt = b["image"].to(device); K = b["K"].to(device); c2w = b["c2w"].to(device)
    H, W = gt.shape[:2]
    with torch.no_grad():
        r, _, _ = render_splats(splats, torch.linalg.inv(c2w), K, W, H, SH,
                                torch.zeros(3, device=device), "antialiased")
    return (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]  # ->BGR


def label(img, txt):
    img = img.copy()
    cv2.putText(img, txt, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    return img


def main():
    # accept "label=ckpt" pairs (e.g. base=... v2=... v3=...); optional data=<dir>, out=<dir>
    global DATA, OUT
    args = []
    for a in sys.argv[1:]:
        if a.startswith("data="):
            DATA = a.split("=", 1)[1]
        elif a.startswith("out="):
            OUT = Path(a.split("=", 1)[1])
        else:
            args.append(a)
    sys.argv = [sys.argv[0]] + args
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)
    test = list(data["test_indices"])
    models = {}
    for arg in sys.argv[1:]:
        lbl, path = arg.split("=", 1)
        models[lbl] = load_splats(path)

    for vp in VIEW_POS:
        idx = test[vp % len(test)]
        b = load_image(data, idx)
        gt = (b["image"].cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
        cols = [label(gt, "GT")]
        for name, sp in models.items():
            cols.append(label(render(sp, b), name))
        strip = np.concatenate(cols, axis=1)
        cv2.imwrite(str(OUT / f"view{vp}_full.jpg"), strip,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        # deck crop, upscaled 2x for inspection
        H, W = gt.shape[:2]
        x0, x1, y0, y1 = DECK
        crops = []
        for c in cols:
            cc = c[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
            crops.append(cv2.resize(cc, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(str(OUT / f"view{vp}_deck.jpg"), np.concatenate(crops, axis=1),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"wrote view{vp}_full.jpg + view{vp}_deck.jpg", flush=True)
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
