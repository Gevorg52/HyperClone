"""Per-frame sharpness diagnosis (variance-of-Laplacian) on the training frames.

Tells us whether the input frames are uniformly soft (capture/motion blur -> a
deblur-aware trainer like BAD-Gaussians could recover real detail) or variable
(some sharp, some blurry -> sharpest-frame selection helps). Higher var = sharper.
"""
import glob
import os

import cv2
import numpy as np

IMG_DIR = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\export\images"
DECK = (0.40, 0.66, 0.60, 0.92)  # deck crop (x0,x1,y0,y1 frac) — the mushy region


def vlap(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
                   glob.glob(os.path.join(IMG_DIR, "*.png")))
    full, deck = [], []
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        full.append(vlap(im))
        h, w = im.shape
        x0, x1, y0, y1 = DECK
        deck.append(vlap(im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]))
    full, deck = np.array(full), np.array(deck)

    def stats(a, name):
        q = np.percentile(a, [0, 10, 25, 50, 75, 90, 100])
        print(f"\n{name} var-of-Laplacian (n={len(a)}):")
        print(f"  min {q[0]:.0f} | p10 {q[1]:.0f} | p25 {q[2]:.0f} | "
              f"median {q[3]:.0f} | p75 {q[4]:.0f} | p90 {q[5]:.0f} | max {q[6]:.0f}")
        print(f"  mean {a.mean():.0f}  std {a.std():.0f}  "
              f"spread(p90/p10) {q[5] / max(q[1], 1e-6):.1f}x")

    stats(full, "FULL FRAME")
    stats(deck, "DECK CROP (textureless region)")
    # how much would top-50% sharpest selection raise the floor?
    thr = np.median(full)
    print(f"\nIf we kept the sharpest 50% of frames (full-frame), "
          f"mean sharpness {full[full >= thr].mean():.0f} vs all {full.mean():.0f} "
          f"(+{100 * (full[full >= thr].mean() / full.mean() - 1):.0f}%)")
    # reference: typical "sharp" photo var-of-Laplacian is ~500-2000+; <100 is soft
    print("\n(reference: crisp photo ~500-2000+; <100 = soft/blurry)")


if __name__ == "__main__":
    main()
