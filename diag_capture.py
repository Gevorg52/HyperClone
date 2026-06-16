"""Capture diagnostics to validate BAD-Gaussians + CLAHE-for-SfM before investing.

(1) MOTION-BLUR TEST: correlate per-frame sharpness (var-of-Laplacian) with
    camera motion speed (||C_i - C_{i-1}|| from COLMAP poses). Negative correlation
    => blurrier frames are faster-moving => CAMERA MOTION BLUR => BAD-Gaussians can
    help. ~Zero correlation => defocus/codec softness => BAD-Gaussians won't help.

(2) DECK SfM-POINT DENSITY: are 3D points starved on the textureless deck (the
    MCMC-starvation root cause)? If so, CLAHE-for-SfM (more features there) could
    seed a denser deck. Compares deck-bbox point density vs full-frame.
"""
import glob
import os

import cv2
import numpy as np
import pycolmap

SPARSE = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\export\sparse\0"
IMG_DIR = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\export\images"
DECK = (0.40, 0.66, 0.60, 0.92)  # x0,x1,y0,y1 frac


def main():
    rec = pycolmap.Reconstruction(SPARSE)

    # ---- per-frame camera center + name ----
    rows = []
    for iid, img in rec.images.items():
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        R = cfw.rotation.matrix()
        t = np.asarray(cfw.translation).reshape(3)
        C = -R.T @ t                       # camera center in world
        rows.append((img.name, iid, C))
    rows.sort(key=lambda r: r[0])          # temporal order by filename

    # ---- per-frame sharpness (var-of-Laplacian) ----
    sharp = {}
    for name, _, _ in rows:
        p = os.path.join(IMG_DIR, name)
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        sharp[name] = float(cv2.Laplacian(g, cv2.CV_64F).var()) if g is not None else np.nan

    names = [r[0] for r in rows]
    C = np.array([r[2] for r in rows])
    speed = np.linalg.norm(np.diff(C, axis=0), axis=1)          # frame-to-frame
    s = np.array([sharp[n] for n in names])
    s_mid = s[1:]                                               # align with speed

    valid = ~np.isnan(s_mid)
    sp, sh = speed[valid], s_mid[valid]
    # correlations (Pearson + Spearman via rank)
    pear = float(np.corrcoef(sp, sh)[0, 1])
    rk = lambda a: np.argsort(np.argsort(a))
    spear = float(np.corrcoef(rk(sp), rk(sh))[0, 1])
    print("=== (1) MOTION-BLUR TEST ===")
    print(f"frames={valid.sum()}  speed(world units/frame): "
          f"median {np.median(sp):.4f}  p90 {np.percentile(sp,90):.4f}")
    print(f"corr(sharpness, motion-speed): Pearson {pear:+.3f}  Spearman {spear:+.3f}")
    # sharpness of slowest-25% vs fastest-25% frames
    q = np.percentile(sp, [25, 75])
    slow = sh[sp <= q[0]].mean()
    fast = sh[sp >= q[1]].mean()
    print(f"mean sharpness: slowest-25% {slow:.0f}  vs  fastest-25% {fast:.0f}  "
          f"(slow/fast {slow / max(fast,1e-6):.2f}x)")
    print("  => negative corr / slow>>fast = MOTION BLUR (BAD-Gaussians helps);"
          " ~0 = defocus/codec (won't help)")

    # ---- (2) deck SfM-point density on a representative frame ----
    print("\n=== (2) DECK SfM-POINT DENSITY ===")
    # use the alphabetically-first image with the most 3D-linked keypoints
    best = max(rec.images.values(),
               key=lambda im: sum(1 for p in im.points2D if p.has_point3D()))
    cam = rec.cameras[best.camera_id]
    W, H = cam.width, cam.height
    x0, x1, y0, y1 = DECK
    bx0, bx1, by0, by1 = x0 * W, x1 * W, y0 * H, y1 * H
    n_full = n_deck = 0
    for p in best.points2D:
        if not p.has_point3D():
            continue
        n_full += 1
        x, y = p.xy
        if bx0 <= x <= bx1 and by0 <= y <= by1:
            n_deck += 1
    area_frac = (x1 - x0) * (y1 - y0)
    deck_density = n_deck / max(area_frac, 1e-9)
    full_density = n_full / 1.0
    print(f"frame {best.name}  ({W}x{H})  3D-linked keypoints: {n_full}")
    print(f"deck bbox: {n_deck} points in {area_frac*100:.0f}% of area")
    print(f"density ratio deck/full = {deck_density/max(full_density,1e-9):.2f}  "
          f"(<1 = deck STARVED of SfM points -> CLAHE-for-SfM may seed more)")


if __name__ == "__main__":
    main()
