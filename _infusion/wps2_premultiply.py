"""WP-S2: build pre-multiplied black-bg factor-2 training frames for arms (b)/(c).

For each factor-2 frame, multiply RGB by the (same-resolution) object mask so the
background goes to pure black — this is exactly gsplat_backend._apply_masks_to_images'
premultiply, but written as lossless PNG (identical stems to gsplat_prod/images_2) so
the trainer's colmap->images_2 name map stays identity and there is no JPEG ringing on
the hard mask boundary.

Usage: python _infusion/wps2_premultiply.py
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "_dataset/GPU_V4/apex_images/training/gsplat_prod/data/images_2"
MASK = ROOT / "_dataset/GPU_V4/apex_images/training/wps2/masks_2"
DST = ROOT / "_dataset/GPU_V4/apex_images/training/wps2/obj/data/images_2"


def main():
    DST.mkdir(parents=True, exist_ok=True)
    imgs = sorted(SRC.glob("*.png"))
    n = 0
    for p in imgs:
        img = cv2.imread(str(p)).astype(np.float32)
        m = cv2.imread(str(MASK / (p.stem + ".png")), cv2.IMREAD_GRAYSCALE)
        assert m is not None, f"missing mask for {p.stem}"
        if m.shape[:2] != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = (img * (m.astype(np.float32) / 255.0)[:, :, None]).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(DST / (p.stem + ".png")), out)
        n += 1
        if n % 100 == 0:
            print(f"  [{n}/{len(imgs)}]", flush=True)
    # sanity: report near-black fraction of the premultiplied frames (should be high)
    s = cv2.imread(str(DST / imgs[0].stem) + ".png")
    nb = float((s.max(axis=2) < 16).mean()) * 100
    print(f"[premult] wrote {n} frames to {DST}; frame0 near-black={nb:.1f}% "
          f"(bg now black) shape={s.shape}")


if __name__ == "__main__":
    main()
