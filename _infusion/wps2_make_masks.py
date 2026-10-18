"""WP-S2 step 2: object masks for GPU_V4 via BiRefNet (headless).

Runs the production masking engine (core.masking.run_birefnet_batch, BiRefNet
'general' — the same auto-matte the GUI masking stage uses) on the factor-2
training frames, so masks are stem-matched + resolution-matched to images_2 (the
trainer resolves masks_2/ first for factor 2). Writes:
  - masks at wps2/masks_2/DSC*.png  (soft alpha 0-255)
  - overlay spot-check strips for 5 spread views (qa/wps2_mask_overlays/)
  - coverage stats JSON (qa/wps2_masks.json)

Usage: python _infusion/wps2_make_masks.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.masking import load_birefnet, run_birefnet_batch, generate_overlay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG2 = ROOT / "_dataset/GPU_V4/apex_images/training/gsplat_prod/data/images_2"
MASK2 = ROOT / "_dataset/GPU_V4/apex_images/training/wps2/masks_2"
OVL = ROOT / "qa/wps2_mask_overlays"
STATS = ROOT / "qa/wps2_masks.json"


def main():
    imgs = sorted(IMG2.glob("*.png"))
    assert imgs, f"no images at {IMG2}"
    MASK2.mkdir(parents=True, exist_ok=True)
    print(f"[masks] {len(imgs)} frames -> {MASK2}")

    model, tf = load_birefnet("general", device="cuda")
    n = 0

    def cb(i, total):
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}]", flush=True)

    n = run_birefnet_batch(model, tf, [str(p) for p in imgs], str(MASK2),
                           device="cuda", threshold=0.0, erosion_px=0,
                           progress_cb=cb)
    print(f"[masks] generated {n} masks")

    # coverage stats over a stride sample + overlays for 5 spread views
    OVL.mkdir(parents=True, exist_ok=True)
    sample = imgs[:: max(1, len(imgs) // 60)]
    fgs = []
    for p in sample:
        m = cv2.imread(str(MASK2 / (p.stem + ".png")), cv2.IMREAD_GRAYSCALE)
        fgs.append(float((m > 128).mean()) * 100.0)
    picks = [imgs[0], imgs[len(imgs) // 4], imgs[len(imgs) // 2],
             imgs[3 * len(imgs) // 4], imgs[-1]]
    for p in picks:
        img = cv2.imread(str(p))
        m = cv2.imread(str(MASK2 / (p.stem + ".png")), cv2.IMREAD_GRAYSCALE)
        ov = generate_overlay(img[..., ::-1], m, alpha=0.5, color=(0, 255, 0))[..., ::-1]
        strip = np.hstack([img, cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), ov])
        strip = cv2.resize(strip, None, fx=0.30, fy=0.30)
        cv2.putText(strip, p.stem, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(OVL / f"overlay_{p.stem}.jpg"), strip,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    stats = dict(
        method="BiRefNet 'general' (ZhengPeng7/BiRefNet, trust_remote_code) soft alpha, "
               "threshold=0 erosion=0, run on factor-2 images_2 (headless "
               "core.masking.run_birefnet_batch)",
        n_masks=n, mask_dir=str(MASK2),
        n_sampled=len(sample),
        mean_fg_pct=float(np.mean(fgs)), min_fg_pct=float(np.min(fgs)),
        max_fg_pct=float(np.max(fgs)), std_fg_pct=float(np.std(fgs)),
        overlays=str(OVL))
    STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"[masks] wrote {STATS} + overlays -> {OVL}")


if __name__ == "__main__":
    main()
