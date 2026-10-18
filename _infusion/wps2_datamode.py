"""WP-S2 step 1: VERIFY the actual production data mode for GPU_V4.

The queued WP-S2 assumed production = object-only pre-multiplied black-bg. WP-G1
found the opposite (frames are FULL SCENES). This script measures it directly and
makes the record self-evident about what each arm trains on.

Measures, over the factor-2 training frames the trainer actually reads
(training/gsplat_prod/data/images_2):
  - near-black pixel fraction  (max(R,G,B) < 16/255) per frame + aggregate
  - mean corner luminance (a black-bg premultiply would leave corners ~0)
  - mean full-frame luminance
and writes a labelled sample strip (full frame + a corner crop) so a reviewer can
SEE whether the background is present.

Usage: python _infusion/wps2_datamode.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG2 = ROOT / "_dataset/GPU_V4/apex_images/training/gsplat_prod/data/images_2"
RAW = ROOT / "_dataset/GPU_V4/apex_images/export/images"
OUT_JSON = ROOT / "qa/wps2_datamode.json"
OUT_STRIP = ROOT / "qa/wps2_datamode_strip.jpg"

NEAR_BLACK = 16  # uint8 threshold on max channel


def frame_stats(bgr: np.ndarray) -> dict:
    h, w = bgr.shape[:2]
    maxc = bgr.max(axis=2)                       # (H,W) max channel
    nb = float((maxc < NEAR_BLACK).mean()) * 100.0
    lum = bgr.astype(np.float32).mean() / 255.0  # gray-ish mean over all channels
    # 4 corners, 12% box each
    ch, cw = int(0.12 * h), int(0.12 * w)
    corners = [bgr[:ch, :cw], bgr[:ch, -cw:], bgr[-ch:, :cw], bgr[-ch:, -cw:]]
    corner_lum = float(np.mean([c.astype(np.float32).mean() for c in corners]))
    return dict(near_black_pct=nb, mean_lum=lum, corner_lum=corner_lum)


def main():
    files = sorted(IMG2.glob("*.png"))
    assert files, f"no images at {IMG2}"
    print(f"[datamode] {len(files)} factor-2 training frames at {IMG2}")
    h, w = cv2.imread(str(files[0])).shape[:2]
    print(f"[datamode] frame HxW = {h}x{w}")

    # measure a stride sample of 60 frames for the aggregate (fast, representative)
    sample = files[:: max(1, len(files) // 60)]
    per = [frame_stats(cv2.imread(str(f))) for f in sample]
    agg = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
    agg_std = {k + "_std": float(np.std([p[k] for p in per])) for k in per[0]}

    verdict = ("FULL_SCENE" if agg["near_black_pct"] < 60 and agg["corner_lum"] > 20
               else "BLACK_BG_PREMULTIPLIED")

    result = dict(
        note="near_black = max(R,G,B)<16/255; corner_lum/mean over 0-255 avg-of-3ch",
        images_dir=str(IMG2), frame_h=h, frame_w=w,
        n_frames=len(files), n_sampled=len(sample),
        near_black_pct=agg["near_black_pct"],
        mean_lum_0_255=agg["mean_lum"] * 255.0,
        corner_lum_0_255=agg["corner_lum"],
        **agg_std,
        DATA_MODE_VERDICT=verdict,
        interpretation=(
            "FULL_SCENE => arm (a) full-scene = CURRENT PRODUCTION (control); "
            "object-only (b/c) is the NEW thing under test. Confirms WP-G1 "
            "correction; refutes the DEVELOPER_NOTES 'pre-multiplied black-bg' claim."))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

    # sample strip: 4 frames, full (downscaled) over a corner crop
    picks = [files[0], files[len(files) // 3], files[2 * len(files) // 3], files[-1]]
    rows = []
    for f in picks:
        im = cv2.imread(str(f))
        small = cv2.resize(im, (400, int(400 * h / w)))
        ch, cw = int(0.20 * im.shape[0]), int(0.20 * im.shape[1])
        corner = cv2.resize(im[:ch, :cw], (400, int(400 * ch / cw)))
        cv2.putText(small, f.stem, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(corner, "TL corner", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
        rows.append(np.vstack([small, corner]))
    cv2.imwrite(str(OUT_STRIP), np.hstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[datamode] wrote {OUT_JSON}\n[datamode] wrote {OUT_STRIP}")


if __name__ == "__main__":
    main()
