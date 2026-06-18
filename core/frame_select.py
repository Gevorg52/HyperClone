"""Sharpest-frame selection for video input (imperfect-capture pipeline, lever #0).

Handheld orbit video has motion blur that varies frame-to-frame. Picking the SHARPEST
frame per short window — while enforcing angular/temporal spacing so orbit COVERAGE is
preserved — feeds COLMAP + 3DGS crisp inputs at near-zero cost. var-of-Laplacian is the
standard sharpness proxy (high = crisp, low = blurry). FAITHFUL, trivial, highest
ROI/lowest risk preprocessing step (research-backed, June 2026). Run BEFORE SfM.

select_sharp_frames(paths, window, min_gap) -> selected paths + report.
Also accepts a video file (extracts frames first).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import cv2


@dataclass
class FrameSelectResult:
    selected: list           # selected frame paths (or indices for a video)
    sharpness: np.ndarray    # var-of-Laplacian per input frame
    report: str


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _sharpness_of(path: str, max_side: int = 800) -> float:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return laplacian_var(img)


def select_indices(sharpness: np.ndarray, window: int = 8, min_gap: int = 3,
                   keep_frac: float = 1.0) -> list[int]:
    """Pick the sharpest frame per non-overlapping window, enforce a minimum index gap
    (preserves orbit coverage / avoids clustering on one sharp burst)."""
    n = len(sharpness)
    cand = []
    for s in range(0, n, window):
        seg = sharpness[s:s + window]
        if len(seg):
            cand.append(s + int(np.argmax(seg)))
    cand.sort()
    out = []
    for i in cand:
        if not out or i - out[-1] >= min_gap:
            out.append(i)
    if keep_frac < 1.0 and out:                       # optionally drop the blurriest kept
        thr = np.quantile(sharpness[out], 1 - keep_frac)
        out = [i for i in out if sharpness[i] >= thr]
    return out


def select_sharp_frames(frame_paths: list[str], window: int = 8, min_gap: int = 3,
                        keep_frac: float = 1.0) -> FrameSelectResult:
    sharp = np.array([_sharpness_of(p) for p in frame_paths])
    idx = select_indices(sharp, window, min_gap, keep_frac)
    sel = [frame_paths[i] for i in idx]
    med, lo, hi = np.median(sharp), sharp.min(), sharp.max()
    sel_med = np.median(sharp[idx]) if idx else 0
    report = (
        f"Frames: {len(frame_paths)} in -> {len(sel)} selected "
        f"(window {window}, min_gap {min_gap}).\n"
        f"Sharpness (var-of-Laplacian): min {lo:.0f}, median {med:.0f}, max {hi:.0f}.\n"
        f"Selected-frame median sharpness {sel_med:.0f} "
        f"(+{100*(sel_med/med-1):.0f}% vs all-frame median).\n"
        f"(crisp photo ~ 500-2000+; low values = motion/defocus blur.)")
    return FrameSelectResult(selected=sel, sharpness=sharp, report=report)


def extract_video_frames(video_path: str, out_dir: str, every: int = 1) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    paths = []; i = 0; k = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            p = os.path.join(out_dir, f"frame_{k:06d}.png")
            cv2.imwrite(p, frame); paths.append(p); k += 1
        i += 1
    cap.release()
    return paths


if __name__ == "__main__":
    import sys, glob
    src = sys.argv[1] if len(sys.argv) > 1 else \
        r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\data\images_4"
    win = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    paths = sorted(glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.png")) +
                   glob.glob(os.path.join(src, "*.JPG")))
    res = select_sharp_frames(paths, window=win, min_gap=3)
    print(res.report)
    # blur histogram viz
    import numpy as np
    s = res.sharpness; H, W = 200, 600
    img = np.full((H, W, 3), 28, np.uint8)
    if len(s):
        sn = np.clip(s / (np.percentile(s, 98) + 1e-6), 0, 1)
        sel_set = set(res.selected)
        for i, v in enumerate(sn):
            x = int(i / max(len(sn) - 1, 1) * (W - 1)); y = int(H - v * (H - 20))
            c = (90, 220, 90) if paths[i] in sel_set else (80, 80, 80)
            cv2.line(img, (x, H), (x, y), c, 1)
    cv2.putText(img, "sharpness per frame (green=selected)", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    out = r"F:\HYPERclone\apex-sfm\_infusion\frame_select.png"
    os.makedirs(os.path.dirname(out), exist_ok=True); cv2.imwrite(out, img)
    print("viz:", out)
