"""gui2 — Capture QC card (design §3 ①).

Runs in seconds on input selection, BEFORE the user commits hours: frame count,
resolution / megapixels, a sharpness histogram (var-of-Laplacian on a ~40-frame
sample, reusing core.frame_select), video detection + extraction offer, a crude
end-to-end time estimate, and the auto-chosen data_factor ("Training at N MP").

Nothing here trains or reconstructs — it only probes the input. The heavy work is
the runner's job (gui2/runner.py).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from core.frame_select import (_sharpness_of, extract_video_frames,
                               select_sharp_frames)
from core.utils import get_image_list
from gui2 import defaults as D

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}

# WP-T1 datapoint: 588 frames ~ 6 h end-to-end -> crude linear per-frame estimate.
_MIN_PER_FRAME = 360.0 / 588.0  # ~0.612 min/frame


@dataclass
class QCReport:
    ok: bool
    kind: str                       # "images" | "video" | "empty" | "error"
    message: str
    n_frames: int = 0
    width: int = 0
    height: int = 0
    megapixels: float = 0.0
    is_video: bool = False
    video_path: str | None = None
    sharpness: list[float] = field(default_factory=list)
    sharpness_sample: int = 0
    hist_png: str | None = None
    est_minutes: float = 0.0
    data_factor: int = 1
    training_mp: float = 0.0
    summary_md: str = ""


def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def _first_image_resolution(paths: list[str]) -> tuple[int, int]:
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    return 0, 0


def _sample_paths(paths: list[str], n: int) -> list[str]:
    if len(paths) <= n:
        return list(paths)
    idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
    return [paths[i] for i in sorted(set(idx.tolist()))]


def render_sharpness_hist(sharpness: list[float], out_png: str,
                          width: int = 640, height: int = 220) -> str:
    """Draw a compact sharpness histogram PNG (green bars) — reuses the frame_select
    viz idiom (cv2, no matplotlib). Returns out_png."""
    s = np.asarray(sharpness, dtype=np.float64)
    img = np.full((height, width, 3), 24, np.uint8)
    if s.size:
        nbins = min(24, max(6, s.size // 2))
        hi = float(np.percentile(s, 98)) if s.size > 2 else float(s.max() + 1e-6)
        hi = max(hi, 1e-6)
        counts, _ = np.histogram(np.clip(s, 0, hi), bins=nbins, range=(0, hi))
        cmax = max(int(counts.max()), 1)
        pad = 12
        bw = (width - 2 * pad) / nbins
        for i, c in enumerate(counts):
            x0 = int(pad + i * bw)
            x1 = int(pad + (i + 1) * bw) - 2
            bh = int((height - 40) * (c / cmax))
            cv2.rectangle(img, (x0, height - 20), (x1, height - 20 - bh),
                          (90, 210, 120), -1)
        med = float(np.median(s))
        cv2.putText(img, f"sharpness (var-Laplacian)  median={med:.0f}",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(img, "blurry", (pad, height - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(img, "sharp", (width - 60, height - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_png, img)
    return out_png


def _fmt_eta(minutes: float) -> str:
    if minutes < 90:
        return f"~{minutes:.0f} min"
    return f"~{minutes / 60:.1f} h"


def run_qc(input_path: str, mode: str, subject: str = "",
           work_dir: str | None = None, sample: int = 40,
           detail: bool = False) -> QCReport:
    """Probe a folder-of-images or a video file and return a QCReport.

    For a video, frames are NOT extracted here (that is an explicit user action via
    `extract_frames`); instead the report flags it and estimates from a decoded count.
    """
    if not input_path or not os.path.exists(input_path):
        return QCReport(ok=False, kind="error",
                        message="Select a folder of images or a video file.")

    work = Path(work_dir) if work_dir else Path(input_path).parent / "_qc"
    work.mkdir(parents=True, exist_ok=True)

    # ---- Video: probe frame count + first-frame resolution, offer extraction ----
    if os.path.isfile(input_path) and is_video_file(input_path):
        cap = cv2.VideoCapture(input_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        # sharpness on a decoded sample (do not write full frames yet)
        sh: list[float] = []
        if n > 0:
            step = max(1, n // sample)
            i = 0
            while len(sh) < sample:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = cap.read()
                if not ok:
                    break
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                s = 800 / max(g.shape[:2])
                if s < 1:
                    g = cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)))
                sh.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
                i += step
        cap.release()
        mp = (w * h) / 1e6
        d = D.defaults_for(mode, detail)
        factor = D.resolve_data_factor(d, mp)
        tmp = D.training_mp(mp, factor)
        hist = render_sharpness_hist(sh, str(work / "qc_hist.png")) if sh else None
        est = n * _MIN_PER_FRAME
        summary = _summary_md(True, n, w, h, mp, factor, tmp, est, sh, mode)
        return QCReport(
            ok=True, kind="video",
            message=f"Video detected: {n} frames at {w}x{h}. "
                    "Click 'Extract frames' to prepare it for scanning.",
            n_frames=n, width=w, height=h, megapixels=mp, is_video=True,
            video_path=input_path, sharpness=sh, sharpness_sample=len(sh),
            hist_png=hist, est_minutes=est, data_factor=factor,
            training_mp=tmp, summary_md=summary)

    # ---- Folder of images ----
    if not os.path.isdir(input_path):
        return QCReport(ok=False, kind="error",
                        message=f"Not a folder or supported video: {input_path}")
    try:
        paths = [str(p) for p in get_image_list(input_path)]
    except FileNotFoundError as e:
        return QCReport(ok=False, kind="error", message=str(e))
    if not paths:
        return QCReport(ok=False, kind="empty",
                        message=f"No images found in {input_path}.")

    w, h = _first_image_resolution(paths)
    mp = (w * h) / 1e6
    sample_paths = _sample_paths(paths, sample)
    sh = [_sharpness_of(p) for p in sample_paths]
    hist = render_sharpness_hist(sh, str(work / "qc_hist.png"))
    d = D.defaults_for(mode, detail)
    factor = D.resolve_data_factor(d, mp)
    tmp = D.training_mp(mp, factor)
    est = len(paths) * _MIN_PER_FRAME
    summary = _summary_md(False, len(paths), w, h, mp, factor, tmp, est, sh, mode)
    return QCReport(
        ok=True, kind="images",
        message=f"{len(paths)} images at {w}x{h} ({mp:.1f} MP).",
        n_frames=len(paths), width=w, height=h, megapixels=mp, is_video=False,
        sharpness=sh, sharpness_sample=len(sh), hist_png=hist,
        est_minutes=est, data_factor=factor, training_mp=tmp, summary_md=summary)


def extract_frames(video_path: str, out_dir: str, every: int = 1) -> list[str]:
    """Extract every Nth frame of a video to PNGs. Thin wrapper over frame_select."""
    return extract_video_frames(video_path, out_dir, every=every)


def extract_and_select_frames(video_path: str, out_dir: str, window: int = 8,
                              min_gap: int = 3) -> tuple[list[str], str]:
    """Extract ALL frames of a video, keep the SHARPEST per short window (frame_select
    lever #0 — preserves orbit coverage, drops motion-blurred frames), and write the
    kept frames into a freshly-CLEARED out_dir. Returns (kept paths, report).

    Preferred over raw every-Nth extraction: handheld orbits have frame-to-frame motion
    blur, so sharpest-per-window feeds SfM/3DGS crisp inputs at near-zero cost.
    """
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)          # clear the extraction dir first
    out.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="apex_vidframes_")
    try:
        allframes = extract_video_frames(video_path, tmp, every=1)
        if not allframes:
            return [], "No frames decoded from the video."
        res = select_sharp_frames(allframes, window=window, min_gap=min_gap)
        kept = []
        for p in res.selected:
            dst = out / Path(p).name
            shutil.copy2(p, str(dst))
            kept.append(str(dst))
        return kept, res.report
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _summary_md(is_video: bool, n: int, w: int, h: int, mp: float, factor: int,
                tmp: float, est: float, sh: list[float], mode: str) -> str:
    med = float(np.median(sh)) if sh else 0.0
    lo = float(np.min(sh)) if sh else 0.0
    blur_note = ""
    if sh:
        blurry = int(np.sum(np.asarray(sh) < 0.4 * (med + 1e-6)))
        if blurry:
            blur_note = (f"\n- {blurry}/{len(sh)} sampled frames look soft "
                         "(motion/defocus blur).")
    coverage = ""
    if mode == "object":
        coverage = ("\n- Coverage: single-ring orbits give a weaker underside — "
                    "add a few low/high angles if you can.")
    src = "video" if is_video else "images"
    return (
        f"### Capture check\n"
        f"- **{n} {'frames' if is_video else 'images'}** ({src}) at "
        f"**{w}x{h}** = **{mp:.1f} MP**\n"
        f"- Auto resolution: **factor {factor}** -> **training at {tmp:.1f} MP** "
        f"(recommended)\n"
        f"- Sharpness (sample of {len(sh)}): median {med:.0f}, min {lo:.0f}"
        f"{blur_note}\n"
        f"- Estimated processing time: **{_fmt_eta(est)}** (crude, linear in frames)"
        f"{coverage}"
    )
