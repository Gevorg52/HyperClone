"""gui2/review.py — UI-3 Mask Review logic (non-Gradio, unit-testable).

The Step-③ "Mask review" screen (GUI_REDESIGN.md §3③) confirms/fixes the object cut-out
so the delivered object SOG is the subject WITHOUT its capture rig (the shoe-stand case).

This module holds the pure logic; gui2/app.py holds the Gradio wiring:
  - pick ~12 representative views spread across the orbit,
  - build the current-matte overlay gallery (BiRefNet by default),
  - run SAM 2.1 point-prompt segmentation live on one frame (gui2.sam_service),
  - propagate the accepted prompt across the whole orbit (subprocess),
  - stage the accepted masks under <run_dir>/masks_reviewed/ for delivery (--mask_dir).

Masks are a DELIVERY input (WP-S2), not a training input, so review runs in parallel with
training at zero wall-clock cost. Accepting the current BiRefNet matte ("Looks right") does
NOT write masks_reviewed — delivery's --gen_masks path reproduces exactly that matte — so
masks_reviewed exists ONLY when the user fixed the cut-out with SAM.
"""

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gui2 import sam_service as SAM

_REPO = str(Path(__file__).resolve().parent.parent)

INCLUDE_TINT = (0, 255, 0)      # green — the kept object
BIREFNET_TINT = (255, 90, 90)   # red   — the current (unreviewed) matte, may include the rig
REVIEWED_TINT = (0, 255, 0)

# Written LAST (inside the atomically-swapped staging dir) by accept_reviewed; holds the
# staged mask count. runner._run_delivery requires this marker AND a matching on-disk
# .png count before it will deliver from masks_reviewed — otherwise it falls back to
# --gen_masks (BiRefNet). This makes the cross-process accept -> delivery hand-off safe.
MASKS_COMPLETE_MARKER = ".complete"


# ---------------------------------------------------------------------------
# View selection + image ordering
# ---------------------------------------------------------------------------

def spread_indices(n_frames: int, k: int = 12) -> list[int]:
    """`k` indices spread evenly across [0, n_frames) by orbit index (uniform stride).

    Orbit index == capture/filename order, so a uniform stride samples the turntable
    evenly. (Coverage-by-pose spread is available in gui2.coverage but uniform stride is
    robust and cheap, and the design permits it.)
    """
    if n_frames <= 0:
        return []
    k = max(1, min(k, n_frames))
    return [int(round(i * (n_frames - 1) / (k - 1))) if k > 1 else 0 for i in range(k)]


def ordered_images(run_dir: str | Path) -> list[str]:
    """Ordered source frames for review = the SfM-registered images (orbit order).

    Prefers <run_dir>/export/images (the COLMAP-registered set whose stems match the
    delivery masks_{factor}); falls back to <run_dir>/input_images.
    """
    run_dir = Path(run_dir)
    for sub in ("export/images", "input_images"):
        d = run_dir / sub
        if d.is_dir():
            imgs = sorted(str(p) for p in d.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if imgs:
                return imgs
    return []


# ---------------------------------------------------------------------------
# Working frames (the downsampled orbit SAM 2 / the gallery operate on)
# ---------------------------------------------------------------------------

def frames_dir(run_dir: str | Path) -> str:
    return str(Path(run_dir) / "sam_work" / "frames")


def masks_reviewed_dir(run_dir: str | Path) -> str:
    return str(Path(run_dir) / "masks_reviewed")


def prepare_working_frames(run_dir: str | Path, max_side: int = SAM.DEFAULT_MAX_SIDE):
    """Downsample the orbit into a SAM 2 video dir. Returns (frames_dir, stems, images)."""
    images = ordered_images(run_dir)
    fdir = frames_dir(run_dir)
    stems = SAM.prepare_frames(images, fdir, max_side=max_side)
    return fdir, stems, images


def load_frame(fdir: str | Path, idx: int) -> np.ndarray:
    """Load working frame `idx` as an RGB uint8 array."""
    return np.array(Image.open(SAM.frame_path(fdir, idx)).convert("RGB"))


# ---------------------------------------------------------------------------
# Current-matte (BiRefNet) gallery
# ---------------------------------------------------------------------------

def ensure_mattes(fdir: str | Path, stems: list[str], indices: list[int],
                  out_dir: str | Path, existing_mask_dir: str | None = None,
                  device: str = "cuda") -> str:
    """Ensure a BiRefNet matte exists for each `indices` frame in out_dir/<stem>.png.

    Reuses masks already present in `existing_mask_dir` (e.g. delivery's masks_{factor});
    otherwise runs BiRefNet on the working frames (cheap — only the gallery views).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo_imgs, todo_stems = [], []
    for i in indices:
        st = stems[i]
        if (out_dir / f"{st}.png").is_file():
            continue
        if existing_mask_dir and (Path(existing_mask_dir) / f"{st}.png").is_file():
            m = cv2.imread(str(Path(existing_mask_dir) / f"{st}.png"), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(out_dir / f"{st}.png"), m)
            continue
        todo_imgs.append(SAM.frame_path(fdir, i))
        todo_stems.append(st)
    if todo_imgs:
        from core.masking import load_birefnet, run_birefnet_single, unload_birefnet
        model, transform = load_birefnet("general", device=device)
        for img, st in zip(todo_imgs, todo_stems):
            run_birefnet_single(model, transform, img, str(out_dir), device=device)
            # run_birefnet_single names by the image stem ({idx:05d}); rename to COLMAP stem
            src = out_dir / f"{Path(img).stem}.png"
            if src.is_file() and src.name != f"{st}.png":
                src.replace(out_dir / f"{st}.png")
        unload_birefnet()
    return str(out_dir)


def overlay_rgb(frame_rgb: np.ndarray, mask: np.ndarray | None, tint,
                points=None, labels=None) -> np.ndarray:
    """Tinted mask overlay on a frame, with include/exclude click dots drawn."""
    from core.masking import generate_overlay
    out = frame_rgb.copy()
    if mask is not None:
        if mask.shape[:2] != frame_rgb.shape[:2]:
            mask = cv2.resize(mask, (frame_rgb.shape[1], frame_rgb.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        out = generate_overlay(out, mask, alpha=0.45, color=tuple(tint))
    if points is not None:
        for (x, y), l in zip(points, labels):
            c = (0, 210, 0) if int(l) == 1 else (255, 40, 40)
            cv2.circle(out, (int(x), int(y)), 7, c, -1)
            cv2.circle(out, (int(x), int(y)), 7, (255, 255, 255), 2)
    return out


def gallery_items(fdir: str | Path, stems: list[str], indices: list[int],
                  matte_dir: str | Path, tint=BIREFNET_TINT) -> list[tuple]:
    """List of (overlay_rgb, caption) for a gr.Gallery of the representative views."""
    items = []
    for i in indices:
        st = stems[i]
        frame = load_frame(fdir, i)
        mp = Path(matte_dir) / f"{st}.png"
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.is_file() else None
        items.append((overlay_rgb(frame, mask, tint), f"{st} (view {i})"))
    return items


# ---------------------------------------------------------------------------
# Propagation (subprocess) + accept
# ---------------------------------------------------------------------------

def propagate_subprocess(fdir: str | Path, prompts: list[dict],
                         out_masks_dir: str | Path, stats_out: str | Path,
                         gpu: str = "0", offload_video: bool = False,
                         log_file=None) -> subprocess.Popen:
    """Launch `python -m gui2.sam_service propagate` (mirrors gui2/preview.py plumbing)."""
    import json
    prompt_path = Path(fdir).parent / "prompt.json"
    prompt_path.write_text(json.dumps({"prompts": prompts}))
    cmd = [sys.executable, "-m", "gui2.sam_service", "propagate",
           "--frames_dir", str(fdir), "--prompt", str(prompt_path),
           "--out_masks_dir", str(out_masks_dir), "--gpu", str(gpu),
           "--stats_out", str(stats_out)]
    if offload_video:
        cmd.append("--offload_video")
    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(cmd, cwd=_REPO, stdout=(log_file or subprocess.PIPE),
                            stderr=subprocess.STDOUT, text=True, env=env)


def accept_reviewed(run_dir: str | Path, src_mask_dir: str | Path,
                    stems: list[str]) -> tuple[str, int]:
    """ATOMICALLY stage the accepted SAM masks into <run_dir>/masks_reviewed/.

    The masks (~one per COLMAP stem) are copied into a sibling masks_reviewed.tmp/
    first; a MASKS_COMPLETE_MARKER file (holding the staged count) is written LAST,
    inside the staging dir; then the whole dir is swapped into place with os.replace.
    So a cross-process reader (runner._run_delivery) never observes a half-copied
    masks_reviewed — it sees either the previous dir or the fully-staged new one
    (marker + all masks together). Delivery consumes the result via --mask_dir.
    Returns (masks_reviewed_dir, n_written).
    """
    import shutil
    dst = Path(masks_reviewed_dir(run_dir))
    tmp = Path(str(dst) + ".tmp")
    old = Path(str(dst) + ".old")

    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    n = 0
    for st in stems:
        src = Path(src_mask_dir) / f"{st}.png"
        if src.is_file():
            shutil.copyfile(src, tmp / f"{st}.png")
            n += 1
    # Completion marker LAST, inside the staging dir, so it swaps in atomically with
    # the masks it certifies.
    (tmp / MASKS_COMPLETE_MARKER).write_text(str(n))

    # Atomic swap. os.replace cannot overwrite an existing dir on Windows, so move any
    # current masks_reviewed aside first, rename the staged dir in, then delete the old.
    # The only non-atomic window (dst briefly absent) makes a concurrent reader fall
    # back to --gen_masks, never read a partial dir.
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)
    if dst.exists():
        os.replace(str(dst), str(old))
    os.replace(str(tmp), str(dst))
    shutil.rmtree(old, ignore_errors=True)
    return str(dst), n


def clear_reviewed(run_dir: str | Path) -> None:
    """"Looks right" path: no reviewed masks -> delivery falls back to BiRefNet (--gen_masks)."""
    import shutil
    d = Path(masks_reviewed_dir(run_dir))
    for p in (d, Path(str(d) + ".tmp"), Path(str(d) + ".old")):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
