"""gui2/sam_service.py — SAM 2.1 mask service for the UI-3 "Mask review" step.

Two operations (design GUI_REDESIGN.md §3③):

  (a) prompt_frame(image, points, labels) -> binary mask       [interactive, in-process]
      Uses the SAM 2.1 IMAGE predictor (via core.masking). Sub-second per click after the
      first warm-up; drives the live "click include / exclude" edit loop in the app.

  (b) propagate(frames_dir, seed_idx, points, labels, out_masks_dir)   [heavy, subprocess]
      Uses the SAM 2.1 VIDEO predictor to track the clicked object across the whole orbit
      sequence (frames ordered by orbit index = a video), forward AND backward from the
      edited seed frame. Mirrors gui2/preview.py: runs as `python -m gui2.sam_service
      propagate ...` so a 588-frame track never blocks/crashes the Gradio server and is
      tree-killable.

Design decisions (see _infusion/ui3_RECORD.md):
  - Frames are downsampled to `--max_side` (default 1024 = the model's internal
    `image_size`) before the video predictor loads them. This keeps a 588-frame orbit
    light on VRAM and fast to load, at zero quality cost (SAM 2 squashes every frame to
    1024x1024 internally regardless). Output masks come back at the fed resolution; mask
    resolution is irrelevant downstream because both the trainer (gsplat_trainer.py:426)
    and delivery (delivery.mask_in_render_frame) cv2.resize masks to the image.
  - The connected-components CUDA extension (`sam2._C`) is intentionally NOT built on
    sm_120 (SAM 2 install ran with SAM2_BUILD_CUDA=0). The video predictor's
    apply_postprocessing sets fill_hole_area=8, which would call _C; we override
    `predictor.fill_hole_area = 0` after build so it never does. Everything else about
    SAM 2 is plain attention / SDPA, which works on sm_120.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# SAM 2.1 config/checkpoint mapping (mirrors core.masking.load_sam2's config_map).
_VIDEO_CONFIG_MAP = {
    "sam2.1_hiera_large": ("configs/sam2.1/sam2.1_hiera_l.yaml",
                           "checkpoints/sam2.1_hiera_large.pt"),
    "sam2.1_hiera_base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml",
                               "checkpoints/sam2.1_hiera_base_plus.pt"),
    "sam2.1_hiera_small": ("configs/sam2.1/sam2.1_hiera_s.yaml",
                           "checkpoints/sam2.1_hiera_small.pt"),
    "sam2.1_hiera_tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml",
                          "checkpoints/sam2.1_hiera_tiny.pt"),
}

DEFAULT_CHECKPOINT = "sam2.1_hiera_large"
DEFAULT_MAX_SIDE = 1024
INDEX_MAP_NAME = "index_map.json"
# Frames a corrective (non-primary) prompt may override on each side of its seed. SAM 2
# tracking is reliable near a good anchor and degrades with distance, so a correction on
# a hard view fixes its local window; the primary seed keeps the rest.
DEFAULT_CORRECTION_RADIUS = 45

# Module-level video predictor cache (subprocess lives briefly, but a warm cache helps
# if the process ever runs multiple propagations).
_video_predictor = None
_video_checkpoint = None


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def sam2_available() -> bool:
    """True iff SAM 2.1 (image + video predictors) is importable."""
    try:
        from sam2.build_sam import build_sam2_video_predictor  # noqa: F401
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_ckpt(rel_path: str) -> str:
    """Resolve a checkpoint path relative to the sam2 repo root (mirrors core.masking)."""
    import sam2
    sam2_root = Path(sam2.__file__).parent.parent
    return str(sam2_root / rel_path)


# ---------------------------------------------------------------------------
# (a) Interactive single-frame prompting — delegates to core.masking (image predictor)
# ---------------------------------------------------------------------------

def prompt_frame(
    image: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    image_key: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Run SAM 2.1 point-prompt segmentation on ONE frame (interactive).

    Args:
        image: RGB image [H, W, 3] uint8 (the working-resolution frame the user clicked).
        points: Nx2 (x, y) pixel coords in `image`.
        labels: N labels — 1 = include (foreground), 0 = exclude (background).
        image_key: stable identity for set_image() caching across clicks on the same frame.

    Returns:
        Binary mask [H, W] uint8 (0/255), or None if SAM 2.1 is unavailable / failed.
    """
    from core.masking import load_sam2, predict_mask_sam2
    predictor = load_sam2(checkpoint=checkpoint, device=device)
    if predictor is None:
        return None
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    lbl = np.asarray(labels, dtype=np.int32).reshape(-1)
    if pts.shape[0] == 0:
        return None
    return predict_mask_sam2(predictor, image, pts, lbl, image_path=image_key)


# ---------------------------------------------------------------------------
# Frame preparation — build the numeric-named JPEG "video" SAM 2 needs
# ---------------------------------------------------------------------------

def prepare_frames(
    image_paths: list[str | Path],
    out_dir: str | Path,
    max_side: int = DEFAULT_MAX_SIDE,
    force: bool = False,
) -> list[str]:
    """Downsample + rename an ordered image list into a SAM 2 video directory.

    SAM 2's `init_state(video_path=<dir>)` requires JPEG frames whose stems are integers
    (it sorts by `int(stem)`). We write `<out_dir>/{idx:05d}.jpg` in the given order and a
    sidecar `index_map.json` = the list of original COLMAP stems, so propagation can name
    each output mask by its original stem.

    Args:
        image_paths: ordered source image paths (orbit order = capture/filename order).
        out_dir: destination frame dir.
        max_side: longest edge of the written frames (0 = keep original size).

    Returns:
        The ordered list of original stems (== index_map).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems = [Path(p).stem for p in image_paths]

    map_path = out_dir / INDEX_MAP_NAME
    if not force and map_path.is_file():
        try:
            existing = json.loads(map_path.read_text())
            n_jpg = len(list(out_dir.glob("*.jpg")))
            if existing == stems and n_jpg == len(stems):
                return stems  # already prepared, identical order
        except Exception:
            pass

    # (Re)build cleanly.
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    for i, p in enumerate(image_paths):
        img = Image.open(str(p))
        # draft() lets libjpeg decode big JPEGs at a reduced scale (powers of 2) — a large
        # speedup on 60 MP source frames before we resize to the exact working size. Sony
        # cameras write MPO (multi-picture JPEG), so accept that too; draft no-ops safely
        # on formats that don't support it.
        if max_side and img.format in ("JPEG", "MPO"):
            try:
                img.draft("RGB", (max_side, max_side))
            except Exception:
                pass
        img = img.convert("RGB")
        if max_side and max(img.size) > max_side:
            scale = max_side / max(img.size)
            new_size = (max(1, round(img.size[0] * scale)),
                        max(1, round(img.size[1] * scale)))
            img = img.resize(new_size, Image.LANCZOS)
        img.save(str(out_dir / f"{i:05d}.jpg"), "JPEG", quality=95)
    map_path.write_text(json.dumps(stems))
    return stems


def load_index_map(frames_dir: str | Path) -> list[str]:
    """Load the ordered original stems written by prepare_frames."""
    return json.loads((Path(frames_dir) / INDEX_MAP_NAME).read_text())


def frame_path(frames_dir: str | Path, idx: int) -> str:
    """Path to the working-resolution frame at orbit index `idx`."""
    return str(Path(frames_dir) / f"{idx:05d}.jpg")


# ---------------------------------------------------------------------------
# (b) Video propagation — track the clicked object across the whole orbit
# ---------------------------------------------------------------------------

def load_video_predictor(checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cuda"):
    """Build (and cache) the SAM 2.1 video predictor with _C-free postprocessing."""
    global _video_predictor, _video_checkpoint
    if _video_predictor is not None and _video_checkpoint == checkpoint:
        return _video_predictor
    if not sam2_available():
        return None
    if checkpoint not in _VIDEO_CONFIG_MAP:
        raise ValueError(f"unknown SAM 2.1 checkpoint: {checkpoint}")
    from sam2.build_sam import build_sam2_video_predictor
    cfg_name, ckpt_rel = _VIDEO_CONFIG_MAP[checkpoint]
    ckpt_path = _resolve_ckpt(ckpt_rel)
    predictor = build_sam2_video_predictor(cfg_name, ckpt_path, device=device)
    # Disable the hole-fill post-process that would call the un-built _C extension.
    predictor.fill_hole_area = 0
    _video_predictor = predictor
    _video_checkpoint = checkpoint
    return predictor


def normalize_prompts(prompts=None, seed_idx=None, points=None, labels=None) -> list[dict]:
    """Accept either the multi-frame `prompts` list or the legacy single-seed args.

    Returns a list of {"frame_idx": int, "points": [[x,y]...], "labels": [1,0...]}.
    Each entry is one frame the user edited; SAM 2 treats them all as conditioning
    frames (this is the design's iterate loop — edit frame A, propagate, see frame B is
    off, add a click on B, re-propagate).
    """
    if prompts:
        out = []
        for p in prompts:
            entry = {
                "frame_idx": int(p["frame_idx"]),
                "points": np.asarray(p["points"], dtype=np.float32).reshape(-1, 2),
                "labels": np.asarray(p["labels"], dtype=np.int32).reshape(-1),
            }
            if p.get("radius") is not None:
                entry["radius"] = int(p["radius"])
            out.append(entry)
        return out
    if seed_idx is None:
        raise ValueError("propagate needs either `prompts` or `seed_idx`/points/labels")
    return [{
        "frame_idx": int(seed_idx),
        "points": np.asarray(points, dtype=np.float32).reshape(-1, 2),
        "labels": np.asarray(labels, dtype=np.int32).reshape(-1),
    }]


def propagate(
    frames_dir: str | Path,
    prompts=None,
    out_masks_dir: str | Path = "masks",
    *,
    seed_idx: int = None,
    points=None,
    labels=None,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cuda",
    offload_video_to_cpu: bool = False,
    offload_state_to_cpu: bool = False,
    obj_id: int = 1,
    progress_cb=None,
) -> dict:
    """Propagate one or more per-frame point-prompts across the whole orbit.

    `prompts` is a list of {"frame_idx", "points", "labels"} (one per edited frame); the
    legacy `seed_idx`/`points`/`labels` single-frame form is also accepted. Frames must be
    a SAM 2 video dir (see prepare_frames). Masks -> `<out_masks_dir>/<stem>.png` (0/255).

    Returns a stats dict: n_frames, n_masks, seconds, fps, peak_vram_gb, seed_idxs, offload.
    """
    import torch

    frames_dir = Path(frames_dir)
    out_masks_dir = Path(out_masks_dir)
    out_masks_dir.mkdir(parents=True, exist_ok=True)
    stems = load_index_map(frames_dir)
    n_frames = len(stems)

    prompt_list = normalize_prompts(prompts, seed_idx, points, labels)

    predictor = load_video_predictor(checkpoint, device)
    if predictor is None:
        raise RuntimeError("SAM 2.1 video predictor unavailable (sam2 not installed)")

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    # bfloat16 autocast is SAM 2's recommended inference dtype (sm_120: plain SDPA).
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device.startswith("cuda") else _nullcontext())
    written = 0
    with torch.inference_mode(), autocast:
        state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        # Propagate from EACH edited frame independently (forward + reverse), then let the
        # nearest seed win per frame (circular orbit distance). SAM 2 propagates
        # sequentially from the min conditioning frame, so mixing several conditioning
        # frames in one pass lets a late/hard correction hijack far regions. Running one
        # track per seed and merging by nearest-anchor keeps each correction LOCAL — which
        # is exactly the design's "fix the bad spot-check views" loop.
        per_seed: list[dict[int, np.ndarray]] = []
        for p in prompt_list:
            predictor.reset_state(state)
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=p["frame_idx"], obj_id=obj_id,
                points=p["points"], labels=p["labels"])
            masks: dict[int, np.ndarray] = {}
            for reverse in (False, True):
                for fidx, _obj_ids, mask_logits in predictor.propagate_in_video(
                        state, start_frame_idx=p["frame_idx"], reverse=reverse):
                    masks[int(fidx)] = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
            per_seed.append(masks)

    seed_idxs = [p["frame_idx"] for p in prompt_list]
    # Per-seed claim radius (linear frame distance). The FIRST prompt is the primary
    # anchor (unlimited claim); later prompts are LOCAL corrections that only override a
    # neighbourhood — so a click on a hard view (e.g. the sole) fixes its window without
    # degrading the well-tracked far frames. SAM 2 propagation is linear over the frame
    # sequence, so mask quality falls off with |frame - seed|, not orbit-angle distance.
    radii = [p.get("radius", (None if i == 0 else DEFAULT_CORRECTION_RADIUS))
             for i, p in enumerate(prompt_list)]

    def _pick_seed(fidx: int) -> int:
        best, bd = None, None
        for si, s in enumerate(seed_idxs):
            d = abs(fidx - s)
            if radii[si] is not None and d > radii[si]:
                continue
            if fidx not in per_seed[si]:
                continue
            if bd is None or d < bd:
                bd, best = d, si
        if best is None:  # nobody eligible -> any seed that reached this frame
            for si in range(len(per_seed)):
                if fidx in per_seed[si]:
                    return si
        return best if best is not None else 0

    for fidx in range(n_frames):
        si = _pick_seed(fidx)
        m = per_seed[si].get(fidx)
        if m is None:
            continue
        stem = stems[fidx]
        if cv2.imwrite(str(out_masks_dir / f"{stem}.png"), m):
            written += 1
        if progress_cb is not None:
            progress_cb(written, n_frames)

    seconds = time.time() - t0
    peak = 0.0
    if device.startswith("cuda") and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return {
        "n_frames": n_frames,
        "n_masks": written,
        "seconds": round(seconds, 2),
        "fps": round(n_frames * len(prompt_list) / seconds, 2) if seconds > 0 else 0.0,
        "peak_vram_gb": round(peak, 2),
        "seed_idxs": seed_idxs,
        "offload_video_to_cpu": offload_video_to_cpu,
        "offload_state_to_cpu": offload_state_to_cpu,
    }


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# Subprocess entry point (mirrors gui2/preview.py)
# ---------------------------------------------------------------------------

def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="gui2 SAM 2.1 mask service")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("propagate", help="track a point-prompt across the orbit")
    pp.add_argument("--frames_dir", required=True,
                    help="SAM 2 video dir ({idx:05d}.jpg + index_map.json)")
    pp.add_argument("--prompt", required=True,
                    help="JSON file: {seed_idx, points:[[x,y]...], labels:[1,0...]}")
    pp.add_argument("--out_masks_dir", required=True)
    pp.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    pp.add_argument("--gpu", default="0")
    pp.add_argument("--offload_video", action="store_true",
                    help="keep frames in CPU RAM (safer alongside training; a bit slower)")
    pp.add_argument("--offload_state", action="store_true")
    pp.add_argument("--stats_out", default="", help="write the stats dict JSON here")

    # `prepare` is exposed for tests / standalone use; the app calls prepare_frames directly.
    pr = sub.add_parser("prepare", help="build a SAM 2 video dir from an image list")
    pr.add_argument("--images_json", required=True, help="JSON list of ordered image paths")
    pr.add_argument("--out_dir", required=True)
    pr.add_argument("--max_side", type=int, default=DEFAULT_MAX_SIDE)

    args = ap.parse_args(argv)

    if args.cmd == "prepare":
        image_paths = json.loads(Path(args.images_json).read_text())
        stems = prepare_frames(image_paths, args.out_dir, max_side=args.max_side)
        print(json.dumps({"n_frames": len(stems)}))
        return 0

    if args.cmd == "propagate":
        prompt = json.loads(Path(args.prompt).read_text())
        # Accept {prompts:[{frame_idx,points,labels}...]} or legacy {seed_idx,points,labels}.
        prompts = prompt.get("prompts")
        if prompts is None:
            prompts = [{"frame_idx": prompt["seed_idx"],
                        "points": prompt["points"], "labels": prompt["labels"]}]
        seeds = [p["frame_idx"] for p in prompts]
        npts = sum(len(p["points"]) for p in prompts)
        print(f"[sam] propagate seeds={seeds} points={npts} "
              f"frames_dir={args.frames_dir}", flush=True)

        def _cb(done, total):
            if done % 50 == 0 or done == total:
                print(f"[sam] saved {done}/{total} masks", flush=True)

        # GPU selection mirrors gui2/preview.py: pick the device explicitly via torch
        # (set_device + cuda:<idx>) instead of mutating CUDA_VISIBLE_DEVICES, so both
        # gui2 GPU subprocesses use ONE pattern.
        import torch
        gpu = str(args.gpu)
        device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(int(gpu))
            except (ValueError, RuntimeError):
                pass
        stats = propagate(
            args.frames_dir, prompts, args.out_masks_dir,
            checkpoint=args.checkpoint, device=device,
            offload_video_to_cpu=args.offload_video,
            offload_state_to_cpu=args.offload_state,
            progress_cb=_cb)
        print("[sam] " + json.dumps(stats), flush=True)
        if args.stats_out:
            Path(args.stats_out).write_text(json.dumps(stats, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
