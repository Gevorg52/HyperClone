"""gui2 — live training preview (UI-2, design §3 ②).

Renders a 6-frame turntable STRIP from a training checkpoint, and (on demand)
exports the current checkpoint to a viewer-ready SOG. Both are meant to run as a
SUBPROCESS out of the trainer's process (a brief, seconds-long GPU burst) so the
trainer never stalls on preview work.

Usage (subprocess entry points):
    python -m gui2.preview strip  --ckpt <ckpt.pt> --data_dir <colmap data dir>
                                  --out <strip.png> --step N --factor F [--frames 6]
                                  [--height 360] [--gpu 0]
    python -m gui2.preview export --ckpt <ckpt.pt> --ply_dir <output/ply>
                                  --out <viewer.sog> [--gpu 0]

The strip renderer reuses the proven render path
(`core.training.gsplat_trainer.load_colmap_data` + `render_splats`, same as
`core.enhancement.distill.render_pseudo_views`) and the object-centering /
orbit-basis math from `core.enhancement.coverage`. Poses are a TURNTABLE: spread
azimuths at the median captured elevation around the object/scene center — the
design's "6-frame mini-turntable" (chosen over pairwise slerp because the spec
asks for spread azimuths *around the center*, not a trajectory tween).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = str(Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ---------------------------------------------------------------------------
# Turntable poses (reuse coverage centering + orbit basis)
# ---------------------------------------------------------------------------

def build_turntable_poses(data: dict, frames: int = 6,
                          elev_override: float | None = None) -> tuple[np.ndarray, dict]:
    """Build `frames` c2w matrices (OpenCV) orbiting the object/scene center in the
    trainer's NORMALIZED world space (same space `load_colmap_data(normalize=True)`
    returns). Reuses `object_center_from_cameras` + `estimate_up` + the orbit basis.
    """
    from core.enhancement.coverage import (object_center_from_cameras,
                                           estimate_up, _basis)

    cams = np.asarray(data["camtoworlds"], dtype=np.float64)
    C = cams[:, :3, 3]                      # (N,3) camera centers
    fwd = cams[:, :3, 2]                    # (N,3) OpenCV +z forward

    center = object_center_from_cameras(C, fwd)
    # Robustness: ray-convergence can fall behind an inside-out (environment)
    # capture. If the recovered center is implausibly far from the camera cloud,
    # fall back to the SfM point median (or the camera centroid).
    cam_centroid = C.mean(0)
    spread = float(np.median(np.linalg.norm(C - cam_centroid, axis=1))) + 1e-9
    if (not np.all(np.isfinite(center))) or \
       np.linalg.norm(center - cam_centroid) > 5.0 * spread:
        pts = data.get("points")
        center = (np.median(np.asarray(pts, np.float64), axis=0)
                  if pts is not None and len(pts) else cam_centroid)

    up = estimate_up(C, center)
    up, ref, perp = _basis(up)

    dirs = C - center
    radius = float(np.median(np.linalg.norm(dirs, axis=1)))
    if radius <= 1e-6:
        radius = spread
    dn = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)

    # Median captured elevation = a "sensible elevation" that frames the subject the
    # way it was actually shot.
    cam_el = np.degrees(np.arcsin(np.clip(dn @ up, -1.0, 1.0)))
    elev = float(np.median(cam_el)) if elev_override is None else float(elev_override)
    elev = float(np.clip(elev, -10.0, 75.0))

    # Azimuth origin = circular mean of the captured azimuths, so frame 0 sits over
    # the densest part of the real ring (well-framed) and the strip sweeps outward.
    cam_az = np.arctan2(dn @ perp, dn @ ref)
    az0 = math.atan2(float(np.sin(cam_az).mean()), float(np.cos(cam_az).mean()))

    el = math.radians(elev)
    poses = []
    for k in range(frames):
        azr = az0 + 2.0 * math.pi * k / frames
        d = (math.cos(el) * (math.cos(azr) * ref + math.sin(azr) * perp)
             + math.sin(el) * up)
        eye = center + radius * d
        f = center - eye
        f /= (np.linalg.norm(f) + 1e-9)
        r = np.cross(f, up)
        r /= (np.linalg.norm(r) + 1e-9)
        u2 = np.cross(r, f)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = r
        c2w[:3, 1] = -u2          # OpenCV: +y down (matches coverage.target_poses)
        c2w[:3, 2] = f
        c2w[:3, 3] = eye
        poses.append(c2w)
    meta = {"center": center.tolist(), "radius": radius, "elev_deg": elev,
            "frames": frames,
            "label": f"{frames}-view turntable  |  elev {elev:.0f}deg"}
    return np.stack(poses), meta


def select_real_poses(data: dict, frames: int = 6) -> tuple[np.ndarray, dict]:
    """Pick `frames` REAL training-camera poses spread across the capture (index order
    ~ capture order). Used for ENVIRONMENT mode: a SYNTHESIZED orbit around a divergent /
    inside-out scene can stare into unreconstructed void, whereas real capture poses
    always look at reconstructed geometry."""
    cams = np.asarray(data["camtoworlds"], dtype=np.float32)
    ti = np.sort(np.asarray(data.get("train_indices",
                                     np.arange(len(cams))), dtype=np.int64))
    if len(ti) == 0:
        ti = np.arange(len(cams), dtype=np.int64)
    k = int(min(frames, len(ti)))
    sel = ti[np.linspace(0, len(ti) - 1, k).astype(int)]
    poses = np.stack([cams[int(i)] for i in sel])
    return poses, {"frames": k, "label": f"{k} capture views"}


# ---------------------------------------------------------------------------
# Strip renderer
# ---------------------------------------------------------------------------

def _load_ckpt_splats(ckpt_path: str, device, retries: int = 3):
    """Load a checkpoint's splats, tolerating a torch.save that is still in flight
    (the trainer writes ckpt files non-atomically; a preview can fire moments after)."""
    import torch
    last = None
    for i in range(retries):
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=True)
            return {k: v.to(device) for k, v in ck["splats"].items()}
        except Exception as e:       # partial write / EOF -> brief backoff + retry
            last = e
            time.sleep(0.5)
    raise RuntimeError(f"could not load checkpoint {ckpt_path}: {last}")


def render_strip(ckpt_path: str, data_dir: str, out_png: str, step: int,
                 factor: int, frames: int = 6, height: int = 360,
                 cols: int = 3, gpu: int = 0, mode: str = "object",
                 test_every: int = 8) -> str:
    """Render a `frames`-frame strip and composite it into one labeled PNG.

    OBJECT mode -> synthesized turntable around the object center (spread azimuths).
    ENVIRONMENT mode -> 6 REAL training-camera poses spread across the capture (a
    synthesized orbit can point into unreconstructed void for divergent scenes)."""
    import torch
    import cv2
    from core.training.gsplat_trainer import load_colmap_data, render_splats

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    splats = _load_ckpt_splats(ckpt_path, device)
    sh_degree = int((-1 + (1 + 4 * (1 + splats["shN"].shape[1])) ** 0.5) / 2)

    data = load_colmap_data(data_dir, factor=factor, test_every=test_every,
                            normalize=True)
    if str(mode).lower().startswith("env"):
        poses, meta = select_real_poses(data, frames=frames)
    else:
        poses, meta = build_turntable_poses(data, frames=frames)

    # Intrinsics/size from the first training camera, scaled to the target height (one
    # reference K for all frames — a capture uses a single camera model).
    ti = data.get("train_indices", [])
    ref_idx = int(ti[0]) if len(ti) else 0
    cid = data["camera_ids"][ref_idx]
    K = np.asarray(data["Ks_dict"][cid], dtype=np.float64).copy()
    W, H = data["imsize_dict"][cid]
    scale = float(height) / float(H)
    rH = int(height)
    rW = max(8, int(round(W * scale)))
    K[:2, :] *= scale
    K_t = torch.from_numpy(K).float().to(device)
    bkgd = torch.zeros(3, device=device)

    tiles = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_gpu = time.time()
    with torch.no_grad():
        for c2w in poses:
            c2w_t = torch.from_numpy(np.asarray(c2w)).float().to(device)
            w2c = torch.linalg.inv(c2w_t)
            renders, _, _ = render_splats(
                splats, w2c, K_t, rW, rH, sh_degree, bkgd, "antialiased")
            img = (renders[0].clamp(0, 1) * 255).byte().cpu().numpy()   # (H,W,3) RGB
            tiles.append(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # The GPU-contention window is JUST this loop; the rest of the subprocess wall time
    # is CPU-side (torch import + COLMAP parse), which does not fight the trainer's GPU.
    print(f"[preview] gpu_render_burst={time.time() - t_gpu:.2f}s "
          f"({len(poses)} frames @ {rW}x{rH}, {splats['means'].shape[0]} gaussians)",
          flush=True)

    strip = _compose_strip(tiles, step, meta, cols=cols)

    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write so the UI timer never reads a half-written PNG. Encode to bytes
    # (cv2.imwrite picks the codec by file extension, so a ".tmp" name would fail) and
    # os.replace the finished file into place.
    ok, buf = cv2.imencode(".png", cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encode failed")
    tmp = out.with_name(out.name + ".part")
    tmp.write_bytes(buf.tobytes())
    os.replace(str(tmp), str(out))

    del splats, K_t, bkgd
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(out)


def _compose_strip(tiles: list[np.ndarray], step: int, meta: dict,
                   cols: int = 3) -> np.ndarray:
    """Lay frames in a grid (turntable reads left->right, top->bottom as the object
    rotates) with thin separators and a top label bar. A 3-col x 2-row grid of ~360px
    frames gives the design's ~720px-tall strip that stays legible in the side panel
    (a single 6-wide row would display as an unreadable sliver). RGB uint8 out."""
    import cv2
    if not tiles:
        raise RuntimeError("no rendered frames")
    cols = max(1, min(cols, len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    fh, fw = tiles[0].shape[:2]
    sep = 4
    bar = 34
    grid_w = cols * fw + sep * (cols - 1)
    grid_h = rows * fh + sep * (rows - 1)
    canvas = np.full((bar + grid_h, grid_w, 3), 22, np.uint8)   # dark bg
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y = bar + r * (fh + sep)
        x = c * (fw + sep)
        canvas[y:y + t.shape[0], x:x + t.shape[1]] = t
    label = f"Step {step}  |  {meta.get('label', str(len(tiles)) + ' views')}"
    cv2.putText(canvas, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# On-demand viewer export: checkpoint -> SOG (reuse the delivery/splat-transform path)
# ---------------------------------------------------------------------------

def _ply_sh_degree(ply_path: str) -> int:
    """SH degree from a PLY header (count f_rest_* props). Reads only the header."""
    n_rest = 0
    try:
        with open(ply_path, "rb") as f:
            for _ in range(400):
                line = f.readline().decode("ascii", "replace").strip()
                if line.startswith("property") and "f_rest_" in line:
                    n_rest += 1
                if line == "end_header" or not line:
                    break
    except Exception:
        return 3
    rest = n_rest // 3
    total = rest + 1
    deg = int(round(total ** 0.5)) - 1
    return max(0, deg)


def export_viewer_sog(ckpt_path: str, ply_dir: str, out_sog: str,
                      gpu: str = "0", iterations: int = 8) -> str:
    """Produce a viewer-ready SOG for `ckpt_path`. Reuses the PLY the trainer already
    wrote at that save step when present (`point_cloud_<step>.ply`); otherwise loads
    the checkpoint and writes a PLY. Then converts PLY->SOG via splat-transform
    (SH2 quick profile: cap harmonics at 2 for a smaller file; morton-ordered). Uses a
    reduced SOG-compression iteration count — this is a fast on-demand PREVIEW, not the
    final delivery SOG (which Step ④/core.delivery produces at full quality)."""
    from core.delivery import run_splat_transform, export_ply, load_splats

    ckpt = Path(ckpt_path)
    step = None
    try:
        step = int(ckpt.name.split("_")[1])
    except (IndexError, ValueError):
        pass

    src_ply = None
    if step is not None and ply_dir:
        cand = Path(ply_dir) / f"point_cloud_{step}.ply"
        if cand.is_file():
            src_ply = str(cand)

    tmp_ply = None
    if src_ply is None:
        # No matching PLY on disk -> materialize one from the checkpoint (CPU).
        tmp_ply = str(Path(out_sog).with_suffix(".src.ply"))
        splats = load_splats(ckpt_path, "cpu")
        export_ply(splats, tmp_ply)
        src_ply = tmp_ply

    actions = []
    if _ply_sh_degree(src_ply) > 2:          # SH2 quick profile
        actions += ["--filter-harmonics", "2"]
    actions += ["--morton-order"]

    Path(out_sog).parent.mkdir(parents=True, exist_ok=True)
    run_splat_transform(src_ply, out_sog, actions, gpu=gpu, iterations=iterations)

    if tmp_ply and os.path.isfile(tmp_ply):
        try:
            os.remove(tmp_ply)
        except OSError:
            pass
    return out_sog


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="gui2 live-preview subprocess")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("strip", help="render a turntable strip PNG")
    ps.add_argument("--ckpt", required=True)
    ps.add_argument("--data_dir", required=True)
    ps.add_argument("--out", required=True)
    ps.add_argument("--step", type=int, default=0)
    ps.add_argument("--factor", type=int, default=1)
    ps.add_argument("--frames", type=int, default=6)
    ps.add_argument("--height", type=int, default=360)
    ps.add_argument("--cols", type=int, default=3)
    ps.add_argument("--gpu", type=int, default=0)
    ps.add_argument("--mode", default="object",
                    help="object=synthesized turntable; environment=real capture poses")
    ps.add_argument("--test_every", type=int, default=8)

    pe = sub.add_parser("export", help="export a viewer-ready SOG from a checkpoint")
    pe.add_argument("--ckpt", required=True)
    pe.add_argument("--ply_dir", default="")
    pe.add_argument("--out", required=True)
    pe.add_argument("--gpu", default="0")
    pe.add_argument("--iterations", type=int, default=8)

    args = ap.parse_args(argv)
    if args.cmd == "strip":
        t0 = time.time()
        out = render_strip(args.ckpt, args.data_dir, args.out, args.step,
                           args.factor, frames=args.frames, height=args.height,
                           cols=args.cols, gpu=args.gpu, mode=args.mode,
                           test_every=args.test_every)
        print(f"[preview] strip -> {out} ({time.time() - t0:.1f}s)", flush=True)
    elif args.cmd == "export":
        t0 = time.time()
        out = export_viewer_sog(args.ckpt, args.ply_dir, args.out, gpu=args.gpu,
                                iterations=args.iterations)
        print(f"[preview] sog -> {out} ({time.time() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
