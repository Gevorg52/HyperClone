"""Shared plumbing for the WP-Q1 phase-2 scorers.

Everything the scorers need on top of the phase-1 SceneContext: render arbitrary
poses from the (possibly corrupted) splats, apply the blur corruption to scorer
renders so blur-type gates stay direction-correct, stage held-out renders / real
photos to disk, find the nearest real photo by COLMAP pose distance, and build
the two rendered clips the temporal scorers consume:

  * a SMOOTH MOVING trajectory (RAFT warp-error, TSED consecutive pairs, NVS-SQA
    sorted-frame sequence), interpolated through a contiguous run of real cameras
  * a NEAR-STATIC micro-jitter clip (VBench flicker) — tiny sub-scene-scale camera
    motion so view-dependent floaters/aliasing pop while content stays ~static

All renders come from `ctx.splats` (which ALREADY carries injected floaters when
the run is corrupted) so consistency/NR scorers see the corruption.

STAGING POLICY (review-hardened): staged files under `<out_dir>/_scorers/` are
ALWAYS regenerated — never reused across runs. (The original out_dir-keyed disk
cache served PRISTINE renders to a corrupted re-run of the same out_dir —
reviewer-reproduced HIGH bug.) Within ONE run, results are memoized in-memory on
the ctx so multiple scorers share the work; clip/render staging dirs are cleared
before writing so stale extra frames from an earlier run can never leak into
directory globs (NVS-SQA, CrossScore).

With `--floater_jitter`, the clip renderers re-randomize the injected floaters'
positions on every frame (deterministic per frame index) — genuine temporal
popping, the positive control for raft_warp/flicker.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# pose / intrinsics helpers  (data is load_colmap_data output; poses = c2w,
# OpenCV convention; w2c = inv(c2w))
# ---------------------------------------------------------------------------

def c2w_of(ctx, idx: int) -> np.ndarray:
    return np.asarray(ctx.data["camtoworlds"][idx], dtype=np.float64)


def w2c_of(ctx, idx: int) -> np.ndarray:
    return np.linalg.inv(c2w_of(ctx, idx))


def K_of(ctx, idx: int) -> np.ndarray:
    cid = ctx.data["camera_ids"][idx]
    return np.asarray(ctx.data["Ks_dict"][cid], dtype=np.float64)


def imsize_of(ctx, idx: int):
    """(W, H) at the loaded factor, matching load_image's undistort/crop."""
    cid = ctx.data["camera_ids"][idx]
    return ctx.data["imsize_dict"][cid]


def cam_center(c2w: np.ndarray) -> np.ndarray:
    return np.asarray(c2w, dtype=np.float64)[:3, 3]


def robust_scene_scale(ctx) -> float:
    """Percentile-robust scene scale: 1–99 percentile bbox diagonal of the camera
    centers (the same robust-bbox pattern phase 1 uses for floater placement).
    The data dict's `scene_scale` is a raw MAX camera radius — a single outlier
    camera inflates it and quietly loosens the flicker static-guard."""
    centers = np.stack([np.asarray(c, np.float64)[:3, 3]
                        for c in ctx.data["camtoworlds"]], axis=0)
    lo = np.percentile(centers, 1.0, axis=0)
    hi = np.percentile(centers, 99.0, axis=0)
    d = float(np.linalg.norm(hi - lo))
    if d <= 1e-9:            # degenerate (e.g. single/coincident cameras)
        d = max(float(ctx.data.get("scene_scale", 1.0)), 1e-9)
    return d


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _blur_sigma(ctx) -> float:
    corr = getattr(ctx, "corruption", None) or {}
    return float(corr.get("blur_sigma", 0.0) or 0.0)


def _splats_for_frame(ctx, frame_idx):
    """Splats to render with. Normally ctx.splats (already carrying the STATIC
    floater injection when the run is corrupted). Under --floater_jitter, clip
    renderers pass a frame_idx and the N injected floaters (appended LAST by
    inject_floaters) are re-drawn at per-frame-seeded random positions ->
    genuine temporal popping (positive control for raft_warp/flicker)."""
    corr = getattr(ctx, "corruption", None) or {}
    n = int(corr.get("inject_floaters", 0) or 0)
    if frame_idx is None or n <= 0 or not corr.get("floater_jitter"):
        return ctx.splats
    from ..corruptions import inject_floaters
    base = {k: v[:-n] for k, v in ctx.splats.items()}
    seed = int(corr.get("floater_seed", 0)) + 1000 + int(frame_idx)
    return inject_floaters(base, n, seed=seed,
                           scale_frac=float(corr.get("floater_scale", 0.01)))


def render_pose(ctx, c2w: np.ndarray, K: np.ndarray, W: int, H: int,
                frame_idx=None) -> torch.Tensor:
    """Render (H,W,3) float [0,1] from ctx.splats for a c2w pose + K. Applies the
    blur corruption if the run set one (keeps blur gates direction-correct);
    `frame_idx` (clips only) enables per-frame floater jitter."""
    from core.training.gsplat_trainer import render_splats
    from ..corruptions import gaussian_blur
    splats = _splats_for_frame(ctx, frame_idx)
    dev = ctx.device
    viewmat = torch.linalg.inv(torch.from_numpy(np.asarray(c2w, np.float32)).to(dev))
    Kt = torch.from_numpy(np.asarray(K, np.float32)).to(dev)
    bg = torch.zeros(3, device=dev)
    with torch.no_grad():
        renders, _, _ = render_splats(splats, viewmat, Kt, int(W), int(H),
                                      ctx.sh_degree, bg, "antialiased")
        img = renders[0].clamp(0, 1)
    sig = _blur_sigma(ctx)
    if sig > 0:
        img = gaussian_blur(img, sig)
    return img


def render_view(ctx, idx: int) -> torch.Tensor:
    """Render held-out (or any) view idx at its native factor resolution."""
    W, H = imsize_of(ctx, idx)
    return render_pose(ctx, c2w_of(ctx, idx), K_of(ctx, idx), W, H)


def to_uint8_rgb(img: torch.Tensor) -> np.ndarray:
    return (img.clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def _stage(ctx) -> Path:
    d = Path(ctx.out_dir) / "_scorers"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fresh_dir(d: Path) -> Path:
    """(Re)create a staging dir EMPTY. Staged content is ALWAYS regenerated —
    never trusted across runs (out_dir-keyed disk caching once served pristine
    renders to a corrupted re-run: the reviewer-reproduced HIGH bug). Clearing
    also removes stale extra frames a directory glob would pick up."""
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _memo(ctx) -> dict:
    """Per-RUN in-memory memo (attached to the ctx object): scorers in the same
    run share staged work legitimately — same process, same splats, same
    corruption. Never persisted, so it cannot go stale across runs."""
    m = getattr(ctx, "_scorer_memo", None)
    if m is None:
        m = {}
        ctx._scorer_memo = m
    return m


# ---------------------------------------------------------------------------
# staging held-out renders + real photos (CrossScore / NOVA)
# ---------------------------------------------------------------------------

def stage_heldout_renders(ctx, subdir="renders") -> List[str]:
    """Full-res render of every held-out view -> <stage>/<subdir>/frame_XXXXX.png.
    Always rendered fresh for this run (dir cleared first); memoized in-memory so
    several scorers in one run share the work. Returns sorted list of paths."""
    memo = _memo(ctx)
    key = ("renders", subdir)
    if key in memo:
        return memo[key]
    out = _fresh_dir(_stage(ctx) / subdir)
    paths = []
    for idx in ctx.test_indices:
        p = out / f"frame_{idx:05d}.png"
        rgb = to_uint8_rgb(render_view(ctx, idx))
        cv2.imwrite(str(p), rgb[..., ::-1])   # RGB->BGR
        paths.append(str(p))
    memo[key] = sorted(paths)
    return memo[key]


def stage_real_photo(ctx, idx: int, subdir="real") -> str:
    """Undistorted real photo for view idx (via load_image, so it matches the
    pose/K exactly) -> <stage>/<subdir>/frame_XXXXX.png. Always (re)written —
    overwriting is cheap and removes the stale-file bug class; memoized
    in-memory within the run (called per-index, so the dir is NOT cleared)."""
    memo = _memo(ctx)
    key = ("real", subdir, int(idx))
    if key in memo:
        return memo[key]
    from core.training.gsplat_trainer import load_image
    out = _stage(ctx) / subdir
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"frame_{idx:05d}.png"
    img = load_image(ctx.data, idx)["image"]        # (H,W,3) [0,1]
    cv2.imwrite(str(p), to_uint8_rgb(img)[..., ::-1])
    memo[key] = str(p)
    return memo[key]


def nearest_real_index(ctx, idx: int) -> int:
    """Index of the nearest OTHER real camera by camera-center distance.

    NOVA wants a NON-ALIGNED real reference near the rendered viewpoint. We
    therefore take the nearest real photo EXCLUDING the view's own index (pairing
    a render with its own GT would be pixel-aligned, defeating NOVA's design).
    This is the tightest valid non-aligned reference; see scorers/nova.py."""
    centers = np.stack([cam_center(np.asarray(c2w, np.float64))
                        for c2w in ctx.data["camtoworlds"]], axis=0)
    d = np.linalg.norm(centers - centers[idx], axis=1)
    d[idx] = np.inf
    return int(np.argmin(d))


# ---------------------------------------------------------------------------
# camera-path generation
# ---------------------------------------------------------------------------

def _quat_from_R(R):
    R = np.asarray(R, np.float64)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _R_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], np.float64)


def _slerp(q0, q1, u):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / (np.linalg.norm(q) + 1e-12)
    th0 = math.acos(max(-1.0, min(1.0, d)))
    th = th0 * u
    q2 = q1 - q0 * d
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    return q0 * math.cos(th) + q2 * math.sin(th)


def _interp_c2w(c2w_a, c2w_b, u):
    qa = _quat_from_R(c2w_a[:3, :3])
    qb = _quat_from_R(c2w_b[:3, :3])
    R = _R_from_quat(_slerp(qa, qb, u))
    t = (1 - u) * c2w_a[:3, 3] + u * c2w_b[:3, 3]
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _resample_K(K, W, H, long_edge):
    """Scale K + (W,H) so the long edge <= long_edge (keeps FOV)."""
    s = min(1.0, long_edge / max(W, H))
    if s >= 1.0:
        return np.asarray(K, np.float64), int(W), int(H)
    Ks = np.asarray(K, np.float64).copy()
    Ks[:2, :] *= s
    return Ks, max(1, round(W * s)), max(1, round(H * s))


def moving_clip(ctx, n_anchors=12, steps=3, long_edge=640):
    """Smooth moving trajectory through a contiguous run of real cameras.

    Anchors = a centered contiguous block of up to `n_anchors` name-sorted views
    (real capture order is already smooth); `steps` interpolated frames between
    each consecutive anchor pair (SLERP rotation + lerp translation). Intrinsics
    fixed to the first anchor's K, downscaled so long edge <= `long_edge`.

    Returns dict: frames_u8 (list uint8 RGB), poses (list of {R,t,K,H,W} w2c
    OpenCV), paths (staged PNGs, sorted). Rendered fresh each run (dir cleared);
    memoized in-memory within the run."""
    memo = _memo(ctx)
    key = ("moving", n_anchors, steps, long_edge)
    if key in memo:
        return memo[key]

    N = len(ctx.data["camtoworlds"])
    n_anchors = max(2, min(n_anchors, N))
    lo = max(0, (N - n_anchors) // 2)
    anchor_ids = list(range(lo, lo + n_anchors))

    K0, W, H = _resample_K(K_of(ctx, anchor_ids[0]), *imsize_of(ctx, anchor_ids[0]),
                           long_edge=long_edge)

    c2ws = [c2w_of(ctx, i) for i in anchor_ids]
    path_c2w = []
    for a in range(len(c2ws) - 1):
        for s in range(steps):
            path_c2w.append(_interp_c2w(c2ws[a], c2ws[a + 1], s / steps))
    path_c2w.append(c2ws[-1])

    out = _fresh_dir(_stage(ctx) / "moving")
    frames_u8, poses, paths = [], [], []
    for fi, c2w in enumerate(path_c2w):
        p = out / f"frame_{fi:05d}.png"
        img = render_pose(ctx, c2w, K0, W, H, frame_idx=fi)
        u8 = to_uint8_rgb(img)
        cv2.imwrite(str(p), u8[..., ::-1])
        frames_u8.append(u8)
        w2c = np.linalg.inv(c2w)
        poses.append({"R": w2c[:3, :3].copy(), "t": w2c[:3, 3].copy(),
                      "K": K0.copy(), "H": int(H), "W": int(W)})
        paths.append(str(p))
    memo[key] = {"frames_u8": frames_u8, "poses": poses, "paths": sorted(paths),
                 "H": int(H), "W": int(W)}
    return memo[key]


def static_clip(ctx, n_frames=16, jitter_frac=0.06, long_edge=640):
    """Near-static MICRO-jitter clip around one central view (VBench flicker).

    Camera center jitters on a tiny circle of radius `jitter_frac * median
    inter-camera baseline` (deterministic, no RNG) — small enough to read as a
    static camera (see max_motion_frac in the returned dict) yet enough parallax
    that view-dependent floaters/aliasing flicker. Orientation is held fixed.

    Returns dict: frames_u8, max_motion_frac (max inter-frame center displacement
    / robust scene scale — the flicker guard checks this), radius, paths.
    Rendered fresh each run (dir cleared); memoized in-memory within the run."""
    memo = _memo(ctx)
    key = ("static", n_frames, jitter_frac, long_edge)
    if key in memo:
        return memo[key]

    N = len(ctx.data["camtoworlds"])
    aidx = N // 2
    c2w0 = c2w_of(ctx, aidx)
    K0, W, H = _resample_K(K_of(ctx, aidx), *imsize_of(ctx, aidx), long_edge=long_edge)

    scale = robust_scene_scale(ctx)
    centers = np.stack([cam_center(np.asarray(c, np.float64))
                        for c in ctx.data["camtoworlds"]], axis=0)
    order = np.argsort(np.linalg.norm(centers - centers[aidx], axis=1))
    baseline = float(np.linalg.norm(centers[order[1]] - centers[aidx])) if N > 1 else scale
    radius = jitter_frac * max(baseline, 1e-6)

    # local camera right/up axes (columns of c2w rotation)
    right = c2w0[:3, 0]
    up = c2w0[:3, 1]

    out = _fresh_dir(_stage(ctx) / "static")
    frames_u8, centers_path, paths = [], [], []
    for fi in range(n_frames):
        ang = 2 * math.pi * fi / n_frames
        off = radius * (math.cos(ang) * right + math.sin(ang) * up)
        c2w = c2w0.copy()
        c2w[:3, 3] = c2w0[:3, 3] + off
        img = render_pose(ctx, c2w, K0, W, H, frame_idx=fi)
        u8 = to_uint8_rgb(img)
        p = out / f"frame_{fi:05d}.png"
        cv2.imwrite(str(p), u8[..., ::-1])
        frames_u8.append(u8)
        centers_path.append(c2w[:3, 3].copy())
        paths.append(str(p))

    cp = np.stack(centers_path, axis=0)
    step_motion = np.linalg.norm(np.diff(cp, axis=0), axis=1) if len(cp) > 1 else np.zeros(1)
    max_motion_frac = float(step_motion.max() / scale)
    memo[key] = {"frames_u8": frames_u8, "max_motion_frac": max_motion_frac,
                 "radius": radius, "paths": sorted(paths)}
    return memo[key]
