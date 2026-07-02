"""Minimal gsplat training script for Apex Reality.

Self-contained trainer that uses:
  - pycolmap.Reconstruction (already installed) for COLMAP data loading
  - gsplat.rendering.rasterization for Gaussian rendering
  - gsplat.strategy.MCMCStrategy (default) or DefaultStrategy for densification
  - Standard PyTorch for everything else (no nerfview, fused_ssim, etc.)

Quality stack (June 2026):
  - MCMC densification with explicit Gaussian budget (cap_max)
  - Mip-Splatting anti-aliasing (rasterize_mode="antialiased")
  - Optional SE(3) camera pose refinement during training
  - Coarse-to-fine resolution schedule (densify at low res, finish at high res)
  - Opacity/scale regularization (required for MCMC)

Designed to be invoked as a subprocess from GsplatBackend.train().
Progress is printed to stdout in a parseable format.

Usage:
    python -m core.training.gsplat_trainer \
        --data_dir <path> --result_dir <path> \
        --data_factor 2 --coarse_factor 4 --coarse_until 15000 \
        --max_steps 30000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

# Must precede any gsplat import (fixes MSVC JIT flags on Windows).
# Works both as a module (-m core.training.gsplat_trainer) and as a script.
try:
    from core.training import gsplat_jit_shim  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gsplat_jit_shim  # noqa: F401

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy


# ---------------------------------------------------------------------------
# Scene normalization (adapted from gsplat examples)
# ---------------------------------------------------------------------------

def _similarity_from_cameras(c2w: np.ndarray) -> np.ndarray:
    """Compute a similarity transform to normalize camera positions."""
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    norm = np.linalg.norm(world_up)
    if norm < 1e-6:
        world_up = np.array([0.0, -1.0, 0.0])
    else:
        world_up /= norm

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) / (1 + c)
    else:
        R_align = np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])

    fwds = np.sum((R_align @ R.transpose(0, 2, 1)).transpose(0, 2, 1) * np.array([0, 0, 1.0]), axis=-1)
    t_aligned = (R_align @ t[..., None])[..., 0]

    nearest = t_aligned + (fwds * -t_aligned).sum(-1)[:, None] * fwds
    translate = -np.median(nearest, axis=0)

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    scale = 1.0 / np.median(np.linalg.norm(t_aligned + translate, axis=-1))
    transform[:3, :] *= scale
    return transform


def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _transform_cameras(matrix: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    c2w = np.einsum("nij, ki -> nkj", c2w, matrix)
    scaling = np.linalg.norm(c2w[:, 0, :3], axis=1)
    c2w[:, :3, :3] /= scaling[:, None, None]
    return c2w


# ---------------------------------------------------------------------------
# COLMAP data loading via pycolmap.Reconstruction
# ---------------------------------------------------------------------------

def load_colmap_data(
    data_dir: str,
    factor: int = 4,
    test_every: int = 8,
    normalize: bool = True,
):
    """Load COLMAP reconstruction and images.

    Returns dict with keys: camtoworlds, Ks, image_paths, points, points_rgb,
    train_indices, test_indices, imsize, distortion_params.
    """
    import pycolmap

    sparse_dir = os.path.join(data_dir, "sparse", "0")
    if not os.path.exists(sparse_dir):
        sparse_dir = os.path.join(data_dir, "sparse")

    rec = pycolmap.Reconstruction(sparse_dir)

    # Collect camera data
    w2c_mats = []
    camera_ids = []
    image_names = []
    Ks_dict = {}
    params_dict = {}
    imsize_dict = {}

    bottom = np.array([0, 0, 0, 1], dtype=np.float64).reshape(1, 4)

    for img_id in sorted(rec.images.keys()):
        img = rec.images[img_id]
        # pycolmap >= 3.12: cam_from_world is a method returning Rigid3d
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        R = cfw.rotation.matrix()
        t = cfw.translation.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([R, t], axis=1), bottom], axis=0)
        w2c_mats.append(w2c)
        camera_ids.append(img.camera_id)
        image_names.append(img.name)

        cam = rec.cameras[img.camera_id]
        params = cam.params
        model_name = cam.model.name

        # Build intrinsic matrix
        if model_name in ("SIMPLE_PINHOLE",):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
            dist = np.empty(0, dtype=np.float32)
        elif model_name in ("PINHOLE",):
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]
            dist = np.empty(0, dtype=np.float32)
        elif model_name in ("SIMPLE_RADIAL",):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
            dist = np.array([params[3], 0, 0, 0], dtype=np.float32)
        elif model_name in ("RADIAL",):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
            dist = np.array([params[3], params[4], 0, 0], dtype=np.float32)
        elif model_name in ("OPENCV",):
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]
            dist = np.array([params[4], params[5], params[6], params[7]], dtype=np.float32)
        else:
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]
            dist = np.empty(0, dtype=np.float32)

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        K[:2, :] /= factor

        Ks_dict[img.camera_id] = K
        params_dict[img.camera_id] = dist
        imsize_dict[img.camera_id] = (cam.width // factor, cam.height // factor)

    w2c_mats = np.stack(w2c_mats, axis=0)
    camtoworlds = np.linalg.inv(w2c_mats).astype(np.float32)

    # Sort by image name (consistent with gsplat convention)
    inds = np.argsort(image_names)
    image_names = [image_names[i] for i in inds]
    camtoworlds = camtoworlds[inds]
    camera_ids = [camera_ids[i] for i in inds]

    # Resolve image paths
    if factor > 1:
        image_dir = os.path.join(data_dir, f"images_{factor}")
    else:
        image_dir = os.path.join(data_dir, "images")

    colmap_image_dir = os.path.join(data_dir, "images")

    # Map colmap image names to downsampled image names
    if os.path.isdir(image_dir) and image_dir != colmap_image_dir:
        colmap_files = sorted(os.listdir(colmap_image_dir))
        image_files = sorted(os.listdir(image_dir))
        name_map = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, name_map.get(n, n)) for n in image_names]
    else:
        image_paths = [os.path.join(colmap_image_dir, n) for n in image_names]

    # Resolve mask paths (for object-mask alpha supervision). Prefer masks_{factor}/
    # (matches the downsampled images), fall back to masks/. Masks are PNG, stem-matched
    # to the COLMAP image name. None if no mask dir exists.
    mask_paths = None
    for md in ([os.path.join(data_dir, f"masks_{factor}")] if factor > 1 else []) + \
              [os.path.join(data_dir, "masks")]:
        if os.path.isdir(md):
            cand = [os.path.join(md, os.path.splitext(n)[0] + ".png") for n in image_names]
            if any(os.path.isfile(p) for p in cand):
                mask_paths = cand
                break

    # Verify first image exists and adjust intrinsics if needed
    first_img = imageio.imread(image_paths[0])[..., :3]
    actual_h, actual_w = first_img.shape[:2]
    colmap_w, colmap_h = imsize_dict[camera_ids[0]]
    s_w, s_h = actual_w / colmap_w, actual_h / colmap_h
    for cid, K in Ks_dict.items():
        K[0, :] *= s_w
        K[1, :] *= s_h
        w, h = imsize_dict[cid]
        imsize_dict[cid] = (int(w * s_w), int(h * s_h))

    # Undistortion maps
    mapx_dict = {}
    mapy_dict = {}
    roi_dict = {}
    for cid, dist in params_dict.items():
        if len(dist) == 0:
            continue
        K = Ks_dict[cid]
        w, h = imsize_dict[cid]
        K_undist, roi = cv2.getOptimalNewCameraMatrix(K.astype(np.float64), dist, (w, h), 0)
        mapx, mapy = cv2.initUndistortRectifyMap(
            K.astype(np.float64), dist, None, K_undist, (w, h), cv2.CV_32FC1,
        )
        mapx_dict[cid] = mapx
        mapy_dict[cid] = mapy
        roi_dict[cid] = roi
        Ks_dict[cid] = K_undist
        if roi[2] > 0 and roi[3] > 0:
            imsize_dict[cid] = (roi[2], roi[3])

    # 3D points
    pts = rec.points3D
    pid_sorted = sorted(pts.keys())
    points = np.array([pts[pid].xyz for pid in pid_sorted], dtype=np.float32)
    points_rgb = np.array([pts[pid].color for pid in pid_sorted], dtype=np.uint8)

    # Per-view SfM observations (for sparse depth supervision) — for each image, the row
    # indices (into `points`) of the 3D points it observes. Built by one pass over the
    # point tracks; keyed by image name so it stays aligned to the name-sorted order.
    point_obs = None
    try:
        from collections import defaultdict
        imgid_to_name = {iid: im.name for iid, im in rec.images.items()}
        by_name = defaultdict(list)
        for row, pid in enumerate(pid_sorted):
            for el in pts[pid].track.elements:
                nm = imgid_to_name.get(el.image_id)
                if nm is not None:
                    by_name[nm].append(row)
        point_obs = [np.asarray(by_name.get(nm, []), dtype=np.int64) for nm in image_names]
    except Exception as e:
        print(f"[Data] sparse-depth observations unavailable ({e}); depth_loss disabled")

    # Normalize
    if normalize:
        T = _similarity_from_cameras(camtoworlds)
        camtoworlds = _transform_cameras(T, camtoworlds)
        points = _transform_points(T, points)

    # Train/test split
    indices = np.arange(len(image_names))
    # Held-out-wedge mode (proof experiments): APEX_HOLDOUT="start,frac" holds out a
    # CONTIGUOUS arc of the trajectory (images are name-sorted ~ capture order) as the
    # test set and trains on the rest — simulates an under-observed/unseen region.
    _holdout = os.environ.get("APEX_HOLDOUT", "")
    if _holdout:
        hs, hf = (float(x) for x in _holdout.split(","))
        n = len(indices)
        lo, hi = int(hs * n), int((hs + hf) * n)
        mask = np.zeros(n, dtype=bool)
        mask[lo:hi] = True
        test_indices = indices[mask]
        train_indices = indices[~mask]
    else:
        test_indices = indices[indices % test_every == 0]
        train_indices = indices[indices % test_every != 0]

    # Scene scale
    cam_locs = camtoworlds[:, :3, 3]
    scene_scale = float(np.max(np.linalg.norm(cam_locs - cam_locs.mean(0), axis=1)))

    print(f"[Data] {len(image_names)} images ({len(train_indices)} train, {len(test_indices)} test)")
    print(f"[Data] {len(points)} SfM points, scene_scale={scene_scale:.3f}")

    return {
        "camtoworlds": camtoworlds,
        "Ks_dict": Ks_dict,
        "camera_ids": camera_ids,
        "image_paths": image_paths,
        "mask_paths": mask_paths,
        "point_obs": point_obs,
        "image_names": image_names,
        "imsize_dict": imsize_dict,
        "params_dict": params_dict,
        "mapx_dict": mapx_dict,
        "mapy_dict": mapy_dict,
        "roi_dict": roi_dict,
        "points": points,
        "points_rgb": points_rgb,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "scene_scale": scene_scale,
        # per-view loss weight (1.0 for captured views; extra/pseudo views may set
        # <1 via append_extra_views -> confidence-gated prior fusion, Phase 1)
        "loss_weight": np.ones(len(image_names), dtype=np.float32),
    }


def append_extra_views(data: dict, extra_views_json: str) -> int:
    """Append externally-defined views (e.g. enhanced pseudo-views) to the
    training pool.

    The JSON holds a list of ``{"image": path, "K": 3x3, "c2w": 4x4}`` with
    poses already in the trainer's NORMALIZED world space (as produced by
    core.enhancement.distill).  Views are added to train_indices only.
    """
    entries = json.loads(Path(extra_views_json).read_text(encoding="utf-8"))
    if not entries:
        return 0

    base = len(data["image_names"])
    extra_c2ws = []
    for i, e in enumerate(entries):
        cid = f"extra_{i}"
        data["Ks_dict"][cid] = np.array(e["K"], dtype=np.float64)
        data["params_dict"][cid] = np.empty(0, dtype=np.float32)
        data["camera_ids"].append(cid)
        data["image_paths"].append(e["image"])
        data["image_names"].append(Path(e["image"]).name)
        extra_c2ws.append(np.array(e["c2w"], dtype=np.float32))

    data["camtoworlds"] = np.concatenate(
        [data["camtoworlds"], np.stack(extra_c2ws)], axis=0,
    )
    data["train_indices"] = np.concatenate(
        [data["train_indices"], np.arange(base, base + len(entries))],
    )
    # per-view loss weights (default 1.0; entries may carry "weight" for
    # confidence-gated prior fusion — high in under-observed regions, ~0 near
    # real cameras so the prior never overrides captured data)
    w = np.array([float(e.get("weight", 1.0)) for e in entries], dtype=np.float32)
    if "loss_weight" not in data:
        data["loss_weight"] = np.ones(base, dtype=np.float32)
    data["loss_weight"] = np.concatenate([data["loss_weight"], w])
    print(f"[Data] Appended {len(entries)} extra views (train pool now "
          f"{len(data['train_indices'])}); extra-view weight "
          f"mean={w.mean():.2f} range=[{w.min():.2f},{w.max():.2f}]")
    return len(entries)


def compute_unseen_mask(means: torch.Tensor, data: dict, device) -> torch.Tensor:
    """Bool (N,) mask: True where a Gaussian center projects into NO training/pseudo
    camera frustum (= floats in empty, never-supervised space). Used by the
    visibility loss to drive such Gaussians' opacity -> 0 (Nerfbusters-style),
    WITHOUT deleting pseudo-view-filled surfaces (those ARE seen by pseudo cams).
    Precomputed once (resume has no densification, so the set is fixed)."""
    N = means.shape[0]
    seen = torch.zeros(N, dtype=torch.bool, device=device)
    Xh = torch.cat([means, torch.ones(N, 1, device=device)], dim=1)  # (N,4)
    for idx in data["train_indices"]:
        idx = int(idx)
        cid = data["camera_ids"][idx]
        w2c = torch.linalg.inv(
            torch.from_numpy(data["camtoworlds"][idx]).float().to(device))
        K = torch.from_numpy(np.asarray(data["Ks_dict"][cid])).float().to(device)
        W, H = data["imsize_dict"].get(cid, (None, None))
        if W is None:
            continue
        xc = (w2c @ Xh.T).T[:, :3]
        z = xc[:, 2].clamp(min=1e-6)
        u = (K[0, 0] * xc[:, 0] + K[0, 2] * z) / z
        v = (K[1, 1] * xc[:, 1] + K[1, 2] * z) / z
        seen |= (xc[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if bool(seen.all()):
            break
    return ~seen


def load_image(data: dict, index: int) -> dict:
    """Load a single training image with undistortion."""
    image = imageio.imread(data["image_paths"][index])[..., :3]
    cid = data["camera_ids"][index]
    K = data["Ks_dict"][cid].copy()
    dist = data["params_dict"][cid]
    c2w = data["camtoworlds"][index]

    # Object mask (optional) — load + undistort/crop identically to the image so it
    # stays pixel-aligned. Resize to the (possibly downsampled) image first.
    mask = None
    mpaths = data.get("mask_paths")
    if mpaths is not None and mpaths[index] and os.path.isfile(mpaths[index]):
        m = imageio.imread(mpaths[index])
        if m.ndim == 3:
            m = m[..., 0]
        if m.shape[:2] != image.shape[:2]:
            m = cv2.resize(m, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (m.astype(np.float32) / 255.0)

    if len(dist) > 0 and cid in data["mapx_dict"]:
        image = cv2.remap(image, data["mapx_dict"][cid], data["mapy_dict"][cid], cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.remap(mask, data["mapx_dict"][cid], data["mapy_dict"][cid], cv2.INTER_LINEAR)
        if cid in data["roi_dict"]:
            x, y, w, h = data["roi_dict"][cid]
            if w > 0 and h > 0:
                image = image[y:y+h, x:x+w]
                if mask is not None:
                    mask = mask[y:y+h, x:x+w]

    lw = data.get("loss_weight")
    out = {
        "image": torch.from_numpy(image).float() / 255.0,
        "K": torch.from_numpy(K).float(),
        "c2w": torch.from_numpy(c2w).float(),
        "loss_weight": float(lw[index]) if lw is not None else 1.0,
    }
    if mask is not None:
        out["mask"] = torch.from_numpy(mask).float()  # (H, W) in [0,1]

    # Dense mono-depth prior (WP-F1): aligned MoGe-2 depth is stored already in
    # the undistorted+cropped training frame, so it drops in without any remap.
    # inf = invalid (masked in the loss). Only attached if the shape matches the
    # (post-undistort) rendered frame — a mismatch means wrong factor -> skip.
    if data.get("load_priors"):
        stem = os.path.splitext(os.path.basename(data["image_paths"][index]))[0]
        ppath = os.path.join(data["prior_dir"], "depth", stem + ".npy")
        if os.path.isfile(ppath):
            pd = np.load(ppath)
            if pd.shape[:2] == image.shape[:2]:
                out["prior_depth"] = torch.from_numpy(pd.astype(np.float32))
                rel, dw = data.get("prior_align", {}).get(stem, (False, 0.0))
                out["prior_reliable"] = bool(rel)
                out["prior_weight"] = float(dw)
    return out


class Prefetcher:
    """Background image loader that overlaps disk I/O + decode with GPU work.

    The training loop samples random views every step, so the order doesn't
    matter — workers sample random train indices, decode/undistort on CPU
    threads (cv2/imageio release the GIL during the heavy parts), and push
    ready batches to a bounded queue.  This keeps the GPU fed instead of
    blocking on per-step image loads (the cause of ~50% GPU utilization).
    """

    def __init__(self, data, train_indices, num_workers=3, queue_size=6,
                 seed=None):
        self.data = data
        self.train_indices = np.asarray(train_indices)
        self.q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        # seed=None -> unseeded per-worker RNGs (current production behavior).
        # With a seed, each worker gets its own deterministic stream
        # (seed + worker_id) so the set of sampled views is reproducible.
        # NOTE: queue interleaving across workers is still OS-scheduling
        # dependent, so step-level order is only approximately reproducible.
        self._seed = seed
        self._threads = [
            threading.Thread(target=self._worker, args=(wid,), daemon=True)
            for wid in range(max(1, num_workers))
        ]
        for t in self._threads:
            t.start()

    def _worker(self, worker_id: int = 0):
        rng = (np.random.default_rng() if self._seed is None
               else np.random.default_rng(self._seed + worker_id))
        n = len(self.train_indices)
        while not self._stop.is_set():
            idx = int(self.train_indices[rng.integers(n)])
            try:
                batch = load_image(self.data, idx)
            except Exception:
                continue
            batch["idx"] = idx
            try:
                batch["image"] = batch["image"].pin_memory()
                if "mask" in batch:
                    batch["mask"] = batch["mask"].pin_memory()
                if "prior_depth" in batch:
                    batch["prior_depth"] = batch["prior_depth"].pin_memory()
            except RuntimeError:
                pass
            while not self._stop.is_set():
                try:
                    self.q.put(batch, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def get(self):
        return self.q.get()

    def close(self):
        self._stop.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass


# ---------------------------------------------------------------------------
# Gaussian initialization
# ---------------------------------------------------------------------------

def init_gaussians_from_sfm(
    points: np.ndarray,
    points_rgb: np.ndarray,
    scene_scale: float,
    device: torch.device,
    sh_degree: int = 3,
    init_opacity: float = 0.1,
) -> dict:
    """Initialize Gaussian parameters from SfM points."""
    N = len(points)
    print(f"[Init] {N} Gaussians from SfM points (opacity={init_opacity})")

    means = torch.from_numpy(points).float().to(device)

    # RGB to SH (degree 0 coefficient)
    C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))
    rgbs = torch.from_numpy(points_rgb).float().to(device) / 255.0
    sh0 = (rgbs - 0.5) / C0
    sh0 = sh0.unsqueeze(1)  # (N, 1, 3)
    num_sh_coeffs = (sh_degree + 1) ** 2 - 1
    shN = torch.zeros(N, num_sh_coeffs, 3, device=device)

    # Initial scales from KNN distances (pure PyTorch, no sklearn)
    pts_gpu = means  # already on device
    # Compute pairwise distances in batches to avoid OOM on large point clouds
    k = 4
    batch_size = min(4096, N)
    avg_dists = torch.zeros(N, device=device)
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        dists = torch.cdist(pts_gpu[i:end], pts_gpu)  # (batch, N)
        topk = dists.topk(k, dim=1, largest=False)  # k smallest (includes self=0)
        avg_dists[i:end] = topk.values[:, 1:].mean(dim=1)  # skip self
    scales = torch.log((avg_dists * 0.5).clamp(min=1e-8))
    scales = scales.unsqueeze(-1).expand(-1, 3).contiguous()  # (N, 3)

    # Initial opacities (logit space)
    opacities = torch.logit(torch.full((N,), init_opacity, device=device))

    # Identity quaternions
    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = 1.0

    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    })


# ---------------------------------------------------------------------------
# Camera pose refinement (SE(3) deltas, adapted from gsplat examples)
# ---------------------------------------------------------------------------

def _rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrices (Zhou et al.)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class CameraOptModule(torch.nn.Module):
    """Per-image learnable SE(3) pose correction.

    Embeddings are zero-initialized; a zero embedding maps to the identity
    transform, so training starts exactly at the SfM poses.
    """

    def __init__(self, num_cameras: int):
        super().__init__()
        # 3 translation + 6D rotation per camera
        self.embeds = torch.nn.Embedding(num_cameras, 9)
        torch.nn.init.zeros_(self.embeds.weight)
        # identity 6D rotation offset: [1,0,0, 0,1,0]
        self.register_buffer(
            "identity_6d", torch.tensor([1.0, 0, 0, 0, 1.0, 0]),
        )

    def forward(self, c2w: torch.Tensor, embed_id: int) -> torch.Tensor:
        """Apply the learned correction for camera *embed_id* to c2w (4, 4)."""
        delta = self.embeds(torch.tensor([embed_id], device=c2w.device))[0]
        trans = delta[:3]
        rot = _rotation_6d_to_matrix(delta[3:] + self.identity_6d)
        transform = torch.eye(4, device=c2w.device, dtype=c2w.dtype)
        transform[:3, :3] = rot
        transform[:3, 3] = trans
        return torch.matmul(c2w, transform)


# ---------------------------------------------------------------------------
# SSIM (standard PyTorch, no fused_ssim dependency)
# ---------------------------------------------------------------------------

def _fspecial_gauss(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    return g / g.sum()


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Compute SSIM between two images. Input: (B, C, H, W) in [0, 1]."""
    window = _fspecial_gauss(window_size, 1.5, img1.device)
    window = window.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=3)
    mu2 = F.conv2d(img2, window, padding=pad, groups=3)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=3) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=3) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# Rendering helper
# ---------------------------------------------------------------------------

def render_splats(
    splats,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int,
    background: torch.Tensor,
    rasterize_mode: str = "antialiased",
    render_mode: str = "RGB",
    eps2d: float = None,
):
    """Rasterize the splats for one camera. Returns (renders, alphas, info).

    render_mode="RGB+ED" appends a 4th channel = expected depth (same pass,
    no extra render), used for the edge-aware geometric regularizer.

    eps2d (SA-GS): gsplat's 2D screen-space dilation (default 0.3 px^2). Leave
    None to keep the built-in default (unchanged production behavior); pass a
    value from compute_eps2d() at zoom/eval render time to keep the low-pass
    filter matched to the render's pixel footprint (anti-aliasing at zoom).
    """
    kwargs = {}
    if eps2d is not None:
        kwargs["eps2d"] = float(eps2d)
    return rasterization(
        means=splats["means"],
        quats=splats["quats"] / (splats["quats"].norm(dim=-1, keepdim=True) + 1e-12),
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
        viewmats=viewmat[None],
        Ks=K[None],
        width=width,
        height=height,
        sh_degree=sh_degree,
        near_plane=0.01,
        far_plane=1e10,
        backgrounds=background[None],
        rasterize_mode=rasterize_mode,
        render_mode=render_mode,
        packed=False,
        **kwargs,
    )


def compute_eps2d(train_res, render_res, train_f, render_f,
                  train_dist=1.0, render_dist=1.0, base_eps=0.3):
    """SA-GS anti-aliasing rule (arXiv 2403.19615) via gsplat's `eps2d` knob.

    gsplat dilates every projected 2D Gaussian by a fixed screen-space variance
    `eps2d` (0.3) so nothing shrinks below ~1 px. That low-pass filter is
    calibrated for the TRAINING sampling rate; at a different render sampling
    rate (zoom / resolution / object distance) it becomes mismatched -> the
    familiar zoom aliasing + dilation halo. SA-GS rescales the filter by the
    squared sampling-rate ratio:

        r = (render_res/train_res) * (render_f/train_f) * (train_dist/render_dist)
        eps2d = 0.3 * r^2

    `train_f`/`render_f` are NORMALIZED focal lengths (fx / image_width) so the
    `render_res/train_res` term is not double-counted (a same-FOV resolution
    change moves render_res and pixel-focal together but leaves normalized focal
    fixed). `*_dist` are optional mean object distances (pass 1.0 when unknown).
    Coherent with rasterize_mode="antialiased" — its compensation reads the SAME
    dilated covariance. Returns the eps2d to pass to render_splats(eps2d=...).
    """
    r = (render_res / train_res) * (render_f / train_f) * (train_dist / render_dist)
    return base_eps * r * r


def edge_aware_depth_smoothness(depth, image, beta: float = 10.0):
    """Edge-aware 2nd-order (curvature) depth smoothness.

    Penalizes depth *curvature* (not gradient) so genuine planar slopes are
    allowed but bumpy "mush" on flat surfaces is removed. Weighted by the
    inverse image gradient so it acts only where the image is flat (textureless
    plastic) and is silent at real edges (keys, labels). This is the unifying
    SOTA lever for textureless 3DGS (AtomGS / PGSR / DN-Splatter).

    depth: (H, W) expected depth. image: (H, W, 3) in [0, 1].
    """
    gray = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2])
    # second derivative of depth (Laplacian, separable)
    d2x = depth[:, 2:] - 2 * depth[:, 1:-1] + depth[:, :-2]
    d2y = depth[2:, :] - 2 * depth[1:-1, :] + depth[:-2, :]
    # central image gradient magnitude (aligned with the 2nd-derivative centers)
    ix = (gray[:, 2:] - gray[:, :-2]).abs()
    iy = (gray[2:, :] - gray[:-2, :]).abs()
    wx = torch.exp(-beta * ix)  # ~1 on flat regions, ~0 at edges
    wy = torch.exp(-beta * iy)
    return (d2x.abs() * wx).mean() + (d2y.abs() * wy).mean()


def erank_loss(log_scales, w_erank=0.01, w_thin=0.0, eps=1e-7):
    """Effective-rank anti-needle regularizer (arXiv 2406.11672, Eq. 7-10).

    A Gaussian's covariance eigenvalues are s_i^2. Their normalized entropy
    exp(H) is the *effective rank* in [1, 3]: 3 = isotropic ball, 1 = a
    degenerate needle/disk. Needles are the dominant floater/aliasing artifact
    (they project to bright specks that shatter under zoom). We penalize the
    effective rank falling below ~2 with a log-barrier hinge — the orthogonal
    anti-needle term MCMC's opacity/scale regs don't cover.

    log_scales: raw (pre-exp) per-Gaussian scales, i.e. splats["scales"] (N, 3).
    Returns a scalar. w_thin stays 0 under MCMC (scale_reg already shrinks size;
    the hinge is the shape term). The `.clamp_min(...)` guards are OURS: MCMC
    relocates dead Gaussians to log_scales ~ -inf, whose exp underflows to 0 and
    would make p/log(p) NaN — the clamps keep the barrier finite there.
    """
    lam = torch.exp(log_scales * 2)                              # s_i^2 = cov eigenvalues
    p = lam / lam.sum(dim=1, keepdim=True).clamp_min(1e-30)
    H = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
    erank = torch.exp(H)                                         # in [1,3]; needle->1
    hinge = torch.clamp(-torch.log(erank - 1.0 + eps), min=0.0)  # active iff erank<~2
    thin = torch.exp(log_scales).min(dim=1).values
    return w_erank * hinge.mean() + w_thin * thin.mean()


_lpips_model = None


def get_lpips(device):
    """Lazily build a frozen VGG-LPIPS for use as a perceptual training loss.

    Perceptual loss is the most-reproduced lever for perceived sharpness; it
    recovers high-frequency detail that L1+SSIM averages out. Computed on a
    random crop per step (bounds 4K memory) — over many steps it covers the
    whole frame.
    """
    global _lpips_model
    if _lpips_model is None:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        m = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
        m = m.to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _lpips_model = m
    return _lpips_model


def lpips_patch_loss(rendered, gt_image, patch: int, device):
    """LPIPS on a matched random crop of rendered vs GT. Inputs (H,W,3) in [0,1].

    Returns (lpips_value, crop_l1_detached). The crop's L1 error is used to GATE
    the perceptual loss toward under-resolved regions (high error) and away from
    already-well-resolved ones (low error) — so LPIPS sharpens the textureless
    deck without over-texturing the well-resolved keyboard.
    """
    H, W = gt_image.shape[:2]
    ph, pw = min(patch, H), min(patch, W)
    top = int(np.random.randint(0, H - ph + 1))
    left = int(np.random.randint(0, W - pw + 1))
    r = rendered[top:top + ph, left:left + pw].permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    g = gt_image[top:top + ph, left:left + pw].permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    crop_l1 = (r - g).abs().mean().detach()
    return get_lpips(device)(r, g), crop_l1, r, g


# ---------------------------------------------------------------------------
# Gram / style loss (Difix3D+ texture-baking term) — VGG feature correlations
# ---------------------------------------------------------------------------

_vgg_model = None
_vgg_norm = None


def get_vgg(device):
    """Frozen VGG16 feature extractor (up to relu3_3) for the Gram/style loss.

    Difix3D+ (arXiv 2503.01774) bakes high-frequency *texture* into the splats
    with a Gram-matrix term (L = L2 + LPIPS + 0.5·Gram) that LPIPS alone misses.
    """
    global _vgg_model
    if _vgg_model is None:
        from torchvision.models import vgg16, VGG16_Weights
        m = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
        m = m.to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _vgg_model = m
    return _vgg_model


def _gram(f):
    b, c, h, w = f.shape
    f = f.reshape(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


def gram_loss(r, g, device):
    """Multi-layer Gram MSE on matched BCHW [0,1] crops (r=rendered, g=target)."""
    global _vgg_norm
    if _vgg_norm is None:
        _vgg_norm = (
            torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1),
            torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1),
        )
    mean, std = _vgg_norm
    x = (r - mean) / std
    y = ((g - mean) / std).detach()   # target: no gradient
    vgg = get_vgg(device)
    taps = {3, 8, 15}  # relu1_2, relu2_2, relu3_3
    loss = 0.0
    for i, layer in enumerate(vgg):
        x, y = layer(x), layer(y)
        if i in taps:
            loss = loss + F.mse_loss(_gram(x), _gram(y))
        if i >= 15:
            break
    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Optional deterministic seed (default None = current stochastic behavior,
    # production unchanged). Used for fair A/B runs. Seeds torch/numpy/random
    # (init, crop positions, MCMC noise) AND the Prefetcher workers' view-
    # sampling streams (seed + worker_id). Residual nondeterminism remains:
    # worker->queue interleaving is OS-scheduling dependent and gsplat's
    # atomic rasterization kernels are not bit-reproducible — so seeded runs
    # are statistically matched, not bit-identical.
    if getattr(args, "seed", None) is not None:
        import random as _random
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        _random.seed(args.seed)
        print(f"[Train] Seeded RNG with {args.seed}")

    rasterize_mode = "antialiased" if args.antialiased else "classic"
    use_mcmc = args.strategy == "mcmc"

    # Load data — coarse-to-fine uses two resolution levels of the same scene
    data_fine = load_colmap_data(
        args.data_dir,
        factor=args.data_factor,
        test_every=args.test_every,
        normalize=args.normalize_world_space,
    )
    coarse_until = args.coarse_until if args.coarse_factor > args.data_factor else 0
    if args.extra_views and coarse_until > 0:
        print("[Train] extra_views given — disabling coarse-to-fine for this run")
        coarse_until = 0
    if coarse_until > 0:
        data_coarse = load_colmap_data(
            args.data_dir,
            factor=args.coarse_factor,
            test_every=args.test_every,
            normalize=args.normalize_world_space,
        )
        print(f"[Train] Coarse-to-fine: factor {args.coarse_factor} until step "
              f"{coarse_until}, then factor {args.data_factor}")
    else:
        data_coarse = data_fine

    # Extra (enhanced pseudo-view) images, poses already normalized
    if args.extra_views:
        append_extra_views(data_fine, args.extra_views)

    # Init Gaussians: from checkpoint (distill/refine rounds) or SfM points
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=True)
        splats = torch.nn.ParameterDict({
            k: torch.nn.Parameter(v.to(device)) for k, v in ckpt["splats"].items()
        })
        print(f"[Init] {splats['means'].shape[0]} Gaussians from checkpoint "
              f"{Path(args.init_ckpt).name}")
    else:
        # MCMC works best with higher initial opacity
        splats = init_gaussians_from_sfm(
            data_fine["points"], data_fine["points_rgb"], data_fine["scene_scale"],
            device, sh_degree=args.sh_degree,
            init_opacity=0.5 if use_mcmc else 0.1,
        )

    # Optimizers (same LR as gsplat defaults)
    optimizers = {
        "means": torch.optim.Adam([splats["means"]], lr=1.6e-4, eps=1e-15),
        "scales": torch.optim.Adam([splats["scales"]], lr=5e-3, eps=1e-15),
        "quats": torch.optim.Adam([splats["quats"]], lr=1e-3, eps=1e-15),
        "opacities": torch.optim.Adam([splats["opacities"]], lr=5e-2, eps=1e-15),
        "sh0": torch.optim.Adam([splats["sh0"]], lr=2.5e-3, eps=1e-15),
        "shN": torch.optim.Adam([splats["shN"]], lr=1.25e-4, eps=1e-15),
    }

    # Means LR scheduler (exponential decay from 1.6e-4 to 1.6e-6)
    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=(1e-2) ** (1.0 / max(args.max_steps, 1)),
    )

    # Densification strategy (refine_stop_iter <= 0 → per-strategy default;
    # checkpoint-resumed refine rounds default to no densification)
    refine_stop = args.refine_stop_iter
    if refine_stop <= 0:
        if args.init_ckpt:
            refine_stop = 1
        else:
            refine_stop = 25_000 if use_mcmc else 15_000
    refine_stop = min(refine_stop, args.max_steps)
    if use_mcmc:
        strategy = MCMCStrategy(
            cap_max=args.cap_max,
            refine_stop_iter=refine_stop,
            verbose=False,
        )
        strategy_state = strategy.initialize_state()
        print(f"[Train] MCMC densification: cap_max={args.cap_max}, "
              f"refine_stop={refine_stop}")
    else:
        strategy = DefaultStrategy(
            refine_stop_iter=refine_stop,
            verbose=False,
        )
        strategy_state = strategy.initialize_state(scene_scale=data_fine["scene_scale"])
        print(f"[Train] Default densification: refine_stop={refine_stop}")

    # Optional camera pose refinement
    pose_module = None
    pose_optimizer = None
    if args.pose_opt:
        pose_module = CameraOptModule(len(data_fine["image_names"])).to(device)
        pose_optimizer = torch.optim.Adam(
            pose_module.parameters(),
            lr=args.pose_opt_lr,
            weight_decay=args.pose_opt_reg,
        )
        print(f"[Train] Pose refinement: lr={args.pose_opt_lr}, reg={args.pose_opt_reg}")

    # Output dirs
    result_dir = Path(args.result_dir)
    ckpt_dir = result_dir / "ckpts"
    ply_dir = result_dir / "ply"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)

    # Parse save/eval steps
    save_steps = set(args.save_steps) if args.save_steps else {args.max_steps}
    save_steps.add(args.max_steps)  # always save final

    # Training
    train_indices = data_fine["train_indices"]
    num_train = len(train_indices)
    ssim_lambda = args.ssim_lambda

    print(f"[Train] Starting {args.max_steps} steps, {num_train} train images")
    print(f"[Train] strategy={args.strategy}, antialiased={args.antialiased}, "
          f"ssim_lambda={ssim_lambda}, sh_degree={args.sh_degree}")
    sys.stdout.flush()

    # Optional per-view bilateral grid (BilaRF) — a learned 3D affine color grid
    # applied to the RENDER during training only (dropped at inference). Absorbs
    # per-image exposure/WB drift so the canonical Gaussians converge to a
    # consistent radiance instead of spawning exposure-compensating floaters.
    bil_grids = bil_opt = _bil_slice = _bil_tv = None
    train_pos = {}
    if args.bilateral_grid:
        try:  # runs as a module (import) or as a subprocess script (bare name)
            from core.training.lib_bilagrid import (
                BilateralGrid, slice as _bil_slice, total_variation_loss as _bil_tv,
            )
        except ModuleNotFoundError:
            from lib_bilagrid import (
                BilateralGrid, slice as _bil_slice, total_variation_loss as _bil_tv,
            )
        bil_grids = BilateralGrid(num_train, grid_X=16, grid_Y=16, grid_W=8).to(device)
        bil_opt = torch.optim.Adam(bil_grids.parameters(), lr=2e-3, eps=1e-15)
        train_pos = {int(g): i for i, g in enumerate(train_indices)}
        print(f"[Train] Bilateral grid ON ({num_train} views, 16x16x8)")

    # Optional Wasserstein-Distortion (WD-R) perceptual loss (texture-statistic;
    # alternative to LPIPS). Warm-up shares --lpips_start; ungated, own random crop.
    _wd_fn = None
    if args.wd_lambda > 0:
        try:
            from core.training.wd_loss import wasserstein_distortion as _wd_fn
        except ModuleNotFoundError:
            from wd_loss import wasserstein_distortion as _wd_fn
        print(f"[Train] WD-R perceptual loss ON (wd_lambda={args.wd_lambda}, "
              f"start={args.lpips_start})")

    # Visibility loss: precompute the never-seen (empty-space floater) mask once
    unseen_mask = None
    if args.vis_reg > 0:
        unseen_mask = compute_unseen_mask(splats["means"].detach(), data_fine, device)
        print(f"[Train] Visibility loss ON (vis_reg={args.vis_reg}): "
              f"{int(unseen_mask.sum())}/{unseen_mask.numel()} Gaussians seen by no camera")

    # Sparse-depth supervision setup (rendered depth vs SfM points at observed pixels)
    depth_obs = None
    points_t = None
    if args.depth_lambda > 0 and data_fine.get("point_obs") is not None:
        depth_obs = data_fine["point_obs"]
        points_t = torch.from_numpy(data_fine["points"]).float().to(device)
        n_obs = int(np.median([len(o) for o in depth_obs]))
        print(f"[Train] Sparse depth loss ON (depth_lambda={args.depth_lambda}, "
              f"start={args.depth_start}): ~{n_obs} SfM points/view")

    # Dense mono-depth supervision setup (WP-F1) — load the per-view reliability
    # + depth_weight from the MoGe-2 prior cache; flag load_priors so load_image
    # pulls the aligned prior-depth map (fine resolution) into each batch.
    if args.mono_depth_lambda > 0:
        prior_dir = Path(args.data_dir) / "priors"
        align_path = prior_dir / "align.json"
        if not align_path.is_file():
            print(f"[Train] WARNING: --mono_depth_lambda set but no prior cache at "
                  f"{align_path}; mono-depth loss DISABLED. Build it with "
                  f"`python -m core.priors {args.data_dir} --factor {args.data_factor}`.")
            args.mono_depth_lambda = 0.0
        else:
            aj = json.loads(align_path.read_text(encoding="utf-8"))
            # The cache stores maps at ONE resolution; a factor mismatch would
            # silently no-op the loss (per-batch shape check skips every view),
            # so it is a hard error, not a warning.
            cache_factor = int(aj.get("meta", {}).get("factor", -1))
            if cache_factor != args.data_factor:
                raise SystemExit(
                    f"[Train] ERROR: prior cache at {prior_dir} was built at "
                    f"factor {cache_factor} but --data_factor={args.data_factor}. "
                    f"Rebuild it: `python -m core.priors {args.data_dir} "
                    f"--factor {args.data_factor}`")
            prior_align = {k: (bool(v["reliable"]), float(v["depth_weight"]))
                           for k, v in aj["images"].items()}
            n_rel = sum(1 for r, _ in prior_align.values() if r)
            # Count train views that will ACTUALLY load a prior (file present
            # + stored shape matches this camera's post-undistort frame).
            # Zero coverage = misbuilt cache -> hard error instead of printing
            # "Mono-depth loss ON" while silently supervising nothing.
            n_covered = 0
            for ti in train_indices:
                ti = int(ti)
                stem = os.path.splitext(
                    os.path.basename(data_fine["image_paths"][ti]))[0]
                dpath = prior_dir / "depth" / f"{stem}.npy"
                if not dpath.is_file():
                    continue
                try:  # mmap reads only the npy header (cheap)
                    shp = np.load(dpath, mmap_mode="r").shape
                except Exception:
                    continue
                wh = data_fine["imsize_dict"].get(data_fine["camera_ids"][ti])
                if wh is not None and tuple(shp[:2]) != (wh[1], wh[0]):
                    continue
                n_covered += 1
            if n_covered == 0:
                raise SystemExit(
                    f"[Train] ERROR: --mono_depth_lambda is set but 0/"
                    f"{len(train_indices)} train views have a usable prior in "
                    f"{prior_dir / 'depth'} (missing files or wrong resolution). "
                    f"Rebuild it: `python -m core.priors {args.data_dir} "
                    f"--factor {args.data_factor}`")
            # priors are at data_factor (fine) resolution -> only the fine
            # dataset loads them (coarse phase renders at coarse_factor), so
            # the loss cannot start before the fine phase begins.
            args.mono_depth_start = max(args.mono_depth_start, coarse_until + 1)
            data_fine["load_priors"] = True
            data_fine["prior_dir"] = str(prior_dir)
            data_fine["prior_align"] = prior_align
            print(f"[Train] Mono-depth loss ON (mono_depth_lambda="
                  f"{args.mono_depth_lambda}, start={args.mono_depth_start}, "
                  f"use_weight={args.mono_depth_use_weight}): mode={aj['meta']['mode']}, "
                  f"{n_rel}/{len(prior_align)} views reliable, "
                  f"{n_covered}/{len(train_indices)} train views covered")

    running_loss = 0.0
    running_psnr = 0.0
    running_crop_l1 = None  # EMA of LPIPS-crop L1, for gated perceptual loss
    running_wd_tex = None   # EMA of GT-crop texture energy, for gated WD
    running_erank = None    # EMA of erank hinge term, for per-loss logging
    running_mono = None     # EMA of mono-depth term, for per-loss logging
    t0 = time.time()

    # Background image prefetch (overlaps disk I/O + decode with GPU compute)
    prefetcher = Prefetcher(data_coarse if coarse_until > 0 else data_fine,
                            train_indices, seed=args.seed)

    for step in range(1, args.max_steps + 1):
        if step == coarse_until + 1 and coarse_until > 0:
            print(f"[Train] Switching to fine resolution (factor {args.data_factor}) at step {step}")
            sys.stdout.flush()
            prefetcher.close()
            # Offset the seed so the fine phase doesn't replay the coarse
            # phase's view sequence (still deterministic given --seed).
            prefetcher = Prefetcher(data_fine, train_indices,
                                    seed=None if args.seed is None
                                    else args.seed + 1000)

        # Random training image (prefetched on background threads)
        batch = prefetcher.get()
        idx = batch["idx"]
        gt_image = batch["image"].to(device, non_blocking=True)  # (H, W, 3)
        K = batch["K"].to(device)  # (3, 3)
        c2w = batch["c2w"].to(device)  # (4, 4)

        if pose_module is not None:
            c2w = pose_module(c2w, int(idx))

        H, W = gt_image.shape[:2]
        w2c = torch.linalg.inv(c2w)  # (4, 4)

        # Random background for masked scenes
        if args.random_bkgd:
            bkgd = torch.rand(3, device=device)
        else:
            bkgd = torch.zeros(3, device=device)

        geo_on = args.geo_reg and step >= args.geo_reg_start
        depth_on = depth_obs is not None and step >= args.depth_start
        # Mono-depth is fine-phase only (priors are at data_factor res); the
        # prior_depth key is present only for the fine dataset's batches.
        mono_on = (args.mono_depth_lambda > 0 and step >= args.mono_depth_start
                   and "prior_depth" in batch)
        renders, alphas, info = render_splats(
            splats, w2c, K, W, H, args.sh_degree, bkgd, rasterize_mode,
            render_mode="RGB+ED" if (geo_on or depth_on or mono_on) else "RGB",
        )
        rendered = renders[0][..., :3].clamp(0.0, 1.0)  # (H, W, 3)

        # Bilateral-grid color correction (training-only) — match this view's
        # exposure/WB so photometric loss doesn't drive exposure floaters.
        if bil_grids is not None:
            gy, gx = torch.meshgrid(
                (torch.arange(H, device=device) + 0.5) / H,
                (torch.arange(W, device=device) + 0.5) / W,
                indexing="ij",
            )
            grid_xy = torch.stack([gx, gy], dim=-1).unsqueeze(0)        # (1,H,W,2)
            iid = torch.tensor([train_pos.get(int(idx), 0)], device=device)
            rendered = _bil_slice(bil_grids, grid_xy,
                                  rendered.unsqueeze(0), iid)["rgb"][0]

        # L1 + SSIM loss
        l1_loss = F.l1_loss(rendered, gt_image)

        if ssim_lambda > 0:
            rendered_bchw = rendered.permute(2, 0, 1).unsqueeze(0)
            gt_bchw = gt_image.permute(2, 0, 1).unsqueeze(0)
            ssim_val = ssim(rendered_bchw, gt_bchw)
            loss = (1.0 - ssim_lambda) * l1_loss + ssim_lambda * (1.0 - ssim_val)
        else:
            loss = l1_loss

        # Object-mask alpha supervision — the rendered alpha (accumulated opacity) must
        # go to 0 OUTSIDE the object mask. Pre-multiplied-black images alone don't teach
        # this (a black background pixel is satisfied by a black-but-opaque Gaussian), so
        # background floaters grow unpenalized. Supervising alpha directly DELETES them.
        if (args.alpha_loss_lambda > 0 or args.alpha_obj_lambda > 0) and "mask" in batch:
            m = batch["mask"].to(device, non_blocking=True)        # (H, W) in [0,1]
            a = alphas[0, ..., 0]                                  # (H, W) rendered alpha
            if args.alpha_loss_lambda > 0:
                loss = loss + args.alpha_loss_lambda * (a * (1.0 - m)).mean()
            if args.alpha_obj_lambda > 0:
                loss = loss + args.alpha_obj_lambda * ((1.0 - a) * m).mean()

        # Opacity/scale regularization (essential for MCMC, harmless otherwise)
        if args.opacity_reg > 0:
            loss = loss + args.opacity_reg * torch.sigmoid(splats["opacities"]).mean()
        if args.scale_reg > 0:
            loss = loss + args.scale_reg * torch.exp(splats["scales"]).mean()
        # Effective-rank anti-needle hinge (arXiv 2406.11672) — orthogonal to
        # scale_reg: it constrains Gaussian SHAPE (no needles/discs), not size.
        # w_thin stays 0 under MCMC (scale_reg covers size). Default lambda 0 = off.
        # erank_end (official repo's erank_end_iter, inclusive) lets scenes with
        # genuinely thin geometry (wires, blades) release the constraint late.
        if (args.erank_lambda > 0 and step >= args.erank_start
                and (args.erank_end is None or step <= args.erank_end)):
            erank_term = erank_loss(splats["scales"], w_erank=args.erank_lambda, w_thin=0.0)
            loss = loss + erank_term
            running_erank = (erank_term.item() if running_erank is None
                             else 0.95 * running_erank + 0.05 * erank_term.item())
        if unseen_mask is not None:
            loss = loss + args.vis_reg * torch.sigmoid(
                splats["opacities"][unseen_mask]).mean()

        # Sparse-depth supervision — anchor rendered depth to the SfM points the view
        # observes. Re-project points with THIS view's K+pose (so it matches the rendered
        # grid regardless of undistortion), sample rendered depth there, robust L1.
        # Faithful + free: grounds geometry where photometric loss is ambiguous.
        if depth_on:
            obs = depth_obs[int(idx)]
            if len(obs) >= 16:
                P = points_t[torch.from_numpy(obs).to(device)]          # (M, 3) world
                Xc = P @ w2c[:3, :3].T + w2c[:3, 3]                     # (M, 3) camera
                z = Xc[:, 2]
                front = z > 1e-4
                uv = (Xc[front] @ K.T)
                u = uv[:, 0] / uv[:, 2]; v = uv[:, 1] / uv[:, 2]
                zf = z[front]
                inb = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
                if inb.any():
                    depth_map = renders[0][..., 3]                      # (H, W) expected depth
                    gx = (u[inb] / (W - 1)) * 2 - 1
                    gy = (v[inb] / (H - 1)) * 2 - 1
                    grid = torch.stack([gx, gy], -1)[None, None]        # (1,1,K,2)
                    d_r = torch.nn.functional.grid_sample(
                        depth_map[None, None], grid, mode="bilinear",
                        align_corners=True)[0, 0, 0]                    # (K,)
                    zt = zf[inb]
                    # scale by median GT depth so the weight is resolution/scene invariant
                    sc = zt.median().detach() + 1e-6
                    loss = loss + args.depth_lambda * (
                        (d_r - zt).abs() / sc).mean()

        # Dense monocular-depth supervision (WP-F1) — aligned MoGe-2 prior depth
        # vs rendered expected-depth, over valid∩reliable pixels only. Normalized
        # by the median prior depth (scale/resolution invariant, exactly like the
        # sparse --depth_lambda loss). inf = invalid in the prior; the isfinite
        # mask is what stops inf from poisoning the loss/grads.
        if mono_on and batch.get("prior_reliable", False):
            prior_d = batch["prior_depth"].to(device, non_blocking=True)   # (H,W)
            pv = torch.isfinite(prior_d)
            if bool(pv.any()):
                rendered_depth = renders[0][..., 3]                         # (H,W)
                pvd = prior_d[pv]
                sc = pvd.median().detach() + 1e-6
                mono_term = ((rendered_depth[pv] - pvd).abs() / sc).mean()
                w = (float(batch.get("prior_weight", 1.0))
                     if args.mono_depth_use_weight else 1.0)
                loss = loss + args.mono_depth_lambda * w * mono_term
                running_mono = (mono_term.item() if running_mono is None
                                else 0.95 * running_mono + 0.05 * mono_term.item())

        # Edge-aware geometric regularization — flattens textureless surfaces
        # (the "mushy black plastic" fix) without softening textured detail.
        if geo_on:
            depth = renders[0][..., 3]  # (H, W) expected depth, differentiable
            loss = loss + args.geo_reg_lambda * edge_aware_depth_smoothness(
                depth, gt_image, beta=args.geo_reg_beta,
            )

        # Perceptual (LPIPS) loss — recovers high-frequency detail averaged out
        # by L1+SSIM. Optionally GATED by relative crop error so it acts on
        # under-resolved regions and spares already-well-resolved ones (avoids
        # over-texturing at full 4K).
        if (args.lpips_lambda > 0 or args.gram_lambda > 0) and step >= args.lpips_start:
            lp, crop_l1, r_crop, g_crop = lpips_patch_loss(
                rendered, gt_image, args.lpips_patch, device)
            # Gram/style term shares the same crop (Difix3D+ texture-baking lever)
            gram = gram_loss(r_crop, g_crop, device) if args.gram_lambda > 0 else 0.0
            if args.lpips_gated:
                if running_crop_l1 is None:
                    running_crop_l1 = crop_l1
                else:
                    running_crop_l1 = 0.99 * running_crop_l1 + 0.01 * crop_l1
                # gate ~0 on below-average-error (well-resolved) crops, ~1 on
                # above-average-error (under-resolved) crops
                gate = torch.sigmoid(
                    (crop_l1 - running_crop_l1) / (0.25 * running_crop_l1 + 1e-6)
                )
                loss = loss + gate * (args.lpips_lambda * lp + args.gram_lambda * gram)
            else:
                loss = loss + args.lpips_lambda * lp + args.gram_lambda * gram

        # Wasserstein Distortion (WD-R) — texture-statistic VGG loss (own crop,
        # ungated). Compares LOCAL mean+std of features (vs LPIPS pointwise).
        if _wd_fn is not None and step >= args.lpips_start:
            Hc, Wc = gt_image.shape[:2]
            ph, pw = min(args.lpips_patch, Hc), min(args.lpips_patch, Wc)
            top = int(np.random.randint(0, Hc - ph + 1))
            left = int(np.random.randint(0, Wc - pw + 1))
            rc = rendered[top:top + ph, left:left + pw].permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
            gc = gt_image[top:top + ph, left:left + pw].permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
            d_wd = _wd_fn(rc, gc, device)
            if args.wd_gated:
                # gate WD by the GT crop's TEXTURE energy: full WD where there is
                # real texture, ~0 on flat/mush crops (so WD can't synthesize grain
                # floaters where there's no signal). |grad| of the GT crop.
                gg = 0.299 * gc[0, 0] + 0.587 * gc[0, 1] + 0.114 * gc[0, 2]
                tex = ((gg[:, 1:] - gg[:, :-1]).abs().mean()
                       + (gg[1:, :] - gg[:-1, :]).abs().mean()).detach()
                if running_wd_tex is None:
                    running_wd_tex = tex
                else:
                    running_wd_tex = 0.99 * running_wd_tex + 0.01 * tex
                gate = torch.sigmoid((tex - running_wd_tex) / (0.5 * running_wd_tex + 1e-6))
                loss = loss + args.wd_lambda * gate * d_wd
            else:
                loss = loss + args.wd_lambda * d_wd

        # Bilateral-grid total-variation smoothness (keeps the grid well-behaved)
        if bil_grids is not None:
            loss = loss + 10.0 * _bil_tv(bil_grids.grids)

        # PSNR for logging
        with torch.no_grad():
            mse = F.mse_loss(rendered, gt_image)
            psnr = 10.0 * math.log10(1.0 / (mse.item() + 1e-10))
            running_loss = 0.95 * running_loss + 0.05 * loss.item()
            running_psnr = 0.95 * running_psnr + 0.05 * psnr

        # Confidence-gated prior fusion (Phase 1): scale this view's gradient by its
        # loss weight — 1.0 for captured views, <1 for extra/pseudo views in
        # well-observed regions so the prior never overrides real data.
        view_w = float(batch.get("loss_weight", 1.0))
        if view_w != 1.0:
            loss = loss * view_w

        # Densification hooks (strategies gate internally on refine windows)
        strategy.step_pre_backward(
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
        )

        # Backward
        loss.backward()

        # Optimizer step
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        if pose_optimizer is not None:
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
        if bil_opt is not None:
            bil_opt.step()
            bil_opt.zero_grad(set_to_none=True)
        means_scheduler.step()

        if use_mcmc:
            strategy.step_post_backward(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                step=step,
                info=info,
                lr=means_scheduler.get_last_lr()[0],
            )
        else:
            strategy.step_post_backward(
                params=splats,
                optimizers=optimizers,
                state=strategy_state,
                step=step,
                info=info,
            )

        # Progress logging (every 100 steps)
        if step % 100 == 0:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            eta = (args.max_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
            n_gs = splats["means"].shape[0]
            mem = torch.cuda.max_memory_allocated() / 1024**2
            erank_str = (f" | erank={running_erank:.3e}"
                         if running_erank is not None else "")
            mono_str = (f" | mono={running_mono:.4f}"
                        if running_mono is not None else "")
            print(
                f"step={step} | loss={running_loss:.6f} | psnr={running_psnr:.2f} "
                f"| #GS={n_gs} | mem={mem:.0f}MB | {steps_per_sec:.1f} it/s | eta={eta:.0f}s"
                f"{erank_str}{mono_str}"
            )
            sys.stdout.flush()

        # Save checkpoint
        if step in save_steps:
            ckpt_path = ckpt_dir / f"ckpt_{step-1}_rank0.pt"
            ckpt_payload = {
                "step": step,
                "splats": {k: v.data for k, v in splats.items()},
            }
            if pose_module is not None:
                ckpt_payload["pose_embeds"] = pose_module.embeds.weight.data
            torch.save(ckpt_payload, str(ckpt_path))
            print(f"[Checkpoint] Saved {ckpt_path.name}")

            # Export PLY
            try:
                from gsplat import export_splats
                ply_path = ply_dir / f"point_cloud_{step-1}.ply"
                # export_splats expects RAW parameters: the standard 3DGS PLY
                # stores scale in log-space and opacity in logit-space (the
                # viewer applies exp()/sigmoid() on load). Passing activated
                # values here double-activates → giant ~scene-sized splats.
                export_splats(
                    means=splats["means"].detach(),
                    scales=splats["scales"].detach(),
                    quats=splats["quats"].detach(),
                    opacities=splats["opacities"].detach(),
                    sh0=splats["sh0"].detach(),
                    shN=splats["shN"].detach(),
                    format="ply",
                    save_to=str(ply_path),
                )
                print(f"[PLY] Saved {ply_path.name}")
            except Exception as e:
                print(f"[PLY] Export failed: {e}")

            sys.stdout.flush()

    prefetcher.close()

    # Final eval on test set (always at fine resolution, raw poses, black bkgd)
    test_indices = data_fine["test_indices"]
    if len(test_indices) > 0:
        print(f"\n[Eval] Running on {len(test_indices)} test images...")
        sys.stdout.flush()
        test_psnrs = []
        test_ssims = []

        with torch.no_grad():
            for idx in test_indices:
                batch = load_image(data_fine, idx)
                gt_image = batch["image"].to(device)
                K = batch["K"].to(device)
                c2w = batch["c2w"].to(device)
                H, W = gt_image.shape[:2]
                w2c = torch.linalg.inv(c2w)

                bkgd = torch.zeros(3, device=device)
                renders, _, _ = render_splats(
                    splats, w2c, K, W, H, args.sh_degree, bkgd, rasterize_mode,
                )
                rendered = renders[0].clamp(0, 1)

                mse = F.mse_loss(rendered, gt_image)
                test_psnr = 10.0 * math.log10(1.0 / (mse.item() + 1e-10))
                test_psnrs.append(test_psnr)

                r_bchw = rendered.permute(2, 0, 1).unsqueeze(0)
                g_bchw = gt_image.permute(2, 0, 1).unsqueeze(0)
                test_ssim = ssim(r_bchw, g_bchw).item()
                test_ssims.append(test_ssim)

        avg_psnr = np.mean(test_psnrs)
        avg_ssim = np.mean(test_ssims)
        print(f"[Eval] PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f}")

        # Write results JSON
        results = {
            "psnr": float(avg_psnr),
            "ssim": float(avg_ssim),
            "num_gaussians": int(splats["means"].shape[0]),
            "steps": args.max_steps,
            "strategy": args.strategy,
            "antialiased": args.antialiased,
            "pose_opt": args.pose_opt,
            "data_factor": args.data_factor,
            "coarse_factor": args.coarse_factor if coarse_until > 0 else None,
        }
        stats_dir = result_dir / "stats"
        stats_dir.mkdir(exist_ok=True)
        (stats_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"[Stats] Written to {stats_dir / 'results.json'}")
    else:
        print("[Eval] No test images (test_every may be larger than dataset)")

    total_time = time.time() - t0
    print(f"\n[Done] Training complete in {total_time:.1f}s")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Minimal gsplat trainer for Apex Reality")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--data_factor", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional RNG seed for matched A/B runs (default "
                             "None = current stochastic behavior). Seeds "
                             "torch/numpy/random + Prefetcher view sampling; "
                             "worker interleaving and gsplat atomics stay "
                             "nondeterministic (statistically matched, not "
                             "bit-identical)")
    parser.add_argument("--test_every", type=int, default=8)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--ssim_lambda", type=float, default=0.2)
    parser.add_argument("--normalize_world_space",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random_bkgd", action="store_true", default=False)
    parser.add_argument("--save_steps", type=int, nargs="*", default=None)
    # Densification
    parser.add_argument("--strategy", type=str, default="mcmc",
                        choices=["default", "mcmc"])
    parser.add_argument("--cap_max", type=int, default=1_000_000,
                        help="MCMC Gaussian budget (VRAM control)")
    parser.add_argument("--refine_stop_iter", type=int, default=0,
                        help="0 = strategy default (MCMC 25000, default 15000)")
    parser.add_argument("--opacity_reg", type=float, default=0.01)
    parser.add_argument("--scale_reg", type=float, default=0.01)
    parser.add_argument("--vis_reg", type=float, default=0.0,
                        help="Visibility loss: push opacity->0 for Gaussians seen by "
                             "no training/pseudo camera (kills empty-space floaters)")
    # Effective-rank anti-needle regularizer (arXiv 2406.11672)
    parser.add_argument("--erank_lambda", type=float, default=0.0,
                        help="Weight of the effective-rank hinge (penalizes "
                             "needle/disc Gaussians; erank<~2). 0 = off. ~0.01 on.")
    parser.add_argument("--erank_start", type=int, default=7000,
                        help="Step to start the erank hinge (after coarse geometry "
                             "forms; too early fights initial anisotropy)")
    parser.add_argument("--erank_end", type=int, default=None,
                        help="Last step (inclusive) the erank hinge is applied "
                             "(official erank_end_iter). None = no cutoff. Set "
                             "for scenes with genuinely thin geometry "
                             "(wires, blades) to release the constraint late")
    # Object-mask alpha supervision (kills background floaters on masked object scans)
    parser.add_argument("--alpha_loss_lambda", type=float, default=0.0,
                        help="Weight of L_bg = mean(rendered_alpha * (1-mask)): pushes "
                             "rendered opacity ->0 OUTSIDE the object mask. Needs masks/ "
                             "or masks_{factor}/ next to images. 0 = off.")
    parser.add_argument("--alpha_obj_lambda", type=float, default=0.0,
                        help="Optional: push alpha ->1 INSIDE the mask "
                             "(mean((1-alpha)*mask)); fills see-through object holes. "
                             "Use sparingly (can thicken thin parts). 0 = off.")
    # Sparse-depth supervision (free, faithful: rendered depth vs SfM points)
    parser.add_argument("--depth_lambda", type=float, default=0.0,
                        help="Weight of sparse depth loss: L1(rendered_depth, SfM_point_z) "
                             "at observed pixels. Grounds geometry in under-textured "
                             "regions; free (uses existing SfM points). 0 = off.")
    parser.add_argument("--depth_start", type=int, default=2000,
                        help="Step to start depth loss (after coarse geometry forms)")
    # Dense monocular-depth supervision (WP-F1): aligned MoGe-2 prior depth vs
    # rendered expected-depth, masked by validity ∩ per-view reliability. Prior
    # cache built by `python -m core.priors <data_dir> --factor N`.
    parser.add_argument("--mono_depth_lambda", type=float, default=0.0,
                        help="Weight of dense masked mono-depth loss "
                             "L1(aligned MoGe prior depth, rendered depth) over "
                             "valid-and-reliable pixels, normalized by median prior "
                             "depth (scale/res invariant). Needs priors/ cache. 0 = off.")
    parser.add_argument("--mono_depth_start", type=int, default=2000,
                        help="Step to start mono-depth loss (fine phase only; "
                             "priors are at data_factor resolution)")
    parser.add_argument("--mono_depth_use_weight",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Scale the per-view mono-depth loss by the "
                             "alignment-residual depth_weight from align.json")
    # Anti-aliasing (Mip-Splatting)
    parser.add_argument("--antialiased",
                        action=argparse.BooleanOptionalAction, default=True)
    # Pose refinement
    parser.add_argument("--pose_opt", action="store_true", default=False)
    parser.add_argument("--pose_opt_lr", type=float, default=1e-5)
    parser.add_argument("--pose_opt_reg", type=float, default=1e-6)
    # Coarse-to-fine schedule
    parser.add_argument("--coarse_factor", type=int, default=0,
                        help="Resolution factor for the densification phase "
                             "(0 = disabled, must be > data_factor)")
    parser.add_argument("--coarse_until", type=int, default=15000,
                        help="Step at which to switch from coarse to fine resolution")
    # Distill / refine rounds
    parser.add_argument("--init_ckpt", type=str, default=None,
                        help="Resume Gaussians from a checkpoint instead of SfM init")
    parser.add_argument("--extra_views", type=str, default=None,
                        help="JSON of additional train views (normalized poses)")
    # Edge-aware geometric regularization (textureless/flat-surface "mush" fix)
    parser.add_argument("--geo_reg", action="store_true", default=False,
                        help="Enable edge-aware 2nd-order depth smoothness")
    parser.add_argument("--geo_reg_lambda", type=float, default=0.05,
                        help="Weight of the geometric regularization term")
    parser.add_argument("--geo_reg_beta", type=float, default=10.0,
                        help="Edge sensitivity: higher = reg confined to flatter regions")
    parser.add_argument("--geo_reg_start", type=int, default=3000,
                        help="Step to start geo reg (after coarse geometry forms)")
    # Perceptual (LPIPS) loss
    parser.add_argument("--lpips_lambda", type=float, default=0.0,
                        help="Weight of LPIPS perceptual loss (0 = off)")
    parser.add_argument("--lpips_patch", type=int, default=512,
                        help="Random crop size for LPIPS (bounds memory at 4K)")
    parser.add_argument("--lpips_start", type=int, default=3000,
                        help="Step to start LPIPS (after coarse structure forms)")
    parser.add_argument("--lpips_gated", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Gate LPIPS toward under-resolved (high-error) crops "
                             "so it doesn't over-texture well-resolved regions")
    parser.add_argument("--gram_lambda", type=float, default=0.0,
                        help="Weight of VGG Gram/style loss (Difix3D+ texture "
                             "baking; 0 = off; shares LPIPS crop+gate+start)")
    parser.add_argument("--wd_lambda", type=float, default=0.0,
                        help="Weight of Wasserstein-Distortion (WD-R) perceptual "
                             "loss (texture-statistic; 0 = off; warm-up = lpips_start)")
    parser.add_argument("--wd_gated", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Gate WD by GT-crop texture energy (fires on textured "
                             "crops, ~0 on flats -> no grain floaters on mush)")
    parser.add_argument("--bilateral_grid", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Per-view BilaRF color grid (training-only) to absorb "
                             "exposure/WB drift; dropped at inference")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
