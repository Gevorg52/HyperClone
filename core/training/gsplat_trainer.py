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
  - Scale regularization; opacity regularization is opt-in (default 0 since 2026-10-17)

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
    holdout_list: str | None = None,
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
    if holdout_list:
        # Explicit test-set membership by COLMAP image name (one per line, '#' comments).
        # Needed whenever views come in GROUPS that must not be split — e.g. the 12
        # rig crops of one 360 panorama (WP-O3): they share an optical centre, so
        # holding out only some of them leaks the held-out view into training.
        # Default None => the every-Nth rule below is untouched.
        wanted = set()
        with open(holdout_list, "r", encoding="utf8") as fh:
            for line in fh:
                line = line.split("#", 1)[0].strip()
                if line:
                    wanted.add(line.replace("\\", "/"))
        by_name = {n.replace("\\", "/"): i for i, n in enumerate(image_names)}
        missing = sorted(wanted - set(by_name))
        if missing:
            raise ValueError(
                f"--holdout_list {holdout_list}: {len(missing)} name(s) are not in the "
                f"reconstruction, e.g. {missing[:3]}")
        test_indices = np.array(sorted(by_name[n] for n in wanted), dtype=np.int64)
        mask = np.zeros(len(indices), dtype=bool)
        mask[test_indices] = True
        train_indices = indices[~mask]
        print(f"[Data] holdout_list: {len(test_indices)} held out by name "
              f"({os.path.basename(holdout_list)})")
    elif _holdout:
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
    Optional ``"hole_mask"`` per entry points to an explicit per-pixel unobserved
    mask (PNG or .npy, >0 = unobserved hole) for WP-R1 observed-preservation gating.
    """
    entries = json.loads(Path(extra_views_json).read_text(encoding="utf-8"))
    if not entries:
        return 0

    base = len(data["image_names"])
    extra_c2ws = []
    extra_indices = set(data.get("extra_view_indices", set()))
    extra_hole_masks = dict(data.get("extra_hole_masks", {}))
    for i, e in enumerate(entries):
        cid = f"extra_{i}"
        idx = base + i
        data["Ks_dict"][cid] = np.array(e["K"], dtype=np.float64)
        data["params_dict"][cid] = np.empty(0, dtype=np.float32)
        data["camera_ids"].append(cid)
        data["image_paths"].append(e["image"])
        data["image_names"].append(Path(e["image"]).name)
        extra_c2ws.append(np.array(e["c2w"], dtype=np.float32))
        extra_indices.add(idx)
        if e.get("hole_mask"):
            extra_hole_masks[idx] = str(e["hole_mask"])

    data["extra_view_indices"] = extra_indices
    data["extra_hole_masks"] = extra_hole_masks
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


def compute_extra_unobs_weight(
    alpha_init: torch.Tensor,
    tau_obs: float = 0.85,
    gamma: float = 1.0,
    hole_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """WP-R1 spatial unobserved weight map w_unobs(u,v) in [0, 1] for an extra view.

    Where the frozen initial checkpoint already has high opacity (alpha_init >= tau_obs),
    the scene is observed and w_unobs = 0. Where alpha_init < tau_obs, w_unobs ramps up as
    clamp((tau_obs - alpha_init) / tau_obs, 0, 1) ** gamma. If an explicit hole_mask is
    also provided (e.g. OGGSplat unobserved mask in extra_views.json), w_unobs is gated by
    hole_mask so semi-transparent observed fringes outside the hole receive zero weight.
    """
    tau = max(float(tau_obs), 1e-6)
    w = ((tau - alpha_init.float()) / tau).clamp(0.0, 1.0)
    if float(gamma) != 1.0:
        w = w.pow(float(gamma))
    if hole_mask is not None:
        hm = hole_mask.to(device=w.device, dtype=w.dtype).clamp(0.0, 1.0)
        w = w * hm
    return w


def blend_extra_view_target(
    gt_pseudo: torch.Tensor,
    render_init: torch.Tensor,
    w_unobs: torch.Tensor,
    anchor_strength: float = 1.0,
) -> torch.Tensor:
    """WP-R1 target replacement/blending on an extra view.

    Blends observed pixels (1 - w_unobs) of the pseudo-view target toward the frozen
    initial render I_init with strength `anchor_strength` in [0, 1]:
      w_anchor = clamp(anchor_strength, 0, 1) * (1 - w_unobs)
      I_target = (1 - w_anchor) * gt_pseudo + w_anchor * render_init
    At anchor_strength == 1.0:
      I_target = w_unobs * gt_pseudo + (1 - w_unobs) * render_init
    so observed pixels (w_unobs == 0) are identical to render_init while unobserved
    hole pixels (w_unobs == 1) pass gt_pseudo through untouched.
    """
    a = float(max(0.0, min(1.0, anchor_strength)))
    if a <= 0.0:
        return gt_pseudo
    w_u = w_unobs.to(device=gt_pseudo.device, dtype=gt_pseudo.dtype)
    if w_u.ndim == 2:
        w_u = w_u.unsqueeze(-1)
    w_anchor = a * (1.0 - w_u.clamp(0.0, 1.0))
    return (1.0 - w_anchor) * gt_pseudo + w_anchor * render_init.to(
        device=gt_pseudo.device, dtype=gt_pseudo.dtype)


def precompute_extra_view_anchors(
    splats,
    data: dict,
    sh_degree: int,
    rasterize_mode: str,
    device: torch.device,
    tau_obs: float = 0.85,
    gamma: float = 1.0,
) -> dict[int, dict]:
    """WP-R1: precompute frozen initial render I_init, alpha_init, and w_unobs
    for every extra_views entry before the training loop begins."""
    extra_indices = sorted(int(i) for i in data.get("extra_view_indices", ()))
    if not extra_indices:
        return {}
    hole_masks = data.get("extra_hole_masks", {})
    anchors: dict[int, dict] = {}
    t0 = time.time()
    bkgd = torch.zeros(3, device=device)
    with torch.no_grad():
        for idx in extra_indices:
            b = load_image(data, idx)
            H, W = b["image"].shape[:2]
            K = b["K"].to(device)
            c2w = b["c2w"].to(device)
            w2c = torch.linalg.inv(c2w)
            renders, alphas, _ = render_splats(
                splats, w2c, K, W, H, sh_degree, bkgd, rasterize_mode,
            )
            render_init = renders[0][..., :3].clamp(0.0, 1.0).detach()
            alpha_init = alphas[0, ..., 0].clamp(0.0, 1.0).detach()
            hm_tensor = None
            hm_path = hole_masks.get(idx)
            if hm_path and os.path.isfile(hm_path):
                if hm_path.lower().endswith(".npy"):
                    hm_np = np.load(hm_path).astype(np.float32)
                else:
                    hm_raw = imageio.imread(hm_path)
                    if hm_raw.ndim == 3:
                        hm_raw = hm_raw[..., 0]
                    hm_np = hm_raw.astype(np.float32) / (255.0 if hm_raw.max() > 1.0 else 1.0)
                if hm_np.shape[:2] != (H, W):
                    hm_np = cv2.resize(hm_np, (W, H), interpolation=cv2.INTER_LINEAR)
                hm_tensor = torch.from_numpy(hm_np).to(device=device, dtype=torch.float32)
            w_unobs = compute_extra_unobs_weight(
                alpha_init, tau_obs=tau_obs, gamma=gamma, hole_mask=hm_tensor,
            )
            w_solid = ((alpha_init - 0.95) / 0.05).clamp(0.0, 1.0)
            anchors[idx] = {
                "render_init": render_init,
                "alpha_init": alpha_init,
                "w_unobs": w_unobs,
                "w_solid": w_solid,
            }
    dt = time.time() - t0
    mean_unobs = float(np.mean([a["w_unobs"].mean().item() for a in anchors.values()]))
    print(f"[WP-R1] Precomputed {len(anchors)} extra-view anchors in {dt:.2f}s "
          f"(tau_obs={tau_obs}, gamma={gamma}, mean w_unobs={mean_unobs:.4f})")
    return anchors



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
    if mpaths is not None and index < len(mpaths) and mpaths[index] and os.path.isfile(mpaths[index]):
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
        "is_extra_view": bool(index in data.get("extra_view_indices", ())),
    }
    if mask is not None:
        out["mask"] = torch.from_numpy(mask).float()  # (H, W) in [0,1]

    # Geometric priors (WP-F1 mono-depth + WP-F4 planar): aligned MoGe-2 maps are
    # stored already in the undistorted+cropped training frame, so they drop in
    # without any remap. inf/0 = invalid (masked in the loss). Attached only when
    # the shape matches the (post-undistort) rendered frame — a mismatch means
    # wrong factor / coarse phase -> skip (keeps the losses fine-phase only).
    need_priors = data.get("load_priors") or data.get("load_planar_priors")
    if need_priors:
        stem = os.path.splitext(os.path.basename(data["image_paths"][index]))[0]
        pdir = data["prior_dir"]
        ppath = os.path.join(pdir, "depth", stem + ".npy")
        if os.path.isfile(ppath):
            pd = np.load(ppath)
            if pd.shape[:2] == image.shape[:2]:
                out["prior_depth"] = torch.from_numpy(pd.astype(np.float32))
                rel, dw = data.get("prior_align", {}).get(stem, (False, 0.0))
                out["prior_reliable"] = bool(rel)
                out["prior_weight"] = float(dw)
                # WP-F4: normal + conf + inverse-Canny edge + plane-instance map.
                # Guard each read so a partially-built cache degrades gracefully
                # (the loss setup already hard-errors on zero coverage).
                if data.get("load_planar_priors"):
                    npath = os.path.join(pdir, "normal", stem + ".npy")
                    cpath = os.path.join(pdir, "conf", stem + ".npy")
                    epath = os.path.join(pdir, "edge_mask", stem + ".png")
                    mpath = os.path.join(pdir, "plane_mask", stem + ".npy")
                    if os.path.isfile(npath):
                        nrm = np.load(npath)
                        if nrm.shape[:2] == image.shape[:2]:
                            out["prior_normal"] = torch.from_numpy(nrm.astype(np.float32))
                    if os.path.isfile(cpath):
                        cf = np.load(cpath)
                        if cf.shape[:2] == image.shape[:2]:
                            out["prior_conf"] = torch.from_numpy(cf.astype(np.float32))
                    if os.path.isfile(epath):
                        em = imageio.imread(epath)
                        if em.ndim == 3:
                            em = em[..., 0]
                        if em.shape[:2] == image.shape[:2]:
                            # 255 = away-from-edge (supervise) -> weight 1.0
                            out["prior_edge"] = torch.from_numpy(
                                (em > 0).astype(np.float32))
                    if os.path.isfile(mpath):
                        plm = np.load(mpath)
                        if plm.shape[:2] == image.shape[:2]:
                            out["prior_plane_mask"] = torch.from_numpy(
                                plm.astype(np.int64))
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
                for k in ("mask", "prior_depth", "prior_normal", "prior_conf",
                          "prior_edge", "prior_plane_mask"):
                    if k in batch:
                        batch[k] = batch[k].pin_memory()
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
    return_map: bool = False,
) -> torch.Tensor:
    """Compute SSIM between two images. Input: (B, C, H, W) in [0, 1].

    return_map=True returns the per-pixel SSIM map (B, C, H, W) instead of its mean, so a
    caller can average it over a validity mask (--loss_mask) rather than over the frame.
    """
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
    return ssim_map if return_map else ssim_map.mean()


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


# ---------------------------------------------------------------------------
# Planar-prior helpers (WP-F4) — ported from PlanarGS (MIT; see THIRD_PARTY.md):
#   planar/co_planar.py::ProjectDepth and common_utils/graphics_utils.py::
#   {Depth2Pointscam, pcd2normal}. Adapted to our RGB+ED renderer (single (H,W)
#   z-depth map, camera-frame; MoGe-2 cache normals point toward the camera,
#   matching pcd2normal's convention — verified 2026-07-03, no sign flip needed).
# ---------------------------------------------------------------------------

def co_planar_depth(depth, plane_mask, inv_K):
    """Per-instance regularized-LS plane fit on the unprojected rendered depth,
    reconstructed back to a z-depth map (PlanarGS L_p).

    depth: (H, W) rendered expected z-depth (differentiable unless detached).
    plane_mask: (H, W) long, 0=bg, 1..M plane instances.
    inv_K: (3, 3) inverse intrinsics of THIS view (undistorted training frame).
    Returns plane_depth (H, W): inside each instance = the fitted-plane z-depth,
    elsewhere = the input depth (masked out by the caller anyway).

    Plane fit: param = (P Pᵀ + 0.01 I)⁻¹ P·1 ; plane z-depth = 1/(paramᵀ K⁻¹ ũ).
    """
    H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    pm = plane_mask
    # Zero an 8-px border so ill-conditioned edge pixels don't enter a fit
    # (PlanarGS does the same to keep gradients well-behaved).
    pm = pm.clone()
    pm[:8, :] = 0; pm[-8:, :] = 0; pm[:, :8] = 0; pm[:, -8:] = 0

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype), indexing="ij")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1),
                          torch.ones(H * W, device=device, dtype=dtype)], dim=0)  # (3,HW)
    rays = (inv_K.to(dtype) @ coords)                          # (3,HW) = K⁻¹ũ
    pts = rays * depth.reshape(1, -1)                          # (3,HW) camera pts
    plane_depth = depth.reshape(-1).clone()
    labels = pm.reshape(-1)
    eye = 0.01 * torch.eye(3, device=device, dtype=dtype)
    for i in torch.unique(labels):
        if int(i) == 0:
            continue
        sel = labels == i
        P = pts[:, sel]                                        # (3,N)
        A = P @ P.T + eye                                      # (3,3)
        param = torch.linalg.solve(A, P.sum(dim=1, keepdim=True))          # (3,1)
        denom = (param.transpose(0, 1) @ rays[:, sel]).reshape(-1)         # (N,)
        # Sign-PRESERVING degenerate guard (matches plane_masks._plane_depth_along):
        # for a grazing ray (paramᵀr → 0) a fixed +1e-6 would map plane_depth to
        # ~1e6 — a spurious huge target that stays *finite* (so the isfinite guard
        # in the loss would miss it) and can dominate that instance's mean. This
        # keeps |plane_depth| ~ O(1) in normalized-world scale instead.
        denom = torch.where(denom.abs() < 1e-6,
                            torch.sign(denom) + (denom == 0).to(denom.dtype), denom)
        plane_depth[sel] = 1.0 / denom
    return plane_depth.reshape(H, W)


def normal_from_depth(depth, inv_K):
    """Camera-frame per-pixel normal from a rendered z-depth map (PlanarGS
    NormalFromDepth = pcd2normal ∘ Depth2Pointscam). Returns (H, W, 3), unit,
    pointing toward the camera on fronto-parallel surfaces (matches MoGe cache).
    """
    H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype), indexing="ij")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1),
                          torch.ones(H * W, device=device, dtype=dtype)], dim=0)
    rays = (inv_K.to(dtype) @ coords)                          # (3,HW)
    xyz = (rays * depth.reshape(1, -1)).transpose(0, 1).reshape(H, W, 3)   # (H,W,3)
    left = xyz[1:-1, :-2]; right = xyz[1:-1, 2:]
    top = xyz[:-2, 1:-1]; bottom = xyz[2:, 1:-1]
    nrm = torch.cross(right - left, top - bottom, dim=-1)      # PlanarGS order
    nrm = torch.nn.functional.normalize(nrm, dim=-1)
    nrm = torch.nn.functional.pad(nrm.permute(2, 0, 1), (1, 1, 1, 1),
                                  mode="replicate").permute(1, 2, 0)
    return nrm


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
        holdout_list=args.holdout_list,
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
            holdout_list=args.holdout_list,
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
        raw_splats = {k: v.to(device) for k, v in ckpt["splats"].items()}
        _repo_root = str(Path(__file__).resolve().parents[2])
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        from core.sh_utils import resize_shN, sh_degree_of
        ckpt_deg = sh_degree_of(raw_splats)
        if ckpt_deg != args.sh_degree:
            raw_splats["shN"] = resize_shN(raw_splats["shN"], args.sh_degree)
            print(f"[Init] Resized checkpoint SH degree {ckpt_deg} -> {args.sh_degree} "
                  f"(shN: {raw_splats['shN'].shape[1]} rest coeffs/ch)")
        splats = torch.nn.ParameterDict({
            k: torch.nn.Parameter(v) for k, v in raw_splats.items()
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

    # WP-R1: Precompute frozen anchor render + alpha + unobserved weight map for extra_views
    extra_anchors = {}
    if args.extra_views and (
        getattr(args, "extra_unobserved_only", False)
        or getattr(args, "extra_anchor_observed", 0.0) > 0.0
    ):
        extra_anchors = precompute_extra_view_anchors(
            splats, data_fine, args.sh_degree, rasterize_mode, device,
            tau_obs=getattr(args, "extra_obs_tau", 0.85),
            gamma=getattr(args, "extra_obs_gamma", 1.0),
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

    # Optional PPISP (WP-F5; NVIDIA nv-tlabs/ppisp, arXiv 2601.18336) — a physically
    # plausible ISP applied to the RENDER: per-frame exposure + colour (white balance)
    # homography, per-camera vignetting + camera response curve. Like the bilateral grid
    # it only touches the image, never the Gaussians, so the exported PLY stays a standard
    # 3DGS PLY holding the canonical (ISP-free) radiance. Default OFF. Mutually exclusive
    # with --bilateral_grid (both absorb the same per-view photometric drift).
    ppisp_mod = ppisp_opts = ppisp_scheds = None
    ppisp_cam_of = {}
    ppisp_act_step = None        # controller distillation start (0-based scheduler epoch)
    splats_frozen = False
    running_ppisp = None         # EMA of the PPISP regularizer, for per-loss logging
    if args.ppisp:
        if args.bilateral_grid:
            raise SystemExit("[Train] ERROR: --ppisp and --bilateral_grid are mutually "
                             "exclusive (both model per-view photometric drift).")
        if args.extra_views:
            raise SystemExit("[Train] ERROR: --ppisp does not support --extra_views "
                             "(pseudo views have no PPISP frame).")
        try:
            from ppisp import PPISP, PPISPConfig
        except ImportError as e:
            raise SystemExit(
                f"[Train] ERROR: --ppisp needs the `ppisp` package ({e}). It is built into "
                f"the gsplat-main venv only: D:/REPOS/HYPERclone/venvs/gsplat_main "
                f"(tools/blackwell_session_2026-06/build_ppisp.bat).")
        # PPISP camera = physical camera (vignetting/CRF); frame = training image.
        ppisp_cam_of = {cid: k for k, cid in enumerate(
            sorted({int(c) for c in data_fine["camera_ids"]}))}
        train_pos = {int(g): i for i, g in enumerate(train_indices)}
        ppisp_cfg = PPISPConfig(
            use_controller=bool(args.ppisp_controller),
            controller_distillation=True,
            controller_activation_ratio=float(args.ppisp_controller_start),
            scheduler_decay_max_steps=int(args.max_steps),
        )
        ppisp_mod = PPISP(num_cameras=len(ppisp_cam_of), num_frames=num_train,
                          config=ppisp_cfg)
        ppisp_opts = ppisp_mod.create_optimizers()
        # create_schedulers is MANDATORY: forward() reads the scheduler's last_epoch to
        # decide when the controller takes over.
        ppisp_scheds = ppisp_mod.create_schedulers(ppisp_opts, args.max_steps)
        if args.ppisp_controller:
            ppisp_act_step = int(args.ppisp_controller_start * args.max_steps)
        ctrl_txt = ("OFF" if ppisp_act_step is None else
                    f"ON, distillation from step {ppisp_act_step + 1} "
                    f"(splats frozen, MCMC stopped)")
        print(f"[Train] PPISP ON: {len(ppisp_cam_of)} camera(s), {num_train} frames, "
              f"controller={ctrl_txt}")

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

    # Planar-prior supervision setup (WP-F4, PlanarGS port) — three optional
    # losses on the MoGe-2 cache + plane-instance masks, all default OFF:
    #   L_p  co-planarity (--planar_lambda, needs plane_mask)
    #   L_rd prior-depth  (--prior_depth_lambda, needs conf+edge)
    #   L_rn prior-normal (--prior_normal_lambda, needs plane_mask+normal+conf+edge)
    # Same fine-phase gating + prior cache as mono-depth (priors at data_factor).
    planar_any = (args.planar_lambda > 0 or args.prior_depth_lambda > 0
                  or args.prior_normal_lambda > 0)
    need_plane_mask = args.planar_lambda > 0 or args.prior_normal_lambda > 0
    if planar_any:
        prior_dir = Path(args.data_dir) / "priors"
        align_path = prior_dir / "align.json"
        if not align_path.is_file():
            raise SystemExit(
                f"[Train] ERROR: planar losses set but no prior cache at "
                f"{align_path}. Build it: `python -m core.priors {args.data_dir} "
                f"--factor {args.data_factor}`"
                + (f" then `python -m core.plane_masks {args.data_dir} "
                   f"--factor {args.data_factor}`" if need_plane_mask else ""))
        aj = json.loads(align_path.read_text(encoding="utf-8"))
        cache_factor = int(aj.get("meta", {}).get("factor", -1))
        if cache_factor != args.data_factor:
            raise SystemExit(
                f"[Train] ERROR: prior cache at {prior_dir} was built at factor "
                f"{cache_factor} but --data_factor={args.data_factor}. Rebuild it.")
        # Reuse prior_align from the mono-depth block if present, else build it.
        if not data_fine.get("prior_align"):
            data_fine["prior_align"] = {
                k: (bool(v["reliable"]), float(v["depth_weight"]))
                for k, v in aj["images"].items()}
        # Coverage check for plane masks (L_p / L_rn) — 0 usable masks is a hard
        # error, not a silent no-op (mirrors the mono-depth coverage guard).
        n_pm = 0
        for ti in train_indices:
            stem = os.path.splitext(
                os.path.basename(data_fine["image_paths"][int(ti)]))[0]
            mpath = prior_dir / "plane_mask" / f"{stem}.npy"
            if not mpath.is_file():
                continue
            try:
                shp = np.load(mpath, mmap_mode="r").shape
            except Exception:
                continue
            wh = data_fine["imsize_dict"].get(data_fine["camera_ids"][int(ti)])
            if wh is not None and tuple(shp[:2]) != (wh[1], wh[0]):
                continue
            n_pm += 1
        if need_plane_mask and n_pm == 0:
            raise SystemExit(
                f"[Train] ERROR: --planar_lambda/--prior_normal_lambda set but "
                f"0/{len(train_indices)} train views have a usable plane_mask in "
                f"{prior_dir / 'plane_mask'}. Build it: "
                f"`python -m core.plane_masks {args.data_dir} "
                f"--factor {args.data_factor}`")
        # conf/edge caches gate L_rd (and L_rn) per-pixel — a missing/empty cache
        # would make those losses a silent no-op, so hard-error (core.priors builds
        # depth/normal/conf/edge_mask together, so this only trips on a broken cache).
        need_conf_edge = args.prior_depth_lambda > 0 or args.prior_normal_lambda > 0
        if need_conf_edge:
            for sub in ("conf", "edge_mask"):
                d = prior_dir / sub
                if not d.is_dir() or not any(d.iterdir()):
                    raise SystemExit(
                        f"[Train] ERROR: L_rd/L_rn need priors/{sub}/ but it is "
                        f"missing/empty at {d}. Rebuild the cache: "
                        f"`python -m core.priors {args.data_dir} "
                        f"--factor {args.data_factor}`")
        # Fine-phase only (priors at data_factor res).
        floor = coarse_until + 1
        args.planar_start = max(args.planar_start, floor)
        args.prior_depth_start = max(args.prior_depth_start, floor)
        args.prior_normal_start = max(args.prior_normal_start, floor)
        data_fine["load_planar_priors"] = True
        data_fine["prior_dir"] = str(prior_dir)
        print(f"[Train] Planar losses ON: "
              f"L_p(planar_lambda={args.planar_lambda}, start={args.planar_start}, "
              f"detach={args.planar_detach}) | "
              f"L_rd(prior_depth_lambda={args.prior_depth_lambda}, "
              f"start={args.prior_depth_start}) | "
              f"L_rn(prior_normal_lambda={args.prior_normal_lambda}, "
              f"start={args.prior_normal_start}) | "
              f"{n_pm}/{len(train_indices)} train views have plane masks")

    running_loss = 0.0
    running_psnr = 0.0
    running_crop_l1 = None  # EMA of LPIPS-crop L1, for gated perceptual loss
    running_wd_tex = None   # EMA of GT-crop texture energy, for gated WD
    running_erank = None    # EMA of erank hinge term, for per-loss logging
    running_mono = None     # EMA of mono-depth term, for per-loss logging
    running_planar = None   # EMA of L_p co-planarity term (WP-F4)
    running_prd = None      # EMA of L_rd prior-depth term (WP-F4)
    running_prn = None      # EMA of L_rn prior-normal term (WP-F4)
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

        # PPISP controller distillation (faithful to upstream): from the activation step on,
        # freeze the splats (and poses) and STOP MCMC relocation + noise injection — freezing
        # grads alone does not stop MCMC noise (EXECUTION_PLAN WP-F5).
        if (ppisp_act_step is not None and not splats_frozen
                and (step - 1) >= ppisp_act_step):
            for p_ in splats.values():
                p_.requires_grad_(False)
            if pose_module is not None:
                for p_ in pose_module.parameters():
                    p_.requires_grad_(False)
            splats_frozen = True
            print(f"[Train] PPISP controller distillation from step {step}: splats + poses "
                  f"frozen, densification/MCMC noise stopped")
            sys.stdout.flush()

        if pose_module is not None:
            c2w = pose_module(c2w, int(idx))

        H, W = gt_image.shape[:2]
        w2c = torch.linalg.inv(c2w)  # (4, 4)

        is_extra_view = bool(batch.get("is_extra_view", False))
        extra_anchor = extra_anchors.get(int(idx)) if is_extra_view else None

        # Random background for masked scenes (keep black on gated extra_views so random
        # background colour never pollutes semi-transparent hole pixels)
        if args.random_bkgd and extra_anchor is None:
            bkgd = torch.rand(3, device=device)
        else:
            bkgd = torch.zeros(3, device=device)

        geo_on = args.geo_reg and step >= args.geo_reg_start
        depth_on = depth_obs is not None and step >= args.depth_start
        # Mono-depth is fine-phase only (priors are at data_factor res); the
        # prior_depth key is present only for the fine dataset's batches.
        mono_on = (args.mono_depth_lambda > 0 and step >= args.mono_depth_start
                   and "prior_depth" in batch)
        # Planar-prior losses (WP-F4), fine-phase only (prior maps present only
        # in the fine dataset's batches). L_p / L_rn need the plane-instance map.
        planar_on = (args.planar_lambda > 0 and step >= args.planar_start
                     and "prior_plane_mask" in batch)
        prd_on = (args.prior_depth_lambda > 0 and step >= args.prior_depth_start
                  and "prior_depth" in batch and "prior_conf" in batch
                  and "prior_edge" in batch)
        prn_on = (args.prior_normal_lambda > 0 and step >= args.prior_normal_start
                  and "prior_plane_mask" in batch and "prior_normal" in batch
                  and "prior_conf" in batch and "prior_edge" in batch)
        sh_deg_step = (min((step - 1) // args.sh_degree_interval, args.sh_degree)
                       if args.sh_degree_interval > 0 else args.sh_degree)
        renders, alphas, info = render_splats(
            splats, w2c, K, W, H, sh_deg_step, bkgd, rasterize_mode,
            render_mode="RGB+ED" if (geo_on or depth_on or mono_on
                                     or planar_on or prd_on or prn_on) else "RGB",
        )
        if ppisp_mod is not None:
            # PPISP takes the UNCLAMPED radiance (its exposure may darken values > 1;
            # the CRF clamps internally) — the canonical radiance is what the PLY holds.
            rendered = ppisp_mod(
                renders[0][..., :3], resolution=(W, H),
                camera_idx=ppisp_cam_of[int(data_fine["camera_ids"][int(idx)])],
                frame_idx=train_pos[int(idx)],
            ).clamp(0.0, 1.0)
        else:
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

        # Object-extract refit (WP-D1): scope the photometric loss to the object.
        # Composite render+GT over black inside the mask so background pixels (which
        # the deleted background Gaussians no longer explain) contribute ~0 loss and
        # exert no pull on the object. Default OFF; alpha_loss (below) still deletes
        # any residual background alpha using the un-masked alpha channel.
        if args.rgb_object_masked and "mask" in batch:
            m_rgb = batch["mask"].to(device, non_blocking=True).unsqueeze(-1)  # (H,W,1)
            rendered = rendered * m_rgb
            gt_image = gt_image * m_rgb

        # Loss validity mask (WP-O3) — EXCLUDE pixels from the photometric loss instead of
        # compositing them over black. `rgb_object_masked` (above) zeroes both sides, which
        # (a) divides the loss magnitude by the valid fraction and (b) manufactures hard
        # black edges at the mask boundary that the SSIM windows then chase. On the 360 rig
        # crops (mean 33% valid) that combination made MCMC blow up — see
        # `_infusion/wpo3_RECORD.md`. Here both terms are averaged over the VALID pixels
        # only: L1 over the mask, and the per-pixel SSIM map over the mask. Default OFF.
        loss_mask = None
        if args.loss_mask and "mask" in batch:
            loss_mask = batch["mask"].to(device, non_blocking=True)          # (H,W) [0,1]

        # WP-R1 observed-preservation gate on extra_views pseudo-views:
        #   (1) --extra_anchor_observed > 0 blends observed pixels (1 - w_unobs) of gt_image
        #       with the frozen initial checkpoint render I_init so generator drift / boundary
        #       feathers cannot corrupt observed targets.
        #   (2) --extra_unobserved_only restricts pixelwise L1/SSIM supervision to w_unobs
        #       AND replaces unsupervised pixels in window/crop targets (11x11 SSIM, 512x512
        #       LPIPS) with rendered.detach() so window/receptive-field bleed into observed or
        #       partially-carved fringe regions is identically zero.
        gt_window = gt_image
        extra_w_unobs = None
        extra_lp_scale = 1.0
        if extra_anchor is not None:
            extra_w_unobs = extra_anchor["w_unobs"].to(device, non_blocking=True)
            extra_w_solid = extra_anchor["w_solid"].to(device, non_blocking=True)
            a_str = float(max(0.0, min(1.0, getattr(args, "extra_anchor_observed", 0.0))))
            unobs_only = bool(getattr(args, "extra_unobserved_only", False))
            w_ext = float(max(0.0, getattr(args, "extra_loss_weight", 1.0)))
            if unobs_only:
                # Three-zone target:
                #   1. Unobserved hole (w_unobs): supervised by gt_pseudo with weight min(1, w_ext) * w_unobs
                #   2. Solid observed (w_solid, alpha_init >= 0.99): anchored to render_init with weight a_str
                #   3. Partially-carved fringe (tau_obs <= alpha_init <= 0.95): replaced with rendered.detach()
                #      so broken semi-transparent fringe renders exert identically zero gradient.
                w_u = (min(1.0, w_ext) * extra_w_unobs).unsqueeze(-1)
                w_a = (a_str * extra_w_solid * (1.0 - extra_w_unobs)).unsqueeze(-1)
                w_free = (1.0 - w_u - w_a).clamp(0.0, 1.0)
                gt_image = (
                    w_u * gt_image
                    + w_a * extra_anchor["render_init"].to(device, non_blocking=True)
                    + w_free * rendered.detach()
                )
                gt_window = gt_image
                extra_lp_scale = float((w_u + w_a).mean().item())
            else:
                if a_str > 0.0:
                    gt_image = blend_extra_view_target(
                        gt_image, extra_anchor["render_init"], extra_w_unobs,
                        anchor_strength=a_str,
                    )
                if w_ext != 1.0:
                    w_e = min(1.0, w_ext)
                    gt_image = w_e * gt_image + (1.0 - w_e) * rendered.detach()
                    extra_lp_scale = w_e
                gt_window = gt_image
        elif is_extra_view and float(getattr(args, "extra_loss_weight", 1.0)) != 1.0:
            w_e = float(max(0.0, min(1.0, getattr(args, "extra_loss_weight", 1.0))))
            gt_image = w_e * gt_image + (1.0 - w_e) * rendered.detach()
            gt_window = gt_image
            extra_lp_scale = w_e

        # L1 + SSIM loss
        if loss_mask is not None:
            w = loss_mask.unsqueeze(-1)
            denom = w.sum() * rendered.shape[-1] + 1e-8
            l1_loss = ((rendered - gt_image).abs() * w).sum() / denom
        else:
            l1_loss = F.l1_loss(rendered, gt_image)

        if ssim_lambda > 0:
            rendered_bchw = rendered.permute(2, 0, 1).unsqueeze(0)
            gt_bchw = gt_window.permute(2, 0, 1).unsqueeze(0)
            if loss_mask is not None:
                smap = ssim(rendered_bchw, gt_bchw, return_map=True)[0]      # (C,H,W)
                ssim_val = ((smap * loss_mask).sum()
                            / (loss_mask.sum() * smap.shape[0] + 1e-8))
            else:
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
            # WP-S2 object-mode: only supervise depth inside the object mask, so
            # object-only training never anchors Gaussians to background depth.
            if args.mono_depth_object_masked and "mask" in batch:
                pv = pv & (batch["mask"].to(device, non_blocking=True) > 0.5)
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

        # ---- Planar-prior losses (WP-F4, PlanarGS port) ---------------------
        # All operate on the rendered expected-depth; all masked to their
        # respective regions; each normalized so its lambda is scene-scale
        # invariant (our depth is in arbitrary normalized-world units). Added one
        # at a time in the staggered schedule so coarse geometry forms first.
        _inv_K = None  # lazy inverse-K (shared by L_p and L_rn)

        # L_rd — prior-depth: rendered depth vs the aligned MoGe cache, gated by
        # conf × inverse-Canny edge × per-view depth_weight (PlanarGS l2_loss,
        # relative-normalized here). Redundant-but-complementary with mono-depth
        # (adds the conf/edge per-pixel gating mono-depth lacks).
        if prd_on and batch.get("prior_reliable", False):
            prior_d = batch["prior_depth"].to(device, non_blocking=True)   # (H,W)
            pv = torch.isfinite(prior_d)
            if bool(pv.any()):
                rendered_depth = renders[0][..., 3]
                conf = batch["prior_conf"].to(device, non_blocking=True)
                edge = batch["prior_edge"].to(device, non_blocking=True)
                m = pv.float() * conf * edge                               # (H,W)
                denom = m.sum().clamp_min(1.0)
                sc = prior_d[pv].median().detach() + 1e-6
                diff = torch.where(pv, rendered_depth - prior_d,
                                   torch.zeros_like(rendered_depth)) / sc
                prd_term = ((diff * diff) * m).sum() / denom
                # Reuses --mono_depth_use_weight (intentional: same per-view
                # depth_weight from align.json; one toggle for both depth losses).
                w = (float(batch.get("prior_weight", 1.0))
                     if args.mono_depth_use_weight else 1.0)
                loss = loss + args.prior_depth_lambda * w * prd_term
                running_prd = (prd_term.item() if running_prd is None
                               else 0.95 * running_prd + 0.05 * prd_term.item())

        # L_p — co-planarity: fit a plane per instance to the rendered depth and
        # pull the rendered depth toward it (masked L1). Faithful PlanarGS keeps
        # the fit attached (grad flows through both depth and the plane params);
        # --planar_detach exposes the detached variant.
        if planar_on:
            pm = batch["prior_plane_mask"].to(device, non_blocking=True)   # (H,W) long
            # Exclude the 8-px border from the LOSS mask too (PlanarGS zeros it in
            # co_planar's in-place segmask, so its planar_mask drops it as well);
            # this keeps the numerator and denominator consistent (no dilution).
            pm[:8, :] = 0; pm[-8:, :] = 0; pm[:, :8] = 0; pm[:, -8:] = 0
            if int(pm.max()) > 0:
                rendered_depth = renders[0][..., 3]
                _inv_K = torch.linalg.inv(K)
                depth_in = rendered_depth.detach() if args.planar_detach else rendered_depth
                plane_depth = co_planar_depth(depth_in, pm, _inv_K)
                planar_mask = (pm > 0).float()
                pmsum = planar_mask.sum().clamp_min(1.0)
                sc = rendered_depth.detach()[pm > 0].median() + 1e-6
                planar_term = (((rendered_depth - plane_depth).abs()
                                * planar_mask).sum() / pmsum) / sc
                if torch.isfinite(planar_term):
                    loss = loss + args.planar_lambda * planar_term
                    running_planar = (planar_term.item() if running_planar is None
                                      else 0.95 * running_planar + 0.05 * planar_term.item())

        # L_rn — prior-normal: normal derived from the rendered depth vs the MoGe
        # cache normal (L1 + (1−cos)), over planar ∩ conf ∩ edge pixels. Uses a
        # depth-derived normal (no Gaussian-normal rasterization — L_dn skipped).
        if prn_on:
            pm = batch["prior_plane_mask"].to(device, non_blocking=True)
            pm[:8, :] = 0; pm[-8:, :] = 0; pm[:, :8] = 0; pm[:, -8:] = 0  # see L_p note
            if int(pm.max()) > 0:
                rendered_depth = renders[0][..., 3]
                if _inv_K is None:
                    _inv_K = torch.linalg.inv(K)
                dnorm = normal_from_depth(rendered_depth, _inv_K)          # (H,W,3)
                pnorm = batch["prior_normal"].to(device, non_blocking=True)
                conf = batch["prior_conf"].to(device, non_blocking=True)
                edge = batch["prior_edge"].to(device, non_blocking=True)
                m = (pm > 0).float() * conf * edge                        # (H,W)
                msum = m.sum().clamp_min(1.0)
                l1n = ((pnorm - dnorm).abs().sum(-1) * m).sum() / msum
                cos = (pnorm * dnorm).sum(-1)                             # (H,W)
                l2n = ((1.0 - cos) * m).sum() / msum
                prn_term = l1n + l2n
                if torch.isfinite(prn_term):
                    loss = loss + args.prior_normal_lambda * prn_term
                    running_prn = (prn_term.item() if running_prn is None
                                   else 0.95 * running_prn + 0.05 * prn_term.item())

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
                rendered, gt_window, args.lpips_patch, device)
            # Gram/style term shares the same crop (Difix3D+ texture-baking lever)
            gram = gram_loss(r_crop, g_crop, device) if args.gram_lambda > 0 else 0.0
            lp_scale = extra_lp_scale
            if args.lpips_gated:
                if not is_extra_view:
                    if running_crop_l1 is None:
                        running_crop_l1 = crop_l1
                    else:
                        running_crop_l1 = 0.99 * running_crop_l1 + 0.01 * crop_l1
                # gate ~0 on below-average-error (well-resolved) crops, ~1 on
                # above-average-error (under-resolved) crops
                ref_l1 = running_crop_l1 if running_crop_l1 is not None else crop_l1
                gate = torch.sigmoid(
                    (crop_l1 - ref_l1) / (0.25 * ref_l1 + 1e-6)
                )
                loss = loss + lp_scale * gate * (
                    args.lpips_lambda * lp + args.gram_lambda * gram)
            else:
                loss = loss + lp_scale * (
                    args.lpips_lambda * lp + args.gram_lambda * gram)

        # Wasserstein Distortion (WD-R) — texture-statistic VGG loss (own crop,
        # ungated). Compares LOCAL mean+std of features (vs LPIPS pointwise).
        if _wd_fn is not None and step >= args.lpips_start:
            Hc, Wc = gt_image.shape[:2]
            ph, pw = min(args.lpips_patch, Hc), min(args.lpips_patch, Wc)
            top = int(np.random.randint(0, Hc - ph + 1))
            left = int(np.random.randint(0, Wc - pw + 1))
            rc = rendered[top:top + ph, left:left + pw].permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
            gc = gt_window[top:top + ph, left:left + pw].permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
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
        if ppisp_mod is not None:
            ppisp_reg = ppisp_mod.get_regularization_loss()
            loss = loss + ppisp_reg
            running_ppisp = (ppisp_reg.item() if running_ppisp is None
                             else 0.95 * running_ppisp + 0.05 * ppisp_reg.item())

        # PSNR for logging (over the valid pixels when --loss_mask is on, so the logged
        # number is comparable to the held-out masked PSNR instead of being inflated by
        # the excluded region)
        with torch.no_grad():
            if loss_mask is not None:
                w = loss_mask.unsqueeze(-1)
                mse = (((rendered - gt_image) ** 2) * w).sum() / (
                    w.sum() * rendered.shape[-1] + 1e-8)
            else:
                mse = F.mse_loss(rendered, gt_image)
            psnr = 10.0 * math.log10(1.0 / (mse.item() + 1e-10))
            running_loss = 0.95 * running_loss + 0.05 * loss.item()
            running_psnr = 0.95 * running_psnr + 0.05 * psnr

        # Confidence-gated prior fusion (Phase 1): scale this view's gradient by its
        # per-entry JSON loss weight (1.0 for captured views).
        view_w = float(batch.get("loss_weight", 1.0))
        if view_w != 1.0:
            loss = loss * view_w

        # Densification hooks (strategies gate internally on refine windows)
        if not splats_frozen:
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
        if ppisp_mod is not None:
            for opt in ppisp_opts:
                opt.step()
                opt.zero_grad(set_to_none=True)
            for sch in ppisp_scheds:
                sch.step()
        means_scheduler.step()

        if splats_frozen:
            pass  # PPISP controller distillation: no relocation, no MCMC noise
        elif use_mcmc:
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
            planar_str = (f" | Lp={running_planar:.4f}"
                          if running_planar is not None else "")
            prd_str = (f" | Lrd={running_prd:.4f}"
                       if running_prd is not None else "")
            prn_str = (f" | Lrn={running_prn:.4f}"
                       if running_prn is not None else "")
            if running_ppisp is not None:
                with torch.no_grad():
                    ev = ppisp_mod.exposure_params
                    prn_str += (f" | ppisp_reg={running_ppisp:.2e} exp_sd={ev.std().item():.3f}"
                                f" exp_rng=[{ev.min().item():.2f},{ev.max().item():.2f}]")
            print(
                f"step={step} | loss={running_loss:.6f} | psnr={running_psnr:.2f} "
                f"| #GS={n_gs} | mem={mem:.0f}MB | {steps_per_sec:.1f} it/s | eta={eta:.0f}s"
                f"{erank_str}{mono_str}{planar_str}{prd_str}{prn_str}"
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
            if ppisp_mod is not None:
                # Side-car state (consumers that read ckpt["splats"] are unaffected).
                ckpt_payload["ppisp"] = {k: v.detach().cpu() for k, v in
                                         ppisp_mod.state_dict().items()}
                ckpt_payload["ppisp_meta"] = {
                    "camera_of": {int(k): int(v) for k, v in ppisp_cam_of.items()},
                    "train_indices": [int(i) for i in train_indices],
                    "use_controller": bool(args.ppisp_controller),
                    "controller_start_step": (None if ppisp_act_step is None
                                              else int(ppisp_act_step) + 1),
                }
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
        isp_psnrs = []   # PPISP novel-view render (camera_idx real, frame_idx=-1)
        isp_ssims = []

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

                if ppisp_mod is not None:
                    # frame_idx=-1: controller prediction if trained, else identity
                    # per-frame corrections (per-camera vignetting + CRF still applied).
                    isp = ppisp_mod(
                        renders[0][..., :3], resolution=(W, H),
                        camera_idx=ppisp_cam_of[int(data_fine["camera_ids"][int(idx)])],
                        frame_idx=-1,
                    ).clamp(0, 1)
                    mse_i = F.mse_loss(isp, gt_image)
                    isp_psnrs.append(10.0 * math.log10(1.0 / (mse_i.item() + 1e-10)))
                    isp_ssims.append(ssim(isp.permute(2, 0, 1).unsqueeze(0), g_bchw).item())

        avg_psnr = np.mean(test_psnrs)
        avg_ssim = np.mean(test_ssims)
        print(f"[Eval] PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f}")
        if isp_psnrs:
            print(f"[Eval] PPISP novel-view (frame_idx=-1) PSNR={np.mean(isp_psnrs):.2f} "
                  f"SSIM={np.mean(isp_ssims):.4f}  (the line above is the raw PLY render)")

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
        if isp_psnrs:
            results["ppisp"] = True
            results["ppisp_controller"] = bool(args.ppisp_controller)
            results["psnr_ppisp_novel"] = float(np.mean(isp_psnrs))
            results["ssim_ppisp_novel"] = float(np.mean(isp_ssims))
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
    parser.add_argument("--holdout_list", type=str, default=None,
                        help="File of COLMAP image names (one per line) to use as the test "
                             "set instead of the every-Nth --test_every rule. For captures "
                             "whose views come in inseparable GROUPS (e.g. the 12 rig crops "
                             "of one 360 pano). Default off.")
    parser.add_argument("--sh_degree", type=int, default=3, choices=[0, 1, 2, 3, 4],
                        help="Spherical-harmonic degree (0..4; gsplat kernel max is 4)")
    parser.add_argument("--sh_degree_interval", type=int, default=0,
                        help="Progressive SH schedule interval in steps (0 = full --sh_degree "
                             "from step 1; >0 = min((step-1)//interval, sh_degree))")
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
    parser.add_argument("--opacity_reg", type=float, default=0.0,
                        help="Opacity L1 regularizer weight (opt-in). Default 0 since "
                             "2026-10-17: 0.01 AND 0.001 silently destroy many-view "
                             "MCMC scenes (WP-O3/O5, DEVELOPER_NOTES section 6). "
                             "Pre-2026-10 runs used 0.01.")
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
    parser.add_argument("--rgb_object_masked",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Delivery object-extract refit (WP-D1): composite BOTH "
                             "render and GT over black inside the object mask before "
                             "the L1+SSIM photometric loss, so a short self-refit of an "
                             "already-extracted object is not pulled by background "
                             "pixels the deleted background Gaussians can no longer "
                             "explain. Default OFF (production unchanged). Needs masks/ "
                             "or masks_{factor}/ next to the images.")
    parser.add_argument("--loss_mask",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="WP-O3: average the L1 and SSIM terms over the VALID mask "
                             "pixels only (mask=1) instead of over the whole frame, and "
                             "exclude the rest entirely. Unlike --rgb_object_masked this "
                             "neither rescales the loss by the valid fraction nor creates "
                             "black edges at the mask boundary — use it when the mask "
                             "removes a LARGE part of the frame (360 rig crops keep about "
                             "a third). Default OFF (production unchanged). Needs masks/.")
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
    parser.add_argument("--mono_depth_object_masked",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Object-mode (WP-S2): restrict mono-depth supervision "
                             "to inside the object mask (pv &= mask>0.5). Default OFF "
                             "= full-frame (production unchanged). Needs masks/ or "
                             "masks_{factor}/ next to the images.")
    # Planar-prior losses (WP-F4, PlanarGS port). All default 0 = OFF. Need the
    # MoGe-2 prior cache (`python -m core.priors`); L_p/L_rn also need plane
    # masks (`python -m core.plane_masks`). Staggered starts (PlanarGS 30k:
    # 7k/14k/20k) let coarse geometry form first; scale them to shorter runs.
    parser.add_argument("--planar_lambda", type=float, default=0.0,
                        help="L_p co-planarity weight: per-instance LS plane fit "
                             "on rendered depth, masked L1 pull toward the plane "
                             "(PlanarGS 0.5). Needs priors/plane_mask. 0 = off.")
    parser.add_argument("--planar_start", type=int, default=14000,
                        help="Step to start L_p (fine phase only)")
    parser.add_argument("--planar_detach",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Detach the plane fit (PlanarGS default keeps it "
                             "attached — grad flows through both depth and fit)")
    parser.add_argument("--prior_depth_lambda", type=float, default=0.0,
                        help="L_rd prior-depth weight: rendered depth vs aligned "
                             "MoGe cache, gated by conf x inverse-Canny edge x "
                             "per-view depth_weight (PlanarGS 0.05). 0 = off.")
    parser.add_argument("--prior_depth_start", type=int, default=7000,
                        help="Step to start L_rd (fine phase only)")
    parser.add_argument("--prior_normal_lambda", type=float, default=0.0,
                        help="L_rn prior-normal weight: depth-derived normal vs "
                             "MoGe cache normal (L1 + 1-cos), over planar-and-conf-"
                             "and-edge pixels (PlanarGS 0.2). Needs plane_mask. 0=off.")
    parser.add_argument("--prior_normal_start", type=int, default=20000,
                        help="Step to start L_rn (fine phase only)")
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
    # WP-R1: Observed-Preservation Gate for Generative Supervision (default OFF)
    parser.add_argument("--extra_unobserved_only",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="WP-R1: Mask photometric/SSIM/LPIPS loss on extra_views "
                             "to unobserved/under-converged pixels only (default OFF)")
    parser.add_argument("--extra_anchor_observed", type=float, default=0.0,
                        help="WP-R1: Blend extra_views target with frozen init_ckpt "
                             "render on observed pixels (0 = off, 1 = pure anchor on "
                             "observed region)")
    parser.add_argument("--extra_loss_weight", type=float, default=1.0,
                        help="WP-R1: Scalar weight multiplier for total loss on "
                             "extra_views steps (default 1.0 = unweighted)")
    parser.add_argument("--extra_obs_tau", type=float, default=0.95,
                        help="WP-R1: Initial alpha threshold tau_obs above which an "
                             "extra_view pixel is considered observed (default 0.95)")
    parser.add_argument("--extra_obs_gamma", type=float, default=1.0,
                        help="WP-R1: Exponent gamma on ((tau_obs - alpha)/tau_obs)^gamma "
                             "for unobserved weight map (default 1.0)")
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
    # PPISP (WP-F5) — needs the ppisp package (gsplat-main venv). Default OFF.
    parser.add_argument("--ppisp", action=argparse.BooleanOptionalAction, default=False,
                        help="NVIDIA PPISP post-render ISP (per-frame exposure + white "
                             "balance, per-camera vignetting + response curve; arXiv "
                             "2601.18336). Training-time only: the PLY keeps canonical "
                             "radiance. Mutually exclusive with --bilateral_grid.")
    parser.add_argument("--ppisp_controller", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Also train PPISP's per-camera controller (predicts exposure/"
                             "colour for novel views) by distillation: from "
                             "--ppisp_controller_start the splats are FROZEN and MCMC "
                             "relocation/noise is stopped (upstream recipe).")
    parser.add_argument("--ppisp_controller_start", type=float, default=0.8,
                        help="Fraction of --max_steps at which controller distillation "
                             "starts (PPISP default 0.8)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
