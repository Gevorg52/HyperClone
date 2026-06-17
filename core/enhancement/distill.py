"""Difix3D+-style progressive distill-back loop.

Implements the documented source of most of Difix3D+'s quality gain, which
NVIDIA's released Fixer does not ship:

    1. Render pseudo-views along trajectories interpolated between the
       training cameras, from the current 3DGS checkpoint.
    2. Enhance the renders with the diffusion fixer (artifact removal).
    3. Append the enhanced views to the training set.
    4. Resume 3DGS training for a few thousand steps (no densification).
    5. Optionally repeat with the improved checkpoint.

The fixer sees renders from viewpoints *between* training views — exactly
where 3DGS artifacts live — and its cleaned outputs pull the representation
toward artifact-free novel views while the real images anchor fidelity.

All poses written to extra_views.json are in the trainer's NORMALIZED world
space; the trainer appends them after its own normalization, so coordinates
stay consistent across rounds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from core.enhancement import EnhancementBackend, EnhancementConfig
from core.training import TrainingBackend, TrainingConfig, TrainingResult

ProgressCallback = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# Pose interpolation
# ---------------------------------------------------------------------------

def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([
            0.25 / s,
            (R[2, 1] - R[1, 2]) * s,
            (R[0, 2] - R[2, 0]) * s,
            (R[1, 0] - R[0, 1]) * s,
        ])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.empty(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def _matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1, 1))
    s0 = np.sin((1 - t) * theta) / np.sin(theta)
    s1 = np.sin(t * theta) / np.sin(theta)
    return s0 * q0 + s1 * q1


def interpolate_pseudo_poses(
    camtoworlds: np.ndarray,
    train_indices: np.ndarray,
    num_views: int,
    t: float = 0.5,
) -> np.ndarray:
    """Midpoint poses between consecutive training cameras (capture order).

    Returns (num_views, 4, 4) c2w matrices in the same (normalized) space
    as the input cameras.
    """
    order = np.sort(train_indices)
    pairs = list(zip(order[:-1], order[1:]))
    if not pairs:
        raise ValueError("Need at least 2 training cameras to interpolate")

    # Evenly sample pairs along the trajectory
    sel = np.linspace(0, len(pairs) - 1, min(num_views, len(pairs))).astype(int)

    poses = []
    for pi in sel:
        a, b = pairs[pi]
        c2w_a, c2w_b = camtoworlds[a], camtoworlds[b]
        q = _slerp(_quat_from_matrix(c2w_a[:3, :3]),
                   _quat_from_matrix(c2w_b[:3, :3]), t)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = _matrix_from_quat(q)
        pose[:3, 3] = (1 - t) * c2w_a[:3, 3] + t * c2w_b[:3, 3]
        poses.append(pose)
    return np.stack(poses)


# ---------------------------------------------------------------------------
# Rendering pseudo-views from a checkpoint
# ---------------------------------------------------------------------------

def render_pseudo_views(
    checkpoint_dir: str,
    data_dir: str,
    output_dir: str,
    num_views: int,
    data_factor: int = 0,
    antialiased: bool = True,
) -> tuple[list[str], list[dict]]:
    """Render interpolated pseudo-views from the latest checkpoint.

    Returns (render_paths, view_entries) where each entry holds the K and
    normalized c2w used, ready for extra_views.json once the image path is
    swapped for the enhanced version.
    """
    import torch

    from core.training.gsplat_trainer import load_colmap_data, render_splats

    device = torch.device("cuda:0")

    ckpts = sorted(Path(checkpoint_dir).glob("ckpt_*_rank0.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {checkpoint_dir}")
    ckpt = torch.load(str(ckpts[-1]), map_location=device, weights_only=True)
    splats = {k: v.to(device) for k, v in ckpt["splats"].items()}
    sh_degree = int((-1 + (1 + 4 * (1 + splats["shN"].shape[1])) ** 0.5) / 2)

    if data_factor <= 0:
        available = sorted(
            int(p.name.split("_")[1])
            for p in Path(data_dir).glob("images_*")
            if p.is_dir() and p.name.split("_")[1].isdigit()
        )
        data_factor = available[0] if available else 1

    data = load_colmap_data(data_dir, factor=data_factor, test_every=8,
                            normalize=True)

    poses = interpolate_pseudo_poses(
        data["camtoworlds"], data["train_indices"], num_views,
    )

    # Phase-1a coverage gate (validated +1.03 dB): weight each pseudo-view by how
    # FAR it is from the nearest real camera, normalized to [0,1]. The prior then
    # supervises strongly in under-observed regions and ~0 near captured views, so
    # it never overrides real data. (per-view distance-to-nearest-real-cam predicts
    # error at Pearson +0.50; gated beat ungated on every metric.)
    train_c = data["camtoworlds"][data["train_indices"], :3, 3]
    cov = np.array([np.min(np.linalg.norm(train_c - c2w[:3, 3], axis=1)) for c2w in poses])
    wcov = np.clip(cov / (np.percentile(cov, 90) + 1e-9), 0.0, 1.0)

    # Use the first training camera's intrinsics/size for all pseudo-views
    ref_idx = int(data["train_indices"][0])
    ref_cid = data["camera_ids"][ref_idx]
    K = data["Ks_dict"][ref_cid].copy()
    W, H = data["imsize_dict"][ref_cid]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    K_t = torch.from_numpy(K).float().to(device)
    bkgd = torch.zeros(3, device=device)
    mode = "antialiased" if antialiased else "classic"

    render_paths: list[str] = []
    entries: list[dict] = []
    from PIL import Image

    with torch.no_grad():
        for i, c2w in enumerate(poses):
            c2w_t = torch.from_numpy(c2w).float().to(device)
            w2c = torch.linalg.inv(c2w_t)
            renders, _, _ = render_splats(
                splats, w2c, K_t, int(W), int(H), sh_degree, bkgd, mode,
            )
            img = (renders[0].clamp(0, 1) * 255).byte().cpu().numpy()
            path = out / f"pseudo_{i:04d}.png"
            Image.fromarray(img).save(str(path))
            render_paths.append(str(path))
            entries.append({
                "image": str(path),  # swapped for enhanced path later
                "K": K.tolist(),
                "c2w": c2w.tolist(),
                "weight": float(wcov[i]),  # coverage gate (Phase 1a)
            })

    # Release the splats + gsplat CUDA cache from THIS process before the Difix
    # subprocess runs — otherwise the parent holds ~2-3 GiB (3M Gaussians) and
    # starves Difix's 1080p VAE decode, which then thrashes (2-3 min/img).
    del splats, K_t, bkgd
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return render_paths, entries


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def run_distill_loop(
    training_backend: TrainingBackend,
    enhancer: EnhancementBackend,
    data_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    rounds: int = 2,
    views_per_round: int = 80,
    steps_per_round: int = 4000,
    data_factor: int = 0,
    enhancer_extra: Optional[dict] = None,
    train_extra: Optional[dict] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> TrainingResult:
    """Run the progressive distill-back loop.

    Each round: render pseudo-views from the current checkpoint → enhance →
    resume training with the enhanced views appended.  Returns the final
    round's TrainingResult (checkpoint + PLY of the improved model).
    """
    def _progress(frac: float, desc: str):
        if progress_cb:
            progress_cb(frac, desc)

    # Resolve the working resolution once — pseudo-view intrinsics and the
    # resumed training runs must use the same factor.
    if data_factor <= 0:
        available = sorted(
            int(p.name.split("_")[1])
            for p in Path(data_dir).glob("images_*")
            if p.is_dir() and p.name.split("_")[1].isdigit()
        )
        data_factor = available[0] if available else 1

    out_root = Path(output_dir)
    current_ckpt_dir = checkpoint_dir
    result: Optional[TrainingResult] = None

    for r in range(1, rounds + 1):
        round_dir = out_root / f"round_{r}"
        base = (r - 1) / rounds

        # 1. Render pseudo-views
        _progress(base + 0.05 / rounds, f"Round {r}/{rounds}: rendering "
                  f"{views_per_round} pseudo-views...")
        render_paths, entries = render_pseudo_views(
            checkpoint_dir=current_ckpt_dir,
            data_dir=data_dir,
            output_dir=str(round_dir / "renders"),
            num_views=views_per_round,
            data_factor=data_factor,
        )

        # 2. Enhance
        _progress(base + 0.25 / rounds, f"Round {r}/{rounds}: enhancing "
                  f"{len(render_paths)} views...")
        config = EnhancementConfig(
            backend=enhancer.name,
            num_views=len(render_paths),
            extra=enhancer_extra or {},
        )
        enhanced = enhancer.enhance_images(
            image_paths=render_paths,
            output_dir=str(round_dir / "enhanced"),
            config=config,
        )
        if len(enhanced) != len(render_paths):
            raise RuntimeError(
                f"Enhancer returned {len(enhanced)} images for "
                f"{len(render_paths)} renders — cannot map poses"
            )

        # Map enhanced images back to their poses (both name-sorted)
        for entry, enh in zip(entries, sorted(enhanced)):
            entry["image"] = enh

        extra_views_json = round_dir / "extra_views.json"
        extra_views_json.write_text(json.dumps(entries), encoding="utf-8")

        # 3. Resume training with the enhanced views
        _progress(base + 0.5 / rounds, f"Round {r}/{rounds}: resuming training "
                  f"({steps_per_round} steps)...")
        ckpts = sorted(Path(current_ckpt_dir).glob("ckpt_*_rank0.pt"))
        # Conservative defaults: geometry frozen (no densification), L1+SSIM only.
        # train_extra overrides these for an aggressive "add detail / fill holes"
        # pass (e.g. refine_stop_iter>1 to densify, lpips_lambda to transfer
        # high-freq detail). GOTCHA: if densification is enabled, cap_max MUST be
        # set to the model's current Gaussian count (the 1M default would prune a
        # 3M model down to 1M). init_ckpt/extra_views are never overridable.
        extra = {
            "pose_opt": False,        # pseudo-poses are exact
            "coarse_factor": 0,       # refine at fine resolution only
            "strategy": "mcmc",
            "refine_stop_iter": 1,    # no densification (conservative default)
        }
        extra.update(train_extra or {})
        extra["init_ckpt"] = str(ckpts[-1])
        extra["extra_views"] = str(extra_views_json)
        train_config = TrainingConfig(
            backend=training_backend.name,
            max_steps=steps_per_round,
            data_factor=data_factor,
            save_steps=[steps_per_round],
            extra=extra,
        )
        result = training_backend.train(
            data_dir=data_dir,
            output_dir=str(round_dir),
            config=train_config,
        )
        if not result.checkpoint_dir:
            raise RuntimeError(f"Round {r} training produced no checkpoint")
        current_ckpt_dir = result.checkpoint_dir
        _progress((base + 1.0 / rounds), f"Round {r}/{rounds} complete "
                  f"(PSNR {result.psnr:.2f})")

    return result
