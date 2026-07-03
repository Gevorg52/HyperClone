"""gsplat training backend.

Drives our bundled ``gsplat_trainer.py`` as a subprocess so that:
  - VRAM is fully freed when training finishes (separate process)
  - Progress is streamed via stdout line parsing
  - Stop is implemented via process termination + latest checkpoint recovery

The trainer uses gsplat's rasterization API + pycolmap for data loading
(both already installed), so no extra dependencies are needed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from core.training import (
    ProgressCallback,
    TrainingBackend,
    TrainingConfig,
    TrainingInterrupted,
    TrainingResult,
)

# Path to our bundled trainer script
_TRAINER_SCRIPT = str(Path(__file__).resolve().parent / "gsplat_trainer.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_masks_to_images(
    src_image_dir: str,
    mask_dir: str,
    dst_image_dir: str,
    progress_cb: Optional[ProgressCallback] = None,
) -> int:
    """Pre-multiply images with masks (black background).

    For each image, composites it with its mask:
      - Masked-out pixels (mask == 0) become black.
      - Foreground pixels are preserved.

    Saves results as JPEG to *dst_image_dir*. Returns count of images processed.
    """
    src_dir = Path(src_image_dir)
    mask_path = Path(mask_dir)
    dst_dir = Path(dst_image_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in src_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )

    def _process_one(idx_path: tuple[int, Path]) -> None:
        idx, img_path = idx_path
        # Find matching mask (same stem, .png)
        mask_file = mask_path / (img_path.stem + ".png")
        img = Image.open(img_path).convert("RGB")

        if mask_file.is_file():
            mask = Image.open(mask_file).convert("L")
            if mask.size != img.size:
                mask = mask.resize(img.size, Image.BILINEAR)
            # Apply mask: foreground kept, background → black
            img_arr = np.array(img, dtype=np.float32)
            mask_arr = np.array(mask, dtype=np.float32) / 255.0
            img_arr *= mask_arr[:, :, np.newaxis]
            img = Image.fromarray(img_arr.astype(np.uint8))

        # Save as JPEG (gsplat reads any common format)
        out_path = dst_dir / (img_path.stem + ".jpg")
        img.save(out_path, quality=95)

    n = len(image_files)
    workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(_process_one, enumerate(image_files))):
            if progress_cb and (i + 1) % 5 == 0:
                progress_cb(i + 1, n, 0.0, 0.0)

    return n


def _downsample_images(src_dir: str, dst_dir: str, factor: int) -> int:
    """Create downsampled copies of images using bicubic interpolation.

    Returns number of images processed.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in src.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )

    def _resize_one(path: Path) -> None:
        img = Image.open(path).convert("RGB")
        new_w = round(img.width / factor)
        new_h = round(img.height / factor)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        out = dst / (path.stem + ".png")
        img.save(out)

    workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_resize_one, image_files))

    return len(image_files)


def _export_masks(src_images: str, mask_dir: str, dst_dir: str, factor: int = 1) -> int:
    """Write per-image object masks (stem-matched to the images) into dst_dir as
    grayscale PNG, optionally downsampled by `factor` with NEAREST (preserves the
    binary edge). Lets the trainer apply object-mask alpha supervision. Returns count."""
    src = Path(src_images)
    md = Path(mask_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    image_files = sorted(
        p for p in src.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )

    def _one(path: Path) -> int:
        mf = md / (path.stem + ".png")
        if not mf.is_file():
            return 0
        m = Image.open(mf).convert("L")
        if factor > 1:
            m = m.resize((round(m.width / factor), round(m.height / factor)), Image.NEAREST)
        m.save(dst / (path.stem + ".png"))
        return 1

    workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(_one, image_files))


# ---------------------------------------------------------------------------
# Progress line parsing
# ---------------------------------------------------------------------------

# Our trainer prints lines like:
#   step=100 | loss=0.1234 | psnr=23.45 | #GS=12345 | mem=1234.5MB
# Eval lines:
#   [Eval] PSNR=25.12 SSIM=0.8901

_STEP_RE = re.compile(r"step[=:\s]+(\d+)", re.IGNORECASE)
_LOSS_RE = re.compile(r"loss[=:\s]+([\d.eE+-]+)", re.IGNORECASE)
_PSNR_RE = re.compile(r"psnr[=:\s]+([\d.eE+-]+)", re.IGNORECASE)
_SSIM_RE = re.compile(r"ssim[=:\s]+([\d.eE+-]+)", re.IGNORECASE)


def _parse_progress_line(line: str) -> Optional[dict]:
    """Try to extract training metrics from a single log line."""
    step_m = _STEP_RE.search(line)
    if not step_m:
        return None
    result = {"step": int(step_m.group(1))}

    loss_m = _LOSS_RE.search(line)
    if loss_m:
        result["loss"] = float(loss_m.group(1))

    psnr_m = _PSNR_RE.search(line)
    if psnr_m:
        result["psnr"] = float(psnr_m.group(1))

    ssim_m = _SSIM_RE.search(line)
    if ssim_m:
        result["ssim"] = float(ssim_m.group(1))

    return result


# ---------------------------------------------------------------------------
# GsplatBackend
# ---------------------------------------------------------------------------

class GsplatBackend(TrainingBackend):
    """3D Gaussian Splatting backend using gsplat."""

    name = "gsplat"

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()

    # -----------------------------------------------------------------------
    # Environment validation
    # -----------------------------------------------------------------------

    def validate_environment(self) -> tuple[bool, str]:
        """Check gsplat is importable and dependencies are available."""
        try:
            import gsplat
            version = getattr(gsplat, "__version__", "unknown")
        except ImportError:
            return False, (
                "gsplat is not installed.  Install with:\n"
                "  pip install gsplat"
            )

        # Verify core API works
        try:
            from gsplat.rendering import rasterization  # noqa: F401
            from gsplat.strategy import DefaultStrategy  # noqa: F401
        except ImportError as e:
            return False, f"gsplat {version} installed but core API import failed: {e}"

        # Verify our trainer script exists
        if not Path(_TRAINER_SCRIPT).is_file():
            return False, f"Bundled trainer not found at: {_TRAINER_SCRIPT}"

        return True, f"gsplat {version} ready.  Trainer: {_TRAINER_SCRIPT}"

    # -----------------------------------------------------------------------
    # Default config
    # -----------------------------------------------------------------------

    def get_default_config(self) -> dict:
        return {
            "max_steps": 30_000,
            "data_factor": 2,
            "test_every": 8,
            "save_steps": [7_000, 30_000],
            "sh_degree": 3,
            "ssim_lambda": 0.2,
            "random_bkgd": False,
            "normalize_world_space": True,
            # June 2026 quality stack
            "strategy": "mcmc",            # mcmc | default
            "cap_max": 1_000_000,          # MCMC Gaussian budget
            "opacity_reg": 0.01,
            "scale_reg": 0.01,
            "vis_reg": 0.0,                # visibility loss (kill empty-space floaters)
            "erank_lambda": 0.0,           # effective-rank anti-needle hinge (2406.11672; ~0.01)
            "erank_start": 7000,           # step to start erank hinge
            "erank_end": None,             # last step (incl.) for the hinge; None = no cutoff
            "alpha_loss_lambda": 0.0,      # object-mask alpha sup. (kills bg floaters; ~0.1)
            "alpha_obj_lambda": 0.0,       # push alpha->1 inside mask (off by default)
            "depth_lambda": 0.0,           # sparse SfM-point depth supervision (~0.05-0.2)
            "depth_start": 2000,           # step to start depth loss
            "mono_depth_lambda": 0.0,      # dense MoGe-2 prior depth sup. (WP-F1; ~0.05)
            "mono_depth_start": 2000,      # step to start mono-depth loss (fine phase)
            "mono_depth_use_weight": True, # scale by per-view align depth_weight
            # EDGS-style dense-matcher triangulation init (WP-F2). OFF = production
            # unchanged (ALIKED sparse init). ON augments the SfM point cloud with
            # RoMa dense-triangulated points on textureless regions before seeding.
            "dense_init": False,
            "dense_init_num_matched": 5,   # covisible pairs/image for dense matching
            "dense_init_weights": "outdoor",  # RoMa weights: outdoor | indoor
            "dense_init_max_matches": 15000,  # RoMa samples/pair
            "dense_init_min_certainty": 0.05,
            "dense_init_voxel_scale": 1.0,    # x median-NN spacing for dedup voxel
            "dense_init_low_density_only": False,  # CoMapGS selectivity (opt-in)
            "dense_init_num_workers": 0,   # DataLoader workers (0 = safe on Win/4K)
            "dense_init_seed": 0,          # torch seed for RoMa match sampling
            "antialiased": True,           # Mip-Splatting anti-aliasing
            "pose_opt": True,              # SE(3) pose refinement
            "pose_opt_lr": 1e-5,
            "coarse_factor": 4,            # densification-phase resolution
            "coarse_until": 15_000,        # step to switch to fine resolution
            # Edge-aware geometric reg (textureless/flat-surface "mush" fix)
            "geo_reg": False,              # off by default; opt-in per run
            "geo_reg_lambda": 0.05,
            "geo_reg_beta": 10.0,
            "geo_reg_start": 3000,
            # Perceptual (LPIPS) loss — perceived-sharpness lever
            "lpips_lambda": 0.0,           # off by default; opt-in (try ~0.1-0.2)
            "lpips_patch": 512,
            "lpips_start": 3000,
            "lpips_gated": True,           # gate toward under-resolved regions
            "gram_lambda": 0.0,            # VGG Gram/style loss (Difix3D+ texture)
            "wd_lambda": 0.0,              # Wasserstein-Distortion (WD-R) perceptual
            "wd_gated": True,             # gate WD by GT texture (no grain on flats)
            "bilateral_grid": False,       # per-view BilaRF color grid (exposure/WB)
        }

    # -----------------------------------------------------------------------
    # Data preparation
    # -----------------------------------------------------------------------

    def prepare_data(
        self,
        export_dir: str,
        mask_dir: Optional[str],
        output_dir: str,
        config: TrainingConfig,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> str:
        """Prepare a gsplat-compatible data directory.

        - Copies sparse reconstruction from export_dir
        - If masks exist, pre-multiplies images (black background)
        - Creates downsampled image directories for the configured data_factor
        """
        export = Path(export_dir)
        data_dir = Path(output_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Single source of truth for option defaults (same merge as train()).
        extra = {**self.get_default_config(), **config.extra}

        # --- Copy sparse reconstruction ---
        sparse_src = export / "sparse" / "0"
        if not sparse_src.is_dir():
            sparse_src = export / "sparse"
        sparse_dst = data_dir / "sparse" / "0"
        sparse_dst.mkdir(parents=True, exist_ok=True)

        for bin_file in sparse_src.glob("*.bin"):
            shutil.copy2(str(bin_file), str(sparse_dst / bin_file.name))

        # --- Prepare images ---
        src_images = export / "images"
        if not src_images.is_dir():
            raise FileNotFoundError(f"No images/ directory in {export_dir}")

        images_dir = data_dir / "images"

        if mask_dir and Path(mask_dir).is_dir():
            # Pre-multiply images with masks → black background
            _apply_masks_to_images(
                str(src_images), mask_dir, str(images_dir), progress_cb,
            )
        else:
            # Copy images directly
            images_dir.mkdir(parents=True, exist_ok=True)
            for img in src_images.iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    shutil.copy2(str(img), str(images_dir / img.name))

        # --- Create downsampled directories ---
        factors = {config.data_factor}
        coarse_factor = int(extra["coarse_factor"] or 0)
        if coarse_factor > config.data_factor:
            factors.add(coarse_factor)
        for factor in factors:
            if factor > 1:
                ds_dir = data_dir / f"images_{factor}"
                if not ds_dir.is_dir():
                    _downsample_images(str(images_dir), str(ds_dir), factor)

        # --- Export object masks for alpha supervision (full-res + per factor) ---
        if mask_dir and Path(mask_dir).is_dir():
            _export_masks(str(src_images), mask_dir, str(data_dir / "masks"), factor=1)
            for factor in factors:
                if factor > 1:
                    _export_masks(str(src_images), mask_dir,
                                  str(data_dir / f"masks_{factor}"), factor=factor)

        # --- EDGS-style dense triangulation init (WP-F2; opt-in, default OFF) ---
        # Augments data_dir/sparse/0 in place with RoMa dense-triangulated points on
        # textureless regions. Reads the full-res (premultiplied) images and the
        # reference model; all reference reads complete before the augmented model is
        # written back, so writing to the same dir is safe. Production is unchanged
        # unless config.extra["dense_init"] is True.
        if extra["dense_init"]:
            from core.dense_init import build_dense_init

            sparse_model = str(data_dir / "sparse" / "0")
            print("[prepare_data] dense_init ENABLED — augmenting SfM cloud with "
                  "RoMa dense triangulation (WP-F2)...", flush=True)
            stats = build_dense_init(
                reference_model=sparse_model,
                image_dir=str(images_dir),
                out_model_dir=sparse_model,
                num_matched=int(extra["dense_init_num_matched"]),
                matcher_weights=extra["dense_init_weights"],
                max_num_matches=int(extra["dense_init_max_matches"]),
                min_certainty=float(extra["dense_init_min_certainty"]),
                voxel_scale=float(extra["dense_init_voxel_scale"]),
                low_density_only=bool(extra["dense_init_low_density_only"]),
                num_workers=int(extra["dense_init_num_workers"]),
                seed=int(extra["dense_init_seed"]),
            )
            print(f"[prepare_data] dense_init: {stats['n_orig_points']} -> "
                  f"{stats['n_final_points']} points "
                  f"(+{stats['n_added_points']}, {stats['final_over_orig']:.2f}x); "
                  f"new-pt mean reproj {stats['new_points_mean_reproj_px']:.2f}px",
                  flush=True)

        return str(data_dir)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(
        self,
        data_dir: str,
        output_dir: str,
        config: TrainingConfig,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        """Run gsplat training as a subprocess.

        Streams progress by parsing stdout.  Supports early stop via
        ``request_stop()``.
        """
        self._stop_event.clear()

        # Build CLI args for our bundled trainer
        result_dir = str(Path(output_dir) / "output")
        defaults = self.get_default_config()
        extra = {**defaults, **config.extra}

        args = [
            sys.executable, _TRAINER_SCRIPT,
            "--data_dir", data_dir,
            "--result_dir", result_dir,
            "--data_factor", str(config.data_factor),
            "--max_steps", str(config.max_steps),
            "--test_every", str(config.test_every),
            "--sh_degree", str(extra["sh_degree"]),
            "--ssim_lambda", str(extra["ssim_lambda"]),
            "--strategy", str(extra["strategy"]),
            "--cap_max", str(extra["cap_max"]),
            "--opacity_reg", str(extra["opacity_reg"]),
            "--scale_reg", str(extra["scale_reg"]),
            "--vis_reg", str(extra.get("vis_reg", 0.0)),
            "--erank_lambda", str(extra.get("erank_lambda", 0.0)),
            "--erank_start", str(extra.get("erank_start", 7000)),
            "--alpha_loss_lambda", str(extra.get("alpha_loss_lambda", 0.0)),
            "--alpha_obj_lambda", str(extra.get("alpha_obj_lambda", 0.0)),
            "--depth_lambda", str(extra.get("depth_lambda", 0.0)),
            "--depth_start", str(extra.get("depth_start", 2000)),
            "--mono_depth_lambda", str(extra.get("mono_depth_lambda", 0.0)),
            "--mono_depth_start", str(extra.get("mono_depth_start", 2000)),
        ]
        # mono-depth per-view weighting (BooleanOptionalAction on the trainer)
        args.append("--mono_depth_use_weight"
                    if extra.get("mono_depth_use_weight", True)
                    else "--no-mono_depth_use_weight")

        # erank_end: optional cutoff step (None = flag omitted = no cutoff)
        if extra.get("erank_end") is not None:
            args.extend(["--erank_end", str(extra["erank_end"])])

        if extra["normalize_world_space"]:
            args.append("--normalize_world_space")
        else:
            args.append("--no-normalize_world_space")

        if extra["random_bkgd"]:
            args.append("--random_bkgd")

        args.append("--antialiased" if extra["antialiased"] else "--no-antialiased")

        if extra["pose_opt"]:
            args.append("--pose_opt")
            args.extend(["--pose_opt_lr", str(extra["pose_opt_lr"])])

        coarse_factor = int(extra.get("coarse_factor", 0) or 0)
        if coarse_factor > config.data_factor:
            args.extend([
                "--coarse_factor", str(coarse_factor),
                "--coarse_until", str(extra["coarse_until"]),
            ])

        # Edge-aware geometric regularization
        if extra.get("geo_reg"):
            args.append("--geo_reg")
            args.extend([
                "--geo_reg_lambda", str(extra["geo_reg_lambda"]),
                "--geo_reg_beta", str(extra["geo_reg_beta"]),
                "--geo_reg_start", str(extra["geo_reg_start"]),
            ])

        # Perceptual (LPIPS) loss
        if extra.get("lpips_lambda", 0):
            args.extend([
                "--lpips_lambda", str(extra["lpips_lambda"]),
                "--lpips_patch", str(extra["lpips_patch"]),
                "--lpips_start", str(extra["lpips_start"]),
            ])
            args.append("--lpips_gated" if extra.get("lpips_gated", True)
                        else "--no-lpips_gated")

        # Wasserstein-Distortion (WD-R) perceptual loss (shares lpips_patch/start)
        if extra.get("wd_lambda", 0):
            args.extend(["--wd_lambda", str(extra["wd_lambda"])])
            args.append("--wd_gated" if extra.get("wd_gated", True)
                        else "--no-wd_gated")
            if not extra.get("lpips_lambda", 0):
                args.extend(["--lpips_patch", str(extra.get("lpips_patch", 512)),
                             "--lpips_start", str(extra.get("lpips_start", 3000))])

        # Per-view bilateral grid (BilaRF) — exposure/WB absorption
        if extra.get("bilateral_grid"):
            args.append("--bilateral_grid")

        # Gram/style loss (texture baking; shares LPIPS crop/gate/start)
        if extra.get("gram_lambda", 0):
            args.extend(["--gram_lambda", str(extra["gram_lambda"])])
            # gram needs the perceptual crop path active even if LPIPS is off
            if not extra.get("lpips_lambda", 0):
                args.extend(["--lpips_patch", str(extra.get("lpips_patch", 512)),
                             "--lpips_start", str(extra.get("lpips_start", 3000))])
                args.append("--lpips_gated" if extra.get("lpips_gated", True)
                            else "--no-lpips_gated")

        # Distill / refine rounds
        if extra.get("init_ckpt"):
            args.extend(["--init_ckpt", str(extra["init_ckpt"])])
        if extra.get("extra_views"):
            args.extend(["--extra_views", str(extra["extra_views"])])
        if extra.get("refine_stop_iter"):
            args.extend(["--refine_stop_iter", str(extra["refine_stop_iter"])])

        # save_steps
        save_steps = config.save_steps or [7_000, config.max_steps]
        for s in save_steps:
            args.extend(["--save_steps", str(s)])

        # Launch subprocess
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

        self._process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        # Stream output and parse progress
        last_step = 0
        last_loss = 0.0
        last_psnr = 0.0
        last_ssim = 0.0
        log_lines: list[str] = []

        try:
            for line in self._process.stdout:
                line = line.rstrip()
                log_lines.append(line)

                # Check stop
                if self._stop_event.is_set():
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                    raise TrainingInterrupted("Training stopped by user")

                # Parse progress
                parsed = _parse_progress_line(line)
                if parsed:
                    last_step = parsed["step"]
                    last_loss = parsed.get("loss", last_loss)
                    last_psnr = parsed.get("psnr", last_psnr)
                    last_ssim = parsed.get("ssim", last_ssim)

                    if progress_cb and last_step % 100 == 0:
                        progress_cb(last_step, config.max_steps, last_loss, last_psnr)

            self._process.wait()

        except TrainingInterrupted:
            raise
        except Exception:
            if self._process and self._process.poll() is None:
                self._process.kill()
            raise
        finally:
            self._process = None
            # Save full log
            log_path = Path(output_dir) / "training_log.txt"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(log_lines), encoding="utf-8")

        # Collect results
        result = TrainingResult(
            steps_completed=last_step,
            final_loss=last_loss,
            psnr=last_psnr,
            ssim=last_ssim,
        )

        # Find checkpoint and PLY
        ckpt_dir = Path(result_dir) / "ckpts"
        if ckpt_dir.is_dir():
            ckpts = sorted(ckpt_dir.glob("ckpt_*_rank0.pt"))
            if ckpts:
                result.checkpoint_dir = str(ckpt_dir)

        ply_dir = Path(result_dir) / "ply"
        if ply_dir.is_dir():
            plys = sorted(ply_dir.glob("point_cloud_*.ply"))
            if plys:
                result.ply_path = str(plys[-1])

        # Read eval metrics from stats JSON
        stats_dir = Path(result_dir) / "stats"
        if stats_dir.is_dir():
            for stats_file in sorted(stats_dir.glob("*.json")):
                try:
                    stats_data = json.loads(stats_file.read_text(encoding="utf-8"))
                    if "psnr" in stats_data:
                        result.psnr = float(stats_data["psnr"])
                    if "ssim" in stats_data:
                        result.ssim = float(stats_data["ssim"])
                except (json.JSONDecodeError, KeyError):
                    pass

        return result

    # -----------------------------------------------------------------------
    # Stop
    # -----------------------------------------------------------------------

    def request_stop(self):
        """Signal the training subprocess to stop."""
        self._stop_event.set()
        if self._process and self._process.poll() is None:
            self._process.terminate()

    # -----------------------------------------------------------------------
    # Render views
    # -----------------------------------------------------------------------

    def render_views(
        self,
        checkpoint_dir: str,
        data_dir: str,
        output_dir: str,
        max_images: int = 0,
        data_factor: int = 0,
        antialiased: bool = True,
    ) -> list[str]:
        """Render test-set views from a trained checkpoint.

        Loads the checkpoint in-process and renders using gsplat rasterization.
        Returns paths to saved PNG images.  *data_factor* should match the
        factor used in training; 0 picks the smallest images_N dir available.
        """
        import torch
        from gsplat.rendering import rasterization as gsplat_raster

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Find latest checkpoint
        ckpt_dir = Path(checkpoint_dir)
        ckpts = sorted(ckpt_dir.glob("ckpt_*_rank0.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        # Load checkpoint
        ckpt = torch.load(str(ckpts[-1]), map_location=device, weights_only=True)
        splats = {k: v.to(device) for k, v in ckpt["splats"].items()}

        # Resolve render factor: prefer explicit, else smallest available dir
        if data_factor <= 0:
            available = sorted(
                int(p.name.split("_")[1])
                for p in Path(data_dir).glob("images_*")
                if p.is_dir() and p.name.split("_")[1].isdigit()
            )
            data_factor = available[0] if available else 1

        # Load scene data
        from core.training.gsplat_trainer import load_colmap_data, load_image
        data = load_colmap_data(
            data_dir, factor=data_factor, test_every=8, normalize=True,
        )

        render_dir = Path(output_dir)
        render_dir.mkdir(parents=True, exist_ok=True)

        test_indices = data["test_indices"]
        if max_images > 0:
            test_indices = test_indices[:max_images]

        rendered_paths = []
        sh_degree = int((-1 + (1 + 4 * (1 + splats["shN"].shape[1])) ** 0.5) / 2)

        with torch.no_grad():
            for idx in test_indices:
                batch = load_image(data, idx)
                K = batch["K"].to(device)
                c2w = batch["c2w"].to(device)
                H, W = batch["image"].shape[:2]
                w2c = torch.linalg.inv(c2w)

                bkgd = torch.zeros(3, device=device)
                renders, _, _ = gsplat_raster(
                    means=splats["means"],
                    quats=splats["quats"] / (splats["quats"].norm(dim=-1, keepdim=True) + 1e-12),
                    scales=torch.exp(splats["scales"]),
                    opacities=torch.sigmoid(splats["opacities"]),
                    colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
                    viewmats=w2c[None],
                    Ks=K[None],
                    width=W,
                    height=H,
                    sh_degree=sh_degree,
                    near_plane=0.01,
                    far_plane=1e10,
                    backgrounds=bkgd[None],
                    rasterize_mode="antialiased" if antialiased else "classic",
                    packed=False,
                )

                img = (renders[0].clamp(0, 1) * 255).byte().cpu().numpy()
                out_path = render_dir / f"render_{data['image_names'][idx]}.png"
                Image.fromarray(img).save(str(out_path))
                rendered_paths.append(str(out_path))

        return rendered_paths
