"""Full Laptop training run with the current (June 2026) quality stack.

Stage 9 (base train): modernized gsplat trainer at FULL 4K
  - MCMC densification, cap_max 3M (validated near-optimal for single-obj+bg)
  - Mip-Splatting anti-aliasing, SE(3) pose_opt
  - coarse-to-fine: densify at factor 4, fine phase at factor 1 (full 3840x2160)
  - gated LPIPS (lambda 0.2) — perceptual sharpness, safe on the well-resolved subject

Run:  conda activate base && python run_train_laptop.py
Outputs base checkpoint + PLY under _dataset/Laptop/train/.  The gated-Difix
distill-back loop is launched separately (run_distill_laptop.py) once the base
model exists, so the long base run can proceed independently.
"""
import argparse
import time
from pathlib import Path

from core.training import TrainingConfig
from core.training.gsplat_backend import GsplatBackend

EXPORT_DIR = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\export"
TRAIN_DIR = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\train"


def banner(msg: str):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bilateral", action="store_true",
                    help="enable per-view bilateral grid (writes to train_bil/)")
    ap.add_argument("--wd", action="store_true",
                    help="use WD-R perceptual loss instead of gated LPIPS")
    ap.add_argument("--wd_lambda", type=float, default=0.025,
                    help="WD-R weight (calibrated ~0.025 for this stack)")
    ap.add_argument("--wd_gated", action=argparse.BooleanOptionalAction, default=True,
                    help="gate WD by GT texture (train_wdg/); --no-wd_gated -> train_wd/")
    args = ap.parse_args()

    t0 = time.time()
    backend = GsplatBackend()
    if args.bilateral:
        suffix = "_bil"
    elif args.wd:
        suffix = "_wdg" if args.wd_gated else "_wd"
    else:
        suffix = ""
    out_dir = TRAIN_DIR + suffix

    config = TrainingConfig(
        backend="gsplat",
        max_steps=30_000,
        data_factor=1,                 # fine phase at full 4K
        test_every=8,
        save_steps=[7_000, 15_000, 30_000],
        extra={
            "strategy": "mcmc",
            "cap_max": 3_000_000,      # validated near-optimal; 8M diverges/OOMs
            "opacity_reg": 0.01,
            "scale_reg": 0.01,
            "antialiased": True,       # Mip-Splatting
            "pose_opt": True,
            "pose_opt_lr": 1e-5,
            "coarse_factor": 4,        # densify at 960x540
            "coarse_until": 15_000,    # switch to full 4K for the fine phase
            "lpips_lambda": 0.0 if args.wd else 0.2,  # WD run turns LPIPS off
            "lpips_patch": 512,
            "lpips_start": 3_000,      # also the WD warm-up step
            "lpips_gated": True,
            "wd_lambda": args.wd_lambda if args.wd else 0.0,
            "wd_gated": args.wd_gated,
            "bilateral_grid": args.bilateral,
        },
    )

    banner("STAGE 9a — prepare data (reuse train/data if present)")
    t = time.time()
    existing = Path(TRAIN_DIR) / "data"
    if (existing / "sparse").exists() and (existing / "images_4").exists():
        data_dir = str(existing)   # reuse the already-prepared 4K data
        print(f"reusing existing data_dir = {data_dir}")
    else:
        data_dir = backend.prepare_data(
            export_dir=EXPORT_DIR, mask_dir=None, output_dir=out_dir, config=config,
        )
    print(f"data_dir = {data_dir}  [{time.time() - t:.0f}s]", flush=True)

    banner("STAGE 9b — train (MCMC 3M, AA, pose_opt, C2F 4->1, gated LPIPS 0.2, 30k)")
    t = time.time()

    def prog(step, total, loss, psnr):
        if step % 500 == 0 or step >= total:
            el = time.time() - t
            print(f"  step {step:>6}/{total}  loss {loss:.4f}  "
                  f"psnr {psnr:.2f}  [{el:.0f}s, {step / max(el, 1):.1f} it/s]",
                  flush=True)

    result = backend.train(
        data_dir=data_dir, output_dir=out_dir, config=config, progress_cb=prog,
    )
    print(f"[train] {time.time() - t:.0f}s", flush=True)

    banner("BASE TRAIN COMPLETE")
    print(f"checkpoint = {result.checkpoint_dir}")
    print(f"ply        = {result.ply_path}")
    print(f"test PSNR  = {result.psnr:.2f} dB")
    print(f"test SSIM  = {result.ssim:.4f}")
    print(f"steps      = {result.steps_completed}")
    print(f"final loss = {result.final_loss:.4f}")
    print(f"TOTAL      = {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
