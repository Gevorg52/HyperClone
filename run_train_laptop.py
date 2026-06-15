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
import time
from pathlib import Path

from core.training import TrainingConfig
from core.training.gsplat_backend import GsplatBackend

EXPORT_DIR = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\export"
TRAIN_DIR = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\train"


def banner(msg: str):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70, flush=True)


def main():
    t0 = time.time()
    backend = GsplatBackend()

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
            "lpips_lambda": 0.2,       # gated perceptual loss (production default)
            "lpips_patch": 512,
            "lpips_start": 3_000,
            "lpips_gated": True,
        },
    )

    banner("STAGE 9a — prepare data (factor 1 + coarse factor 4)")
    t = time.time()
    data_dir = backend.prepare_data(
        export_dir=EXPORT_DIR, mask_dir=None, output_dir=TRAIN_DIR, config=config,
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
        data_dir=data_dir, output_dir=TRAIN_DIR, config=config, progress_cb=prog,
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
