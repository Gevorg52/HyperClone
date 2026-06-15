"""Gated-Difix distill-back loop on the trained Laptop model.

Run AFTER run_train_laptop.py finishes (needs the base checkpoint).  Each round:
render pseudo-views from the current model -> GATED Difix (default-on) -> resume
training with the enhanced views appended.  Produces a distilled checkpoint + PLY.

Usage:  python run_distill_laptop.py [--factor N] [--rounds R] [--views V] [--steps S]
  --factor : working resolution for pseudo-views + resume training (1=4K, 2=1080p).
             Difix at 4K is VRAM-heavy; verify feasibility first (defaults to 2).
"""
import argparse
import time
from pathlib import Path

from core.training import get_backend as get_training_backend
from core.enhancement import get_backend as get_enhancer
from core.enhancement.distill import run_distill_loop

TRAIN_DIR = Path(r"F:\HYPERclone\apex-sfm\_dataset\Laptop\train")
DATA_DIR = TRAIN_DIR / "data"
CKPT_DIR = TRAIN_DIR / "output" / "ckpts"
DISTILL_DIR = TRAIN_DIR / "distill"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--views", type=int, default=80)
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()

    ckpts = sorted(CKPT_DIR.glob("ckpt_*_rank0.pt"))
    if not ckpts:
        raise SystemExit(f"No base checkpoint in {CKPT_DIR} — run base train first.")

    # The loader silently falls back to full-res images/ when images_{factor} is
    # missing, which renders pseudo-views at 4K -> Difix OOM. Ensure it exists.
    if args.factor > 1:
        ds = DATA_DIR / f"images_{args.factor}"
        if not ds.is_dir():
            from core.training.gsplat_backend import _downsample_images
            print(f"creating {ds.name} (downsample x{args.factor})...", flush=True)
            _downsample_images(str(DATA_DIR / "images"), str(ds), args.factor)
    print(f"base checkpoint dir = {CKPT_DIR}")
    print(f"latest ckpt         = {ckpts[-1].name}")
    print(f"distill: factor={args.factor} rounds={args.rounds} "
          f"views/round={args.views} steps/round={args.steps}", flush=True)

    tb = get_training_backend("gsplat")
    enh = get_enhancer("difix")           # gated on by default
    ok, msg = enh.validate_environment()
    if not ok:
        raise SystemExit(f"difix not ready: {msg}")

    t0 = time.time()

    def prog(frac, desc):
        print(f"  [{frac * 100:5.1f}%] {desc}  ({(time.time() - t0) / 60:.1f} min)",
              flush=True)

    result = run_distill_loop(
        training_backend=tb,
        enhancer=enh,
        data_dir=str(DATA_DIR),
        checkpoint_dir=str(CKPT_DIR),
        output_dir=str(DISTILL_DIR),
        rounds=args.rounds,
        views_per_round=args.views,
        steps_per_round=args.steps,
        data_factor=args.factor,
        enhancer_extra={"gated": True},
        progress_cb=prog,
    )

    print("\n" + "=" * 70)
    print("DISTILL COMPLETE")
    print(f"checkpoint = {result.checkpoint_dir}")
    print(f"ply        = {result.ply_path}")
    print(f"test PSNR  = {result.psnr:.2f} dB")
    print(f"test SSIM  = {result.ssim:.4f}")
    print(f"TOTAL      = {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
