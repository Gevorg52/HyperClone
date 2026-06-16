"""General A/B training driver: train one dataset with a chosen perceptual loss.

Used to validate WD-R vs gated-LPIPS across scenes. Reuses prepared data if present.

Usage: python run_train_ab.py --dataset GPU --loss {lpips|wd|none} [--factor 2]
                              [--steps 30000] [--wd_lambda 0.025] [--cap 3000000]
Writes to _dataset/<dataset>/train_<loss>/.
"""
import argparse
import time
from pathlib import Path

from core.training import TrainingConfig
from core.training.gsplat_backend import GsplatBackend

ROOT = Path(r"F:\HYPERclone\apex-sfm\_dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--loss", choices=["lpips", "wd", "none"], required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--wd_lambda", type=float, default=0.025)
    ap.add_argument("--cap", type=int, default=3_000_000)
    args = ap.parse_args()

    export = ROOT / args.dataset / "export"
    out_dir = ROOT / args.dataset / f"train_{args.loss}"
    backend = GsplatBackend()

    extra = {
        "strategy": "mcmc", "cap_max": args.cap,
        "opacity_reg": 0.01, "scale_reg": 0.01,
        "antialiased": True, "pose_opt": True, "pose_opt_lr": 1e-5,
        "coarse_factor": 4, "coarse_until": args.steps // 2,
        "lpips_lambda": 0.2 if args.loss == "lpips" else 0.0,
        "lpips_patch": 512, "lpips_start": max(3_000, args.steps // 10),
        "lpips_gated": True,
        "wd_lambda": args.wd_lambda if args.loss == "wd" else 0.0,
    }
    config = TrainingConfig(backend="gsplat", max_steps=args.steps,
                            data_factor=args.factor, test_every=8,
                            save_steps=[args.steps], extra=extra)

    print(f"=== A/B: dataset={args.dataset} loss={args.loss} factor={args.factor} "
          f"steps={args.steps} -> {out_dir.name} ===", flush=True)
    t0 = time.time()
    # reuse prepared data if present
    data = out_dir.parent / "train_lpips" / "data"
    if not (data / "sparse").exists():
        data = backend.prepare_data(export_dir=str(export), mask_dir=None,
                                    output_dir=str(out_dir), config=config)
    else:
        data = str(data)
        print(f"reusing data {data}")

    def prog(s, t, l, p):
        if s % 1000 == 0 or s >= t:
            print(f"  step {s}/{t} loss {l:.4f} psnr {p:.2f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    r = backend.train(data_dir=data, output_dir=str(out_dir), config=config,
                      progress_cb=prog)
    print(f"\nDONE {args.dataset}/{args.loss}: PSNR {r.psnr:.2f} SSIM {r.ssim:.4f} "
          f"ckpt={r.checkpoint_dir} ply={r.ply_path} [{(time.time()-t0)/60:.1f} min]",
          flush=True)


if __name__ == "__main__":
    main()
