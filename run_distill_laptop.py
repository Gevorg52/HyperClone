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
CAP_MAX = 3_000_000   # MUST match the base model's Gaussian count when densifying


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--views", type=int, default=80)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--aggressive", action="store_true",
                    help="v2: push detail into the PLY — gated LPIPS + MCMC "
                         "relocation @ cap + geo_reg. Writes to distill_v2/.")
    ap.add_argument("--v3", action="store_true",
                    help="v3: v2 + Gram/style loss + cap_max 3M->4M (grow into "
                         "holes) + stronger scale_reg. Writes to distill_v3/.")
    # WP-R1: Observed-preservation gate levers (default OFF)
    ap.add_argument("--extra_unobserved_only",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="WP-R1: Restrict extra_views loss to unobserved pixels")
    ap.add_argument("--extra_anchor_observed", type=float, default=0.0,
                    help="WP-R1: Blend extra_views target with init_ckpt render on observed pixels")
    ap.add_argument("--extra_loss_weight", type=float, default=1.0,
                    help="WP-R1: Scalar weight multiplier for extra_views steps")
    ap.add_argument("--extra_obs_tau", type=float, default=0.95,
                    help="WP-R1: Alpha threshold tau_obs for observed pixels")
    ap.add_argument("--extra_obs_gamma", type=float, default=1.0,
                    help="WP-R1: Exponent gamma for unobserved weight map")
    args = ap.parse_args()
    if args.v3:
        args.aggressive = True   # v3 builds on the v2 recipe

    # v2 "push the limits" defaults (only applied if user left the conservative
    # defaults in place, so explicit flags still win).
    out_dir = DISTILL_DIR
    train_extra = None
    if args.aggressive:
        out_dir = TRAIN_DIR / "distill_v2"
        if args.rounds == 2:
            args.rounds = 3
        if args.views == 80:
            args.views = 120
        if args.steps == 4000:
            args.steps = 8000
        train_extra = {
            "cap_max": CAP_MAX,                     # never prune the 3M model down
            "refine_stop_iter": int(0.8 * args.steps),  # MCMC relocates ~80% of round
            "lpips_lambda": 0.2,                    # transfer high-freq detail
            "lpips_patch": 512,
            "lpips_start": 200,                     # resumed model -> start early
            "lpips_gated": True,                    # protect the well-resolved subject
            "geo_reg": True,                        # keep relocated splats on surfaces
            "geo_reg_start": 200,
            "opacity_reg": 0.0,                     # was 0.01 pre-2026-10 (WP-O3/O5)
            "scale_reg": 0.01,
        }
    if args.v3:
        out_dir = TRAIN_DIR / "distill_v3"
        train_extra["gram_lambda"] = 0.5           # Difix3D+ texture-baking term
        train_extra["cap_max"] = 4_000_000         # grow new Gaussians into holes
        train_extra["scale_reg"] = 0.02            # stronger anti-giant-blob for growth

    # Forward WP-R1 observed-preservation levers when active
    if (args.extra_unobserved_only or args.extra_anchor_observed > 0.0
            or args.extra_loss_weight != 1.0):
        if train_extra is None:
            train_extra = {}
        train_extra["extra_unobserved_only"] = bool(args.extra_unobserved_only)
        train_extra["extra_anchor_observed"] = float(args.extra_anchor_observed)
        train_extra["extra_loss_weight"] = float(args.extra_loss_weight)
        train_extra["extra_obs_tau"] = float(args.extra_obs_tau)
        train_extra["extra_obs_gamma"] = float(args.extra_obs_gamma)

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
    print(f"mode={'AGGRESSIVE v2' if args.aggressive else 'conservative v1'}  "
          f"out={out_dir.name}")
    print(f"distill: factor={args.factor} rounds={args.rounds} "
          f"views/round={args.views} steps/round={args.steps}")
    if train_extra:
        print(f"train_extra = {train_extra}", flush=True)

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
        output_dir=str(out_dir),
        rounds=args.rounds,
        views_per_round=args.views,
        steps_per_round=args.steps,
        data_factor=args.factor,
        enhancer_extra={"gated": True},
        train_extra=train_extra,
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
