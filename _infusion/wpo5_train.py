"""WP-O5 training driver: 3-arm free-viewpoint sweep on the 8K crop set.

All three arms share the WP-O3 ARM-2 recipe that is known-good on 360 walkthroughs
(`--opacity_reg 0` is load-bearing — see DEVELOPER_NOTES section 6) and the same dataset,
holdout list, seed, cap and step budget. One lever moves per arm:

  A  baseline-360   the WP-O3 ARM 2 recipe at 30 k steps
  B  A + --pose_opt          SfM pose jitter is a candidate cause of free-view mush; the 12
                             crops of a pano share an optical centre by construction, so this
                             hands each crop independent SE(3) freedom (WP-O3 hand-off item 1)
  C  A + --opacity_reg 0.001 10x BELOW the value that destroyed WP-O3 ARM 1. Tests whether a
                             little opacity pressure removes floater fog without re-triggering
                             the collapse. Checkpoint forensics run on every arm regardless.
  D  A + --erank_lambda 0.01 --erank_start 7000
                             ADDED MID-WP on measurement, not on the original plan
                             (_infusion/wpo5_needle_probe.py on the splat the user rejected):
                             deleting all 1 000 308 sub-0.05-opacity Gaussians changes its
                             render at NO viewpoint, so fog is ruled out; the off-path renders
                             are full of SPIKES; and the checkpoint is 52.4 % needles
                             (erank < 2), 27.8 % severe, median aspect 6.9:1. Deleting the
                             needles instead is catastrophic even ON-path, so they are
                             load-bearing and cannot be pruned away — which means the only
                             admissible intervention is at TRAINING time. WP-Q2 validated this
                             loss at these settings (89.6 % -> 16.6 % needles at equal
                             PSNR/LPIPS). Whether fewer needles actually improves free view is
                             the open question this arm exists to answer.

30 k steps (not WP-O3's 200 k) because WP-O3's own step ablation showed the last 170 k steps
bought only -5.5 % LPIPS / +0.29 dB: 3 arms x 30 k buys more information than 1 arm x 200 k.

  python _infusion/wpo5_train.py --data_dir <packaged> --work _dataset/360/train \
      --out _infusion/wpo5/train_runs.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE = [
    "--data_factor", "1",          # crops are 1920^2 native; factor 2 would throw away the 8K
    "--strategy", "mcmc",
    "--cap_max", "3000000",
    "--antialiased",
    "--opacity_reg", "0",          # THE WP-O3 FIX
    "--scale_reg", "0.01",
    "--ssim_lambda", "0.2",
    "--sh_degree", "3",
    "--loss_mask",
    "--coarse_factor", "0",
]

ARMS = {
    "A": [],
    "B": ["--pose_opt"],
    "C": ["--opacity_reg", "0.001"],      # overrides BASE's 0 (argparse: last wins)
    "D": ["--erank_lambda", "0.01", "--erank_start", "7000"],
}


def ckpt_health(ckpt: Path) -> dict:
    sys.path.insert(0, str(PROJECT_ROOT))
    import torch
    from tools.prune_ply import splat_stats

    payload = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    raw = payload["splats"]
    n = raw["means"].shape[0]
    out = splat_stats({"means": raw["means"].float(), "scales": raw["scales"].float(),
                       "quats": raw["quats"].float(),
                       "opacities": raw["opacities"].float().reshape(n),
                       "sh0": raw["sh0"].float().reshape(n, 1, 3),
                       "shN": raw["shN"].float().reshape(n, -1, 3)})
    # --pose_opt only: how far did BA-fixed poses actually move? The eval renders held-out
    # and off-path views at the ORIGINAL SfM poses, so a large drift would silently penalise
    # this arm; recording the magnitude is what makes the A/B readable.
    if "pose_embeds" in payload:
        # CameraOptModule layout (gsplat_trainer.py:673-688): [0:3] translation,
        # [3:9] 6D-rotation OFFSET added to the identity [1,0,0, 0,1,0].
        e = payload["pose_embeds"].float()
        trans, rot6 = e[:, :3], e[:, 3:]
        out["pose_opt"] = {
            "num_embeds": int(e.shape[0]),
            "rot6_offset_absmax": round(float(rot6.abs().max()), 8),
            "rot6_offset_absmean": round(float(rot6.abs().mean()), 8),
            "trans_norm_mean": round(float(trans.norm(dim=1).mean()), 8),
            "trans_norm_p99": round(float(trans.norm(dim=1).quantile(0.99)), 8),
            "trans_norm_max": round(float(trans.norm(dim=1).max()), 8),
            "note": "translation is a CAMERA-frame offset in the trainer's normalised world "
                    "units (camera cloud radius ~1); compare against the median inter-pano "
                    "step. The 6D-rotation numbers are raw parameter offsets, not angles."}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python _infusion/wpo5_train.py")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--work", default="_dataset/360/train")
    p.add_argument("--out", required=True)
    p.add_argument("--prefix", default="wpo5_run")
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", default="A,B,C,D")
    args = p.parse_args(argv)

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    holdout = str(Path(args.data_dir) / "holdout_test.txt")
    runs = []
    for tag in [a.strip() for a in args.arms.split(",") if a.strip()]:
        result_dir = work / f"{args.prefix}_{tag}"
        cmd = [sys.executable, str(PROJECT_ROOT / "core" / "training" / "gsplat_trainer.py"),
               "--data_dir", args.data_dir,
               "--result_dir", str(result_dir),
               "--holdout_list", holdout,
               "--max_steps", str(args.max_steps),
               # the mid-run save is the opacity-decay probe: WP-O3's collapse at
               # opacity_reg 0.01 was already visible by step ~1 000, so at arm C's 0.001 a
               # 10 k checkpoint says whether the model is on the same trajectory or not.
               "--save_steps", str(args.max_steps // 3), str(args.max_steps),
               "--seed", str(args.seed)] + BASE + ARMS[tag]
        log = work / f"{args.prefix}_{tag}.log"
        print(f"\n+ [arm {tag}] {' '.join(cmd)}", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf8", errors="replace") as fh:
            rc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        wall = time.time() - t0
        ckpt = result_dir / "ckpts" / f"ckpt_{args.max_steps - 1}_rank0.pt"
        ply = result_dir / "ply" / f"point_cloud_{args.max_steps - 1}.ply"
        stats = result_dir / "stats" / "results.json"
        row = {"arm": tag, "returncode": rc, "wall_clock_s": round(wall, 2),
               "argv": cmd, "log": str(log), "result_dir": str(result_dir),
               "ckpt": str(ckpt) if ckpt.is_file() else None,
               "ply": str(ply) if ply.is_file() else None,
               "ply_bytes": ply.stat().st_size if ply.is_file() else None,
               "trainer_results": json.loads(stats.read_text()) if stats.is_file() else None}
        mid = result_dir / "ckpts" / f"ckpt_{args.max_steps // 3 - 1}_rank0.pt"
        if mid.is_file():
            row["ckpt_health_mid"] = ckpt_health(mid)
            row["ckpt_mid"] = str(mid)
        if ckpt.is_file():
            row["ckpt_health"] = ckpt_health(ckpt)
            h = row["ckpt_health"]
            print(f"  [arm {tag}] rc={rc} {wall:.0f}s · {h['num_gaussians']} gaussians · "
                  f"mean opacity {h['opacity_mean']:.4f} · {h['opacity_frac_below']['0.05']*100:.1f}% "
                  f"< 0.05 · |mean| p99 {h['mean_norm_p99']:.2f}", flush=True)
        else:
            print(f"  [arm {tag}] rc={rc} {wall:.0f}s · NO CHECKPOINT", flush=True)
        runs.append(row)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"wp": "WP-O5", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "argv": [sys.executable, __file__] + list(sys.argv[1:]),
             "data_dir": args.data_dir, "holdout_list": holdout,
             "max_steps": args.max_steps, "seed": args.seed,
             "base_flags": BASE, "arm_flags": ARMS, "runs": runs}, indent=1), encoding="utf8")
    print(f"\nruns -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
