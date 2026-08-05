"""WP-O3 diagnostic: why the baseline arm diverged (opacity -> 0, means -> 1e6).

Hypothesis. `--rgb_object_masked` composites BOTH render and GT over black inside the
mask and then takes a plain `mean()` over the WHOLE frame. The WP-O2 crop masks keep only
~33% of each crop (exclusive nearest-camera tiling AND-ed with WP-O1's operator/nadir/seam
exclusions), so the photometric term is ~3x smaller than it would be on an unmasked frame
while `opacity_reg`/`scale_reg` are unchanged. The regularizers therefore dominate,
opacities collapse, and gsplat's MCMC noise injection is *maximal* at low opacity
(`op_sigmoid(1 - opacity)`, k=100, x0=0.995) -> positions random-walk away -> runaway.

Four short arms (default 5000 steps each) isolate it:
  A masks_on_prod_reg   the baseline arm's config          (expected: collapsing)
  B masks_off_prod_reg  same, --no-rgb_object_masked       (isolates the mask)
  C masks_on_scaled_reg masks on, regs x mean valid frac   (isolates the dilution)
  D masks_on_no_reg     masks on, opacity_reg=scale_reg=0  (upper bound on the effect)

Records mean opacity, dead fraction, position spread and train PSNR per arm.

Usage:
  python _infusion/wpo3_diag_collapse.py --data_dir <packaged> --work <scratch dir> \
      --out _infusion/wpo3/diag_collapse.json [--steps 5000]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINER = PROJECT_ROOT / "core" / "training" / "gsplat_trainer.py"

# mean valid fraction of the WP-O2 crop masks (sfm_report.json -> valid_frac_mean)
VALID_FRAC = 0.331

ARMS = {
    "A_masks_on_prod_reg":   ["--rgb_object_masked", "--opacity_reg", "0.01",
                              "--scale_reg", "0.01"],
    "B_masks_off_prod_reg":  ["--no-rgb_object_masked", "--opacity_reg", "0.01",
                              "--scale_reg", "0.01"],
    "C_masks_on_scaled_reg": ["--rgb_object_masked",
                              "--opacity_reg", f"{0.01 * VALID_FRAC:.5f}",
                              "--scale_reg", f"{0.01 * VALID_FRAC:.5f}"],
    "D_masks_on_no_reg":     ["--rgb_object_masked", "--opacity_reg", "0",
                              "--scale_reg", "0"],
    # Added after A-D: the fix candidate. --loss_mask averages L1 and the SSIM map over
    # the VALID pixels instead of compositing over black, so the mask costs neither loss
    # magnitude nor boundary edges. NOTE its logged PSNR is already the MASKED one, i.e.
    # ~4.8 dB lower than the black-composited PSNR arms A/C/D print.
    "E_loss_mask_no_reg":    ["--loss_mask", "--opacity_reg", "0", "--scale_reg", "0"],
    "F_loss_mask_prod_reg":  ["--loss_mask", "--opacity_reg", "0.01", "--scale_reg", "0.01"],
    # A re-run, alone on the GPU (A and C originally overlapped an eval job).
    "A2_masks_on_prod_reg":  ["--rgb_object_masked", "--opacity_reg", "0.01",
                              "--scale_reg", "0.01"],
    # Which of the two regularizers is the killer?
    "G_loss_mask_scale_only": ["--loss_mask", "--opacity_reg", "0", "--scale_reg", "0.01"],
    "H_loss_mask_opacity_only": ["--loss_mask", "--opacity_reg", "0.01", "--scale_reg", "0"],
}


def ckpt_stats(path: Path) -> dict:
    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    sp = ck["splats"]
    m = sp["means"].float()
    o = torch.sigmoid(sp["opacities"].float())
    s = torch.exp(sp["scales"].float()).max(dim=1).values
    n = m.norm(dim=1)
    return {
        "step": int(ck["step"]),
        "num_gaussians": int(m.shape[0]),
        "opacity_mean": float(o.mean()),
        "opacity_median": float(o.median()),
        "dead_frac_lt_0.01": float((o < 0.01).float().mean()),
        "means_norm_median": float(n.median()),
        "means_norm_p99": float(torch.quantile(n, 0.99)),
        "means_abs_max": float(m.abs().max()),
        "scale_max_axis_median": float(s.median()),
        "scale_max_axis_max": float(s.max()),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--cap_max", type=int, default=3_000_000)
    p.add_argument("--arms", default=",".join(ARMS))
    args = p.parse_args(argv)

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    results = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "argv": [sys.executable, __file__] + list(sys.argv[1:]),
               "steps": args.steps, "cap_max": args.cap_max,
               "valid_frac_used": VALID_FRAC, "arms": {}}

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        rd = work / arm
        cmd = [sys.executable, str(TRAINER),
               "--data_dir", args.data_dir, "--result_dir", str(rd),
               "--holdout_list", str(Path(args.data_dir) / "holdout_test.txt"),
               "--data_factor", "1", "--max_steps", str(args.steps),
               "--save_steps", str(args.steps), "--seed", "0",
               "--strategy", "mcmc", "--cap_max", str(args.cap_max),
               "--antialiased", "--ssim_lambda", "0.2", "--sh_degree", "3",
               "--coarse_factor", "0"] + ARMS[arm]
        print(f"\n===== {arm} =====\n{' '.join(cmd)}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log = (work / f"{arm}.log")
        log.write_text(proc.stdout + "\n=== STDERR ===\n" + proc.stderr, encoding="utf8")
        entry = {"argv": cmd, "returncode": proc.returncode,
                 "seconds": round(time.time() - t0, 1), "log": str(log)}
        steps_lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("step=")]
        entry["last_step_line"] = steps_lines[-1] if steps_lines else None
        entry["train_psnr_last"] = (float(steps_lines[-1].split("psnr=")[1].split()[0])
                                    if steps_lines else None)
        ck = rd / "ckpts" / f"ckpt_{args.steps - 1}_rank0.pt"
        entry["ckpt_stats"] = ckpt_stats(ck) if ck.is_file() else None
        results["arms"][arm] = entry
        print(json.dumps(entry["ckpt_stats"], indent=1), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf8")

    print(f"\n[diag] -> {args.out}")
    hdr = f"{'arm':24s} {'opacity_mean':>13s} {'dead<0.01':>10s} {'|mean| p99':>12s} {'trainPSNR':>10s}"
    print(hdr)
    for arm, e in results["arms"].items():
        s = e["ckpt_stats"]
        if s:
            print(f"{arm:24s} {s['opacity_mean']:13.5f} {s['dead_frac_lt_0.01']:10.4f} "
                  f"{s['means_norm_p99']:12.2f} {e['train_psnr_last'] or 0:10.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
