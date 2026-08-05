"""WP-O5 chain: wait for the running 8K SfM, then package it and run the training sweep.

Exists only so the packaging + 4-arm sweep start the instant the SfM finishes instead of
waiting for a human/agent turn. Every step is the same command written in
`_infusion/wpo5_RECORD.md` section 0; nothing here changes behaviour.

  python _infusion/wpo5_pipeline.py --sfm _dataset/360/sfm/wpo5_theater_027_8k \
      --data_dir _dataset/360/train/wpo5_theater_027_8k --arms A,B,C,D
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd, log: Path) -> int:
    print(f"\n+ {' '.join(str(c) for c in cmd)}\n  -> {log}", flush=True)
    t0 = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf8", errors="replace") as fh:
        rc = subprocess.run([str(c) for c in cmd], cwd=PROJECT_ROOT, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    print(f"  rc={rc} in {time.time() - t0:.0f}s", flush=True)
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python _infusion/wpo5_pipeline.py")
    p.add_argument("--sfm", required=True, help="the running SfM out_dir")
    p.add_argument("--data_dir", required=True, help="packaged trainer dataset to create")
    p.add_argument("--work", default="_dataset/360/train")
    p.add_argument("--arms", default="A,B,C,D")
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--poll_s", type=float, default=60.0)
    p.add_argument("--wait_timeout_s", type=float, default=28800.0)
    args = p.parse_args(argv)

    report = Path(args.sfm) / "sfm_report.json"
    incomplete = Path(args.sfm) / "INCOMPLETE"
    t0 = time.time()
    while not report.is_file():
        if incomplete.is_file():
            print(f"ABORT: {incomplete} exists — the SfM failed; not packaging.", flush=True)
            return 1
        if time.time() - t0 > args.wait_timeout_s:
            print(f"ABORT: no {report} after {args.wait_timeout_s:.0f}s", flush=True)
            return 1
        time.sleep(args.poll_s)
    time.sleep(5)                                     # let the writer flush
    rep = json.loads(report.read_text(encoding="utf8"))
    m = rep.get("metrics_equirect") or rep["metrics_crop"]
    print(f"[chain] SfM done: {m['num_registered_panos']}/{m['num_input_panos']} panos "
          f"({m['registered_pct']:.2f} %), reproj {m['mean_reproj_error_px']:.4f} px, "
          f"{m['num_points3D']} points, {rep['timings_s']['total']:.0f}s", flush=True)
    if m["registered_pct"] < 95.0:
        print("ABORT: SfM gate (>=95 % panos registered) FAILED — not training.", flush=True)
        return 1

    rc = run([sys.executable, "-m", "core.train_360_prep", "--capture-type", "360",
              "--sfm", args.sfm, "--out_dir", args.data_dir, "--overwrite", "--progress", "25"],
             Path(args.work) / "wpo5_prep.log")
    if rc != 0:
        print("ABORT: train_360_prep failed", flush=True)
        return rc

    prep = Path(args.data_dir) / "prep_report.json"
    if prep.is_file():
        pr = json.loads(prep.read_text(encoding="utf8"))
        keys = ("num_images", "num_train", "num_test", "crops_per_pano", "num_panos",
                "num_panos_fully_registered", "split")
        print("[chain] packaged: "
              + json.dumps({k: v for k, v in pr.items() if k in keys}), flush=True)

    return run([sys.executable, str(PROJECT_ROOT / "_infusion" / "wpo5_train.py"),
                "--data_dir", args.data_dir, "--work", args.work,
                "--out", "_infusion/wpo5/train_runs.json", "--arms", args.arms,
                "--max_steps", str(args.max_steps)],
               Path(args.work) / "wpo5_train_driver.log")


if __name__ == "__main__":
    raise SystemExit(main())
