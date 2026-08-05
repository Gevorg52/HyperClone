"""WP-O5 SIFT-parity guard: crop-rig SfM twice on the same panos, SIFT vs ALIKED+LightGlue.

The quality guard before the 8K production run. Both arms share EVERYTHING that defines the
reconstruction problem — the same WP-O1 pano dataset, the same 12-crop rig, the same
SIMPLE_PINHOLE intrinsics, the same WP-O1 masks, the same `SequentialPairingOptions`, the same
`rig_verification`, the same mapper options and the same seed. Only the feature stack differs
(`--features sift` vs `--features aliked`). Loop detection is OFF in both so the pair sets are
provably identical (the report records the pair counts to prove it).

Writes one scorecard JSON with both arms' full argv, seed, per-stage timings, database
statistics and reconstruction metrics.

  python _infusion/wpo5_sfm_ab.py --dataset _dataset/360/theater_027_8k --panos 20 \
      --work _dataset/360/sfm --out _infusion/wpo5/sfm_parity_ab.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def db_stats(db_path: Path) -> dict:
    import numpy as np
    import pycolmap

    pycolmap.logging.minloglevel = 3
    with pycolmap.Database.open(db_path) as db:
        ks = [db.num_keypoints_for_image(im.image_id) for im in db.read_all_images()]
        return {"num_images": db.num_images(),
                "keypoints_total": int(sum(ks)),
                "keypoints_per_image_mean": round(float(np.mean(ks)), 1),
                "keypoints_per_image_min": int(min(ks)),
                "keypoints_per_image_max": int(max(ks)),
                "matched_image_pairs": db.num_matched_image_pairs(),
                "verified_image_pairs": db.num_verified_image_pairs(),
                "inlier_matches": db.num_inlier_matches()}


def run_arm(tag: str, features: str, args) -> dict:
    out_dir = Path(args.work) / f"wpo5_ab_{tag}"
    cmd = [sys.executable, "-m", "core.sfm_360", "--capture-type", "360",
           "--arm", "crop_rig", "--features", features,
           "--dataset", args.dataset, "--out_dir", str(out_dir), "--overwrite",
           "--max_panos", str(args.panos), "--matcher", "sequential",
           "--overlap", str(args.overlap), "--quadratic_overlap", "1",
           "--loop_detection", "0", "--retrieval_num", "0",
           "--seed", str(args.seed), "--timeout_s", str(args.timeout_s),
           "--heartbeat_s", "300", "--progress", "10"]
    if features == "aliked":
        cmd += ["--aliked_max_keypoints", str(args.aliked_max_keypoints),
                "--aliked_resize_max", str(args.aliked_resize_max)]
    log = Path(args.work) / f"wpo5_ab_{tag}.log"
    print(f"\n+ [{tag}] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf8", errors="replace") as fh:
        rc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    report_path = out_dir / "sfm_report.json"
    report = json.loads(report_path.read_text(encoding="utf8")) \
        if report_path.is_file() else None
    arm = {"tag": tag, "features": features, "returncode": rc,
           "wall_clock_s": round(wall, 2), "argv": cmd, "log": str(log),
           "out_dir": str(out_dir)}
    if report is not None:
        arm["timings_s"] = report["timings_s"]
        arm["metrics_crop"] = report["metrics_crop"]
        arm["metrics_equirect"] = report["metrics_equirect"]
        arm["trajectory"] = report["trajectory"]
        arm["matching"] = report["matching"]
        arm["hloc"] = report.get("hloc")
        arm["crop_size"] = report["crop_size"]
        arm["crop_focal"] = report["crop_focal"]
    if (out_dir / "database.db").is_file():
        arm["database"] = db_stats(out_dir / "database.db")
    print(f"  [{tag}] rc={rc} in {wall:.0f}s", flush=True)
    return arm


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python _infusion/wpo5_sfm_ab.py")
    p.add_argument("--dataset", required=True, help="WP-O1 pano dataset")
    p.add_argument("--panos", type=int, default=20)
    p.add_argument("--work", default="_dataset/360/sfm")
    p.add_argument("--out", required=True, help="scorecard JSON")
    p.add_argument("--overlap", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout_s", type=float, default=21600.0)
    p.add_argument("--aliked_max_keypoints", type=int, default=8192)
    p.add_argument("--aliked_resize_max", type=int, default=1600)
    p.add_argument("--arms", default="sift,aliked")
    args = p.parse_args(argv)

    Path(args.work).mkdir(parents=True, exist_ok=True)
    arms = []
    for spec in [a.strip() for a in args.arms.split(",") if a.strip()]:
        # "tag" or "tag=features" (so the same feature stack can be run twice with
        # different knobs and still land in separate out_dirs)
        tag, _, features = spec.partition("=")
        arms.append(run_arm(tag, features or tag, args))

    card = {"wp": "WP-O5", "check": "SIFT-parity guard (crop-rig SfM, features-only A/B)",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "argv": [sys.executable, __file__] + list(sys.argv[1:]),
            "dataset": args.dataset, "panos": args.panos, "seed": args.seed,
            "shared": "rig config, SIMPLE_PINHOLE intrinsics, WP-O1 masks, "
                      "SequentialPairingOptions(overlap=%d, quadratic, loop_detection=0), "
                      "rig_verification=True, skip_image_pairs_in_same_frame, mapper options"
                      % args.overlap,
            "arms": arms}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=1), encoding="utf8")

    print("\n| arm | reg panos | reproj px (crop) | reproj px (ERP) | points3D | obs | "
          "kpts/img | inlier matches | wall s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        m, e, d = a.get("metrics_crop"), a.get("metrics_equirect"), a.get("database", {})
        if not m:
            print(f"| {a['tag']} | FAILED rc={a['returncode']} | | | | | | | "
                  f"{a['wall_clock_s']:.0f} |")
            continue
        print(f"| {a['tag']} | {m['num_registered_panos']}/{m['num_input_panos']} "
              f"({m['registered_pct']:.1f}%) | {m['mean_reproj_error_px']:.4f} | "
              f"{e['mean_reproj_error_px']:.4f} | {m['num_points3D']} | "
              f"{m['num_observations']} | {d.get('keypoints_per_image_mean')} | "
              f"{d.get('inlier_matches')} | {a['wall_clock_s']:.0f} |")
    print(f"\nscorecard -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
