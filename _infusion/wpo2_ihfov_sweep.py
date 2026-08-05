"""WP-O2 — ih_fov / iv_fov stitch calibration sweep (deferred from WP-O1 §3.7).

WP-O1 shipped with `--ih_fov 200 --iv_fov 200` UNCALIBRATED (the value is nominal for the
Insta360 lens; 200/200 stitched coherently and the seams are masked anyway). WP-O2 is the
first stage that can actually SCORE a stitch, because a wrong input FOV warps the whole
fisheye->ERP angular mapping, and the EQUIRECTANGULAR camera model assumes that mapping is
exact -> the error shows up as reprojection error.

Protocol (per value v in --values):
  1. re-ingest the SAME source frames with `core.ingest_360 --ih_fov v --iv_fov v
     --max_frames N` (frame SELECTION is done on the raw fisheye stream and is independent
     of the stitch, so all values see byte-identical source frames),
  2. reconstruct with `core.sfm_360 --arm spherical` (ARM B; masks ON = production config),
  3. score by registration % first, mean reprojection error second (the brief's rule).

Everything is a subprocess so the recorded argv is exactly what ran.

    python _infusion/wpo2_ihfov_sweep.py \
        --input D:/HYPERCLONE/FOOTAGE/FlowerShop/VID_20260721_133413_00_077.insv \
        --work _dataset/360/_ihfov --out _infusion/wpo2/ihfov_sweep.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    with log.open("w", encoding="utf8", errors="replace") as fh:
        fh.write("+ " + " ".join(str(c) for c in cmd) + "\n")
        fh.flush()
        p = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    return p.returncode


def collect_noise(work: Path, out: Path) -> int:
    """Summarise repeat runs (`noise_<value>_s<seed>/`) into a run-to-run noise floor.

    A sweep is only readable against its own noise: COLMAP's incremental mapper is
    multi-threaded and is NOT bit-reproducible even at a fixed --seed, so a difference in
    mean reprojection error only means something if it is larger than this spread. Each run's
    own `sfm_report.json` carries the argv that produced it.
    """
    import statistics

    groups: dict = {}
    for rep_path in sorted(work.glob("noise_*/sfm_report.json")):
        rep = json.loads(rep_path.read_text(encoding="utf8"))
        value = rep_path.parent.name.split("_")[1]
        groups.setdefault(value, []).append({
            "run": rep_path.parent.name, "seed": rep["seed"],
            "mean_reproj_error_px": rep["metrics_equirect"]["mean_reproj_error_px"],
            "num_points3D": rep["metrics_equirect"]["num_points3D"],
            "registered_pct": rep["metrics_equirect"]["registered_pct"],
            "argv": rep["argv"]})
    # Fold in the sweep run for the same value, which used identical argv but seed 0.
    for value in groups:
        sweep_rep = work / f"sfm_{value}" / "sfm_report.json"
        if sweep_rep.exists():
            rep = json.loads(sweep_rep.read_text(encoding="utf8"))
            groups[value].append({
                "run": f"sfm_{value} (sweep)", "seed": rep["seed"],
                "mean_reproj_error_px": rep["metrics_equirect"]["mean_reproj_error_px"],
                "num_points3D": rep["metrics_equirect"]["num_points3D"],
                "registered_pct": rep["metrics_equirect"]["registered_pct"],
                "argv": rep["argv"]})
    summary = {}
    for value, runs in groups.items():
        e = [r["mean_reproj_error_px"] for r in runs]
        summary[value] = {
            "n_runs": len(e), "mean": round(statistics.fmean(e), 5),
            "min": round(min(e), 5), "max": round(max(e), 5),
            "range": round(max(e) - min(e), 5),
            "stdev": round(statistics.stdev(e), 5) if len(e) > 1 else None,
            "runs": runs}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "wp": "WP-O2", "what": "ih_fov sweep run-to-run noise floor",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "same dataset, same argv, seeds 0/1/2 plus the sweep run; COLMAP's mapper is "
                "multi-threaded and not bit-reproducible, so this is the resolution limit of "
                "the sweep",
        "by_value": summary}, indent=1), encoding="utf8")
    for value, s in sorted(summary.items()):
        print(f"ih_fov={value}: n={s['n_runs']} mean={s['mean']:.5f} "
              f"range={s['range']:.5f} sd={s['stdev']}")
    print(f"-> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collect_noise", action="store_true",
                    help="do not sweep; summarise existing noise_<value>_s<seed> runs in "
                         "--work into --out")
    ap.add_argument("--input", help="source .insv/.mov")
    ap.add_argument("--values", default="192,194,196,198,200")
    ap.add_argument("--n_frames", type=int, default=20)
    ap.add_argument("--work", default="_dataset/360/_ihfov", help="heavy outputs (gitignored)")
    ap.add_argument("--out", default="_infusion/wpo2/ihfov_sweep.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    work = (REPO / args.work).resolve()
    out = (REPO / args.out).resolve()
    if args.collect_noise:
        return collect_noise(work, out)
    if not args.input:
        ap.error("--input is required unless --collect_noise is passed")
    values = [float(v) for v in args.values.split(",")]
    rows = []
    t_all = time.time()
    for v in values:
        tag = f"{v:g}".replace(".", "p")
        ds = work / f"ingest_{tag}"
        sfm = work / f"sfm_{tag}"
        t0 = time.time()
        ingest_cmd = [sys.executable, "-m", "core.ingest_360",
                      "--capture-type", "insta360", "--input", args.input,
                      "--out_dir", str(ds), "--overwrite",
                      "--max_frames", str(args.n_frames),
                      "--ih_fov", f"{v}", "--iv_fov", f"{v}",
                      "--contact_sheet", "0", "--progress", "5000", "--seed", str(args.seed)]
        rc = run(ingest_cmd, work / f"log_ingest_{tag}.txt")
        if rc != 0:
            rows.append({"ih_fov": v, "status": f"ingest failed rc={rc}"})
            continue
        t_ingest = time.time() - t0
        t0 = time.time()
        sfm_cmd = [sys.executable, "-m", "core.sfm_360",
                   "--capture-type", "360", "--arm", "spherical",
                   "--dataset", str(ds), "--out_dir", str(sfm), "--overwrite",
                   "--matcher", "exhaustive", "--seed", str(args.seed),
                   "--timeout_s", "3600", "--heartbeat_s", "120"]
        rc = run(sfm_cmd, work / f"log_sfm_{tag}.txt")
        if rc != 0:
            rows.append({"ih_fov": v, "status": f"sfm failed rc={rc}",
                         "ingest_argv": ingest_cmd, "sfm_argv": sfm_cmd})
            continue
        rep = json.loads((sfm / "sfm_report.json").read_text(encoding="utf8"))
        m = rep["metrics_equirect"]
        rows.append({
            "ih_fov": v, "iv_fov": v, "status": "ok",
            "n_panos": m["num_input_panos"],
            "registered": m["num_registered_panos"],
            "registered_pct": m["registered_pct"],
            "mean_reproj_error_px": m["mean_reproj_error_px"],
            "num_points3D": m["num_points3D"],
            "mean_track_length": m["mean_track_length"],
            "mean_obs_per_reg_image": m["mean_obs_per_reg_image"],
            "num_reconstructions": rep["num_reconstructions"],
            "ingest_s": round(t_ingest, 1),
            "sfm_s": round(time.time() - t0, 1),
            "ingest_argv": ingest_cmd, "sfm_argv": sfm_cmd,
            "dataset_dir": str(ds), "sfm_dir": str(sfm),
        })
        print(f"[sweep] ih_fov={v}: {rows[-1]['registered']}/{rows[-1]['n_panos']} reg, "
              f"reproj {rows[-1]['mean_reproj_error_px']:.4f} px, "
              f"{rows[-1]['num_points3D']} pts", flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    winner = None
    if ok:
        winner = sorted(ok, key=lambda r: (-r["registered_pct"], r["mean_reproj_error_px"]))[0]
    report = {
        "wp": "WP-O2",
        "what": "ih_fov/iv_fov stitch calibration sweep (deferred from WP-O1)",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": args.input,
        "n_frames_per_value": args.n_frames,
        "arm": "spherical (ARM B), masks ON, exhaustive matcher",
        "score_rule": "registration % first, mean reprojection error second",
        "seed": args.seed,
        "total_s": round(time.time() - t_all, 1),
        "winner_ih_fov": winner["ih_fov"] if winner else None,
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf8")

    hdr = f"| ih_fov | reg | reg% | mean reproj px | points3D | track | obs/img | sfm s |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [hdr, sep]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| {r['ih_fov']:g} | {r.get('status')} | | | | | | |")
            continue
        lines.append(f"| {r['ih_fov']:g} | {r['registered']}/{r['n_panos']} | "
                     f"{r['registered_pct']:.2f} | {r['mean_reproj_error_px']:.4f} | "
                     f"{r['num_points3D']} | {r['mean_track_length']:.3f} | "
                     f"{r['mean_obs_per_reg_image']:.1f} | {r['sfm_s']:.0f} |")
    table = "\n".join(lines)
    out.with_suffix(".md").write_text(
        f"# WP-O2 ih_fov sweep ({args.n_frames} frames, ARM B spherical)\n\n{table}\n\n"
        f"winner: **{report['winner_ih_fov']}**\n", encoding="utf8")
    print("\n" + table + f"\n\nwinner: {report['winner_ih_fov']}\n-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
