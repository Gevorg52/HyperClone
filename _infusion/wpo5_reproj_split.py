"""WP-O5: de-confound the SIFT-vs-ALIKED mean-reprojection-error comparison.

Mean reprojection error is NOT directly comparable across feature detectors when one of them
produced 4x the tracks: the extra tracks are, by construction, the weaker/longer-baseline
correspondences the other detector never matched, and they carry higher residuals. Before
calling a +0.17 px difference a quality regression, split the error by track length and
report the distribution, not just the mean.

  python _infusion/wpo5_reproj_split.py --model sift=<sparse/0> --model aliked=<sparse/0> \
      --out _infusion/wpo5/reproj_split.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def per_point_errors(model_dir: Path) -> dict:
    import pycolmap

    pycolmap.logging.minloglevel = 3
    rec = pycolmap.Reconstruction(str(model_dir))
    rec.update_point_3d_errors()
    err, tlen = [], []
    for pt in rec.points3D.values():
        err.append(float(pt.error))
        tlen.append(int(pt.track.length()))
    err = np.asarray(err)
    tlen = np.asarray(tlen)
    ok = np.isfinite(err) & (err >= 0)
    return {"error": err[ok], "track_length": tlen[ok],
            "num_reg_images": rec.num_reg_images(),
            "num_observations": rec.compute_num_observations()}


def summarise(err: np.ndarray) -> dict:
    if err.size == 0:
        return None
    return {"n": int(err.size), "mean": round(float(err.mean()), 4),
            "median": round(float(np.median(err)), 4),
            "p90": round(float(np.percentile(err, 90)), 4),
            "p99": round(float(np.percentile(err, 99)), 4),
            "max": round(float(err.max()), 4)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python _infusion/wpo5_reproj_split.py")
    p.add_argument("--model", action="append", required=True, metavar="NAME=DIR")
    p.add_argument("--out", required=True)
    p.add_argument("--bins", type=int, nargs="*", default=[2, 3, 4, 6, 10])
    args = p.parse_args(argv)

    out = {"wp": "WP-O5", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "argv": [sys.executable, __file__] + list(sys.argv[1:]),
           "note": "reprojection error per 3D point, split by track length. Errors are in "
                   "the 1920x1920 crop model's pixels.",
           "models": {}}
    data = {}
    for spec in args.model:
        name, d = spec.split("=", 1)
        data[name] = per_point_errors(Path(d))
        e, t = data[name]["error"], data[name]["track_length"]
        rows = {"all": summarise(e)}
        for lo in args.bins:
            rows[f"track>={lo}"] = summarise(e[t >= lo])
        rows["track==2"] = summarise(e[t == 2])
        out["models"][name] = {
            "dir": d, "num_reg_images": data[name]["num_reg_images"],
            "num_points3D": int(e.size), "num_observations": data[name]["num_observations"],
            "mean_track_length": round(float(t.mean()), 4),
            "track_length_hist": {str(k): int((t == k).sum()) for k in range(2, 11)},
            "track_length_ge_11": int((t >= 11).sum()),
            "by_track_length": rows}

    # matched-population comparison: top-N points by track length in each model, N = the
    # smaller model's point count. Removes "one model simply kept more weak tracks".
    names = list(data)
    if len(names) == 2:
        a, b = names
        n = min(data[a]["error"].size, data[b]["error"].size)
        matched = {}
        for name in names:
            e, t = data[name]["error"], data[name]["track_length"]
            order = np.argsort(-t, kind="stable")[:n]
            matched[name] = summarise(e[order])
        out["matched_top_tracks"] = {"n_points_each": int(n), "by_model": matched}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf8")

    print("\n| model | points | mean | median | track>=4 mean | track==2 mean | "
          "mean track len |")
    print("|---|---|---|---|---|---|---|")
    for name, m in out["models"].items():
        r = m["by_track_length"]
        print(f"| {name} | {m['num_points3D']} | {r['all']['mean']} | "
              f"{r['all']['median']} | {r['track>=4']['mean'] if r.get('track>=4') else '-'} "
              f"| {r['track==2']['mean'] if r.get('track==2') else '-'} | "
              f"{m['mean_track_length']} |")
    if "matched_top_tracks" in out:
        mt = out["matched_top_tracks"]
        print(f"\nmatched population (top {mt['n_points_each']} points by track length):")
        for name, s in mt["by_model"].items():
            print(f"  {name:8s} mean {s['mean']}  median {s['median']}  p90 {s['p90']}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
