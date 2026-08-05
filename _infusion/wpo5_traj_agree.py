"""WP-O5: do the SIFT and ALIKED reconstructions actually disagree about the CAMERA POSES?

Mean reprojection error is a keypoint-localisation statistic, not a pose statistic, and there
is no ground-truth trajectory for this capture. What we CAN measure is whether the two feature
stacks land on the same trajectory: align the two panorama-centre tracks with a similarity
(Umeyama) transform and report the residual as a fraction of the median inter-panorama step.

If the tracks agree to well under a step, the reprojection difference is about where keypoints
sit inside an image, not about where the camera is — which is the honest reading to record.

  python _infusion/wpo5_traj_agree.py --ref sift=<sparse_equirectangular/0> \
      --other aliked=<sparse_equirectangular/0> --out _infusion/wpo5/traj_agree.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def pano_centres(model_dir: Path) -> tuple:
    import pycolmap

    pycolmap.logging.minloglevel = 3
    rec = pycolmap.Reconstruction(str(model_dir))
    rows = []
    for image_id in rec.reg_image_ids():
        im = rec.images[image_id]
        rows.append((im.name, np.asarray(im.cam_from_world().inverse().translation,
                                         dtype=np.float64)))
    rows.sort(key=lambda r: r[0])
    return [r[0] for r in rows], np.stack([r[1] for r in rows])


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple:
    """Similarity transform (scale, R, t) mapping src onto dst (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / src.shape[0]
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1
    rot = u @ s @ vt
    var_s = (s0 ** 2).sum() / src.shape[0]
    scale = float(np.trace(np.diag(d) @ s) / var_s) if var_s > 1e-12 else 1.0
    trans = mu_d - scale * rot @ mu_s
    return scale, rot, trans


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python _infusion/wpo5_traj_agree.py")
    p.add_argument("--ref", required=True, metavar="NAME=DIR")
    p.add_argument("--other", action="append", required=True, metavar="NAME=DIR")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    ref_name, ref_dir = args.ref.split("=", 1)
    ref_names, ref_c = pano_centres(Path(ref_dir))
    ref_steps = np.linalg.norm(np.diff(ref_c, axis=0), axis=1)
    ref_step = float(np.median(ref_steps))

    out = {"wp": "WP-O5", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "argv": [sys.executable, __file__] + list(sys.argv[1:]),
           "reference": {"name": ref_name, "dir": ref_dir, "num_panos": len(ref_names),
                         "step_median_units": round(ref_step, 6),
                         "path_length_units": round(float(ref_steps.sum()), 6)},
           "comparisons": {}}

    for spec in args.other:
        name, d = spec.split("=", 1)
        names, c = pano_centres(Path(d))
        common = [n for n in ref_names if n in set(names)]
        i_ref = [ref_names.index(n) for n in common]
        i_oth = [names.index(n) for n in common]
        a, b = c[i_oth], ref_c[i_ref]
        scale, rot, trans = umeyama(a, b)
        aligned = (scale * (rot @ a.T)).T + trans
        err = np.linalg.norm(aligned - b, axis=1)
        out["comparisons"][name] = {
            "dir": d, "num_common_panos": len(common),
            "similarity_scale": round(scale, 6),
            "rmse_units": round(float(np.sqrt((err ** 2).mean())), 6),
            "max_units": round(float(err.max()), 6),
            "rmse_in_reference_steps": round(float(np.sqrt((err ** 2).mean()) / ref_step), 5),
            "max_in_reference_steps": round(float(err.max() / ref_step), 5)}
        r = out["comparisons"][name]
        print(f"{name} vs {ref_name}: {len(common)} panos, scale {scale:.4f}, "
              f"RMSE {r['rmse_units']:.5f} units = {r['rmse_in_reference_steps']:.3f} "
              f"inter-pano steps (max {r['max_in_reference_steps']:.3f} steps)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
