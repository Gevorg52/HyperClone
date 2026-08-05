"""WP-O5: an objective, ground-truth-free free-viewpoint metric — VIEWPOINT STABILITY.

The off-path contact sheets show arms A/B/D looking very similar, and no ground truth exists
off the capture path, so "which arm wins off-path" needs something better than an eyeball.

What is measured: take a panorama pose, slide the camera sideways in many small equal steps
(a pure translation, same look direction), and measure LPIPS between CONSECUTIVE renders. A
Gaussian that is a needle is correct only from the direction it was fitted at, so it appears,
streaks and vanishes as the camera moves — a per-step discontinuity. A well-conditioned model
changes smoothly. Averaged over many start poses this is a real free-viewpoint property.

**The confound, stated up front:** a blurrier model also scores lower, because blur is smooth.
This number is therefore only meaningful BETWEEN ARMS WHOSE ON-PATH FIDELITY IS EQUAL. In
WP-O5 that condition holds by measurement — held-out masked PSNR 21.469 / 21.366 / 21.456 and
LPIPS 0.5505 / 0.5499 / 0.5517 for A / B / D are all within noise — so a difference here is
about view-conditioning, not about one arm being softer. Do not reuse this metric without
re-checking that condition.

  python _infusion/wpo5_viewstab.py --data_dir <packaged> --arm A=<ckpt> --arm D=<ckpt> \
      --out _infusion/wpo5/viewstab.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("wpo5_eval", PROJECT_ROOT / "_infusion" / "wpo5_eval.py")
wpo5_eval = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(wpo5_eval)

from core.training.gsplat_trainer import get_lpips, load_colmap_data, load_image  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python _infusion/wpo5_viewstab.py")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=CKPT")
    p.add_argument("--out", required=True)
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--rasterize_mode", default="antialiased")
    p.add_argument("--frame_cam", type=int, default=6)
    p.add_argument("--num_poses", type=int, default=8, help="start panoramas, evenly spaced")
    p.add_argument("--num_steps", type=int, default=12, help="lateral samples per sweep")
    p.add_argument("--span_steps", type=float, default=2.0,
                   help="total lateral sweep length, in median inter-panorama steps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_colmap_data(args.data_dir, factor=args.factor, normalize=True,
                            holdout_list=str(Path(args.data_dir) / "holdout_test.txt"))
    frames = wpo5_eval.pano_frames(data, args.frame_cam)
    centres, up = frames["centres"], frames["up"]
    travel = wpo5_eval.travel_dirs(centres)
    step = float(np.median(np.linalg.norm(np.diff(centres, axis=0), axis=1)))
    span = args.span_steps * step
    picks = np.unique(np.linspace(0, len(frames["panos"]) - 1, args.num_poses).astype(int))

    K = data["Ks_dict"][data["camera_ids"][frames["indices"][0]]]
    img0 = load_image(data, frames["indices"][0])["image"]
    H, W = int(img0.shape[0]), int(img0.shape[1])
    lp = get_lpips(device)

    # the sweep geometry is identical for every arm
    sweeps = []
    for pi in picks:
        c2w0 = frames["c2w"][pi]
        lateral = np.cross(travel[pi], up)
        n = np.linalg.norm(lateral)
        if n < 1e-9:
            continue
        lateral = lateral / n
        offs = np.linspace(-span / 2, span / 2, args.num_steps)
        poses = []
        for o in offs:
            w = c2w0.copy()
            w[:3, 3] = c2w0[:3, 3] + lateral * o
            poses.append(w)
        sweeps.append({"pano": frames["panos"][pi], "poses": poses,
                       "delta_units": float(offs[1] - offs[0])})

    out = {"wp": "WP-O5", "metric": "viewpoint stability (LPIPS between consecutive renders "
                                    "of a pure lateral translation)",
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "argv": [sys.executable, __file__] + list(sys.argv[1:]),
           "data_dir": str(Path(args.data_dir).resolve()),
           "frame_cam": args.frame_cam, "num_poses": len(sweeps),
           "num_steps": args.num_steps,
           "span_in_pano_steps": args.span_steps,
           "step_median_units": round(step, 6),
           "delta_per_sample_in_pano_steps": round(args.span_steps / (args.num_steps - 1), 4),
           "confound": "lower is smoother, but a blurrier model also scores lower — only "
                       "comparable across arms with equal on-path fidelity (see docstring)",
           "arms": {}}

    for spec in args.arm:
        name, ckpt = spec.split("=", 1)
        splats, _, _ = wpo5_eval.load_splats(Path(ckpt), device)
        t0 = time.time()
        per_sweep = []
        for sw in sweeps:
            imgs = []
            for w in sw["poses"]:
                bgr = wpo5_eval.render_at(splats, w, K, W, H, device, args.sh_degree,
                                          args.rasterize_mode)
                rgb = torch.from_numpy(bgr[..., ::-1].copy()).to(device).float() / 255.0
                # the repo's get_lpips() is torchmetrics LPIPS(normalize=True) -> wants [0,1]
                imgs.append(rgb.permute(2, 0, 1)[None])
            with torch.no_grad():
                vals = [float(lp(imgs[i], imgs[i + 1]).item())
                        for i in range(len(imgs) - 1)]
            per_sweep.append({"pano": sw["pano"], "mean": float(np.mean(vals)),
                              "max": float(np.max(vals))})
        allv = [s["mean"] for s in per_sweep]
        out["arms"][name] = {"ckpt": str(Path(ckpt).resolve()),
                             "mean_consecutive_lpips": round(float(np.mean(allv)), 6),
                             "median_consecutive_lpips": round(float(np.median(allv)), 6),
                             "worst_sweep_mean": round(float(np.max(allv)), 6),
                             "seconds": round(time.time() - t0, 1),
                             "per_sweep": per_sweep}
        print(f"[viewstab] {name:6s} mean {out['arms'][name]['mean_consecutive_lpips']:.5f} "
              f"median {out['arms'][name]['median_consecutive_lpips']:.5f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del splats
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf8")
    base = None
    print("\n| arm | mean consecutive LPIPS (lower = more view-stable) | vs baseline |")
    print("|---|---|---|")
    for name, a in out["arms"].items():
        v = a["mean_consecutive_lpips"]
        base = v if base is None else base
        print(f"| {name} | {v:.5f} | {(v / base - 1) * 100:+.1f} % |")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
