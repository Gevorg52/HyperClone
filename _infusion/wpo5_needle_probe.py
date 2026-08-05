"""WP-O5: is the free-viewpoint mush made of NEEDLES or of low-opacity FOG?

A direct ablation on an existing checkpoint, with no retraining. The same off-path and
"unvisited" viewpoints are rendered four ways:

    all            every Gaussian
    no-fog         sigmoid(opacity) >= --opacity_min deleted below threshold
    no-needles     effective rank >= --erank_min kept (needles deleted)
    neither        both filters

Deleting needles leaves holes where they were carrying real surface, so a cleaner render in
the "no-needles" column is NOT a usable fix — it is evidence about which population is
producing the spikes, which is what decides whether `--opacity_reg` or `--erank_lambda` is the
right training lever to reach for.

  python _infusion/wpo5_needle_probe.py --data_dir <packaged> --ckpt <ckpt.pt> \
      --out _infusion/wpo5/needle_probe
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("wpo5_eval", PROJECT_ROOT / "_infusion" / "wpo5_eval.py")
wpo5_eval = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(wpo5_eval)

from core.training.gsplat_trainer import load_colmap_data, load_image  # noqa: E402
from tools.prune_ply import effective_rank, splat_stats  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python _infusion/wpo5_needle_probe.py")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--opacity_min", type=float, default=0.05)
    p.add_argument("--erank_min", type=float, default=2.0)
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--rasterize_mode", default="antialiased")
    p.add_argument("--offpath_points", type=int, default=2)
    p.add_argument("--lateral_m", type=float, nargs="*", default=[-1.0, 1.0])
    p.add_argument("--vertical_m", type=float, nargs="*", default=[])
    p.add_argument("--unvisited_views", type=int, default=2)
    p.add_argument("--tile_px", type=int, default=520)
    p.add_argument("--ceiling_m", type=float, default=3.0)
    p.add_argument("--walk_mps", type=float, default=1.2)
    p.add_argument("--pano_fps", type=float, default=2.0)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_colmap_data(args.data_dir, factor=args.factor, normalize=True,
                            holdout_list=str(Path(args.data_dir) / "holdout_test.txt"))
    frames = wpo5_eval.pano_frames(data)
    scale = wpo5_eval.estimate_metres_per_unit(data, frames, args.pano_fps, args.ceiling_m,
                                               args.walk_mps)
    mpu = scale["metres_per_unit"]
    views, _ = wpo5_eval.build_offpath_views(data, frames, mpu, args)
    views = [v for v in views if v["kind"] in ("onpath", "lateral", "unvisited")]

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=True)["splats"]
    splats = {k: v.float() for k, v in raw.items()}
    n = splats["means"].shape[0]
    op = torch.sigmoid(splats["opacities"]).reshape(-1)
    er = effective_rank(splats["scales"])
    variants = {
        "all": torch.ones(n, dtype=torch.bool),
        f"no-fog (opacity>={args.opacity_min})": op >= args.opacity_min,
        f"no-needles (erank>={args.erank_min})": er >= args.erank_min,
        "neither": (op >= args.opacity_min) & (er >= args.erank_min),
    }

    K = data["Ks_dict"][data["camera_ids"][frames["indices"][0]]]
    img0 = load_image(data, frames["indices"][0])["image"]
    H, W = int(img0.shape[0]), int(img0.shape[1])

    renders, counts = {}, {}
    for label, keep in variants.items():
        sub = {k: v[keep].to(device) for k, v in splats.items()}
        counts[label] = int(keep.sum())
        t0 = time.time()
        renders[label] = [wpo5_eval.render_at(sub, v["c2w"], K, W, H, device,
                                              args.sh_degree, args.rasterize_mode)
                          for v in views]
        print(f"[probe] {label:34s} {counts[label]:>9d} gaussians "
              f"({counts[label] / n * 100:5.1f} %)  {time.time() - t0:.0f}s", flush=True)
        del sub
        torch.cuda.empty_cache()

    written = {}
    labels = list(variants)
    for i, v in enumerate(views):
        tiles = [wpo5_eval.label(renders[lb][i], lb, f"{v['group']}  {v['label']}")
                 for lb in labels]
        pth = out / f"probe_{v['group']}_{v['kind']}_{i:02d}.jpg"
        cv2.imwrite(str(pth), wpo5_eval.sheet(tiles, len(labels), args.tile_px),
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
        written[pth.name] = f"{v['group']} / {v['label']}: " + " | ".join(labels)

    payload = {"wp": "WP-O5", "probe": "needles vs fog, no retraining",
               "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "argv": [sys.executable, __file__] + list(sys.argv[1:]),
               "ckpt": str(Path(args.ckpt).resolve()), "num_gaussians": n,
               "kept_per_variant": counts,
               "kept_frac_per_variant": {k: round(c / n, 6) for k, c in counts.items()},
               "stats": splat_stats(splats), "scale": scale,
               "views": [{"group": v["group"], "label": v["label"], "kind": v["kind"]}
                         for v in views],
               "figures": written}
    (out / "needle_probe.json").write_text(json.dumps(payload, indent=1), encoding="utf8")
    print(f"[probe] {len(written)} sheets + needle_probe.json -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
