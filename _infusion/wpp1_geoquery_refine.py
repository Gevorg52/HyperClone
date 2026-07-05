"""WP-P1 GeoQuery pipeline, stage 2 (GEOQUERY venv): read the manifest written by
wpp1_geoquery_prep.py, refine every target render with GeoQuery.sample(), write
refined/<i>.png into the same work dir.

Run from D:\\REPOS\\HYPERclone\\GeoQuery with PYTHONPATH=src and the geoquery venv python:
  set PYTHONPATH=src
  <geoquery python> _infusion/wpp1_geoquery_refine.py \
     --work <abs work dir> --pkl weights/geoquery_window_size3_down1.pkl --window_size 3
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--window_size", type=int, default=3)
    ap.add_argument("--refine_max", type=int, default=0,
                    help="if >0, downscale long side to this before refine, then upscale back")
    ap.add_argument("--prompt", default="remove degradation")
    args = ap.parse_args()

    work = Path(args.work)
    man = json.loads((work / "manifest.json").read_text())
    entries = man["entries"]

    from model import GeoQuery
    gq = GeoQuery(pretrained_path=args.pkl, timestep=199,
                  neighborhood_size=args.window_size, low_res_only=True)
    if hasattr(gq, "set_eval"):
        gq.set_eval()
    gq.to("cuda")

    (work / "refined").mkdir(parents=True, exist_ok=True)
    times = []
    for e in entries:
        tgt = Image.open(work / e["target_png"]).convert("RGB")
        ref = Image.open(work / e["ref_png"]).convert("RGB")
        depth = np.load(work / e["ref_depth_npy"]).astype(np.float32)
        K = np.array(e["K"], dtype=np.float32)
        ref_pose = np.array(e["ref_pose_raw"], dtype=np.float32)
        tgt_pose = np.array(e["target_pose_raw"], dtype=np.float32)

        orig = tgt.size  # (W,H)
        scale = 1.0
        if args.refine_max > 0 and max(orig) > args.refine_max:
            scale = args.refine_max / max(orig)
            nw, nh = int(round(orig[0] * scale)), int(round(orig[1] * scale))
            tgt_r = tgt.resize((nw, nh), Image.LANCZOS)
            ref_r = ref.resize((nw, nh), Image.LANCZOS)
            K_r = K.copy(); K_r[0, :] *= scale; K_r[1, :] *= scale
        else:
            tgt_r, ref_r, K_r = tgt, ref, K

        with torch.no_grad():
            out, dt = gq.sample(tgt_r, ref_r, depth, K_r, ref_pose, tgt_pose, prompt=args.prompt)
        if out.size != orig:
            out = out.resize(orig, Image.LANCZOS)
        out.save(work / "refined" / f"{e['i']:04d}.png")
        times.append(dt)
        print(f"[refine {e['i']:04d}] {out.size} fwd {dt:.2f}s", flush=True)

    print(f"REFINE DONE {len(entries)} views, mean fwd {np.mean(times):.2f}s", flush=True)


if __name__ == "__main__":
    main()
