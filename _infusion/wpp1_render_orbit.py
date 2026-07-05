"""WP-P1 SwiftVR stage-1 (MAIN env): render the FULL orbit (all registered views, in
sorted/capture order) from a gsplat ckpt at --factor, as a PNG sequence for a temporal
video restorer (SwiftVR). Also saves held-out (every-8) GT + an index manifest so the
restored held-out frames can be scored vs GT afterwards.

Usage:
  python _infusion/wpp1_render_orbit.py --ckpt <stack.pt> --data_dir <data> --factor 2 \
    --work qa/wpp1_sv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.gsplat_trainer import (  # noqa: E402
    load_colmap_data, load_image, render_splats,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    device = "cuda"

    data = load_colmap_data(args.data_dir, factor=args.factor, test_every=8, normalize=True)
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    splats = {k: v.to(device) for k, v in ck["splats"].items()}

    work = Path(args.work)
    (work / "in").mkdir(parents=True, exist_ok=True)
    (work / "gt").mkdir(parents=True, exist_ok=True)

    n = len(data["image_names"])
    test_set = set(int(x) for x in data["test_indices"])
    heldout = []  # (seq_pos, orig_idx, stem)
    with torch.no_grad():
        for pos in range(n):
            b = load_image(data, pos)
            gt = b["image"].to(device)
            H, W = gt.shape[:2]
            K = b["K"].to(device)
            w2c = torch.linalg.inv(b["c2w"].to(device))
            r, _, _ = render_splats(splats, w2c, K, W, H, args.sh_degree,
                                    torch.zeros(3, device=device), "antialiased")
            img = (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(work / "in" / f"{pos:04d}.png")
            if pos in test_set:
                stem = Path(data["image_names"][pos]).stem
                Image.fromarray((gt.cpu().numpy() * 255).astype(np.uint8)).save(
                    work / "gt" / f"{pos:04d}.png")
                heldout.append({"seq": pos, "stem": stem})
            if pos % 50 == 0:
                print(f"rendered {pos}/{n} ({W}x{H})", flush=True)

    (work / "orbit_manifest.json").write_text(json.dumps(
        {"n": n, "factor": args.factor, "ckpt": str(args.ckpt),
         "data_dir": str(args.data_dir), "heldout": heldout}, indent=2), encoding="utf-8")
    print(f"WROTE {n} frames -> {work/'in'}; {len(heldout)} held-out GT saved", flush=True)


if __name__ == "__main__":
    main()
