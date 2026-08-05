"""WP-O3 bystander audit — WP-O1 masks the OPERATOR only, so other people in the scene are
unmasked and train as if they were geometry. This finds the held-out crops most likely to
contain one and renders them, so "no bystander ghosts" is a checked claim, not a hope.

Heuristic: skin-tone pixels (YCrCb box, the standard Cr/Cb skin gate) inside the VALID part
of the WP-O2 crop mask. The operator is masked out, so anything the heuristic finds in the
valid region is either a bystander or a false positive (wood, warm paper) — which is why the
output is a GT|render|mask strip a human can read, not a number.

Usage:
  python _infusion/wpo3_bystander_check.py --ckpt <ckpt.pt> --data_dir <packaged> \
      --out _infusion/wpo3/arm2_final --top 4
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.training.gsplat_trainer import load_colmap_data  # noqa: E402
from wpo3_eval import load_splats, render_view, strip  # noqa: E402  (same dir)


def skin_frac(bgr: np.ndarray, valid: np.ndarray) -> float:
    """Fraction of the VALID region that is skin-toned (YCrCb Cr 133-173, Cb 77-127)."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    n = int(valid.sum())
    return float((skin & valid).sum()) / max(n, 1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=4)
    p.add_argument("--min_blob_px", type=int, default=1500,
                   help="ignore skin blobs smaller than this (paper/wood speckle)")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    holdout = str(Path(args.data_dir) / "holdout_test.txt")
    data = load_colmap_data(args.data_dir, factor=1, normalize=True, holdout_list=holdout)

    rows = []
    for idx in data["test_indices"]:
        idx = int(idx)
        path = data["image_paths"][idx]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        mp = data["mask_paths"][idx]
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        valid = m > 127
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[..., 1], ycrcb[..., 2]
        skin = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & valid).astype(np.uint8)
        n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(skin, 8)
        biggest = int(stats[1:, cv2.CC_STAT_AREA].max()) if n_lbl > 1 else 0
        rows.append({"name": Path(path).name, "index": idx,
                     "skin_frac_in_valid": round(skin_frac(bgr, valid), 5),
                     "largest_skin_blob_px": biggest})

    rows.sort(key=lambda r: -r["largest_skin_blob_px"])
    hits = [r for r in rows if r["largest_skin_blob_px"] >= args.min_blob_px]

    splats, step = load_splats(Path(args.ckpt), device)
    names = [Path(x).name for x in data["image_paths"]]
    by_name = {n: i for i, n in enumerate(names)}
    written = []
    for r in rows[:args.top]:
        rr, g, mk = render_view(splats, data, by_name[r["name"]], device)
        f = out / f"bystander_{r['name'].replace('.jpg', '')}.jpg"
        cv2.imwrite(str(f), strip(rr, g, mk,
                                  f"{r['name']} (largest skin blob {r['largest_skin_blob_px']} px "
                                  f"in the VALID region)", 1.0),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append(f.name)

    payload = {"ckpt": str(Path(args.ckpt).resolve()), "ckpt_step": step,
               "num_heldout": len(rows), "min_blob_px": args.min_blob_px,
               "num_crops_over_threshold": len(hits),
               "rendered": written, "ranked": rows[:25]}
    (out / "bystander_check.json").write_text(json.dumps(payload, indent=1), encoding="utf8")
    print(f"[bystander] {len(hits)}/{len(rows)} held-out crops have a skin blob "
          f">= {args.min_blob_px} px inside the valid region; rendered top {len(written)}")
    for r in rows[:args.top]:
        print(f"    {r['name']}  blob={r['largest_skin_blob_px']}px  "
              f"skin_frac={r['skin_frac_in_valid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
