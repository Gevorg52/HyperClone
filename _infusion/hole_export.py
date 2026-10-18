"""Export per-view fill inputs for the hole arc (feeds WP-C1 Part 2).

For each hole-crossing view: dump the holed-splat RGB render, accumulated alpha,
the OGGSplat unobserved mask (white=fill, per core/qa/visibility.py), the real GT
photo, and the camera (K, c2w) in the trainer's normalized frame. These are the
GaMO latent-gate inputs (observed render to anchor generation + the mask to gate).

Usage: python _infusion/hole_export.py --holed <pt> --data <dir> --factor 2 \
          --split 293:316 --out _infusion/hole/arc
"""
import argparse, os, sys, json
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("APEX_HOLDOUT", None)
from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats
from core.qa.visibility import VisibilityConfig, build_visibility_mask
from qa_report import load_splats

SH = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holed", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--split", default="293:316", help="LO:HI absolute name-sorted indices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha_thresh", type=float, default=0.3)
    args = ap.parse_args()

    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    sp = load_splats(args.holed, dev)
    data = load_colmap_data(args.data, factor=args.factor, test_every=8, normalize=True)
    lo, hi = (int(x) for x in args.split.split(":"))
    idxs = list(range(lo, hi))
    vcfg = VisibilityConfig(alpha_thresh=args.alpha_thresh)
    meta = {}
    for idx in idxs:
        b = load_image(data, idx)
        gt = b["image"].to(dev); K = b["K"].to(dev); c2w = b["c2w"].to(dev)
        H, W = gt.shape[:2]
        with torch.no_grad():
            r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH,
                                    torch.zeros(3, device=dev), "antialiased")
        render = r[0].clamp(0, 1); alpha = a[0, ..., 0]
        vis = build_visibility_mask(alpha, vcfg)
        stem = f"v{idx:04d}"
        cv2.imwrite(os.path.join(args.out, f"{stem}_render.png"),
                    cv2.cvtColor((render.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.out, f"{stem}_gt.png"),
                    cv2.cvtColor((gt.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.out, f"{stem}_mask.png"),
                    (vis["unobserved"].astype(np.uint8) * 255))
        np.save(os.path.join(args.out, f"{stem}_alpha.npy"), alpha.cpu().numpy().astype(np.float32))
        meta[stem] = {"index": idx, "name": data["image_names"][idx],
                      "K": K.cpu().numpy().tolist(), "c2w": c2w.cpu().numpy().tolist(),
                      "W": W, "H": H, "observed_pct": vis["observed_pct"]}
        print(f"  {stem} {data['image_names'][idx]} obs {vis['observed_pct']:.1f}%")
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] {len(idxs)} views -> {args.out}")


if __name__ == "__main__":
    main()
