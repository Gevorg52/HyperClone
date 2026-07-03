"""GENUINE-HOLE testbed carver (WP-C1 prerequisite, Part 1).

Creates a REAL geometric void (observed% < 100 under core/qa/visibility.py) in a
trained production splat, in a full-scene dome capture where naive foreground
deletion only reveals background.

Mechanism = DEPTH-COLUMN (frustum-wedge) deletion. Pick a central hole-crossing
view V and an image box; delete every Gaussian whose projected centre lands in the
box AND whose view-space depth is within [d0,d1]*median_object_depth (d1=inf =>
the whole column). Deleting the whole depth column guarantees the rays through the
box hit NOTHING -> rendered alpha collapses -> the OGGSplat visibility mask reports
that region UNOBSERVED. Neighbouring views see the same 3D wedge under parallax, so
the void is hole-crossing (not a single-view artefact).

This is the task's slab/region-deletion route, chosen over pure view-withholding
because both production scenes are dense ~complete domes (Laptop 100% / GPU_V4 95%
hemisphere coverage) where withholding a locally-seen region's views would require
withholding a destructive fraction and would still heal over (the June-wedge
finding). Deletion + a fine-tune that EXCLUDES the hole-viewing arc (hole_finetune)
guarantees a clean 'generate-into-emptiness' void with real held-out reference
photos.

Measures observed% with the SAME core/qa/visibility.py the gate uses (ground truth).

Usage:
  python _infusion/hole_carve.py --ckpt <pt> --data <dir> --factor 2 \
     --view 176 --box cx,cy,w,h --depth d0,d1 --span 20 --out _infusion/hole/holed.pt
"""
import argparse, os, sys, json
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("APEX_HOLDOUT", None)
from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats
from core.qa.visibility import VisibilityConfig, build_visibility_mask

SH = 3


def load_splat(ckpt, dev):
    c = torch.load(ckpt, map_location=dev, weights_only=False)
    sp = c["splats"] if isinstance(c, dict) and "splats" in c else c
    return {k: v.to(dev).float() for k, v in sp.items()}


def render(sp, c2w, K, W, H, dev, mode="RGB+ED"):
    with torch.no_grad():
        r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH,
                                torch.zeros(3, device=dev), "antialiased", render_mode=mode)
    return r[0], a[0, ..., 0]


def obj_median_depth(sp, c2w, K, W, H, dev):
    r, a = render(sp, c2w, K, W, H, dev, "RGB+ED")
    d = r[..., 3]
    m = (a > 0.5) & (d > 1e-6)
    return float(d[m].median()) if m.any() else 1.0


def column_delete_mask(sp, c2w, K, W, H, box, depth_rng, dev, footprint=True):
    """Boolean (N,) = True for Gaussians to DELETE (image-box depth column).

    footprint=True (default): delete any Gaussian whose 3-sigma projected FOOTPRINT
    overlaps the box, not just whose centre lands in it. This is required to clear
    the pervasive low-opacity HAZE in these full-scene captures: large background
    Gaussians centred outside the box still splat into it and keep alpha>0.3 (a
    'present-but-smudged' smear instead of a clean alpha<0.3 void)."""
    means = sp["means"]
    w2c = torch.linalg.inv(c2w)
    cam = (means @ w2c[:3, :3].T) + w2c[:3, 3]           # (N,3) camera frame
    z = cam[:, 2]
    x = cam[:, 0] / z.clamp(min=1e-6) * K[0, 0] + K[0, 2]
    y = cam[:, 1] / z.clamp(min=1e-6) * K[1, 1] + K[1, 2]
    cx, cy, bw, bh = box
    x0, x1 = (cx - bw / 2) * W, (cx + bw / 2) * W
    y0, y1 = (cy - bh / 2) * H, (cy + bh / 2) * H
    d0, d1 = depth_rng
    if footprint:
        # 3-sigma screen radius (px) ~ f * (3*max_scale) / depth; only expand by the
        # portion of the footprint that lies INSIDE the box (>=50% overlap heuristic:
        # centre within box grown by 0.5*radius) to avoid nuking giant Gaussians that
        # merely graze the box (which would punch voids elsewhere).
        s3 = 3.0 * torch.exp(sp["scales"].float()).max(dim=1).values
        f = 0.5 * (K[0, 0] + K[1, 1])
        r = (f * s3 / z.clamp(min=1e-6)) * 0.5
        inbox = (z > 1e-6) & (x >= x0 - r) & (x < x1 + r) & (y >= y0 - r) & (y < y1 + r) \
            & (z >= d0) & (z < d1)
    else:
        inbox = (z > 1e-6) & (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & (z >= d0) & (z < d1)
    return inbox


def observed_pct(sp, c2w, K, W, H, dev, alpha_thresh=0.3):
    _, a = render(sp, c2w, K, W, H, dev, "RGB")
    vis = build_visibility_mask(a, VisibilityConfig(alpha_thresh=alpha_thresh))
    return vis["observed_pct"], vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--view", type=int, required=True, help="central hole-crossing view index")
    ap.add_argument("--box", default="0.5,0.45,0.3,0.3", help="cx,cy,w,h (fractions)")
    ap.add_argument("--depth", default="0,inf", help="d0,d1 in units of median object depth (inf ok)")
    ap.add_argument("--span", type=int, default=24, help="+/- neighbour views to scan")
    ap.add_argument("--step", type=int, default=3, help="neighbour stride")
    ap.add_argument("--out", default=None, help="save holed ckpt here (optional)")
    ap.add_argument("--emit_split", default=None, help="write train/holdout split json here")
    ap.add_argument("--obs_band", default="40,85", help="target observed%% band for hole-crossing views")
    args = ap.parse_args()

    dev = "cuda"
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hole")
    os.makedirs(outdir, exist_ok=True)
    sp = load_splat(args.ckpt, dev)
    N = sp["means"].shape[0]
    data = load_colmap_data(args.data, factor=args.factor, test_every=8, normalize=True)
    names = data["image_names"]
    nimg = len(names)
    box = [float(v) for v in args.box.split(",")]
    d0s, d1s = args.depth.split(",")
    d0 = float(d0s); d1 = float("inf") if d1s.strip() == "inf" else float(d1s)

    V = args.view
    b = load_image(data, V); K = b["K"].to(dev); c2w = b["c2w"].to(dev)
    H, W = b["image"].shape[:2]
    md = obj_median_depth(sp, c2w, K, W, H, dev)
    depth_rng = (d0 * md, d1 * md)
    print(f"[carve] view {V} ({names[V]}) {W}x{H}  median obj depth {md:.3f}  "
          f"depth column [{depth_rng[0]:.3f},{depth_rng[1]}]")

    dmask = column_delete_mask(sp, c2w, K, W, H, box, depth_rng, dev)
    ndel = int(dmask.sum())
    print(f"[carve] box {box} deletes {ndel:,}/{N:,} Gaussians ({100*ndel/N:.2f}%)")
    keep = ~dmask
    holed = {k: v[keep].clone() for k, v in sp.items()}

    # scan neighbours: observed% before/after
    lo, hi = int(args.obs_band.split(",")[0]), int(args.obs_band.split(",")[1])
    scan = list(range(max(0, V - args.span), min(nimg, V + args.span + 1), args.step))
    if V not in scan:
        scan.append(V); scan.sort()
    crossing = []
    print(f"[carve] scanning neighbours (observed%% full -> holed):")
    for j in scan:
        bj = load_image(data, j); Kj = bj["K"].to(dev); c2wj = bj["c2w"].to(dev)
        Hj, Wj = bj["image"].shape[:2]
        of, _ = observed_pct(sp, c2wj, Kj, Wj, Hj, dev)
        oh, vish = observed_pct(holed, c2wj, Kj, Wj, Hj, dev)
        tag = ""
        if lo <= oh <= hi:
            crossing.append(j); tag = "  <-- hole-crossing"
        print(f"   v{j:04d} {names[j]:>20s}: {of:6.1f} -> {oh:6.1f}{tag}")
        # save overlay for the central + a couple
        if j == V or j in (scan[0], scan[-1]):
            rh, ah = render(holed, c2wj, Kj, Wj, Hj, dev, "RGB")
            img = (rh[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            um = vish["unobserved"]
            img[um] = (0.4 * img[um] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
            cv2.imwrite(os.path.join(outdir, f"holed_v{j:04d}.jpg"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[carve] hole-crossing views ({lo}-{hi}%): {crossing}")

    if args.out:
        torch.save({"splats": {k: v.cpu() for k, v in holed.items()},
                    "carve": {"view": V, "box": box, "depth": [d0, d1], "n_deleted": ndel}},
                   args.out)
        print(f"[carve] saved holed ckpt -> {args.out}  ({int(keep.sum()):,} Gaussians)")
    if args.emit_split and crossing:
        split = {"hole_views": crossing, "central_view": V, "box": box, "depth": [d0, d1],
                 "n_deleted": ndel, "data": args.data, "factor": args.factor}
        with open(args.emit_split, "w") as f:
            json.dump(split, f, indent=2)
        print(f"[carve] wrote split -> {args.emit_split}")


if __name__ == "__main__":
    main()
