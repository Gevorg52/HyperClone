"""WP-T2 region-LPIPS + GT|arm|arm zoom strips on SPECIFIC held-out views with
PER-VIEW crop boxes (unlike wps1_region_eval which uses one box over all views).

For each (view_idx, box, name) and each arm ckpt, at --factor:
  full_lpips   : torchmetrics VGG-LPIPS (normalize=True) whole frame
  region_lpips : same LPIPS on the fractional crop box (x0,x1,y0,y1)
  region_psnr  : PSNR from crop MSE (dB)
and writes GT | arm1 | arm2 ... strips (full + 2x NEAREST region crop).

Both models are 3D Gaussians -> renderable at any factor, so the SAME script does
the common-res (factor 3) and native (factor 2) strips. LPIPS model == trainer's
get_lpips, directly comparable to the WP-S1/N1/T1 region evals.

view spec: "IDX:x0,x1,y0,y1:NAME"  (fractional box)

Usage:
  python _infusion/wpt2_regions.py --data_dir <data> --factor 3 \
    --arms prod=<ckpt3> detail=<ckpt2> \
    --views "448:0.45,0.78,0.42,0.80:lace_mesh" "96:0.46,0.88,0.38,0.74:sole_tread" \
            "248:0.13,0.58,0.26,0.62:metallic" \
    --out _infusion/wpt2_regions_f3.json --strip_out qa/wpt2_strips_f3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.gsplat_trainer import (  # noqa: E402
    load_colmap_data, load_image, render_splats, get_lpips,
)

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False


def load_splats(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    return {k: v.to(device) for k, v in ck["splats"].items()}


def render(splats, b, sh_degree, device):
    gt = b["image"].to(device)
    K = b["K"].to(device); c2w = b["c2w"].to(device)
    H, W = gt.shape[:2]
    w2c = torch.linalg.inv(c2w)
    r, _, _ = render_splats(splats, w2c, K, W, H, sh_degree,
                            torch.zeros(3, device=device), "antialiased")
    return r[0].clamp(0, 1), gt


def label(img_bgr, text):
    if not _HAVE_CV2:
        return img_bgr
    out = img_bgr.copy()
    cv2.putText(out, text, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(out, text, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--factor", type=int, default=3)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--arms", nargs="+", required=True, help="label=ckpt pairs")
    ap.add_argument("--views", nargs="+", required=True,
                    help="IDX:x0,x1,y0,y1:NAME per-view crop specs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strip_out", default=None)
    args = ap.parse_args()
    device = "cuda"

    data = load_colmap_data(args.data_dir, factor=args.factor, test_every=8, normalize=True)
    lp = get_lpips(device)
    arms = {}
    for spec in args.arms:
        lbl, ck = spec.split("=", 1)
        arms[lbl] = load_splats(ck, device)

    specs = []
    for v in args.views:
        idx_s, box_s, name = v.split(":")
        box = tuple(float(x) for x in box_s.split(","))
        specs.append((int(idx_s), box, name))

    result = dict(data_dir=str(args.data_dir), factor=args.factor,
                  metric="VGG-LPIPS normalize=True; region crop box fractional; "
                         "region_psnr from crop MSE", views={})
    strip_dir = Path(args.strip_out) if args.strip_out else None
    if strip_dir:
        strip_dir.mkdir(parents=True, exist_ok=True)

    for idx, box, name in specs:
        b = load_image(data, idx)
        gt = b["image"].to(device)
        H, W = gt.shape[:2]
        x0, x1 = int(box[0] * W), int(box[1] * W)
        y0, y1 = int(box[2] * H), int(box[3] * H)
        gb = gt.permute(2, 0, 1).unsqueeze(0)
        entry = dict(view_idx=idx, name=name, box=list(box),
                     image=data["image_names"][idx], arms={})
        cols_full = [label((gt.cpu().numpy() * 255).astype(np.uint8)[..., ::-1], f"GT {name}")]
        cols_crop = [(gt.cpu().numpy() * 255).astype(np.uint8)[..., ::-1][y0:y1, x0:x1]]
        for lbl, sp in arms.items():
            with torch.no_grad():
                r, _ = render(sp, b, args.sh_degree, device)
            rb = r.permute(2, 0, 1).unsqueeze(0)
            full = float(lp(rb, gb))
            rc, gc = rb[:, :, y0:y1, x0:x1], gb[:, :, y0:y1, x0:x1]
            reg = float(lp(rc, gc))
            mse = torch.mean((rc - gc) ** 2).item()
            reg_psnr = 10.0 * np.log10(1.0 / (mse + 1e-10))
            entry["arms"][lbl] = dict(full_lpips=full, region_lpips=reg,
                                      region_psnr=reg_psnr)
            rimg = (r.cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
            cols_full.append(label(rimg, lbl))
            cols_crop.append(rimg[y0:y1, x0:x1])
        result["views"][f"{idx}_{name}"] = entry
        line = f"view {idx} {name}: " + " | ".join(
            f"{l} reg-LPIPS {entry['arms'][l]['region_lpips']:.4f}" for l in arms)
        print(line, flush=True)
        if strip_dir and _HAVE_CV2:
            cv2.imwrite(str(strip_dir / f"{name}_view{idx}_full.jpg"),
                        np.concatenate(cols_full, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 92])
            crops2x = [cv2.resize(c, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                       for c in cols_crop]
            cv2.imwrite(str(strip_dir / f"{name}_view{idx}_crop2x.jpg"),
                        np.concatenate(crops2x, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 95])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}" + (f" + strips {strip_dir}" if strip_dir else ""), flush=True)


if __name__ == "__main__":
    main()
