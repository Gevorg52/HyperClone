"""WP-S1 gate evidence: region-crop + full-frame metrics AND zoom strips on the
flat/shiny pain surfaces, for N arms (stack vs ctrl) on held-out views.

Method (mirrors _infusion/wpf1_deck_eval.py, generalized to a configurable box):
render each every-8 held-out view at --factor (black bg, antialiased), clamp to
[0,1]. Metrics per view:
  full_lpips   : torchmetrics VGG-LPIPS (normalize=True) over the whole frame
  region_lpips : same LPIPS on the fractional crop box (x0,x1,y0,y1)
  region_psnr  : PSNR from MSE on the crop (dB)
The LPIPS model is the SAME as the trainer's get_lpips (torchmetrics VGG,
normalize=True), so numbers are directly comparable to the WP-F1/F2 deck evals.

Optionally (--strip_out DIR) writes GT | arm1 | arm2 ... side-by-side strips:
per view a full-frame strip and a 2x-upscaled region-crop strip (INTER_NEAREST,
render_compare.py convention) for visual inspection of the shiny surfaces.

Usage (repo root, e.g. Laptop deck):
  python _infusion/wps1_region_eval.py \
    --data_dir <data> --box 0.30 0.72 0.55 0.95 \
    --arms ctrl=<ckpt> stack=<ckpt> \
    --out _infusion/wps1_laptop_region.json \
    --strip_out qa/wps1_laptop_strips --strip_views 0 4 8 12
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


def _render(splats, b, sh_degree, device):
    gt = b["image"].to(device)
    K = b["K"].to(device)
    c2w = b["c2w"].to(device)
    H, W = gt.shape[:2]
    w2c = torch.linalg.inv(c2w)
    r, _, _ = render_splats(splats, w2c, K, W, H, sh_degree,
                            torch.zeros(3, device=device), "antialiased")
    return r[0].clamp(0, 1), gt


def eval_arm(ckpt, data, box, lp, sh_degree, device):
    splats = load_splats(ckpt, device)
    full, reg_lp, reg_psnr = [], [], []
    with torch.no_grad():
        for idx in sorted(int(x) for x in data["test_indices"]):
            b = load_image(data, idx)
            r, gt = _render(splats, b, sh_degree, device)
            H, W = gt.shape[:2]
            rb = r.permute(2, 0, 1).unsqueeze(0)
            gb = gt.permute(2, 0, 1).unsqueeze(0)
            full.append(float(lp(rb, gb)))
            x0, x1 = int(box[0] * W), int(box[1] * W)
            y0, y1 = int(box[2] * H), int(box[3] * H)
            rc = rb[:, :, y0:y1, x0:x1]
            gc = gb[:, :, y0:y1, x0:x1]
            reg_lp.append(float(lp(rc, gc)))
            mse = torch.mean((rc - gc) ** 2).item()
            reg_psnr.append(10.0 * np.log10(1.0 / (mse + 1e-10)))
    return dict(ckpt=str(ckpt), n_views=len(full),
                full_lpips=float(np.mean(full)),
                region_lpips=float(np.mean(reg_lp)),
                region_psnr=float(np.mean(reg_psnr)))


def label_img(img_bgr, text):
    if not _HAVE_CV2:
        return img_bgr
    out = img_bgr.copy()
    cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_strips(arms, data, box, sh_degree, device, strip_out, strip_views):
    if not _HAVE_CV2:
        print("cv2 unavailable; skipping strips")
        return
    strip_out = Path(strip_out)
    strip_out.mkdir(parents=True, exist_ok=True)
    test = sorted(int(x) for x in data["test_indices"])
    loaded = {lbl: load_splats(ck, device) for lbl, ck in arms.items()}
    for vp in strip_views:
        idx = test[vp % len(test)]
        b = load_image(data, idx)
        # GT
        _, gt = _render(next(iter(loaded.values())), b, sh_degree, device)
        gt_bgr = (gt.cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
        H, W = gt_bgr.shape[:2]
        cols = [label_img(gt_bgr, "GT")]
        for lbl, sp in loaded.items():
            with torch.no_grad():
                r, _ = _render(sp, b, sh_degree, device)
            r_bgr = (r.cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
            cols.append(label_img(r_bgr, lbl))
        strip = np.concatenate(cols, axis=1)
        cv2.imwrite(str(strip_out / f"view{vp}_full.jpg"), strip,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        # 2x region crop
        x0, x1 = int(box[0] * W), int(box[1] * W)
        y0, y1 = int(box[2] * H), int(box[3] * H)
        crops = []
        for c in cols:
            cc = c[y0:y1, x0:x1]
            crops.append(cv2.resize(cc, None, fx=2, fy=2,
                                    interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(str(strip_out / f"view{vp}_region2x.jpg"),
                    np.concatenate(crops, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  strip view{vp}: full + region2x (idx {idx})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="label=ckpt pairs, e.g. ctrl=... stack=...")
    ap.add_argument("--box", nargs=4, type=float, required=True,
                    metavar=("X0", "X1", "Y0", "Y1"),
                    help="fractional crop box x0 x1 y0 y1")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strip_out", default=None)
    ap.add_argument("--strip_views", nargs="*", type=int, default=[0, 4, 8, 12])
    ap.add_argument("--region_name", default="region")
    args = ap.parse_args()
    device = "cuda"

    data = load_colmap_data(args.data_dir, factor=args.factor, test_every=8,
                            normalize=True)
    lp = get_lpips(device)
    box = tuple(args.box)
    arms = {}
    for spec in args.arms:
        lbl, ck = spec.split("=", 1)
        arms[lbl] = ck

    result = dict(
        metric="mean VGG-LPIPS (torchmetrics normalize=True) over every8 held-out "
               "views; region = fractional crop box; region_psnr from crop MSE",
        region_name=args.region_name,
        box_frac=dict(x0=box[0], x1=box[1], y0=box[2], y1=box[3]),
        data_dir=str(args.data_dir), factor=args.factor, arms={})
    for lbl, ck in arms.items():
        r = eval_arm(ck, data, box, lp, args.sh_degree, device)
        result["arms"][lbl] = r
        print(f"{lbl:6s} views={r['n_views']}  full_lpips={r['full_lpips']:.4f}  "
              f"{args.region_name}_lpips={r['region_lpips']:.4f}  "
              f"{args.region_name}_psnr={r['region_psnr']:.3f}", flush=True)

    labels = list(result["arms"])
    if len(labels) == 2:
        a, b = (result["arms"][l] for l in labels)
        result["delta"] = {
            k: dict(abs=b[k] - a[k],
                    pct=(b[k] - a[k]) / a[k] * 100.0 if a[k] else 0.0)
            for k in ("full_lpips", "region_lpips", "region_psnr")}
        print(f"delta {labels[1]} vs {labels[0]}: " + "  ".join(
            f"{k} {result['delta'][k]['abs']:+.4f} ({result['delta'][k]['pct']:+.1f}%)"
            for k in ("full_lpips", "region_lpips", "region_psnr")), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)

    if args.strip_out:
        write_strips(arms, data, box, args.sh_degree, device,
                     args.strip_out, args.strip_views)
        print(f"strips -> {args.strip_out}", flush=True)


if __name__ == "__main__":
    main()
