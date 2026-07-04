"""WP-G1 stage-2: apples-to-apples specular-region A/B between 3DGS-DR (pre-rendered
PNGs) and the WP-S1 STACK (rendered live from its gsplat ckpt), BOTH scored against
the SAME canonical held-out GT (our undistorted load_image frames), with the SAME
LPIPS (torchmetrics VGG normalize=True) and the SAME region box as WP-S1
(wps1_region_eval.py). This is the gate evaluator.

Held-out set parity: 3DGS-DR sorts by image_name and holds out idx%8==0; our
load_colmap_data(test_every=8) does the same on the same registered set -> identical
test views. We match DR renders to GT by image stem.

Metrics per held-out view, per arm:
  full_lpips   : VGG-LPIPS over the whole frame
  region_lpips : VGG-LPIPS on the fractional crop box (the specular surface) [GATE]
  region_psnr  : PSNR from crop MSE
  full_psnr    : PSNR whole frame
  obj_psnr     : PSNR over the GT-object mask (luminance>thr) — black-bg-fair global

Usage:
  python _infusion/wpg1_eval.py \
    --data_dir _dataset/GPU_V4/.../wps1/stack/data \
    --stack_ckpt .../wps1/stack/output/ckpts/ckpt_29999_rank0.pt \
    --dr_render_dir .../wpg1/3dgsdr_render \
    --box 0.28 0.72 0.30 0.55 --region_name shiny \
    --out qa/wpg1_gpu/scorecard.json \
    --strip_out qa/wpg1_gpu_strips --strip_views 0 3 6 --factor 2
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


def render_stack(splats, b, sh_degree, device):
    K = b["K"].to(device)
    c2w = b["c2w"].to(device)
    gt = b["image"].to(device)
    H, W = gt.shape[:2]
    w2c = torch.linalg.inv(c2w)
    r, _, _ = render_splats(splats, w2c, K, W, H, sh_degree,
                            torch.zeros(3, device=device), "antialiased")
    return r[0].clamp(0, 1)  # (H,W,3)


def load_dr(dr_dir, stem, H, W, device):
    p = Path(dr_dir) / (stem + ".png")
    if not p.exists():
        raise FileNotFoundError(f"DR render missing: {p}")
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)[..., ::-1]  # BGR->RGB
    if im.shape[0] != H or im.shape[1] != W:
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(im)).float().to(device) / 255.0


def metrics(pred, gt, lp, box, obj_thr=0.02):
    # pred,gt: (H,W,3) [0,1] on cuda
    H, W = gt.shape[:2]
    pb = pred.permute(2, 0, 1).unsqueeze(0)
    gb = gt.permute(2, 0, 1).unsqueeze(0)
    x0, x1 = int(box[0] * W), int(box[1] * W)
    y0, y1 = int(box[2] * H), int(box[3] * H)
    rc, gc = pb[:, :, y0:y1, x0:x1], gb[:, :, y0:y1, x0:x1]
    full_mse = torch.mean((pb - gb) ** 2).item()
    reg_mse = torch.mean((rc - gc) ** 2).item()
    lum = gt.mean(dim=-1)
    om = (lum > obj_thr)
    if om.sum() > 0:
        obj_mse = (((pred - gt) ** 2).mean(dim=-1)[om]).mean().item()
    else:
        obj_mse = full_mse
    return {
        "full_lpips": float(lp(pb, gb)),
        "region_lpips": float(lp(rc, gc)),
        "region_psnr": 10.0 * np.log10(1.0 / (reg_mse + 1e-10)),
        "full_psnr": 10.0 * np.log10(1.0 / (full_mse + 1e-10)),
        "obj_psnr": 10.0 * np.log10(1.0 / (obj_mse + 1e-10)),
    }


def label(img_bgr, text):
    out = img_bgr.copy()
    cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--stack_ckpt", required=True)
    ap.add_argument("--dr_render_dir", required=True)
    ap.add_argument("--box", nargs=4, type=float, required=True)
    ap.add_argument("--region_name", default="region")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strip_out", default=None)
    ap.add_argument("--strip_views", nargs="*", type=int, default=[0, 3, 6])
    args = ap.parse_args()
    device = "cuda"
    box = tuple(args.box)

    data = load_colmap_data(args.data_dir, factor=args.factor, test_every=8, normalize=True)
    lp = get_lpips(device)
    stack = load_splats(args.stack_ckpt, device)

    test = sorted(int(x) for x in data["test_indices"])
    acc = {"stack": [], "dr": []}
    per_view = []
    with torch.no_grad():
        for idx in test:
            b = load_image(data, idx)
            gt = b["image"].to(device)
            H, W = gt.shape[:2]
            stem = Path(data["image_names"][idx]).stem
            sr = render_stack(stack, b, args.sh_degree, device)
            dr = load_dr(args.dr_render_dir, stem, H, W, device)
            ms = metrics(sr, gt, lp, box)
            md = metrics(dr, gt, lp, box)
            acc["stack"].append(ms)
            acc["dr"].append(md)
            per_view.append({"idx": idx, "stem": stem, "stack": ms, "dr": md})

    def mean(arm):
        keys = acc[arm][0].keys()
        return {k: float(np.mean([m[k] for m in acc[arm]])) for k in keys}

    res = {
        "metric": "VGG-LPIPS (torchmetrics normalize=True); region=specular crop box; "
                  "both arms vs canonical undistorted load_image GT",
        "box_frac": {"x0": box[0], "x1": box[1], "y0": box[2], "y1": box[3]},
        "region_name": args.region_name,
        "data_dir": str(args.data_dir), "stack_ckpt": str(args.stack_ckpt),
        "dr_render_dir": str(args.dr_render_dir), "factor": args.factor,
        "n_views": len(test),
        "stack": mean("stack"), "dr": mean("dr"),
    }
    res["delta_dr_minus_stack"] = {
        k: {"abs": res["dr"][k] - res["stack"][k],
            "pct": (res["dr"][k] - res["stack"][k]) / res["stack"][k] * 100.0
                   if res["stack"][k] else 0.0}
        for k in res["stack"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({**res, "per_view": per_view}, indent=2),
                              encoding="utf-8")

    def line(arm):
        m = res[arm]
        return (f"{arm:6s} region_lpips={m['region_lpips']:.4f} "
                f"full_lpips={m['full_lpips']:.4f} region_psnr={m['region_psnr']:.3f} "
                f"full_psnr={m['full_psnr']:.3f} obj_psnr={m['obj_psnr']:.3f}")
    print(line("stack"), flush=True)
    print(line("dr"), flush=True)
    d = res["delta_dr_minus_stack"]
    print(f"delta dr-stack: region_lpips {d['region_lpips']['abs']:+.4f} "
          f"({d['region_lpips']['pct']:+.1f}%)  full_lpips {d['full_lpips']['abs']:+.4f} "
          f"({d['full_lpips']['pct']:+.1f}%)", flush=True)
    print(f"wrote {args.out}", flush=True)

    # strips
    if args.strip_out and _HAVE_CV2:
        so = Path(args.strip_out); so.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for vp in args.strip_views:
                idx = test[vp % len(test)]
                b = load_image(data, idx)
                gt = b["image"].to(device); H, W = gt.shape[:2]
                stem = Path(data["image_names"][idx]).stem
                sr = render_stack(stack, b, args.sh_degree, device)
                dr = load_dr(args.dr_render_dir, stem, H, W, device)
                cols_t = [gt, sr, dr]; labels = ["GT", "wps1-stack", "3dgs-dr"]
                bgr = [(t.cpu().numpy() * 255).astype(np.uint8)[..., ::-1] for t in cols_t]
                full = np.concatenate([label(im, l) for im, l in zip(bgr, labels)], axis=1)
                cv2.imwrite(str(so / f"view{vp}_full.jpg"), full, [cv2.IMWRITE_JPEG_QUALITY, 95])
                x0, x1 = int(box[0] * W), int(box[1] * W)
                y0, y1 = int(box[2] * H), int(box[3] * H)
                crops = [cv2.resize(im[y0:y1, x0:x1], None, fx=2, fy=2,
                                    interpolation=cv2.INTER_NEAREST) for im in bgr]
                cv2.imwrite(str(so / f"view{vp}_region2x.jpg"),
                            np.concatenate(crops, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"  strip view{vp} (idx {idx}, {stem})", flush=True)
        print(f"strips -> {so}", flush=True)


if __name__ == "__main__":
    main()
