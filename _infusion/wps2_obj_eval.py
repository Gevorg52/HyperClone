"""WP-S2 object-centric eval: OBJECT-region masked metrics for the 3 arms, all scored
against the SAME full-scene GT object pixels (not the visibility mask).

eval_dir (wps2/eval/data) junctions FULL-SCENE images_2 (GT) + object masks_2 + the
dense sparse (shared cameras). For every held-out view and every arm ckpt we render
(black bg, antialiased), then compute the WP-Q1 masked metrics
(core.qa.masked_metrics, VGG-LPIPS normalize=True) on:
  - OBJECT region  = (object_mask > 0.5)   -> the gate metric (render vs full-scene GT)
  - BG region      = (object_mask <= 0.5)  -> artifact check (meaningful for arm a only,
                                              which is the only arm modelling background)
plus full-frame LPIPS/PSNR. All arms use the identical held-out views + GT + mask, so
the object-region numbers are directly comparable across arms.

Usage:
  python _infusion/wps2_obj_eval.py --eval_dir <eval/data> \
    --arms a_full=<ckpt> b_obj=<ckpt> c_obj_alpha=<ckpt> \
    --out _infusion/wps2_obj_metrics.json \
    --strip_out qa/wps2_strips --strip_views 0 6 12 18 24
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
from core.qa.masked_metrics import masked_psnr, masked_ssim, masked_lpips  # noqa: E402

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False


def load_splats(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    sp = ck["splats"] if isinstance(ck, dict) and "splats" in ck else ck
    return {k: v.to(device).float() for k, v in sp.items()}


def _render(splats, b, sh_degree, device):
    K = b["K"].to(device)
    c2w = b["c2w"].to(device)
    gt = b["image"].to(device)
    H, W = gt.shape[:2]
    w2c = torch.linalg.inv(c2w)
    r, _, _ = render_splats(splats, w2c, K, W, H, sh_degree,
                            torch.zeros(3, device=device), "antialiased")
    return r[0].clamp(0, 1), gt


def _obj_lpips(lp, r, gt, om, device, min_size=64):
    """Object-region LPIPS: pre-multiply BOTH render and GT by the object mask (black
    outside), crop the object's bounding box, then a single LPIPS forward. Pre-multiply
    makes the excluded background identical (black) in both arms' inputs, so full-scene
    (real bg in the bbox corners) and object-only (black bg) are compared FAIRLY on the
    object pixels only. One forward per view (fast; tiled-over-bg was pathological)."""
    m3 = om.float().unsqueeze(-1)
    rp = (r * m3)
    gp = (gt * m3)
    ys, xs = torch.where(om)
    if len(ys) == 0:
        return float("nan")
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    H, W = om.shape
    # expand to >= min_size within bounds
    if y1 - y0 < min_size:
        c = (y0 + y1) // 2; y0 = max(0, c - min_size // 2); y1 = min(H, y0 + min_size); y0 = max(0, y1 - min_size)
    if x1 - x0 < min_size:
        c = (x0 + x1) // 2; x0 = max(0, c - min_size // 2); x1 = min(W, x0 + min_size); x0 = max(0, x1 - min_size)
    rc = rp[y0:y1, x0:x1].permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    gc = gp[y0:y1, x0:x1].permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    return float(lp(rc, gc))


def eval_arm(ckpt, data, sh_degree, device):
    splats = load_splats(ckpt, device)
    lp = get_lpips(device)
    obj = dict(lpips=[], psnr=[], ssim=[])
    bg = dict(psnr=[], ssim=[])
    full_lpips, full_psnr, obj_frac = [], [], []
    with torch.no_grad():
        for idx in sorted(int(x) for x in data["test_indices"]):
            b = load_image(data, idx)
            if "mask" not in b:
                raise SystemExit(f"view {idx} has no object mask (eval_dir needs masks_2/)")
            r, gt = _render(splats, b, sh_degree, device)
            m = b["mask"].to(device)
            om = m > 0.5                    # object region (bool HxW)
            bm = ~om                        # background region
            obj_frac.append(float(om.float().mean()))
            # object region (mask-premultiplied bbox LPIPS; pixelwise PSNR/SSIM over mask)
            obj["psnr"].append(masked_psnr(r, gt, om))
            obj["ssim"].append(masked_ssim(r, gt, om))
            obj["lpips"].append(_obj_lpips(lp, r, gt, om, device))
            # background region artifact check (pixelwise; interpret for arm a — only arm
            # modelling bg. b/c bg = black-vs-realbg by construction, expected poor)
            bg["psnr"].append(masked_psnr(r, gt, bm))
            bg["ssim"].append(masked_ssim(r, gt, bm))
            # full frame
            rb = r.permute(2, 0, 1).unsqueeze(0)
            gb = gt.permute(2, 0, 1).unsqueeze(0)
            full_lpips.append(float(lp(rb, gb)))
            mse = float(((r - gt) ** 2).mean())
            full_psnr.append(10.0 * np.log10(1.0 / (mse + 1e-10)))
    nm = lambda a: float(np.nanmean(a))
    return dict(
        ckpt=str(ckpt), n_views=len(full_psnr), n_gaussians=int(splats["means"].shape[0]),
        obj_mean_frac=nm(obj_frac),
        obj_lpips=nm(obj["lpips"]), obj_psnr=nm(obj["psnr"]), obj_ssim=nm(obj["ssim"]),
        bg_psnr=nm(bg["psnr"]), bg_ssim=nm(bg["ssim"]),
        full_lpips=nm(full_lpips), full_psnr=nm(full_psnr))


def _label(img_bgr, text):
    if not _HAVE_CV2:
        return img_bgr
    o = img_bgr.copy()
    cv2.putText(o, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(o, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return o


def write_strips(arms, data, sh_degree, device, strip_out, strip_views):
    if not _HAVE_CV2:
        print("cv2 unavailable; skipping strips"); return
    strip_out = Path(strip_out); strip_out.mkdir(parents=True, exist_ok=True)
    test = sorted(int(x) for x in data["test_indices"])
    loaded = {lbl: load_splats(ck, device) for lbl, ck in arms.items()}
    for vp in strip_views:
        idx = test[vp % len(test)]
        b = load_image(data, idx)
        _, gt = _render(next(iter(loaded.values())), b, sh_degree, device)
        gt_bgr = (gt.cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
        H, W = gt_bgr.shape[:2]
        m = (b["mask"].cpu().numpy() > 0.5)
        ys, xs = np.where(m)
        if len(ys):
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        else:
            y0, y1, x0, x1 = 0, H, 0, W
        cols = [_label(gt_bgr, "GT")]
        for lbl, sp in loaded.items():
            with torch.no_grad():
                r, _ = _render(sp, b, sh_degree, device)
            cols.append(_label((r.cpu().numpy() * 255).astype(np.uint8)[..., ::-1], lbl))
        cv2.imwrite(str(strip_out / f"view{vp}_full.jpg"),
                    np.concatenate(cols, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 92])
        crops = [cv2.resize(c[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                 for c in cols]
        cv2.imwrite(str(strip_out / f"view{vp}_objcrop2x.jpg"),
                    np.concatenate(crops, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  strip view{vp} (idx {idx}, obj {m.mean()*100:.1f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="label=ckpt ...")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strip_out", default=None)
    ap.add_argument("--strip_views", nargs="*", type=int, default=[0, 6, 12, 18, 24])
    args = ap.parse_args()
    device = "cuda"

    data = load_colmap_data(args.eval_dir, factor=args.factor, test_every=8, normalize=True)
    arms = {}
    for spec in args.arms:
        lbl, ck = spec.split("=", 1)
        arms[lbl] = ck

    result = dict(
        metric="OBJECT-region masked metrics (WP-Q1 core.qa.masked_metrics, tiled "
               "VGG-LPIPS normalize=True) on full-scene GT; obj=(mask>0.5), "
               "bg=(mask<=0.5); render black-bg antialiased on held-out every-8 views",
        eval_dir=str(args.eval_dir), factor=args.factor, arms={})
    for lbl, ck in arms.items():
        r = eval_arm(ck, data, args.sh_degree, device)
        result["arms"][lbl] = r
        print(f"{lbl:12s} views={r['n_views']} nGS={r['n_gaussians']:,} "
              f"objfrac={r['obj_mean_frac']*100:.1f}% | OBJ lpips={r['obj_lpips']:.4f} "
              f"psnr={r['obj_psnr']:.3f} ssim={r['obj_ssim']:.4f} | BG psnr={r['bg_psnr']:.3f} "
              f"ssim={r['bg_ssim']:.4f} | FULL lpips={r['full_lpips']:.4f} psnr={r['full_psnr']:.3f}",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    if args.strip_out:
        write_strips(arms, data, args.sh_degree, device, args.strip_out, args.strip_views)
        print(f"strips -> {args.strip_out}", flush=True)


if __name__ == "__main__":
    main()
