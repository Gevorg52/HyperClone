"""Side-by-side held-out renders: GT | base | v1 | v2, full frame + deck crop.

Lets you SEE what the distill transferred into each PLY (vs the captured GT).
Writes PNGs to _dataset/Laptop/compare/.

Usage:  python render_compare.py <base_ckpt> <v1_ckpt> <v2_ckpt>

SA-GS eps2d gate (WP-Q2b) — one ckpt, OFF vs ON the anti-alias rule at a zoom:
  python render_compare.py sags zoom=2 <label=ckpt> data=<dir> out=<dir>
  python render_compare.py sags zoom=0.5 eval_ss=4 <label=ckpt> data=<dir> out=<dir>
Zoom scales the camera focal length (>1 in, <1 out) about the principal point.
"ON" passes eps2d=compute_eps2d(...) (0.3*r^2); for zoom-out, eval_ss>1 renders
at K x res with K-scaled intrinsics + eps2d=0.3*(rK)^2 and box-downsamples.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from core.training.gsplat_trainer import (
    load_colmap_data, load_image, render_splats, compute_eps2d,
)

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\train\data"
OUT = Path(r"F:\HYPERclone\apex-sfm\_dataset\Laptop\compare")
SH = 3
FACTOR = 2
device = "cuda"
# held-out views to render (indices into the test split) + a deck crop (x0,x1,y0,y1 frac)
VIEW_POS = [0, 12, 24]
DECK = (0.30, 0.72, 0.55, 0.95)


def load_splats(p):
    ck = torch.load(p, weights_only=True, map_location=device)
    return {k: v.to(device) for k, v in ck["splats"].items()}


def render(splats, b, zoom=1.0, eps2d=None, ss=1):
    """Render a held-out view. zoom scales the focal length about the principal
    point (>1 zoom in, <1 zoom out). eps2d overrides gsplat's 2D dilation
    (None = default 0.3). ss>1 supersamples at K x res then box-downsamples."""
    gt = b["image"].to(device); K = b["K"].to(device); c2w = b["c2w"].to(device)
    H, W = gt.shape[:2]
    Kz = K.clone()
    Kz[0, 0] *= zoom; Kz[1, 1] *= zoom                 # zoom = focal scaling
    if ss > 1:
        Kr = Kz.clone()
        Kr[0, 0] *= ss; Kr[1, 1] *= ss; Kr[0, 2] *= ss; Kr[1, 2] *= ss
        Wr, Hr = W * ss, H * ss
    else:
        Kr, Wr, Hr = Kz, W, H
    with torch.no_grad():
        r, _, _ = render_splats(splats, torch.linalg.inv(c2w), Kr, Wr, Hr, SH,
                                torch.zeros(3, device=device), "antialiased",
                                eps2d=eps2d)
        img = r[0].clamp(0, 1)                          # (Hr, Wr, 3)
        if ss > 1:
            img = F.avg_pool2d(img.permute(2, 0, 1).unsqueeze(0), ss)[0].permute(1, 2, 0)
    return (img.cpu().numpy() * 255).astype(np.uint8)[..., ::-1]  # ->BGR


def label(img, txt):
    img = img.copy()
    cv2.putText(img, txt, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    return img


def sags_gate(models, data, test, zoom, eval_ss, factor):
    """WP-Q2b gate: for each ckpt at a given `zoom`, render OFF (default eps2d
    0.3) vs ON (SA-GS rule eps2d=0.3*r^2, + eval_ss supersample for zoom-out)
    and write a side-by-side [OFF | ON] strip per view. No GT column (there is
    no captured ground truth at an arbitrary zoom)."""
    ztag = f"z{zoom:g}".replace(".", "p")
    ss = eval_ss if zoom < 1.0 else 1        # supersample only helps zoom-OUT
    for vp in VIEW_POS:
        idx = test[vp % len(test)]
        b = load_image(data, idx)
        H, W = b["image"].shape[:2]
        # SA-GS ratio: same-FOV output res == train res, so normalized focal
        # ratio == zoom; the ss-supersampled raster runs at ss*W (rasterization
        # res), giving eps2d = 0.3*(zoom*ss)^2. dist ratio unknown -> 1.0.
        eps = compute_eps2d(train_res=W, render_res=W * ss,
                            train_f=1.0, render_f=zoom)
        cols = []
        for name, sp in models.items():
            off = render(sp, b, zoom=zoom, eps2d=None, ss=1)          # default 0.3
            on = render(sp, b, zoom=zoom, eps2d=eps, ss=ss)           # SA-GS rule
            on_lbl = f"{name} ON eps{eps:.2f}" + (f" ss{ss}" if ss > 1 else "")
            cols.append(label(off, f"{name} OFF eps0.30"))
            cols.append(label(on, on_lbl))
        strip = np.concatenate(cols, axis=1)
        out_path = OUT / f"sags_{ztag}_view{vp}.jpg"
        cv2.imwrite(str(out_path), strip, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"wrote {out_path.name}  (zoom={zoom} ss={ss} eps2d={eps:.3f})", flush=True)
    print(f"\nOutput: {OUT}")


def main():
    # accept "label=ckpt" pairs (e.g. base=... v2=... v3=...); optional data=<dir>, out=<dir>
    global DATA, OUT
    args = []
    sags = False
    zoom, eval_ss = 1.0, 1
    for a in sys.argv[1:]:
        if a.startswith("data="):
            DATA = a.split("=", 1)[1]
        elif a.startswith("out="):
            OUT = Path(a.split("=", 1)[1])
        elif a == "sags":
            sags = True
        elif a.startswith("zoom="):
            zoom = float(a.split("=", 1)[1])
        elif a.startswith("eval_ss="):
            eval_ss = int(a.split("=", 1)[1])
        else:
            args.append(a)
    sys.argv = [sys.argv[0]] + args
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)
    test = list(data["test_indices"])
    models = {}
    for arg in sys.argv[1:]:
        lbl, path = arg.split("=", 1)
        models[lbl] = load_splats(path)

    if sags:
        sags_gate(models, data, test, zoom, eval_ss, FACTOR)
        return

    for vp in VIEW_POS:
        idx = test[vp % len(test)]
        b = load_image(data, idx)
        gt = (b["image"].cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
        cols = [label(gt, "GT")]
        for name, sp in models.items():
            cols.append(label(render(sp, b), name))
        strip = np.concatenate(cols, axis=1)
        cv2.imwrite(str(OUT / f"view{vp}_full.jpg"), strip,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        # deck crop, upscaled 2x for inspection
        H, W = gt.shape[:2]
        x0, x1, y0, y1 = DECK
        crops = []
        for c in cols:
            cc = c[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
            crops.append(cv2.resize(cc, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(str(OUT / f"view{vp}_deck.jpg"), np.concatenate(crops, axis=1),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"wrote view{vp}_full.jpg + view{vp}_deck.jpg", flush=True)
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
