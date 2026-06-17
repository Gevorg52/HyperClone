"""Hole detector v2 — the signals that actually catch wrong-geometry defects.

v1 finding: alpha-deficit misses the deleted region because the gap is filled by
bleed-through (high alpha, WRONG content), not transparency. Two grounded signals
fix this:
  A. RENDER-vs-PHOTO error  |render - real photo|  -> defect MAGNITUDE (grounded).
  B. WARP consistency (WarpRF-lite): warp neighbour PHOTOS into this view using the
     splat's depth. Where geometry is WRONG (hole bleed-through -> wrong depth) the
     warp misaligns -> high disagreement. Where geometry is correct-but-soft (Type 4,
     capture-limited) the warp ALIGNS -> low disagreement. This is the discriminator
     between a fixable hole and an unfixable capture limit.

Validation (known answer): both signals must spike on the deleted region's image
footprint and stay low on the intact, well-reconstructed surface.

Usage: python hole_detect2.py
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\output\ckpts\ckpt_29999_rank0.pt"
IO2 = r"F:\HYPERclone\apex-sfm\_infusion\io2"
OUT = r"F:\HYPERclone\apex-sfm\_infusion\detect"
SH, dev = 3, "cuda"
FACTOR, A = 4, 256
os.makedirs(OUT, exist_ok=True)


def render(sp, c2w, K, W, H, mode="RGB+ED"):
    r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH,
                            torch.zeros(3, device=dev), "antialiased", render_mode=mode)
    return r[0], a[0, ..., 0]


def warp_photo(photoB, dA, KA, c2wA, KB, c2wB):
    """Warp B's photo into view A using A's depth -> (H,W,3) sampled, (H,W) valid."""
    H, W = dA.shape
    vv, uu = torch.meshgrid(torch.arange(H, device=dev).float(),
                            torch.arange(W, device=dev).float(), indexing="ij")
    cam = torch.stack([(uu - KA[0, 2]) / KA[0, 0] * dA,
                       (vv - KA[1, 2]) / KA[1, 1] * dA, dA, torch.ones_like(dA)], -1)
    world = cam @ c2wA.T                                   # (H,W,4)
    camB = world @ torch.linalg.inv(c2wB).T
    zB = camB[..., 2].clamp(min=1e-6)
    uB = (camB[..., 0] / zB) * KB[0, 0] + KB[0, 2]
    vB = (camB[..., 1] / zB) * KB[1, 1] + KB[1, 2]
    gx = uB / (W - 1) * 2 - 1; gy = vB / (H - 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)[None]
    samp = F.grid_sample(photoB.permute(2, 0, 1)[None], grid, align_corners=True,
                         mode="bilinear", padding_mode="zeros")[0].permute(1, 2, 0)
    valid = (camB[..., 2] > 0) & (gx.abs() <= 1) & (gy.abs() <= 1)
    return samp, valid


full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
holed = {k: v.to(dev) for k, v in torch.load(os.path.join(IO2, "holed.pt"), weights_only=True)["splats"].items()}
data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)

bA = load_image(data, A); photoA = bA["image"].to(dev)
KA = bA["K"].to(dev); c2wA = bA["c2w"].to(dev); H, W = photoA.shape[:2]

# neighbours by camera-centre distance
C = torch.tensor(np.stack([data["camtoworlds"][i][:3, 3] for i in range(len(data["camtoworlds"]))]),
                 device=dev).float()
d = (C - C[A]).norm(dim=1); d[A] = 1e9
nbrs = torch.argsort(d)[:4].tolist()
print(f"view {A}, neighbours {nbrs}")

# defect footprint (known answer): where holed render differs from full render
rgbF, _ = render(full, c2wA, KA, W, H, "RGB")
rgbH, _ = render(holed, c2wA, KA, W, H, "RGB+ED")
depthH = rgbH[..., 3]
footprint = ((rgbF[..., :3] - rgbH[..., :3]).abs().mean(-1) > 0.05).cpu().numpy()

# A. render-vs-photo error (the splat we actually have = holed)
errH = (rgbH[..., :3].clamp(0, 1) - photoA).abs().mean(-1)

# B. warp consistency using holed-splat depth
dis = []
for B in nbrs:
    bB = load_image(data, B); photoB = bB["image"].to(dev)
    KB = bB["K"].to(dev); c2wB = bB["c2w"].to(dev)
    samp, valid = warp_photo(photoB, depthH, KA, c2wA, KB, c2wB)
    dd = (samp - photoA).abs().mean(-1)
    dd[~valid] = float("nan")
    dis.append(dd)
warp = torch.nanmedian(torch.stack(dis), dim=0).values
warp = torch.nan_to_num(warp, nan=0.0)

obj = depthH.cpu().numpy() > 1e-6
fp = footprint & obj
good = obj & ~footprint
e = errH.detach().cpu().numpy(); wv = warp.detach().cpu().numpy()
print("\n            mean on DELETED-footprint   mean on GOOD surface   ratio")
print(f"render-vs-photo err   {e[fp].mean():.4f}                {e[good].mean():.4f}"
      f"             {e[fp].mean()/max(e[good].mean(),1e-6):.1f}x")
print(f"warp inconsistency    {wv[fp].mean():.4f}                {wv[good].mean():.4f}"
      f"             {wv[fp].mean()/max(wv[good].mean(),1e-6):.1f}x")

# combined defect score, gated to object
defect = (e * wv)
defect[~obj] = 0
thr = np.percentile(defect[good], 95)            # threshold from good-surface stats
flagged = defect > thr
recall = flagged[fp].mean()
fpr = flagged[good].mean()
print(f"\ncombined defect = err*warp,  threshold = 95th pct of GOOD surface ({thr:.4f})")
print(f"  recall on deleted footprint: {100*recall:.1f}%   false-positive on good: {100*fpr:.1f}%")


def heat(x, m=None):
    v = x / (np.percentile(x[obj], 99) + 1e-6)
    img = cv2.applyColorMap((np.clip(v, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    if m is not None:
        img[~m] = 30
    return img


pa = (photoA.cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
ph = (rgbH[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
row = np.concatenate([pa, ph, heat(e, obj), heat(wv, obj), heat(defect, obj)], 1)
for i, t in enumerate(["PHOTO", "HOLED render", "err", "warp-inconsist", "defect=err*warp"]):
    cv2.putText(row, t, (i * W + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.imwrite(os.path.join(OUT, "3_warp.png"), row)
print(f"viz: {OUT}\\3_warp.png")
