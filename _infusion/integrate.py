"""Inject InFusion's filled points into the holed splat, re-optimize, compare.

Pipeline: holed splat + InFusion mask.ply -> new Gaussians (color from ply,
small init scales/opacity) -> render test view BEFORE opt -> optimize ONLY the
new Gaussians on training views (frozen background) -> render AFTER. Compare
both to the original FULL splat (the ground truth for the deleted geometry) and
to the real GT photo. 4-panel: FULL(GT) | HOLED | FILLED-raw | FILLED-opt.

Usage: python integrate.py [steps]
"""
import math
import os
import sys

import numpy as np
import torch
import cv2

from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats, ssim

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\output\ckpts\ckpt_29999_rank0.pt"
IO2 = r"F:\HYPERclone\apex-sfm\_infusion\io2"
SH, dev = 3, "cuda"
IDX, FACTOR = 256, 4
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 800
C0 = 0.28209479177387814             # SH DC factor


def render(sp, c2w, K, W, H):
    r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH,
                            torch.zeros(3, device=dev), "antialiased")
    return r[0].clamp(0, 1)


def metrics(x, ref):
    p = 10 * math.log10(1.0 / (torch.mean((x - ref) ** 2).item() + 1e-9))
    s = ssim(x.permute(2, 0, 1)[None], ref.permute(2, 0, 1)[None]).item()
    return p, s


full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
holed = {k: v.to(dev) for k, v in torch.load(os.path.join(IO2, "holed.pt"), weights_only=True)["splats"].items()}

# Load InFusion filled points (ascii ply: 10 header lines, then x y z r g b)
raw = np.loadtxt(os.path.join(IO2, "out", "rgb_mask.ply"), skiprows=10)
pts = torch.from_numpy(raw[:, :3]).float().to(dev)
col = torch.from_numpy(raw[:, 3:6]).float().to(dev) / 255.0       # RGB [0,1]
# subsample to keep it light (one depth sheet is dense)
if pts.shape[0] > 120000:
    sub = torch.randperm(pts.shape[0], device=dev)[:120000]
    pts, col = pts[sub], col[sub]
M = pts.shape[0]
print(f"InFusion filled points: {M:,}")

vs = 0.02 * 1.772
new = {
    "means": pts.clone(),
    "scales": torch.full((M, 3), math.log(vs * 0.5), device=dev),
    "quats": torch.tensor([1.0, 0, 0, 0], device=dev).repeat(M, 1),
    "opacities": torch.full((holed["opacities"].shape[1:] and (M,) + tuple(holed["opacities"].shape[1:])
                             or (M,)), -2.197, device=dev),
    "sh0": ((col - 0.5) / C0).reshape(M, 1, 3),
    "shN": torch.zeros((M,) + tuple(holed["shN"].shape[1:]), device=dev),
}
# match opacity rank to holed
if holed["opacities"].dim() == 2:
    new["opacities"] = new["opacities"].reshape(M, 1)

data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)
b = load_image(data, IDX); gt_photo = b["image"].to(dev)
Kt = b["K"].to(dev); c2wt = b["c2w"].to(dev); H, W = gt_photo.shape[:2]

rgb_full = render(full, c2wt, Kt, W, H).detach()
rgb_holed = render(holed, c2wt, Kt, W, H).detach()


def combine(trainable):
    return {k: torch.cat([holed[k], trainable[k]], 0) for k in holed}


filled_raw = render(combine(new), c2wt, Kt, W, H).detach()

# Optimize ONLY the new Gaussians on training views (frozen background)
params = {k: torch.nn.Parameter(v.clone()) for k, v in new.items()}
opt = torch.optim.Adam([
    {"params": [params["means"]], "lr": 1.6e-4 * 1.772},
    {"params": [params["scales"]], "lr": 5e-3},
    {"params": [params["quats"]], "lr": 1e-3},
    {"params": [params["opacities"]], "lr": 5e-2},
    {"params": [params["sh0"]], "lr": 2.5e-3},
    {"params": [params["shN"]], "lr": 1.25e-4},
])
train_idx = list(data["train_indices"])
g = torch.Generator().manual_seed(0)
for step in range(STEPS):
    j = train_idx[int(torch.randint(len(train_idx), (1,), generator=g))]
    bt = load_image(data, j)
    gt = bt["image"].to(dev); K = bt["K"].to(dev); c2w = bt["c2w"].to(dev)
    h, w = gt.shape[:2]
    r = render_splats(combine(params), torch.linalg.inv(c2w), K, w, h, SH,
                      torch.zeros(3, device=dev), "antialiased")[0][0].clamp(0, 1)
    l1 = (r - gt).abs().mean()
    s = 1 - ssim(r.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
    loss = 0.8 * l1 + 0.2 * s
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0:
        print(f"  step {step:4d}  loss {loss.item():.4f}")

filled_opt = render(combine(params), c2wt, Kt, W, H).detach()

print("\n            vs FULL-splat (deleted-geom GT)   vs REAL photo")
for name, img in [("holed     ", rgb_holed), ("filled-raw", filled_raw), ("filled-opt", filled_opt)]:
    pf, sf = metrics(img, rgb_full); pr, sr = metrics(img, gt_photo)
    print(f"{name}   PSNR {pf:6.2f}  SSIM {sf:.4f}      PSNR {pr:6.2f}  SSIM {sr:.4f}")
pf0, sf0 = metrics(rgb_full, gt_photo)
print(f"{'FULL (ref) ':12s} (=GT)                     PSNR {pf0:6.2f}  SSIM {sf0:.4f}")

panel = torch.cat([rgb_full, rgb_holed, filled_raw, filled_opt], 1).cpu().numpy()
cv2.imwrite(os.path.join(IO2, "integrate.png"),
            cv2.cvtColor((panel * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
print("viz:", os.path.join(IO2, "integrate.png"))
