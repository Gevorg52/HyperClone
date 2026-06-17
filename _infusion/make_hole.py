"""Carve a GENUINE hole in the splat, then emit InFusion inputs for it.

Back-project the (observed) hole pixels of the test view to 3D, take their
inflated bbox, and DELETE every Gaussian inside it -> a real geometry gap.
The original full splat is the ground truth we'll compare the fill against.
Render full vs holed (test view) and write the holed disparity + camera files
for InFusion (reusing the real RGB to isolate geometry/integration quality).

Usage: python make_hole.py
"""
import os
import math

import numpy as np
import torch
import cv2

from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\output\ckpts\ckpt_29999_rank0.pt"
IO = r"F:\HYPERclone\apex-sfm\_infusion\io"        # has true_depth/alpha/mask/intri/c2w/rgb
IO2 = r"F:\HYPERclone\apex-sfm\_infusion\io2"
SH, dev = 3, "cuda"
IDX, FACTOR = 256, 4
os.makedirs(IO2, exist_ok=True)


def render_view(splats, c2w, K, W, H):
    with torch.no_grad():
        r, a, _ = render_splats(splats, torch.linalg.inv(c2w), K, W, H, SH,
                                torch.zeros(3, device=dev), "antialiased", render_mode="RGB+ED")
    return r[0, ..., :3].clamp(0, 1), r[0, ..., 3], a[0, ..., 0]


def psnr(x, y):
    return 10 * math.log10(1.0 / (torch.mean((x - y) ** 2).item() + 1e-9))


s = torch.load(CK, weights_only=True, map_location=dev)["splats"]
splats = {k: v.to(dev) for k, v in s.items()}
data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)
b = load_image(data, IDX)
gt_photo = b["image"].to(dev); K = b["K"].to(dev); c2w = b["c2w"].to(dev)
H, W = gt_photo.shape[:2]

true_d = np.load(os.path.join(IO, "true_depth.npy"))
alpha = np.load(os.path.join(IO, "alpha.npy"))
# Smaller, localized hole (20% box) -> a realistic missing surface PATCH
mask = np.zeros((H, W), np.uint8)
bw, bh = int(0.20 * W), int(0.20 * H)
x0, y0 = (W - bw) // 2, (H - bh) // 2
mask[y0:y0 + bh, x0:x0 + bw] = 255
cv2.imwrite(os.path.join(IO2, "mask.png"), mask)
mask = mask > 127
sel = mask & (alpha > 0.5)

# Back-project hole-surface pixels -> 3D (OpenCV pinhole, world = c2w @ cam)
Kn = K.cpu().numpy(); c2wn = c2w.cpu().numpy()
ys, xs = np.nonzero(sel)
d = true_d[ys, xs]
cam = np.stack([(xs - Kn[0, 2]) / Kn[0, 0] * d,
                (ys - Kn[1, 2]) / Kn[1, 1] * d, d, np.ones_like(d)], 1)
world = (cam @ c2wn.T)[:, :3]

# Delete a thin SURFACE SHELL: voxelize the surface points and delete only
# Gaussians sharing a voxel (dilated +/-1) -> localized patch, not a big box.
vs = 0.02 * float(data["scene_scale"])            # voxel size ~ 2% of scene
origin = world.min(0) - 4 * vs
means = splats["means"].cpu().numpy()
def vkey(pts):
    ijk = np.floor((pts - origin) / vs).astype(np.int64)
    return ijk
surf_ijk = np.unique(vkey(world), axis=0)
# dilate surface voxels by +/-1 in each axis (3^3 neighborhood) for shell thickness
offs = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
dil = np.unique((surf_ijk[:, None, :] + offs[None]).reshape(-1, 3), axis=0)
P = 1 << 21
def lin(ijk):
    a = ijk + P // 2
    return (a[:, 0] * P + a[:, 1]) * P + a[:, 2]
inside = np.isin(lin(vkey(means)), lin(dil))
print(f"hole: {len(world)} surf pts, voxel {vs:.4f}, {len(dil)} shell voxels")
keep = torch.from_numpy(~inside).to(dev)
holed = {k: v[keep].clone() for k, v in splats.items()}
print(f"deleted {int(inside.sum()):,} / {len(means):,} gaussians -> holed splat "
      f"{int(keep.sum()):,}")
torch.save({"splats": {k: v.cpu() for k, v in holed.items()}},
           os.path.join(IO2, "holed.pt"))

# Render full vs holed (test view) for reference + the holed disparity for InFusion
rgb_full, _, _ = render_view(splats, c2w, K, W, H)
rgb_holed, depth_holed, alpha_holed = render_view(holed, c2w, K, W, H)
print(f"test-view PSNR holed-vs-full: {psnr(rgb_holed, rgb_full):.2f} dB "
      f"(lower = bigger visible gap)")

valid = (alpha_holed > 0.5) & (depth_holed > 1e-6)
disp = torch.where(valid, 1.0 / depth_holed.clamp(min=1e-6), torch.zeros_like(depth_holed))
np.save(os.path.join(IO2, "disp.npy"), disp.cpu().numpy().astype(np.float32))
for f in ("intri.npy", "c2w.npy", "rgb.png", "true_depth.npy", "alpha.npy"):
    src = os.path.join(IO, f); dst = os.path.join(IO2, f)
    if f.endswith(".png"):
        cv2.imwrite(dst, cv2.imread(src))
    else:
        np.save(dst, np.load(src))

cmp = (torch.cat([rgb_full, rgb_holed], 1).cpu().numpy() * 255).astype(np.uint8)
cv2.imwrite(os.path.join(IO2, "full_vs_holed.png"), cv2.cvtColor(cmp, cv2.COLOR_RGB2BGR))
print("OUT", IO2)
