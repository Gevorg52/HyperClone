"""GPU held-out-90deg-wedge: PREP (base env). Simulate a 270deg capture, hold out a
90deg azimuth wedge, and prepare EscherNet inputs: mask the object (BiRefNet), square-
center-crop it (EscherNet expects centered object on white bg), resize 256. Save N
reference RGBA + ref/target c2w + object centre + masked GT for the held-out wedge.
Object centre = least-squares intersection of camera viewing rays (robust for an orbit).
"""
import os
import numpy as np
import cv2
from core.training.gsplat_trainer import load_colmap_data, load_image
from core.masking import load_birefnet, run_birefnet_single
from core.enhancement.coverage import estimate_up

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
OUT = r"F:\HYPERclone\apex-sfm\_infusion\wedge"
N_REF, N_TGT, WEDGE_DEG = 10, 6, 90
os.makedirs(OUT, exist_ok=True)
for s in ("ref", "tgt_gt"):
    os.makedirs(os.path.join(OUT, s), exist_ok=True)

data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True)
N = len(data["camtoworlds"])
c2ws = [data["camtoworlds"][i] for i in range(N)]
cen = np.stack([c[:3, 3] for c in c2ws])
fwd = np.stack([c[:3, 2] for c in c2ws])              # OpenCV +z forward
# object centre = point closest to all camera rays
A = np.zeros((3, 3)); b = np.zeros(3)
for C, d in zip(cen, fwd):
    P = np.eye(3) - np.outer(d, d); A += P; b += P @ C
O = np.linalg.solve(A, b)
up = estimate_up(cen, O)
ref0 = (cen[0] - O) - (cen[0] - O).dot(up) * up; ref0 /= np.linalg.norm(ref0) + 1e-9
perp = np.cross(up, ref0)
proj = (cen - O) - np.outer((cen - O).dot(up), up)
az = np.degrees(np.arctan2(proj.dot(perp), proj.dot(ref0))) % 360
print(f"object centre {O.round(3)}; {N} cams; az span {az.min():.0f}-{az.max():.0f}")

# hold out a 90deg azimuth wedge centred at a well-populated azimuth
wlo, whi = 135, 135 + WEDGE_DEG
in_wedge = (az >= wlo) & (az < whi)
ref_pool = np.where(~in_wedge)[0]; tgt_pool = np.where(in_wedge)[0]
print(f"wedge {wlo}-{whi}deg: {len(tgt_pool)} held-out, {len(ref_pool)} reference pool")
ref_idx = ref_pool[np.argsort(az[ref_pool])][np.linspace(0, len(ref_pool) - 1, N_REF).astype(int)]
tgt_idx = tgt_pool[np.argsort(az[tgt_pool])][np.linspace(0, len(tgt_pool) - 1, N_TGT).astype(int)]

birefnet, tf = load_birefnet("general")


def masked_crop(idx, save_dir, white_bg):
    b = load_image(data, idx)
    img = (b["image"].numpy() * 255).astype(np.uint8)              # HWC RGB
    ip = os.path.join(OUT, f"_tmp_{idx}.png")
    cv2.imwrite(ip, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    m = run_birefnet_single(birefnet, tf, ip, os.path.join(OUT, "_m"))
    os.remove(ip)
    if m is None or (m > 40).sum() < 500:
        return False
    ys, xs = np.where(m > 40)
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    half = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.62) + 8
    H, W = img.shape[:2]
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(W, cx + half), min(H, cy + half)
    crop = img[y0:y1, x0:x1].astype(np.float32) / 255.0
    a = (m[y0:y1, x0:x1].astype(np.float32) / 255.0)[..., None]
    if white_bg:
        out = (crop * a + (1 - a))                                 # white-bg RGB
        out = cv2.resize((out * 255).astype(np.uint8), (256, 256))
        cv2.imwrite(os.path.join(save_dir, f"{idx}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    else:                                                          # masked GT (white bg too, for fair compare)
        out = (crop * a + (1 - a))
        out = cv2.resize((out * 255).astype(np.uint8), (256, 256))
        cv2.imwrite(os.path.join(save_dir, f"{idx}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return True


ref_keep, tgt_keep = [], []
for i in ref_idx:
    if masked_crop(int(i), os.path.join(OUT, "ref"), True):
        ref_keep.append(int(i))
for i in tgt_idx:
    if masked_crop(int(i), os.path.join(OUT, "tgt_gt"), False):
        tgt_keep.append(int(i))

np.save(os.path.join(OUT, "ref_c2w.npy"), np.stack([c2ws[i] for i in ref_keep]))
np.save(os.path.join(OUT, "tgt_c2w.npy"), np.stack([c2ws[i] for i in tgt_keep]))
np.save(os.path.join(OUT, "object_center.npy"), O)
np.save(os.path.join(OUT, "ref_idx.npy"), np.array(ref_keep))
np.save(os.path.join(OUT, "tgt_idx.npy"), np.array(tgt_keep))
print(f"saved {len(ref_keep)} refs + {len(tgt_keep)} targets to {OUT}")
