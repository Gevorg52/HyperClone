"""Generalization test: run the hole-detector on the REAL (un-holed) masked Laptop
splat. Critical question: does WARP correctly SUPPRESS the textureless deck (Type-4,
capture-limited) so it is NOT flagged as a hole? (err is high there because it's
soft, but geometry is CORRECT -> warp low -> defect low -> not masked.)

Shows per view: PHOTO | render | err | warp | defect=err*warp | auto-mask.
No centering prior (masked object -> object IS the subject, no room clutter).

Usage: python laptop_detect.py
"""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\output_4k_gated\ckpts\ckpt_29999_rank0.pt"
OUT = r"F:\HYPERclone\apex-sfm\_infusion\laptop"
SH, dev = 3, "cuda"
FACTOR = 2
os.makedirs(OUT, exist_ok=True)


def render(sp, c2w, K, W, H, mode="RGB+ED"):
    r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH,
                            torch.zeros(3, device=dev), "antialiased", render_mode=mode)
    return r[0], a[0, ..., 0]


def warp_disagree(photoA, dA, KA, c2wA, nbrs, data):
    H, W = dA.shape
    vv, uu = torch.meshgrid(torch.arange(H, device=dev).float(),
                            torch.arange(W, device=dev).float(), indexing="ij")
    cam = torch.stack([(uu - KA[0, 2]) / KA[0, 0] * dA, (vv - KA[1, 2]) / KA[1, 1] * dA,
                       dA, torch.ones_like(dA)], -1)
    world = cam @ c2wA.T
    dis = []
    for B in nbrs:
        bB = load_image(data, B); pB = bB["image"].to(dev)
        KB = bB["K"].to(dev); c2wB = bB["c2w"].to(dev)
        camB = world @ torch.linalg.inv(c2wB).T
        zB = camB[..., 2].clamp(min=1e-6)
        gx = ((camB[..., 0] / zB) * KB[0, 0] + KB[0, 2]) / (W - 1) * 2 - 1
        gy = ((camB[..., 1] / zB) * KB[1, 1] + KB[1, 2]) / (H - 1) * 2 - 1
        samp = F.grid_sample(pB.permute(2, 0, 1)[None], torch.stack([gx, gy], -1)[None],
                             align_corners=True, padding_mode="zeros")[0].permute(1, 2, 0)
        dd = (samp - photoA).abs().mean(-1)
        dd[(camB[..., 2] <= 0) | (gx.abs() > 1) | (gy.abs() > 1)] = float("nan")
        dis.append(dd)
    return torch.nan_to_num(torch.nanmedian(torch.stack(dis), 0).values, nan=0.0)


def auto_mask(defect, obj):
    thr = np.percentile(defect[obj], 97.0)
    m = (defect > max(thr, 1e-5)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m); floor = 0.004 * obj.sum()
    for a_, i in sorted([(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)], reverse=True)[:3]:
        if a_ >= floor:
            keep[lab == i] = 255
    keep = cv2.dilate(keep, np.ones((9, 9), np.uint8)); keep[~obj] = 0
    return keep


full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)
C = torch.tensor(np.stack([data["camtoworlds"][i][:3, 3] for i in range(len(data["camtoworlds"]))]),
                 device=dev).float()
print(f"laptop splat: {full['means'].shape[0]:,} gaussians, {len(data['camtoworlds'])} views")


def heat(x, obj):
    v = x / (np.percentile(x[obj], 99) + 1e-6)
    img = cv2.applyColorMap((np.clip(v, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~obj] = 30
    return img


tests = list(data["test_indices"])[::max(1, len(data["test_indices"]) // 4)][:4]
rows = []
tot_mask = tot_obj = 0
for idx in tests:
    b = load_image(data, idx); photo = b["image"].to(dev)
    K = b["K"].to(dev); c2w = b["c2w"].to(dev); H, W = photo.shape[:2]
    rr, a = render(full, c2w, K, W, H, "RGB+ED")
    depth = rr[..., 3]; obj = (a > 0.5).cpu().numpy()
    err = (rr[..., :3].clamp(0, 1) - photo).abs().mean(-1)
    dvec = (C - C[idx]).norm(dim=1); dvec[idx] = 1e9
    nbrs = torch.argsort(dvec)[:4].tolist()
    warp = warp_disagree(photo, depth, K, c2w, nbrs, data)
    e = err.cpu().numpy(); wv = warp.cpu().numpy()
    defect = e * wv; defect[~obj] = 0
    mask = auto_mask(defect, obj)
    tot_mask += int((mask > 0).sum()); tot_obj += int(obj.sum())
    print(f"view {idx:3d}: err(obj) {e[obj].mean():.4f}  warp(obj) {wv[obj].mean():.4f}  "
          f"flagged {100*(mask>0).sum()/max(obj.sum(),1):.2f}% of object")
    pa = (photo.cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
    pr = (rr[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
    ov = pa.copy(); ov[mask > 0] = (0, 0, 255)
    mk = cv2.addWeighted(pa, 0.6, ov, 0.4, 0)
    row = np.concatenate([pa, pr, heat(e, obj), heat(wv, obj), heat(defect, obj), mk], 1)
    rows.append(cv2.resize(row, (row.shape[1] // 3, row.shape[0] // 3)))
labels = "PHOTO | render | err | warp | defect | mask"
print(f"\nTOTAL flagged: {100*tot_mask/max(tot_obj,1):.2f}% of object across {len(tests)} views")
print(f"(want: LOW % and mask NOT on the textureless deck -> warp suppresses Type-4)")
cv2.imwrite(os.path.join(OUT, "laptop_detect.png"), np.concatenate(rows, 0))
print(f"layout: {labels}\nviz: {os.path.join(OUT, 'laptop_detect.png')}")
