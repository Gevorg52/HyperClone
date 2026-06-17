"""Honest multi-view eval of the Laptop auto-complete: reproduce the (deterministic)
carve to recover `holed` + the deleted-region image footprint (FIXED eval crop), then
compare holed vs completed vs full across ALL test views in that fixed region.
Single-view eval was misleading (carve view barely damaged; crop varied per run)."""
import math
import os
import numpy as np
import torch
import cv2
from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats, ssim, get_lpips

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\output_4k_gated\ckpts\ckpt_29999_rank0.pt"
import sys
COMP = sys.argv[1] if len(sys.argv) > 1 else r"F:\HYPERclone\apex-sfm\_infusion\full_laptop\completed.pt"
SH, dev = 3, "cuda"; FACTOR = 2


def render(sp, c2w, K, W, H, mode="RGB"):
    r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH,
                            torch.zeros(3, device=dev), "antialiased", render_mode=mode)
    return r[0], a[0, ..., 0]


full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)
scene = float(data["scene_scale"]); lp = get_lpips(dev)

# reproduce carve (deterministic): most-detailed test view, centre box, shell delete
best = None
for idx in list(data["test_indices"]):
    ph = load_image(data, idx)["image"].cpu().numpy(); H, W = ph.shape[:2]
    c = ph[int(.4 * H):int(.6 * H), int(.4 * W):int(.6 * W)]
    gx = np.abs(np.diff(c.mean(-1), axis=1)).mean()
    if best is None or gx > best[0]:
        best = (gx, idx)
hv = best[1]
b = load_image(data, hv); K = b["K"].to(dev); c2w = b["c2w"].to(dev); H, W = b["image"].shape[:2]
rr, a = render(full, c2w, K, W, H, "RGB+ED"); depth = rr[..., 3]
selm = np.zeros((H, W), bool); bw, bh = int(.18 * W), int(.18 * H)
selm[(H - bh) // 2:(H + bh) // 2, (W - bw) // 2:(W + bw) // 2] = True
sel = ((a > 0.5) & (depth > 1e-6)).cpu().numpy() & selm
Kn = K.cpu().numpy(); c2wn = c2w.cpu().numpy(); dd = depth.cpu().numpy()
ys, xs = np.nonzero(sel); dv = dd[ys, xs]
cam = np.stack([(xs - Kn[0, 2]) / Kn[0, 0] * dv, (ys - Kn[1, 2]) / Kn[1, 1] * dv, dv, np.ones_like(dv)], 1)
world = (cam @ c2wn.T)[:, :3]
vs = 0.02 * scene; origin = world.min(0) - 4 * vs
means = full["means"].cpu().numpy()
def vkey(p): return np.floor((p - origin) / vs).astype(np.int64)
surf = np.unique(vkey(world), axis=0)
offs = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
dil = np.unique((surf[:, None, :] + offs[None]).reshape(-1, 3), axis=0)
P = 1 << 21
def lin(x): return ((x[:, 0] + P // 2) * P + (x[:, 1] + P // 2)) * P + (x[:, 2] + P // 2)
inside = np.isin(lin(vkey(means)), lin(dil))
holed = {k: v[torch.from_numpy(~inside).to(dev)].clone() for k, v in full.items()}
del_w = torch.from_numpy(means[inside]).float().to(dev)            # deleted region (fixed eval target)
comp = {k: v.to(dev) for k, v in torch.load(COMP, weights_only=True)["splats"].items()}
print(f"carve view {hv}, deleted {int(inside.sum()):,}; eval across test views on deleted-region crop")


def footprint_pts(pts, c2w, K, W, H):
    xc = (torch.cat([pts, torch.ones(len(pts), 1, device=dev)], 1) @ torch.linalg.inv(c2w).T)[:, :3]
    z = xc[:, 2]; uv = (xc / z.clamp(min=1e-6)[:, None]) @ K.T
    u = uv[:, 0]; v = uv[:, 1]; ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if ok.sum() < 50:
        return None
    u = u[ok]; v = v[ok]
    return int(u.min()), int(v.min()), int(u.max()), int(v.max())


def met(x, ref, box):
    x0, y0, x1, y1 = box
    xc_, rc_ = x[y0:y1, x0:x1], ref[y0:y1, x0:x1]
    p = 10 * math.log10(1.0 / (torch.mean((xc_ - rc_) ** 2).item() + 1e-9))
    l = lp(xc_.permute(2, 0, 1)[None].clamp(0, 1), rc_.permute(2, 0, 1)[None].clamp(0, 1)).item()
    return p, l


rh = {"p": [], "l": []}; rc = {"p": [], "l": []}; nval = 0; worst = None
for idx in list(data["test_indices"]):
    b = load_image(data, idx); K = b["K"].to(dev); c2w = b["c2w"].to(dev); H, W = b["image"].shape[:2]
    box = footprint_pts(del_w, c2w, K, W, H)
    if box is None:
        continue
    x0, y0, x1, y1 = box
    if x1 - x0 < 24 or y1 - y0 < 24:
        continue
    rf = render(full, c2w, K, W, H)[0][..., :3].clamp(0, 1).detach()
    rho = render(holed, c2w, K, W, H)[0][..., :3].clamp(0, 1).detach()
    rco = render(comp, c2w, K, W, H)[0][..., :3].clamp(0, 1).detach()
    ph, lh = met(rho, rf, box); pc, lc = met(rco, rf, box)
    rh["p"].append(ph); rh["l"].append(lh); rc["p"].append(pc); rc["l"].append(lc); nval += 1
    if worst is None or ph < worst[0]:                # view where hole is most visible
        pad = 30; bx = (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))
        worst = (ph, idx, [im[bx[1]:bx[3], bx[0]:bx[2]] for im in (rf, rho, rco)])
print(f"\n{nval} test views see the deleted region. Mean over deleted-region crop, vs FULL (GT):")
print(f"  holed         PSNR {np.mean(rh['p']):6.2f}  LPIPS {np.mean(rh['l']):.4f}")
print(f"  auto-complete PSNR {np.mean(rc['p']):6.2f}  LPIPS {np.mean(rc['l']):.4f}")
dp = np.mean(rc['p']) - np.mean(rh['p']); dl = np.mean(rc['l']) - np.mean(rh['l'])
print(f"  delta         PSNR {dp:+.2f}        LPIPS {dl:+.4f}   "
      f"({'IMPROVED' if dp > 0 and dl < 0 else 'MIXED' if dp > 0 or dl < 0 else 'WORSE'})")
zc = []
for im, t in zip(worst[2], ["FULL", "holed", "auto-complete"]):
    c = (im.cpu().numpy() * 255).astype(np.uint8)[..., ::-1].copy()
    c = cv2.resize(c, (340, int(340 * c.shape[0] / max(c.shape[1], 1))), interpolation=cv2.INTER_NEAREST)
    cv2.putText(c, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2); zc.append(c)
hh = min(x.shape[0] for x in zc); zc = [x[:hh] for x in zc]
cv2.imwrite(r"F:\HYPERclone\apex-sfm\_infusion\full_laptop\result_holeview.png", np.concatenate(zc, 1))
print(f"viz at worst-hole view {worst[1]} (holed PSNR {worst[0]:.1f}): full_laptop\\result_holeview.png")
