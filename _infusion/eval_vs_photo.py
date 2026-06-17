"""Definitive multi-view eval vs the GT PHOTOS (not the imperfect full splat).
Compares original / holed / custom-refine / real-trainer on the deleted-region crop
across all test views that see the hole. LPIPS = the honest faithfulness metric."""
import numpy as np
import torch
import cv2
from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats, get_lpips

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\output_4k_gated\ckpts\ckpt_29999_rank0.pt"
CUSTOM = r"F:\HYPERclone\apex-sfm\_infusion\full_laptop\completed.pt"
REAL = r"F:\HYPERclone\apex-sfm\_infusion\diag_laptop\ckpts\ckpt_4999_rank0.pt"
SH, dev = 3, "cuda"
lp = get_lpips(dev)


def render(sp, c2w, K, W, H):
    r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH, torch.zeros(3, device=dev), "antialiased")
    return r[0][..., :3].clamp(0, 1)


full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True); scene = float(data["scene_scale"])
best = None
for idx in list(data["test_indices"]):
    ph = load_image(data, idx)["image"].cpu().numpy(); H, W = ph.shape[:2]
    c = ph[int(.4 * H):int(.6 * H), int(.4 * W):int(.6 * W)]
    gx = np.abs(np.diff(c.mean(-1), axis=1)).mean()
    if best is None or gx > best[0]: best = (gx, idx)
hv = best[1]; b = load_image(data, hv); K = b["K"].to(dev); c2w = b["c2w"].to(dev); H, W = b["image"].shape[:2]
r, a, _ = render_splats(full, torch.linalg.inv(c2w), K, W, H, SH, torch.zeros(3, device=dev), "antialiased", render_mode="RGB+ED")
depth = r[0, ..., 3]; selm = np.zeros((H, W), bool); bw, bh = int(.18 * W), int(.18 * H)
selm[(H - bh) // 2:(H + bh) // 2, (W - bw) // 2:(W + bw) // 2] = True
sel = ((a[0, ..., 0] > 0.5) & (depth > 1e-6)).cpu().numpy() & selm
Kn = K.cpu().numpy(); c2wn = c2w.cpu().numpy(); dd = depth.cpu().numpy()
ys, xs = np.nonzero(sel); dv = dd[ys, xs]
cam = np.stack([(xs - Kn[0, 2]) / Kn[0, 0] * dv, (ys - Kn[1, 2]) / Kn[1, 1] * dv, dv, np.ones_like(dv)], 1)
world = (cam @ c2wn.T)[:, :3]; vs = 0.02 * scene; origin = world.min(0) - 4 * vs
means = full["means"].cpu().numpy()
def vkey(p): return np.floor((p - origin) / vs).astype(np.int64)
surf = np.unique(vkey(world), axis=0)
offs = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
dil = np.unique((surf[:, None, :] + offs[None]).reshape(-1, 3), axis=0); P = 1 << 21
def lin(x): return ((x[:, 0] + P // 2) * P + (x[:, 1] + P // 2)) * P + (x[:, 2] + P // 2)
inside = np.isin(lin(vkey(means)), lin(dil))
holed = {k: v[torch.from_numpy(~inside).to(dev)].clone() for k, v in full.items()}
del_w = torch.from_numpy(means[inside]).float().to(dev)
custom = {k: v.to(dev) for k, v in torch.load(CUSTOM, weights_only=True)["splats"].items()}
real = {k: v.to(dev) for k, v in torch.load(REAL, weights_only=True)["splats"].items()}
states = [("original(ceiling)", full), ("holed(floor)", holed), ("custom-refine", custom), ("real-trainer", real)]
res = {n: [] for n, _ in states}
nv = 0
for idx in list(data["test_indices"]):
    b = load_image(data, idx); K = b["K"].to(dev); c2w = b["c2w"].to(dev); H, W = b["image"].shape[:2]; gt = b["image"].to(dev)
    xc = (torch.cat([del_w, torch.ones(len(del_w), 1, device=dev)], 1) @ torch.linalg.inv(c2w).T)[:, :3]
    z = xc[:, 2]; uv = (xc / z.clamp(min=1e-6)[:, None]) @ K.T
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if ok.sum() < 50: continue
    u, v = uv[ok, 0], uv[ok, 1]
    x0, y0, x1, y1 = max(0, int(u.min())), max(0, int(v.min())), min(W, int(u.max())), min(H, int(v.max()))
    if x1 - x0 < 24 or y1 - y0 < 24: continue
    g = gt[y0:y1, x0:x1].permute(2, 0, 1)[None].clamp(0, 1)
    for n, sp in states:
        im = render(sp, c2w, K, W, H)[y0:y1, x0:x1].permute(2, 0, 1)[None].clamp(0, 1)
        res[n].append(lp(im, g).item())
    nv += 1
print(f"\n{nv} test views see the hole. Mean LPIPS vs GT PHOTO on deleted-region crop (lower=faithful):")
for n, _ in states:
    print(f"  {n:18s} {np.mean(res[n]):.4f}")
