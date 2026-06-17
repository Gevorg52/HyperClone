"""Build init for the diagnosis: reproduce the (deterministic) Laptop carve -> holed,
add the RAW InFusion seed points (cold-start in the empty hole) -> save as an
init_ckpt for the REAL production trainer (MCMC + gated-LPIPS) to reconstruct.
"""
import glob
import math
import os
import numpy as np
import torch
from gsplat import export_splats  # noqa (ensures gsplat importable)
from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\output_4k_gated\ckpts\ckpt_29999_rank0.pt"
SEEDS = r"F:\HYPERclone\apex-sfm\_infusion\full_laptop"
OUT = r"F:\HYPERclone\apex-sfm\_infusion\diag\init_laptop.pt"
SH, dev, C0 = 3, "cuda", 0.28209479177387814
os.makedirs(os.path.dirname(OUT), exist_ok=True)

full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True)
scene = float(data["scene_scale"])

# reproduce carve (deterministic)
best = None
for idx in list(data["test_indices"]):
    ph = load_image(data, idx)["image"].cpu().numpy(); H, W = ph.shape[:2]
    c = ph[int(.4 * H):int(.6 * H), int(.4 * W):int(.6 * W)]
    gx = np.abs(np.diff(c.mean(-1), axis=1)).mean()
    if best is None or gx > best[0]:
        best = (gx, idx)
hv = best[1]
b = load_image(data, hv); K = b["K"].to(dev); c2w = b["c2w"].to(dev); H, W = b["image"].shape[:2]
r, a, _ = render_splats(full, torch.linalg.inv(c2w), K, W, H, SH, torch.zeros(3, device=dev),
                        "antialiased", render_mode="RGB+ED")
depth = r[0, ..., 3]; selm = np.zeros((H, W), bool); bw, bh = int(.18 * W), int(.18 * H)
selm[(H - bh) // 2:(H + bh) // 2, (W - bw) // 2:(W + bw) // 2] = True
sel = ((a[0, ..., 0] > 0.5) & (depth > 1e-6)).cpu().numpy() & selm
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
print(f"carve view {hv}: holed {holed['means'].shape[0]:,} (deleted {int(inside.sum()):,})")

# gather ALL raw InFusion seeds (maximize cold-start coverage in the hole)
plys = sorted(glob.glob(os.path.join(SEEDS, "r*", "out", "rgb_mask.ply")))
pts, cols = [], []
for p in plys:
    raw = np.loadtxt(p, skiprows=10)
    if raw.ndim == 2 and len(raw):
        pts.append(raw[:, :3]); cols.append(raw[:, 3:6] / 255.0)
pts = torch.from_numpy(np.concatenate(pts)).float().to(dev)
col = torch.from_numpy(np.concatenate(cols)).float().to(dev)
M = pts.shape[0]
seed = {"means": pts, "scales": torch.full((M, 3), math.log(vs * 0.5), device=dev),
        "quats": torch.tensor([1.0, 0, 0, 0], device=dev).repeat(M, 1),
        "opacities": torch.full((M,) + tuple(holed["opacities"].shape[1:]), -2.197, device=dev),
        "sh0": ((col - 0.5) / C0).reshape(M, 1, 3),
        "shN": torch.zeros((M,) + tuple(holed["shN"].shape[1:]), device=dev)}
comb = {k: torch.cat([holed[k], seed[k]], 0) for k in holed}
print(f"seed {M:,} from {len(plys)} views -> init {comb['means'].shape[0]:,} gaussians")
torch.save({"splats": {k: v.detach().cpu() for k, v in comb.items()}}, OUT)
print("saved", OUT)
