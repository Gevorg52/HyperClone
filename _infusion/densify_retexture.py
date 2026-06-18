"""Test the user's hypothesis on the REAL laptop (no carving): the fuzzy region is
GAUSSIAN-STARVED (few large soft Gaussians, because MCMC drains textureless regions).
Add anchors by SPLITTING the large soft Gaussians into many small ones, then RE-TEXTURE
them against the REAL dataset images (gated-LPIPS multi-view). If the photos hold detail,
this recovers it FAITHFULLY (better than FLUX hallucination); if they don't, it only
smooths (-> FLUX is then the only way to add detail).

Eval: LPIPS vs the REAL PHOTOS on held-out test views (lower = more faithful detail),
before (original) vs after (densified + re-textured). + visual at a fuzzy view.

Usage: python densify_retexture.py [split_pct] [steps]
"""
import math
import os
import sys

import numpy as np
import torch
import cv2

from core.training.gsplat_trainer import (load_colmap_data, load_image, render_splats,
                                          ssim, get_lpips)

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\output_4k_gated\ckpts\ckpt_29999_rank0.pt"
OUT = r"F:\HYPERclone\apex-sfm\_infusion\densify_retex"
SH, dev = 3, "cuda"
PCT = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0      # top-% scale to split
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
LPIPS_L, PATCH = 0.3, 256
os.makedirs(OUT, exist_ok=True)


def render(sp, c2w, K, W, H):
    r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH,
                            torch.zeros(3, device=dev), "antialiased")
    return r[0][..., :3].clamp(0, 1)


full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True)
scene = float(data["scene_scale"])
tr_idx = list(data["train_indices"]); te_idx = list(data["test_indices"])
lp = get_lpips(dev); g = torch.Generator().manual_seed(0)
gc = torch.Generator(device=dev).manual_seed(0)   # cuda generator for randn

# --- select the large soft Gaussians (the starved/fuzzy ones) ---
gscale = torch.exp(full["scales"]).mean(1)                   # mean radius per Gaussian
thr = torch.quantile(gscale, 1 - PCT / 100)
sel = gscale > thr
N = full["means"].shape[0]
print(f"splatting {N:,} gaussians; splitting top {PCT:.0f}% by scale = {int(sel.sum()):,}")

frozen = {k: v[~sel] for k, v in full.items()}

# --- SPLIT each selected Gaussian into K small anchors (standard 3DGS split) ---
K = 3
selg = {k: v[sel] for k, v in full.items()}
M = selg["means"].shape[0]
R = torch.exp(selg["scales"])                                # (M,3) per-axis radius
new = {k: [] for k in full}
for _ in range(K):
    pert = torch.randn(M, 3, generator=gc, device=dev) * R * 0.6
    nm = selg["means"] + pert
    new["means"].append(nm)
    new["scales"].append(selg["scales"] - math.log(1.6))     # smaller children
    new["quats"].append(selg["quats"].clone())
    new["opacities"].append(selg["opacities"].clone())
    new["sh0"].append(selg["sh0"].clone())
    new["shN"].append(selg["shN"].clone())
new = {k: torch.cat(v, 0) for k, v in new.items()}
print(f"-> {new['means'].shape[0]:,} new anchors (total {frozen['means'].shape[0]+new['means'].shape[0]:,})")


def comb(p):
    return {k: torch.cat([frozen[k], p[k]], 0) for k in frozen}


# --- baseline eval (original splat) vs real photos on held-out test views ---
def eval_lpips(splat, n=18):
    vals = []
    with torch.no_grad():
        for idx in te_idx[:n]:
            b = load_image(data, idx); gt = b["image"].to(dev); Kc = b["K"].to(dev); c2w = b["c2w"].to(dev)
            H, W = gt.shape[:2]
            r = render(splat, c2w, Kc, W, H)
            vals.append(lp(r.permute(2, 0, 1)[None].clamp(0, 1), gt.permute(2, 0, 1)[None].clamp(0, 1)).item())
    return float(np.mean(vals))


base_lpips = eval_lpips(full)
print(f"\noriginal splat: held-out LPIPS vs photos {base_lpips:.4f}")

# --- re-texture the anchors against the REAL photos (gated LPIPS) ---
params = {k: torch.nn.Parameter(v.clone()) for k, v in new.items()}
opt = torch.optim.Adam([
    {"params": [params["means"]], "lr": 8e-5 * scene}, {"params": [params["scales"]], "lr": 5e-3},
    {"params": [params["quats"]], "lr": 1e-3}, {"params": [params["opacities"]], "lr": 3e-2},
    {"params": [params["sh0"]], "lr": 2.5e-3}, {"params": [params["shN"]], "lr": 1.25e-4}])
ema = None
for step in range(STEPS):
    idx = tr_idx[int(torch.randint(len(tr_idx), (1,), generator=g))]
    b = load_image(data, idx); gt = b["image"].to(dev); Kc = b["K"].to(dev); c2w = b["c2w"].to(dev)
    H, W = gt.shape[:2]
    r = render(comb(params), c2w, Kc, W, H)
    l1 = (r - gt).abs().mean()
    sl = 1 - ssim(r.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
    top = int(torch.randint(0, H - PATCH + 1, (1,), generator=g)); lft = int(torch.randint(0, W - PATCH + 1, (1,), generator=g))
    rc = r[top:top + PATCH, lft:lft + PATCH]; gc = gt[top:top + PATCH, lft:lft + PATCH]
    lpv = lp(rc.permute(2, 0, 1)[None].clamp(0, 1), gc.permute(2, 0, 1)[None].clamp(0, 1))
    cl1 = (rc - gc).abs().mean().detach()
    ema = cl1 if ema is None else 0.99 * ema + 0.01 * cl1
    gate = torch.sigmoid((cl1 - ema) / (0.25 * ema + 1e-6))
    loss = 0.8 * l1 + 0.2 * sl + LPIPS_L * gate * lpv
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(f"  step {step:4d}  l1 {l1.item():.4f}")

dens = comb(params)
new_lpips = eval_lpips(dens)
print(f"\n=== held-out LPIPS vs REAL PHOTOS (lower = more faithful detail) ===")
print(f"  original            {base_lpips:.4f}")
print(f"  densified+retexture {new_lpips:.4f}   ({'BETTER' if new_lpips < base_lpips else 'no gain'}, "
      f"d={new_lpips-base_lpips:+.4f})")

# visual at a test view: original | densified | real photo (cropped to a soft area)
vi = te_idx[len(te_idx) // 2]; b = load_image(data, vi); gt = b["image"].to(dev); Kc = b["K"].to(dev); c2w = b["c2w"].to(dev)
H, W = gt.shape[:2]
o = (render(full, c2w, Kc, W, H).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
d = (render(dens, c2w, Kc, W, H).detach().cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
p = (gt.cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
panel = np.concatenate([o, d, p], 1)
for i, t in enumerate(["original", "densified+retex", "REAL photo"]):
    cv2.putText(panel, t, (i * W + 10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
cv2.imwrite(os.path.join(OUT, "compare.png"), cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2)))
torch.save({"splats": {k: v.detach().cpu() for k, v in dens.items()}}, os.path.join(OUT, "densified.pt"))
print("viz:", os.path.join(OUT, "compare.png"), "(original | densified+retex | REAL photo)")
