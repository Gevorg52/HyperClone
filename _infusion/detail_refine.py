"""Detail-refinement of the InFusion-seeded fill: sharpen the mushy hole using the
REAL multi-view photos + gated LPIPS (the detail exists in the raw images). InFusion
seeds geometry (cold-start); this stage recovers high-frequency detail the same way
the original splat got sharp — multi-view photometric optimization + perceptual loss.

Background frozen (no drift). Only the fill Gaussians train. Supervision is CROPPED
to the fill's image footprint per view (that's where detail must come from, and the
only place fill gradients flow). Loss = L1 + SSIM + gated-LPIPS (validated recipe).

Compares to the original FULL splat (GT) on the hole crop: PSNR/SSIM + LPIPS (the
detail metric). before (mushy) vs after (refined).

Usage: python detail_refine.py [steps]
"""
import math
import os
import sys

import numpy as np
import torch
import cv2

from core.training.gsplat_trainer import (load_colmap_data, load_image, render_splats,
                                          ssim, get_lpips)

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\output\ckpts\ckpt_29999_rank0.pt"
IO2 = r"F:\HYPERclone\apex-sfm\_infusion\io2"
COMP = r"F:\HYPERclone\apex-sfm\_infusion\prog\completed.pt"
OUT = r"F:\HYPERclone\apex-sfm\_infusion\refine"
SH, dev = 3, "cuda"
FACTOR = 4
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
LPIPS_L, PATCH = 0.3, 256
os.makedirs(OUT, exist_ok=True)


def render(sp, c2w, K, W, H, mode="RGB"):
    r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH,
                            torch.zeros(3, device=dev), "antialiased", render_mode=mode)
    return r[0], a[0, ..., 0]


def footprint(fill, c2w, K, W, H):
    """bbox of where the fill Gaussians render (alpha>0.1); None if too small."""
    with torch.no_grad():
        _, a = render(fill, c2w, K, W, H)
    m = (a > 0.1).cpu().numpy()
    if m.sum() < 1500:
        return None
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max(), ys.max()


def crop_box(box, W, H, pad=24, mn=PATCH):
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    # enforce min size
    if x1 - x0 < mn:
        c = (x0 + x1) // 2; x0, x1 = c - mn // 2, c + mn // 2
    if y1 - y0 < mn:
        c = (y0 + y1) // 2; y0, y1 = c - mn // 2, c + mn // 2
    x0 = max(0, min(x0, W - 1)); x1 = max(x0 + 1, min(x1, W))
    y0 = max(0, min(y0, H - 1)); y1 = max(y0 + 1, min(y1, H))
    return int(x0), int(y0), int(x1), int(y1)


full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
holed = {k: v.to(dev) for k, v in torch.load(os.path.join(IO2, "holed.pt"), weights_only=True)["splats"].items()}
comp = {k: v.to(dev) for k, v in torch.load(COMP, weights_only=True)["splats"].items()}
Nh = holed["means"].shape[0]
fill0 = {k: v[Nh:].clone() for k, v in comp.items()}        # the seeded+coarse fill
print(f"holed {Nh:,}  fill {fill0['means'].shape[0]:,}")

data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)
# observing views = where the fill renders (it sees the hole)
cand = sorted(set(int(i) for i in (list(data["train_indices"])[::4] + [255, 256, 257, 264, 240, 400])))
obsv = []
for i in cand:
    b = load_image(data, i); fp = footprint(fill0, b["c2w"].to(dev), b["K"].to(dev),
                                            b["image"].shape[1], b["image"].shape[0])
    if fp:
        obsv.append(i)
print(f"{len(obsv)} observing views for refinement")

DENSIFY = len(sys.argv) > 2 and sys.argv[2] == "densify"
DENS_EVERY, DENS_FRAC, DENS_CAP = 500, 0.12, 3.0
state = {k: v.clone() for k, v in fill0.items()}
N0 = state["means"].shape[0]


def make_opt(state):
    p = {k: torch.nn.Parameter(v.clone()) for k, v in state.items()}
    o = torch.optim.Adam([
        {"params": [p["means"]], "lr": 8e-5 * 1.772}, {"params": [p["scales"]], "lr": 8e-3},
        {"params": [p["quats"]], "lr": 1e-3}, {"params": [p["opacities"]], "lr": 3e-2},
        {"params": [p["sh0"]], "lr": 2.5e-3}, {"params": [p["shN"]], "lr": 1.25e-4}])
    return p, o


def densify(state, gacc):
    """Split high-gradient fill Gaussians (clone + halve scale); prune dead opacity."""
    if state["means"].shape[0] >= DENS_CAP * N0:
        return state
    keep = torch.sigmoid(state["opacities"].squeeze(-1)) > 0.02      # prune dead
    state = {k: v[keep] for k, v in state.items()}; gacc = gacc[keep]
    k = max(1, int(DENS_FRAC * state["means"].shape[0]))
    sel = torch.topk(gacc, k).indices
    child = {kk: state[kk][sel].clone() for kk in state}
    child["scales"] = child["scales"] - math.log(1.6)               # finer children
    child["means"] = child["means"] + torch.randn_like(child["means"]) * math.exp(
        child["scales"].mean().item()) * 0.5
    return {kk: torch.cat([state[kk], child[kk]], 0) for kk in state}


params, opt = make_opt(state)
gacc = torch.zeros(params["means"].shape[0], device=dev)
lpips = get_lpips(dev)
g = torch.Generator().manual_seed(0)
ema = None


def comb(p):
    return {k: torch.cat([holed[k], p[k]], 0) for k in holed}


for step in range(STEPS):
    if DENSIFY and step > 0 and step % DENS_EVERY == 0 and step < STEPS - 400:
        state = {k: params[k].detach() for k in params}
        state = densify(state, gacc)
        params, opt = make_opt(state)
        gacc = torch.zeros(params["means"].shape[0], device=dev)
        print(f"  [densify] -> {params['means'].shape[0]:,} fill gaussians")
    i = obsv[int(torch.randint(len(obsv), (1,), generator=g))]
    b = load_image(data, i); gt = b["image"].to(dev); K = b["K"].to(dev); c2w = b["c2w"].to(dev)
    H, W = gt.shape[:2]
    with torch.no_grad():
        fp = footprint(params, c2w, K, W, H)
    if fp is None:
        continue
    x0, y0, x1, y1 = crop_box(fp, W, H)
    r = render(comb(params), c2w, K, W, H)[0][..., :3].clamp(0, 1)
    rc = r[y0:y1, x0:x1]; gc = gt[y0:y1, x0:x1]
    l1 = (rc - gc).abs().mean()
    sl = 1 - ssim(rc.permute(2, 0, 1)[None], gc.permute(2, 0, 1)[None])
    lp = lpips(rc.permute(2, 0, 1)[None].clamp(0, 1), gc.permute(2, 0, 1)[None].clamp(0, 1))
    cl1 = l1.detach()
    ema = cl1 if ema is None else 0.99 * ema + 0.01 * cl1
    gate = torch.sigmoid((cl1 - ema) / (0.25 * ema + 1e-6))
    loss = 0.8 * l1 + 0.2 * sl + LPIPS_L * gate * lp
    opt.zero_grad(); loss.backward()
    if params["means"].grad is not None:
        gacc += params["means"].grad.detach().norm(dim=1)
    opt.step()
    if step % 500 == 0:
        print(f"  step {step:4d}  l1 {l1.item():.4f}  lpips {lp.item():.4f}")

# ---- evaluate on view 256 vs FULL (GT), cropped to hole ----
b = load_image(data, 256); K = b["K"].to(dev); c2w = b["c2w"].to(dev); H, W = b["image"].shape[:2]
rf = render(full, c2w, K, W, H)[0][..., :3].clamp(0, 1).detach()
rm = render(comp, c2w, K, W, H)[0][..., :3].clamp(0, 1).detach()      # mushy (before)
rr = render(comb(params), c2w, K, W, H)[0][..., :3].clamp(0, 1).detach()  # refined (after)
x0, y0, x1, y1 = crop_box(footprint(fill0, c2w, K, W, H), W, H, pad=40, mn=320)


def stats(x, ref, tag):
    p = 10 * math.log10(1.0 / (torch.mean((x - ref) ** 2).item() + 1e-9))
    s = ssim(x.permute(2, 0, 1)[None], ref.permute(2, 0, 1)[None]).item()
    lpv = lpips(x[y0:y1, x0:x1].permute(2, 0, 1)[None].clamp(0, 1),
                ref[y0:y1, x0:x1].permute(2, 0, 1)[None].clamp(0, 1)).item()
    print(f"{tag:18s} PSNR {p:6.2f}  SSIM {s:.4f}   hole-LPIPS(vs full) {lpv:.4f}")
    return lpv


print("\n=== vs FULL splat (GT). hole-LPIPS lower = sharper detail ===")
stats(rm, rf, "mushy (before)")
stats(rr, rf, "refined (after)")
# zoom panel: FULL | mushy | refined  (hole crop)
zc = []
for im, t in [(rf, "FULL"), (rm, "mushy"), (rr, "refined")]:
    c = (im[y0:y1, x0:x1].cpu().numpy() * 255).astype(np.uint8)[..., ::-1].copy()
    c = cv2.resize(c, (360, int(360 * c.shape[0] / c.shape[1])), interpolation=cv2.INTER_NEAREST)
    cv2.putText(c, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    zc.append(c)
cv2.imwrite(os.path.join(OUT, "refine_zoom.png"), np.concatenate(zc, 1))
torch.save({"splats": {k: v.detach().cpu() for k, v in comb(params).items()}},
           os.path.join(OUT, "refined.pt"))
print("viz:", os.path.join(OUT, "refine_zoom.png"), "(FULL | mushy | refined)")
