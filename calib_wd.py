"""Calibrate wd_lambda: measure typical d_WD vs L_orig (0.8*L1+0.2*DSSIM) on real
render-vs-GT crops from the base model, so the WD term is scaled comparably to the
L1+SSIM term (target: WD contribution ~1-1.6x L_orig, per Apple WD-R)."""
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
from core.training.gsplat_trainer import (
    load_colmap_data, load_image, render_splats, ssim, get_lpips,
)
from core.training.wd_loss import wasserstein_distortion as wd

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\train\data"
BASE = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\train\output\ckpts\ckpt_29999_rank0.pt"
SH, FACTOR, PATCH, device = 3, 1, 512, "cuda"

s = torch.load(BASE, weights_only=True, map_location=device)["splats"]
splats = {k: v.to(device) for k, v in s.items()}
data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)
lp = get_lpips(device)

wds, lorigs, lps = [], [], []
with torch.no_grad():
    for idx in list(data["test_indices"])[:6]:
        b = load_image(data, idx)
        gt = b["image"].to(device); K = b["K"].to(device); c2w = b["c2w"].to(device)
        H, W = gt.shape[:2]
        r, _, _ = render_splats(splats, torch.linalg.inv(c2w), K, W, H, SH,
                                torch.zeros(3, device=device), "antialiased")
        r = r[0].clamp(0, 1)
        for _ in range(3):  # 3 random crops per view
            top = np.random.randint(0, H - PATCH); left = np.random.randint(0, W - PATCH)
            rc = r[top:top+PATCH, left:left+PATCH].permute(2, 0, 1).unsqueeze(0)
            gc = gt[top:top+PATCH, left:left+PATCH].permute(2, 0, 1).unsqueeze(0)
            wds.append(float(wd(rc, gc, device)))
            l1 = (rc - gc).abs().mean().item()
            dssim = 1.0 - ssim(rc, gc).item()
            lorigs.append(0.8 * l1 + 0.2 * dssim)
            lps.append(float(lp(rc, gc)))

wd_m, lo_m, lp_m = np.mean(wds), np.mean(lorigs), np.mean(lps)
print(f"mean d_WD       = {wd_m:.4f}")
print(f"mean L_orig     = {lo_m:.4f}  (0.8*L1+0.2*DSSIM)")
print(f"mean LPIPS      = {lp_m:.4f}  (current: 0.2*LPIPS = {0.2*lp_m:.4f} contribution)")
for tgt in (1.0, 1.6):
    print(f"wd_lambda for WD contribution = {tgt:.1f}x L_orig : {tgt*lo_m/wd_m:.4f}")
print(f"wd_lambda to match current 0.2*LPIPS contribution    : {0.2*lp_m/wd_m:.4f}")
