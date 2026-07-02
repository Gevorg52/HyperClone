import glob, os, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from core.training.gsplat_trainer import get_lpips, ssim
d = "cuda"; lp = get_lpips(d)
base = r"D:\REPOS\HYPERclone\artifixer-checkpoints\laptop720_dyn_sample"

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(d)

def psnr(x, gt): return 10*np.log10(1.0/(F.mse_loss(x, gt).item()+1e-10))

frames = sorted(os.path.basename(p) for p in glob.glob(base + r"\gt\*.png"))
rows = {"base 3DGRUT": [], "ArtiFixer": []}
print(f"per-frame LPIPS (lower=better), {len(frames)} held-out views @720p:")
print(f"{'frame':10s} {'base':>8s} {'ArtiFixer':>10s} {'delta%':>8s}")
agg = {"base 3DGRUT": {"l":[], "p":[], "s":[]}, "ArtiFixer": {"l":[], "p":[], "s":[]}}
for f in frames:
    gt = load(os.path.join(base, "gt", f))
    rend = load(os.path.join(base, "rendered", f))
    pred = load(os.path.join(base, "pred", f))
    for nm, t in [("base 3DGRUT", rend), ("ArtiFixer", pred)]:
        if t.shape[-2:] != gt.shape[-2:]:
            t = F.interpolate(t, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        agg[nm]["l"].append(float(lp(t, gt))); agg[nm]["p"].append(psnr(t, gt)); agg[nm]["s"].append(ssim(t, gt).item())
    lb, la = agg["base 3DGRUT"]["l"][-1], agg["ArtiFixer"]["l"][-1]
    print(f"{f:10s} {lb:8.4f} {la:10.4f} {(-(la-lb)/lb*100):+7.1f}%")
print("-"*40)
print(f"{'model':14s} {'LPIPS':>8s} {'PSNR':>8s} {'SSIM':>8s}")
for nm in ("base 3DGRUT", "ArtiFixer"):
    L, P, S = np.mean(agg[nm]["l"]), np.mean(agg[nm]["p"]), np.mean(agg[nm]["s"])
    print(f"{nm:14s} {L:8.4f} {P:8.3f} {S:8.4f}")
lb, la = np.mean(agg["base 3DGRUT"]["l"]), np.mean(agg["ArtiFixer"]["l"])
print(f"\nArtiFixer vs base: LPIPS {lb:.4f} -> {la:.4f} = {(-(la-lb)/lb*100):+.1f}%  (n={len(frames)})")
print(f"For reference: our Difix-v2 was base 0.355 -> 0.324 = -8.7% (38 views, factor2/1080p)")
