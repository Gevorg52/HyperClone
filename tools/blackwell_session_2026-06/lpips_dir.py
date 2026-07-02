import glob, os, sys, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from core.training.gsplat_trainer import get_lpips, ssim
d = "cuda"; lp = get_lpips(d)
base = sys.argv[1]

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(d)
def psnr(x, gt): return 10*np.log10(1.0/(F.mse_loss(x, gt).item()+1e-10))

frames = sorted(os.path.basename(p) for p in glob.glob(base + r"\gt\*.png"))
agg = {"base 3DGRUT": {"l":[],"p":[],"s":[]}, "ArtiFixer(bidir)": {"l":[],"p":[],"s":[]}}
imp = []
for f in frames:
    gt = load(os.path.join(base,"gt",f)); rend = load(os.path.join(base,"rendered",f)); pred = load(os.path.join(base,"pred",f))
    for nm,t in [("base 3DGRUT",rend),("ArtiFixer(bidir)",pred)]:
        if t.shape[-2:]!=gt.shape[-2:]: t=F.interpolate(t,size=gt.shape[-2:],mode="bilinear",align_corners=False)
        agg[nm]["l"].append(float(lp(t,gt))); agg[nm]["p"].append(psnr(t,gt)); agg[nm]["s"].append(ssim(t,gt).item())
    imp.append(-(agg["ArtiFixer(bidir)"]["l"][-1]-agg["base 3DGRUT"]["l"][-1])/agg["base 3DGRUT"]["l"][-1]*100)
print(f"n={len(frames)} held-out views @720p")
print(f"{'model':18s} {'LPIPS':>8s} {'PSNR':>8s} {'SSIM':>8s}")
for nm in ("base 3DGRUT","ArtiFixer(bidir)"):
    print(f"{nm:18s} {np.mean(agg[nm]['l']):8.4f} {np.mean(agg[nm]['p']):8.3f} {np.mean(agg[nm]['s']):8.4f}")
lb,la=np.mean(agg["base 3DGRUT"]["l"]),np.mean(agg["ArtiFixer(bidir)"]["l"])
print(f"\nArtiFixer vs base: LPIPS {lb:.4f} -> {la:.4f} = {(-(la-lb)/lb*100):+.1f}%  (n={len(frames)})")
print(f"frames improved: {sum(1 for x in imp if x>0)}/{len(imp)}; range {min(imp):+.1f}%..{max(imp):+.1f}%")
print(f"REF: our Difix-v2 = -8.7% (38 views, factor2); causal n=6 ArtiFixer = -12.9%")
