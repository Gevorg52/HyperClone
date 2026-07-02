import glob, os, sys, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from core.training.gsplat_trainer import get_lpips, ssim
d="cuda"; lp=get_lpips(d)
base=sys.argv[1]
def load(p):
    a=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(d)
def psnr(x,gt): return 10*np.log10(1.0/(F.mse_loss(x,gt).item()+1e-10))
srcs={"base 3DGRUT":"rendered","ArtiFixer 2D(bidir)":"pred","ArtiFixer3D":"artifixer3d"}
frames=sorted(os.path.basename(p) for p in glob.glob(base+r"\gt\*.png"))
agg={k:{"l":[],"p":[],"s":[]} for k in srcs}
for f in frames:
    gt=load(os.path.join(base,"gt",f))
    for nm,sub in srcs.items():
        fp=os.path.join(base,sub,f)
        if not os.path.exists(fp): continue
        t=load(fp)
        if t.shape[-2:]!=gt.shape[-2:]: t=F.interpolate(t,size=gt.shape[-2:],mode="bilinear",align_corners=False)
        agg[nm]["l"].append(float(lp(t,gt))); agg[nm]["p"].append(psnr(t,gt)); agg[nm]["s"].append(ssim(t,gt).item())
print(f"n={len(frames)} held-out views   (LPIPS lower=better)")
print(f"{'model':22s} {'LPIPS':>8s} {'PSNR':>8s} {'SSIM':>8s} {'vs base':>9s}")
lb=np.mean(agg['base 3DGRUT']['l'])
for nm in srcs:
    if not agg[nm]["l"]: continue
    L,P,S=np.mean(agg[nm]["l"]),np.mean(agg[nm]["p"]),np.mean(agg[nm]["s"])
    print(f"{nm:22s} {L:8.4f} {P:8.3f} {S:8.4f} {(-(L-lb)/lb*100):+8.1f}%")
