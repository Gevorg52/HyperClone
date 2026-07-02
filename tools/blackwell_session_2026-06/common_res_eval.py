import glob, os, numpy as np, torch
import torch.nn.functional as F
from PIL import Image
from core.training.gsplat_trainer import get_lpips, ssim
d="cuda"; lp=get_lpips(d)
S720=r"D:\REPOS\HYPERclone\artifixer-checkpoints\laptop720_bidir_sample"
S960=r"D:\REPOS\HYPERclone\artifixer-checkpoints\laptop960_sample"
def load(p):
    a=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.0
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(d)
def psnr(x,gt): return 10*np.log10(1.0/(F.mse_loss(x,gt).item()+1e-10))
# reference = 720p GT (1280x720). All candidates resized to this.
cands={"base @720p":(S720,"rendered"),"ArtiFixer3D @720p":(S720,"artifixer3d"),
       "base @960p->720":(S960,"rendered"),"ArtiFixer3D @960p->720":(S960,"artifixer3d")}
frames=sorted(os.path.basename(p) for p in glob.glob(S720+r"\gt\*.png"))
agg={k:{"l":[],"p":[],"s":[]} for k in cands}
for f in frames:
    gt=load(os.path.join(S720,"gt",f)); H,W=gt.shape[-2:]
    for nm,(root,sub) in cands.items():
        fp=os.path.join(root,sub,f)
        if not os.path.exists(fp): continue
        t=load(fp)
        if t.shape[-2:]!=(H,W): t=F.interpolate(t,size=(H,W),mode="bilinear",align_corners=False)
        agg[nm]["l"].append(float(lp(t,gt))); agg[nm]["p"].append(psnr(t,gt)); agg[nm]["s"].append(ssim(t,gt).item())
print(f"COMMON-RES (all evaluated at 720p vs the SAME 720p GT)   n={len(frames)}")
print(f"{'model':24s} {'LPIPS':>8s} {'PSNR':>8s} {'SSIM':>8s}")
for nm in cands:
    if agg[nm]["l"]:
        print(f"{nm:24s} {np.mean(agg[nm]['l']):8.4f} {np.mean(agg[nm]['p']):8.3f} {np.mean(agg[nm]['s']):8.4f}")
