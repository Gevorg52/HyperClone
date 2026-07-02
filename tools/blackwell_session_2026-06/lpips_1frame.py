import torch, numpy as np
from PIL import Image
from core.training.gsplat_trainer import get_lpips, ssim
import torch.nn.functional as F
d = "cuda"
lp = get_lpips(d)
base = r"D:\REPOS\HYPERclone\artifixer-checkpoints\laptop720_sample"

def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(d)

gt = load(base + r"\gt.png")
pred = load(base + r"\artifixer_pred.png")
rend = load(base + r"\base_3dgrut.png")
# align sizes if needed
for name, t in [("pred", pred), ("rend", rend)]:
    if t.shape[-2:] != gt.shape[-2:]:
        if name == "pred": pred = F.interpolate(t, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        else: rend = F.interpolate(t, size=gt.shape[-2:], mode="bilinear", align_corners=False)

def psnr(x): return 10*np.log10(1.0/(F.mse_loss(x, gt).item()+1e-10))
print(f"resolution: {tuple(gt.shape[-2:])}  (n=1 held-out frame, frame 00000 = first held-out view)")
print(f"{'image':18s} {'LPIPS':>8s} {'PSNR':>8s} {'SSIM':>8s}   (LPIPS lower=better)")
for name, t in [("base 3DGRUT", rend), ("ArtiFixer pred", pred)]:
    print(f"{name:18s} {float(lp(t, gt)):8.4f} {psnr(t):8.3f} {ssim(t, gt).item():8.4f}")
