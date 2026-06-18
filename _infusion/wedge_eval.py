"""GPU wedge: EVAL (base env). Score EscherNet-generated held-out-wedge views vs the
REAL held-out photos (masked, same crop). LPIPS/PSNR/SSIM + montage. This measures how
faithfully EscherNet generates a genuinely-missing 90deg from real multi-view conditioning
(the OOD question: does real-view conditioning overcome its Objaverse training bias?)."""
import os, math
import numpy as np
import torch
import cv2
from core.training.gsplat_trainer import ssim, get_lpips

W = r"F:\HYPERclone\apex-sfm\_infusion\wedge"
dev = "cuda"; lp = get_lpips(dev)
tgt_idx = np.load(os.path.join(W, "tgt_idx.npy"))
ref_idx = np.load(os.path.join(W, "ref_idx.npy"))


def t(img):
    return torch.from_numpy(img[..., ::-1].copy()).float().permute(2, 0, 1)[None].to(dev) / 255.0


ps, ss, ls = [], [], []
rows_gen, rows_gt = [], []
for ti in tgt_idx:
    g = cv2.imread(os.path.join(W, "gen", f"{ti}.png"))
    gt = cv2.imread(os.path.join(W, "tgt_gt", f"{ti}.png"))
    if g is None or gt is None:
        continue
    g = cv2.resize(g, (256, 256)); gt = cv2.resize(gt, (256, 256))
    tg, tgtt = t(g), t(gt)
    ps.append(10 * math.log10(1 / (torch.mean((tg - tgtt) ** 2).item() + 1e-9)))
    ss.append(ssim(tg, tgtt).item())
    ls.append(lp(tg.clamp(0, 1), tgtt.clamp(0, 1)).item())
    rows_gen.append(cv2.resize(g, (150, 150))); rows_gt.append(cv2.resize(gt, (150, 150)))

print(f"held-out 90deg wedge ({len(ps)} views) — EscherNet generated vs REAL photos:")
print(f"  PSNR {np.mean(ps):.2f}   SSIM {np.mean(ss):.4f}   LPIPS {np.mean(ls):.4f}")

refs = [cv2.resize(cv2.imread(os.path.join(W, "ref", f"{i}.png")), (150, 150))
        for i in ref_idx[np.linspace(0, len(ref_idx) - 1, 4).astype(int)]]
refrow = np.concatenate(refs + [np.full((150, 150, 3), 30, np.uint8)] * (len(rows_gen) - 4), 1) if len(rows_gen) >= 4 else np.concatenate(refs, 1)
genrow = np.concatenate(rows_gen, 1); gtrow = np.concatenate(rows_gt, 1)
for r, lab in [(genrow, "GENERATED (missing 90deg)"), (gtrow, "REAL held-out")]:
    cv2.putText(r, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.putText(refrow, "4 of 10 reference views (270deg)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
cv2.imwrite(os.path.join(W, "wedge_eval.png"), np.concatenate([refrow, genrow, gtrow], 0))
print("viz:", os.path.join(W, "wedge_eval.png"))
