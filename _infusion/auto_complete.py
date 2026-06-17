"""Automatic splat completion: detect holes -> auto-mask -> InFusion fill -> inject
-> distill. NO hand-specified mask. Validates on the holed GPU scan (recover the
deleted region automatically; full splat = ground truth).

Type-1 insight: a hole we can DETECT in a real view A has A's real photo showing the
true content (the hole is a GEOMETRY defect, not appearance) -> fill RGB = A's own
photo, fill geometry = InFusion depth-completion. Faithful, SDXL-free.

Stages:  python auto_complete.py detect   # preview auto-masks (overlay) for chosen views
         python auto_complete.py run      # full: fill + inject + distill + compare

Detector = err*warp (validated): err=|render-photo| magnitude, warp=multi-view
geometry inconsistency (discriminator). Object-gated, percentile threshold, cleaned.
"""
import math
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats, ssim

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\output\ckpts\ckpt_29999_rank0.pt"
IO2 = r"F:\HYPERclone\apex-sfm\_infusion\io2"
AUTO = r"F:\HYPERclone\apex-sfm\_infusion\auto"
RUN_DIR = r"F:\HYPERclone\infusion\depth_inpainting\run"
MODEL = r"F:\HYPERclone\infusion\checkpoints\Infusion"
SH, dev = 3, "cuda"
FACTOR, NVIEWS = 4, 6
C0 = 0.28209479177387814
os.makedirs(AUTO, exist_ok=True)


def render(sp, c2w, K, W, H, mode="RGB+ED"):
    r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH,
                            torch.zeros(3, device=dev), "antialiased", render_mode=mode)
    return r[0], a[0, ..., 0]


def warp_disagree(sp, photoA, dA, KA, c2wA, nbrs, data):
    H, W = dA.shape
    vv, uu = torch.meshgrid(torch.arange(H, device=dev).float(),
                            torch.arange(W, device=dev).float(), indexing="ij")
    cam = torch.stack([(uu - KA[0, 2]) / KA[0, 0] * dA, (vv - KA[1, 2]) / KA[1, 1] * dA,
                       dA, torch.ones_like(dA)], -1)
    world = cam @ c2wA.T
    dis = []
    for B in nbrs:
        bB = load_image(data, B); pB = bB["image"].to(dev)
        KB = bB["K"].to(dev); c2wB = bB["c2w"].to(dev)
        camB = world @ torch.linalg.inv(c2wB).T
        zB = camB[..., 2].clamp(min=1e-6)
        uB = (camB[..., 0] / zB) * KB[0, 0] + KB[0, 2]
        vB = (camB[..., 1] / zB) * KB[1, 1] + KB[1, 2]
        gx = uB / (W - 1) * 2 - 1; gy = vB / (H - 1) * 2 - 1
        samp = F.grid_sample(pB.permute(2, 0, 1)[None], torch.stack([gx, gy], -1)[None],
                             align_corners=True, padding_mode="zeros")[0].permute(1, 2, 0)
        dd = (samp - photoA).abs().mean(-1)
        dd[(camB[..., 2] <= 0) | (gx.abs() > 1) | (gy.abs() > 1)] = float("nan")
        dis.append(dd)
    return torch.nan_to_num(torch.nanmedian(torch.stack(dis), 0).values, nan=0.0)


def detect_mask(sp, idx, data, C):
    b = load_image(data, idx); photo = b["image"].to(dev)
    K = b["K"].to(dev); c2w = b["c2w"].to(dev); H, W = photo.shape[:2]
    rr, a = render(sp, c2w, K, W, H, "RGB+ED")
    depth = rr[..., 3]
    obj = (a > 0.5).cpu().numpy()
    err = (rr[..., :3].clamp(0, 1) - photo).abs().mean(-1)
    dvec = (C - C[idx]).norm(dim=1); dvec[idx] = 1e9
    nbrs = torch.argsort(dvec)[:4].tolist()
    warp = warp_disagree(sp, photo, depth, K, c2w, nbrs, data)
    defect = (err * warp).cpu().numpy(); defect[~obj] = 0
    if obj.sum() < 100:
        return None
    # subject-centering prior: object scans frame the subject centrally -> down-weight
    # the image periphery so room/background clutter doesn't dominate the threshold
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2
    sig = 0.42 * math.hypot(H, W) / 2
    cw = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2)))
    defect = defect * cw
    thr = np.percentile(defect[obj], 97.0)
    m = (defect > max(thr, 1e-5)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    # keep only the largest contiguous defect(s): the genuine hole is one big blob,
    # background clutter is scattered small/medium components -> drop them
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    floor = 0.01 * obj.sum()
    areas = sorted([(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)], reverse=True)
    for a_, i in areas[:2]:                       # top-2 largest blobs only
        if a_ >= floor:
            keep[lab == i] = 255
    keep = cv2.dilate(keep, np.ones((9, 9), np.uint8))
    keep[~obj] = 0
    area = (keep > 0).sum()
    return dict(idx=idx, mask=keep, photo=photo, K=K, c2w=c2w, depth=depth, a=a,
               H=H, W=W, area=area, objarea=int(obj.sum()),
               defect=float(defect[obj].mean()))


def pick_views(sp, data, C):
    """Rank views by detected hole area; return top NVIEWS with real holes."""
    cands = list(data["test_indices"])[::3] + [256, 255, 257]
    cands = sorted(set(int(i) for i in cands))
    scored = []
    for i in cands:
        d = detect_mask(sp, i, data, C)
        if d and d["area"] > 0.003 * d["objarea"]:
            scored.append((d["area"], i, d))
            print(f"  view {i:3d}: hole area {d['area']:>7d} "
                  f"({100*d['area']/d['objarea']:.1f}% of object)")
    scored.sort(reverse=True)
    return [d for _, _, d in scored[:NVIEWS]]


def write_inputs(d, vdir):
    os.makedirs(vdir, exist_ok=True)
    valid = (d["a"] > 0.5) & (d["depth"] > 1e-6)
    disp = torch.where(valid, 1.0 / d["depth"].clamp(min=1e-6), torch.zeros_like(d["depth"]))
    cv2.imwrite(os.path.join(vdir, "rgb.png"),
                cv2.cvtColor((d["photo"].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(vdir, "mask.png"), d["mask"])
    np.save(os.path.join(vdir, "disp.npy"), disp.cpu().numpy().astype(np.float32))
    np.save(os.path.join(vdir, "intri.npy"), d["K"].cpu().numpy().astype(np.float64))
    np.save(os.path.join(vdir, "c2w.npy"), d["c2w"].cpu().numpy().astype(np.float64))


def run_infusion(vdir):
    out = os.path.join(vdir, "out"); os.makedirs(out, exist_ok=True)
    cmd = [r"C:\Users\gevor\anaconda3\envs\difix\python.exe",
           "run_inference_inpainting.py",
           "--input_rgb_path", os.path.join(vdir, "rgb.png"),
           "--input_mask", os.path.join(vdir, "mask.png"),
           "--input_depth_path", os.path.join(vdir, "disp.npy"),
           "--model_path", MODEL, "--output_dir", out, "--denoise_steps", "20",
           "--intri", os.path.join(vdir, "intri.npy"),
           "--c2w", os.path.join(vdir, "c2w.npy"), "--use_mask", "--blend"]
    subprocess.run(cmd, cwd=RUN_DIR, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return os.path.join(out, "rgb_mask.ply")


def metrics(x, ref):
    p = 10 * math.log10(1.0 / (torch.mean((x - ref) ** 2).item() + 1e-9))
    s = ssim(x.permute(2, 0, 1)[None], ref.permute(2, 0, 1)[None]).item()
    return p, s


# ---------------------------------------------------------------------------
holed = {k: v.to(dev) for k, v in torch.load(os.path.join(IO2, "holed.pt"), weights_only=True)["splats"].items()}
data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)
C = torch.tensor(np.stack([data["camtoworlds"][i][:3, 3] for i in range(len(data["camtoworlds"]))]),
                 device=dev).float()
mode = sys.argv[1] if len(sys.argv) > 1 else "detect"

print(f"[{mode}] ranking views by auto-detected hole area...")
views = pick_views(holed, data, C)
print(f"selected {len(views)} views: {[d['idx'] for d in views]}")

if mode == "detect":
    panels = []
    for d in views[:4]:
        img = (d["photo"].cpu().numpy() * 255).astype(np.uint8).copy()
        ov = img.copy(); ov[d["mask"] > 0] = (255, 0, 0)
        img = cv2.addWeighted(img, 0.6, ov, 0.4, 0)
        cv2.putText(img, f"v{d['idx']}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        panels.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(AUTO, "auto_masks.png"), np.concatenate(panels, 1))
    print("viz:", os.path.join(AUTO, "auto_masks.png"))
    sys.exit(0)

# ---- run: fill each view with InFusion, collect points ----
allpts, allcol = [], []
for d in views:
    vdir = os.path.join(AUTO, f"v{d['idx']}")
    write_inputs(d, vdir)
    ply = run_infusion(vdir)
    raw = np.loadtxt(ply, skiprows=10)
    if raw.ndim == 1 or len(raw) == 0:
        continue
    allpts.append(raw[:, :3]); allcol.append(raw[:, 3:6] / 255.0)
    print(f"  view {d['idx']}: +{len(raw):,} points")
pts = torch.from_numpy(np.concatenate(allpts)).float().to(dev)
col = torch.from_numpy(np.concatenate(allcol)).float().to(dev)
if pts.shape[0] > 200000:
    s = torch.randperm(pts.shape[0], device=dev)[:200000]; pts, col = pts[s], col[s]
M = pts.shape[0]
print(f"total injected points: {M:,}")

vs = 0.02 * float(data["scene_scale"])
new = {"means": pts.clone(),
       "scales": torch.full((M, 3), math.log(vs * 0.5), device=dev),
       "quats": torch.tensor([1.0, 0, 0, 0], device=dev).repeat(M, 1),
       "opacities": torch.full((M,) + tuple(holed["opacities"].shape[1:]), -2.197, device=dev),
       "sh0": ((col - 0.5) / C0).reshape(M, 1, 3),
       "shN": torch.zeros((M,) + tuple(holed["shN"].shape[1:]), device=dev)}

full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
b = load_image(data, 256); photo = b["image"].to(dev); Kt = b["K"].to(dev); c2wt = b["c2w"].to(dev)
H, W = photo.shape[:2]
rgb_full = render(full, c2wt, Kt, W, H, "RGB")[0][..., :3].clamp(0, 1).detach()
rgb_holed = render(holed, c2wt, Kt, W, H, "RGB")[0][..., :3].clamp(0, 1).detach()


def combine(tr):
    return {k: torch.cat([holed[k], tr[k]], 0) for k in holed}


filled_raw = render(combine(new), c2wt, Kt, W, H, "RGB")[0][..., :3].clamp(0, 1).detach()

params = {k: torch.nn.Parameter(v.clone()) for k, v in new.items()}
opt = torch.optim.Adam([
    {"params": [params["means"]], "lr": 1.6e-4 * 1.772}, {"params": [params["scales"]], "lr": 5e-3},
    {"params": [params["quats"]], "lr": 1e-3}, {"params": [params["opacities"]], "lr": 5e-2},
    {"params": [params["sh0"]], "lr": 2.5e-3}, {"params": [params["shN"]], "lr": 1.25e-4}])
tr_idx = list(data["train_indices"]); g = torch.Generator().manual_seed(0)
for step in range(800):
    j = tr_idx[int(torch.randint(len(tr_idx), (1,), generator=g))]
    bt = load_image(data, j); gt = bt["image"].to(dev); K = bt["K"].to(dev); c2w = bt["c2w"].to(dev)
    h, w = gt.shape[:2]
    r = render_splats(combine(params), torch.linalg.inv(c2w), K, w, h, SH,
                      torch.zeros(3, device=dev), "antialiased")[0][0].clamp(0, 1)
    loss = 0.8 * (r - gt).abs().mean() + 0.2 * (1 - ssim(r.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 200 == 0:
        print(f"  distill step {step}  loss {loss.item():.4f}")

filled_opt = render(combine(params), c2wt, Kt, W, H, "RGB")[0][..., :3].clamp(0, 1).detach()
print("\n            vs FULL-splat (GT)          vs REAL photo")
for nm, img in [("holed     ", rgb_holed), ("auto-raw  ", filled_raw), ("auto-opt  ", filled_opt)]:
    pf, sf = metrics(img, rgb_full); pr, sr = metrics(img, photo)
    print(f"{nm}  PSNR {pf:6.2f} SSIM {sf:.4f}    PSNR {pr:6.2f} SSIM {sr:.4f}")
panel = torch.cat([rgb_full, rgb_holed, filled_raw, filled_opt], 1).cpu().numpy()
cv2.imwrite(os.path.join(AUTO, "auto_result.png"),
            cv2.cvtColor((panel * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
print("viz:", os.path.join(AUTO, "auto_result.png"), "(FULL | HOLED | auto-raw | auto-opt)")
