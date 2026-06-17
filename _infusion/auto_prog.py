"""Progressive automatic completion: fill the worst hole -> distill -> re-detect on
the UPDATED splat -> fill next-worst -> ... until the hole-field converges. Each
fill sees the already-corrected geometry => mutually consistent (no raw multi-sheet
pileup). Validates on the holed GPU scan (full splat = GT).

Usage: python auto_prog.py [max_rounds]
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
AUTO = r"F:\HYPERclone\apex-sfm\_infusion\prog"
RUN_DIR = r"F:\HYPERclone\infusion\depth_inpainting\run"
MODEL = r"F:\HYPERclone\infusion\checkpoints\Infusion"
DIFIX_PY = r"C:\Users\gevor\anaconda3\envs\difix\python.exe"
SH, dev = 3, "cuda"
FACTOR = 4
C0 = 0.28209479177387814
MAXR = int(sys.argv[1]) if len(sys.argv) > 1 else 6
DISTILL = 300
os.makedirs(AUTO, exist_ok=True)


def render(sp, c2w, K, W, H, mode="RGB+ED"):
    r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH,
                            torch.zeros(3, device=dev), "antialiased", render_mode=mode)
    return r[0], a[0, ..., 0]


def warp_disagree(photoA, dA, KA, c2wA, nbrs, data):
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
        gx = ((camB[..., 0] / zB) * KB[0, 0] + KB[0, 2]) / (W - 1) * 2 - 1
        gy = ((camB[..., 1] / zB) * KB[1, 1] + KB[1, 2]) / (H - 1) * 2 - 1
        samp = F.grid_sample(pB.permute(2, 0, 1)[None], torch.stack([gx, gy], -1)[None],
                             align_corners=True, padding_mode="zeros")[0].permute(1, 2, 0)
        dd = (samp - photoA).abs().mean(-1)
        dd[(camB[..., 2] <= 0) | (gx.abs() > 1) | (gy.abs() > 1)] = float("nan")
        dis.append(dd)
    return torch.nan_to_num(torch.nanmedian(torch.stack(dis), 0).values, nan=0.0)


def detect(sp, idx, data, C):
    b = load_image(data, idx); photo = b["image"].to(dev)
    K = b["K"].to(dev); c2w = b["c2w"].to(dev); H, W = photo.shape[:2]
    rr, a = render(sp, c2w, K, W, H, "RGB+ED")
    depth = rr[..., 3]; obj = (a > 0.5).cpu().numpy()
    if obj.sum() < 100:
        return None
    err = (rr[..., :3].clamp(0, 1) - photo).abs().mean(-1)
    dvec = (C - C[idx]).norm(dim=1); dvec[idx] = 1e9
    nbrs = torch.argsort(dvec)[:4].tolist()
    warp = warp_disagree(photo, depth, K, c2w, nbrs, data)
    defect = (err * warp).cpu().numpy(); defect[~obj] = 0
    yy, xx = np.mgrid[0:H, 0:W]
    sig = 0.42 * math.hypot(H, W) / 2
    defect = defect * np.exp(-(((yy - H / 2) ** 2 + (xx - W / 2) ** 2) / (2 * sig ** 2)))
    thr = np.percentile(defect[obj], 97.0)
    m = (defect > max(thr, 1e-5)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m); floor = 0.004 * obj.sum()
    for a_, i in sorted([(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)], reverse=True)[:2]:
        if a_ >= floor:
            keep[lab == i] = 255
    keep = cv2.dilate(keep, np.ones((9, 9), np.uint8)); keep[~obj] = 0
    return dict(idx=idx, mask=keep, photo=photo, K=K, c2w=c2w, depth=depth, a=a,
               H=H, W=W, area=int((keep > 0).sum()), objarea=int(obj.sum()))


def fill_view(d, rdir):
    os.makedirs(rdir, exist_ok=True)
    valid = (d["a"] > 0.5) & (d["depth"] > 1e-6)
    disp = torch.where(valid, 1.0 / d["depth"].clamp(min=1e-6), torch.zeros_like(d["depth"]))
    cv2.imwrite(os.path.join(rdir, "rgb.png"),
                cv2.cvtColor((d["photo"].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(rdir, "mask.png"), d["mask"])
    np.save(os.path.join(rdir, "disp.npy"), disp.cpu().numpy().astype(np.float32))
    np.save(os.path.join(rdir, "intri.npy"), d["K"].cpu().numpy().astype(np.float64))
    np.save(os.path.join(rdir, "c2w.npy"), d["c2w"].cpu().numpy().astype(np.float64))
    out = os.path.join(rdir, "out"); os.makedirs(out, exist_ok=True)
    subprocess.run([DIFIX_PY, "run_inference_inpainting.py",
                    "--input_rgb_path", os.path.join(rdir, "rgb.png"),
                    "--input_mask", os.path.join(rdir, "mask.png"),
                    "--input_depth_path", os.path.join(rdir, "disp.npy"),
                    "--model_path", MODEL, "--output_dir", out, "--denoise_steps", "20",
                    "--intri", os.path.join(rdir, "intri.npy"),
                    "--c2w", os.path.join(rdir, "c2w.npy"), "--use_mask", "--blend"],
                   cwd=RUN_DIR, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raw = np.loadtxt(os.path.join(out, "rgb_mask.ply"), skiprows=10)
    return raw if raw.ndim == 2 and len(raw) else None


def as_gaussians(raw, ref):
    pts = torch.from_numpy(raw[:, :3]).float().to(dev)
    col = torch.from_numpy(raw[:, 3:6]).float().to(dev) / 255.0
    if pts.shape[0] > 60000:
        s = torch.randperm(pts.shape[0], device=dev)[:60000]; pts, col = pts[s], col[s]
    M = pts.shape[0]; vs = 0.02 * 1.772
    return {"means": pts, "scales": torch.full((M, 3), math.log(vs * 0.5), device=dev),
            "quats": torch.tensor([1.0, 0, 0, 0], device=dev).repeat(M, 1),
            "opacities": torch.full((M,) + tuple(ref["opacities"].shape[1:]), -2.197, device=dev),
            "sh0": ((col - 0.5) / C0).reshape(M, 1, 3),
            "shN": torch.zeros((M,) + tuple(ref["shN"].shape[1:]), device=dev)}


def metrics(x, ref):
    p = 10 * math.log10(1.0 / (torch.mean((x - ref) ** 2).item() + 1e-9))
    return p, ssim(x.permute(2, 0, 1)[None], ref.permute(2, 0, 1)[None]).item()


# ---------------------------------------------------------------------------
holed = {k: v.to(dev) for k, v in torch.load(os.path.join(IO2, "holed.pt"), weights_only=True)["splats"].items()}
full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)
C = torch.tensor(np.stack([data["camtoworlds"][i][:3, 3] for i in range(len(data["camtoworlds"]))]),
                 device=dev).float()
cand = sorted(set(int(i) for i in list(data["test_indices"])[::2]) | {255, 256, 257, 264})
tr_idx = list(data["train_indices"]); g = torch.Generator().manual_seed(0)

b = load_image(data, 256); photo256 = b["image"].to(dev); Kt = b["K"].to(dev); c2wt = b["c2w"].to(dev)
H, W = photo256.shape[:2]
rgb_full = render(full, c2wt, Kt, W, H, "RGB")[0][..., :3].clamp(0, 1).detach()
rgb_holed = render(holed, c2wt, Kt, W, H, "RGB")[0][..., :3].clamp(0, 1).detach()
print(f"holed: PSNR {metrics(rgb_holed, rgb_full)[0]:.2f} vs full; {len(cand)} candidate views")

acc = None                                            # accumulated optimized fill gaussians
done = []
for r in range(MAXR):
    cur = holed if acc is None else {k: torch.cat([holed[k], acc[k]], 0) for k in holed}
    scored = []
    for i in cand:
        if i in done:
            continue
        d = detect(cur, i, data, C)
        if d and d["area"] > 0.004 * d["objarea"]:
            scored.append((d["area"], d))
    if not scored:
        print(f"round {r}: hole-field converged (no view above floor)"); break
    scored.sort(key=lambda t: t[0], reverse=True)
    d = scored[0][1]; done.append(d["idx"])
    raw = fill_view(d, os.path.join(AUTO, f"r{r}_v{d['idx']}"))
    if raw is None:
        continue
    newg = as_gaussians(raw, holed)
    acc = newg if acc is None else {k: torch.cat([acc[k], newg[k]], 0) for k in newg}
    # distill all accumulated fill gaussians (frozen background)
    params = {k: torch.nn.Parameter(v.clone()) for k, v in acc.items()}
    opt = torch.optim.Adam([
        {"params": [params["means"]], "lr": 1.6e-4 * 1.772}, {"params": [params["scales"]], "lr": 5e-3},
        {"params": [params["quats"]], "lr": 1e-3}, {"params": [params["opacities"]], "lr": 5e-2},
        {"params": [params["sh0"]], "lr": 2.5e-3}, {"params": [params["shN"]], "lr": 1.25e-4}])
    for step in range(DISTILL):
        j = tr_idx[int(torch.randint(len(tr_idx), (1,), generator=g))]
        bt = load_image(data, j); gt = bt["image"].to(dev); K = bt["K"].to(dev); c2w = bt["c2w"].to(dev)
        h, w = gt.shape[:2]
        comb = {k: torch.cat([holed[k], params[k]], 0) for k in holed}
        rr = render_splats(comb, torch.linalg.inv(c2w), K, w, h, SH,
                           torch.zeros(3, device=dev), "antialiased")[0][0].clamp(0, 1)
        loss = 0.8 * (rr - gt).abs().mean() + 0.2 * (1 - ssim(rr.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        opt.zero_grad(); loss.backward(); opt.step()
    acc = {k: params[k].detach() for k in params}
    cur = {k: torch.cat([holed[k], acc[k]], 0) for k in holed}
    pm = metrics(render(cur, c2wt, Kt, W, H, "RGB")[0][..., :3].clamp(0, 1), rgb_full)
    print(f"round {r}: filled v{d['idx']} (area {d['area']}, +{len(newg['means']):,}g) "
          f"-> total {len(acc['means']):,}g  PSNR-256 {pm[0]:.2f} SSIM {pm[1]:.4f}")

cur = {k: torch.cat([holed[k], acc[k]], 0) for k in holed} if acc else holed
final = render(cur, c2wt, Kt, W, H, "RGB")[0][..., :3].clamp(0, 1).detach()
pf, sf = metrics(final, rgb_full); pr, sr = metrics(final, photo256)
print(f"\nFINAL prog: PSNR {pf:.2f} SSIM {sf:.4f} vs full; PSNR {pr:.2f} SSIM {sr:.4f} vs photo")
print(f"(holed {metrics(rgb_holed, rgb_full)[0]:.2f} -> prog {pf:.2f}; manual single-view was 26.88; "
      f"multi-view-batch was 26.72)")
panel = torch.cat([rgb_full, rgb_holed, final], 1).cpu().numpy()
cv2.imwrite(os.path.join(AUTO, "prog_result.png"),
            cv2.cvtColor((panel * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
torch.save({"splats": {k: v.cpu() for k, v in cur.items()}}, os.path.join(AUTO, "completed.pt"))
print("viz:", os.path.join(AUTO, "prog_result.png"), "(FULL | HOLED | PROG)")
