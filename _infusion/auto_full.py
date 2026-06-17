"""Full auto-complete ENGINE (the GUI will call this): edge-gated detect -> InFusion
seed -> progressive distill -> detail-refine (+densify). Parameterized for any scan.

Usage:
  python auto_full.py laptop          # carve a hole in the Laptop, recover, evaluate
  python auto_full.py gpu             # same on GPU scan
  python auto_full.py laptop real     # run on the REAL (un-holed) splat (safety/inertness)

Config switch at top. For the carve test the ORIGINAL full splat = ground truth.
"""
import math
import os
import subprocess
import sys

import numpy as np
import torch
import cv2

from core.training.gsplat_trainer import (load_colmap_data, load_image, render_splats,
                                          ssim, get_lpips)

RUN_DIR = r"F:\HYPERclone\infusion\depth_inpainting\run"
MODEL = r"F:\HYPERclone\infusion\checkpoints\Infusion"
DIFIX_PY = r"C:\Users\gevor\anaconda3\envs\difix\python.exe"
SH, dev, C0 = 3, "cuda", 0.28209479177387814

CFG = {
    "gpu": dict(
        DATA=r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data",
        CK=r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\output\ckpts\ckpt_29999_rank0.pt",
        factor=4, center_prior=True),
    "laptop": dict(
        DATA=r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\data",
        CK=r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\output_4k_gated\ckpts\ckpt_29999_rank0.pt",
        factor=2, center_prior=False),
}
which = sys.argv[1] if len(sys.argv) > 1 else "laptop"
REAL = len(sys.argv) > 2 and sys.argv[2] == "real"
cfg = CFG[which]
OUT = rf"F:\HYPERclone\apex-sfm\_infusion\full_{which}"
os.makedirs(OUT, exist_ok=True)


def render(sp, c2w, K, W, H, mode="RGB+ED"):
    r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH,
                            torch.zeros(3, device=dev), "antialiased", render_mode=mode)
    return r[0], a[0, ..., 0]


def warp_disagree(photoA, dA, KA, c2wA, nbrs, data):
    H, W = dA.shape
    import torch.nn.functional as F
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


def footprint(fill, c2w, K, W, H):
    with torch.no_grad():
        _, a = render(fill, c2w, K, W, H, "RGB")
    m = (a > 0.1).cpu().numpy()
    if m.sum() < 1500:
        return None
    ys, xs = np.nonzero(m); return xs.min(), ys.min(), xs.max(), ys.max()


def cbox(box, W, H, pad=24, mn=256):
    x0, y0, x1, y1 = box; x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    if x1 - x0 < mn: c = (x0 + x1) // 2; x0, x1 = c - mn // 2, c + mn // 2
    if y1 - y0 < mn: c = (y0 + y1) // 2; y0, y1 = c - mn // 2, c + mn // 2
    x0 = max(0, min(x0, W - 1)); x1 = max(x0 + 1, min(x1, W))
    y0 = max(0, min(y0, H - 1)); y1 = max(y0 + 1, min(y1, H))
    return int(x0), int(y0), int(x1), int(y1)


def fill_gain(holed, acc_before, acc_new, newg, fill_idx, data, C):
    """Accept a fill only if it improves agreement with the REAL photos on held-out
    NEIGHBOUR views (not the InFusion RGB-source view). Returns mean L1 improvement;
    >0 => the fill is genuinely better geometry, not over-filling decent geometry."""
    prev_sp = holed if acc_before is None else {k: torch.cat([holed[k], acc_before[k]], 0) for k in holed}
    new_sp = {k: torch.cat([holed[k], acc_new[k]], 0) for k in holed}
    order = torch.argsort((C - C[fill_idx]).norm(dim=1)).tolist()
    gains = []
    for v in order:
        if v == fill_idx or len(gains) >= 3:
            continue
        b = load_image(data, v); Kv = b["K"].to(dev); c2wv = b["c2w"].to(dev)
        pv = b["image"].to(dev); Hv, Wv = pv.shape[:2]
        fp = footprint(newg, c2wv, Kv, Wv, Hv)
        if fp is None:
            continue
        x0, y0, x1, y1 = cbox(fp, Wv, Hv)
        rp = render(prev_sp, c2wv, Kv, Wv, Hv, "RGB")[0][..., :3].clamp(0, 1)
        rn = render(new_sp, c2wv, Kv, Wv, Hv, "RGB")[0][..., :3].clamp(0, 1)
        ep = (rp[y0:y1, x0:x1] - pv[y0:y1, x0:x1]).abs().mean().item()
        en = (rn[y0:y1, x0:x1] - pv[y0:y1, x0:x1]).abs().mean().item()
        gains.append(ep - en)
    return (sum(gains) / len(gains)) if gains else -1.0


def detect(sp, idx, data, C, scene, center_prior):
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
    # EDGE-GATE: suppress silhouette + depth-discontinuity false positives (WarpRF
    # always disagrees across a depth cliff; those are edges, not surface holes)
    d = np.where(obj, depth.cpu().numpy(), 0).astype(np.float32)
    gy, gx = np.gradient(d); gmag = np.hypot(gx, gy)
    edge = cv2.dilate((gmag > 0.02 * scene).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    defect[edge] = 0
    if center_prior:
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


def carve(full, data, scene, factor):
    """Pick the most-detailed test view, box its centre, delete a surface shell."""
    best = None
    for idx in list(data["test_indices"]):
        b = load_image(data, idx); ph = b["image"].cpu().numpy()
        H, W = ph.shape[:2]
        c = ph[int(.4 * H):int(.6 * H), int(.4 * W):int(.6 * W)]
        gx = np.abs(np.diff(c.mean(-1), axis=1)).mean()
        if best is None or gx > best[0]:
            best = (gx, idx)
    idx = best[1]
    b = load_image(data, idx); K = b["K"].to(dev); c2w = b["c2w"].to(dev)
    H, W = b["image"].shape[:2]
    rr, a = render(full, c2w, K, W, H, "RGB+ED")
    depth = rr[..., 3]; sel = (a > 0.5) & (depth > 1e-6)
    mask = np.zeros((H, W), bool)
    bw, bh = int(0.18 * W), int(0.18 * H)
    mask[(H - bh) // 2:(H + bh) // 2, (W - bw) // 2:(W + bw) // 2] = True
    sel = sel.cpu().numpy() & mask
    Kn = K.cpu().numpy(); c2wn = c2w.cpu().numpy(); dd = depth.cpu().numpy()
    ys, xs = np.nonzero(sel); dv = dd[ys, xs]
    cam = np.stack([(xs - Kn[0, 2]) / Kn[0, 0] * dv, (ys - Kn[1, 2]) / Kn[1, 1] * dv, dv, np.ones_like(dv)], 1)
    world = (cam @ c2wn.T)[:, :3]
    vs = 0.02 * scene; origin = world.min(0) - 4 * vs
    means = full["means"].cpu().numpy()
    def vkey(p): return np.floor((p - origin) / vs).astype(np.int64)
    surf = np.unique(vkey(world), axis=0)
    offs = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    dil = np.unique((surf[:, None, :] + offs[None]).reshape(-1, 3), axis=0)
    P = 1 << 21
    def lin(x): return ((x[:, 0] + P // 2) * P + (x[:, 1] + P // 2)) * P + (x[:, 2] + P // 2)
    inside = np.isin(lin(vkey(means)), lin(dil))
    keep = torch.from_numpy(~inside).to(dev)
    holed = {k: v[keep].clone() for k, v in full.items()}
    print(f"carved hole at view {idx}: deleted {int(inside.sum()):,} gaussians")
    return holed, idx


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


def as_gaussians(raw, ref, scene):
    pts = torch.from_numpy(raw[:, :3]).float().to(dev)
    col = torch.from_numpy(raw[:, 3:6]).float().to(dev) / 255.0
    if pts.shape[0] > 60000:
        s = torch.randperm(pts.shape[0], device=dev)[:60000]; pts, col = pts[s], col[s]
    M = pts.shape[0]; vs = 0.02 * scene
    return {"means": pts, "scales": torch.full((M, 3), math.log(vs * 0.5), device=dev),
            "quats": torch.tensor([1.0, 0, 0, 0], device=dev).repeat(M, 1),
            "opacities": torch.full((M,) + tuple(ref["opacities"].shape[1:]), -2.197, device=dev),
            "sh0": ((col - 0.5) / C0).reshape(M, 1, 3),
            "shN": torch.zeros((M,) + tuple(ref["shN"].shape[1:]), device=dev)}


def opt_for(state, lr_mean):
    p = {k: torch.nn.Parameter(v.clone()) for k, v in state.items()}
    o = torch.optim.Adam([
        {"params": [p["means"]], "lr": lr_mean}, {"params": [p["scales"]], "lr": 8e-3},
        {"params": [p["quats"]], "lr": 1e-3}, {"params": [p["opacities"]], "lr": 3e-2},
        {"params": [p["sh0"]], "lr": 2.5e-3}, {"params": [p["shN"]], "lr": 1.25e-4}])
    return p, o


def metrics(x, ref, lp=None, crop=None):
    p = 10 * math.log10(1.0 / (torch.mean((x - ref) ** 2).item() + 1e-9))
    s = ssim(x.permute(2, 0, 1)[None], ref.permute(2, 0, 1)[None]).item()
    lv = None
    if lp is not None and crop is not None:
        x0, y0, x1, y1 = crop
        lv = lp(x[y0:y1, x0:x1].permute(2, 0, 1)[None].clamp(0, 1),
                ref[y0:y1, x0:x1].permute(2, 0, 1)[None].clamp(0, 1)).item()
    return p, s, lv


# ===========================================================================
full = {k: v.to(dev) for k, v in torch.load(cfg["CK"], weights_only=True, map_location=dev)["splats"].items()}
data = load_colmap_data(cfg["DATA"], factor=cfg["factor"], test_every=8, normalize=True)
scene = float(data["scene_scale"])
C = torch.tensor(np.stack([data["camtoworlds"][i][:3, 3] for i in range(len(data["camtoworlds"]))]),
                 device=dev).float()
lp = get_lpips(dev)
print(f"[{which}] {full['means'].shape[0]:,} gaussians, {len(data['camtoworlds'])} views, scene {scene:.2f}")

if REAL:
    holed = full; hole_view = int(list(data["test_indices"])[len(data["test_indices"]) // 2])
    print("REAL mode: detecting on the un-holed splat (inertness test)")
else:
    holed = None
    holed, hole_view = carve(full, data, scene, cfg["factor"])

# ---- progressive fill ----
cand = sorted(set(int(i) for i in list(data["test_indices"])[::2]) | {hole_view})
tr_idx = list(data["train_indices"]); g = torch.Generator().manual_seed(0)
acc = None; done = []
for r in range(6):
    cur = holed if acc is None else {k: torch.cat([holed[k], acc[k]], 0) for k in holed}
    scored = []
    for i in cand:
        if i in done:
            continue
        d = detect(cur, i, data, C, scene, cfg["center_prior"])
        if d and d["area"] > 0.004 * d["objarea"]:
            scored.append((d["area"], d))
    if not scored:
        print(f"round {r}: converged"); break
    scored.sort(key=lambda t: t[0], reverse=True); d = scored[0][1]; done.append(d["idx"])
    raw = fill_view(d, os.path.join(OUT, f"r{r}_v{d['idx']}"))
    if raw is None:
        continue
    acc_before = acc
    newg = as_gaussians(raw, holed, scene)
    trial = newg if acc is None else {k: torch.cat([acc[k], newg[k]], 0) for k in newg}
    params, opt = opt_for(trial, 1.6e-4 * scene)
    for step in range(300):
        j = tr_idx[int(torch.randint(len(tr_idx), (1,), generator=g))]
        bt = load_image(data, j); gt = bt["image"].to(dev); K = bt["K"].to(dev); c2w = bt["c2w"].to(dev)
        h, w = gt.shape[:2]
        comb = {k: torch.cat([holed[k], params[k]], 0) for k in holed}
        rr = render_splats(comb, torch.linalg.inv(c2w), K, w, h, SH,
                           torch.zeros(3, device=dev), "antialiased")[0][0].clamp(0, 1)
        loss = 0.8 * (rr - gt).abs().mean() + 0.2 * (1 - ssim(rr.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        opt.zero_grad(); loss.backward(); opt.step()
    acc_new = {k: params[k].detach() for k in params}
    # ACCEPT/REJECT gate: commit only if it improves held-out real-photo agreement
    gain = fill_gain(holed, acc_before, acc_new, newg, d["idx"], data, C)
    if gain > 0:
        acc = acc_new
        print(f"round {r}: filled v{d['idx']} ACCEPT (gain {gain:+.4f}) total {len(acc['means']):,}g")
    else:
        print(f"round {r}: filled v{d['idx']} REJECT (gain {gain:+.4f}) — fill didn't improve, keeping original")

if acc is None:
    print("\nNo holes filled (inert) — done."); sys.exit(0)

# ---- detail-refine (+densify) ----
print("detail-refine (gated-LPIPS multi-view + densify)...")
import torch.nn.functional as F


def footprint(fill, c2w, K, W, H):
    with torch.no_grad():
        _, a = render(fill, c2w, K, W, H, "RGB")
    m = (a > 0.1).cpu().numpy()
    if m.sum() < 1500:
        return None
    ys, xs = np.nonzero(m); return xs.min(), ys.min(), xs.max(), ys.max()


def cbox(box, W, H, pad=24, mn=256):
    x0, y0, x1, y1 = box; x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    if x1 - x0 < mn: c = (x0 + x1) // 2; x0, x1 = c - mn // 2, c + mn // 2
    if y1 - y0 < mn: c = (y0 + y1) // 2; y0, y1 = c - mn // 2, c + mn // 2
    x0 = max(0, min(x0, W - 1)); x1 = max(x0 + 1, min(x1, W))
    y0 = max(0, min(y0, H - 1)); y1 = max(y0 + 1, min(y1, H))
    return int(x0), int(y0), int(x1), int(y1)


fill0 = acc  # committed InFusion seeds (the cold-start)

# ---- RECONSTRUCT with the REAL production trainer (init = splat + committed seeds) ----
# The custom detail-refine produced MUSH (geometry stuck in local minima). The
# production trainer's MCMC inject_noise + relocation + gated-LPIPS migrates Gaussians
# to the TRUE surface and reconstructs faithfully — validated 85% LPIPS-gap recovery
# vs GT photos (eval_vs_photo.py). InFusion only supplies the cold-start seed.
print("reconstruct: production trainer (MCMC + gated LPIPS) on splat+seeds...")
init = {k: torch.cat([holed[k], acc[k]], 0) for k in holed}
init_path = os.path.join(OUT, "recon_init.pt")
torch.save({"splats": {k: v.detach().cpu() for k, v in init.items()}}, init_path)
recon_dir = os.path.join(OUT, "recon")
cap = int(init["means"].shape[0] * 1.15)
env = os.environ.copy(); env["PYTHONPATH"] = r"F:\HYPERclone\apex-sfm"
subprocess.run([sys.executable, r"F:\HYPERclone\apex-sfm\core\training\gsplat_trainer.py",
                "--data_dir", cfg["DATA"], "--result_dir", recon_dir,
                "--data_factor", str(cfg["factor"]), "--max_steps", "5000", "--test_every", "8",
                "--strategy", "mcmc", "--cap_max", str(cap), "--refine_stop_iter", "4500",
                "--init_ckpt", init_path, "--lpips_lambda", "0.2", "--lpips_gated",
                "--lpips_start", "200", "--antialiased", "--random_bkgd", "--save_steps", "5000"],
               check=True, env=env)
final_splat = {k: v.to(dev) for k, v in torch.load(
    os.path.join(recon_dir, "ckpts", "ckpt_4999_rank0.pt"),
    weights_only=True, map_location=dev)["splats"].items()}

# ---- evaluate at the hole view vs FULL (GT) ----
b = load_image(data, hole_view); K = b["K"].to(dev); c2w = b["c2w"].to(dev); H, W = b["image"].shape[:2]
rf = render(full, c2w, K, W, H, "RGB")[0][..., :3].clamp(0, 1).detach()
rh = render(holed, c2w, K, W, H, "RGB")[0][..., :3].clamp(0, 1).detach()
rfin = render(final_splat, c2w, K, W, H, "RGB")[0][..., :3].clamp(0, 1).detach()
crop = cbox(footprint(fill0, c2w, K, W, H) or (W // 3, H // 3, 2 * W // 3, 2 * H // 3), W, H, pad=40, mn=320)
print(f"\n=== {which} hole-view {hole_view} vs FULL (GT) ===")
for nm, im in [("holed", rh), ("auto-complete", rfin)]:
    p, s, l = metrics(im, rf, lp, crop)
    print(f"{nm:16s} PSNR {p:6.2f}  SSIM {s:.4f}  hole-LPIPS {l:.4f}")
zc = []
for im, t in [(rf, "FULL"), (rh, "holed"), (rfin, "auto-complete")]:
    x0, y0, x1, y1 = crop; c = (im[y0:y1, x0:x1].cpu().numpy() * 255).astype(np.uint8)[..., ::-1].copy()
    c = cv2.resize(c, (360, int(360 * c.shape[0] / max(c.shape[1], 1))), interpolation=cv2.INTER_NEAREST)
    cv2.putText(c, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2); zc.append(c)
cv2.imwrite(os.path.join(OUT, "result.png"), np.concatenate(zc, 1))
torch.save({"splats": {k: v.cpu() for k, v in final_splat.items()}}, os.path.join(OUT, "completed.pt"))
print("viz:", os.path.join(OUT, "result.png"), "(FULL | holed | auto-complete)")
