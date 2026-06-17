"""Automatic hole / under-observation DETECTOR + classifier (validation harness).

Three signals, all from machinery we already have:
  1. GEOMETRIC HOLES (Type 1/3): render alpha from real views; an interior hole =
     low-alpha pixel enclosed by the object silhouette (cv2 flood-fill). Genuine
     missing surface.
  2. OCCLUSION-AWARE COVERAGE (Type 2 vs 4): for every Gaussian, count how many real
     cameras see it AS THE FRONT SURFACE (project -> compare camera-z to that camera's
     rendered depth). Low count = under-observed; high = reliable/well-observed.
  3. REPROJECT-COVERAGE of a hole (Type 1 vs 3): estimate the hole's depth by
     inpainting surrounding depth, back-project, and count how many OTHER cameras
     observe that surface. >=1 -> faithfully fillable (Type 1); 0 -> unseen (Type 3).

VALIDATION (known answer): on the artificially-holed GPU splat the detector must
(a) light up the deleted region, (b) classify it Type 1 (other cams saw it),
(c) NOT flag the intact full splat / well-observed surfaces.

Usage: python hole_detect.py
"""
import math
import os

import numpy as np
import torch
import cv2

from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\output\ckpts\ckpt_29999_rank0.pt"
IO2 = r"F:\HYPERclone\apex-sfm\_infusion\io2"
OUT = r"F:\HYPERclone\apex-sfm\_infusion\detect"
SH, dev = 3, "cuda"
FACTOR = 4
C0 = 0.28209479177387814
os.makedirs(OUT, exist_ok=True)


def render(sp, c2w, K, W, H, mode="RGB+ED"):
    r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH,
                            torch.zeros(3, device=dev), "antialiased", render_mode=mode)
    return r[0], a[0, ..., 0]


def interior_holes(alpha_np, k=9):
    """Low-alpha pixels enclosed by the object silhouette (= genuine surface gaps)."""
    obj = (alpha_np > 0.5).astype(np.uint8) * 255
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # denoise
    ff = obj.copy()
    m = np.zeros((obj.shape[0] + 2, obj.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 255)            # flood background from corner
    enclosed = cv2.bitwise_not(ff)               # interior not reached from border
    # also catch near-silhouette gaps via closing
    closed = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((k * 3, k * 3), np.uint8))
    near_edge = cv2.bitwise_and(closed, cv2.bitwise_not(obj))
    holes = cv2.bitwise_or(enclosed, near_edge)
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))  # drop specks
    return holes > 0


full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
holed = {k: v.to(dev) for k, v in torch.load(os.path.join(IO2, "holed.pt"), weights_only=True)["splats"].items()}
data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)
scene = float(data["scene_scale"])
all_idx = list(range(len(data["camtoworlds"])))

# ---------- 1. GEOMETRIC HOLES: full vs holed across views ----------
print("=== 1. GEOMETRIC HOLE DETECTION (interior alpha-deficit) ===")
view_set = sorted(set([256] + list(data["test_indices"])[::6]))[:14]
tot_full = tot_holed = npix = 0
for vi in view_set:
    b = load_image(data, vi); K = b["K"].to(dev); c2w = b["c2w"].to(dev)
    H, W = b["image"].shape[:2]; npix += H * W
    _, af = render(full, c2w, K, W, H, "RGB")
    _, ah = render(holed, c2w, K, W, H, "RGB")
    hf = interior_holes(af.cpu().numpy()); hh = interior_holes(ah.cpu().numpy())
    tot_full += hf.sum(); tot_holed += hh.sum()
print(f"views={len(view_set)}  interior-hole px:  FULL {tot_full:>8d} ({100*tot_full/npix:.3f}%)"
      f"   HOLED {tot_holed:>8d} ({100*tot_holed/npix:.3f}%)   ratio {tot_holed/max(tot_full,1):.1f}x")

# viz at the hole view
b = load_image(data, 256); K = b["K"].to(dev); c2w = b["c2w"].to(dev)
H, W = b["image"].shape[:2]
panels = []
for sp, nm in [(full, "FULL"), (holed, "HOLED")]:
    rgb, a = render(sp, c2w, K, W, H, "RGB")
    img = (rgb[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).copy()
    hm = interior_holes(a.cpu().numpy())
    img[hm] = (0, 0, 255)                                   # paint detected holes red
    cv2.putText(img, nm + " holes", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    panels.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUT, "1_holes.png"), np.concatenate(panels, 1))

# ---------- 2. OCCLUSION-AWARE PER-GAUSSIAN COVERAGE (full splat) ----------
print("\n=== 2. OCCLUSION-AWARE COVERAGE per Gaussian (front-surface visibility) ===")
means = full["means"]; M = means.shape[0]
ones = torch.ones(M, 1, device=dev)
Xh = torch.cat([means, ones], 1)                            # (M,4)
cov = torch.zeros(M, device=dev)
tol = 0.015 * scene
b0 = load_image(data, 256); Wc, Hc = b0["image"].shape[1] // 2, b0["image"].shape[0] // 2
for i in all_idx:
    bi = load_image(data, i); K = bi["K"].to(dev).clone(); c2w = bi["c2w"].to(dev)
    K[:2] *= 0.5                                            # half-res coverage depth
    rr, aa = render(full, c2w, K, Wc, Hc, "RGB+ED")
    depth = rr[..., 3]
    w2c = torch.linalg.inv(c2w)
    xc = (Xh @ w2c.T)[:, :3]
    z = xc[:, 2]
    uv = (xc / z.clamp(min=1e-6)[:, None]) @ K.T
    u = uv[:, 0].long(); v = uv[:, 1].long()
    ok = (z > 0) & (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)
    uu = u.clamp(0, Wc - 1); vv = v.clamp(0, Hc - 1)
    dz = depth[vv, uu]; av = aa[vv, uu]
    seen = ok & (av > 0.5) & ((z - dz).abs() < tol)         # front surface in this cam
    cov += seen.float()
cov_np = cov.cpu().numpy()
qs = np.percentile(cov_np, [1, 5, 25, 50, 75, 95])
print(f"coverage count per Gaussian (over {len(all_idx)} cams): "
      f"min {cov_np.min():.0f}  p1 {qs[0]:.0f}  p5 {qs[1]:.0f}  p25 {qs[2]:.0f}  "
      f"med {qs[3]:.0f}  p75 {qs[4]:.0f}  p95 {qs[5]:.0f}  max {cov_np.max():.0f}")
print(f"  never-seen (cov==0): {100*(cov_np==0).mean():.2f}%   "
      f"under-observed (cov<=3): {100*(cov_np<=3).mean():.2f}%   "
      f"well-observed (cov>=20): {100*(cov_np>=20).mean():.2f}%")

# coverage heatmap render at view 256 (turbo: blue=under-observed, red=well-observed)
cn = np.clip(cov_np / np.percentile(cov_np, 95), 0, 1)
colors = cv2.applyColorMap((cn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, 0, ::-1] / 255.0
heat = {k: v.clone() for k, v in full.items()}
heat["sh0"] = ((torch.from_numpy(colors.copy()).float().to(dev) - 0.5) / C0).reshape(M, 1, 3)
heat["shN"] = torch.zeros_like(full["shN"])
rgb_h, _ = render(heat, c2w, K * 2 if False else b["K"].to(dev), W, H, "RGB")
cv2.imwrite(os.path.join(OUT, "2_coverage.png"),
            cv2.cvtColor((rgb_h[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                         cv2.COLOR_RGB2BGR))

# ---------- 3. REPROJECT-COVERAGE: classify the deleted region (Type 1 vs 3) ----------
print("\n=== 3. HOLE CLASSIFICATION (reproject-coverage of the holed region) ===")
b = load_image(data, 256); K = b["K"].to(dev); c2w = b["c2w"].to(dev)
H, W = b["image"].shape[:2]
rr, ah = render(holed, c2w, K, W, H, "RGB+ED")
depth_h = rr[..., 3].cpu().numpy()
hole = interior_holes(ah.cpu().numpy())
# estimate hole depth by inpainting surrounding rendered depth
dmask = (hole & (depth_h <= 1e-6)).astype(np.uint8)
known = np.where(depth_h > 1e-6, depth_h, 0).astype(np.float32)
dfill = cv2.inpaint((known / (known.max() + 1e-6) * 255).astype(np.uint8),
                    dmask, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255.0 * (known.max() + 1e-6)
ys, xs = np.nonzero(hole)
if len(ys) > 4000:
    s = np.random.RandomState(0).choice(len(ys), 4000, replace=False); ys, xs = ys[s], xs[s]
d = dfill[ys, xs]
Kn = K.cpu().numpy(); c2wn = c2w.cpu().numpy()
camp = np.stack([(xs - Kn[0, 2]) / Kn[0, 0] * d, (ys - Kn[1, 2]) / Kn[1, 1] * d, d, np.ones_like(d)], 1)
world = torch.from_numpy((camp @ c2wn.T)[:, :3]).float().to(dev)
Xh2 = torch.cat([world, torch.ones(len(world), 1, device=dev)], 1)
obs = torch.zeros(len(world), device=dev)
for i in all_idx:
    if i == 256:
        continue
    bi = load_image(data, i); Ki = bi["K"].to(dev).clone(); c2wi = bi["c2w"].to(dev)
    Ki[:2] *= 0.5
    rr2, aa2 = render(full, c2wi, Ki, Wc, Hc, "RGB+ED")
    depth2 = rr2[..., 3]
    w2c = torch.linalg.inv(c2wi)
    xc = (Xh2 @ w2c.T)[:, :3]; z = xc[:, 2]
    uv = (xc / z.clamp(min=1e-6)[:, None]) @ Ki.T
    u = uv[:, 0].long().clamp(0, Wc - 1); v = uv[:, 1].long().clamp(0, Hc - 1)
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < Wc) & (uv[:, 1] >= 0) & (uv[:, 1] < Hc)
    seen = ok & (aa2[v, u] > 0.5) & ((z - depth2[v, u]).abs() < tol)
    obs += seen.float()
obs_np = obs.cpu().numpy()
type1 = (obs_np >= 1).mean()
print(f"hole surface points: {len(world)}  mean observing cams {obs_np.mean():.1f}  "
      f"median {np.median(obs_np):.0f}")
print(f"  Type-1 (fillable from >=1 real view): {100*type1:.1f}%   "
      f"Type-3 (unseen, 0 views): {100*(obs_np==0).mean():.1f}%")
print(f"\nviz: {OUT}\\1_holes.png (red=detected holes)  2_coverage.png (blue=under-observed)")
