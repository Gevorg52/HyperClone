"""Diagnose the real Laptop splat (no carving): is the fuzzy back UNDER-OBSERVED
(few cameras -> fixable by using the real grazing views better) or CAPTURE-LIMITED
(well-seen but soft -> only generative detail can add to it)? Characterise the orbit,
per-Gaussian coverage, and render the under-observed direction so we target the right
tool."""
import math
import os
import numpy as np
import torch
import cv2
from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\output_4k_gated\ckpts\ckpt_29999_rank0.pt"
OUT = r"F:\HYPERclone\apex-sfm\_infusion\diag_back"
SH, dev, C0 = 3, "cuda", 0.28209479177387814
os.makedirs(OUT, exist_ok=True)


def render(sp, c2w, K, W, H, mode="RGB"):
    r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH, torch.zeros(3, device=dev), "antialiased", render_mode=mode)
    return r[0], a[0, ..., 0]


full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True)
scene = float(data["scene_scale"])
cen = np.stack([data["camtoworlds"][i][:3, 3] for i in range(len(data["camtoworlds"]))])
O = full["means"].median(0).values.cpu().numpy()

# --- orbit extent: azimuth spread of cameras around the object, in the orbit plane ---
c0 = cen - cen.mean(0)
_, _, V = np.linalg.svd(c0, full_matrices=False)
axis = V[2]                                          # orbit-plane normal
ref = c0[0] - c0[0].dot(axis) * axis; ref /= np.linalg.norm(ref) + 1e-9
perp = np.cross(axis, ref)
proj = c0 - np.outer(c0.dot(axis), axis)
az = np.degrees(np.arctan2(proj.dot(perp), proj.dot(ref)))
az_sorted = np.sort(az)
gaps = np.diff(np.concatenate([az_sorted, [az_sorted[0] + 360]]))
biggest_gap = gaps.max(); gap_center = az_sorted[gaps.argmax()] + gaps.max() / 2
print(f"cameras: {len(cen)}  azimuth span {az.max()-az.min():.0f} deg  "
      f"biggest uncovered arc {biggest_gap:.0f} deg centered @ {gap_center:.0f} deg")
print(f"-> {'FULL orbit' if biggest_gap < 40 else 'PARTIAL arc — the back is in the ' + f'{biggest_gap:.0f}-deg gap (likely under-observed/unseen)'}")

# --- occlusion-aware per-Gaussian coverage (how many cams see each as front surface) ---
means = full["means"]; M = means.shape[0]
Xh = torch.cat([means, torch.ones(M, 1, device=dev)], 1)
cov = torch.zeros(M, device=dev); tol = 0.015 * scene
b0 = load_image(data, 0); Wc, Hc = b0["image"].shape[1] // 2, b0["image"].shape[0] // 2
for i in range(len(data["camtoworlds"])):
    bi = load_image(data, i); K = bi["K"].to(dev).clone(); c2w = bi["c2w"].to(dev); K[:2] *= 0.5
    rr, aa = render(full, c2w, K, Wc, Hc, "RGB+ED"); depth = rr[..., 3]
    xc = (Xh @ torch.linalg.inv(c2w).T)[:, :3]; z = xc[:, 2]
    uv = (xc / z.clamp(min=1e-6)[:, None]) @ K.T
    u = uv[:, 0].long().clamp(0, Wc - 1); v = uv[:, 1].long().clamp(0, Hc - 1)
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < Wc) & (uv[:, 1] >= 0) & (uv[:, 1] < Hc)
    cov += (ok & (aa[v, u] > 0.5) & ((z - depth[v, u]).abs() < tol)).float()
cn = cov.cpu().numpy()
print(f"\nper-Gaussian coverage: med {np.median(cn):.0f}  p10 {np.percentile(cn,10):.0f}  "
      f"under-observed (<=3 cams) {100*(cn<=3).mean():.1f}%  never-seen {100*(cn==0).mean():.1f}%")

# --- render: a normal test view + a novel view from the uncovered (back) azimuth ---
def look_at(eye, target, up=np.array([0, 0, 1.0])):
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r) + 1e-9
    u2 = np.cross(r, f)
    c2w = np.eye(4); c2w[:3, 0] = r; c2w[:3, 1] = -u2; c2w[:3, 2] = f; c2w[:3, 3] = eye
    return c2w


rad = np.linalg.norm(cen - cen.mean(0), axis=1).mean()
ga = math.radians(gap_center)
eye = cen.mean(0) + rad * (math.cos(ga) * ref + math.sin(ga) * perp)
eye[2] = cen[:, 2].mean()
bview = torch.from_numpy(look_at(eye, O).astype(np.float32)).to(dev)
Kt = b0["K"].to(dev); H, W = b0["image"].shape[:2]
back_rgb, back_a = render(full, bview, Kt, W, H, "RGB")
cv2.imwrite(os.path.join(OUT, "back_view.png"),
            cv2.cvtColor((back_rgb[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

# coverage heatmap at a normal test view (blue=under-observed, red=well-observed)
vi = int(list(data["test_indices"])[len(data["test_indices"]) // 2])
bb = load_image(vi if False else data, vi); Kv = bb["K"].to(dev); c2wv = bb["c2w"].to(dev)
colmap = cv2.applyColorMap((np.clip(cn / max(np.percentile(cn, 95), 1), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, 0, ::-1] / 255.0
heat = {k: v.clone() for k, v in full.items()}
heat["sh0"] = ((torch.from_numpy(colmap.copy()).float().to(dev) - 0.5) / C0).reshape(M, 1, 3)
heat["shN"] = torch.zeros_like(full["shN"])
hr, _ = render(heat, c2wv, Kv, W, H, "RGB")
cv2.imwrite(os.path.join(OUT, "coverage.png"),
            cv2.cvtColor((hr[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
print(f"\nviz: {OUT}\\back_view.png (novel view from the uncovered azimuth = the fuzzy back?)")
print(f"     {OUT}\\coverage.png (blue=under-observed gaussians, red=well-observed)")
