"""Fill the genuinely-missing BELOW-THE-TABLE region (legs/underside/floor never
captured) from a head-on zoomed-out view. FLUX generates the structure, InFusion gives
depth (anchored by the visible table edge), seed Gaussians. Honest limit: InFusion does
smooth depth completion, so it reconstructs planar-ish extension well; complex vertical
legs are approximate and will look best from ~this head-on angle (where FLUX painted
them) and weaker from oblique angles. Plausible, not faithful (nothing was captured)."""
import math
import os
import numpy as np
import torch
import cv2
from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats
from core.enhancement.autocomplete import AutoCompleteConfig
from core.enhancement.recreate import complete_view, _render

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\output_4k_gated\ckpts\ckpt_29999_rank0.pt"
OUT = r"F:\HYPERclone\apex-sfm\_infusion\headon"
SH, dev = 3, "cuda"
os.makedirs(OUT, exist_ok=True)

full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True)
scene = float(data["scene_scale"])
cen = np.stack([data["camtoworlds"][i][:3, 3] for i in range(len(data["camtoworlds"]))])
O = full["means"].median(0).values.cpu().numpy()
c0 = cen - cen.mean(0); _, _, V = np.linalg.svd(c0, full_matrices=False); axis = V[2]
if (cen - O).dot(axis).mean() < 0:
    axis = -axis
elev = (cen - O).dot(axis) / (np.linalg.norm(cen - O, axis=1) + 1e-9)
low = int(elev.argmin())
b = load_image(data, low); K0 = b["K"].to(dev); H, W = b["image"].shape[:2]
c2w0 = b["c2w"].cpu().numpy(); fwd = O - c2w0[:3, 3]; fwd /= np.linalg.norm(fwd)
rad = np.linalg.norm(c2w0[:3, 3] - O)


def look_at(eye, target, up):
    f = target - eye; f /= np.linalg.norm(f); r = np.cross(f, up); r /= np.linalg.norm(r) + 1e-9; u2 = np.cross(r, f)
    c = np.eye(4); c[:3, 0] = r; c[:3, 1] = -u2; c[:3, 2] = f; c[:3, 3] = eye; return c


eye = O - fwd * rad * 1.9
c2w = torch.from_numpy(look_at(eye, O, axis).astype(np.float32)).to(dev)
K = K0.clone(); K[0, 0] /= 1.7; K[1, 1] /= 1.7

rr, a = _render(full, c2w, K, W, H, dev, "RGB+ED")
alpha = a.cpu().numpy()
# Below-the-table isn't cleanly empty (fuzzy floor/ground bleeds through). FORCE a mask
# over the lower-center region (where legs/underside belong) that is NOT the solid
# table/laptop, so we can see what FLUX+InFusion produce for this hardest case.
solid = (alpha > 0.85)
band = np.zeros((H, W), np.uint8)
band[int(0.52 * H):int(0.92 * H), int(0.10 * W):int(0.90 * W)] = 255   # below-table strip
band[solid] = 0                                                         # keep the real table/laptop
band = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
npx = int((band > 0).sum())
print(f"head-on view: below-table band {npx:,}px ({100*npx/(H*W):.1f}% of frame)")
cv2.imwrite(os.path.join(OUT, "band.png"), band)

cfg = AutoCompleteConfig()
res = complete_view(full, c2w, K, W, H, scene, os.path.join(OUT, "fill"),
                    inpaint_backend="flux-fill",
                    prompt="the legs and underside of the table, and the paved ground "
                           "beneath the table, same materials, color and lighting, "
                           "photorealistic, outdoor, no new objects",
                    mask_np=band, cfg=cfg, dev=dev)
if res is None:
    print("no fill produced"); raise SystemExit
s = res["seed"]; s["opacities"] = torch.full_like(s["opacities"], 1.5)
ext = {k: torch.cat([full[k], s[k]], 0) for k in full}
torch.save({"splats": {k: v.cpu() for k, v in ext.items()}}, os.path.join(OUT, "extended.pt"))

bef = _render(full, c2w, K, W, H, dev, "RGB")[0][..., :3].clamp(0, 1)
aft = _render(ext, c2w, K, W, H, dev, "RGB")[0][..., :3].clamp(0, 1)
mvis = (bef.cpu().numpy() * 255).astype(np.uint8).copy(); mvis[band > 0] = (0, 0, 255)
panel = np.ascontiguousarray(np.concatenate([
    cv2.addWeighted((bef.cpu().numpy() * 255).astype(np.uint8), 0.6, mvis, 0.4, 0),
    (aft.cpu().numpy() * 255).astype(np.uint8)], 1)[..., ::-1])
for i, t in enumerate(["before (red=missing below table)", "after (FLUX-generated)"]):
    cv2.putText(panel, t, (i * W + 12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imwrite(os.path.join(OUT, "headon_fill.png"), cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2)))
print(f"+{res['n_points']:,} seed points; viz:", os.path.join(OUT, "headon_fill.png"))
