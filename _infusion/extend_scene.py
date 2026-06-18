"""Extend the captured scene OUTWARD in 3D (fill the genuinely-missing ground/table
periphery). Wider-FOV outpainting: from captured camera positions, render with a wider
field of view so the real scene sits in the center and the missing surroundings form a
large border; mask the empty GROUND band (lower, anchored to the visible edge), FLUX-
outpaint the surface, InFusion-extend the depth (plane anchored by the visible ground),
seed Gaussians. Repeat over a few azimuths. Plausible (generated where no camera saw),
appropriate because the region is genuinely ABSENT (not present-but-smooth).

Usage: python extend_scene.py [n_views] [fov_zoom]
"""
import math
import os
import sys

import numpy as np
import torch
import cv2
from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats
from core.enhancement.autocomplete import AutoCompleteConfig
from core.enhancement.recreate import complete_view, _render

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\output_4k_gated\ckpts\ckpt_29999_rank0.pt"
OUT = r"F:\HYPERclone\apex-sfm\_infusion\extend"
SH, dev, C0 = 3, "cuda", 0.28209479177387814
NV = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ZOOM = float(sys.argv[2]) if len(sys.argv) > 2 else 1.6     # wider FOV = /ZOOM focal
os.makedirs(OUT, exist_ok=True)

full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True)
scene = float(data["scene_scale"])
cfg = AutoCompleteConfig()

te = list(data["test_indices"])
views = [te[i] for i in np.linspace(0, len(te) - 1, NV).astype(int)]
seeds = []
for vi in views:
    b = load_image(data, vi); c2w = b["c2w"].to(dev); H, W = b["image"].shape[:2]
    K = b["K"].to(dev).clone(); K[0, 0] /= ZOOM; K[1, 1] /= ZOOM     # wider FOV, same center
    _, a = _render(full, c2w, K, W, H, dev, "RGB+ED")
    alpha = a.cpu().numpy()
    obj = (alpha > 0.5).astype(np.uint8) * 255
    empty = (alpha < 0.3).astype(np.uint8) * 255
    # ground extension band: empty pixels in the LOWER frame, adjacent to real geometry
    band = cv2.bitwise_and(empty, cv2.dilate(obj, np.ones((90, 90), np.uint8)))
    band[:int(0.42 * H)] = 0                                          # keep ground (lower), not sky
    band = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    npx = int((band > 0).sum())
    print(f"view {vi}: ground-extension band {npx:,}px ({100*npx/(H*W):.1f}% of frame)")
    if npx < 3000:
        continue
    res = complete_view(full, c2w, K, W, H, scene, os.path.join(OUT, f"v{vi}"),
                        inpaint_backend="flux-fill",
                        prompt="seamless continuation of the table surface and paved "
                               "ground, same material, color and lighting, "
                               "photorealistic, outdoor, no new objects",
                        mask_np=band, cfg=cfg, dev=dev)
    if res is not None:
        s = res["seed"]
        # make the extension render solid (no real views will optimise it): boost opacity
        s["opacities"] = torch.full_like(s["opacities"], 1.5)         # sigmoid~0.82
        seeds.append(s)
        print(f"  +{res['n_points']:,} seed points")

if not seeds:
    print("no extension produced"); raise SystemExit
acc = {k: torch.cat([s[k] for s in seeds], 0) for k in seeds[0]}
ext = {k: torch.cat([full[k], acc[k]], 0) for k in full}
torch.save({"splats": {k: v.cpu() for k, v in ext.items()}}, os.path.join(OUT, "extended.pt"))
print(f"\nextended splat: +{acc['means'].shape[0]:,} points ({full['means'].shape[0]:,} -> {ext['means'].shape[0]:,})")

# before/after at a pulled-back wide view (show the periphery extended)
vi = te[len(te) // 2]; b = load_image(data, vi); c2w = b["c2w"].clone().to(dev)
c2w[:3, 3] = c2w[:3, 3] - 1.8 * c2w[:3, 2]
K = b["K"].to(dev).clone(); K[0, 0] /= 1.6; K[1, 1] /= 1.6; H, W = b["image"].shape[:2]
bef = _render(full, c2w, K, W, H, dev, "RGB")[0][..., :3].clamp(0, 1)
aft = _render(ext, c2w, K, W, H, dev, "RGB")[0][..., :3].clamp(0, 1)
panel = np.ascontiguousarray(np.concatenate([
    (bef.cpu().numpy() * 255).astype(np.uint8), (aft.cpu().numpy() * 255).astype(np.uint8)], 1)[..., ::-1])
for i, t in enumerate(["before (hard boundary)", "after (extended)"]):
    cv2.putText(panel, t, (i * W + 12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
cv2.imwrite(os.path.join(OUT, "extend.png"), cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2)))
print("viz:", os.path.join(OUT, "extend.png"), "(before | after, wide pulled-back view)")
