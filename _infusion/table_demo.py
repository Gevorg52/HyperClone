"""Direct demo of generating a MISSING surface region: take a real view showing the
table, mask a large chunk ('pretend the bottom half of the table wasn't captured'),
and FLUX-generate the continuation. Answers 'can we plausibly hallucinate the missing
table/ground?' on the real texture. (2D capability demo; the 3D seed+integrate is the
validated recreate pipeline.)"""
import os
import numpy as np
import cv2
from core.training.gsplat_trainer import load_colmap_data, load_image
from core.enhancement.inpaint_backend import inpaint

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\data"
OUT = r"F:\HYPERclone\apex-sfm\_infusion\table"
os.makedirs(OUT, exist_ok=True)

data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True)
# a view looking across the table
idx = int(list(data["test_indices"])[8])
photo = (load_image(data, idx)["image"].numpy() * 255).astype(np.uint8)
H, W = photo.shape[:2]
# 'missing bottom half of the table' = mask the lower 45% of the frame
mask = np.zeros((H, W), np.uint8)
mask[int(0.55 * H):, :] = 255

rgb_p = os.path.join(OUT, "demo_rgb.png"); mask_p = os.path.join(OUT, "demo_mask.png")
out_p = os.path.join(OUT, "demo_flux.png")
cv2.imwrite(rgb_p, cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)); cv2.imwrite(mask_p, mask)
print(f"view {idx}, masked lower 45% (simulated missing table); FLUX generating...")
inpaint("flux-fill", rgb_p, mask_p, out_p,
        prompt="seamless continuation of the table surface and paved ground, same "
               "wood/material, color and lighting, photorealistic, outdoor, no new objects")

flux = cv2.imread(out_p)
masked_vis = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR).copy(); masked_vis[mask > 0] = (0, 0, 200)
cols = [cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), masked_vis, flux]
labels = ["real photo", "missing (masked)", "FLUX-generated"]
hh = min(c.shape[0] for c in cols)
panel = np.ascontiguousarray(np.concatenate([c[:hh] for c in cols], 1))
for i, t in enumerate(labels):
    cv2.putText(panel, t, (i * W + 12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
cv2.imwrite(os.path.join(OUT, "table_demo.png"), cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2)))
print("viz:", os.path.join(OUT, "table_demo.png"), "(real | missing | FLUX-generated)")
