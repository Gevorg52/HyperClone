"""Validate the Z-Image inpainter on our data BEFORE building the Type-3 path.
Take a real Laptop photo, mask a center region, inpaint, and check (a) the fill is
plausible + matches surroundings, (b) the unmasked pixels are preserved (paste-back).
"""
import os
import numpy as np
import cv2
from core.training.gsplat_trainer import load_colmap_data, load_image
from core.enhancement.inpaint_backend import inpaint, validate_backend

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\data"
OUT = r"F:\HYPERclone\apex-sfm\_infusion\inpaint_test"
os.makedirs(OUT, exist_ok=True)

ok, msg = validate_backend("zimage")
print("backend:", ok, msg)

data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True)
idx = int(list(data["test_indices"])[len(data["test_indices"]) // 2])
photo = (load_image(data, idx)["image"].numpy() * 255).astype(np.uint8)  # HWC RGB
H, W = photo.shape[:2]
mask = np.zeros((H, W), np.uint8)
bw, bh = int(0.28 * W), int(0.28 * H)
mask[(H - bh) // 2:(H + bh) // 2, (W - bw) // 2:(W + bw) // 2] = 255

rgb_p = os.path.join(OUT, "rgb.png"); mask_p = os.path.join(OUT, "mask.png")
out_p = os.path.join(OUT, "inpainted.png")
cv2.imwrite(rgb_p, cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))
cv2.imwrite(mask_p, mask)
print(f"view {idx}, {W}x{H}, hole {bw}x{bh}; inpainting...")

inpaint("zimage", rgb_p, mask_p, out_p,
        prompt="laptop and table surface, same material and lighting, photorealistic, seamless")

inp = cv2.imread(out_p)
# show: original | masked | inpainted
masked = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR).copy(); masked[mask > 0] = (0, 0, 255)
# preservation check: unmasked pixel diff
base = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR).astype(np.float32)
diff_unmasked = np.abs(inp.astype(np.float32) - base)[mask == 0].mean()
print(f"unmasked-region mean |diff| (should be ~0 = preserved): {diff_unmasked:.2f}")
panel = np.concatenate([cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), masked, inp], 1)
panel = cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2))
cv2.imwrite(os.path.join(OUT, "compare.png"), panel)
print("viz:", os.path.join(OUT, "compare.png"), "(original | masked | inpainted)")
