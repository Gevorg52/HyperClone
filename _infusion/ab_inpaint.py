"""A/B the inpainters on the same masked Laptop photo: original | masked | Z-Image | FLUX.
Confirms whether FLUX.1-Fill's quality upgrade is worth making it the Type-3 default."""
import os
import numpy as np
import cv2
from core.training.gsplat_trainer import load_colmap_data, load_image
from core.enhancement.inpaint_backend import inpaint, validate_backend

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\data"
OUT = r"F:\HYPERclone\apex-sfm\_infusion\ab_inpaint"
PROMPT = ("seamless continuation of the existing surface, same material, texture, "
          "color and lighting, photorealistic, no new objects")
os.makedirs(OUT, exist_ok=True)

data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True)
idx = int(list(data["test_indices"])[len(data["test_indices"]) // 2])
photo = (load_image(data, idx)["image"].numpy() * 255).astype(np.uint8)
H, W = photo.shape[:2]
mask = np.zeros((H, W), np.uint8)
bw, bh = int(0.28 * W), int(0.28 * H)
mask[(H - bh) // 2:(H + bh) // 2, (W - bw) // 2:(W + bw) // 2] = 255
rgb_p = os.path.join(OUT, "rgb.png"); mask_p = os.path.join(OUT, "mask.png")
cv2.imwrite(rgb_p, cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)); cv2.imwrite(mask_p, mask)
masked = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR).copy(); masked[mask > 0] = (0, 0, 255)
base = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR).astype(np.float32)

cols = [cv2.cvtColor(photo, cv2.COLOR_RGB2BGR), masked]
labels = ["original", "masked"]
for bk in ["zimage", "flux-fill"]:
    ok, msg = validate_backend(bk)
    print(f"{bk}: {ok} {msg}")
    out_p = os.path.join(OUT, f"{bk}.png")
    try:
        inpaint(bk, rgb_p, mask_p, out_p, prompt=PROMPT)
        img = cv2.imread(out_p)
        d = np.abs(img.astype(np.float32) - base)[mask == 0].mean()
        print(f"  {bk}: unmasked |diff| {d:.2f} (preserve), wrote {out_p}")
        cols.append(img); labels.append(bk)
    except Exception as e:
        print(f"  {bk} FAILED: {type(e).__name__}: {str(e)[:200]}")

hh = min(c.shape[0] for c in cols)
panel = np.concatenate([c[:hh] for c in cols], 1)
for i, t in enumerate(labels):
    cv2.putText(panel, t, (i * W + 10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
panel = cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2))
cv2.imwrite(os.path.join(OUT, "ab.png"), panel)
print("viz:", os.path.join(OUT, "ab.png"), "|", " | ".join(labels))
