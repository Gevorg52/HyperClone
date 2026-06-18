"""Demo: generate the GENUINELY-MISSING periphery (table/ground beyond the capture)
— the right case for generation (empty, not featureless-present). Pull a camera back
to reveal the capture boundary, mask the empty band just outside the real geometry
(so InFusion's depth is anchored by the visible edge), FLUX-outpaint the surface, seed,
and show before/after. This is plausible completion of absent regions, not invention
on present ones."""
import os
import numpy as np
import torch
import cv2
from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats
from core.enhancement.autocomplete import AutoCompleteConfig
from core.enhancement.recreate import complete_view, _render

DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\data"
CK = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\output_4k_gated\ckpts\ckpt_29999_rank0.pt"
OUT = r"F:\HYPERclone\apex-sfm\_infusion\table"
SH, dev = 3, "cuda"
os.makedirs(OUT, exist_ok=True)

full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True)
scene = float(data["scene_scale"])
te = list(data["test_indices"]); idx = te[15]
b = load_image(data, idx); K = b["K"].to(dev); H, W = b["image"].shape[:2]
# pull the camera back to reveal the missing periphery
c2w = b["c2w"].clone().to(dev); c2w[:3, 3] = c2w[:3, 3] - 1.6 * c2w[:3, 2]

rr, a = _render(full, c2w, K, W, H, dev, "RGB+ED")
alpha = a.cpu().numpy()
obj = (alpha > 0.5).astype(np.uint8) * 255
empty = (alpha < 0.3).astype(np.uint8) * 255
# outpaint band = empty pixels within ~70px of real geometry (anchored extension)
band = cv2.bitwise_and(empty, cv2.dilate(obj, np.ones((70, 70), np.uint8)))
band = cv2.morphologyEx(band, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
print(f"view {idx} pulled-back; outpaint band {int((band>0).sum()):,}px "
      f"({100*(band>0).sum()/(H*W):.1f}% of frame)")

cfg = AutoCompleteConfig()
res = complete_view(full, c2w, K, W, H, scene, os.path.join(OUT, "complete"),
                    inpaint_backend="flux-fill",
                    prompt="seamless continuation of the table surface and the paved "
                           "ground around it, same material, color and lighting, "
                           "photorealistic, outdoor, no new objects",
                    mask_np=band, cfg=cfg, dev=dev)
if res is None:
    print("no completion produced"); raise SystemExit

comb = {k: torch.cat([full[k], res["seed"][k]], 0) for k in full}
before = _render(full, c2w, K, W, H, dev, "RGB")[0][..., :3].clamp(0, 1)
after = _render(comb, c2w, K, W, H, dev, "RGB")[0][..., :3].clamp(0, 1)
ov = (before.cpu().numpy() * 255).astype(np.uint8).copy(); ov[band > 0] = (255, 0, 0)
panel = np.concatenate([
    cv2.addWeighted((before.cpu().numpy() * 255).astype(np.uint8), 0.6, ov, 0.4, 0),
    (before.cpu().numpy() * 255).astype(np.uint8),
    (after.cpu().numpy() * 255).astype(np.uint8)], 1)[..., ::-1]
for i, t in enumerate(["missing (red)", "before", "after (FLUX-completed)"]):
    cv2.putText(panel, t, (i * W + 10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imwrite(os.path.join(OUT, "table_complete.png"),
            cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2)))
print(f"+{res['n_points']:,} seed points; viz:", os.path.join(OUT, "table_complete.png"))
