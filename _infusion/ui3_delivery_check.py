"""UI-3 delivery gate: does the reviewed-mask object SOG contain the shoe WITHOUT the stand?

Renders 4 spread views of:
  (A) the REVIEWED extract  (deliverables_reviewed/sony_shoe_object.ply, carved with the
      SAM shoe-only masks_reviewed), and
  (B) the BASELINE extract  (deliverables/sony_shoe_detail_object.ply, carved with the
      original BiRefNet masks that INCLUDE the stand),
then reports, per view, the mean rendered alpha OUTSIDE the reviewed shoe-only mask
(background bleed) — ~0 means a clean cut with no stand; a higher value means stand/rig
Gaussians survive outside the shoe. Writes a side-by-side strip + a metrics JSON.

Re-runnable:  python _infusion/ui3_delivery_check.py
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.delivery import load_splats, sh_degree_of, mask_in_render_frame  # noqa: E402
from core.training.gsplat_trainer import load_colmap_data, render_splats, load_image  # noqa: E402
from core.qa.visibility import VisibilityConfig  # noqa: E402

DS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_dataset", "SONY_TEST")
DATA_DIR = os.path.join(DS, "training", "gsplat", "data")
REVIEWED_PLY = os.path.join(DS, "deliverables_reviewed", "sony_shoe_object.ply")
BASELINE_PLY = os.path.join(DS, "deliverables", "sony_shoe_detail_object.ply")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui3_strips")
FACTOR = 2
MASK_THRESH = 0.5
DEVICE = "cuda"


def render_view(splats, sh, data, idx, vcfg):
    b = load_image(data, idx)
    gt = b["image"].to(DEVICE)
    H, W = gt.shape[:2]
    K = b["K"].to(DEVICE)
    viewmat = torch.linalg.inv(b["c2w"].to(DEVICE))
    with torch.no_grad():
        renders, alphas, _ = render_splats(
            splats, viewmat, K, W, H, sh, torch.zeros(3, device=DEVICE), vcfg.rasterize_mode)
    return renders[0].clamp(0, 1).cpu().numpy(), alphas[0, ..., 0].cpu().numpy(), gt.cpu().numpy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    vcfg = VisibilityConfig()
    # Stage masks_2 <- the accepted shoe-only reviewed masks via the (hardened) delivery
    # mask-staging path, so the bleed check always scores against the SHOE-ONLY reference
    # and is re-runnable regardless of what masks_2 holds at rest (BiRefNet by default).
    # This also exercises core.delivery.ensure_masks end-to-end on the real 588-image set.
    from core.delivery import ensure_masks  # noqa: E402
    _probe = load_colmap_data(DATA_DIR, factor=FACTOR, normalize=True)
    ensure_masks(DATA_DIR, FACTOR, os.path.join(DS, "masks_reviewed"), False,
                 _probe["image_paths"], _probe["image_names"], DEVICE)
    # masks_2 now holds the reviewed shoe-only masks; reload so they auto-attach. This is
    # the reference the bleed check is scored against for BOTH renders.
    data = load_colmap_data(DATA_DIR, factor=FACTOR, normalize=True)
    n = len(data["image_names"])
    idxs = [int(round(x)) for x in np.linspace(0, n - 1, 4)]

    rev = load_splats(REVIEWED_PLY, DEVICE)
    base = load_splats(BASELINE_PLY, DEVICE)
    sh_rev, sh_base = sh_degree_of(rev), sh_degree_of(base)
    result = {
        "reviewed_ply": REVIEWED_PLY, "baseline_ply": BASELINE_PLY,
        "reviewed_n_gaussians": int(rev["means"].shape[0]),
        "baseline_n_gaussians": int(base["means"].shape[0]),
        "factor": FACTOR, "views": [],
    }

    rows = []
    for idx in idxs:
        m = mask_in_render_frame(data, idx)          # reviewed shoe-only mask, render frame
        r_rev, a_rev, gt = render_view(rev, sh_rev, data, idx, vcfg)
        r_base, a_base, _ = render_view(base, sh_base, data, idx, vcfg)
        H, W = a_rev.shape
        if m is None:
            outside = np.ones((H, W), bool)
            obj = np.zeros((H, W), bool)
        else:
            if m.shape != a_rev.shape:
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            obj = m > MASK_THRESH
            outside = ~obj
        out_area = max(1, int(outside.sum()))
        bleed_rev = float((a_rev * outside).sum() / out_area)
        bleed_base = float((a_base * outside).sum() / out_area)
        result["views"].append({
            "idx": idx, "stem": data["image_names"][idx],
            "reviewed_out_of_mask_alpha": round(bleed_rev, 5),
            "baseline_out_of_mask_alpha": round(bleed_base, 5),
        })

        def strip(r, a, tag):
            bgr = np.ascontiguousarray((r[..., ::-1] * 255).astype(np.uint8))
            bleedpix = outside & (a > 0.1)             # rig/floaters outside the shoe mask
            bgr[bleedpix] = (0.35 * bgr[bleedpix] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
            sc = 480 / max(H, W)
            bgr = cv2.resize(bgr, (int(W * sc), int(H * sc)))
            cv2.putText(bgr, tag, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return bgr
        rows.append(np.hstack([
            strip(r_base, a_base, f"BASELINE(+stand) {data['image_names'][idx]}"),
            strip(r_rev, a_rev, "REVIEWED(shoe-only)")]))

    strip_img = np.vstack(rows)
    cv2.imwrite(os.path.join(OUT_DIR, "delivery_render_check.png"), strip_img)

    br = np.mean([v["reviewed_out_of_mask_alpha"] for v in result["views"]])
    bb = np.mean([v["baseline_out_of_mask_alpha"] for v in result["views"]])
    result["mean_reviewed_out_of_mask_alpha"] = round(float(br), 5)
    result["mean_baseline_out_of_mask_alpha"] = round(float(bb), 5)
    result["stand_removed"] = bool(br < 0.02 and br < bb * 0.5)
    with open(os.path.join(OUT_DIR, "..", "ui3_delivery_check.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
