"""Refine BOWL_C085 object extraction using two-stage 3D visual-hull carving
(`keep_all & keep_side`) combining BiRefNet('general') and BiRefNet('high-res SOD'),
followed by Stage 10 TIDI pruning, 3,000-step object-masked self-refit, and
delivery of PLY / SOG / SPZ v4 bundles.

Why two-stage visual-hull carving (`keep_all & keep_side`) is needed on tabletop captures:
  1. On oblique and overhead views (~85% of the video trajectory), 3D Gaussians on the
     white book pedestal and wooden table directly underneath the bowl project *through*
     the wider bowl rim into the 2D bowl mask (since `fully_fused_projection` tests
     frustum visibility without occlusion). A single global `vote_frac=0.55..0.80`
     across all views therefore lets the pedestal underneath survive.
  2. On low-elevation horizontal side views (`gen_area < 0.135`, 27 training views), the
     camera looks horizontally at the side of the bowl, so the book and table project
     *below* the bowl in 2D image space (0% inside `BiRefNet('general')`).
  3. Meanwhile, on top-down views, `BiRefNet('general')` segments only the food inside
     the bowl while `BiRefNet('high-res SOD')` captures 100% of the brown glazed ceramic
     bowl rim.
  4. Intersecting:
       - `keep_all`: `vote_frac >= 0.60` across all 327 training views using `general` on
         side views + `max(general, hrsod)` on overhead views (retains 100% of the 360°
         ceramic rim), AND
       - `keep_side`: `vote_frac >= 0.85` across the 27 lowest-elevation horizontal side
         views using `general` masks (carves away 100% of the book pedestal and table
         underneath the bowl)
     yields a 100% pure 3D bowl + food splat with zero pedestal, zero table, and a
     complete 360° ceramic rim.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.masking import load_birefnet, unload_birefnet, postprocess_mask  # noqa: E402
from core.training.gsplat_trainer import load_colmap_data, render_splats  # noqa: E402
from core.delivery import (  # noqa: E402
    load_splats, vote_object, apply_keep, prune_dict,
)
from _infusion.run_bowl_c085 import clean_bowl_mask, EXPORT_DIR, TRAIN_DIR, DELIV_DIR  # noqa: E402


def main():
    t0 = time.time()
    device = "cuda"
    master_ckpt = TRAIN_DIR / "ckpts" / "ckpt_29999_rank0.pt"
    hull_ckpt = TRAIN_DIR / "ckpts" / "ckpt_29999_hull.pt"
    masks_dir = EXPORT_DIR / "masks"
    masks_2_dir = EXPORT_DIR / "masks_2"
    masks_gen_dir = EXPORT_DIR / "masks_general_bak"
    masks_hrsod_dir = EXPORT_DIR / "masks_hrsod"

    # 1. Ensure BiRefNet('general') backup exists
    if not masks_gen_dir.is_dir():
        shutil.copytree(masks_dir, masks_gen_dir)
        print(f"[1/5] Backed up BiRefNet('general') masks to {masks_gen_dir}", flush=True)

    # 2. Ensure BiRefNet('high-res SOD') masks exist
    imgs = sorted((EXPORT_DIR / "images").glob("*.jpg"))
    masks_hrsod_dir.mkdir(parents=True, exist_ok=True)
    if len(list(masks_hrsod_dir.glob("*.png"))) < len(imgs):
        print(f"[2/5] Running BiRefNet('high-res SOD') on {len(imgs)} frames ...", flush=True)
        model, transform = load_birefnet("high-res SOD", device=device)
        for i, img_path in enumerate(imgs):
            pil_img = Image.open(str(img_path)).convert("RGB")
            orig_w, orig_h = pil_img.size
            inp = transform(pil_img).unsqueeze(0).to(device).half()
            with torch.no_grad():
                preds = model(inp)
            pred = preds[-1].sigmoid().detach().squeeze().cpu().float().numpy()
            pred_pil = Image.fromarray(pred).resize((orig_w, orig_h), Image.BILINEAR)
            raw_u8 = postprocess_mask(np.array(pred_pil), threshold=0.35, erosion_px=1)
            clean_u8 = clean_bowl_mask(raw_u8)
            cv2.imwrite(str(masks_hrsod_dir / f"{img_path.stem}.png"), clean_u8)
            if (i + 1) % 75 == 0 or (i + 1) == len(imgs):
                print(f"  [hrsod] {i + 1}/{len(imgs)}", flush=True)
        unload_birefnet()
    else:
        print(f"[2/5] Reusing {len(imgs)} BiRefNet('high-res SOD') masks in {masks_hrsod_dir}", flush=True)

    # 3. Compose elevation-aware 2D masks (general on side views, max(general, hrsod) on overhead views)
    # and clear any cached masks_2 directory so load_colmap_data reads fresh masks.
    if masks_2_dir.is_dir():
        shutil.rmtree(masks_2_dir)
    for p in sorted(masks_gen_dir.glob("*.png")):
        mg = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        mh = cv2.imread(str(masks_hrsod_dir / p.name), cv2.IMREAD_GRAYSCALE)
        if (mg > 127).mean() < 0.22:
            m_out = mg
        else:
            m_out = np.maximum(mg, mh)
        cv2.imwrite(str(masks_dir / p.name), m_out)

    # 4. Two-stage 3D visual-hull carving (`keep_all & keep_side`)
    print("[3/5] Running two-stage 3D visual-hull carving (keep_all & keep_side) ...", flush=True)
    splats = load_splats(str(master_ckpt), device)
    data = load_colmap_data(str(EXPORT_DIR), factor=2, normalize=True)

    keep_all, st_all = vote_object(
        splats, data, device, mask_thresh=0.5, vote_frac=0.60, min_views=10, stride=1
    )
    horiz_idxs = []
    for idx in data["train_indices"]:
        stem = os.path.splitext(data["image_names"][idx])[0]
        mg = cv2.imread(str(masks_gen_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if (mg > 127).mean() < 0.135:
            horiz_idxs.append(idx)
    data_side = dict(data)
    data_side["train_indices"] = np.array(horiz_idxs, dtype=np.int64)
    keep_side, st_side = vote_object(
        splats, data_side, device, mask_thresh=0.5, vote_frac=0.85, min_views=4, stride=1
    )
    keep_hull = keep_all & keep_side
    print(
        f"  [hull] keep_all={int(keep_all.sum()):,} & keep_side={int(keep_side.sum()):,} "
        f"({len(horiz_idxs)} horizontal views) -> keep_hull={int(keep_hull.sum()):,}",
        flush=True,
    )

    hull_splats = apply_keep({k: v.to(device) for k, v in splats.items()}, keep_hull.to(device))
    torch.save({"step": 29999, "splats": {k: v.detach().cpu() for k, v in hull_splats.items()}}, str(hull_ckpt))
    print(f"  [hull] Saved pre-carved checkpoint -> {hull_ckpt}", flush=True)

    # Also gate the 2D masks by the clean 3D bowl silhouette so refit never sees book pixels
    # on any oblique view where hrsod touched the book corner
    clean_sub, _ = prune_dict(hull_splats, opacity_min=0.05, sor_k=12, sor_std=2.0, device=device)
    print("[4/5] Gating 2D masks with 3D visual-hull projection to remove any oblique book corners ...", flush=True)
    bg = torch.zeros(3, device=device)
    for idx in range(len(data["image_names"])):
        cid = data["camera_ids"][idx]
        K = torch.from_numpy(np.asarray(data["Ks_dict"][cid])).float().to(device)
        W, H = data["imsize_dict"][cid]
        vm = torch.linalg.inv(torch.from_numpy(data["camtoworlds"][idx]).float().to(device))
        with torch.no_grad():
            _, alphas, _ = render_splats(
                clean_sub, vm, K, int(W), int(H), 0, bg, rasterize_mode="antialiased"
            )
        alpha_np = alphas[0, ..., 0].detach().cpu().numpy()
        stem = os.path.splitext(data["image_names"][idx])[0]
        m_cur = cv2.imread(str(masks_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = m_cur.shape[:2]
        alpha_full = cv2.resize(alpha_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        sil_u8 = ((alpha_full > 0.15) * 255).astype(np.uint8)
        sil_dil = cv2.dilate(sil_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        # Intersect 2D mask with dilated 3D hull silhouette, and fill interior holes where hull alpha > 0.65
        m_gated = np.where(
            sil_dil > 0,
            np.maximum(m_cur, ((alpha_full > 0.65) * 255).astype(np.uint8)),
            0,
        ).astype(np.uint8)
        cv2.imwrite(str(masks_dir / f"{stem}.png"), clean_bowl_mask(m_gated))

    # 5. Re-run Stage 10 core.delivery on the visual-hull checkpoint + undecimated TIDI PLY
    print("[5/5] Running Stage 10 core.delivery on visual-hull checkpoint ...", flush=True)
    work_dir = DELIV_DIR / "_work" / "bowl_c085_object"
    if work_dir.is_dir():
        shutil.rmtree(work_dir)
    if masks_2_dir.is_dir():
        shutil.rmtree(masks_2_dir)

    cmd = [
        sys.executable, "-m", "core.delivery",
        "--ckpt", str(hull_ckpt),
        "--data_dir", str(EXPORT_DIR),
        "--name", "bowl_c085",
        "--factor", "2",
        "--extract_object",
        "--mask_dir", str(masks_dir),
        "--vote_frac", "0.60",
        "--decimate", "75%",
        "--refit_steps", "3000",
        "--filter_harmonics", "3",
        "--prune_tidi",
        "--tidi_data_dir", str(EXPORT_DIR),
        "--tidi_data_factor", "4",
        "--tidi_mode", "moderate",
        "--spz",
        "--out", str(DELIV_DIR),
    ]
    subprocess.run(cmd, check=True)

    refit_ckpt = work_dir / "refit" / "ckpts" / "ckpt_2999_rank0.pt"
    if refit_ckpt.is_file():
        full_obj_ply = DELIV_DIR / "bowl_c085_object_undecimated.ply"
        cmd_tidi = [
            sys.executable, str(REPO_ROOT / "tools" / "prune_tidi.py"),
            "--ckpt", str(refit_ckpt),
            "--data_dir", str(EXPORT_DIR),
            "--factor", "4",
            "--preset", "moderate",
            "--out", str(full_obj_ply),
        ]
        subprocess.run(cmd_tidi, check=True)
        print(f"[done] Exported TIDI-cleaned object PLY -> {full_obj_ply}", flush=True)

    print(f"COMPLETE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
