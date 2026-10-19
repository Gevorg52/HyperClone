"""End-to-end object-only 3DGS pipeline for C:\\Users\\gevor\\Downloads\\A001_03201343_C085.mov.

Implements the full October 2026 Clean-Splat object recipe (DEVELOPER_NOTES.md §5.2):
  1. Window-relative Tenengrad sharpest-frame extraction from 4K HEVC MOV (~373 frames @ 3840x2160).
  2. Stages 1-7 SfM: ALIKED (8192 kpts @ 1600px) + LightGlue (NetVLAD retrieval top-50)
     + COLMAP incremental mapper + 200-iter BA + gravity-aligned 3DGS export.
  3. Stage 8 Masking: BiRefNet ('general') + largest-connected-component filter (keeps ONLY the
     central food bowl, stripping any detached background plant/book islands in 2D).
  4. Stage 9 Master Training (WP-S2 full-scene anchor + STEP 0 opacity_reg=0 + WP-Q2 erank + gated LPIPS):
     30,000 steps, factor 2 (1920x1080), coarse-to-fine (factor 4 -> 2 @ 5k), cap 2.5M, pose_opt.
  5. Stage 10 Object-Only Extraction, TIDI-GS Pruning & Self-Refit (WP-D1 + WP-C3 + WP-D2):
     - Multi-view BiRefNet mask voting (`vote_object`) + 3D spatial cluster filter
     - 3,000-step object-masked self-refit (`--rgb_object_masked` + `--alpha_loss_lambda 0.1`)
     - TIDI-GS cross-view evidence & isolation pruner (`tools/prune_tidi.py` on object views)
     - Exports:
         * `_dataset/BOWL_C085/deliverables/bowl_c085_object_master_refit.ply` (full-density refit object PLY)
         * `deliverables/bowl_c085_object.ply` (clean delivery PLY)
         * `deliverables/bowl_c085_object.sog` (SuperSplat web bundle)
         * `deliverables/bowl_c085_object.spz` (Niantic SPZ v4 8/8 bits)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gradio as gr  # noqa: F401,E402
from core.pipeline_state import PipelineState  # noqa: E402
from core.utils import create_project_dir, get_image_list, get_reconstruction_stats  # noqa: E402
from gui.stages.extraction import run_extraction  # noqa: E402
from gui.stages.matching import run_matching  # noqa: E402
from gui.stages.reconstruction import run_reconstruction  # noqa: E402
from gui.stages.export import run_export, run_refinement  # noqa: E402
from core.masking import load_birefnet, unload_birefnet, postprocess_mask  # noqa: E402
from PIL import Image  # noqa: E402

NOOP = lambda *a, **k: None  # noqa: E731

VIDEO_PATH = Path(r"C:\Users\gevor\Downloads\A001_03201343_C085.mov")
DATASET_DIR = REPO_ROOT / "_dataset" / "BOWL_C085"
RAW_CANDIDATES_DIR = DATASET_DIR / "_candidates_3fps"
IMAGES_DIR = DATASET_DIR / "images"
EXPORT_DIR = DATASET_DIR / "export"
TRAIN_DIR = DATASET_DIR / "training" / "master"
DELIV_DIR = REPO_ROOT / "deliverables"


def banner(msg: str):
    print(f"\n{'=' * 76}\n[{time.strftime('%H:%M:%S')}] {msg}\n{'=' * 76}", flush=True)


def tenengrad_score(img_bgr: np.ndarray) -> float:
    """Compute Tenengrad sharpness on a 960px-wide center-weighted grayscale thumbnail."""
    h, w = img_bgr.shape[:2]
    scale = 960.0 / max(w, 1)
    small = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Center 60% ROI so background motion/blur doesn't dominate object sharpness
    sh, sw = gray.shape
    y0, y1 = int(sh * 0.15), int(sh * 0.85)
    x0, x1 = int(sw * 0.15), int(sw * 0.85)
    roi = gray[y0:y1, x0:x1]
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def stage0_extract_sharpest_frames(window_size: int = 3, candidate_fps: float = 3.0) -> list[Path]:
    """Decode at `candidate_fps` (3 fps) and keep the sharpest frame in each `window_size`
    window (~1 sharpest frame per second -> ~373 frames at full 3840x2160)."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(IMAGES_DIR.glob("frame_*.jpg"))
    if len(existing) >= 300:
        print(f"[stage 0] Reusing {len(existing)} already-extracted sharpest frames in {IMAGES_DIR}", flush=True)
        return existing

    banner(f"STAGE 0 — Extracting candidate frames at {candidate_fps} fps + window-relative sharpest selection")
    RAW_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hwaccel", "cuda",
        "-i", str(VIDEO_PATH),
        "-vf", f"fps={candidate_fps}",
        "-qscale:v", "2",
        str(RAW_CANDIDATES_DIR / "cand_%05d.jpg"),
    ]
    subprocess.run(cmd, check=True)
    cands = sorted(RAW_CANDIDATES_DIR.glob("cand_*.jpg"))
    print(f"[stage 0] Extracted {len(cands)} candidate frames; scoring Tenengrad sharpness...", flush=True)

    kept_paths: list[Path] = []
    scores_log = []
    out_idx = 1
    for i in range(0, len(cands), window_size):
        chunk = cands[i:i + window_size]
        best_p = None
        best_s = -1.0
        for p in chunk:
            im = cv2.imread(str(p))
            if im is None:
                continue
            s = tenengrad_score(im)
            if s > best_s:
                best_s = s
                best_p = p
        if best_p is not None:
            dst = IMAGES_DIR / f"frame_{out_idx:04d}.jpg"
            shutil.move(str(best_p), str(dst))
            kept_paths.append(dst)
            scores_log.append({"frame": dst.name, "src": best_p.name, "tenengrad": round(best_s, 2)})
            out_idx += 1

    shutil.rmtree(RAW_CANDIDATES_DIR, ignore_errors=True)
    with open(DATASET_DIR / "ingest_sharpness.json", "w") as f:
        json.dump({
            "video": str(VIDEO_PATH),
            "candidate_fps": candidate_fps,
            "window_size": window_size,
            "kept_frames": len(kept_paths),
            "tenengrad_p10": float(np.percentile([x["tenengrad"] for x in scores_log], 10)),
            "tenengrad_median": float(np.median([x["tenengrad"] for x in scores_log])),
            "frames": scores_log,
        }, f, indent=2)
    print(f"[stage 0] Kept {len(kept_paths)} sharpest 4K frames in {IMAGES_DIR}", flush=True)
    return kept_paths


def stage1_to_7_sfm():
    sparse0 = EXPORT_DIR / "sparse" / "0"
    if (sparse0 / "images.bin").is_file() and (EXPORT_DIR / "images").is_dir():
        stats = get_reconstruction_stats(str(sparse0))
        print(f"[stages 1-7] Reusing existing COLMAP export at {EXPORT_DIR}: {stats}", flush=True)
        return

    banner("STAGES 1-7 — Headless SfM (ALIKED 8192 + LightGlue retrieval top-50 + COLMAP + BA + Export)")
    t0 = time.time()
    state = PipelineState()
    state.image_dir = str(IMAGES_DIR)
    state.project_dir = str(create_project_dir(str(DATASET_DIR), "bowl_c085"))
    all_imgs = get_image_list(str(IMAGES_DIR))
    state.image_list = [str(p) for p in all_imgs]
    state.excluded_images = []
    state.num_images = len(all_imgs)
    state.save()
    print(f"project_dir = {state.project_dir} ({state.num_images} images)", flush=True)

    banner("STAGE 2 — ALIKED extraction (8192 kpts, resize 1600)")
    run_extraction("aliked", 8192, 3, False, 1600, False, state, NOOP)

    banner("STAGE 3 — LightGlue matching (NetVLAD retrieval top-50)")
    run_matching("retrieval", True, 0.1, 50, state, NOOP)

    banner("STAGE 4-5 — Incremental SfM (OPENCV, single camera)")
    run_reconstruction("incremental", "OPENCV", True, 15, 40, 100, True, True, True, state, NOOP)
    reg = state.num_registered or 0
    print(f"Registered {reg}/{state.num_images} ({100.0 * reg / max(state.num_images, 1):.1f}%), "
          f"3D pts = {state.num_points3d}, mean reproj = {state.mean_reprojection_error:.3f} px", flush=True)

    banner("STAGE 6 — Final Bundle Adjustment (200 iters)")
    run_refinement(200, True, True, True, state, NOOP)

    banner("STAGE 7 — Export to 3DGS layout (gravity alignment + centering ON)")
    run_export(str(EXPORT_DIR), "binary", True, "y", True, True, False, "bw_masks", state, NOOP)
    stats = get_reconstruction_stats(str(EXPORT_DIR / "sparse" / "0"))
    print(f"[stages 1-7] Complete in {(time.time() - t0) / 60:.1f} min — stats: {stats}", flush=True)


def clean_bowl_mask(mask_u8: np.ndarray) -> np.ndarray:
    """Keep ONLY the primary foreground connected component (the bowl) and fill internal holes."""
    import torch
    binary = (mask_u8 >= 128).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return mask_u8
    h, w = mask_u8.shape[:2]
    cx, cy = w * 0.5, h * 0.5
    # Pick component with largest area weighted by proximity to frame center
    best_label = 1
    best_score = -1.0
    for lbl in range(1, num_labels):
        area = float(stats[lbl, cv2.CC_STAT_AREA])
        if area < 0.005 * (h * w):
            continue
        dx = (centroids[lbl][0] - cx) / w
        dy = (centroids[lbl][1] - cy) / h
        dist_w = 1.0 / (0.25 + dx * dx + dy * dy)
        score = area * dist_w
        if score > best_score:
            best_score = score
            best_label = lbl
    comp = (labels == best_label).astype(np.uint8) * 255
    # Fill internal holes inside the bowl silhouette
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, k)
    # Preserve soft edge inside the kept component
    out = np.where(comp > 0, np.maximum(mask_u8, 200), 0).astype(np.uint8)
    return out


def stage8_generate_object_masks():
    import torch
    mask_dir = EXPORT_DIR / "masks"
    img_dir = EXPORT_DIR / "images"
    imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    mask_dir.mkdir(parents=True, exist_ok=True)
    existing = list(mask_dir.glob("*.png"))
    if len(existing) == len(imgs) and len(imgs) > 0:
        print(f"[stage 8] Reusing {len(existing)} object masks in {mask_dir}", flush=True)
        return

    banner(f"STAGE 8 — BiRefNet ('general') + largest-component bowl mask on {len(imgs)} frames")
    t0 = time.time()
    model, transform = load_birefnet("general", device="cuda")
    coverages = []
    for i, img_path in enumerate(imgs):
        pil_img = Image.open(str(img_path)).convert("RGB")
        orig_w, orig_h = pil_img.size
        inp = transform(pil_img).unsqueeze(0).to("cuda").half()
        with torch.no_grad():
            preds = model(inp)
        pred = preds[-1].sigmoid().squeeze().cpu().float().numpy()
        pred_pil = Image.fromarray(pred).resize((orig_w, orig_h), Image.BILINEAR)
        raw_u8 = postprocess_mask(np.array(pred_pil), threshold=0.35, erosion_px=1)
        clean_u8 = clean_bowl_mask(raw_u8)
        out_path = mask_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), clean_u8)
        coverages.append(float((clean_u8 > 127).mean() * 100.0))
        if (i + 1) % 50 == 0 or (i + 1) == len(imgs):
            print(f"  [masks] {i + 1}/{len(imgs)} (mean coverage {np.mean(coverages):.1f}%)", flush=True)
    unload_birefnet()
    print(f"[stage 8] Generated {len(imgs)} masks in {time.time() - t0:.1f}s "
          f"(coverage p10={np.percentile(coverages, 10):.1f}%, median={np.median(coverages):.1f}%)", flush=True)


def stage9_train_master() -> Path:
    from core.training.gsplat_backend import _downsample_images
    for f in (2, 4):
        dst_img = EXPORT_DIR / f"images_{f}"
        if not dst_img.is_dir() or len(list(dst_img.iterdir())) == 0:
            print(f"[stage 9] Creating downsampled images_{f} ...", flush=True)
            _downsample_images(str(EXPORT_DIR / "images"), str(dst_img), f)

    ckpt_path = TRAIN_DIR / "ckpts" / "ckpt_29999_rank0.pt"
    if ckpt_path.is_file():
        print(f"[stage 9] Reusing trained 30k master checkpoint: {ckpt_path}", flush=True)
        return ckpt_path

    banner("STAGE 9 — Training 30k Master Splat (MCMC 2.5M, opacity_reg=0, erank=0.01, gated-LPIPS, pose_opt)")
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    # Note: We move export/masks temporarily to export/masks_object during full-scene master training
    # so load_colmap_data does NOT treat object masks as exclusion masks during full-scene training (WP-S2 rule),
    # then restore export/masks for Stage 10 object extraction & object-masked refit!
    masks_dir = EXPORT_DIR / "masks"
    masks_hold = EXPORT_DIR / "masks_object_hold"
    if masks_dir.is_dir():
        if masks_hold.is_dir():
            shutil.rmtree(masks_hold)
        masks_dir.rename(masks_hold)

    try:
        cmd = [
            sys.executable, str(REPO_ROOT / "core" / "training" / "gsplat_trainer.py"),
            "--data_dir", str(EXPORT_DIR),
            "--result_dir", str(TRAIN_DIR),
            "--data_factor", "2",
            "--coarse_factor", "4",
            "--coarse_until", "5000",
            "--max_steps", "30000",
            "--strategy", "mcmc",
            "--cap_max", "2500000",
            "--opacity_reg", "0.0",
            "--scale_reg", "0.01",
            "--erank_lambda", "0.01",
            "--erank_start", "7000",
            "--lpips_lambda", "0.2",
            "--lpips_gated",
            "--lpips_start", "5000",
            "--pose_opt",
            "--antialiased",
            "--seed", "0",
        ]
        print("[stage 9] Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    finally:
        if masks_hold.is_dir() and not masks_dir.is_dir():
            masks_hold.rename(masks_dir)

    assert ckpt_path.is_file(), f"Expected checkpoint not found: {ckpt_path}"
    return ckpt_path


def stage10_extract_and_deliver_object(master_ckpt: Path):
    banner("STAGE 10 — Extracting ONLY the Bowl Object + 3k Self-Refit + TIDI Pruning + SPZ/PLY/SOG Delivery")
    DELIV_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "core.delivery",
        "--ckpt", str(master_ckpt),
        "--data_dir", str(EXPORT_DIR),
        "--name", "bowl_c085",
        "--factor", "2",
        "--extract_object",
        "--mask_dir", str(EXPORT_DIR / "masks"),
        "--vote_frac", "0.55",
        "--refit_steps", "3000",
        "--filter_harmonics", "3",
        "--prune_tidi",
        "--tidi_data_dir", str(EXPORT_DIR),
        "--tidi_data_factor", "4",
        "--tidi_mode", "moderate",
        "--spz",
        "--out", str(DELIV_DIR),
    ]
    print("[stage 10] Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # Also run TIDI prune directly on the 3,000-step refit full-resolution checkpoint so the user
    # has both the decimated delivery PLY/SOG/SPZ AND the undecimated refit object PLY!
    refit_ckpt = DELIV_DIR / "_work" / "bowl_c085_object" / "refit" / "ckpts" / "ckpt_2999_rank0.pt"
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
        print(f"[stage 10] Exported undecimated TIDI-cleaned object PLY -> {full_obj_ply}", flush=True)


def main():
    t_all = time.time()
    stage0_extract_sharpest_frames(window_size=3, candidate_fps=3.0)
    stage1_to_7_sfm()
    stage8_generate_object_masks()
    master_ckpt = stage9_train_master()
    stage10_extract_and_deliver_object(master_ckpt)
    banner(f"BOWL_C085 OBJECT PIPELINE COMPLETE in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
