"""WP-C3 Part 1 Gate Evaluation: TIDI-GS Cross-View Evidence, Importance & 3D Isolation Pruner.

Evaluates on the 3 banked 30k-step checkpoints:
  - Laptop: `_dataset/Laptop/apex_images/training/opreg0/or0/output/ckpts/ckpt_29999_rank0.pt`
  - GPU_V4: `_dataset/GPU_V4/apex_images/training/wpf5/base/output/ckpts/ckpt_29999_rank0.pt`
  - 360:    `_dataset/360/train/wpf5_base/output/ckpts/ckpt_29999_rank0.pt`

Compares:
  (a) `unpruned` (full 3,000,000 Gaussians)
  (b) `prune_op05` (`tools/prune_ply.py --opacity_min 0.05`)
  (c) `prune_tidi_cons` (`tools/prune_tidi.py --preset conservative`)
  (d) `prune_tidi` (`tools/prune_tidi.py --preset moderate`, default)

Persists all results to `_infusion/wpc3/tidi_gate.json`.
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.qa.masked_metrics import masked_lpips, masked_psnr, masked_ssim  # noqa: E402
from core.qa.visibility import VisibilityConfig, build_visibility_mask  # noqa: E402
from core.training.gsplat_trainer import (  # noqa: E402
    get_lpips,
    load_colmap_data,
    load_image,
    render_splats,
)
from _infusion.offpath_orbit import build_views, fog_proxies, render_rgba  # noqa: E402
from tools.prune_ply import effective_rank, prune as prune_ply_fn, splat_stats  # noqa: E402
from tools.prune_tidi import TidiConfig, compute_tidi_scores, tidi_keep_mask  # noqa: E402

_spec_w5 = _ilu.spec_from_file_location("wpo5_eval", PROJECT_ROOT / "_infusion" / "wpo5_eval.py")
_w5 = _ilu.module_from_spec(_spec_w5)
_spec_w5.loader.exec_module(_w5)


SCENES = {
    "laptop": {
        "ckpt": "_dataset/Laptop/apex_images/training/opreg0/or0/output/ckpts/ckpt_29999_rank0.pt",
        "data_dir": "_dataset/Laptop/apex_images/training/gsplat_prod/data",
        "factor": 2,
        "holdout_list": None,
        "region_name": "deck",
        "region_box": (0.30, 0.72, 0.55, 0.95),
        "out_ply": "_dataset/Laptop/apex_images/training/opreg0/or0/output/ply/point_cloud_29999_pruned_tidi.ply",
    },
    "gpu_v4": {
        "ckpt": "_dataset/GPU_V4/apex_images/training/wpf5/base/output/ckpts/ckpt_29999_rank0.pt",
        "data_dir": "_dataset/GPU_V4/apex_images/training/gsplat_prod/data",
        "factor": 2,
        "holdout_list": None,
        "region_name": "shiny",
        "region_box": (0.28, 0.72, 0.30, 0.55),
        "out_ply": "_dataset/GPU_V4/apex_images/training/wpf5/base/output/ply/point_cloud_29999_pruned_tidi.ply",
    },
    "360": {
        "ckpt": "_dataset/360/train/wpf5_base/output/ckpts/ckpt_29999_rank0.pt",
        "data_dir": "_dataset/360/train/wpo5_theater_027_8k",
        "factor": 1,
        "holdout_list": "_dataset/360/train/wpo5_theater_027_8k/holdout_test.txt",
        "region_name": None,
        "region_box": None,
        "out_ply": "_dataset/360/train/wpf5_base/output/ply/point_cloud_29999_pruned_tidi.ply",
    },
}


def subset_splats(splats: dict, keep_mask: torch.Tensor) -> dict:
    km = keep_mask.to(splats["means"].device)
    return {k: v[km] for k, v in splats.items()}


def compute_3d_floater_stats(
    splats_cpu: dict,
    keep_mask_cpu: torch.Tensor,
    scores: dict,
) -> dict:
    """Compute 3D spatial floater / geometry health metrics for an arm."""
    n_base = int(splats_cpu["means"].shape[0])
    n_kept = int(keep_mask_cpu.sum())
    sub = {k: v[keep_mask_cpu] for k, v in splats_cpu.items()}
    st = splat_stats(sub)

    sor_002_thr = float(scores["sor_002_threshold"])
    sor_op05_thr = float(scores["sor_threshold"])
    knn_kept = scores["knn_dist"][keep_mask_cpu].numpy()

    # Recompute kNN on the kept point cloud itself as well
    means_k = sub["means"].numpy()
    d_k, _ = cKDTree(means_k).query(means_k, k=9, workers=-1)
    knn_self = d_k[:, 1:].mean(axis=1)
    sor_self_thr = float(knn_self.mean() + 2.0 * knn_self.std())

    scales_lin = np.exp(sub["scales"].numpy())
    maxscale = scales_lin.max(axis=1)
    giant_pct = float((maxscale > 10.0 * np.median(maxscale)).mean()) * 100.0
    opac = torch.sigmoid(sub["opacities"]).numpy()
    low_opa_001_pct = float((opac < 0.01).mean()) * 100.0

    cnt_sor002 = int((knn_self > sor_002_thr).sum())
    cnt_sor_op05 = int((knn_self > sor_op05_thr).sum())
    cnt_sor_self = int((knn_self > sor_self_thr).sum())

    return {
        "num_gaussians": n_kept,
        "kept_pct": round(100.0 * n_kept / max(n_base, 1), 3),
        "opacity_mean": round(st["opacity_mean"], 4),
        "opacity_median": round(st["opacity_median"], 4),
        "opacity_frac_below_005": round(st["opacity_frac_below"]["0.05"], 6),
        "low_opa_lt_001_pct": round(low_opa_001_pct, 3),
        "outliers_beyond_3x_bbox": int(st["outliers_beyond_3x_bbox"]),
        "outliers_beyond_10x_bbox": int(st["outliers_beyond_10x_bbox"]),
        "needle_pct_erank_lt_2": round(st["needle_frac_erank_lt_2"] * 100.0, 3),
        "giant_pct": round(giant_pct, 3),
        "erank_mean": round(st["erank_mean"], 4),
        "floating_sor_002_threshold": round(sor_002_thr, 6),
        "floating_sor_002_count": cnt_sor002,
        "floating_sor_002_pct_of_base": round(100.0 * cnt_sor002 / max(n_base, 1), 4),
        "floating_sor_002_pct_of_kept": round(100.0 * cnt_sor002 / max(n_kept, 1), 4),
        "sor_op05_threshold": round(sor_op05_thr, 6),
        "sor_op05_count": cnt_sor_op05,
        "sor_op05_pct_of_base": round(100.0 * cnt_sor_op05 / max(n_base, 1), 4),
        "sor_op05_pct_of_kept": round(100.0 * cnt_sor_op05 / max(n_kept, 1), 4),
        "sor_self_pct": round(100.0 * cnt_sor_self / max(n_kept, 1), 4),
        "knn_median": round(float(np.median(knn_self)), 6),
        "knn_p99": round(float(np.percentile(knn_self, 99)), 6),
        "knn_max": round(float(knn_self.max()), 6),
    }


def eval_heldout_orbit(
    splats_gpu: dict,
    test_batches: list[dict],
    region_box: tuple[float, float, float, float],
    lpips_vgg,
    device: torch.device,
) -> dict:
    """Evaluate held-out masked PSNR/SSIM/LPIPS + region LPIPS/PSNR on object-orbit scenes."""
    vcfg = VisibilityConfig(alpha_thresh=0.3)
    m_psnr, m_ssim, m_lpips, obs_pct = [], [], [], []
    f_lpips, r_lpips, r_psnr = [], [], []
    x0f, x1f, y0f, y1f = region_box

    with torch.no_grad():
        for b in test_batches:
            gt = b["image"].to(device)
            K = b["K"].to(device)
            c2w = b["c2w"].to(device)
            H, W = gt.shape[:2]
            renders, alphas, _ = render_splats(
                splats_gpu,
                torch.linalg.inv(c2w),
                K,
                W,
                H,
                3,
                torch.zeros(3, device=device),
                "antialiased",
            )
            render = renders[0].clamp(0, 1)
            alpha = alphas[0, ..., 0]

            vis = build_visibility_mask(alpha, vcfg)
            obs = torch.from_numpy(vis["observed"]).to(device)
            obs_pct.append(vis["observed_pct"])
            m_psnr.append(masked_psnr(render, gt, obs))
            m_ssim.append(masked_ssim(render, gt, obs))
            m_lpips.append(masked_lpips(render, gt, obs, device, mode="bbox"))

            # Region LPIPS / PSNR (same VGG-LPIPS as _infusion/wps1_region_eval.py)
            rb = render.permute(2, 0, 1).unsqueeze(0)
            gb = gt.permute(2, 0, 1).unsqueeze(0)
            f_lpips.append(float(lpips_vgg(rb, gb)))
            lpips_vgg.reset()
            x0, x1 = int(x0f * W), int(x1f * W)
            y0, y1 = int(y0f * H), int(y1f * H)
            rc = rb[:, :, y0:y1, x0:x1]
            gc = gb[:, :, y0:y1, x0:x1]
            r_lpips.append(float(lpips_vgg(rc, gc)))
            lpips_vgg.reset()
            mse = torch.mean((rc - gc) ** 2).item()
            r_psnr.append(10.0 * np.log10(1.0 / (mse + 1e-10)))

    return {
        "n_views": len(test_batches),
        "observed_area_pct": round(float(np.mean(obs_pct)), 3),
        "masked_psnr": round(float(np.mean(m_psnr)), 4),
        "masked_ssim": round(float(np.mean(m_ssim)), 4),
        "masked_lpips": round(float(np.mean(m_lpips)), 4),
        "full_lpips_vgg": round(float(np.mean(f_lpips)), 4),
        "region_lpips": round(float(np.mean(r_lpips)), 4),
        "region_psnr": round(float(np.mean(r_psnr)), 4),
    }


def eval_heldout_360(
    splats_gpu: dict,
    test_batches: list[dict],
    device: torch.device,
) -> dict:
    """Evaluate held-out 360 crops (exact protocol of `_infusion/wpo3_eval.py`)."""
    m_psnr, m_ssim, m_lpips_bbox, m_lpips_tiled = [], [], [], []
    f_psnr, f_ssim, f_lpips = [], [], []

    with torch.no_grad():
        for b in test_batches:
            gt = b["image"].to(device)
            K = b["K"].to(device)
            c2w = b["c2w"].to(device)
            H, W = gt.shape[:2]
            renders, _, _ = render_splats(
                splats_gpu,
                torch.linalg.inv(c2w),
                K,
                W,
                H,
                3,
                torch.zeros(3, device=device),
                "antialiased",
            )
            render = renders[0][..., :3].clamp(0, 1)
            full_mask = torch.ones((H, W), dtype=torch.bool, device=device)
            f_psnr.append(masked_psnr(render, gt, full_mask))
            f_ssim.append(masked_ssim(render, gt, full_mask))
            f_lpips.append(masked_lpips(render, gt, full_mask, device, mode="bbox"))

            if "mask" in b:
                mask = b["mask"].to(device) > 0.5
                m_psnr.append(masked_psnr(render, gt, mask))
                m_ssim.append(masked_ssim(render, gt, mask))
                m_lpips_bbox.append(masked_lpips(render, gt, mask, device, mode="bbox"))
                m_lpips_tiled.append(
                    masked_lpips(render, gt, mask, device, mode="tiled", tile=128, cover=0.8)
                )

    return {
        "n_views": len(test_batches),
        "masked_psnr": round(float(np.mean(m_psnr)), 4),
        "masked_ssim": round(float(np.mean(m_ssim)), 4),
        "masked_lpips_tiled": round(float(np.mean(m_lpips_tiled)), 4),
        "masked_lpips_bbox": round(float(np.mean(m_lpips_bbox)), 4),
        "full_psnr": round(float(np.mean(f_psnr)), 4),
        "full_ssim": round(float(np.mean(f_ssim)), 4),
        "full_lpips": round(float(np.mean(f_lpips)), 4),
    }


def eval_offpath_fog(
    splats_gpu: dict,
    arm_mask_gpu: torch.Tensor,
    clean_tidi_mask_gpu: torch.Tensor,
    views: list[dict],
    K: np.ndarray,
    W: int,
    H: int,
    device: torch.device,
) -> dict:
    """Evaluate off-path fog metrics across off-path virtual cameras.

    Measures:
    1. `op05_fog_layer_alpha_mean` & `op05_prune_dMAE`: alpha deposited by `op < 0.05` splats
       still present in the arm (`offpath_orbit.py` definition).
    2. `residual_floater_layer_alpha_mean` & `residual_floater_prune_dMAE`: alpha deposited by
       all floater splats (`~clean_tidi_mask`) still present inside this arm (`arm_mask & ~clean_tidi_mask`),
       and the MAE render change when those remaining floaters are removed.
       - In `unpruned`, this is the total off-path fog/floater layer (`op < 0.05` + `op >= 0.05` floaters).
       - In `prune_op05`, this is the surviving `op >= 0.05` floater layer missed by `prune_ply.py`.
       - In `prune_tidi`, this is `0.0`.
    3. `translucent_op20_layer_alpha_mean`: alpha deposited by translucent `op < 0.20` splats
       present in the arm at off-path views.
    4. Full-render alpha proxies (`semi_alpha_frac`, `fog_alpha_frac`, `coverage`).
    """
    arm_splats = subset_splats(splats_gpu, arm_mask_gpu)
    clean_splats = subset_splats(splats_gpu, clean_tidi_mask_gpu)

    opac_all = torch.sigmoid(splats_gpu["opacities"].reshape(-1))
    op05_fog_mask = arm_mask_gpu & (opac_all < 0.05)
    op05_dense_mask = arm_mask_gpu & (opac_all >= 0.05)
    res_floater_mask = arm_mask_gpu & (~clean_tidi_mask_gpu)
    op20_mask = arm_mask_gpu & (opac_all < 0.20)

    has_op05_fog = int(op05_fog_mask.sum()) > 0
    has_res_floater = int(res_floater_mask.sum()) > 0
    has_op20 = int(op20_mask.sum()) > 0

    op05_fog_splats = subset_splats(splats_gpu, op05_fog_mask) if has_op05_fog else None
    op05_dense_splats = subset_splats(splats_gpu, op05_dense_mask) if has_op05_fog else None
    res_floater_splats = subset_splats(splats_gpu, res_floater_mask) if has_res_floater else None
    op20_splats = subset_splats(splats_gpu, op20_mask) if has_op20 else None

    semi_list, fog_list, cov_list = [], [], []
    op05_a_list, op05_dmae_list = [], []
    res_a_list, res_px_list, res_dmae_list = [], [], []
    op20_a_list = []

    for v in views:
        c2w = v["c2w"]
        img, a = render_rgba(arm_splats, c2w, K, W, H, device, 3)
        fp = fog_proxies(a)
        semi_list.append(fp["semi_alpha_frac"])
        fog_list.append(fp["fog_alpha_frac"])
        cov_list.append(fp["coverage"])

        if has_op05_fog:
            _, a_op05 = render_rgba(op05_fog_splats, c2w, K, W, H, device, 3)
            img_d05, _ = render_rgba(op05_dense_splats, c2w, K, W, H, device, 3)
            op05_a_list.append(float(a_op05.mean()))
            op05_dmae_list.append(
                float(np.abs(img.astype(np.float32) - img_d05.astype(np.float32)).mean())
            )
        else:
            op05_a_list.append(0.0)
            op05_dmae_list.append(0.0)

        if has_res_floater:
            _, a_res = render_rgba(res_floater_splats, c2w, K, W, H, device, 3)
            img_clean, _ = render_rgba(clean_splats, c2w, K, W, H, device, 3)
            res_a_list.append(float(a_res.mean()))
            res_px_list.append(float((a_res > 0.1).mean()))
            res_dmae_list.append(
                float(np.abs(img.astype(np.float32) - img_clean.astype(np.float32)).mean())
            )
        else:
            res_a_list.append(0.0)
            res_px_list.append(0.0)
            res_dmae_list.append(0.0)

        if has_op20:
            _, a_op20 = render_rgba(op20_splats, c2w, K, W, H, device, 3)
            op20_a_list.append(float(a_op20.mean()))
        else:
            op20_a_list.append(0.0)

    return {
        "num_offpath_views": len(views),
        "semi_alpha_frac": round(float(np.mean(semi_list)), 6),
        "fog_alpha_frac": round(float(np.mean(fog_list)), 6),
        "coverage": round(float(np.mean(cov_list)), 6),
        "op05_fog_layer_alpha_mean": round(float(np.mean(op05_a_list)), 6),
        "op05_prune_delta_mae": round(float(np.mean(op05_dmae_list)), 4),
        "residual_floater_count": int(res_floater_mask.sum()),
        "residual_floater_layer_alpha_mean": round(float(np.mean(res_a_list)), 6),
        "residual_floater_layer_px_frac": round(float(np.mean(res_px_list)), 6),
        "residual_floater_prune_delta_mae": round(float(np.mean(res_dmae_list)), 4),
        "translucent_op20_layer_alpha_mean": round(float(np.mean(op20_a_list)), 6),
    }


def evaluate_scene(scene_key: str, device: torch.device, lpips_vgg) -> dict:
    import gsplat

    scfg = SCENES[scene_key]
    ckpt_path = PROJECT_ROOT / scfg["ckpt"]
    data_dir = PROJECT_ROOT / scfg["data_dir"]
    holdout = str(PROJECT_ROOT / scfg["holdout_list"]) if scfg["holdout_list"] else None

    print(f"\n==================== Evaluating {scene_key.upper()} ====================", flush=True)
    t_scene = time.time()

    raw_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    s_raw = raw_ckpt["splats"]
    N = int(s_raw["means"].shape[0])
    splats_cpu = {
        "means": s_raw["means"].float(),
        "scales": s_raw["scales"].float(),
        "quats": s_raw["quats"].float(),
        "opacities": s_raw["opacities"].float().reshape(N),
        "sh0": s_raw["sh0"].float().reshape(N, 1, 3),
        "shN": s_raw["shN"].float().reshape(N, -1, 3),
    }
    pose_embeds = raw_ckpt.get("pose_embeds")
    if pose_embeds is not None:
        pose_embeds = pose_embeds.float()

    data = load_colmap_data(
        str(data_dir),
        factor=scfg["factor"],
        test_every=8,
        normalize=True,
        holdout_list=holdout,
    )

    # 1. Compute TIDI scores once across all training views
    t_score = time.time()
    cfg_mod = TidiConfig.from_preset("moderate")
    cfg_cons = TidiConfig.from_preset("conservative")
    scores = compute_tidi_scores(
        splats_cpu, data, cfg=cfg_mod, device=device, pose_embeds=pose_embeds
    )
    score_runtime_s = round(time.time() - t_score, 3)
    print(f"[{scene_key}] TIDI score computation across {scores['num_train_views']} train views: {score_runtime_s:.2f}s", flush=True)

    # 2. Build keep masks for all arms
    keep_unpruned = torch.ones(N, dtype=torch.bool)
    _, reasons_op05 = prune_ply_fn(splats_cpu, opacity_min=0.05, bbox_scale=0.0)
    keep_op05 = scores["opacities"] >= 0.05
    keep_tidi_cons, reasons_cons = tidi_keep_mask(scores, cfg=cfg_cons)
    keep_tidi_mod, reasons_mod = tidi_keep_mask(scores, cfg=cfg_mod)

    # Export pruned TIDI PLY
    out_ply = PROJECT_ROOT / scfg["out_ply"]
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    kept_mod_splats = {k: v[keep_tidi_mod] for k, v in splats_cpu.items()}
    gsplat.export_splats(
        means=kept_mod_splats["means"],
        scales=kept_mod_splats["scales"],
        quats=kept_mod_splats["quats"],
        opacities=kept_mod_splats["opacities"],
        sh0=kept_mod_splats["sh0"],
        shN=kept_mod_splats["shN"],
        format="ply",
        save_to=str(out_ply),
    )

    # Preload held-out test batches
    test_indices = sorted(int(i) for i in data["test_indices"])
    test_batches = [load_image(data, idx) for idx in test_indices]

    # Build off-path virtual camera views
    if scene_key != "360":
        offpath_views, _ = build_views(data, 4)
        cam0 = data["camera_ids"][0]
        W0, H0 = data["imsize_dict"][cam0]
        W_off, H_off = int(round(W0 * 0.5)), int(round(H0 * 0.5))
        K_off = data["Ks_dict"][cam0].astype(np.float64).copy()
        K_off[:2] *= (W_off / W0)
    else:
        frames = _w5.pano_frames(data, 6)
        scale_info = _w5.estimate_metres_per_unit(data, frames, 2.0, 3.0, 1.2)
        mpu = scale_info["metres_per_unit"]
        ns_off = argparse.Namespace(
            offpath_points=4,
            lateral_m=[-1.0, -0.5, 0.5, 1.0],
            vertical_m=[0.5, -0.3],
            unvisited_views=3,
        )
        offpath_views, _ = _w5.build_offpath_views(data, frames, mpu, ns_off)
        cid0 = data["camera_ids"][frames["indices"][0]]
        W0, H0 = data["imsize_dict"][cid0]
        W_off, H_off = int(round(W0 * 0.5)), int(round(H0 * 0.5))
        K_off = data["Ks_dict"][cid0].astype(np.float64).copy()
        K_off[:2] *= (W_off / W0)

    splats_gpu = {k: v.to(device) for k, v in splats_cpu.items()}
    clean_tidi_mask_gpu = keep_tidi_mod.to(device)

    arms_spec = [
        ("unpruned", keep_unpruned, {"kept": N, "dropped_total": 0, "kept_frac": 1.0}),
        ("prune_op05", keep_op05, reasons_op05),
        ("prune_tidi_cons", keep_tidi_cons, reasons_cons),
        ("prune_tidi", keep_tidi_mod, reasons_mod),
    ]

    arms_out = {}
    for arm_name, km_cpu, reasons in arms_spec:
        t_arm = time.time()
        km_gpu = km_cpu.to(device)
        arm_splats_gpu = subset_splats(splats_gpu, km_gpu)

        if scene_key != "360":
            heldout = eval_heldout_orbit(
                arm_splats_gpu, test_batches, scfg["region_box"], lpips_vgg, device
            )
        else:
            heldout = eval_heldout_360(arm_splats_gpu, test_batches, device)

        offpath = eval_offpath_fog(
            splats_gpu=splats_gpu,
            arm_mask_gpu=km_gpu,
            clean_tidi_mask_gpu=clean_tidi_mask_gpu,
            views=offpath_views,
            K=K_off,
            W=W_off,
            H=H_off,
            device=device,
        )
        floaters_3d = compute_3d_floater_stats(splats_cpu, km_cpu, scores)

        arms_out[arm_name] = {
            "prune_reasons": reasons,
            "heldout": heldout,
            "offpath_fog": offpath,
            "floaters_3d": floaters_3d,
            "eval_seconds": round(time.time() - t_arm, 2),
        }
        lp_key = "masked_lpips" if scene_key != "360" else "masked_lpips_tiled"
        print(
            f"  [{scene_key}:{arm_name:15s}] kept={floaters_3d['num_gaussians']:8,d} ({floaters_3d['kept_pct']:5.1f}%) | "
            f"mPSNR={heldout['masked_psnr']:.4f} | mSSIM={heldout['masked_ssim']:.4f} | "
            f"{lp_key}={heldout[lp_key]:.4f} | "
            f"res_fogA={offpath['residual_floater_layer_alpha_mean']:.4f} | "
            f"op20_fogA={offpath['translucent_op20_layer_alpha_mean']:.4f} | "
            f"sor002={floaters_3d['floating_sor_002_count']:,} ({floaters_3d['floating_sor_002_pct_of_base']:.2f}%)",
            flush=True,
        )

    del splats_gpu
    torch.cuda.empty_cache()

    # Compute gate deltas: prune_tidi vs prune_op05
    h_op05 = arms_out["prune_op05"]["heldout"]
    h_tidi = arms_out["prune_tidi"]["heldout"]
    lp_key = "masked_lpips" if scene_key != "360" else "masked_lpips_tiled"
    delta_psnr = round(h_tidi["masked_psnr"] - h_op05["masked_psnr"], 5)
    delta_ssim = round(h_tidi["masked_ssim"] - h_op05["masked_ssim"], 5)
    delta_lpips = round(h_tidi[lp_key] - h_op05[lp_key], 5)
    gate_pass = bool((delta_lpips <= 0.0005) and (delta_psnr >= -0.05))

    return {
        "ckpt": scfg["ckpt"],
        "data_dir": scfg["data_dir"],
        "factor": scfg["factor"],
        "out_ply": scfg["out_ply"],
        "out_ply_mb": round(out_ply.stat().st_size / 1e6, 2),
        "score_runtime_s": score_runtime_s,
        "total_scene_eval_s": round(time.time() - t_scene, 2),
        "tidi_config_moderate": asdict(cfg_mod),
        "tidi_config_conservative": asdict(cfg_cons),
        "arms": arms_out,
        "gate_vs_prune_op05": {
            "delta_masked_psnr_db": delta_psnr,
            "delta_masked_ssim": delta_ssim,
            "delta_masked_lpips": delta_lpips,
            "extra_gaussians_removed_beyond_op05": int(
                arms_out["prune_op05"]["floaters_3d"]["num_gaussians"]
                - arms_out["prune_tidi"]["floaters_3d"]["num_gaussians"]
            ),
            "pct_reduction_vs_op05": round(
                100.0
                * (
                    arms_out["prune_op05"]["floaters_3d"]["num_gaussians"]
                    - arms_out["prune_tidi"]["floaters_3d"]["num_gaussians"]
                )
                / max(arms_out["prune_op05"]["floaters_3d"]["num_gaussians"], 1),
                2,
            ),
            "gate_passed": gate_pass,
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WP-C3 Part 1 TIDI Pruner Gate Evaluation")
    ap.add_argument("--scenes", nargs="+", default=["laptop", "gpu_v4", "360"], choices=["laptop", "gpu_v4", "360"])
    ap.add_argument("--out", default="_infusion/wpc3/tidi_gate.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    lpips_vgg = get_lpips(device)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    if out_path.is_file():
        try:
            prior = json.loads(out_path.read_text(encoding="utf8"))
            results = prior.get("scenes", {})
        except Exception:
            results = {}

    t0 = time.time()
    for sc in args.scenes:
        results[sc] = evaluate_scene(sc, device, lpips_vgg)
        payload = {
            "wp": "WP-C3 Part 1 (TIDI-GS Cross-View Evidence, Importance & 3D Isolation Pruner)",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "argv": [sys.executable, __file__] + list(sys.argv[1:]),
            "all_gates_passed": all(
                v.get("gate_vs_prune_op05", {}).get("gate_passed", False) for v in results.values()
            ),
            "total_runtime_s": round(time.time() - t0, 2),
            "scenes": results,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf8")
        print(f"[wpc3_tidi_eval] updated {out_path}", flush=True)

    print(f"\n[wpc3_tidi_eval] ALL DONE in {time.time() - t0:.1f}s -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
