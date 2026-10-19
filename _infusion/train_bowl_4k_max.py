"""End-to-end maximized 4K training + object delivery pipeline for BOWL_C085.

Combines all October 2026 performance and quality upgrades:
1. GPU VRAM uint8 frame + mask preload (3.09 GB in coarse 1080p phase, 12.35 GB in fine 4K phase)
2. Fused multi-tensor CUDA Adam + separable 1D Gaussian SSIM + stateless bfloat16 LPIPS + zero CPU syncs
3. Coarse-to-fine training: 1920x1080 (steps 1..15,000) -> native 4K 3840x2160 (steps 15,001..30,000)
4. Cosine-decayed MCMC noise_lr (5e5 -> 2.5e4 over steps 8,333..25,000)
5. Zero-mean canonical Bilateral Grid (--bilateral_grid) to absorb handheld exposure/WB flicker
6. Effective-rank anti-needle regularizer (--erank_lambda 0.01) + --opacity_reg 0.0 + gated 768x768 LPIPS
7. Upgraded core.delivery with two-stage horizon-stratified visual-hull carving (--vote_horizon_frac 0.85),
   pre+post-refit TIDI pruning (--prune_tidi), 4K self_refit, and SH4 + SPZ v4 export.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA_DIR = REPO / "_dataset" / "BOWL_C085" / "export"
TRAIN_ROOT = REPO / "_dataset" / "BOWL_C085" / "training"
MASTER_4K_DIR = TRAIN_ROOT / "master_4k"
DELIV_DIR = REPO / "deliverables"
WORK_DIR = TRAIN_ROOT / "delivery_4k_work"


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n{'=' * 72}\n[{desc}]\n  {' '.join(str(x) for x in cmd)}\n{'=' * 72}", flush=True)
    t0 = time.time()
    r = subprocess.run([str(x) for x in cmd], cwd=str(REPO))
    if r.returncode != 0:
        raise SystemExit(f"[{desc}] FAILED with exit code {r.returncode}")
    print(f"[{desc}] finished in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    t_all = time.time()
    py = sys.executable
    trainer = REPO / "core" / "training" / "gsplat_trainer.py"

    # ------------------------------------------------------------------
    # Stage 1: Full 30,000-step C2F (1080p -> 4K) Master Training
    # ------------------------------------------------------------------
    master_ckpt = MASTER_4K_DIR / "ckpts" / "ckpt_29999_rank0.pt"
    if not master_ckpt.is_file():
        run_cmd([
            py, str(trainer),
            "--data_dir", str(DATA_DIR),
            "--result_dir", str(MASTER_4K_DIR),
            "--coarse_factor", "2",
            "--data_factor", "1",
            "--coarse_until", "15000",
            "--max_steps", "30000",
            "--save_steps", "15000", "30000",
            "--strategy", "mcmc",
            "--cap_max", "2500000",
            "--mcmc_noise_lr", "500000",
            "--mcmc_noise_decay",
            "--sh_degree", "3",
            "--sh_degree_interval", "1000",
            "--opacity_reg", "0.0",
            "--scale_reg", "0.01",
            "--erank_lambda", "0.01",
            "--erank_start", "7000",
            "--lpips_lambda", "0.2",
            "--lpips_gated",
            "--lpips_bf16",
            "--lpips_patch", "768",
            "--lpips_start", "3000",
            "--bilateral_grid",
            "--alpha_loss_lambda", "0.1",
            "--seed", "42",
        ], "Stage 1: 30k Master Training (1080p -> 4K C2F + MCMC Noise Decay + Zero-Mean BilaGrid)")
    else:
        print(f"[Stage 1] Using existing 4K master checkpoint: {master_ckpt}", flush=True)

    # ------------------------------------------------------------------
    # Stage 2: Object Delivery via upgraded core.delivery (4K self_refit)
    # ------------------------------------------------------------------
    run_cmd([
        py, "-m", "core.delivery",
        "--ckpt", str(master_ckpt),
        "--data_dir", str(DATA_DIR),
        "--name", "bowl_c085",
        "--scene", "BOWL_C085_4K",
        "--factor", "1",
        "--out", str(DELIV_DIR),
        "--work", str(WORK_DIR),
        "--extract_object",
        "--vote_frac", "0.60",
        "--vote_horizon_frac", "0.85",
        "--prune_tidi",
        "--tidi_mode", "moderate",
        "--tidi_data_factor", "4",
        "--decimate", "60%",
        "--refit",
        "--refit_steps", "3000",
        "--alpha_loss_lambda", "0.5",
        "--spz",
        "--spz_sh1_bits", "8",
        "--spz_sh_rest_bits", "8",
    ], "Stage 2: Object Delivery (Horizon Visual-Hull + Pre/Post TIDI + 4K Self-Refit + SPZ v4)")

    # ------------------------------------------------------------------
    # Stage 2b: Apples-to-apples QA at factor=2 (1920x1080) as well as factor=1 (4K)
    # ------------------------------------------------------------------
    from core.delivery import (
        load_splats, qa_object_region, spatial_floater_stats,
        robust_scene_scale, object_center,
    )
    from core.training.gsplat_trainer import load_colmap_data

    delivered_ply = DELIV_DIR / "bowl_c085_object.ply"
    deliv_splats = load_splats(str(delivered_ply), "cuda")
    data_f2 = load_colmap_data(str(DATA_DIR), factor=2, normalize=True)
    qa_f2 = qa_object_region(
        deliv_splats, data_f2, data_f2["test_indices"],
        sh_degree=3, device="cuda", mask_thresh=0.5,
        out_strip=str(REPO / "qa" / "wpd1_bowl_c085_object" / "strip_delivered_4k_at_f2.png"),
    )
    floaters = spatial_floater_stats(deliv_splats, object_center(data_f2), robust_scene_scale(data_f2))
    print(f"\n[QA @ factor=2 (1080p apples-to-apples)] "
          f"PSNR={qa_f2['obj_masked_psnr']:.2f} dB | "
          f"SSIM={qa_f2['obj_masked_ssim']:.4f} | "
          f"LPIPS={qa_f2['obj_masked_lpips']:.4f} | "
          f"bleed={qa_f2['mean_bg_bleed_alpha']:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Stage 3: SH Degree 4 (WP-SH4) 4K Specular Warm-Start + TIDI + SPZ v4
    # ------------------------------------------------------------------
    sh3_post_ply = WORK_DIR / "refit_tidi.ply"
    if not sh3_post_ply.is_file():
        sh3_post_ply = delivered_ply
    sh3_post_splats = load_splats(str(sh3_post_ply), "cuda")
    sh3_post_ckpt = WORK_DIR / "sh3_post_tidi_ckpt.pt"
    torch.save({"step": 3000, "splats": {k: v.detach().cpu() for k, v in sh3_post_splats.items()}},
               str(sh3_post_ckpt))

    sh4_dir = WORK_DIR / "sh4_refit_4k"
    n_sh3 = int(sh3_post_splats["means"].shape[0])
    run_cmd([
        py, str(trainer),
        "--data_dir", str(DATA_DIR),
        "--result_dir", str(sh4_dir),
        "--init_ckpt", str(sh3_post_ckpt),
        "--data_factor", "1",
        "--max_steps", "2000",
        "--save_steps", "2000",
        "--refine_stop_iter", "1",
        "--sh_degree", "4",
        "--strategy", "mcmc",
        "--cap_max", str(max(n_sh3, 1000)),
        "--opacity_reg", "0.0",
        "--scale_reg", "0.01",
        "--erank_lambda", "0.01",
        "--erank_start", "1",
        "--lpips_lambda", "0.2",
        "--lpips_gated",
        "--lpips_bf16",
        "--lpips_patch", "768",
        "--lpips_start", "1",
        "--rgb_object_masked",
        "--alpha_loss_lambda", "0.5",
        "--seed", "42",
    ], "Stage 3: SH Degree 4 (WP-SH4) 4K Specular Warm-Start (2,000 steps)")

    from tools.prune_tidi import TidiConfig, prune_ckpt_tidi
    from core.delivery import export_ply
    from core.spz_io import write_spz

    sh4_ckpt = sh4_dir / "ckpts" / "ckpt_1999_rank0.pt"
    kept_sh4, sh4_reasons, _ = prune_ckpt_tidi(
        source_path=str(sh4_ckpt),
        data_dir=str(DATA_DIR),
        factor=4,
        cfg=TidiConfig.from_preset("moderate"),
        device="cuda",
    )
    sh4_ply = DELIV_DIR / "bowl_c085_object_sh4.ply"
    sh4_spz = DELIV_DIR / "bowl_c085_object_sh4.spz"
    export_ply({k: v.to("cuda") for k, v in kept_sh4.items()}, str(sh4_ply))
    spz4_rec = write_spz({k: v.to("cuda") for k, v in kept_sh4.items()}, str(sh4_spz),
                         4, sh1_bits=8, sh_rest_bits=8)

    qa_sh4_f2 = qa_object_region(
        {k: v.to("cuda") for k, v in kept_sh4.items()},
        data_f2, data_f2["test_indices"],
        sh_degree=4, device="cuda", mask_thresh=0.5,
    )
    summary = {
        "total_wall_seconds": round(time.time() - t_all, 1),
        "qa_sh3_at_factor2": qa_f2,
        "qa_sh4_at_factor2": qa_sh4_f2,
        "spatial_floaters_sh3": floaters,
        "sh4_tidi_reasons": sh4_reasons,
        "sh4_spz": spz4_rec,
    }
    (DELIV_DIR / "bowl_c085_4k_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{'=' * 72}\n[ALL DONE in {summary['total_wall_seconds']}s]\n"
          f"  SH3 (f=2 QA): PSNR={qa_f2['obj_masked_psnr']:.2f} dB, "
          f"SSIM={qa_f2['obj_masked_ssim']:.4f}, LPIPS={qa_f2['obj_masked_lpips']:.4f}, "
          f"bleed={qa_f2['mean_bg_bleed_alpha']:.4f}\n"
          f"  SH4 (f=2 QA): PSNR={qa_sh4_f2['obj_masked_psnr']:.2f} dB, "
          f"SSIM={qa_sh4_f2['obj_masked_ssim']:.4f}, LPIPS={qa_sh4_f2['obj_masked_lpips']:.4f}, "
          f"bleed={qa_sh4_f2['mean_bg_bleed_alpha']:.4f}\n{'=' * 72}", flush=True)


if __name__ == "__main__":
    main()
