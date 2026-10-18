"""WP-SH4 Phase B Evaluator: SH Degree 4 vs Degree 3 Quality A/B.

Evaluates Experiment B1 (Warm-Start Fine-Phase Refinement on GPU_V4 and Laptop)
and Experiment B2 (From-Scratch Arc-Withheld A/B on GPU_V4) across:
  1. Overall masked & full-frame PSNR / SSIM / LPIPS on the held-out orbit arc
     (plus strict_unseen views never seen in the 30k base run for B1).
  2. Specular-region (top-10% view-dependent energy ||I_SH - I_SH0||_2) PSNR & LPIPS.
  3. Specular-box (512x512 crop around specular centroid) LPIPS.
  4. Train-vs-holdout PSNR gap (overfit diagnostic).
  5. Postshot degree-3 truncated twin check via truncate_ply_file().
  6. Off-path & geometry sanity (fog_layer_alpha_mean, op < 0.05 fraction).
  7. Wall time, peak VRAM, PLY size, and SPZ v4 (8/8 bits) round-trip PSNR.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.pop("APEX_HOLDOUT", None)

from core.qa.masked_metrics import masked_lpips, masked_psnr, masked_ssim
from core.sh_utils import ply_sh_degree, read_ply_header, sh_degree_of, truncate_ply_file
from core.spz_io import read_spz, spz_header, write_spz
from core.training.gsplat_trainer import get_lpips, load_colmap_data, load_image, render_splats, ssim
import qa_report

DEV = "cuda"


def parse_log_stats(log_path: Path) -> dict:
    if not log_path.is_file():
        return {"train_seconds": None, "peak_mem_mb": None, "final_it_per_s": None}
    raw = log_path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw[:200]:
        txt = raw.decode("utf-16", errors="replace")
    else:
        txt = raw.decode("utf-8", errors="replace")
    steps = re.findall(r"step=(\d+) .*?mem=(\d+)MB \| ([\d.]+) it/s", txt)
    m_time = re.search(r"Training complete in ([\d.]+)s", txt)
    return {
        "train_seconds": float(m_time.group(1)) if m_time else None,
        "peak_mem_mb": max(int(m) for _, m, _ in steps) if steps else None,
        "final_it_per_s": float(steps[-1][2]) if steps else None,
    }


def compute_opacity_and_fog_stats(sp: dict) -> dict:
    with torch.no_grad():
        op = torch.sigmoid(sp["opacities"].flatten())
        n_gs = int(op.numel())
        op_lt_005_frac = float((op < 0.05).float().mean().item())
        fog_mask = (op > 0.005) & (op < 0.10)
        fog_layer_frac = float(fog_mask.float().mean().item())
        fog_layer_alpha_mean = float((op * fog_mask.float()).mean().item())
        shN = sp["shN"]
        l4_rms = float(shN[:, 15:24, :].pow(2).mean().sqrt().item()) if shN.shape[1] >= 24 else 0.0
        l13_rms = float(shN[:, :15, :].pow(2).mean().sqrt().item()) if shN.shape[1] >= 15 else 0.0
    return {
        "num_gaussians": n_gs,
        "op_lt_005_frac": op_lt_005_frac,
        "fog_layer_frac": fog_layer_frac,
        "fog_layer_alpha_mean": fog_layer_alpha_mean,
        "sh_l1_to_l3_rms": l13_rms,
        "sh_l4_rms": l4_rms,
    }


def _crop_512_around_centroid(mask_2d: torch.Tensor, H: int, W: int, size: int = 512) -> tuple[int, int, int, int]:
    ys, xs = torch.where(mask_2d)
    if ys.numel() > 0:
        cy = int(ys.float().mean().item())
        cx = int(xs.float().mean().item())
    else:
        cy, cx = H // 2, W // 2
    h_box = min(size, H)
    w_box = min(size, W)
    y0 = max(0, min(H - h_box, cy - h_box // 2))
    x0 = max(0, min(W - w_box, cx - w_box // 2))
    return y0, y0 + h_box, x0, x0 + w_box


def evaluate_pair_on_views(
    sp3: dict,
    sp4: dict,
    data: dict,
    view_indices: list[int],
    train_sample_indices: list[int],
) -> dict:
    lpips_fn = get_lpips(DEV)
    bg = torch.zeros(3, device=DEV)

    rows_sh3, rows_sh4, rows_trunc = [], [], []
    for idx in view_indices:
        b = load_image(data, int(idx))
        gt = b["image"].to(DEV)
        K = b["K"].to(DEV)
        c2w = b["c2w"].to(DEV)
        w2c = torch.linalg.inv(c2w)
        H, W = gt.shape[:2]

        # Object foreground mask (explicit mask if present, else non-black foreground)
        if "mask" in b and b["mask"] is not None:
            fg = (b["mask"].to(DEV) > 0.5)
        else:
            fg = (gt.max(dim=-1).values > (2.0 / 255.0))
        if int(fg.sum().item()) < 100:
            fg = torch.ones((H, W), dtype=torch.bool, device=DEV)

        with torch.no_grad():
            r3_full = render_splats(sp3, w2c, K, W, H, 3, bg, "antialiased")[0][0].clamp(0.0, 1.0)
            r3_deg0 = render_splats(sp3, w2c, K, W, H, 0, bg, "antialiased")[0][0].clamp(0.0, 1.0)
            r4_full = render_splats(sp4, w2c, K, W, H, 4, bg, "antialiased")[0][0].clamp(0.0, 1.0)
            r4_deg0 = render_splats(sp4, w2c, K, W, H, 0, bg, "antialiased")[0][0].clamp(0.0, 1.0)
            # Postshot degree-3 truncated twin (rendering sp4 at sh_degree=3 uses only bands 0..3)
            r4_trunc3 = render_splats(sp4, w2c, K, W, H, 3, bg, "antialiased")[0][0].clamp(0.0, 1.0)

            # View-dependent energy ||I_SH - I_SH0||_2 per pixel inside fg
            e3 = torch.norm(r3_full - r3_deg0, dim=-1)
            e4 = torch.norm(r4_full - r4_deg0, dim=-1)
            e_sym = torch.maximum(e3, e4)
            fg_vals = e_sym[fg]
            q90 = torch.quantile(fg_vals, 0.90) if fg_vals.numel() > 10 else torch.tensor(0.0, device=DEV)
            spec_mask = fg & (e_sym >= q90)
            if int(spec_mask.sum().item()) < 32:
                spec_mask = fg

            # Also SH4-only view-dependent top-10% mask (exact formula from spec)
            fg_vals4 = e4[fg]
            q90_4 = torch.quantile(fg_vals4, 0.90) if fg_vals4.numel() > 10 else torch.tensor(0.0, device=DEV)
            spec_mask_sh4 = fg & (e4 >= q90_4)
            if int(spec_mask_sh4.sum().item()) < 32:
                spec_mask_sh4 = fg

            y0, y1, x0, x1 = _crop_512_around_centroid(spec_mask, H, W, size=512)

            def _score_one(ren: torch.Tensor) -> dict:
                mse_full = float(((ren - gt) ** 2).mean().item())
                psnr_full = 10.0 * math.log10(1.0 / max(mse_full, 1e-12))
                ssim_full = float(ssim(
                    ren.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
                ).item())
                psnr_fg = masked_psnr(ren, gt, fg)
                ssim_fg = masked_ssim(ren, gt, fg)
                lpips_fg = masked_lpips(ren, gt, fg, DEV, mode="bbox")

                psnr_spec = masked_psnr(ren, gt, spec_mask)
                psnr_spec_sh4mask = masked_psnr(ren, gt, spec_mask_sh4)
                # Specular-box LPIPS on the 512x512 shiny crop (get_lpips is constructed with normalize=True)
                ren_box = ren[y0:y1, x0:x1].permute(2, 0, 1)[None].clamp(0.0, 1.0)
                gt_box = gt[y0:y1, x0:x1].permute(2, 0, 1)[None].clamp(0.0, 1.0)
                lpips_spec_box = float(lpips_fn(ren_box, gt_box).item())
                # Composited LPIPS restricted to the top-10% specular pixels inside the box
                sm_box = spec_mask[y0:y1, x0:x1].unsqueeze(-1).float()
                ren_comp = (ren[y0:y1, x0:x1] * sm_box + gt[y0:y1, x0:x1] * (1.0 - sm_box)).permute(2, 0, 1)[None].clamp(0.0, 1.0)
                lpips_spec_top10 = float(lpips_fn(ren_comp, gt_box).item())
                return {
                    "psnr_full": psnr_full,
                    "ssim_full": ssim_full,
                    "psnr_masked": psnr_fg,
                    "ssim_masked": ssim_fg,
                    "lpips_masked": lpips_fg,
                    "specular_top10_psnr": psnr_spec,
                    "specular_top10_sh4mask_psnr": psnr_spec_sh4mask,
                    "specular_top10_lpips": lpips_spec_top10,
                    "specular_box_lpips": lpips_spec_box,
                }

            rows_sh3.append(_score_one(r3_full))
            rows_sh4.append(_score_one(r4_full))
            rows_trunc.append(_score_one(r4_trunc3))

    def _agg(rows: list[dict]) -> dict:
        keys = rows[0].keys()
        return {k: float(np.mean([r[k] for r in rows])) for k in keys}

    agg3 = _agg(rows_sh3)
    agg4 = _agg(rows_sh4)
    agg_trunc = _agg(rows_trunc)

    # Train sample PSNR for train-vs-holdout overfit gap
    train_psnr_3, train_psnr_4 = [], []
    for idx in train_sample_indices:
        b = load_image(data, int(idx))
        gt = b["image"].to(DEV)
        K = b["K"].to(DEV)
        w2c = torch.linalg.inv(b["c2w"].to(DEV))
        H, W = gt.shape[:2]
        with torch.no_grad():
            r3 = render_splats(sp3, w2c, K, W, H, 3, bg, "antialiased")[0][0].clamp(0.0, 1.0)
            r4 = render_splats(sp4, w2c, K, W, H, 4, bg, "antialiased")[0][0].clamp(0.0, 1.0)
        train_psnr_3.append(10.0 * math.log10(1.0 / max(float(((r3 - gt) ** 2).mean().item()), 1e-12)))
        train_psnr_4.append(10.0 * math.log10(1.0 / max(float(((r4 - gt) ** 2).mean().item()), 1e-12)))

    tr3 = float(np.mean(train_psnr_3))
    tr4 = float(np.mean(train_psnr_4))
    gap3 = tr3 - agg3["psnr_full"]
    gap4 = tr4 - agg4["psnr_full"]

    return {
        "num_eval_views": len(view_indices),
        "num_train_sample_views": len(train_sample_indices),
        "sh3": {**agg3, "train_psnr_full": tr3, "train_minus_holdout_gap_db": gap3},
        "sh4": {**agg4, "train_psnr_full": tr4, "train_minus_holdout_gap_db": gap4},
        "sh4_truncated_to_sh3": agg_trunc,
        "deltas_sh4_minus_sh3": {
            "psnr_full_db": agg4["psnr_full"] - agg3["psnr_full"],
            "psnr_masked_db": agg4["psnr_masked"] - agg3["psnr_masked"],
            "ssim_masked": agg4["ssim_masked"] - agg3["ssim_masked"],
            "lpips_masked": agg4["lpips_masked"] - agg3["lpips_masked"],
            "specular_top10_psnr_db": agg4["specular_top10_psnr"] - agg3["specular_top10_psnr"],
            "specular_top10_sh4mask_psnr_db": agg4["specular_top10_sh4mask_psnr"] - agg3["specular_top10_sh4mask_psnr"],
            "specular_top10_lpips": agg4["specular_top10_lpips"] - agg3["specular_top10_lpips"],
            "specular_box_lpips": agg4["specular_box_lpips"] - agg3["specular_box_lpips"],
            "overfit_gap_delta_db": gap4 - gap3,
            "trunc_sh3_minus_native_sh3_psnr_db": agg_trunc["psnr_masked"] - agg3["psnr_masked"],
            "trunc_sh3_minus_native_sh3_full_db": agg_trunc["psnr_full"] - agg3["psnr_full"],
        },
    }


def eval_spz_and_trunc_ply(
    ply4_path: Path,
    sp4_gpu: dict,
    data: dict,
    eval_indices: list[int],
    work_dir: Path,
    tag: str,
) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    s_cpu = {k: v.detach().cpu() for k, v in sp4_gpu.items()}
    spz_path = work_dir / f"{tag}_sh4_b88.spz"
    t0 = time.time()
    rec = write_spz(s_cpu, spz_path, sh1_bits=8, sh_rest_bits=8)
    t_write = time.time() - t0
    decoded_cpu = read_spz(spz_path)
    decoded_gpu = {k: v.to(DEV) for k, v in decoded_cpu.items()}

    # Render orig SH4 vs SPZ-decoded SH4 on first 15 held-out views
    sub_idx = eval_indices[:15]
    bg = torch.zeros(3, device=DEV)
    psnr_orig_vs_dec, psnr_dec_vs_gt = [], []
    for i in sub_idx:
        b = load_image(data, int(i))
        gt = b["image"].to(DEV)
        K = b["K"].to(DEV)
        w2c = torch.linalg.inv(b["c2w"].to(DEV))
        H, W = gt.shape[:2]
        with torch.no_grad():
            ro = render_splats(sp4_gpu, w2c, K, W, H, 4, bg, "antialiased")[0][0].clamp(0.0, 1.0)
            rd = render_splats(decoded_gpu, w2c, K, W, H, 4, bg, "antialiased")[0][0].clamp(0.0, 1.0)
        mse_od = float(((ro - rd) ** 2).mean().item())
        mse_dg = float(((rd - gt) ** 2).mean().item())
        psnr_orig_vs_dec.append(10.0 * math.log10(1.0 / max(mse_od, 1e-12)))
        psnr_dec_vs_gt.append(10.0 * math.log10(1.0 / max(mse_dg, 1e-12)))

    # Disk PLY truncation test via truncate_ply_file
    trunc_ply = work_dir / f"{tag}_trunc_sh3.ply"
    truncate_ply_file(ply4_path, trunc_ply, 3)
    trunc_deg = ply_sh_degree(trunc_ply)
    trunc_bytes = trunc_ply.stat().st_size
    ply4_bytes = ply4_path.stat().st_size

    # Clean up temporary large files in work_dir after measuring
    spz_bytes = spz_path.stat().st_size
    try:
        spz_path.unlink(missing_ok=True)
        trunc_ply.unlink(missing_ok=True)
    except Exception:
        pass

    return {
        "ply_sh4_bytes": ply4_bytes,
        "ply_trunc_sh3_bytes": trunc_bytes,
        "ply_trunc_sh3_degree": trunc_deg,
        "spz_v4_88_bytes": spz_bytes,
        "ply_over_spz_ratio": ply4_bytes / max(spz_bytes, 1),
        "spz_write_seconds": round(t_write, 2),
        "spz_dropped_out_of_range": rec["dropped_out_of_range"],
        "spz_88_orig_vs_decoded_psnr_mean": float(np.mean(psnr_orig_vs_dec)),
        "spz_88_orig_vs_decoded_psnr_min": float(np.min(psnr_orig_vs_dec)),
        "spz_88_decoded_vs_gt_psnr_mean": float(np.mean(psnr_dec_vs_gt)),
    }


def evaluate_experiment(
    exp_name: str,
    data_dir: str,
    holdout_list: str,
    sh3_dir: Path,
    sh4_dir: Path,
    ckpt_step: int,
    is_warm_start_from_test_every_8: bool = False,
) -> dict:
    print(f"\n=== Evaluating {exp_name} ===", flush=True)
    data = load_colmap_data(
        data_dir, factor=2, test_every=8, normalize=True, holdout_list=holdout_list
    )
    ckpt3 = sh3_dir / "ckpts" / f"ckpt_{ckpt_step}_rank0.pt"
    ckpt4 = sh4_dir / "ckpts" / f"ckpt_{ckpt_step}_rank0.pt"
    ply3 = sh3_dir / "ply" / f"point_cloud_{ckpt_step}.ply"
    ply4 = sh4_dir / "ply" / f"point_cloud_{ckpt_step}.ply"

    sp3 = qa_report.load_splats(str(ckpt3), DEV)
    sp4 = qa_report.load_splats(str(ckpt4), DEV)

    arc_indices = [int(i) for i in data["test_indices"]]
    train_all = [int(i) for i in data["train_indices"]]
    step_tr = max(1, len(train_all) // 24)
    train_sample = train_all[::step_tr][:24]

    arc_eval = evaluate_pair_on_views(sp3, sp4, data, arc_indices, train_sample)

    strict_eval = None
    if is_warm_start_from_test_every_8:
        # Views inside the held-out arc that ALSO had index % 8 == 0 during the 30k base run
        strict_unseen = [i for i in arc_indices if i % 8 == 0]
        if strict_unseen:
            strict_eval = evaluate_pair_on_views(sp3, sp4, data, strict_unseen, train_sample)

    stats3 = compute_opacity_and_fog_stats(sp3)
    stats4 = compute_opacity_and_fog_stats(sp4)
    log3 = parse_log_stats(sh3_dir.parent / f"{sh3_dir.name}.log")
    log4 = parse_log_stats(sh4_dir.parent / f"{sh4_dir.name}.log")

    spz_info = eval_spz_and_trunc_ply(
        ply4, sp4, data, arc_indices, sh4_dir / "_tmp_spz", exp_name
    )

    out = {
        "experiment": exp_name,
        "data_dir": data_dir,
        "holdout_list": holdout_list,
        "sh3_ckpt": str(ckpt3),
        "sh4_ckpt": str(ckpt4),
        "holdout_arc_eval": arc_eval,
        "strict_unseen_arc_eval": strict_eval,
        "sh3_splat_stats": {**stats3, **log3, "ply_bytes": ply3.stat().st_size if ply3.is_file() else None},
        "sh4_splat_stats": {**stats4, **log4, "ply_bytes": ply4.stat().st_size if ply4.is_file() else None},
        "spz_and_truncation": spz_info,
    }
    d = arc_eval["deltas_sh4_minus_sh3"]
    print(
        f"[{exp_name}] Arc ({len(arc_indices)} views): "
        f"dPSNR_masked={d['psnr_masked_db']:+.3f} dB, "
        f"dPSNR_spec10={d['specular_top10_psnr_db']:+.3f} dB, "
        f"dLPIPS_spec_box={d['specular_box_lpips']:+.4f}, "
        f"dGap={d['overfit_gap_delta_db']:+.3f} dB, "
        f"dTruncSH3={d['trunc_sh3_minus_native_sh3_psnr_db']:+.3f} dB",
        flush=True,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_infusion/wpsh4/phaseB_gate.json")
    args = ap.parse_args()

    base_dir = REPO / "_infusion/wpsh4/phaseB"
    specs = [
        (
            "B1_GPU_V4_warm5k",
            "_dataset/GPU_V4/apex_images/training/gsplat_prod/data",
            "_infusion/wpsh4/phaseB/gpu_holdout_arc.txt",
            base_dir / "b1_gpu_sh3",
            base_dir / "b1_gpu_sh4",
            4999,
            True,
        ),
        (
            "B1_Laptop_warm5k",
            "_dataset/Laptop/apex_images/training/gsplat_prod/data",
            "_infusion/wpsh4/phaseB/laptop_holdout_arc.txt",
            base_dir / "b1_laptop_sh3",
            base_dir / "b1_laptop_sh4",
            4999,
            True,
        ),
        (
            "B2_GPU_V4_scratch15k",
            "_dataset/GPU_V4/apex_images/training/gsplat_prod/data",
            "_infusion/wpsh4/phaseB/gpu_holdout_arc.txt",
            base_dir / "b2_gpu_sh3",
            base_dir / "b2_gpu_sh4",
            14999,
            False,
        ),
    ]

    results = {
        "script": "_infusion/wpsh4_phaseB_eval.py",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0),
        "experiments": {},
    }
    for exp_name, data_dir, holdout_list, sh3_dir, sh4_dir, step, is_warm in specs:
        c3 = sh3_dir / "ckpts" / f"ckpt_{step}_rank0.pt"
        c4 = sh4_dir / "ckpts" / f"ckpt_{step}_rank0.pt"
        if c3.is_file() and c4.is_file():
            results["experiments"][exp_name] = evaluate_experiment(
                exp_name, data_dir, holdout_list, sh3_dir, sh4_dir, step, is_warm
            )
        else:
            print(f"[skip] {exp_name}: missing {c3} or {c4}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[WP-SH4 Phase B] Wrote {out_path}")


if __name__ == "__main__":
    main()

