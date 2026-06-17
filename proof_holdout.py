"""Proof experiment: does a generative prior recover an UNDER-OBSERVED / unseen
region faithfully? Hold out a contiguous wedge of the trajectory (APEX_HOLDOUT),
train a baseline on the rest, then resume with gated-Difix-enhanced renders at the
held-out poses, and compare BOTH against the held-out GT (the wedge = test set).

Validates the direction before building the full confidence-gated fusion. Honest
read: Difix cleans/extrapolates but cannot INVENT truly-unseen content — a positive
result supports Phase-1 (under-observed recovery); a flat result motivates a real
3D-structured generator (Phase-3) for genuine gaps.

Usage: python proof_holdout.py [start frac]   (default 0.40 0.15 -> 15% wedge)
"""
import json
import os
import sys
from pathlib import Path

HOLD_START = float(sys.argv[1]) if len(sys.argv) > 1 else 0.40
HOLD_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
os.environ["APEX_HOLDOUT"] = f"{HOLD_START},{HOLD_FRAC}"  # inherited by trainer subprocs

import cv2
import numpy as np
import torch

from core.training import TrainingConfig
from core.training.gsplat_backend import GsplatBackend
from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats
from core.enhancement import EnhancementConfig, get_backend as get_enh

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
OUT = Path(r"F:\HYPERclone\apex-sfm\_dataset\GPU\proof")
FACTOR, SH, device = 2, 3, "cuda"
backend = GsplatBackend()


def cfg(steps, extra_more=None):
    e = {"strategy": "mcmc", "cap_max": 3_000_000, "antialiased": True,
         "pose_opt": True, "coarse_factor": 4, "coarse_until": steps // 2,
         "lpips_lambda": 0.2, "lpips_gated": True, "lpips_start": max(2000, steps // 10)}
    if extra_more:
        e.update(extra_more)
    return TrainingConfig(backend="gsplat", max_steps=steps, data_factor=FACTOR,
                          test_every=8, save_steps=[steps], extra=e)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== PROOF: holdout wedge start={HOLD_START} frac={HOLD_FRAC} ===", flush=True)

    # 1. Baseline on non-wedge views; its [Eval] is on the held-out wedge
    rB = backend.train(data_dir=DATA, output_dir=str(OUT / "baseline"), config=cfg(15000),
                       progress_cb=lambda s, t, l, p: s % 3000 == 0 and
                       print(f"  B {s}/{t} psnr {p:.2f}", flush=True))
    print(f"[BASELINE] wedge PSNR {rB.psnr:.2f} SSIM {rB.ssim:.4f}", flush=True)
    ckptB = sorted((OUT / "baseline" / "output" / "ckpts").glob("ckpt_*_rank0.pt"))[-1]

    # 2. Render baseline at the held-out wedge poses, gated-Difix enhance them
    data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)
    wedge = list(data["test_indices"])
    ck = torch.load(str(ckptB), weights_only=True, map_location=device)
    splats = {k: v.to(device) for k, v in ck["splats"].items()}
    rdir = OUT / "wedge_renders"; rdir.mkdir(exist_ok=True)
    entries = []
    with torch.no_grad():
        for j, idx in enumerate(wedge):
            b = load_image(data, idx); K = b["K"].to(device); c2w = b["c2w"].to(device)
            H, W = b["image"].shape[:2]
            r, _, _ = render_splats(splats, torch.linalg.inv(c2w), K, W, H, SH,
                                    torch.zeros(3, device=device), "antialiased")
            img = (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            p = rdir / f"wedge_{j:04d}.png"
            cv2.imwrite(str(p), img[..., ::-1])
            entries.append({"image": str(p), "K": K.cpu().tolist(), "c2w": c2w.cpu().tolist()})
    del splats; torch.cuda.empty_cache()
    print(f"rendered {len(entries)} wedge views from baseline", flush=True)

    enh = get_enh("difix")
    enhanced = enh.enhance_images([e["image"] for e in entries], str(OUT / "wedge_difix"),
                                  EnhancementConfig(backend="difix", num_views=len(entries),
                                                    extra={"gated": True}))
    for e, ep in zip(entries, sorted(enhanced)):
        e["image"] = ep
    evj = OUT / "extra_views.json"; evj.write_text(json.dumps(entries), encoding="utf-8")

    # 3. Resume baseline with the gated-Difix wedge views; [Eval] again on the wedge GT
    rG = backend.train(data_dir=DATA, output_dir=str(OUT / "completed"),
                       config=cfg(5000, {"init_ckpt": str(ckptB), "extra_views": str(evj),
                                         "pose_opt": False, "coarse_factor": 0,
                                         "refine_stop_iter": 1, "lpips_start": 0}),
                       progress_cb=lambda s, t, l, p: s % 2000 == 0 and
                       print(f"  G {s}/{t} psnr {p:.2f}", flush=True))
    print(f"[COMPLETED] wedge PSNR {rG.psnr:.2f} SSIM {rG.ssim:.4f}", flush=True)

    print("\n=== PROOF RESULT (held-out wedge, vs real GT) ===")
    print(f"baseline  PSNR {rB.psnr:.2f}  SSIM {rB.ssim:.4f}")
    print(f"difix-fill PSNR {rG.psnr:.2f}  SSIM {rG.ssim:.4f}  "
          f"(dPSNR {rG.psnr - rB.psnr:+.2f}, dSSIM {rG.ssim - rB.ssim:+.4f})")
    print(f"ckpts: baseline={ckptB}")
    print(f"       completed={sorted((OUT/'completed'/'output'/'ckpts').glob('ckpt_*_rank0.pt'))[-1]}")


if __name__ == "__main__":
    main()
