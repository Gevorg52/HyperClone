"""Phase-1a: coverage-GATED prior fusion vs UNGATED, on the held-out wedge.

Reuses the proof baseline (wedge held out). Renders the baseline at poses spanning
the WHOLE orbit (wedge + observed), gated-Difix-enhances them, then resumes twice:
  - ungated : every pseudo-view weight 1.0 (prior acts everywhere, incl. observed)
  - gated   : pseudo-view weight = coverage (dist to nearest real cam, normalized)
              -> full in the under-observed wedge, ~0 near real cameras
Both evaluated on the held-out wedge GT. Coverage-gating should >= ungated (it
concentrates the prior where data is weak instead of diluting on observed views).
Per-view coverage validated as a gate signal earlier (Pearson +0.50 vs error).
"""
import json
import os

os.environ["APEX_HOLDOUT"] = "0.275,0.45"

from pathlib import Path

import cv2
import numpy as np
import torch

from core.training import TrainingConfig
from core.training.gsplat_backend import GsplatBackend
from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats
from core.enhancement import EnhancementConfig, get_backend as get_enh

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
OUT = Path(r"F:\HYPERclone\apex-sfm\_dataset\GPU\phase1a")
CKPTB = r"F:\HYPERclone\apex-sfm\_dataset\GPU\proof\baseline\output\ckpts\ckpt_14999_rank0.pt"
FACTOR, SH, dev = 2, 3, "cuda"
backend = GsplatBackend()


def resume_cfg(extra_views, steps=5000):
    return TrainingConfig(backend="gsplat", max_steps=steps, data_factor=FACTOR,
                          test_every=8, save_steps=[steps],
                          extra={"strategy": "mcmc", "cap_max": 3_000_000,
                                 "init_ckpt": CKPTB, "extra_views": extra_views,
                                 "pose_opt": False, "coarse_factor": 0,
                                 "refine_stop_iter": 1, "lpips_lambda": 0.2,
                                 "lpips_gated": True, "lpips_start": 0})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_colmap_data(DATA, factor=FACTOR, test_every=8, normalize=True)
    train_c = data["camtoworlds"][data["train_indices"]][:, :3, 3]
    poses = np.arange(len(data["image_names"]))[::4]          # ~every 4th cam, whole orbit
    print(f"pseudo-poses: {len(poses)} spanning the orbit", flush=True)

    s = torch.load(CKPTB, weights_only=True, map_location=dev)["splats"]
    splats = {k: v.to(dev) for k, v in s.items()}
    rdir = OUT / "renders"; rdir.mkdir(exist_ok=True)
    rec = []  # (path, K_scaled_list, c2w_list, coverage_dist)
    with torch.no_grad():
        for j, idx in enumerate(poses):
            b = load_image(data, int(idx)); K = b["K"].to(dev); c2w = b["c2w"].to(dev)
            H, W = b["image"].shape[:2]
            r, _, _ = render_splats(splats, torch.linalg.inv(c2w), K, W, H, SH,
                                    torch.zeros(3, device=dev), "antialiased")
            bgr = (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]
            sc = min(1.0, 1280.0 / max(H, W))
            if sc < 1.0:
                bgr = cv2.resize(bgr, (round(W * sc), round(H * sc)), interpolation=cv2.INTER_AREA)
            Ksc = K.clone(); Ksc[0, :] *= sc; Ksc[1, :] *= sc
            p = rdir / f"p_{j:04d}.png"; cv2.imwrite(str(p), bgr)
            cov = float(np.min(np.linalg.norm(train_c - c2w[:3, 3].cpu().numpy(), axis=1)))
            rec.append([str(p), Ksc.cpu().tolist(), c2w.cpu().tolist(), cov])
    del splats; torch.cuda.empty_cache()

    # gated-Difix enhance all pseudo-views
    enh = get_enh("difix")
    enhanced = enh.enhance_images([r[0] for r in rec], str(OUT / "difix"),
                                  EnhancementConfig(backend="difix", num_views=len(rec),
                                                    extra={"gated": True}))
    enhanced = sorted(enhanced)
    for r, ep in zip(rec, enhanced):
        r[0] = ep

    # coverage weight in [0,1]: 0 near real cams, ->1 in the under-observed wedge
    cov = np.array([r[3] for r in rec])
    p90 = np.percentile(cov, 90) + 1e-9
    wcov = np.clip(cov / p90, 0.0, 1.0)
    print(f"coverage weights: mean {wcov.mean():.2f}  "
          f"observed(min) {wcov.min():.2f}  wedge(max) {wcov.max():.2f}", flush=True)

    results = {}
    for arm, weights in [("ungated", np.ones_like(wcov)), ("gated", wcov)]:
        entries = [{"image": r[0], "K": r[1], "c2w": r[2], "weight": float(w)}
                   for r, w in zip(rec, weights)]
        evj = OUT / f"extra_{arm}.json"; evj.write_text(json.dumps(entries), encoding="utf-8")
        res = backend.train(data_dir=DATA, output_dir=str(OUT / arm),
                            config=resume_cfg(str(evj)),
                            progress_cb=lambda s, t, l, p: s % 2500 == 0 and
                            print(f"  {arm} {s}/{t} psnr {p:.2f}", flush=True))
        results[arm] = (res.psnr, res.ssim)
        print(f"[{arm.upper()}] wedge PSNR {res.psnr:.2f} SSIM {res.ssim:.4f}", flush=True)

    print("\n=== PHASE-1a RESULT (held-out wedge GT) ===")
    print("baseline (floor) PSNR 21.56  SSIM 0.8376   [proof: ungated-all-wedge +0.53]")
    for arm in ("ungated", "gated"):
        p, q = results[arm]
        print(f"{arm:8s} PSNR {p:.2f} ({p-21.56:+.2f})  SSIM {q:.4f}")


if __name__ == "__main__":
    main()
