"""Phase-1c: native geometry-grounded prior vs the Difix floor.

Render the aligned VGGT cloud (vggt_cloud.npz) at the SAME 130 poses + coverage
weights used in phase1a (extra_gated.json) -- footprint-splat to densify + cv2
inpaint residual holes -- then run the SAME gated-fusion distill and eval on the
held-out wedge. Compare to the Difix-fill floor (phase1a gated: 22.59).
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

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
CKPTB = r"F:\HYPERclone\apex-sfm\_dataset\GPU\proof\baseline\output\ckpts\ckpt_14999_rank0.pt"
CLOUD = r"F:\HYPERclone\apex-sfm\_dataset\GPU\proof\vggt_cloud.npz"
SRC = r"F:\HYPERclone\apex-sfm\_dataset\GPU\phase1a\extra_gated.json"
OUT = Path(r"F:\HYPERclone\apex-sfm\_dataset\GPU\phase1c")
backend = GsplatBackend()


def render_dense(P, C, w2c, K, H, W, r=2):
    Pc = (w2c[:3, :3] @ P.T + w2c[:3, 3:4]).T
    z = Pc[:, 2]; f = z > 1e-3
    Pc, Cc, z = Pc[f], C[f], z[f]
    u = K[0, 0] * Pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / z + K[1, 2]
    order = np.argsort(-z)                                   # far first, near overwrites
    uo, vo = u[order], v[order]
    co = (np.clip(Cc[order], 0, 1) * 255).astype(np.uint8)   # RGB
    img = np.zeros((H, W, 3), np.uint8); filled = np.zeros((H, W), bool)
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            uu = np.round(uo + dc).astype(np.int64); vv = np.round(vo + dr).astype(np.int64)
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            img[vv[ok], uu[ok]] = co[ok]; filled[vv[ok], uu[ok]] = True
    fill_frac = float(filled.mean())
    holes = (~filled).astype(np.uint8) * 255
    if holes.any():
        img = cv2.inpaint(img, holes, 6, cv2.INPAINT_NS)
    return img[..., ::-1], fill_frac                         # BGR for imwrite


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(CLOUD); P = d["points"]; C = d["colors"]
    src = json.loads(Path(SRC).read_text())
    rdir = OUT / "renders"; rdir.mkdir(exist_ok=True)
    entries = []; fills = []
    for i, e in enumerate(src):
        K = np.array(e["K"], np.float64); c2w = np.array(e["c2w"], np.float64)
        w2c = np.linalg.inv(c2w)
        H, W = cv2.imread(e["image"]).shape[:2]
        img, ff = render_dense(P, C, w2c, K, H, W)
        fills.append(ff)
        p = rdir / f"vggt_{i:04d}.png"; cv2.imwrite(str(p), img)
        entries.append({"image": str(p), "K": e["K"], "c2w": e["c2w"], "weight": e["weight"]})
    print(f"rendered {len(entries)} VGGT-cloud views; mean pre-inpaint fill "
          f"{100*np.mean(fills):.0f}%", flush=True)
    evj = OUT / "extra_vggt.json"; evj.write_text(json.dumps(entries))

    cfg = TrainingConfig(backend="gsplat", max_steps=5000, data_factor=2, test_every=8,
                         save_steps=[5000],
                         extra={"strategy": "mcmc", "cap_max": 3_000_000,
                                "init_ckpt": CKPTB, "extra_views": str(evj),
                                "pose_opt": False, "coarse_factor": 0,
                                "refine_stop_iter": 1, "lpips_lambda": 0.2,
                                "lpips_gated": True, "lpips_start": 0})
    r = backend.train(data_dir=DATA, output_dir=str(OUT / "vggt_gated"), config=cfg,
                      progress_cb=lambda s, t, l, p: s % 2500 == 0 and
                      print(f"  {s}/{t} psnr {p:.2f}", flush=True))
    print(f"\n=== PHASE-1c RESULT (held-out wedge GT) ===")
    print(f"baseline           21.56 / .8376")
    print(f"difix-fill (1a)    22.59 / .8532   <- floor to beat")
    print(f"VGGT-geom (1c)     {r.psnr:.2f} / {r.ssim:.4f}   (dPSNR vs floor {r.psnr-22.59:+.2f})")
    print(f"ckpt={r.checkpoint_dir}")


if __name__ == "__main__":
    main()
