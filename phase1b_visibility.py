"""Phase-1b (i): add visibility loss + stronger scale_reg to the gated fusion to
fix the 1a giant-blob uptick (7%->11.6%) WITHOUT losing the quality gain.
Reuses the already-Difix'd, coverage-weighted pseudo-views from phase1a (extra_gated
.json) -> only the resume changes -> fast. Must hold the gated wedge PSNR (22.59)
and restore the giant-blob count.
"""
import os
os.environ["APEX_HOLDOUT"] = "0.275,0.45"

from pathlib import Path
from core.training import TrainingConfig
from core.training.gsplat_backend import GsplatBackend

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
CKPTB = r"F:\HYPERclone\apex-sfm\_dataset\GPU\proof\baseline\output\ckpts\ckpt_14999_rank0.pt"
EVJ = r"F:\HYPERclone\apex-sfm\_dataset\GPU\phase1a\extra_gated.json"
OUT = Path(r"F:\HYPERclone\apex-sfm\_dataset\GPU\phase1b")
backend = GsplatBackend()


def cfg(vis_reg, scale_reg):
    return TrainingConfig(backend="gsplat", max_steps=5000, data_factor=2,
                          test_every=8, save_steps=[5000],
                          extra={"strategy": "mcmc", "cap_max": 3_000_000,
                                 "init_ckpt": CKPTB, "extra_views": EVJ,
                                 "pose_opt": False, "coarse_factor": 0,
                                 "refine_stop_iter": 1, "lpips_lambda": 0.2,
                                 "lpips_gated": True, "lpips_start": 0,
                                 "scale_reg": scale_reg, "vis_reg": vis_reg})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arms = {"gated_vis": (0.1, 0.02)}   # vis_reg, scale_reg
    for name, (vis, sreg) in arms.items():
        r = backend.train(data_dir=DATA, output_dir=str(OUT / name),
                          config=cfg(vis, sreg),
                          progress_cb=lambda s, t, l, p: s % 2500 == 0 and
                          print(f"  {name} {s}/{t} psnr {p:.2f}", flush=True))
        print(f"[{name}] wedge PSNR {r.psnr:.2f} SSIM {r.ssim:.4f}  ckpt={r.checkpoint_dir}",
              flush=True)
    print("\ncompare to phase1a gated: PSNR 22.59 SSIM .8532 (giant-blobs 11.6%)")


if __name__ == "__main__":
    main()
