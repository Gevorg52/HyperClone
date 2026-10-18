"""WP-F5 tone diagnostics: how far is the exported PLY's colour from the photos, and how much of an
arm's held-out error is TONE (fixable by a global per-image colour map) vs STRUCTURE/DETAIL?

PPISP / the bilateral grid absorb per-image exposure, white balance, vignetting and the camera
response curve during training, so the Gaussians converge to a canonical radiance that is NOT
necessarily the photos' tone. Postshot shows that radiance as-is. Per held-out view this script
reports, full frame, VGG-LPIPS (torchmetrics normalize=True, same model as the trainer) + PSNR for:
  raw   the PLY render exactly as a viewer shows it                       (the product metric)
  cc    raw after a per-image per-channel AFFINE colour fit to the GT     (gsplat.color_correct_affine;
        removes tone -> isolates structure/detail; multinerf protocol)
  isp   PPISP arms only: render -> PPISP(camera_idx, frame_idx=-1)        (controller if trained,
        else identity per-frame correction + per-camera vignetting/CRF)
Also: mean RGB of raw render vs GT (global tone shift).

Runs in the gsplat-main venv (needs ppisp + gsplat.color_correct_affine):
  venvs/gsplat_main/Scripts/python.exe _infusion/wpf5_tone_eval.py --data_dir <data> --factor 2 \
      --arm base=<ckpt> --arm pp=<ckpt> --out _infusion/wpf5/<scene>_tone.json [--holdout_list f]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from core.training.gsplat_trainer import (  # noqa: E402
    get_lpips, load_colmap_data, load_image, render_splats)
from gsplat import color_correct_affine  # noqa: E402


def psnr(a, b):
    return float(10 * np.log10(1.0 / (F.mse_loss(a, b).item() + 1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--holdout_list", default=None)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--max_views", type=int, default=0, help="0 = all held-out views")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    data = load_colmap_data(a.data_dir, factor=a.factor, test_every=8, normalize=True,
                            holdout_list=a.holdout_list)
    test = [int(i) for i in data["test_indices"]]
    if a.max_views:
        test = test[:: max(1, len(test) // a.max_views)][: a.max_views]
    lp = get_lpips(dev)
    res = {"argv": sys.argv, "n_views": len(test), "arms": {}}
    for spec in a.arm:
        name, ck = spec.split("=", 1)
        payload = torch.load(ck, map_location="cpu", weights_only=True)
        splats = {k: v.float().to(dev) for k, v in payload["splats"].items()}
        sh_degree = int(round((splats["shN"].shape[1] + 1) ** 0.5)) - 1
        isp = None
        if "ppisp" in payload:
            from ppisp import PPISP, PPISPConfig
            meta = payload["ppisp_meta"]
            isp = PPISP.from_state_dict(
                {k: v.to(dev) for k, v in payload["ppisp"].items()},
                config=PPISPConfig(use_controller=bool(meta["use_controller"]))).eval()
            cam_of = {int(k): int(v) for k, v in meta["camera_of"].items()}
        acc = {k: [] for k in ("raw_lpips", "raw_psnr", "cc_lpips", "cc_psnr",
                               "isp_lpips", "isp_psnr", "raw_mean_rgb", "gt_mean_rgb")}
        with torch.no_grad():
            for i in test:
                b = load_image(data, i)
                gt = b["image"].to(dev); K = b["K"].to(dev); c2w = b["c2w"].to(dev)
                H, W = gt.shape[:2]
                r, _, _ = render_splats(splats, torch.linalg.inv(c2w), K, W, H, sh_degree,
                                        torch.zeros(3, device=dev), "antialiased")
                rawu = r[0][..., :3]
                raw = rawu.clamp(0, 1)
                cc = color_correct_affine(raw, gt).clamp(0, 1)
                g4 = gt.permute(2, 0, 1)[None]
                acc["raw_lpips"].append(lp(raw.permute(2, 0, 1)[None], g4).item())
                acc["raw_psnr"].append(psnr(raw, gt))
                acc["cc_lpips"].append(lp(cc.permute(2, 0, 1)[None], g4).item())
                acc["cc_psnr"].append(psnr(cc, gt))
                acc["raw_mean_rgb"].append(raw.mean((0, 1)).tolist())
                acc["gt_mean_rgb"].append(gt.mean((0, 1)).tolist())
                if isp is not None:
                    o = isp(rawu.contiguous(), resolution=(W, H),
                            camera_idx=cam_of[int(data["camera_ids"][i])], frame_idx=-1).clamp(0, 1)
                    acc["isp_lpips"].append(lp(o.permute(2, 0, 1)[None], g4).item())
                    acc["isp_psnr"].append(psnr(o, gt))
        summ = {k: (float(np.mean(v)) if v and not isinstance(v[0], list) else
                    (np.mean(np.array(v), 0).round(4).tolist() if v else None))
                for k, v in acc.items()}
        res["arms"][name] = {"ckpt": ck, "summary": summ}
        print(name, json.dumps(summ), flush=True)
        del splats; torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
