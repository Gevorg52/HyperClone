"""WP-T2 mechanism check: train/test masked-PSNR gap for the factor-2 detail model
(pose_opt OFF) vs the factor-3 production model (pose_opt ON). WP-N1 showed pose_opt
overfits train poses at higher-res photometric gradient -> a large train/test gap;
pose_opt OFF should give ~0 gap. Renders a deterministic even sample of train views
+ all test views at the given factor, masked-PSNR (visibility mask), gap = mean(train)-mean(test).

NOTE: for a pose_opt-ON model this renders train views at ORIGINAL (un-refined) poses,
so its train PSNR is a conservative (low) proxy -> the pose_opt-ON gap is UNDER-stated
here; the point is the pose_opt-OFF model's gap is ~0.

Usage:
  python _infusion/wpt2_gap.py --data_dir <data> --factor 2 --train_sample 60 \
    --arms prod=<ckpt3> detail=<ckpt2> --out _infusion/wpt2_gap.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats  # noqa: E402
from core.qa.visibility import VisibilityConfig, build_visibility_mask  # noqa: E402
from core.qa.masked_metrics import masked_psnr  # noqa: E402


def load_splats(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    return {k: v.to(device).float() for k, v in ck["splats"].items()}


def psnr_over(splats, data, idxs, sh_degree, device, vcfg):
    out = []
    with torch.no_grad():
        for idx in idxs:
            b = load_image(data, int(idx))
            gt = b["image"].to(device); K = b["K"].to(device); c2w = b["c2w"].to(device)
            H, W = gt.shape[:2]; w2c = torch.linalg.inv(c2w)
            r, a, _ = render_splats(splats, w2c, K, W, H, sh_degree,
                                    torch.zeros(3, device=device), "antialiased")
            m = torch.from_numpy(build_visibility_mask(a[0, ..., 0].clamp(0, 1), vcfg)["observed"]).to(device)
            out.append(masked_psnr(r[0].clamp(0, 1), gt, m))
            del r, a, gt
    return float(np.nanmean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--train_sample", type=int, default=60)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    data = load_colmap_data(args.data_dir, factor=args.factor, test_every=8, normalize=True)
    tr = np.array(sorted(int(x) for x in data["train_indices"]))
    k = min(args.train_sample, len(tr))
    train_idx = tr[np.linspace(0, len(tr) - 1, k).round().astype(int)]
    test_idx = sorted(int(x) for x in data["test_indices"])
    vcfg = VisibilityConfig()

    res = {"data_dir": str(args.data_dir), "factor": args.factor,
           "n_train_sample": int(k), "n_test": len(test_idx), "arms": {}}
    for spec in args.arms:
        lbl, ck = spec.split("=", 1)
        sp = load_splats(ck, device)
        tr_p = psnr_over(sp, data, train_idx, args.sh_degree, device, vcfg)
        te_p = psnr_over(sp, data, test_idx, args.sh_degree, device, vcfg)
        res["arms"][lbl] = dict(ckpt=str(ck), train_psnr=tr_p, test_psnr=te_p,
                                gap=tr_p - te_p)
        print(f"{lbl:8s} train {tr_p:.2f}  test {te_p:.2f}  gap {tr_p-te_p:+.2f} dB", flush=True)
        del sp; torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
