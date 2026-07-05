"""WP-N1 gate evidence: masked held-out metrics for the native-res isolation
experiment, computed with the WP-S1 / qa_report masked methodology (rendered-
alpha visibility mask, masked PSNR/SSIM/LPIPS; VGG-LPIPS normalize=True).

For each arm ckpt we render every-8 held-out view ONCE at native factor-1
(antialiased, black bg) and derive, from that single render pass:

  NATIVE-res     : score render_f1 vs GT_f1, mask = build_visibility_mask(alpha_f1).
                   Full-frame LPIPS uses TILED mode (64px tiles) so VGG never sees
                   a 24 MP crop (bbox mode would OOM at native res).
  COMMON-res (B) : the June decisive protocol — downsample render_f1 + alpha_f1 to
                   the factor-2 GT resolution (area filter), score vs GT_f2, mask =
                   build_visibility_mask(alpha_f2). LPIPS bbox mode (6 MP, matches
                   qa_report --factor 2 exactly -> directly comparable to the banked
                   F2 stack scorecard and to qa_report --factor 2 run on N-a/N-b).

Train/test gap (the June ~6 dB overfitting signature): a deterministic even
sample of train views is rendered at native factor-1 and scored with native
masked PSNR; gap = mean(train PSNR) - mean(test PSNR).

Usage (repo root):
  python _infusion/wpn1_eval.py \
    --data_dir _dataset/GPU_V4/apex_images/training/wps1/n1/data \
    --arms F2=<stack_ckpt> N-a=<na_ckpt> N-b=<nb_ckpt> \
    --common_factor 2 --train_sample 72 \
    --out _infusion/wpn1_gpu_masked.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.gsplat_trainer import (  # noqa: E402
    load_colmap_data, load_image, render_splats,
)
from core.qa.visibility import VisibilityConfig, build_visibility_mask  # noqa: E402
from core.qa.masked_metrics import masked_psnr, masked_ssim, masked_lpips  # noqa: E402


def load_splats(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    return {k: v.to(device).float() for k, v in ck["splats"].items()}


def render_one(splats, b, sh_degree, device):
    """Render view -> (render(H,W,3) [0,1], alpha(H,W) [0,1], gt(H,W,3))."""
    gt = b["image"].to(device)
    K = b["K"].to(device)
    c2w = b["c2w"].to(device)
    H, W = gt.shape[:2]
    w2c = torch.linalg.inv(c2w)
    r, a, _ = render_splats(splats, w2c, K, W, H, sh_degree,
                            torch.zeros(3, device=device), "antialiased")
    return r[0].clamp(0, 1), a[0, ..., 0].clamp(0, 1), gt


def resize_hw3(x, H, W):
    """x (H0,W0,3) float -> (H,W,3) area-filtered."""
    y = x.permute(2, 0, 1).unsqueeze(0)
    y = F.interpolate(y, size=(H, W), mode="area")
    return y[0].permute(1, 2, 0)


def resize_hw(x, H, W):
    """x (H0,W0) float -> (H,W) area-filtered."""
    y = x.unsqueeze(0).unsqueeze(0)
    y = F.interpolate(y, size=(H, W), mode="area")
    return y[0, 0]


def masked_triplet(render, gt, alpha, device, vcfg, lpips_mode):
    obs = build_visibility_mask(alpha, vcfg)["observed"]
    m = torch.from_numpy(obs).to(device)
    return (masked_psnr(render, gt, m),
            masked_ssim(render, gt, m),
            masked_lpips(render, gt, m, device, mode=lpips_mode),
            100.0 * float(m.float().mean()))


def masked_ps(render, gt, alpha, device, vcfg):
    """PSNR+SSIM only (no LPIPS). Used for NATIVE 24 MP where a full-frame LPIPS
    is neither cheap nor well-defined; native perceptual quality is measured by
    the microstructure region-LPIPS (wps1_region_eval, factor 1)."""
    obs = build_visibility_mask(alpha, vcfg)["observed"]
    m = torch.from_numpy(obs).to(device)
    return (masked_psnr(render, gt, m),
            masked_ssim(render, gt, m),
            100.0 * float(m.float().mean()))


def eval_arm(ckpt, data1, data2, sh_degree, device, train_idx):
    splats = load_splats(ckpt, device)
    vcfg = VisibilityConfig()
    test = sorted(int(x) for x in data1["test_indices"])
    nat = {"psnr": [], "ssim": [], "obs": []}
    com = {"psnr": [], "ssim": [], "lpips": [], "obs": []}
    with torch.no_grad():
        for idx in test:
            b1 = load_image(data1, idx)
            r1, a1, gt1 = render_one(splats, b1, sh_degree, device)
            # NATIVE 24 MP: PSNR+SSIM only (perceptual = region_eval region-LPIPS)
            p, s, o = masked_ps(r1, gt1, a1, device, vcfg)
            nat["psnr"].append(p); nat["ssim"].append(s); nat["obs"].append(o)
            # COMMON-res B: downsample render+alpha to the factor-2 GT shape
            gt2 = load_image(data2, idx)["image"].to(device)
            H2, W2 = gt2.shape[:2]
            r2 = resize_hw3(r1, H2, W2)
            a2 = resize_hw(a1, H2, W2)
            p, s, l, o = masked_triplet(r2, gt2, a2, device, vcfg, "bbox")
            com["psnr"].append(p); com["ssim"].append(s)
            com["lpips"].append(l); com["obs"].append(o)
            del r1, a1, gt1, r2, a2, gt2
        # Train/test gap: native masked PSNR on a deterministic sample of train views
        train_psnr = []
        for idx in train_idx:
            b1 = load_image(data1, int(idx))
            r1, a1, gt1 = render_one(splats, b1, sh_degree, device)
            obs = build_visibility_mask(a1, vcfg)["observed"]
            m = torch.from_numpy(obs).to(device)
            train_psnr.append(masked_psnr(r1, gt1, m))
            del r1, a1, gt1
    torch.cuda.empty_cache()

    def agg(d):
        return {k: float(np.nanmean(v)) for k, v in d.items()}
    out = {"ckpt": str(ckpt), "n_test": len(test),
           "native": agg(nat), "common_res": agg(com),
           "n_train_sample": len(train_idx),
           "train_psnr_native": float(np.nanmean(train_psnr)),
           "test_psnr_native": float(np.nanmean(nat["psnr"]))}
    out["train_test_gap_native"] = out["train_psnr_native"] - out["test_psnr_native"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="label=ckpt pairs")
    ap.add_argument("--common_factor", type=int, default=2)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--train_sample", type=int, default=72)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"

    print(f"[WPN1] loading data (factor 1 native + factor {args.common_factor} common)")
    data1 = load_colmap_data(args.data_dir, factor=1, test_every=8, normalize=True)
    data2 = load_colmap_data(args.data_dir, factor=args.common_factor,
                             test_every=8, normalize=True)
    # deterministic even sample of train views for the gap
    tr = np.array(sorted(int(x) for x in data1["train_indices"]))
    k = min(args.train_sample, len(tr))
    train_idx = tr[np.linspace(0, len(tr) - 1, k).round().astype(int)]

    arms = {}
    for spec in args.arms:
        lbl, ck = spec.split("=", 1)
        arms[lbl] = ck

    result = {"data_dir": str(args.data_dir), "common_factor": args.common_factor,
              "note": "native: PSNR+SSIM at full 24 MP (perceptual = region_eval "
                      "region-LPIPS). common_res: render@f1 -> area-downsample to f2 "
                      "GT shape, bbox LPIPS (== qa_report --factor 2). "
                      "mask = build_visibility_mask (alpha_thresh 0.3). "
                      "VGG-LPIPS normalize=True.",
              "arms": {}}
    for lbl, ck in arms.items():
        print(f"[WPN1] arm {lbl} ...", flush=True)
        r = eval_arm(ck, data1, data2, args.sh_degree, device, train_idx)
        result["arms"][lbl] = r
        print(f"  {lbl}: common LPIPS={r['common_res']['lpips']:.4f} "
              f"PSNR={r['common_res']['psnr']:.3f} SSIM={r['common_res']['ssim']:.4f} | "
              f"native PSNR={r['native']['psnr']:.3f} SSIM={r['native']['ssim']:.4f} | "
              f"train/test gap(native)={r['train_test_gap_native']:.2f} dB "
              f"(train {r['train_psnr_native']:.2f} / test {r['test_psnr_native']:.2f})",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[WPN1] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
