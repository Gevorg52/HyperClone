"""WP-C1 gate evaluation (Part 2). Consumer of core/qa — nothing in core/qa touched.

Two modes:

  --mode fills2d : 2D gate integrity + plausible-vs-actual on the per-view fills,
    BEFORE distillation. For each hole view (fixed OGGSplat observed/unobserved mask
    from the HOLED render alpha):
      * gate integrity = |fill - holed_render| on the OBSERVED region. GATED must be
        byte-identical (feathered paste-back) -> the fill is structurally unable to
        touch real pixels; UNGATED drifts (VAE round-trip). Decisive.
      * plausible-vs-actual = void-region PSNR/SSIM/LPIPS(fill vs real GT). REFERENCE
        ONLY (unobserved regions have no GT claim), never a gate criterion.

  --mode splat : splat-level fixed-mask gate. Given a distilled (filled) splat, render
    each hole view and, using the HOLED-BASELINE observed mask (FIXED, not the filled
    render's own mask), compute observed-region masked PSNR/SSIM/LPIPS vs GT and vs the
    baseline render -> observed-region gate integrity after distill. Plausibility +
    consistency come from qa_report --scorers on the same splat (run separately).

Usage:
  python _infusion/hole_gate.py --mode fills2d --arc _infusion/hole/arc \
      --gated _infusion/hole/fills_gated --ungated _infusion/hole/fills_ungated --out _infusion/hole/gate2d.json
  python _infusion/hole_gate.py --mode splat --arc _infusion/hole/arc \
      --splat <distilled.pt> --data <dir> --factor 2 --out _infusion/hole/gate_splat.json
"""
import argparse, json, os, sys
import numpy as np
import torch
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.pop("APEX_HOLDOUT", None)
from core.qa.visibility import VisibilityConfig, build_visibility_mask
from core.qa.masked_metrics import masked_psnr, masked_ssim, masked_lpips


def _load_rgb(path):
    im = cv2.imread(path)
    return torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).float() / 255.0


def fills2d(args):
    meta = json.load(open(os.path.join(args.arc, "meta.json")))
    dev = "cuda"
    vcfg = VisibilityConfig(alpha_thresh=0.3)
    rows = {"gated": [], "ungated": []}
    for stem in sorted(meta.keys()):
        holed = _load_rgb(os.path.join(args.arc, f"{stem}_render.png")).to(dev)
        gt = _load_rgb(os.path.join(args.arc, f"{stem}_gt.png")).to(dev)
        alpha = torch.from_numpy(np.load(os.path.join(args.arc, f"{stem}_alpha.npy"))).to(dev)
        vis = build_visibility_mask(alpha, vcfg)
        obs = torch.from_numpy(vis["observed"]).to(dev)
        unobs = torch.from_numpy(vis["unobserved"]).to(dev)
        for mode, d in (("gated", args.gated), ("ungated", args.ungated)):
            if not d:
                continue
            fp = os.path.join(d, f"{stem}_fill.png")
            if not os.path.isfile(fp):
                continue
            fill = _load_rgb(fp).to(dev)
            # gate integrity on OBSERVED pixels: fill vs holed render
            om = obs[..., None].float()
            n = om.sum().clamp(min=1)
            mad = float(((fill - holed).abs() * om).sum() / (n * 3))          # mean abs diff
            mse = float((((fill - holed) ** 2) * om).sum() / (n * 3))
            gi_psnr = 99.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)
            # plausible-vs-actual on the VOID (reference-only)
            vp = masked_psnr(fill, gt, unobs); vl = masked_lpips(fill, gt, unobs, dev, mode="bbox")
            rows[mode].append((stem, mad, gi_psnr, float(vis["observed_pct"]), vp, vl))
    out = {"per_view": {}, "summary": {}}
    for mode, rr in rows.items():
        if not rr:
            continue
        mad = np.array([r[1] for r in rr]); gi = np.array([r[2] for r in rr])
        vp = np.array([r[4] for r in rr]); vl = np.array([r[5] for r in rr])
        out["summary"][mode] = {
            "n": len(rr),
            "observed_MAD_mean": float(mad.mean()),          # 0 == cannot touch observed
            "observed_MAD_max": float(mad.max()),
            "observed_gate_PSNR_mean": float(gi.mean()),     # inf-ish == identical
            "void_vs_GT_PSNR_mean": float(vp.mean()),        # reference-only
            "void_vs_GT_LPIPS_mean": float(vl.mean()),
        }
        out["per_view"][mode] = [
            {"view": r[0], "observed_MAD": r[1], "observed_gate_PSNR": r[2],
             "observed_pct": r[3], "void_PSNR": r[4], "void_LPIPS": r[5]} for r in rr]
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[gate2d] wrote {args.out}")
    for mode, s in out["summary"].items():
        print(f"  {mode:8s}: observed MAD {s['observed_MAD_mean']:.6f} "
              f"(max {s['observed_MAD_max']:.6f}), observed gate-PSNR {s['observed_gate_PSNR_mean']:.1f} dB | "
              f"void-vs-GT PSNR {s['void_vs_GT_PSNR_mean']:.2f} LPIPS {s['void_vs_GT_LPIPS_mean']:.4f}")


def splat_mode(args):
    from qa_report import load_splats
    from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats
    dev = "cuda"
    meta = json.load(open(os.path.join(args.arc, "meta.json")))
    data = load_colmap_data(args.data, factor=args.factor, test_every=8, normalize=True)
    sp = load_splats(args.splat, dev)
    vcfg = VisibilityConfig(alpha_thresh=0.3)
    rows = []
    for stem in sorted(meta.keys()):
        idx = meta[stem]["index"]
        b = load_image(data, idx)
        gt = b["image"].to(dev); K = b["K"].to(dev); c2w = b["c2w"].to(dev)
        H, W = gt.shape[:2]
        with torch.no_grad():
            r, _, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, 3,
                                    torch.zeros(3, device=dev), "antialiased")
        render = r[0].clamp(0, 1)
        # FIXED baseline observed mask (from the holed-baseline alpha exported earlier)
        alpha_b = torch.from_numpy(np.load(os.path.join(args.arc, f"{stem}_alpha.npy"))).to(dev)
        obs = torch.from_numpy(build_visibility_mask(alpha_b, vcfg)["observed"]).to(dev)
        holed = _load_rgb(os.path.join(args.arc, f"{stem}_render.png")).to(dev)
        mp = masked_psnr(render, gt, obs); ms = masked_ssim(render, gt, obs)
        # LPIPS on the observed region, TWO ways:
        #  * TILED (primary, review-corrected H1): tile the FULL-RES frame into
        #    `tile`px crops, keep only tiles with >=`cover` observed coverage, average.
        #    This is the only sound observed-region LPIPS when the observed region is an
        #    ANNULUS around a central void — bbox mode's crop = whole frame there
        #    (bbox area frac -> 1.0), so it silently measures FULL-FRAME LPIPS incl. the
        #    plausible-not-faithful void interior (see wpc1_RECORD H1).
        #  * BBOX (deprecated_fullframe): kept only for provenance/back-compat.
        ml_tiled = masked_lpips(render, gt, obs, dev, mode="tiled",
                                tile=args.tile, cover=args.cover)
        ml_bbox = masked_lpips(render, gt, obs, dev, mode="bbox")
        # drift vs holed baseline on observed (splat-level gate integrity)
        om = obs[..., None].float(); n = om.sum().clamp(min=1)
        mad = float(((render - holed).abs() * om).sum() / (n * 3))
        rows.append((stem, mp, ms, ml_tiled, ml_bbox, mad))
    mp = np.array([r[1] for r in rows]); ms = np.array([r[2] for r in rows])
    mlt = np.array([r[3] for r in rows]); mlb = np.array([r[4] for r in rows])
    mad = np.array([r[5] for r in rows])
    out = {
        "n": len(rows), "fixed_baseline_mask": True,
        "lpips_tile": args.tile, "lpips_cover": args.cover, "full_resolution": True,
        "observed_masked_PSNR": float(mp.mean()), "observed_masked_SSIM": float(ms.mean()),
        # H1-corrected primary: tiled observed-region LPIPS (fixed baseline mask, full res)
        "observed_masked_LPIPS_tiled": float(np.nanmean(mlt)),
        # DEPRECATED — accidental full-frame LPIPS (bbox crop == whole frame around a
        # central void); retained for provenance only, NOT a gate criterion.
        "observed_masked_LPIPS_deprecated_fullframe": float(mlb.mean()),
        "observed_drift_vs_holed_MAD": float(mad.mean()),
        "per_view": [
            {"view": r[0], "observed_masked_LPIPS_tiled": r[3],
             "observed_masked_LPIPS_deprecated_fullframe": r[4],
             "observed_masked_PSNR": r[1], "observed_masked_SSIM": r[2],
             "observed_drift_vs_holed_MAD": r[5]} for r in rows],
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[gate_splat] {args.splat}")
    print(f"  observed (fixed baseline mask, tile={args.tile} cover={args.cover}, full res): "
          f"PSNR {out['observed_masked_PSNR']:.3f} SSIM {out['observed_masked_SSIM']:.4f} "
          f"LPIPS_tiled {out['observed_masked_LPIPS_tiled']:.4f} "
          f"(deprecated fullframe {out['observed_masked_LPIPS_deprecated_fullframe']:.4f}) | "
          f"drift-vs-holed MAD {out['observed_drift_vs_holed_MAD']:.6f}")
    print(f"[gate_splat] wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fills2d", "splat"], required=True)
    ap.add_argument("--arc", required=True)
    ap.add_argument("--gated", default=None)
    ap.add_argument("--ungated", default=None)
    ap.add_argument("--splat", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--tile", type=int, default=256, help="tiled-LPIPS tile size (px, full-res)")
    ap.add_argument("--cover", type=float, default=0.8, help="tiled-LPIPS min observed coverage")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "fills2d":
        fills2d(args)
    else:
        splat_mode(args)


if __name__ == "__main__":
    main()
