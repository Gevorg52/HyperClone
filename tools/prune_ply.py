"""Prune low-opacity ("fog") and runaway Gaussians out of a 3DGS PLY / checkpoint.

Built for WP-O5. Low-opacity Gaussians contribute almost nothing to an on-path render but
they are what a viewer flies THROUGH when you leave the capture path, so they read as haze,
veils and floaters in free-viewpoint navigation. Deleting them is lossy but cheap and
reversible (the source PLY/checkpoint is never modified).

Two filters, both default-OFF-ish in the sense that you opt into thresholds:
  --opacity_min   drop Gaussians with sigmoid(opacity) < t     (WP-O5 deliverable uses 0.05)
  --bbox_scale    drop Gaussians further than s x the half-extent of the robust (1-99
                  percentile) point bbox from its centre                    (0 = off)

Input may be either a gsplat PLY (`gsplat.export_splats(format="ply")`, which is what
`core/training/gsplat_trainer.py` writes) or a trainer checkpoint (`ckpt["splats"]`). Output is
always a gsplat PLY written by `gsplat.export_splats`, so it stays byte-compatible with every
viewer that reads the trainer's own output (Postshot, SuperSplat, ...).

IMPORTANT — parameters are stored RAW: `opacity` is a logit and `scale_*` are logs. The
thresholds above are applied to the ACTIVATED values (sigmoid / exp), which is what a viewer
displays.

  python tools/prune_ply.py --ply run/ply/point_cloud_199999.ply --opacity_min 0.05 \
      --out run/ply/point_cloud_199999_op05.ply --stats_json run/ply/prune_stats.json
  python tools/prune_ply.py --ckpt run/ckpts/ckpt_29999_rank0.pt --opacity_min 0.05 --out x.ply
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch


def read_gsplat_ply(path: Path) -> dict:
    """Read a binary_little_endian float32 PLY into the trainer's `splats` layout."""
    with open(path, "rb") as fh:
        header = b""
        while b"end_header\n" not in header:
            chunk = fh.read(1)
            if not chunk:
                raise ValueError(f"{path}: no end_header found")
            header += chunk
        text = header.decode("ascii", errors="replace")
        if "binary_little_endian" not in text:
            raise ValueError(f"{path}: only binary_little_endian PLYs are supported")
        n = int(re.search(r"element vertex (\d+)", text).group(1))
        props = re.findall(r"property float (\S+)", text)
        if len(props) * 4 * n <= 0:
            raise ValueError(f"{path}: empty PLY")
        raw = np.fromfile(fh, dtype="<f4", count=n * len(props)).reshape(n, len(props))
    col = {name: i for i, name in enumerate(props)}

    def take(names):
        return torch.from_numpy(np.ascontiguousarray(raw[:, [col[x] for x in names]]))

    f_rest = sorted((p for p in props if p.startswith("f_rest_")),
                    key=lambda s: int(s.split("_")[-1]))
    sh0 = take(["f_dc_0", "f_dc_1", "f_dc_2"]).reshape(n, 1, 3)
    # gsplat writes shN as (N, 3, K) flattened C-order, i.e. all K of channel 0 first.
    shN = take(f_rest).reshape(n, 3, -1).permute(0, 2, 1).contiguous() if f_rest \
        else torch.zeros(n, 0, 3)
    return {"means": take(["x", "y", "z"]),
            "sh0": sh0, "shN": shN,
            "opacities": take(["opacity"]).reshape(n),
            "scales": take(["scale_0", "scale_1", "scale_2"]),
            "quats": take([p for p in ("rot_0", "rot_1", "rot_2", "rot_3") if p in col])}


def read_ckpt(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    s = {k: v.float() for k, v in ckpt["splats"].items()}
    n = s["means"].shape[0]
    return {"means": s["means"], "scales": s["scales"], "quats": s["quats"],
            "opacities": s["opacities"].reshape(n),
            "sh0": s["sh0"].reshape(n, 1, 3),
            "shN": s["shN"].reshape(n, -1, 3)}


def effective_rank(log_scales: torch.Tensor) -> torch.Tensor:
    """Per-Gaussian effective rank in [1, 3]; needle -> 1, isotropic blob -> 3.

    The metric behind WP-Q2's anti-needle loss (arXiv:2406.11672 Eq. 7-10, and the same
    formula `gsplat_trainer.py::erank_loss` optimises). Reported here because a needle only
    looks correct from the viewpoint it was fitted at — it is the free-viewpoint failure mode
    that opacity statistics cannot see.
    """
    lam = torch.exp(log_scales.double() * 2)                     # s_i^2 = cov eigenvalues
    p = lam / lam.sum(dim=1, keepdim=True).clamp_min(1e-30)
    h = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1)
    return torch.exp(h)


def splat_stats(splats: dict, opacity_bins=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)) -> dict:
    """Opacity histogram + spatial spread + needle fraction.

    The WP-O3 collapse detector (opacity/means), the WP-O5 fog fractions, and the WP-Q2
    needle metric.
    """
    op = torch.sigmoid(splats["opacities"]).numpy()
    means = splats["means"].numpy()
    scale = np.exp(splats["scales"].numpy())
    er = effective_rank(splats["scales"]).numpy()
    lo, hi = np.percentile(means, [1, 99], axis=0)
    centre = (lo + hi) / 2.0
    half = np.maximum((hi - lo) / 2.0, 1e-9)
    rel = np.abs(means - centre) / half
    radius = rel.max(axis=1)
    return {
        "num_gaussians": int(op.shape[0]),
        "opacity_mean": float(op.mean()), "opacity_median": float(np.median(op)),
        "opacity_frac_below": {str(t): float((op < t).mean()) for t in opacity_bins},
        "opacity_hist_edges": [0.0, *opacity_bins, 1.0],
        "opacity_hist": np.histogram(op, bins=[0.0, *opacity_bins, 1.0])[0].tolist(),
        "bbox_p1": lo.tolist(), "bbox_p99": hi.tolist(),
        "bbox_extent": (hi - lo).tolist(),
        "outliers_beyond_3x_bbox": int((radius > 3.0).sum()),
        "outliers_beyond_10x_bbox": int((radius > 10.0).sum()),
        "rel_radius_p99": float(np.percentile(radius, 99)),
        "rel_radius_max": float(radius.max()),
        "mean_norm_median": float(np.median(np.linalg.norm(means, axis=1))),
        "mean_norm_p99": float(np.percentile(np.linalg.norm(means, axis=1), 99)),
        "scale_max_axis_median": float(np.median(scale.max(axis=1))),
        "erank_mean": float(er.mean()),
        "erank_median": float(np.median(er)),
        "needle_frac_erank_lt_2": float((er < 2.0).mean()),
        "needle_frac_erank_lt_1_5": float((er < 1.5).mean()),
        "aspect_ratio_median": float(np.median(scale.max(axis=1)
                                               / np.maximum(scale.min(axis=1), 1e-20))),
    }


def prune(splats: dict, opacity_min: float, bbox_scale: float,
          erank_min: float = 0.0) -> tuple:
    n = splats["means"].shape[0]
    keep = torch.ones(n, dtype=torch.bool)
    reasons = {}
    if erank_min > 0:
        fat = effective_rank(splats["scales"]) >= erank_min
        reasons["dropped_needle"] = int((~fat).sum())
        keep &= fat
    if opacity_min > 0:
        alive = torch.sigmoid(splats["opacities"]) >= opacity_min
        reasons["dropped_low_opacity"] = int((~alive).sum())
        keep &= alive
    if bbox_scale > 0:
        means = splats["means"].numpy()
        lo, hi = np.percentile(means, [1, 99], axis=0)
        centre, half = (lo + hi) / 2.0, np.maximum((hi - lo) / 2.0, 1e-9)
        inside = torch.from_numpy(
            (np.abs(means - centre) / half).max(axis=1) <= bbox_scale)
        reasons["dropped_spatial_outlier"] = int((~inside & keep).sum())
        keep &= inside
    out = {k: v[keep] for k, v in splats.items()}
    reasons["kept"] = int(keep.sum())
    reasons["dropped_total"] = int(n - keep.sum())
    reasons["kept_frac"] = round(float(keep.float().mean()), 6)
    return out, reasons


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python tools/prune_ply.py",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ply", help="gsplat PLY to prune")
    src.add_argument("--ckpt", help="gsplat_trainer checkpoint to prune")
    p.add_argument("--out", help="output PLY (omit with --stats_only)")
    p.add_argument("--opacity_min", type=float, default=0.05,
                   help="drop Gaussians whose sigmoid(opacity) is below this (0 = off)")
    p.add_argument("--bbox_scale", type=float, default=0.0,
                   help="drop Gaussians beyond this multiple of the robust p1-p99 bbox "
                        "half-extent from its centre (0 = off)")
    p.add_argument("--erank_min", type=float, default=0.0,
                   help="drop Gaussians whose effective rank is below this — i.e. needles, "
                        "which are only correct from the view they were fitted at (0 = off). "
                        "DIAGNOSTIC: deleting needles leaves holes; the real fix is to not "
                        "grow them (--erank_lambda at train time)")
    p.add_argument("--stats_json", help="write before/after stats here")
    p.add_argument("--stats_only", action="store_true", help="report stats, write no PLY")
    args = p.parse_args(argv)
    if not args.stats_only and not args.out:
        p.error("--out is required unless --stats_only")

    import gsplat

    t0 = time.time()
    source = Path(args.ply or args.ckpt)
    splats = read_gsplat_ply(source) if args.ply else read_ckpt(source)
    before = splat_stats(splats)
    print(f"[prune] {source} -> {before['num_gaussians']} gaussians, mean opacity "
          f"{before['opacity_mean']:.4f}, {before['opacity_frac_below']['0.05'] * 100:.1f}% "
          f"below 0.05, {before['outliers_beyond_3x_bbox']} beyond 3x bbox", flush=True)

    payload = {"tool": "tools/prune_ply.py",
               "argv": [sys.executable, __file__] + list(sys.argv[1:]),
               "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "source": str(source.resolve()),
               "opacity_min": args.opacity_min, "bbox_scale": args.bbox_scale,
               "erank_min": args.erank_min, "before": before}

    if not args.stats_only:
        kept, reasons = prune(splats, args.opacity_min, args.bbox_scale, args.erank_min)
        after = splat_stats(kept)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        gsplat.export_splats(means=kept["means"], scales=kept["scales"],
                             quats=kept["quats"], opacities=kept["opacities"],
                             sh0=kept["sh0"], shN=kept["shN"],
                             format="ply", save_to=str(out))
        payload.update(pruned=reasons, after=after, out=str(out.resolve()),
                       out_bytes=out.stat().st_size)
        print(f"[prune] kept {reasons['kept']} ({reasons['kept_frac'] * 100:.1f}%), dropped "
              f"{reasons['dropped_total']} -> {out} "
              f"({out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)", flush=True)

    if args.stats_json:
        Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_json).write_text(json.dumps(payload, indent=1), encoding="utf8")
        print(f"[prune] stats -> {args.stats_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
