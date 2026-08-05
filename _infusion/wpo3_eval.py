"""WP-O3 gate evidence: held-out metrics, visual checks and a walk-through video.

Consumes a `core.train_360_prep` dataset + a `gsplat_trainer.py` checkpoint and writes
everything the WP-O3 gate needs into one directory:

  metrics.json    per-view + aggregate masked/full PSNR-SSIM-LPIPS on the held-out crops
  scorecard.json  the aggregate + the run's argv/seed/config (the committable summary)
  grid_heldout_*.jpg   render | GT | mask strips over held-out panoramas
  closeup_seam_*.jpg   the two equator crops that look straight at the stitch seams
  closeup_operator_*.jpg  the held-out crops with the most WP-O1-masked area
  closeup_thin_*.jpg   the held-out crops with the most high-frequency detail
  walkthrough.mp4      forward-facing render along the SfM trajectory, pano order

The masked metrics use the WP-O2 per-crop mask (255 = VALID). That mask composites the
exclusive nearest-camera tiling with WP-O1's operator/nadir/seam exclusions, so "masked"
here means "on the pixels this crop is actually responsible for and that are real scene".

Usage:
  python _infusion/wpo3_eval.py --ckpt <ckpt.pt> --data_dir <packaged> --out _infusion/wpo3
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.qa.masked_metrics import masked_lpips, masked_psnr, masked_ssim  # noqa: E402
from core.training.gsplat_trainer import (  # noqa: E402
    load_colmap_data,
    load_image,
    render_splats,
)

# WP-O2 crop layout (core.sfm_360.get_virtual_rotations, perspective_overlapping):
# indices 0-3   pitch -35 deg, yaw   0/90/180/270
#         4-7   pitch   0 deg, yaw   0/90/180/270   <- equator
#         8-11  pitch +35 deg, yaw  45/135/225/315
# WP-O1 puts the stitch seams at ERP x = W/4 and 3W/4, i.e. yaw -90 and +90, which is
# exactly the optical axis of equator crops 5 (yaw 90) and 7 (yaw 270 = -90).
SEAM_CAMS = (5, 7)


def load_splats(ckpt_path: Path, device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    splats = {k: v.to(device) for k, v in ckpt["splats"].items()}
    return splats, int(ckpt.get("step", -1))


def render_view(splats, data, idx, device, sh_degree=3, rasterize_mode="antialiased"):
    """(render HxWx3 [0,1], gt HxWx3, mask HxW bool or None) for one dataset index."""
    batch = load_image(data, idx)
    gt = batch["image"].to(device)
    K = batch["K"].to(device)
    c2w = batch["c2w"].to(device)
    H, W = gt.shape[:2]
    with torch.no_grad():
        renders, _, _ = render_splats(
            splats, torch.linalg.inv(c2w), K, W, H, sh_degree,
            torch.zeros(3, device=device), rasterize_mode)
    render = renders[0][..., :3].clamp(0, 1)
    mask = None
    if "mask" in batch:
        mask = (batch["mask"].to(device) > 0.5)
    return render, gt, mask


def to_bgr(t: torch.Tensor) -> np.ndarray:
    return cv2.cvtColor((t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return out


def mask_overlay(gt_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """GT dimmed + red where the mask EXCLUDES (so the excluded region is obvious)."""
    out = gt_bgr.copy()
    excl = ~mask
    out[excl] = (0.35 * out[excl] + 0.65 * np.array([0, 0, 200], np.float32)).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# 1. metrics
# ---------------------------------------------------------------------------

def run_metrics(args, splats, data, device, out: Path) -> dict:
    rows = []
    t0 = time.time()
    test_indices = list(data["test_indices"])
    for n, idx in enumerate(test_indices):
        render, gt, mask = render_view(splats, data, int(idx), device, args.sh_degree,
                                       args.rasterize_mode)
        name = Path(data["image_paths"][int(idx)]).name
        row = {"name": name, "index": int(idx)}
        full_mask = torch.ones(gt.shape[:2], dtype=torch.bool, device=device)
        row["full_psnr"] = masked_psnr(render, gt, full_mask)
        row["full_ssim"] = masked_ssim(render, gt, full_mask)
        row["full_lpips"] = masked_lpips(render, gt, full_mask, device, mode="bbox")
        if mask is not None:
            row["valid_frac"] = float(mask.float().mean())
            row["masked_psnr"] = masked_psnr(render, gt, mask)
            row["masked_ssim"] = masked_ssim(render, gt, mask)
            row["masked_lpips_bbox"] = masked_lpips(render, gt, mask, device, mode="bbox")
            row["masked_lpips_tiled"] = masked_lpips(
                render, gt, mask, device, mode="tiled", tile=args.tile, cover=args.cover)
        rows.append(row)
        if args.progress and (n + 1) % args.progress == 0:
            print(f"[metrics] {n + 1}/{len(test_indices)} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    def agg(key):
        vals = [r[key] for r in rows if key in r and not np.isnan(r[key])]
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
                "std": float(np.std(vals)), "min": float(np.min(vals)),
                "max": float(np.max(vals)), "n": len(vals)} if vals else None

    keys = ["full_psnr", "full_ssim", "full_lpips", "valid_frac",
            "masked_psnr", "masked_ssim", "masked_lpips_bbox", "masked_lpips_tiled"]
    summary = {k: agg(k) for k in keys}
    panos = sorted({r["name"].split("_cam")[0] for r in rows})
    payload = {"num_views": len(rows), "num_panos": len(panos), "panos": panos,
               "summary": summary, "views": rows,
               "seconds": round(time.time() - t0, 2)}
    (out / "metrics.json").write_text(json.dumps(payload, indent=1), encoding="utf8")
    print(f"[metrics] {len(rows)} held-out crops over {len(panos)} panos in "
          f"{payload['seconds']:.0f}s", flush=True)
    for k in keys:
        if summary[k]:
            print(f"    {k:22s} mean {summary[k]['mean']:.4f}  "
                  f"median {summary[k]['median']:.4f}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# 2. figures
# ---------------------------------------------------------------------------

def strip(render, gt, mask, name: str, scale: float) -> np.ndarray:
    r, g = to_bgr(render), to_bgr(gt)
    tiles = [label(g, f"GT  {name}"), label(r, "RENDER")]
    if mask is not None:
        tiles.append(label(mask_overlay(g, mask.cpu().numpy()),
                           f"MASK (red = excluded, {float(mask.float().mean()) * 100:.0f}% valid)"))
    row = np.concatenate(tiles, axis=1)
    if scale != 1.0:
        row = cv2.resize(row, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return row


def run_figures(args, splats, data, device, out: Path, metrics: dict) -> dict:
    names = [Path(p).name for p in data["image_paths"]]
    by_name = {n: i for i, n in enumerate(names)}
    test_names = [r["name"] for r in metrics["views"]]
    test_panos = metrics["panos"]
    written = {}

    # (0) held-out grids: for a few evenly spaced held-out panos, the equator ring.
    picks = [test_panos[i] for i in
             np.linspace(0, len(test_panos) - 1, min(4, len(test_panos))).astype(int)]
    for pano in picks:
        rows = []
        for cam in (4, 5, 6, 7):
            nm = f"{pano}_cam{cam:02d}.jpg"
            if nm not in by_name:
                continue
            r, g, m = render_view(splats, data, by_name[nm], device, args.sh_degree,
                                  args.rasterize_mode)
            rows.append(strip(r, g, m, nm, args.fig_scale))
        if rows:
            p = out / f"grid_heldout_{pano}.jpg"
            cv2.imwrite(str(p), np.concatenate(rows, axis=0),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            written[p.name] = f"held-out pano {pano}, equator crops 4-7 (GT|render|mask)"

    # (1) seam band: the equator crops whose optical axis IS the stitch seam.
    for pano in picks[:2]:
        rows = []
        for cam in SEAM_CAMS:
            nm = f"{pano}_cam{cam:02d}.jpg"
            if nm not in by_name:
                continue
            r, g, m = render_view(splats, data, by_name[nm], device, args.sh_degree,
                                  args.rasterize_mode)
            rows.append(strip(r, g, m, f"{nm} (seam through centre)", args.fig_scale))
        if rows:
            p = out / f"closeup_seam_{pano}.jpg"
            cv2.imwrite(str(p), np.concatenate(rows, axis=0),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            written[p.name] = (f"seam check, pano {pano}: crops {SEAM_CAMS} centre on the "
                               f"ffmpeg v360 stitch seams (yaw +-90 deg)")

    # (2) operator / most-masked held-out crops. The WP-O1 exclusions (operator, nadir
    # stick+feet, seam bands) are the part of the mask that is NOT the exclusive tiling,
    # so rank by masked area inside the crop and show the top ones.
    ranked = sorted((r for r in metrics["views"] if "valid_frac" in r),
                    key=lambda r: r["valid_frac"])
    for r in ranked[:args.n_closeups]:
        idx = by_name[r["name"]]
        rr, g, m = render_view(splats, data, idx, device, args.sh_degree,
                               args.rasterize_mode)
        p = out / f"closeup_operator_{r['name'].replace('.jpg', '')}.jpg"
        cv2.imwrite(str(p), strip(rr, g, m, f"{r['name']} (most-excluded held-out crop)",
                                  1.0), [cv2.IMWRITE_JPEG_QUALITY, 95])
        written[p.name] = (f"ghost check: {r['name']}, only "
                           f"{r['valid_frac'] * 100:.0f}% of the crop is valid — the red "
                           f"region is operator/nadir/seam, unsupervised in EVERY view")

    # (3) thin structures: rank held-out crops by high-frequency energy in GT.
    sharp = []
    for nm in test_names:
        img = cv2.imread(data["image_paths"][by_name[nm]], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        sharp.append((float(cv2.Laplacian(img, cv2.CV_64F).var()), nm))
    sharp.sort(reverse=True)
    for _, nm in sharp[:args.n_closeups]:
        rr, g, m = render_view(splats, data, by_name[nm], device, args.sh_degree,
                               args.rasterize_mode)
        p = out / f"closeup_thin_{nm.replace('.jpg', '')}.jpg"
        cv2.imwrite(str(p), strip(rr, g, m, f"{nm} (highest-detail held-out crop)", 1.0),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        written[p.name] = f"thin-structure check: {nm} (Laplacian-var rank 1..N of held-out)"

    (out / "figures.json").write_text(json.dumps(written, indent=1), encoding="utf8")
    print(f"[figures] wrote {len(written)} images", flush=True)
    return written


# ---------------------------------------------------------------------------
# 3. walk-through video
# ---------------------------------------------------------------------------

def run_video(args, splats, data, device, out: Path) -> dict:
    names = [Path(p).name for p in data["image_paths"]]
    by_name = {n: i for i, n in enumerate(names)}
    cam = f"_cam{args.video_cam:02d}.jpg"
    seq = sorted(n for n in names if n.endswith(cam))
    if not seq:
        return {}
    frames_dir = out / "_walkthrough_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    test_set = {r for r in (Path(data["image_paths"][int(i)]).name
                            for i in data["test_indices"])}
    for k, nm in enumerate(seq):
        r, _, _ = render_view(splats, data, by_name[nm], device, args.sh_degree,
                              args.rasterize_mode)
        img = to_bgr(r)
        tag = "HELD OUT" if nm in test_set else "train"
        cv2.putText(img, f"{nm}  [{tag}]", (10, img.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"{nm}  [{tag}]", (10, img.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(frames_dir / f"{k:05d}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    mp4 = out / "walkthrough.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
           "-i", str(frames_dir / "%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", str(mp4)]
    subprocess.run(cmd, check=True)
    print(f"[video] {len(seq)} frames @ {args.fps} fps -> {mp4}", flush=True)
    return {"frames": len(seq), "fps": args.fps, "camera": args.video_cam,
            "path": str(mp4), "ffmpeg": " ".join(cmd)}


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python _infusion/wpo3_eval.py",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--holdout_list", default=None,
                   help="default: <data_dir>/holdout_test.txt")
    p.add_argument("--out", required=True)
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--rasterize_mode", default="antialiased",
                   choices=["antialiased", "classic"])
    p.add_argument("--do", default="metrics,figures,video")
    p.add_argument("--tile", type=int, default=128, help="masked LPIPS tile size")
    p.add_argument("--cover", type=float, default=0.8, help="masked LPIPS tile coverage")
    p.add_argument("--n_closeups", type=int, default=4)
    p.add_argument("--fig_scale", type=float, default=0.55)
    p.add_argument("--video_cam", type=int, default=4,
                   help="virtual camera index for the walk-through (4 = equator, yaw 0)")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--progress", type=int, default=40)
    p.add_argument("--train_log", default=None, help="training log, parsed into the scorecard")
    p.add_argument("--results_json", default=None,
                   help="trainer stats/results.json (default: <ckpt>/../../stats/results.json)")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    holdout = args.holdout_list or str(Path(args.data_dir) / "holdout_test.txt")

    splats, step = load_splats(Path(args.ckpt), device)
    data = load_colmap_data(args.data_dir, factor=args.factor, normalize=True,
                            holdout_list=holdout)
    print(f"[eval] ckpt step {step}, {splats['means'].shape[0]} gaussians, "
          f"{len(data['test_indices'])} held-out views", flush=True)

    todo = {s.strip() for s in args.do.split(",") if s.strip()}
    metrics = None
    if "metrics" in todo:
        metrics = run_metrics(args, splats, data, device, out)
    elif (out / "metrics.json").is_file():
        metrics = json.loads((out / "metrics.json").read_text())
    figures = run_figures(args, splats, data, device, out, metrics) \
        if ("figures" in todo and metrics) else {}
    video = run_video(args, splats, data, device, out) if "video" in todo else {}

    train_cfg = None
    if args.train_log and Path(args.train_log).is_file():
        lines = Path(args.train_log).read_text(errors="replace").splitlines()
        train_cfg = {"log": args.train_log,
                     "head": lines[:20],
                     "tail": lines[-15:],
                     "num_lines": len(lines)}
    # the trainer's own stats/results.json, next to the checkpoint's result_dir
    results_json = (Path(args.results_json) if args.results_json
                    else Path(args.ckpt).resolve().parent.parent / "stats" / "results.json")
    scorecard = {
        "wp": "WP-O3",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": [sys.executable, __file__] + list(sys.argv[1:]),
        "ckpt": os.path.abspath(args.ckpt),
        "ckpt_step": step,
        "num_gaussians": int(splats["means"].shape[0]),
        "data_dir": os.path.abspath(args.data_dir),
        "holdout_list": os.path.abspath(holdout),
        "factor": args.factor,
        "rasterize_mode": args.rasterize_mode,
        "masked_lpips_tile": args.tile,
        "masked_lpips_cover": args.cover,
        "heldout": {"num_views": metrics["num_views"], "num_panos": metrics["num_panos"],
                    **{k: v for k, v in metrics["summary"].items()}} if metrics else None,
        "trainer_results_json": json.loads(results_json.read_text())
        if results_json.is_file() else None,
        "figures": figures,
        "video": video,
        "train_log": train_cfg,
    }
    (out / "scorecard.json").write_text(json.dumps(scorecard, indent=1), encoding="utf8")
    print(f"[eval] scorecard -> {out / 'scorecard.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
