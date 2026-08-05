"""WP-O5 gate evidence: free-viewpoint quality, not just interpolative held-out metrics.

WP-O3's held-out numbers (masked LPIPS 0.5222) were an INTERPOLATION split — every 8th
panorama withheld while its ~0.4-unit neighbours stayed in training — and the user's Postshot
verdict on that splat was "hot mess, only lines up from key angles". So this script keeps the
WP-O3 metrics for comparability and adds the two things that actually predict free-viewpoint
usability:

  offpath  virtual cameras TRANSLATED off the capture path (lateral +-0.5/+-1 m, vertical
           +0.5/-0.3 m) at several trajectory points, plus "unvisited" viewpoints placed at
           the position in the scene FURTHEST from any capture centre. No ground truth exists
           for these, so they are visual artifacts — rendered at MATCHED viewpoints for every
           arm and laid out as per-viewpoint comparison sheets so the arms can be judged
           against each other rather than against a number.
  fog      per-checkpoint opacity histogram / fraction below 0.05 / spatial-outlier counts
           (tools/prune_ply.splat_stats), plus a with-pruning vs without-pruning render pair
           at an off-path viewpoint, which is what makes the fog contribution visible.

Metric scale: the SfM/trainer world is normalised (camera cloud radius 1), so metres are
recovered by an explicitly recorded anchor — see `estimate_metres_per_unit`. Two independent
estimates are computed and BOTH are written to the scorecard; the offsets use the ceiling one.

  python _infusion/wpo5_eval.py --data_dir <packaged> --out _infusion/wpo5/eval \
      --arm A=<ckptA.pt> --arm B=<ckptB.pt> --arm C=<ckptC.pt> --do metrics,offpath,fog,video
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

from core.training.gsplat_trainer import (  # noqa: E402
    load_colmap_data,
    load_image,
    render_splats,
)
from tools.prune_ply import splat_stats  # noqa: E402
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("wpo3_eval", PROJECT_ROOT / "_infusion" / "wpo3_eval.py")
wpo3_eval = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(wpo3_eval)

# WP-O2 crop indices 4-7 are the equator ring: yaw 0/90/180/270.
# Index 4 (yaw 0) IS the panorama's own frame (cam_from_pano = identity), which makes it the
# natural reference — but on the `_027` capture yaw 0 points straight at the OPERATOR, whose
# pixels are masked out of every view and are therefore unsupervised everywhere. Rendering
# off-path sheets down that axis measures the one direction the splat was never taught, so
# `--frame_cam` exists and the 8K run uses 6 (yaw 180 = the clean, scene-facing side).
# The choice only rotates the LOOK direction: `up` is a yaw-invariant axis and the off-path
# offsets are built from travel x up, so the camera POSITIONS are identical either way.
EQUATOR_CAM = 4


# ---------------------------------------------------------------------------
# trajectory frame
# ---------------------------------------------------------------------------

def pano_frames(data: dict, cam: int = EQUATOR_CAM) -> dict:
    """Per-panorama world pose taken from its equator crop (cam04 == the pano frame).

    The 12 crops of a panorama share an optical centre by construction (WP-O2 rig), and
    cam04 has cam_from_pano = identity, so its c2w IS the panorama's world pose.
    """
    names = [Path(p).name for p in data["image_paths"]]
    idx_of = {n: i for i, n in enumerate(names)}
    suffix = f"_cam{cam:02d}.jpg"
    panos = sorted(n[: -len(suffix)] for n in names if n.endswith(suffix))
    if not panos:
        raise RuntimeError(f"no crops ending in {suffix}")
    c2w = np.stack([data["camtoworlds"][idx_of[p + suffix]] for p in panos]).astype(np.float64)
    centres = c2w[:, :3, 3]
    # OpenCV camera axes in world: X right, Y down, Z forward.
    up = -c2w[:, :3, 1].mean(axis=0)
    up /= max(np.linalg.norm(up), 1e-9)
    return {"panos": panos, "c2w": c2w, "centres": centres, "up": up,
            "indices": [idx_of[p + suffix] for p in panos]}


def travel_dirs(centres: np.ndarray) -> np.ndarray:
    d = np.zeros_like(centres)
    d[1:-1] = centres[2:] - centres[:-2]
    d[0] = centres[1] - centres[0]
    d[-1] = centres[-1] - centres[-2]
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(n, 1e-9)


def estimate_metres_per_unit(data: dict, frames: dict, pano_fps: float,
                             ceiling_m: float, walk_mps: float) -> dict:
    """Scale the normalised trainer world so off-path offsets can be named in metres.

    HONEST STATUS (measured on WP-O3's `_077`, recorded in `_infusion/wpo5_RECORD.md`):
    there is no clean automatic metric anchor for these captures, so the primary scale here
    is deliberately CAPTURE-RELATIVE and the metre labels are indicative only.

      BASELINE (used): the median inter-panorama step IS the capture's own baseline, and it
        is directly measured. Calling it `walk_mps / pano_fps` metres turns every offset into
        a stated multiple of that baseline: "lateral 0.5 m" means "0.83 x one inter-pano
        step sideways" whatever the true metre value is. That relation holds exactly.
      CEILING (diagnostic only): p1-p99 of the SfM points along world up. On `_077` it
        disagreed with BASELINE by 4.5x, and both are individually implausible — under
        BASELINE the camera height varies by 4.7 m over the walk, under CEILING the whole
        151-panorama walk fits in a 3.9 x 1.8 m room. The likely cause is the world-up
        estimate: WP-O2 measured zenith tilt median 11 deg / max 49 deg on these panos, so
        "vertical" leaks horizontal extent, and long sight-lines through doorways inflate
        the point spread further.

    Everything above is written into the scorecard so a reader can re-judge the metre labels.
    """
    pts = np.asarray(data["points"], dtype=np.float64)
    h = pts @ frames["up"]
    vertical_units = float(np.ptp(np.percentile(h, [1, 99])))
    steps = np.linalg.norm(np.diff(frames["centres"], axis=0), axis=1)
    step_median = float(np.median(steps))
    cam_height_spread = float(np.ptp(np.percentile(frames["centres"] @ frames["up"],
                                                   [1, 99])))
    mpu_ceiling = ceiling_m / vertical_units if vertical_units > 1e-9 else float("nan")
    mpu_baseline = (walk_mps / pano_fps) / step_median if step_median > 1e-9 else float("nan")
    return {"used": "baseline",
            "metres_per_unit": round(mpu_baseline, 6),
            "step_median_units": round(step_median, 6),
            "assumed_walk_mps": walk_mps, "pano_fps": pano_fps,
            "baseline_step_m": round(walk_mps / pano_fps, 4),
            "diagnostic_ceiling": {
                "assumed_ceiling_m": ceiling_m,
                "points_vertical_extent_units_p1_p99": round(vertical_units, 6),
                "metres_per_unit_ceiling": round(mpu_ceiling, 6),
                "ratio_baseline_over_ceiling": round(mpu_baseline / mpu_ceiling, 4)
                if mpu_ceiling == mpu_ceiling and mpu_ceiling > 0 else None},
            "diagnostic_camera_height_spread_units_p1_p99": round(cam_height_spread, 6),
            "diagnostic_camera_height_spread_in_steps":
                round(cam_height_spread / step_median, 3) if step_median > 1e-9 else None,
            "caveat": "metre labels are indicative; the exact relation is offset_m / "
                      "baseline_step_m = offset in inter-panorama steps"}


def unvisited_point(frames: dict, grid: int = 48) -> dict:
    """The position INSIDE the walked region that is furthest from every capture centre.

    Sampled on a grid in the plane spanned by the two non-up principal directions of the
    track, at the median capture height, and restricted to the 2D convex hull of the
    projected capture positions. The hull restriction matters: without it the winner is
    always a corner of the bounding box, which is usually outside the building, and a garbage
    render there says nothing about splat quality. Inside the hull it is a spot the operator
    provably walked around but never stood on — exactly what a free-viewpoint viewer wanders
    into.
    """
    from scipy.spatial import Delaunay

    c, up = frames["centres"], frames["up"]
    centred = c - c.mean(axis=0)
    horiz = centred - np.outer(centred @ up, up)
    _, _, vt = np.linalg.svd(horiz, full_matrices=False)
    e0, e1 = vt[0], vt[1]
    a, b = horiz @ e0, horiz @ e1
    ga = np.linspace(a.min(), a.max(), grid)
    gb = np.linspace(b.min(), b.max(), grid)
    A, B = np.meshgrid(ga, gb, indexing="ij")
    uv = np.column_stack([A.ravel(), B.ravel()])
    inside = np.ones(uv.shape[0], dtype=bool)
    try:
        inside = Delaunay(np.column_stack([a, b])).find_simplex(uv) >= 0
    except Exception as exc:                       # degenerate (collinear) track
        print(f"[offpath] convex-hull restriction unavailable ({exc}); "
              f"falling back to the bbox", flush=True)
    if not inside.any():
        inside = np.ones(uv.shape[0], dtype=bool)
    cand = (c.mean(axis=0) + np.outer(uv[:, 0], e0) + np.outer(uv[:, 1], e1)
            + np.outer(np.full(uv.shape[0], float(np.median(centred @ up))), up))
    d = np.linalg.norm(cand[:, None, :] - c[None, :, :], axis=2).min(axis=1)
    d = np.where(inside, d, -np.inf)
    k = int(np.argmax(d))
    return {"position": cand[k], "min_dist_to_capture_units": float(d[k]),
            "inside_hull": bool(inside[k]), "hull_restricted": bool(inside.sum() < inside.size),
            "axes": (e0, e1)}


# ---------------------------------------------------------------------------
# virtual cameras
# ---------------------------------------------------------------------------

def build_offpath_views(data: dict, frames: dict, mpu: float, args) -> list:
    """Matched virtual cameras: pure translations off the path + unvisited look-arounds."""
    centres, up = frames["centres"], frames["up"]
    travel = travel_dirs(centres)
    step = float(np.median(np.linalg.norm(np.diff(centres, axis=0), axis=1)))

    def tag(dm: float) -> str:
        return f"{dm:+.2f} m ({dm / mpu / max(step, 1e-9):+.2f} steps)"

    n = len(frames["panos"])
    picks = np.unique(np.linspace(0, n - 1, args.offpath_points).astype(int))
    views = []
    for p in picks:
        c2w = frames["c2w"][p].copy()
        fwd = travel[p]
        lateral = np.cross(fwd, up)
        ln = np.linalg.norm(lateral)
        lateral = lateral / ln if ln > 1e-9 else np.cross(c2w[:3, 2], up)
        base = frames["panos"][p]
        views.append({"group": base, "label": "on-path (reference)", "kind": "onpath",
                      "c2w": c2w, "offset_m": [0.0, 0.0]})
        for dm in args.lateral_m:
            w = c2w.copy()
            w[:3, 3] = c2w[:3, 3] + lateral * (dm / mpu)
            views.append({"group": base, "label": f"lateral {tag(dm)}", "kind": "lateral",
                          "c2w": w, "offset_m": [float(dm), 0.0]})
        for dm in args.vertical_m:
            w = c2w.copy()
            w[:3, 3] = c2w[:3, 3] + up * (dm / mpu)
            views.append({"group": base, "label": f"vertical {tag(dm)}", "kind": "vertical",
                          "c2w": w, "offset_m": [0.0, float(dm)]})

    # unvisited viewpoints: stand where nobody stood, look around
    uv = unvisited_point(frames)
    e0, _ = uv["axes"]
    for k in range(args.unvisited_views):
        ang = 2 * np.pi * k / max(args.unvisited_views, 1)
        fwd = np.cos(ang) * e0 + np.sin(ang) * np.cross(up, e0)
        fwd /= max(np.linalg.norm(fwd), 1e-9)
        right = np.cross(fwd, up)
        right /= max(np.linalg.norm(right), 1e-9)
        down = np.cross(fwd, right)
        w = np.eye(4)
        w[:3, 0], w[:3, 1], w[:3, 2] = right, down, fwd
        w[:3, 3] = uv["position"]
        views.append({"group": "unvisited",
                      "label": f"unvisited az {np.degrees(ang):.0f} deg "
                               f"({uv['min_dist_to_capture_units'] * mpu:.2f} m from any "
                               f"capture position)",
                      "kind": "unvisited", "c2w": w, "offset_m": None})
    return views, uv


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def load_splats(ckpt_path: Path, device, opacity_min: float = 0.0):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    splats = {k: v.float() for k, v in ckpt["splats"].items()}
    kept = splats["means"].shape[0]
    if opacity_min > 0:
        keep = torch.sigmoid(splats["opacities"]).reshape(-1) >= opacity_min
        splats = {k: v[keep] for k, v in splats.items()}
        kept = int(keep.sum())
    return ({k: v.to(device) for k, v in splats.items()},
            int(ckpt.get("step", -1)), kept)


def render_at(splats, c2w: np.ndarray, K: np.ndarray, W: int, H: int, device,
              sh_degree=3, rasterize_mode="antialiased") -> np.ndarray:
    c2w_t = torch.from_numpy(np.ascontiguousarray(c2w, dtype=np.float32)).to(device)
    K_t = torch.from_numpy(np.ascontiguousarray(K, dtype=np.float32)).to(device)
    with torch.no_grad():
        renders, _, _ = render_splats(splats, torch.linalg.inv(c2w_t), K_t, W, H,
                                      sh_degree, torch.zeros(3, device=device),
                                      rasterize_mode)
    img = renders[0][..., :3].clamp(0, 1).cpu().numpy()
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def label(img, text, sub=None):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30 if sub is None else 50), (0, 0, 0), -1)
    cv2.putText(out, text, (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (7, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 220, 255), 1,
                    cv2.LINE_AA)
    return out


def sheet(tiles, cols: int, tile_px: int) -> np.ndarray:
    if not tiles:
        return np.zeros((10, 10, 3), np.uint8)
    scaled = [cv2.resize(t, (tile_px, int(round(tile_px * t.shape[0] / t.shape[1]))),
                         interpolation=cv2.INTER_AREA) for t in tiles]
    h = max(t.shape[0] for t in scaled)
    rows = []
    for i in range(0, len(scaled), cols):
        row = scaled[i:i + cols]
        row = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in row]
        while len(row) < cols:
            row.append(np.zeros((h, tile_px, 3), np.uint8))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def load_existing_metrics(arms, out: Path) -> dict:
    """Per-arm metrics.json already on disk (so a re-run for figures alone keeps the numbers)."""
    res = {}
    for name, _ in arms:
        p = out / name / "metrics.json"
        if p.is_file():
            res[name] = json.loads(p.read_text(encoding="utf8"))["summary"]
    return res


def stage_metrics(args, arms, data, device, out: Path) -> dict:
    res = {}
    for name, ckpt in arms:
        adir = out / name
        adir.mkdir(parents=True, exist_ok=True)
        if args.skip_existing_metrics and (adir / "metrics.json").is_file():
            res[name] = json.loads((adir / "metrics.json").read_text(
                encoding="utf8"))["summary"]
            print(f"[metrics] arm {name}: reusing {adir / 'metrics.json'}", flush=True)
            continue
        splats, step, _ = load_splats(Path(ckpt), device)
        print(f"\n[metrics] arm {name}: step {step}, {splats['means'].shape[0]} gaussians",
              flush=True)
        ns = argparse.Namespace(sh_degree=args.sh_degree, rasterize_mode=args.rasterize_mode,
                                tile=args.tile, cover=args.cover, progress=args.progress)
        res[name] = wpo3_eval.run_metrics(ns, splats, data, device, adir)["summary"]
        del splats
        torch.cuda.empty_cache()
    return res


def stage_offpath(args, arms, data, frames, mpu: float, device, out: Path) -> dict:
    views, uv = build_offpath_views(data, frames, mpu, args)
    K = data["Ks_dict"][data["camera_ids"][frames["indices"][0]]]
    img0 = load_image(data, frames["indices"][0])["image"]
    H, W = int(img0.shape[0]), int(img0.shape[1])
    odir = out / "offpath"
    odir.mkdir(parents=True, exist_ok=True)

    renders = {}
    for name, ckpt in arms:
        splats, step, _ = load_splats(Path(ckpt), device)
        t0 = time.time()
        renders[name] = [render_at(splats, v["c2w"], K, W, H, device, args.sh_degree,
                                   args.rasterize_mode) for v in views]
        print(f"[offpath] arm {name}: {len(views)} views in {time.time() - t0:.0f}s",
              flush=True)
        del splats
        torch.cuda.empty_cache()

    written = {}
    # (a) per-arm sheet: every viewpoint, one arm
    for name, _ in arms:
        tiles = [label(renders[name][i], f"{v['group']}  {v['label']}", f"arm {name}")
                 for i, v in enumerate(views)]
        p = odir / f"sheet_arm_{name}.jpg"
        cv2.imwrite(str(p), sheet(tiles, args.cols, args.tile_px),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        written[p.name] = f"arm {name}: all {len(views)} off-path viewpoints"
    # (b) per-viewpoint sheet: every arm, one viewpoint — the decisive comparison
    for i, v in enumerate(views):
        tiles = [label(renders[name][i], f"arm {name}", f"{v['group']}  {v['label']}")
                 for name, _ in arms]
        tag = f"{v['group']}_{v['kind']}_{i:02d}"
        p = odir / f"cmp_{tag}.jpg"
        cv2.imwrite(str(p), sheet(tiles, len(arms), args.tile_px),
                    [cv2.IMWRITE_JPEG_QUALITY, 94])
        written[p.name] = f"{v['group']} / {v['label']}: arms " + \
                          ", ".join(n for n, _ in arms)
    print(f"[offpath] wrote {len(written)} sheets -> {odir}", flush=True)
    return {"views": [{"group": v["group"], "label": v["label"], "kind": v["kind"],
                       "offset_m": v["offset_m"],
                       "position": v["c2w"][:3, 3].tolist()} for v in views],
            "unvisited": {"position": uv["position"].tolist(),
                          "min_dist_to_capture_units": uv["min_dist_to_capture_units"],
                          "min_dist_to_capture_m": uv["min_dist_to_capture_units"] * mpu,
                          "inside_capture_hull": uv["inside_hull"],
                          "hull_restriction_applied": uv["hull_restricted"]},
            "num_views": len(views), "sheets": written, "image_size": [W, H]}


def stage_fog(args, arms, data, frames, mpu: float, device, out: Path) -> dict:
    fdir = out / "fog"
    fdir.mkdir(parents=True, exist_ok=True)
    views, _ = build_offpath_views(data, frames, mpu, args)
    # the most-displaced lateral view of the middle trajectory point: fog shows up worst there
    lat = [v for v in views if v["kind"] == "lateral"]
    pick = [lat[len(lat) // 2]] if lat else [views[0]]
    pick += [v for v in views if v["kind"] == "unvisited"][:1]
    K = data["Ks_dict"][data["camera_ids"][frames["indices"][0]]]
    img0 = load_image(data, frames["indices"][0])["image"]
    H, W = int(img0.shape[0]), int(img0.shape[1])

    stats, written = {}, {}
    for name, ckpt in arms:
        raw = torch.load(str(ckpt), map_location="cpu", weights_only=True)["splats"]
        n = raw["means"].shape[0]
        st = splat_stats({"means": raw["means"].float(), "scales": raw["scales"].float(),
                          "quats": raw["quats"].float(),
                          "opacities": raw["opacities"].float().reshape(n),
                          "sh0": raw["sh0"].float().reshape(n, 1, 3),
                          "shN": raw["shN"].float().reshape(n, -1, 3)})
        stats[name] = st
        del raw
        print(f"[fog] arm {name}: {st['num_gaussians']} gaussians, mean opacity "
              f"{st['opacity_mean']:.4f}, {st['opacity_frac_below']['0.05'] * 100:.1f}% "
              f"< 0.05, {st['outliers_beyond_3x_bbox']} beyond 3x bbox", flush=True)

        full, _, n_full = load_splats(Path(ckpt), device, 0.0)
        pruned, _, n_pruned = load_splats(Path(ckpt), device, args.prune_opacity)
        for j, v in enumerate(pick):
            a = label(render_at(full, v["c2w"], K, W, H, device, args.sh_degree,
                                args.rasterize_mode),
                      f"arm {name} — ALL {n_full} gaussians", v["label"])
            b = label(render_at(pruned, v["c2w"], K, W, H, device, args.sh_degree,
                                args.rasterize_mode),
                      f"arm {name} — opacity >= {args.prune_opacity} ({n_pruned})",
                      v["label"])
            p = fdir / f"fogpair_{name}_{v['kind']}_{j}.jpg"
            cv2.imwrite(str(p), sheet([a, b], 2, args.tile_px),
                        [cv2.IMWRITE_JPEG_QUALITY, 94])
            written[p.name] = (f"arm {name}, {v['label']}: full render vs low-opacity-pruned "
                               f"(<{args.prune_opacity} dropped, "
                               f"{n_full - n_pruned} gaussians)")
        del full, pruned
        torch.cuda.empty_cache()
    (fdir / "ckpt_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf8")
    return {"stats": stats, "figures": written, "prune_opacity": args.prune_opacity}


def stage_video(args, arms, data, frames, mpu: float, device, out: Path) -> dict:
    """Walk-through at the SfM poses + the honest off-path walk-through, for one arm."""
    name, ckpt = arms[0] if len(arms) == 1 else \
        next(((n, c) for n, c in arms if n == args.video_arm), arms[0])
    splats, _, _ = load_splats(Path(ckpt), device)
    K = data["Ks_dict"][data["camera_ids"][frames["indices"][0]]]
    img0 = load_image(data, frames["indices"][0])["image"]
    H, W = int(img0.shape[0]), int(img0.shape[1])
    up, centres = frames["up"], frames["centres"]
    travel = travel_dirs(centres)
    vids = {}
    for tag, dm in (("walkthrough", 0.0), ("offpath_lateral", args.video_offset_m)):
        fdir = out / f"_frames_{tag}"
        fdir.mkdir(parents=True, exist_ok=True)
        for k in range(len(frames["panos"])):
            c2w = frames["c2w"][k].copy()
            if dm:
                lateral = np.cross(travel[k], up)
                ln = np.linalg.norm(lateral)
                if ln > 1e-9:
                    c2w[:3, 3] = c2w[:3, 3] + (lateral / ln) * (dm / mpu)
            img = render_at(splats, c2w, K, W, H, device, args.sh_degree,
                            args.rasterize_mode)
            txt = f"{frames['panos'][k]}  arm {name}" + (f"  lateral {dm:+.2f} m" if dm else "")
            cv2.putText(img, txt, (10, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (10, img.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(fdir / f"{k:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        mp4 = out / f"{tag}_{name}.mp4"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
               "-i", str(fdir / "%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-crf", "18", str(mp4)]
        subprocess.run(cmd, check=True)
        vids[tag] = {"path": str(mp4), "frames": len(frames["panos"]), "fps": args.fps,
                     "arm": name, "lateral_offset_m": dm}
        print(f"[video] {tag}: {len(frames['panos'])} frames -> {mp4}", flush=True)
    del splats
    torch.cuda.empty_cache()
    return vids


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python _infusion/wpo5_eval.py",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=CKPT",
                   help="repeatable: arm label and checkpoint path")
    p.add_argument("--out", required=True)
    p.add_argument("--holdout_list", default=None)
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--rasterize_mode", default="antialiased",
                   choices=["antialiased", "classic"])
    p.add_argument("--do", default="metrics,offpath,fog,video")
    p.add_argument("--skip_existing_metrics", action="store_true",
                   help="reuse <out>/<arm>/metrics.json instead of recomputing it")
    p.add_argument("--tile", type=int, default=128, help="masked LPIPS tile size")
    p.add_argument("--cover", type=float, default=0.8)
    p.add_argument("--progress", type=int, default=40)
    p.add_argument("--offpath_points", type=int, default=4,
                   help="trajectory points at which to build the off-path fan")
    p.add_argument("--lateral_m", type=float, nargs="*",
                   default=[-1.0, -0.5, 0.5, 1.0])
    p.add_argument("--vertical_m", type=float, nargs="*", default=[0.5, -0.3])
    p.add_argument("--unvisited_views", type=int, default=3)
    p.add_argument("--cols", type=int, default=7)
    p.add_argument("--tile_px", type=int, default=560)
    p.add_argument("--prune_opacity", type=float, default=0.05)
    p.add_argument("--ceiling_m", type=float, default=3.0,
                   help="assumed floor-to-ceiling height used to scale the normalised world")
    p.add_argument("--walk_mps", type=float, default=1.2,
                   help="assumed walking speed for the cross-check scale estimate")
    p.add_argument("--pano_fps", type=float, default=2.0)
    p.add_argument("--frame_cam", type=int, default=EQUATOR_CAM,
                   help="equator crop index whose rotation defines the look direction "
                        "of every off-path view and of the walk-through (see the note at "
                        "the top of this file: on _027, 4 faces the masked operator and "
                        "6 faces the scene). Camera POSITIONS are unaffected.")
    p.add_argument("--video_arm", default=None)
    p.add_argument("--video_offset_m", type=float, default=0.75)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--note", default=None)
    args = p.parse_args(argv)

    arms = []
    for spec in args.arm:
        if "=" not in spec:
            p.error(f"--arm expects NAME=CKPT, got {spec}")
        name, ckpt = spec.split("=", 1)
        arms.append((name, ckpt))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    holdout = args.holdout_list or str(Path(args.data_dir) / "holdout_test.txt")
    data = load_colmap_data(args.data_dir, factor=args.factor, normalize=True,
                            holdout_list=holdout)
    frames = pano_frames(data, args.frame_cam)
    scale = estimate_metres_per_unit(data, frames, args.pano_fps, args.ceiling_m,
                                     args.walk_mps)
    mpu = scale["metres_per_unit"]
    print(f"[eval] {len(frames['panos'])} panos, {len(data['test_indices'])} held-out crops; "
          f"1 unit = {mpu:.3f} m (baseline anchor: one inter-pano step = "
          f"{scale['baseline_step_m']} m); ceiling diagnostic disagrees by "
          f"{scale['diagnostic_ceiling']['ratio_baseline_over_ceiling']}x — metre labels are "
          f"indicative, see scorecard['scale']", flush=True)

    todo = {s.strip() for s in args.do.split(",") if s.strip()}
    card = {"wp": "WP-O5", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "argv": [sys.executable, __file__] + list(sys.argv[1:]),
            "data_dir": os.path.abspath(args.data_dir),
            "holdout_list": os.path.abspath(holdout),
            "arms": {n: os.path.abspath(c) for n, c in arms},
            "num_panos": len(frames["panos"]),
            "num_heldout_views": len(data["test_indices"]),
            "scale": scale, "note": args.note}
    card["heldout"] = (stage_metrics(args, arms, data, device, out) if "metrics" in todo
                       else load_existing_metrics(arms, out))
    if "offpath" in todo:
        card["offpath"] = stage_offpath(args, arms, data, frames, mpu, device, out)
    if "fog" in todo:
        card["fog"] = stage_fog(args, arms, data, frames, mpu, device, out)
    if "video" in todo:
        card["video"] = stage_video(args, arms, data, frames, mpu, device, out)
    (out / "scorecard.json").write_text(json.dumps(card, indent=1), encoding="utf8")
    print(f"\n[eval] scorecard -> {out / 'scorecard.json'}", flush=True)

    if "heldout" in card:
        print("\n| arm | masked LPIPS (tiled) | masked PSNR | masked SSIM | full LPIPS |")
        print("|---|---|---|---|---|")
        for n, s in card["heldout"].items():
            g = lambda k: s[k]["mean"] if s.get(k) else float("nan")  # noqa: E731
            print(f"| {n} | {g('masked_lpips_tiled'):.4f} | {g('masked_psnr'):.3f} | "
                  f"{g('masked_ssim'):.4f} | {g('full_lpips'):.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
