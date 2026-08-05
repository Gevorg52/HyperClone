"""WP-O5 addendum: a "where are the good spaces" guide for the Postshot user.

The user's second complaint was navigational — "maybe I'm just not finding the right spaces".
A 360 walkthrough splat is only well-conditioned near where the operator actually stood, so
this renders a single sheet: a metric top-down floor plan of the capture path with numbered
stops, beside the view you get standing at each stop.

Stops are chosen by SURROUNDING VIEW DENSITY (how many capture stations are within
`--density_radius` metres), greedily, with a minimum separation so they spread over the
building — those are exactly the places the reconstruction is best supported. The look
direction at each stop is the bearing with the most reconstructed geometry in front of it.

Consumes the ALIGNED model + its transform JSON (`tools/align_ply.py`), so every number on
the plan is in metres under that file's stated scale assumption.

  python _infusion/wpo5_guide.py --ply <aligned.ply> --transform _infusion/wpo5/align_armD.json \
      --data_dir _dataset/360/train/wpo5_theater_027_8k --out _infusion/wpo5/eval/GUIDE.jpg
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.training.gsplat_trainer import load_colmap_data, render_splats  # noqa: E402
from tools.align_ply import AXES, AXIS_INDEX, distinct_centres, transform_camera  # noqa: E402
from tools.prune_ply import read_gsplat_ply  # noqa: E402


def pick_stops(stations: np.ndarray, horiz: list, n: int, radius: float,
               min_sep: float) -> list:
    """Greedy: densest neighbourhood first, then exclude everything within `min_sep`."""
    xy = stations[:, horiz]
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    density = (d <= radius).sum(axis=1)
    order = np.argsort(-density)
    chosen, used = [], np.zeros(len(xy), dtype=bool)
    for i in order:
        if used[i]:
            continue
        chosen.append(int(i))
        used |= d[i] <= min_sep
        if len(chosen) >= n:
            break
    return sorted(chosen), density


def best_bearing(station: np.ndarray, points: np.ndarray, horiz: list, up_i: int,
                 up_sign: float, radius: float, bins: int = 72,
                 near_m: float = 1.8, far_m: float = 3.0) -> np.ndarray:
    """The most WORTH LOOKING AT horizontal direction: open, with mid-distance geometry.

    The obvious objective — "most reconstructed points in front" — is wrong and was tried
    first: the densest direction is always the nearest WALL, which renders as a blurry
    close-up and tells the user nothing. Instead score each bearing by the geometry sitting
    at a useful distance (`far_m`..`radius`) and penalise anything inside `near_m`, so the
    chosen view looks down a corridor or into a room.
    """
    rel = points - station
    h = rel[:, up_i] * up_sign
    flat = rel[:, horiz]
    dist = np.linalg.norm(flat, axis=1)
    band = np.abs(h) < 2.5                       # ignore floor/ceiling clutter
    ang = np.arctan2(flat[:, 1], flat[:, 0])

    def hist_of(mask):
        hh, edges = np.histogram(ang[mask], bins=bins, range=(-np.pi, np.pi))
        # circular smoothing so a bearing is credited for its neighbourhood
        hh = np.convolve(np.r_[hh, hh, hh], np.ones(9) / 9, mode="same")[bins:2 * bins]
        return hh.astype(np.float64), edges

    useful, edges = hist_of(band & (dist >= far_m) & (dist < radius))
    close, _ = hist_of(band & (dist < near_m))
    if useful.sum() < 1.0:
        return np.array([1.0, 0.0])
    score = useful / max(useful.max(), 1e-9) - 2.0 * close / max(close.max(), 1e-9)
    k = int(np.argmax(score))
    a = 0.5 * (edges[k] + edges[k + 1])
    return np.array([np.cos(a), np.sin(a)])


def build_c2w(pos: np.ndarray, bearing: np.ndarray, horiz: list, up_i: int,
              up_sign: float) -> np.ndarray:
    """Level camera at `pos` looking along `bearing` (OpenCV axes: X right, Y down, Z fwd)."""
    up = np.zeros(3)
    up[up_i] = up_sign
    fwd = np.zeros(3)
    fwd[horiz[0]], fwd[horiz[1]] = bearing[0], bearing[1]
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, fwd, pos
    return c2w


def floor_plan(stations: np.ndarray, stops: list, bearings: dict, horiz: list,
               size: int, margin: int = 70) -> np.ndarray:
    xy = stations[:, horiz]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scale = (size - 2 * margin) / span.max()
    off = (np.array([size, size]) - span * scale) / 2.0

    def to_px(p):
        q = (p - lo) * scale + off
        return int(round(q[0])), int(round(size - q[1]))

    img = np.full((size, size, 3), 22, np.uint8)
    # 1 m grid
    g0 = np.floor(lo)
    for gx in np.arange(g0[0], hi[0] + 1, 1.0):
        p = to_px(np.array([gx, lo[1]]))
        q = to_px(np.array([gx, hi[1]]))
        cv2.line(img, p, q, (45, 45, 45), 1, cv2.LINE_AA)
    for gy in np.arange(g0[1], hi[1] + 1, 1.0):
        p = to_px(np.array([lo[0], gy]))
        q = to_px(np.array([hi[0], gy]))
        cv2.line(img, p, q, (45, 45, 45), 1, cv2.LINE_AA)
    # walked path
    pts = np.array([to_px(p) for p in xy], np.int32)
    cv2.polylines(img, [pts], False, (120, 190, 255), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 3, (90, 150, 220), -1, cv2.LINE_AA)
    # stops
    for n, idx in enumerate(stops, start=1):
        c = to_px(xy[idx])
        b = bearings[idx]
        tip = to_px(xy[idx] + b * 1.6)
        cv2.arrowedLine(img, c, tip, (90, 255, 170), 3, cv2.LINE_AA, tipLength=0.32)
        cv2.circle(img, c, 17, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, c, 17, (90, 255, 170), 2, cv2.LINE_AA)
        cv2.putText(img, str(n), (c[0] - (7 if n < 10 else 13), c[1] + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 255, 170), 2, cv2.LINE_AA)
    # scale bar
    bar = to_px(np.array([lo[0], lo[1]]))[0], size - 28
    px_per_m = scale
    cv2.line(img, (bar[0], bar[1]), (int(bar[0] + 5 * px_per_m), bar[1]),
             (235, 235, 235), 3, cv2.LINE_AA)
    cv2.putText(img, "5 m", (bar[0] + 6, bar[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(img, "capture path (top-down, metres) - numbered = best-supported stops",
                (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return img


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python _infusion/wpo5_guide.py")
    p.add_argument("--ply", required=True, help="the ALIGNED ply")
    p.add_argument("--transform", required=True, help="align_ply transform JSON")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stops", type=int, default=8)
    p.add_argument("--density_radius", type=float, default=2.5, help="metres")
    p.add_argument("--min_sep", type=float, default=2.5, help="metres between stops")
    p.add_argument("--bearing_radius", type=float, default=9.0, help="metres")
    p.add_argument("--tile", type=int, default=560)
    p.add_argument("--fov_deg", type=float, default=75.0)
    p.add_argument("--eye_height_m", type=float, default=None,
                   help="default: the capture's own median camera height above the floor, "
                        "which is where the splat is actually supported")
    p.add_argument("--factor", type=int, default=1)
    p.add_argument("--sh_degree", type=int, default=3)
    args = p.parse_args(argv)

    T = json.loads(Path(args.transform).read_text(encoding="utf8"))
    up_axis = T["up_axis"]
    up_i = AXIS_INDEX[up_axis]
    up_sign = float(AXES[up_axis][up_i])
    horiz = [i for i in range(3) if i != up_i]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_colmap_data(args.data_dir, factor=args.factor, normalize=True,
                            holdout_list=str(Path(args.data_dir) / "holdout_test.txt"))
    Tm = {"R": np.asarray(T["rotation_matrix"]),
          "centre": np.asarray(T["centre_source_frame"]),
          "scale": float(T["scale_metres_per_unit"])}
    c2w = np.asarray(data["camtoworlds"], dtype=np.float64)
    stations = distinct_centres(np.stack([transform_camera(m, Tm) for m in c2w])[:, :3, 3])
    pts = (np.asarray(data["points"], dtype=np.float64) - Tm["centre"]) @ Tm["R"].T \
        * Tm["scale"]

    eye = args.eye_height_m if args.eye_height_m is not None \
        else float(T["camera_height_above_floor_m"]["median"])
    stops, density = pick_stops(stations, horiz, args.stops, args.density_radius,
                                args.min_sep)
    bearings = {i: best_bearing(stations[i], pts, horiz, up_i, up_sign,
                                args.bearing_radius) for i in stops}
    print(f"[guide] {len(stations)} stations, {len(stops)} stops, eye height {eye:.2f} m, "
          f"up {up_axis}", flush=True)

    splats = {k: v.to(device) for k, v in read_gsplat_ply(Path(args.ply)).items()}
    tile = args.tile
    f = tile / (2.0 * np.tan(np.deg2rad(args.fov_deg) / 2.0))
    K = torch.tensor([[f, 0, tile / 2], [0, f, tile / 2], [0, 0, 1.0]],
                     dtype=torch.float32, device=device)
    bg = torch.zeros(3, device=device)

    renders, rows = [], []
    for n, idx in enumerate(stops, start=1):
        # the floor plane is at 0 along the up axis by construction (align_ply recentres
        # onto it), so "eye height" is simply that coordinate
        pos = stations[idx].copy()
        pos[up_i] = up_sign * eye
        pose = build_c2w(pos, bearings[idx], horiz, up_i, up_sign)
        vm = torch.linalg.inv(torch.tensor(pose, dtype=torch.float32, device=device))
        with torch.no_grad():
            out, _, _ = render_splats(splats, vm, K, tile, tile, args.sh_degree, bg,
                                      "antialiased")
        img = cv2.cvtColor((out[0][..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                           cv2.COLOR_RGB2BGR)
        cv2.rectangle(img, (0, 0), (tile, 30), (0, 0, 0), -1)
        xy = stations[idx][horiz]
        cv2.putText(img, f"{n}   x={xy[0]:+.1f} m  y={xy[1]:+.1f} m   "
                         f"({int(density[idx])} nearby stations)",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (90, 255, 170), 1, cv2.LINE_AA)
        renders.append(img)
        rows.append({"stop": n, "station_index": int(idx),
                     "position_m": stations[idx].tolist(),
                     "plan_xy_m": [float(xy[0]), float(xy[1])],
                     "bearing": bearings[idx].tolist(),
                     "nearby_stations": int(density[idx])})

    cols = 4 if len(renders) > 4 else max(len(renders), 1)
    grid_rows = []
    for i in range(0, len(renders), cols):
        row = renders[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(renders[0]))
        grid_rows.append(np.concatenate(row, axis=1))
    grid = np.concatenate(grid_rows, axis=0)

    plan = floor_plan(stations, stops, bearings, horiz, size=grid.shape[0])
    sheet = np.concatenate([plan, grid], axis=1)
    banner = np.full((66, sheet.shape[1], 3), 16, np.uint8)
    cv2.putText(banner, "WP-O5 - where the splat holds up: stand at a numbered stop, "
                        f"look along the arrow (eye height {eye:.1f} m)",
                (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(banner, "quality degrades beyond ~1.5 m from the walked path; "
                        "far from it the geometry is not reconstructed at all",
                (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 170, 190), 1, cv2.LINE_AA)
    sheet = np.concatenate([banner, sheet], axis=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    meta = out.with_suffix(".json")
    meta.write_text(json.dumps(
        {"wp": "WP-O5 addendum", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "argv": [sys.executable, __file__] + list(sys.argv[1:]),
         "ply": str(Path(args.ply).resolve()), "transform": str(Path(args.transform).resolve()),
         "up_axis": up_axis, "eye_height_m": eye, "num_stations": int(len(stations)),
         "density_radius_m": args.density_radius, "min_separation_m": args.min_sep,
         "fov_deg": args.fov_deg, "stops": rows}, indent=1), encoding="utf8")
    print(f"[guide] {out} ({sheet.shape[1]}x{sheet.shape[0]}) + {meta}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
