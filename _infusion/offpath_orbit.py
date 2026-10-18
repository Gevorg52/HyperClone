"""Free-viewpoint (off-path) + fog evidence for OBJECT-ORBIT captures (Laptop, GPU_V4, ...).

Why: interpolative held-out LPIPS cannot see what a Postshot user sees when they fly AWAY from
the capture path (splat-quality-bar). `_infusion/wpo5_eval.py` does this for 360 walkthroughs
(pano frames, metric offsets); this is the orbit counterpart. For every arm it renders the SAME
virtual cameras, none of which coincide with a capture pose:

  dolly_in   camera moved toward the orbit's look-at centre to 0.6x its capture distance
  pull_out   moved away to 1.5x (shows background haze / fog shells around the object)
  elev_up    rotated about the centre to +25 deg ABOVE its own elevation, re-aimed at centre
  elev_down  -20 deg below, re-aimed at centre
  lateral    translated sideways by 0.25x its distance, orientation kept (pure parallax)
  gap        the midpoint of the largest azimuth gap between capture cameras (least-observed
             direction of the orbit), at median radius/elevation

No ground truth exists there, so the per-view outputs are (a) comparison sheets at matched
viewpoints (the visual judgement) and (b) GT-free *fog proxies* from the rendered alpha:
  semi_alpha_frac   fraction of pixels with 0.05 < alpha < 0.6  (translucent haze / veils)
  fog_alpha_frac    fraction of pixels with 0.005 < alpha < 0.2 (faint fog)
  coverage          fraction of pixels with alpha > 0.6
plus per-checkpoint Gaussian statistics (tools/prune_ply.splat_stats: opacity histogram,
fraction < 0.05, outliers beyond 3x bbox, needle fraction) and, per arm, a full-vs-pruned
(opacity < --prune_opacity removed) render pair at the most-displaced view.

Reproducible: no randomness; the view set depends only on the dataset's camera poses.

  python _infusion/offpath_orbit.py --data_dir <data> --factor 2 --out qa/<tag>_offpath \
      --arm A=<ckptA.pt> --arm B=<ckptB.pt>
"""

import argparse
import importlib.util as _ilu
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
from tools.prune_ply import splat_stats  # noqa: E402

_spec = _ilu.spec_from_file_location("wpo5_eval", PROJECT_ROOT / "_infusion" / "wpo5_eval.py")
_w5 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_w5)
load_splats, label, sheet = _w5.load_splats, _w5.label, _w5.sheet


def look_at_centre(c2ws: np.ndarray) -> np.ndarray:
    """Least-squares point closest to every camera's optical axis (the orbit target)."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c2w in c2ws:
        o, d = c2w[:3, 3], c2w[:3, 2] / np.linalg.norm(c2w[:3, 2])
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    return np.linalg.solve(A, b)


def look_at(pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenCV c2w (x right, y down, z forward) at `pos` looking at `target`."""
    z = target - pos; z /= np.linalg.norm(z)
    x = np.cross(z, -up); x /= np.linalg.norm(x)   # -up is image-down (y) direction
    y = np.cross(z, x)
    c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, pos
    return c2w


def rot_about(axis: np.ndarray, ang: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def build_views(data: dict, n_anchor: int) -> tuple:
    c2ws = data["camtoworlds"]
    centre = look_at_centre(c2ws)
    up = -c2ws[:, :3, 1].mean(0); up /= np.linalg.norm(up)       # mean camera up (-y)
    rel = c2ws[:, :3, 3] - centre
    # orbit-plane basis for azimuth / elevation
    e1 = rel[0] - (rel[0] @ up) * up; e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    az = np.arctan2(rel @ e2, rel @ e1)
    r = np.linalg.norm(rel, axis=1)
    elev = np.arcsin(np.clip((rel @ up) / r, -1, 1))

    # anchors evenly spaced in capture (= sorted image name) order
    anchors = [int(i) for i in np.linspace(0, len(c2ws) - 1, n_anchor + 1)[:-1]]
    views = []
    for a in anchors:
        p = c2ws[a, :3, 3]; d = p - centre; dist = np.linalg.norm(d)
        views.append(dict(kind="dolly_in", anchor=a, c2w=look_at(centre + 0.6 * d, centre, up)))
        views.append(dict(kind="pull_out", anchor=a, c2w=look_at(centre + 1.5 * d, centre, up)))
        side = np.cross(d / dist, up); side /= np.linalg.norm(side)
        for kind, deg in (("elev_up", 25.0), ("elev_down", -20.0)):
            R = rot_about(side, np.deg2rad(deg))
            q = centre + R @ d
            # rotate toward +up for positive angles (flip if the cross product points away)
            if np.sign((R @ d - d) @ up) != np.sign(deg):
                q = centre + rot_about(side, -np.deg2rad(deg)) @ d
            views.append(dict(kind=kind, anchor=a, c2w=look_at(q, centre, up)))
        lat = c2ws[a].copy(); lat[:3, 3] = p + 0.25 * dist * lat[:3, 0]
        views.append(dict(kind="lateral", anchor=a, c2w=lat))
    # largest azimuth gap
    s = np.sort(az); gaps = np.diff(np.concatenate([s, s[:1] + 2 * np.pi]))
    g = int(np.argmax(gaps)); mid = s[g] + gaps[g] / 2
    rm, em = float(np.median(r)), float(np.median(elev))
    q = centre + rm * (np.cos(em) * (np.cos(mid) * e1 + np.sin(mid) * e2) + np.sin(em) * up)
    views.append(dict(kind="gap", anchor=-1, c2w=look_at(q, centre, up)))
    geo = dict(centre=centre.tolist(), up=up.tolist(), radius_median=rm,
               elev_deg_range=[float(np.rad2deg(elev.min())), float(np.rad2deg(elev.max()))],
               largest_azimuth_gap_deg=float(np.rad2deg(gaps[g])), anchors=anchors)
    return views, geo


def render_rgba(splats, c2w, K, W, H, device, sh_degree):
    c2w_t = torch.from_numpy(np.ascontiguousarray(c2w, dtype=np.float32)).to(device)
    K_t = torch.from_numpy(np.ascontiguousarray(K, dtype=np.float32)).to(device)
    with torch.no_grad():
        rend, alph, _ = render_splats(splats, torch.linalg.inv(c2w_t), K_t, W, H, sh_degree,
                                      torch.zeros(3, device=device), "antialiased")
    rgb = (rend[0][..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), alph[0, ..., 0].cpu().numpy()


def fog_proxies(alpha: np.ndarray) -> dict:
    return {"semi_alpha_frac": float(((alpha > 0.05) & (alpha < 0.6)).mean()),
            "fog_alpha_frac": float(((alpha > 0.005) & (alpha < 0.2)).mean()),
            "coverage": float((alpha > 0.6).mean())}


def subset(splats: dict, keep: torch.Tensor) -> dict:
    return {k: v[keep] for k, v in splats.items()}


FOG_KEYS = ("semi_alpha_frac", "fog_alpha_frac", "coverage", "fog_layer_alpha_mean",
            "fog_layer_px_frac", "prune_delta_mae")


def ckpt_stats(ckpt: Path) -> dict:
    raw = torch.load(str(ckpt), map_location="cpu", weights_only=True)["splats"]
    n = raw["means"].shape[0]
    return splat_stats({"means": raw["means"].float(), "scales": raw["scales"].float(),
                        "quats": raw["quats"].float(),
                        "opacities": raw["opacities"].float().reshape(n),
                        "sh0": raw["sh0"].float().reshape(n, 1, 3),
                        "shN": raw["shN"].float().reshape(n, -1, 3)})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python _infusion/offpath_orbit.py")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--factor", type=int, default=2)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=CKPT")
    p.add_argument("--out", required=True)
    p.add_argument("--anchors", type=int, default=4)
    p.add_argument("--sh_degree", type=int, default=3)
    p.add_argument("--render_scale", type=float, default=0.5,
                   help="render at this fraction of the factor-N image size (speed)")
    p.add_argument("--prune_opacity", type=float, default=0.05)
    p.add_argument("--tile_px", type=int, default=640)
    args = p.parse_args(argv)

    arms = [tuple(a.split("=", 1)) for a in args.arm]
    out = Path(args.out); (out / "sheets").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    data = load_colmap_data(args.data_dir, factor=args.factor, test_every=8, normalize=True)
    views, geo = build_views(data, args.anchors)
    cam0 = data["camera_ids"][0]
    W0, H0 = data["imsize_dict"][cam0]
    W, H = int(round(W0 * args.render_scale)), int(round(H0 * args.render_scale))
    K = data["Ks_dict"][cam0].astype(np.float64).copy(); K[:2] *= (W / W0)

    result = {"argv": sys.argv, "geometry": geo, "image_size": [W, H],
              "views": [{"kind": v["kind"], "anchor": v["anchor"],
                         "position": v["c2w"][:3, 3].tolist()} for v in views],
              "arms": {}}
    renders = {}
    for name, ckpt in arms:
        t0 = time.time()
        splats, step, _ = load_splats(Path(ckpt), device)
        # fog layer = ONLY the low-opacity Gaussians; dense = everything else. On full
        # scenes alpha saturates (coverage ~1), so the informative fog numbers are how much
        # alpha the low-opacity layer alone deposits at each off-path view, and how much the
        # full render changes (MAE, 0-255) when that layer is deleted.
        low = torch.sigmoid(splats["opacities"]).reshape(-1) < args.prune_opacity
        fog_l, dense_l = subset(splats, low), subset(splats, ~low)
        rs, per = [], []
        for v in views:
            img, a = render_rgba(splats, v["c2w"], K, W, H, device, args.sh_degree)
            _, a_fog = render_rgba(fog_l, v["c2w"], K, W, H, device, args.sh_degree)
            img_d, _ = render_rgba(dense_l, v["c2w"], K, W, H, device, args.sh_degree)
            rs.append(img)
            per.append({"kind": v["kind"], "anchor": v["anchor"], **fog_proxies(a),
                        "fog_layer_alpha_mean": float(a_fog.mean()),
                        "fog_layer_px_frac": float((a_fog > 0.1).mean()),
                        "prune_delta_mae": float(np.abs(img.astype(np.float32)
                                                        - img_d.astype(np.float32)).mean())})
        del fog_l, dense_l
        renders[name] = rs
        agg = {k: float(np.mean([q[k] for q in per])) for k in FOG_KEYS}
        by_kind = {kd: {k: float(np.mean([q[k] for q in per if q["kind"] == kd]))
                        for k in FOG_KEYS}
                   for kd in sorted({q["kind"] for q in per})}
        # fog pair at the most displaced view (pull_out of anchor 0)
        pick = next(i for i, v in enumerate(views) if v["kind"] == "pull_out")
        pr, _, n_pr = load_splats(Path(ckpt), device, args.prune_opacity)
        img_p, a_p = render_rgba(pr, views[pick]["c2w"], K, W, H, device, args.sh_degree)
        n_full = splats["means"].shape[0]
        cv2.imwrite(str(out / "sheets" / f"fogpair_{name}.jpg"),
                    sheet([label(rs[pick], f"{name} ALL {n_full}", "pull_out a0"),
                           label(img_p, f"{name} op>={args.prune_opacity} ({n_pr})",
                                 "pull_out a0")], 2, args.tile_px),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        del splats, pr; torch.cuda.empty_cache()
        result["arms"][name] = {"ckpt": str(ckpt), "step": step,
                                "fog_proxy_mean": agg, "fog_proxy_by_kind": by_kind,
                                "per_view": per, "ckpt_stats": ckpt_stats(Path(ckpt)),
                                "pruned_count_at_prune_opacity": n_pr,
                                "pruned_fog_proxy_pull_out_a0": fog_proxies(a_p)}
        print(f"[offpath] {name}: {len(views)} views {time.time() - t0:.0f}s  "
              f"semi={agg['semi_alpha_frac']:.4f} fog={agg['fog_alpha_frac']:.4f} "
              f"cov={agg['coverage']:.4f} foglayer={agg['fog_layer_alpha_mean']:.4f} "
              f"prune_dMAE={agg['prune_delta_mae']:.3f}", flush=True)

    for i, v in enumerate(views):
        tiles = [label(renders[n][i], n, f"{v['kind']} a{v['anchor']}") for n, _ in arms]
        cv2.imwrite(str(out / "sheets" / f"cmp_{i:02d}_{v['kind']}_a{v['anchor']}.jpg"),
                    sheet(tiles, len(arms), args.tile_px), [cv2.IMWRITE_JPEG_QUALITY, 90])
    (out / "offpath.json").write_text(json.dumps(result, indent=1), encoding="utf8")
    print(f"[offpath] wrote {out / 'offpath.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
