"""WP-F2 gate evidence: deck-region SfM-point density before vs after dense init.

Reproduces the quality-roadmap "0.28x SfM-init density" profiling: projects every
SfM point into each held-out view and counts points landing in the flat DECK box vs
the textured KEYBOARD box, per unit image area (points per kpx = per 1000 px).
Reports deck density, keyboard density, and the deck/keyboard ratio for the ORIGINAL
sparse model and the dense-init AUGMENTED model.

Boxes (fractional x0,x1,y0,y1):
  DECK     = (0.30, 0.72, 0.55, 0.95)  # identical to wpf1_deck_eval.py / render_compare.py
  KEYBOARD = (0.36, 0.80, 0.50, 0.68)  # textured keys — DEFINED + RECORDED here (WP-F2);
                                        # the original 0.28x profiling's keyboard region
                                        # was never committed to a box, so we fix one.

Projection is pinhole (fx,fy,cx,cy from the camera; radial distortion ignored — a
sub-pixel effect that does not change deck-vs-keyboard counts). Both models share the
same cameras/poses, so the geometry is identical and only the point SET differs.

Held-out views = name-sorted every-`test_every` images (the trainer's default split).

Usage (repo root):
  python _infusion/wpf2_density_eval.py \
    --off <orig sparse/0> --on <augmented model dir> --out _infusion/wpf2_density.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap

DECK = (0.30, 0.72, 0.55, 0.95)
KEYBOARD = (0.36, 0.80, 0.50, 0.68)


def _K_and_size(cam):
    p = cam.params
    m = cam.model.name
    if m in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:  # PINHOLE, OPENCV, ...
        fx, fy = p[0], p[1]; cx, cy = p[2], p[3]
    return float(fx), float(fy), float(cx), float(cy), int(cam.width), int(cam.height)


def _project(rec, points_xyz):
    """Return list of (frac_uv (M,2), W, H) per held-out image for the given points."""
    out = []
    Xh = points_xyz  # (N,3)
    for img in _heldout_images(rec):
        cam = rec.cameras[img.camera_id]
        fx, fy, cx, cy, W, H = _K_and_size(cam)
        cfw = img.cam_from_world()
        R = np.asarray(cfw.rotation.matrix())
        t = np.asarray(cfw.translation).reshape(3)
        Xc = Xh @ R.T + t  # (N,3)
        z = Xc[:, 2]
        front = z > 1e-6
        u = fx * Xc[:, 0] / np.where(front, z, 1.0) + cx
        v = fy * Xc[:, 1] / np.where(front, z, 1.0) + cy
        fu, fv = u / W, v / H
        inside = front & (fu >= 0) & (fu < 1) & (fv >= 0) & (fv < 1)
        out.append((np.stack([fu[inside], fv[inside]], axis=1), W, H))
    return out


def _heldout_images(rec, test_every=8):
    imgs = sorted(rec.images.values(), key=lambda im: im.name)
    return [im for i, im in enumerate(imgs) if i % test_every == 0]


def _box_density(projections, box):
    """Mean over views of (points-in-box / box-area-in-kpx)."""
    x0, x1, y0, y1 = box
    dens = []
    counts = []
    for uv, W, H in projections:
        if len(uv) == 0:
            area_kpx = (x1 - x0) * W * (y1 - y0) * H / 1000.0
            dens.append(0.0); counts.append(0); continue
        m = (uv[:, 0] >= x0) & (uv[:, 0] < x1) & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
        n = int(m.sum())
        area_kpx = (x1 - x0) * W * (y1 - y0) * H / 1000.0
        dens.append(n / area_kpx)
        counts.append(n)
    return float(np.mean(dens)), int(np.mean(counts)), float(np.mean([d for d in dens]))


def _eval_model(model_dir):
    rec = pycolmap.Reconstruction(str(model_dir))
    xyz = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64)
    proj = _project(rec, xyz)
    deck_d, deck_n, _ = _box_density(proj, DECK)
    kb_d, kb_n, _ = _box_density(proj, KEYBOARD)
    # whole-frame density for reference
    all_d = []
    for uv, W, H in proj:
        all_d.append(len(uv) / (W * H / 1000.0))
    return {
        "n_points_total": int(len(xyz)),
        "n_heldout_views": len(proj),
        "deck_density_per_kpx": deck_d,
        "deck_points_mean_per_view": deck_n,
        "keyboard_density_per_kpx": kb_d,
        "keyboard_points_mean_per_view": kb_n,
        "deck_over_keyboard_ratio": deck_d / kb_d if kb_d > 0 else float("nan"),
        "frame_density_per_kpx": float(np.mean(all_d)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True, help="original sparse model dir")
    ap.add_argument("--on", required=True, help="augmented (dense-init) model dir")
    ap.add_argument("--out", default="_infusion/wpf2_density.json")
    args = ap.parse_args()

    res = {
        "method": "project all SfM points into every-8 held-out views; count in "
                  "fractional DECK vs KEYBOARD boxes; density = points / box-kpx, "
                  "mean over views (pinhole, distortion ignored)",
        "deck_box_frac": dict(zip("x0 x1 y0 y1".split(), DECK)),
        "keyboard_box_frac": dict(zip("x0 x1 y0 y1".split(), KEYBOARD)),
        "off": _eval_model(args.off),
        "on": _eval_model(args.on),
    }
    o, n = res["off"], res["on"]
    res["delta"] = {
        "deck_density_x": n["deck_density_per_kpx"] / o["deck_density_per_kpx"]
        if o["deck_density_per_kpx"] > 0 else float("nan"),
        "keyboard_density_x": n["keyboard_density_per_kpx"] / o["keyboard_density_per_kpx"]
        if o["keyboard_density_per_kpx"] > 0 else float("nan"),
        "deck_over_keyboard_before": o["deck_over_keyboard_ratio"],
        "deck_over_keyboard_after": n["deck_over_keyboard_ratio"],
    }
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")
    print(f"DECK density  off {o['deck_density_per_kpx']:.2f} -> on "
          f"{n['deck_density_per_kpx']:.2f}  ({res['delta']['deck_density_x']:.2f}x)")
    print(f"deck/keyboard off {o['deck_over_keyboard_ratio']:.3f} -> on "
          f"{n['deck_over_keyboard_ratio']:.3f}")


if __name__ == "__main__":
    main()
