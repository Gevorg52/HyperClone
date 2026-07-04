"""WP-G1: export an Apex COLMAP scene into a 3DGS-DR-ready (classic Inria)
COLMAP dataset.

Why: 3DGS-DR is an Inria-3DGS fork whose loader (a) asserts on the OPENCV camera
model and (b) cannot undistort, while our capture cameras ARE OPENCV (k1,k2~0.1)
and our trainer trains/evaluates on UNDISTORTED, ROI-cropped frames. To give
3DGS-DR a fair shot at the specular ceiling AND keep the held-out GT byte-identical
to the WP-S1 stack, we reuse OUR OWN loader (load_colmap_data + load_image) to
produce, per view: the exact undistorted+cropped factor-N image, the undistorted
PINHOLE intrinsics, and the (optionally world-normalized) pose. We then write a
classic COLMAP text model (cameras.txt / images.txt / points3D.txt) + the images.

Split parity: 3DGS-DR sorts cameras by image_name and holds out idx%8==0; our
load_colmap_data sorts by name and holds out idx%test_every==0 — identical sets
(verified: GPU_V4 569->72 test, Laptop 304->38 test). We name exported images by
stem+.png so the sort order is preserved.

Usage:
  python _infusion/wpg1_export_colmap.py \
    --data_dir _dataset/GPU_V4/apex_images/training/wps1/ctrl/data \
    --out_dir  _dataset/GPU_V4/apex_images/training/wpg1/3dgsdr_data \
    --factor 2 --test_every 8
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.gsplat_trainer import load_colmap_data, load_image  # noqa: E402

# rotmat2qvec from 3DGS-DR's own colmap_loader (same convention it reads back)
_cl_path = r"D:\REPOS\HYPERclone\3DGS-DR\scene\colmap_loader.py"
_spec = importlib.util.spec_from_file_location("dr_colmap_loader", _cl_path)
_cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cl)
rotmat2qvec = _cl.rotmat2qvec
qvec2rotmat = _cl.qvec2rotmat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--test_every", type=int, default=8)
    ap.add_argument("--normalize", action="store_true", default=True)
    args = ap.parse_args()

    data = load_colmap_data(args.data_dir, factor=args.factor,
                            test_every=args.test_every, normalize=args.normalize)
    n = len(data["image_names"])
    out = Path(args.out_dir)
    img_out = out / "images"
    sparse_out = out / "sparse" / "0"
    img_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    test_set = set(int(i) for i in data["test_indices"])

    # --- per-view export -----------------------------------------------------
    images_lines = []
    manifest = []
    K_ref = None
    HW_ref = None
    max_pose_err = 0.0
    for idx in range(n):
        b = load_image(data, idx)
        img = (b["image"].clamp(0, 1).numpy() * 255.0 + 0.5).astype(np.uint8)  # HxWx3 RGB
        K = b["K"].numpy().astype(np.float64)
        c2w = b["c2w"].numpy().astype(np.float64)
        H, W = img.shape[:2]
        if K_ref is None:
            K_ref, HW_ref = K.copy(), (H, W)
        else:
            assert (H, W) == HW_ref, f"view {idx} size {(H,W)} != {HW_ref}"
            assert np.allclose(K, K_ref, atol=1e-3), f"view {idx} K differs"

        stem = Path(data["image_names"][idx]).stem
        name = stem + ".png"
        imageio.imwrite(str(img_out / name), img)

        w2c = np.linalg.inv(c2w)
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]
        qvec = rotmat2qvec(R_w2c)
        # round-trip check (reconstruct c2w exactly as 3DGS-DR will)
        R_back = np.transpose(qvec2rotmat(qvec))   # = c2w rotation
        Rt = np.eye(4); Rt[:3, :3] = R_back.T; Rt[:3, 3] = t_w2c
        c2w_back = np.linalg.inv(Rt)
        max_pose_err = max(max_pose_err, float(np.abs(c2w_back - c2w).max()))

        img_id = idx + 1
        images_lines.append(
            f"{img_id} {qvec[0]:.9f} {qvec[1]:.9f} {qvec[2]:.9f} {qvec[3]:.9f} "
            f"{t_w2c[0]:.9f} {t_w2c[1]:.9f} {t_w2c[2]:.9f} 1 {name}\n")
        images_lines.append("\n")  # empty points2D line (required by reader)
        manifest.append({"idx": idx, "name": name, "stem": stem,
                         "is_test": idx in test_set})
        if (idx + 1) % 50 == 0 or idx == n - 1:
            print(f"  exported {idx+1}/{n}", flush=True)

    # --- cameras.txt (single shared PINHOLE) ---------------------------------
    fx, fy = K_ref[0, 0], K_ref[1, 1]
    cx, cy = K_ref[0, 2], K_ref[1, 2]
    Hh, Ww = HW_ref
    (sparse_out / "cameras.txt").write_text(
        "# Camera list\n"
        f"1 PINHOLE {Ww} {Hh} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n",
        encoding="utf-8")

    (sparse_out / "images.txt").write_text(
        "# Image list\n" + "".join(images_lines), encoding="utf-8")

    # --- points3D.txt --------------------------------------------------------
    pts = data["points"]
    rgb = data["points_rgb"]
    with open(sparse_out / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# 3D point list\n")
        for i in range(pts.shape[0]):
            x, y, z = pts[i]
            r, g, bb = int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])
            f.write(f"{i+1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {bb} 0\n")

    # --- env_scope suggestion (median center + robust radius) ---------------
    c = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - c, axis=1)
    env = {
        "center": [float(v) for v in c],
        "scene_scale": float(data["scene_scale"]),
        "radius_scene_scale": float(data["scene_scale"]),
        "dist_p50": float(np.percentile(dist, 50)),
        "dist_p85": float(np.percentile(dist, 85)),
        "dist_p90": float(np.percentile(dist, 90)),
    }

    meta = {
        "data_dir": str(args.data_dir), "out_dir": str(args.out_dir),
        "factor": args.factor, "test_every": args.test_every,
        "normalize": bool(args.normalize),
        "n_images": n, "n_test": len(test_set), "n_train": n - len(test_set),
        "n_points": int(pts.shape[0]),
        "image_size_HW": [Hh, Ww],
        "K": K_ref.tolist(),
        "max_pose_roundtrip_err": max_pose_err,
        "env_scope_suggestion": env,
        "manifest": manifest,
    }
    (out / "wpg1_export_meta.json").write_text(json.dumps(meta, indent=2),
                                               encoding="utf-8")
    print(f"DONE: {n} images ({len(test_set)} test), {pts.shape[0]} points -> {out}")
    print(f"  image size HxW = {Hh}x{Ww}; max pose round-trip err = {max_pose_err:.2e}")
    print(f"  env_scope suggestion: center={np.round(c,3).tolist()} "
          f"radius(scene_scale)={env['radius_scene_scale']:.3f} "
          f"p85={env['dist_p85']:.3f} p90={env['dist_p90']:.3f}")


if __name__ == "__main__":
    main()
