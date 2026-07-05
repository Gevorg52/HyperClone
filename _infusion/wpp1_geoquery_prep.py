"""WP-P1 GeoQuery pipeline, stage 1 (MAIN env): render targets from the WP-S1 stack
ckpt and gather per-target GeoQuery inputs (reference GT image + MoGe metric depth +
raw-COLMAP poses + K). Writes a work dir + manifest that the geoquery-venv refiner
(wpp1_geoquery_refine.py) consumes.

Unit-consistency note: GeoQuery's cross-view warp (ref_depth + poses + K) is invariant
under a global similarity transform, so we feed it RAW-COLMAP poses + RAW MoGe depth
(already COLMAP-aligned) while rendering targets from the normalized-frame stack ckpt
with the correspondingly-normalized poses — images and pixel-correspondence are identical
either way, sidestepping the normalization-scale problem. (target_pose_norm is also saved
so novel-mode refined views can be appended as extra_views in the trainer's normalized space.)

Modes:
  frame  : targets = every-8 HELD-OUT views (clean 2D [PC] measurement; also saves held-out GT).
  novel  : targets = midpoint-interpolated NOVEL poses between consecutive train cams
           (distill supervision; no leakage — held-out views never enhanced/fed back).

Usage (repo root, main python):
  python _infusion/wpp1_geoquery_prep.py --mode frame \
    --ckpt <stack.pt> --data_dir <data> --factor 2 --work qa/wpp1_gq_frame
  python _infusion/wpp1_geoquery_prep.py --mode novel  --n_novel 60 \
    --ckpt <stack.pt> --data_dir <data> --factor 2 --work qa/wpp1_gq_novel
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.training.gsplat_trainer import (  # noqa: E402
    load_colmap_data, load_image, render_splats,
)
from core.enhancement.distill import (  # noqa: E402
    interpolate_pseudo_poses, _slerp, _quat_from_matrix, _matrix_from_quat,
)


def load_splats(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    return {k: v.to(device) for k, v in ck["splats"].items()}


def render_at(splats, c2w, K, W, H, sh_degree, device):
    w2c = torch.linalg.inv(torch.from_numpy(c2w).float().to(device))
    Kt = torch.from_numpy(np.asarray(K)).float().to(device)
    r, _, _ = render_splats(splats, w2c, Kt, int(W), int(H), sh_degree,
                            torch.zeros(3, device=device), "antialiased")
    return (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def interp_raw(c2w_a, c2w_b, t=0.5):
    q = _slerp(_quat_from_matrix(c2w_a[:3, :3]), _quat_from_matrix(c2w_b[:3, :3]), t)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = _matrix_from_quat(q)
    pose[:3, 3] = (1 - t) * c2w_a[:3, 3] + t * c2w_b[:3, 3]
    return pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frame", "novel"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--n_novel", type=int, default=60)
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    device = "cuda"

    dn = load_colmap_data(args.data_dir, factor=args.factor, test_every=8, normalize=True)
    dr = load_colmap_data(args.data_dir, factor=args.factor, test_every=8, normalize=False)
    splats = load_splats(args.ckpt, device)

    work = Path(args.work)
    for sub in ("targets", "refs", "ref_depth", "gt", "refined"):
        (work / sub).mkdir(parents=True, exist_ok=True)

    train_idx = np.sort(np.asarray(dn["train_indices"]))
    train_centers = dn["camtoworlds"][train_idx, :3, 3]
    depth_dir = Path(args.data_dir) / "priors" / "depth"

    def nearest_train(center_norm, exclude_idx=None):
        d = np.linalg.norm(train_centers - center_norm, axis=1)
        if exclude_idx is not None:
            eq = np.where(train_idx == exclude_idx)[0]
            if len(eq):
                d[eq[0]] = np.inf
        return int(train_idx[int(np.argmin(d))])

    def ref_inputs(ref_idx):
        b = load_image(dn, ref_idx)
        ref_img = (b["image"].cpu().numpy() * 255).astype(np.uint8)  # real GT
        stem = Path(dn["image_names"][ref_idx]).stem
        dpath = depth_dir / f"{stem}.npy"
        depth = np.load(dpath).astype(np.float32)
        K = np.asarray(dn["Ks_dict"][dn["camera_ids"][ref_idx]], dtype=np.float64)
        ref_pose_raw = dr["camtoworlds"][ref_idx].astype(np.float64)
        return ref_img, depth, K, ref_pose_raw

    manifest = []
    with torch.no_grad():
        if args.mode == "frame":
            targets = sorted(int(x) for x in dn["test_indices"])
            for i, ti in enumerate(targets):
                bt = load_image(dn, ti)
                gt = (bt["image"].cpu().numpy() * 255).astype(np.uint8)
                H, W = gt.shape[:2]
                K = np.asarray(dn["Ks_dict"][dn["camera_ids"][ti]], dtype=np.float64)
                tgt = render_at(splats, dn["camtoworlds"][ti], K, W, H, args.sh_degree, device)
                r_idx = nearest_train(dn["camtoworlds"][ti, :3, 3])
                ref_img, depth, Kref, ref_pose_raw = ref_inputs(r_idx)
                tgt_pose_raw = dr["camtoworlds"][ti].astype(np.float64)
                Image.fromarray(tgt).save(work / "targets" / f"{i:04d}.png")
                Image.fromarray(ref_img).save(work / "refs" / f"{i:04d}.png")
                Image.fromarray(gt).save(work / "gt" / f"{i:04d}.png")
                np.save(work / "ref_depth" / f"{i:04d}.npy", depth)
                manifest.append(dict(
                    i=i, mode="frame", target_idx=ti, ref_idx=r_idx,
                    stem=Path(dn["image_names"][ti]).stem,
                    target_png=f"targets/{i:04d}.png", ref_png=f"refs/{i:04d}.png",
                    ref_depth_npy=f"ref_depth/{i:04d}.npy", gt_png=f"gt/{i:04d}.png",
                    K=Kref.tolist(), ref_pose_raw=ref_pose_raw.tolist(),
                    target_pose_raw=tgt_pose_raw.tolist()))
                print(f"[frame {i:03d}] tgt idx {ti} ref idx {r_idx} {W}x{H}", flush=True)
        else:  # novel
            order = train_idx
            pairs = list(zip(order[:-1], order[1:]))
            sel = np.linspace(0, len(pairs) - 1, min(args.n_novel, len(pairs))).astype(int)
            # reference intrinsics/size from first train cam (single-camera capture)
            ref0 = int(order[0])
            b0 = load_image(dn, ref0)
            H, W = b0["image"].shape[:2]
            for i, pi in enumerate(sel):
                a, bb = int(pairs[pi][0]), int(pairs[pi][1])
                c2w_norm = interp_raw(dn["camtoworlds"][a], dn["camtoworlds"][bb], 0.5)
                c2w_raw = interp_raw(dr["camtoworlds"][a], dr["camtoworlds"][bb], 0.5)
                K = np.asarray(dn["Ks_dict"][dn["camera_ids"][a]], dtype=np.float64)
                tgt = render_at(splats, c2w_norm, K, W, H, args.sh_degree, device)
                r_idx = nearest_train(c2w_norm[:3, 3])
                ref_img, depth, Kref, ref_pose_raw = ref_inputs(r_idx)
                Image.fromarray(tgt).save(work / "targets" / f"{i:04d}.png")
                Image.fromarray(ref_img).save(work / "refs" / f"{i:04d}.png")
                np.save(work / "ref_depth" / f"{i:04d}.npy", depth)
                manifest.append(dict(
                    i=i, mode="novel", pair=[a, bb], ref_idx=r_idx,
                    target_png=f"targets/{i:04d}.png", ref_png=f"refs/{i:04d}.png",
                    ref_depth_npy=f"ref_depth/{i:04d}.npy",
                    K=Kref.tolist(), ref_pose_raw=ref_pose_raw.tolist(),
                    target_pose_raw=c2w_raw.tolist(),
                    target_pose_norm=c2w_norm.tolist()))
                print(f"[novel {i:03d}] pair ({a},{bb}) ref idx {r_idx} {W}x{H}", flush=True)

    (work / "manifest.json").write_text(json.dumps(dict(
        mode=args.mode, factor=args.factor, sh_degree=args.sh_degree,
        ckpt=str(args.ckpt), data_dir=str(args.data_dir),
        n=len(manifest), entries=manifest), indent=2), encoding="utf-8")
    print(f"WROTE {work/'manifest.json'} ({len(manifest)} entries)", flush=True)


if __name__ == "__main__":
    main()
