"""Phase-1c step 2: VGGT dense geometry -> align to our COLMAP frame -> point render.

Run VGGT on observed views, get dense world-points + VGGT camera poses, align VGGT
frame to our (normalized) COLMAP frame via Umeyama sim3 on the shared camera centers,
then render the aligned point cloud at a target pose (z-buffer splat). SANITY: render
a held-out OBSERVED view (not fed to VGGT) and compare to its GT -> confirms alignment
+ render are correct before we render the wedge poses.

Saves the aligned cloud (points+colors) to _dataset/GPU/proof/vggt_cloud.npz for step 3.
"""
import os
os.environ["APEX_HOLDOUT"] = "0.275,0.45"

import glob
from pathlib import Path

import cv2
import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from core.training.gsplat_trainer import load_colmap_data, load_image

DATA = r"F:\HYPERclone\apex-sfm\_dataset\GPU\train_lpips\data"
IMGDIR = r"F:\HYPERclone\apex-sfm\_dataset\GPU\export\images"
OUT = Path(r"F:\HYPERclone\apex-sfm\_dataset\GPU\proof")
dev = "cuda"
dtype = torch.bfloat16


def umeyama(src, dst):
    """sim3 (s,R,t) mapping src->dst (Nx3 each). Returns s,R,t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = (D.T @ S) / src.shape[0]
    U, d, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    var_s = (S ** 2).sum() / src.shape[0]
    s = np.trace(np.diag(d)) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def render_cloud(P, C, w2c, K, H, W):
    """z-buffer splat of points P(M,3)+colors C(M,3) into a (H,W) image. BGR uint8 + mask."""
    Pc = (w2c[:3, :3] @ P.T + w2c[:3, 3:4]).T               # (M,3) camera space
    z = Pc[:, 2]
    front = z > 1e-3
    Pc, Cc, z = Pc[front], C[front], z[front]
    u = (K[0, 0] * Pc[:, 0] / z + K[0, 2]).round().astype(np.int64)
    v = (K[1, 1] * Pc[:, 1] / z + K[1, 2]).round().astype(np.int64)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, Cc = u[ok], v[ok], z[ok], Cc[ok]
    order = np.argsort(-z)                                   # far first; near overwrites
    pix = v[order] * W + u[order]
    img = np.zeros((H * W, 3), np.uint8); filled = np.zeros(H * W, bool)
    img[pix] = (Cc[order] * 255).astype(np.uint8); filled[pix] = True
    return img.reshape(H, W, 3)[..., ::-1], filled.reshape(H, W)


def main():
    data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True)
    name2idx = {n: i for i, n in enumerate(data["image_names"])}
    files = sorted(glob.glob(os.path.join(IMGDIR, "*.jpg")))
    train_names = set(data["image_names"][i] for i in data["train_indices"])
    obs = [f for f in files if os.path.basename(f) in train_names]
    vggt_in = [obs[i] for i in np.linspace(0, len(obs) - 1, 28).astype(int)]
    holdout_sanity = [f for f in obs if f not in vggt_in][len(obs) // 2]  # observed, not fed to VGGT

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
    images = load_and_preprocess_images(vggt_in).to(dev)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        pred = model(images)
    HW = images.shape[-2:]
    extr, _ = pose_encoding_to_extri_intri(pred["pose_enc"], HW)   # (1,N,3,4) w2c
    extr = extr[0].float().cpu().numpy()
    wp = pred["world_points"][0].float().cpu().numpy()            # (N,H,W,3) VGGT frame
    conf = pred["world_points_conf"][0].float().cpu().numpy()     # (N,H,W)
    cols = images.permute(0, 2, 3, 1).float().cpu().numpy()  # input (N,H,W,3) [0,1] (ToTensor)

    # VGGT cam centers + matched COLMAP cam centers (same images, same order)
    vc = np.array([(-e[:3, :3].T @ e[:3, 3]) for e in extr])      # (N,3) VGGT frame
    cc = np.array([data["camtoworlds"][name2idx[os.path.basename(p)]][:3, 3]
                   for p in vggt_in])                            # (N,3) COLMAP frame
    s, R, t = umeyama(vc, cc)
    align_err = np.linalg.norm((s * (R @ vc.T).T + t) - cc, axis=1).mean()
    print(f"VGGT->COLMAP sim3: scale {s:.3f}  cam-center align err {align_err:.4f} "
          f"(scene_scale {data['scene_scale']:.2f})", flush=True)

    # transform dense points to COLMAP frame, confidence-filter
    m = conf > np.quantile(conf, 0.4)
    P = wp[m]; C = cols[m]
    P = (s * (R @ P.T).T + t)
    np.savez(OUT / "vggt_cloud.npz", points=P.astype(np.float32), colors=C.astype(np.float32),
             s=s, R=R, t=t)
    print(f"aligned cloud: {len(P):,} points -> {OUT/'vggt_cloud.npz'}", flush=True)

    # SANITY: render a held-out observed view, compare to GT
    sidx = name2idx[os.path.basename(holdout_sanity)]
    b = load_image(data, sidx)
    gt = (b["image"].numpy() * 255).astype(np.uint8)[..., ::-1]; H, W = gt.shape[:2]
    K = b["K"].numpy(); w2c = np.linalg.inv(b["c2w"].numpy())
    rnd, filled = render_cloud(P, C, w2c, K, H, W)
    mse = ((rnd[filled].astype(float) - gt[filled].astype(float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / (mse + 1e-9))
    print(f"SANITY observed view {os.path.basename(holdout_sanity)}: "
          f"filled {100*filled.mean():.0f}%  PSNR(on filled) {psnr:.2f}", flush=True)
    cv2.imwrite(str(OUT / "vggt_sanity.jpg"),
                np.concatenate([gt, rnd], axis=1), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"wrote {OUT/'vggt_sanity.jpg'} (GT | VGGT-cloud render)")


if __name__ == "__main__":
    main()
