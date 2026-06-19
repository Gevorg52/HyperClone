"""Validate SplatSuRe's per-Gaussian fidelity score on a real trained splat
(imperfect-capture lever #4). Faithful re-implementation of the paper's math:

  r_t   = per-view screen-space 3-sigma radius of each Gaussian (gsplat projection,
          eps2d=0 to exclude the 0.3 AA dilation, exactly as the paper specifies)
  rho   = r_max / r_min  over the views that observe the Gaussian (>=3 views, else 0)
  score = sigmoid((rho - 1.1) / 0.05)

Interpretation:
  high score (rho >> 1)  -> some view saw it CLOSE -> multi-view fusion already has the
                            detail -> TRUST the splat, suppress generative SR (faithful)
  low score  (rho ~ 1)   -> every view saw it equally coarse -> UNDERSAMPLED -> SR headroom

The decisive question for OUR capture style: do our orbits have DISTANCE VARIATION?
A constant-radius orbit gives rho~1 everywhere -> the gate can't separate "faithful SR
possible" from "must hallucinate" -> SplatSuRe degenerates to global SR. So we also
report the camera-distance spread.

Usage: python val_splatsure.py <ckpt.pt> <data_dir> [factor] [n_views]
"""
import os, sys, glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.training.gsplat_trainer import load_colmap_data
from gsplat.cuda._wrapper import fully_fused_projection

TAU, K_SIG = 1.1, 0.05


def main(ckpt_path, data_dir, factor=2, n_views=0, band=0.0):
    dev = "cuda"
    c = torch.load(ckpt_path, map_location=dev, weights_only=False)
    sp = c["splats"] if isinstance(c, dict) and "splats" in c else c
    means = sp["means"].to(dev).float()
    quats = sp["quats"].to(dev).float()
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-12)
    scales = torch.exp(sp["scales"].to(dev).float())
    opac = torch.sigmoid(sp["opacities"].to(dev).float())
    N = means.shape[0]
    print(f"[ckpt] {N:,} Gaussians")

    data = load_colmap_data(data_dir, factor=factor, normalize=True)
    c2ws = data["camtoworlds"]                       # (V,4,4) normalized
    cam_ids = data["camera_ids"]
    Ks_dict = data["Ks_dict"]; imsize = data["imsize_dict"]
    V = len(c2ws)

    # object center (centroid of SfM points) + camera distances
    obj = torch.from_numpy(data["points"]).float().mean(0)
    cam_pos = torch.from_numpy(c2ws[:, :3, 3]).float()
    dists = (cam_pos - obj).norm(dim=-1).numpy()

    sel = list(range(V))
    if band > 0:   # keep only a near-constant-distance band (simulate a constant-radius orbit)
        med = float(np.median(dists))
        sel = [j for j in sel if abs(dists[j] - med) <= band * med]
        print(f"[band]  constant-distance subset: {len(sel)}/{V} views within +/-{band:.0%} "
              f"of median distance {med:.3f}")
    elif n_views > 0:
        sel = list(range(0, V, max(1, V // n_views)))
    print(f"[cams] {V} views, using {len(sel)}")
    print(f"[capture] camera->object distance: min {dists.min():.3f}, "
          f"median {np.median(dists):.3f}, max {dists.max():.3f}, "
          f"spread max/min = {dists.max()/max(dists.min(),1e-6):.2f}x")

    r_min = torch.full((N,), 1e30, device=dev)
    r_max = torch.zeros((N,), device=dev)
    n_obs = torch.zeros((N,), dtype=torch.int32, device=dev)

    with torch.no_grad():
        for j in sel:
            cid = cam_ids[j]
            K = torch.from_numpy(Ks_dict[cid]).float().to(dev)[None]
            W, H = imsize[cid]
            viewmat = torch.linalg.inv(torch.from_numpy(c2ws[j]).float().to(dev))[None]
            out = fully_fused_projection(
                means, None, quats, scales, viewmat, K, int(W), int(H),
                eps2d=0.0, near_plane=0.01, far_plane=1e10, packed=False,
                opacities=opac,
            )
            radii = out[0][0]                         # (N,) or (N,2)
            if radii.ndim == 2:
                radii = radii.max(dim=-1).values      # 3-sigma extent
            radii = radii.float()
            vis = radii > 0
            n_obs += vis.int()
            r_max = torch.where(vis & (radii > r_max), radii, r_max)
            r_min = torch.where(vis & (radii < r_min), radii, r_min)

    enough = n_obs >= 3
    rho = torch.where(enough, r_max / r_min.clamp(min=1e-6), torch.ones_like(r_max))
    score = torch.sigmoid((rho - TAU) / K_SIG)
    score = torch.where(enough, score, torch.zeros_like(score))

    # weight by opacity (invisible Gaussians don't matter for renders)
    w = opac.clamp(0, 1)
    vis_g = (n_obs >= 1)
    print(f"\n[visibility] seen by >=1 view: {int(vis_g.sum()):,} "
          f"({100*float(vis_g.float().mean()):.1f}%), "
          f">=3 views: {int(enough.sum()):,} ({100*float(enough.float().mean()):.1f}%)")
    rr = rho[enough].cpu().numpy()
    print(f"[rho]   over >=3-view Gaussians: "
          f"p10 {np.percentile(rr,10):.2f}, median {np.median(rr):.2f}, "
          f"p90 {np.percentile(rr,90):.2f}, max {rr.max():.1f}")

    # SR headroom = low-score (undersampled) opacity-weighted fraction among constrained
    sc = score[enough]; ww = w[enough]
    frac_under = float((ww * (sc < 0.5)).sum() / ww.sum())
    frac_well = float((ww * (sc >= 0.5)).sum() / ww.sum())
    print(f"[gate]  among >=3-view Gaussians (opacity-weighted): "
          f"well-sampled (trust splat) {100*frac_well:.1f}%, "
          f"undersampled (SR headroom) {100*frac_under:.1f}%")

    # --- heatmap render: where is detail under-resolved? (red) vs well-resolved (blue) ---
    try:
        import imageio.v2 as imageio
        from gsplat import rasterization
        under = torch.where(enough, 1.0 - score, torch.ones_like(score))  # <3 views = under
        heat = torch.stack([under, torch.zeros_like(under), 1.0 - under], dim=-1)  # R..B
        j = sel[len(sel) // 2]
        cid = cam_ids[j]; K = torch.from_numpy(Ks_dict[cid]).float().to(dev)[None]
        W, H = imsize[cid]
        vm = torch.linalg.inv(torch.from_numpy(c2ws[j]).float().to(dev))[None]
        img, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac,
            colors=heat[None], viewmats=vm, Ks=K, width=int(W), height=int(H),
            sh_degree=None, render_mode="RGB")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splatsure_heat.png")
        imageio.imwrite(out, (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        print(f"[heat]  under-resolved=red, well-resolved=blue -> {out}")
    except Exception as e:
        print(f"[heat]  skipped ({e})")

    return dict(score=score.cpu(), rho=rho.cpu(), n_obs=n_obs.cpu(),
                dists=dists, frac_under=frac_under, frac_well=frac_well)


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        r"_dataset/GPU_V4/apex_images/training/gsplat_prod/output/ckpts/ckpt_29999_rank0.pt"
    ddir = sys.argv[2] if len(sys.argv) > 2 else \
        r"_dataset/GPU_V4/apex_images/training/gsplat_prod/data"
    fac = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    nv = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    band = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    main(ckpt, ddir, fac, nv, band)
