"""Type-3 "recreate" core: complete a view's gap with a 2D inpainter + InFusion depth.

This is the shared operation behind both flows the user wants:
  (1) inpaint fallback inside auto-complete — when a hole's reference photo lacks good
      RGB, fill the rendered view's gap with the inpainter instead of the real photo;
  (2) user-guided "complete this view" — aim a novel camera at an unseen region and
      recreate it (plausible, not faithful).

complete_view(splat, c2w, K, ...): render the splat at the view -> auto-mask the gap
(low-alpha enclosed by real geometry) -> inpaint (Z-Image, paste-back) -> InFusion
depth-completion on the inpainted view -> back-project new Gaussians. Returns the seed
Gaussians + the inpainted preview. InFusion needs SOME surrounding real depth to anchor
the completion, so this works for PARTIAL gaps; a fully-empty view (no anchor) degrades
— that is the honest limit of single-reference completion.
"""
from __future__ import annotations

import math
import os
import subprocess

import numpy as np
import torch
import cv2

from core.training.gsplat_trainer import load_image, render_splats
from core.enhancement.autocomplete import (
    AutoCompleteConfig, _render, _as_gaussians, SH,
)
from core.enhancement.inpaint_backend import inpaint, DEFAULT_BACKEND

SURFACE_PROMPT = ("seamless continuation of the existing surface, same material, "
                  "texture, color and lighting, photorealistic, no new objects")


def interior_gap_mask(alpha_np, k=9):
    """Low-alpha pixels enclosed by the object silhouette = a gap to complete."""
    obj = (alpha_np > 0.5).astype(np.uint8) * 255
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ff = obj.copy()
    m = np.zeros((obj.shape[0] + 2, obj.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 255)
    enclosed = cv2.bitwise_not(ff)
    enclosed = cv2.morphologyEx(enclosed, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    return enclosed


def _infusion_complete(rgb_np, mask_np, depth, alpha, K, c2w, cfg, rdir, dev):
    """Run InFusion depth-completion on a given RGB (the inpainted view) + mask.
    Returns raw back-projected colored points (N,6) or None."""
    os.makedirs(rdir, exist_ok=True)
    valid = (alpha > 0.5) & (depth > 1e-6)
    disp = torch.where(valid, 1.0 / depth.clamp(min=1e-6), torch.zeros_like(depth))
    cv2.imwrite(os.path.join(rdir, "rgb.png"), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(rdir, "mask.png"), mask_np)
    np.save(os.path.join(rdir, "disp.npy"), disp.cpu().numpy().astype(np.float32))
    np.save(os.path.join(rdir, "intri.npy"), K.cpu().numpy().astype(np.float64))
    np.save(os.path.join(rdir, "c2w.npy"), c2w.cpu().numpy().astype(np.float64))
    out = os.path.join(rdir, "out"); os.makedirs(out, exist_ok=True)
    run_dir = os.path.join(cfg.infusion_repo, "depth_inpainting", "run")
    subprocess.run([cfg.difix_python, "run_inference_inpainting.py",
                    "--input_rgb_path", os.path.join(rdir, "rgb.png"),
                    "--input_mask", os.path.join(rdir, "mask.png"),
                    "--input_depth_path", os.path.join(rdir, "disp.npy"),
                    "--model_path", cfg.infusion_ckpt, "--output_dir", out,
                    "--denoise_steps", "20", "--intri", os.path.join(rdir, "intri.npy"),
                    "--c2w", os.path.join(rdir, "c2w.npy"), "--use_mask", "--blend"],
                   cwd=run_dir, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raw = np.loadtxt(os.path.join(out, "rgb_mask.ply"), skiprows=10)
    return raw if raw.ndim == 2 and len(raw) else None


def complete_view(splat, c2w, K, W, H, scene, work_dir,
                  inpaint_backend=DEFAULT_BACKEND, prompt=SURFACE_PROMPT,
                  mask_np=None, cfg: AutoCompleteConfig = None, seed=0, dev="cuda"):
    """Inpaint a view's gap + InFusion-complete it. Returns dict with seed Gaussians,
    the auto/explicit mask, and rendered/inpainted previews (or None if no gap)."""
    cfg = cfg or AutoCompleteConfig()
    os.makedirs(work_dir, exist_ok=True)
    rr, a = _render(splat, c2w, K, W, H, dev, "RGB+ED")
    rgb = (rr[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    depth = rr[..., 3]
    if mask_np is None:
        mask_np = interior_gap_mask(a.cpu().numpy())
    if (mask_np > 0).sum() < 500:
        return None                                   # no gap to complete from this view

    rgb_p = os.path.join(work_dir, "render.png")
    mask_p = os.path.join(work_dir, "mask.png")
    inp_p = os.path.join(work_dir, "inpainted.png")
    cv2.imwrite(rgb_p, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_p, mask_np)
    inpaint(inpaint_backend, rgb_p, mask_p, inp_p, prompt=prompt, seed=seed)
    inp = cv2.cvtColor(cv2.imread(inp_p), cv2.COLOR_BGR2RGB)

    raw = _infusion_complete(inp, mask_np, depth, a, K, c2w, cfg,
                             os.path.join(work_dir, "infusion"), dev)
    if raw is None:
        return None
    seed_g = _as_gaussians(raw, splat, scene, dev)
    return dict(seed=seed_g, mask=mask_np, render=rgb, inpainted=inp,
                n_points=int(raw.shape[0]))


def _rot(axis, ang):
    a = axis / (np.linalg.norm(axis) + 1e-9); x, y, z = a; c, s = math.cos(ang), math.sin(ang)
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def propose_orbit_views(splat, data, n, extend_deg, dev):
    """Extend the captured orbit beyond its ends into UNSEEN angles (rotate the two
    extreme cameras outward around the orbit axis). These views look at the object
    from just-uncaptured directions — partial gaps with surrounding real geometry to
    anchor InFusion. Returns list of (c2w, K)."""
    O = splat["means"].median(0).values.cpu().numpy()
    cen = np.stack([data["camtoworlds"][i][:3, 3] for i in range(len(data["camtoworlds"]))])
    c0 = cen - cen.mean(0)
    _, _, V = np.linalg.svd(c0, full_matrices=False)
    axis = V[2]                                        # orbit-plane normal = up axis
    # angle of each camera around the axis
    ref = c0[0] - c0[0].dot(axis) * axis; ref /= np.linalg.norm(ref) + 1e-9
    perp = np.cross(axis, ref)
    proj = c0 - np.outer(c0.dot(axis), axis)
    ang = np.arctan2(proj.dot(perp), proj.dot(ref))
    lo_i, hi_i = int(ang.argmin()), int(ang.argmax())
    Kmed = torch.from_numpy(np.stack([data["Ks_dict"][data["camera_ids"][i]]
                                      for i in range(len(data["camtoworlds"]))]).mean(0)).float()
    step = math.radians(extend_deg) / max(n // 2, 1)
    views = []
    for base_i, sign in ((hi_i, +1), (lo_i, -1)):
        c2w = data["camtoworlds"][base_i].astype(np.float64)
        for k in range(1, n // 2 + 1):
            R = _rot(axis, sign * step * k)
            new = np.eye(4)
            new[:3, :3] = R @ c2w[:3, :3]
            new[:3, 3] = O + R @ (c2w[:3, 3] - O)
            views.append((torch.from_numpy(new).float().to(dev), Kmed.to(dev)))
    return views


def run_recreate(checkpoint_dir, data_dir, output_dir, config=None, progress_cb=None,
                 n_views=8, extend_deg=30, prompt=SURFACE_PROMPT):
    """Type-3 recreate-the-back (PLAUSIBLE, not faithful): extend the orbit into unseen
    angles, inpaint+complete each partial gap, integrate with a light cleanup that
    constrains the boundary against real views. Returns dict with the completed ckpt."""
    from pathlib import Path
    import glob
    from core.training.gsplat_trainer import load_colmap_data
    from core.enhancement.autocomplete import _find_ckpt, _short_distill, _before_after
    cfg = config or AutoCompleteConfig(); dev = cfg.device
    os.makedirs(output_dir, exist_ok=True)

    def prog(f, d):
        if progress_cb:
            progress_cb(min(max(f, 0.0), 1.0), d)

    prog(0.03, "Loading splat...")
    ck = _find_ckpt(checkpoint_dir)
    base = {k: v.to(dev) for k, v in torch.load(ck, weights_only=True, map_location=dev)["splats"].items()}
    data = load_colmap_data(data_dir, factor=cfg.data_factor, test_every=8, normalize=True)
    scene = float(data["scene_scale"])
    views = propose_orbit_views(base, data, n_views, extend_deg, dev)

    seeds = []; completed = 0
    for vi, (c2w, K) in enumerate(views):
        prog(0.05 + 0.6 * vi / max(len(views), 1), f"Recreating unseen view {vi+1}/{len(views)}...")
        b0 = load_image(data, 0); H, W = b0["image"].shape[:2]
        res = complete_view(base, c2w, K, W, H, scene, os.path.join(output_dir, f"nv{vi}"),
                            inpaint_backend=cfg.extra.get("inpaint_backend", DEFAULT_BACKEND),
                            prompt=prompt, cfg=cfg, dev=dev)
        if res is not None:
            seeds.append(res["seed"]); completed += 1

    if not seeds:
        prog(1.0, "No completable gaps found in the proposed unseen views.")
        return dict(checkpoint=ck, ply_path=None, completed=0,
                    report="No partial gaps with surrounding geometry to anchor "
                           "completion. (Fully-empty unseen views can't be completed "
                           "by single-reference depth completion — re-capture needed.)")
    acc = {k: torch.cat([s[k] for s in seeds], 0) for k in seeds[0]}
    prog(0.7, f"Integrating {acc['means'].shape[0]:,} plausible points (light cleanup)...")
    tr_idx = list(data["train_indices"]); g = torch.Generator().manual_seed(0)
    trial = {k: torch.cat([base[k], acc[k]], 0) for k in base}
    params, _ = _short_distill(base, acc, data, tr_idx, scene, dev, g, steps=400)
    final = {k: torch.cat([base[k], params[k]], 0) for k in base}
    out_ck = os.path.join(output_dir, "recreated.pt")
    torch.save({"splats": {k: v.cpu() for k, v in final.items()}}, out_ck)
    prog(1.0, f"Recreate done: {completed} views, plausible (not faithful).")
    return dict(checkpoint=out_ck, ply_path=None, completed=completed,
                report=f"Recreated {completed} unseen view(s), +{acc['means'].shape[0]:,} "
                       f"PLAUSIBLE points (not faithful — generated where no camera saw). "
                       f"For faithful geometry, re-capture these angles.")


if __name__ == "__main__":
    # Standalone validation: re-carve a Laptop hole, complete a hole-view with the
    # INPAINTER (not the real photo) -> confirm the inpaint->InFusion->seed chain
    # produces a plausible fill.
    import sys
    from core.training.gsplat_trainer import load_colmap_data
    DATA = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\data"
    CK = r"F:\HYPERclone\apex-sfm\_dataset\Laptop\apex_images\training\gsplat_prod\output_4k_gated\ckpts\ckpt_29999_rank0.pt"
    OUT = r"F:\HYPERclone\apex-sfm\_infusion\recreate_test"
    dev = "cuda"; os.makedirs(OUT, exist_ok=True)
    full = {k: v.to(dev) for k, v in torch.load(CK, weights_only=True, map_location=dev)["splats"].items()}
    data = load_colmap_data(DATA, factor=2, test_every=8, normalize=True)
    scene = float(data["scene_scale"])
    # carve (deterministic, most-detailed centre view)
    best = None
    for idx in list(data["test_indices"]):
        ph = load_image(data, idx)["image"].cpu().numpy(); H, W = ph.shape[:2]
        c = ph[int(.4 * H):int(.6 * H), int(.4 * W):int(.6 * W)]
        gx = np.abs(np.diff(c.mean(-1), axis=1)).mean()
        if best is None or gx > best[0]:
            best = (gx, idx)
    hv = best[1]; b = load_image(data, hv); K = b["K"].to(dev); c2w = b["c2w"].to(dev); H, W = b["image"].shape[:2]
    rr, a = _render(full, c2w, K, W, H, dev, "RGB+ED"); depth = rr[..., 3]
    selm = np.zeros((H, W), bool); bw, bh = int(.18 * W), int(.18 * H)
    selm[(H - bh) // 2:(H + bh) // 2, (W - bw) // 2:(W + bw) // 2] = True
    sel = ((a > 0.5) & (depth > 1e-6)).cpu().numpy() & selm
    Kn = K.cpu().numpy(); c2wn = c2w.cpu().numpy(); dd = depth.cpu().numpy()
    ys, xs = np.nonzero(sel); dv = dd[ys, xs]
    cam = np.stack([(xs - Kn[0, 2]) / Kn[0, 0] * dv, (ys - Kn[1, 2]) / Kn[1, 1] * dv, dv, np.ones_like(dv)], 1)
    world = (cam @ c2wn.T)[:, :3]; vs = 0.02 * scene; origin = world.min(0) - 4 * vs
    means = full["means"].cpu().numpy()
    def vkey(p): return np.floor((p - origin) / vs).astype(np.int64)
    surf = np.unique(vkey(world), axis=0)
    offs = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    dil = np.unique((surf[:, None, :] + offs[None]).reshape(-1, 3), axis=0); P = 1 << 21
    def lin(x): return ((x[:, 0] + P // 2) * P + (x[:, 1] + P // 2)) * P + (x[:, 2] + P // 2)
    inside = np.isin(lin(vkey(means)), lin(dil))
    holed = {k: v[torch.from_numpy(~inside).to(dev)].clone() for k, v in full.items()}
    # complete a side view of the hole (192) with the inpainter. The carved hole
    # renders as bleed-through (high alpha) from a real view, so pass the known
    # deleted-region footprint as the mask (validates the inpaint->InFusion->seed
    # chain; auto low-alpha masking applies to genuinely-empty recreate-back views).
    vv = 192; b = load_image(data, vv); K = b["K"].to(dev); c2w = b["c2w"].to(dev); H, W = b["image"].shape[:2]
    del_w = torch.from_numpy(means[inside]).float().to(dev)
    xc = (torch.cat([del_w, torch.ones(len(del_w), 1, device=dev)], 1) @ torch.linalg.inv(c2w).T)[:, :3]
    z = xc[:, 2]; uv = (xc / z.clamp(min=1e-6)[:, None]) @ K.T
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    mnp = np.zeros((H, W), np.uint8)
    uu = uv[ok, 0].long().cpu().numpy(); vv2 = uv[ok, 1].long().cpu().numpy()
    mnp[vv2, uu] = 255
    mnp = cv2.dilate(mnp, np.ones((15, 15), np.uint8))
    res = complete_view(holed, c2w, K, W, H, scene, OUT, inpaint_backend="zimage", mask_np=mnp)
    if res is None:
        print("no gap detected from view", vv); sys.exit(0)
    comb = {k: torch.cat([holed[k], res["seed"][k]], 0) for k in holed}
    before = _render(holed, c2w, K, W, H, dev, "RGB")[0][..., :3].clamp(0, 1)
    after = _render(comb, c2w, K, W, H, dev, "RGB")[0][..., :3].clamp(0, 1)
    gt = b["image"].to(dev)
    panel = torch.cat([gt, before, after], 1).cpu().numpy()
    cv2.imwrite(os.path.join(OUT, "recreate.png"),
                cv2.cvtColor((panel * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"completed view {vv}: +{res['n_points']:,} seed points")
    print("viz:", os.path.join(OUT, "recreate.png"), "(GT photo | holed | inpaint-completed seed)")
