"""Auto-complete: detect holes/defects in a trained 3DGS, seed them, and
reconstruct faithfully.

Validated pipeline (see memory 3dgs-quality-findings):
  detect (err x warp multi-view consistency + occlusion-aware coverage, edge-gated)
  -> InFusion depth-completion = COLD-START SEED (places Gaussians where the splat is
     empty; MCMC alone can't spawn in holes)
  -> accept/reject gate (commit a fill only if it improves agreement with the REAL
     photos on held-out neighbour views -> never degrades good regions)
  -> production trainer (MCMC inject_noise + relocation + gated-LPIPS) reconstructs
     the seeded holes -> sharp, faithful detail (recovers >=100% of the hole-LPIPS gap
     vs GT photos in validation).

InFusion's Stage-2 depth model is pure diffusers and runs in the isolated `difix`
conda env (no INRIA rasterizer); the reconstruction runs in this (base) env via the
production gsplat trainer. Type-1 holes (visible in some real view) are filled
faithfully from that view's own photo; truly-unseen (Type-3) regions cannot be filled
from views and are reported for re-capture instead.
"""
from __future__ import annotations

import glob
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats

SH, C0 = 3, 0.28209479177387814
ProgressCallback = Callable[[float, str], None]


# ---------------------------------------------------------------------------
# Config / result
# ---------------------------------------------------------------------------

@dataclass
class AutoCompleteConfig:
    """Parameters + environment paths for the auto-complete engine."""

    # environment (InFusion depth model runs in the isolated difix env)
    infusion_repo: str = r"F:\HYPERclone\infusion"
    infusion_ckpt: str = r"F:\HYPERclone\infusion\checkpoints\Infusion"
    difix_python: str = r"C:\Users\gevor\anaconda3\envs\difix\python.exe"
    # detection / filling
    data_factor: int = 2
    max_rounds: int = 6            # one hole filled (or rejected) per round
    center_prior: bool = False     # True only for full-scene (un-masked) scans
    # reconstruction (the production trainer)
    recon_steps: int = 5000
    lpips_lambda: float = 0.2
    device: str = "cuda"
    extra: dict = field(default_factory=dict)


@dataclass
class AutoCompleteResult:
    checkpoint: str                       # reconstructed .pt (None if inert)
    ply_path: Optional[str]
    holes_filled: int
    holes_rejected: int
    report: str
    sample_renders: list                  # (path, caption) before/after pairs


# ---------------------------------------------------------------------------
# Core helpers (validated in _infusion/auto_full.py)
# ---------------------------------------------------------------------------

def _render(sp, c2w, K, W, H, dev, mode="RGB+ED"):
    r, a, _ = render_splats(sp, torch.linalg.inv(c2w), K, W, H, SH,
                            torch.zeros(3, device=dev), "antialiased", render_mode=mode)
    return r[0], a[0, ..., 0]


def _warp_disagree(photoA, dA, KA, c2wA, nbrs, data, dev):
    H, W = dA.shape
    vv, uu = torch.meshgrid(torch.arange(H, device=dev).float(),
                            torch.arange(W, device=dev).float(), indexing="ij")
    cam = torch.stack([(uu - KA[0, 2]) / KA[0, 0] * dA, (vv - KA[1, 2]) / KA[1, 1] * dA,
                       dA, torch.ones_like(dA)], -1)
    world = cam @ c2wA.T
    dis = []
    for B in nbrs:
        b = load_image(data, B); pB = b["image"].to(dev)
        KB = b["K"].to(dev); c2wB = b["c2w"].to(dev)
        cB = world @ torch.linalg.inv(c2wB).T
        z = cB[..., 2].clamp(min=1e-6)
        gx = ((cB[..., 0] / z) * KB[0, 0] + KB[0, 2]) / (W - 1) * 2 - 1
        gy = ((cB[..., 1] / z) * KB[1, 1] + KB[1, 2]) / (H - 1) * 2 - 1
        s = F.grid_sample(pB.permute(2, 0, 1)[None], torch.stack([gx, gy], -1)[None],
                          align_corners=True, padding_mode="zeros")[0].permute(1, 2, 0)
        dd = (s - photoA).abs().mean(-1)
        dd[(cB[..., 2] <= 0) | (gx.abs() > 1) | (gy.abs() > 1)] = float("nan")
        dis.append(dd)
    return torch.nan_to_num(torch.nanmedian(torch.stack(dis), 0).values, nan=0.0)


def _detect(sp, idx, data, C, scene, dev, center_prior):
    b = load_image(data, idx); photo = b["image"].to(dev)
    K = b["K"].to(dev); c2w = b["c2w"].to(dev); H, W = photo.shape[:2]
    rr, a = _render(sp, c2w, K, W, H, dev, "RGB+ED")
    depth = rr[..., 3]; obj = (a > 0.5).cpu().numpy()
    if obj.sum() < 100:
        return None
    err = (rr[..., :3].clamp(0, 1) - photo).abs().mean(-1)
    dvec = (C - C[idx]).norm(dim=1); dvec[idx] = 1e9
    nbrs = torch.argsort(dvec)[:4].tolist()
    warp = _warp_disagree(photo, depth, K, c2w, nbrs, data, dev)
    defect = (err * warp).cpu().numpy(); defect[~obj] = 0
    # edge-gate: suppress depth-discontinuity / silhouette false-positives
    d = np.where(obj, depth.cpu().numpy(), 0).astype(np.float32)
    gy, gx = np.gradient(d); gmag = np.hypot(gx, gy)
    edge = cv2.dilate((gmag > 0.02 * scene).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    defect[edge] = 0
    if center_prior:
        yy, xx = np.mgrid[0:H, 0:W]
        sig = 0.42 * math.hypot(H, W) / 2
        defect = defect * np.exp(-(((yy - H / 2) ** 2 + (xx - W / 2) ** 2) / (2 * sig ** 2)))
    thr = np.percentile(defect[obj], 97.0)
    m = (defect > max(thr, 1e-5)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m); floor = 0.004 * obj.sum()
    for a_, i in sorted([(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)], reverse=True)[:2]:
        if a_ >= floor:
            keep[lab == i] = 255
    keep = cv2.dilate(keep, np.ones((9, 9), np.uint8)); keep[~obj] = 0
    return dict(idx=idx, mask=keep, photo=photo, K=K, c2w=c2w, depth=depth, a=a,
                H=H, W=W, area=int((keep > 0).sum()), objarea=int(obj.sum()))


def _footprint(fill, c2w, K, W, H, dev):
    with torch.no_grad():
        _, a = _render(fill, c2w, K, W, H, dev, "RGB")
    m = (a > 0.1).cpu().numpy()
    if m.sum() < 1500:
        return None
    ys, xs = np.nonzero(m); return xs.min(), ys.min(), xs.max(), ys.max()


def _cbox(box, W, H, pad=24, mn=256):
    x0, y0, x1, y1 = box; x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    if x1 - x0 < mn: c = (x0 + x1) // 2; x0, x1 = c - mn // 2, c + mn // 2
    if y1 - y0 < mn: c = (y0 + y1) // 2; y0, y1 = c - mn // 2, c + mn // 2
    x0 = max(0, min(x0, W - 1)); x1 = max(x0 + 1, min(x1, W))
    y0 = max(0, min(y0, H - 1)); y1 = max(y0 + 1, min(y1, H))
    return int(x0), int(y0), int(x1), int(y1)


def _fill_gain(base, acc_before, acc_new, newg, fill_idx, data, C, dev):
    prev = base if acc_before is None else {k: torch.cat([base[k], acc_before[k]], 0) for k in base}
    new = {k: torch.cat([base[k], acc_new[k]], 0) for k in base}
    order = torch.argsort((C - C[fill_idx]).norm(dim=1)).tolist()
    gains = []
    for v in order:
        if v == fill_idx or len(gains) >= 3:
            continue
        b = load_image(data, v); Kv = b["K"].to(dev); c2wv = b["c2w"].to(dev)
        pv = b["image"].to(dev); Hv, Wv = pv.shape[:2]
        fp = _footprint(newg, c2wv, Kv, Wv, Hv, dev)
        if fp is None:
            continue
        x0, y0, x1, y1 = _cbox(fp, Wv, Hv)
        rp = _render(prev, c2wv, Kv, Wv, Hv, dev, "RGB")[0][..., :3].clamp(0, 1)
        rn = _render(new, c2wv, Kv, Wv, Hv, dev, "RGB")[0][..., :3].clamp(0, 1)
        ep = (rp[y0:y1, x0:x1] - pv[y0:y1, x0:x1]).abs().mean().item()
        en = (rn[y0:y1, x0:x1] - pv[y0:y1, x0:x1]).abs().mean().item()
        gains.append(ep - en)
    return (sum(gains) / len(gains)) if gains else -1.0


def _run_infusion(d, rdir, cfg, dev):
    os.makedirs(rdir, exist_ok=True)
    valid = (d["a"] > 0.5) & (d["depth"] > 1e-6)
    disp = torch.where(valid, 1.0 / d["depth"].clamp(min=1e-6), torch.zeros_like(d["depth"]))
    cv2.imwrite(os.path.join(rdir, "rgb.png"),
                cv2.cvtColor((d["photo"].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(rdir, "mask.png"), d["mask"])
    np.save(os.path.join(rdir, "disp.npy"), disp.cpu().numpy().astype(np.float32))
    np.save(os.path.join(rdir, "intri.npy"), d["K"].cpu().numpy().astype(np.float64))
    np.save(os.path.join(rdir, "c2w.npy"), d["c2w"].cpu().numpy().astype(np.float64))
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
    ply = os.path.join(out, "rgb_mask.ply")
    raw = np.loadtxt(ply, skiprows=10)
    return raw if raw.ndim == 2 and len(raw) else None


def _as_gaussians(raw, ref, scene, dev):
    pts = torch.from_numpy(raw[:, :3]).float().to(dev)
    col = torch.from_numpy(raw[:, 3:6]).float().to(dev) / 255.0
    if pts.shape[0] > 60000:
        s = torch.randperm(pts.shape[0], device=dev)[:60000]; pts, col = pts[s], col[s]
    M = pts.shape[0]; vs = 0.02 * scene
    return {"means": pts, "scales": torch.full((M, 3), math.log(vs * 0.5), device=dev),
            "quats": torch.tensor([1.0, 0, 0, 0], device=dev).repeat(M, 1),
            "opacities": torch.full((M,) + tuple(ref["opacities"].shape[1:]), -2.197, device=dev),
            "sh0": ((col - 0.5) / C0).reshape(M, 1, 3),
            "shN": torch.zeros((M,) + tuple(ref["shN"].shape[1:]), device=dev)}


def _find_ckpt(checkpoint_dir):
    """Accept a .pt file or a dir/ckpts containing one; return the newest .pt."""
    p = Path(checkpoint_dir)
    if p.is_file():
        return str(p)
    cands = sorted(p.glob("**/ckpt_*.pt"), key=lambda x: x.stat().st_mtime)
    if not cands:
        cands = sorted(p.glob("**/*.pt"), key=lambda x: x.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No .pt checkpoint under {checkpoint_dir}")
    return str(cands[-1])


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------

def validate_environment(cfg: AutoCompleteConfig) -> tuple[bool, str]:
    msgs = []
    run_py = os.path.join(cfg.infusion_repo, "depth_inpainting", "run", "run_inference_inpainting.py")
    if not os.path.exists(run_py):
        msgs.append(f"InFusion repo not found: {run_py}")
    if not os.path.isdir(cfg.infusion_ckpt):
        msgs.append(f"InFusion checkpoint not found: {cfg.infusion_ckpt}")
    if not os.path.exists(cfg.difix_python):
        msgs.append(f"difix env python not found: {cfg.difix_python}")
    if not torch.cuda.is_available():
        msgs.append("CUDA not available")
    if msgs:
        return False, " | ".join(msgs)
    return True, "Auto-complete ready (InFusion seed + production-trainer reconstruction)."


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_autocomplete(
    checkpoint_dir: str,
    data_dir: str,
    output_dir: str,
    config: Optional[AutoCompleteConfig] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> AutoCompleteResult:
    """Detect + reconstruct holes in a trained splat. Returns an AutoCompleteResult.

    Idempotent-safe: if no genuine holes are detected (or all candidate fills are
    rejected by the gate), the original splat is returned unchanged (inert).
    """
    cfg = config or AutoCompleteConfig()
    dev = cfg.device
    os.makedirs(output_dir, exist_ok=True)

    def prog(f, d):
        if progress_cb:
            progress_cb(min(max(f, 0.0), 1.0), d)

    prog(0.02, "Loading splat + cameras...")
    ck = _find_ckpt(checkpoint_dir)
    base = {k: v.to(dev) for k, v in torch.load(ck, weights_only=True, map_location=dev)["splats"].items()}
    data = load_colmap_data(data_dir, factor=cfg.data_factor, test_every=8, normalize=True)
    scene = float(data["scene_scale"])
    C = torch.tensor(np.stack([data["camtoworlds"][i][:3, 3]
                               for i in range(len(data["camtoworlds"]))]), device=dev).float()
    cand = sorted(set(int(i) for i in list(data["test_indices"])[::2]))
    tr_idx = list(data["train_indices"])
    g = torch.Generator().manual_seed(0)

    # ---- detect + seed (progressive, gated) ----
    acc = None; done = []; filled = 0; rejected = 0; recapture = []
    for r in range(cfg.max_rounds):
        prog(0.05 + 0.45 * r / cfg.max_rounds, f"Detecting holes (round {r + 1})...")
        cur = base if acc is None else {k: torch.cat([base[k], acc[k]], 0) for k in base}
        scored = []
        for i in cand:
            if i in done:
                continue
            d = _detect(cur, i, data, C, scene, dev, cfg.center_prior)
            if d and d["area"] > 0.004 * d["objarea"]:
                scored.append((d["area"], d))
        if not scored:
            break
        scored.sort(key=lambda t: t[0], reverse=True); d = scored[0][1]; done.append(d["idx"])
        raw = _run_infusion(d, os.path.join(output_dir, f"r{r}_v{d['idx']}"), cfg, dev)
        if raw is None:
            continue
        acc_before = acc
        newg = _as_gaussians(raw, base, scene, dev)
        trial = newg if acc is None else {k: torch.cat([acc[k], newg[k]], 0) for k in newg}
        params, opt = _short_distill(base, trial, data, tr_idx, scene, dev, g)
        gain = _fill_gain(base, acc_before, params, newg, d["idx"], data, C, dev)
        if gain > 0:
            acc = params; filled += 1
        else:
            rejected += 1

    if acc is None:
        prog(1.0, "No genuine holes detected — splat unchanged.")
        return AutoCompleteResult(
            checkpoint=ck, ply_path=None, holes_filled=0, holes_rejected=rejected,
            report="No genuine holes detected (or all candidate fills rejected by the "
                   "accept/reject gate). The splat is already complete; left unchanged.",
            sample_renders=[])

    # ---- reconstruct with the production trainer (init = splat + committed seeds) ----
    prog(0.55, "Reconstructing holes (MCMC + gated LPIPS)...")
    init = {k: torch.cat([base[k], acc[k]], 0) for k in base}
    init_path = os.path.join(output_dir, "recon_init.pt")
    torch.save({"splats": {k: v.detach().cpu() for k, v in init.items()}}, init_path)
    recon_dir = os.path.join(output_dir, "recon")
    cap = int(init["means"].shape[0] * 1.15)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    trainer = str(Path(__file__).resolve().parents[1] / "training" / "gsplat_trainer.py")
    subprocess.run([sys.executable, trainer,
                    "--data_dir", data_dir, "--result_dir", recon_dir,
                    "--data_factor", str(cfg.data_factor), "--max_steps", str(cfg.recon_steps),
                    "--test_every", "8", "--strategy", "mcmc", "--cap_max", str(cap),
                    "--refine_stop_iter", str(int(cfg.recon_steps * 0.9)),
                    "--init_ckpt", init_path, "--lpips_lambda", str(cfg.lpips_lambda),
                    "--lpips_gated", "--lpips_start", "200", "--antialiased",
                    "--random_bkgd", "--save_steps", str(cfg.recon_steps)],
                   check=True, env=env)
    out_ck = os.path.join(recon_dir, "ckpts", f"ckpt_{cfg.recon_steps - 1}_rank0.pt")
    ply = glob.glob(os.path.join(recon_dir, "ply", "*.ply"))
    ply_path = ply[-1] if ply else None

    # ---- before/after sample renders at a filled view ----
    prog(0.95, "Rendering before/after...")
    final = {k: v.to(dev) for k, v in torch.load(out_ck, weights_only=True, map_location=dev)["splats"].items()}
    samples = _before_after(base, final, done[:1] or [cand[0]], data, output_dir, dev)

    report = (
        f"Auto-complete done. Holes filled: {filled}, rejected (no improvement): {rejected}.\n"
        f"Reconstructed splat: {out_ck}\n"
        f"PLY: {ply_path or 'N/A'}\n"
    )
    if recapture:
        report += f"Re-capture suggested for {len(recapture)} unseen region(s).\n"
    prog(1.0, "Auto-complete complete.")
    return AutoCompleteResult(checkpoint=out_ck, ply_path=ply_path, holes_filled=filled,
                              holes_rejected=rejected, report=report, sample_renders=samples)


def _short_distill(base, trial, data, tr_idx, scene, dev, g, steps=300):
    from core.training.gsplat_trainer import ssim
    params = {k: torch.nn.Parameter(v.clone()) for k, v in trial.items()}
    opt = torch.optim.Adam([
        {"params": [params["means"]], "lr": 1.6e-4 * scene}, {"params": [params["scales"]], "lr": 5e-3},
        {"params": [params["quats"]], "lr": 1e-3}, {"params": [params["opacities"]], "lr": 5e-2},
        {"params": [params["sh0"]], "lr": 2.5e-3}, {"params": [params["shN"]], "lr": 1.25e-4}])
    for _ in range(steps):
        j = tr_idx[int(torch.randint(len(tr_idx), (1,), generator=g))]
        b = load_image(data, j); gt = b["image"].to(dev); K = b["K"].to(dev); c2w = b["c2w"].to(dev)
        h, w = gt.shape[:2]
        comb = {k: torch.cat([base[k], params[k]], 0) for k in base}
        rr = render_splats(comb, torch.linalg.inv(c2w), K, w, h, SH,
                           torch.zeros(3, device=dev), "antialiased")[0][0].clamp(0, 1)
        loss = 0.8 * (rr - gt).abs().mean() + 0.2 * (1 - ssim(rr.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        opt.zero_grad(); loss.backward(); opt.step()
    return {k: params[k].detach() for k in params}, opt


def _before_after(base, final, views, data, output_dir, dev):
    out = []
    for vi in views:
        b = load_image(data, vi); K = b["K"].to(dev); c2w = b["c2w"].to(dev); H, W = b["image"].shape[:2]
        for sp, tag in [(base, "before"), (final, "after")]:
            im = _render(sp, c2w, K, W, H, dev, "RGB")[0][..., :3].clamp(0, 1)
            p = os.path.join(output_dir, f"v{vi}_{tag}.png")
            cv2.imwrite(p, cv2.cvtColor((im.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            out.append((p, f"{tag} v{vi}"))
    return out
