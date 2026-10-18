"""QA report (WP-Q1 CLI) — visibility-split quality scorecard for a trained splat.

Renders the held-out views of a scene from a checkpoint, builds observed/unobserved
visibility masks (OGGSplat protocol), computes masked PSNR/SSIM/LPIPS on the
observed region only, and writes a per-scene scorecard (JSON) + an HTML strip
(render | mask-overlay | GT | per-view metrics) for eyeballing.

Deterministic: fixed seeds + sorted view iteration + no random crops.  Running the
same command twice produces a byte-identical scorecard.json (a WP gate requirement).

Usage:
  python qa_report.py <ckpt.pt|ply> <data_dir> [--factor 2]
        [--split every8|wedge:START:END] [--out qa/<scene>/]
        [--sh_degree auto] [--alpha_thresh 0.3] [--lpips_mode bbox|tiled]
        [--vis_view_count] [--max_views N] [--thumb 512] [--seed 0]
        [--inject_floaters N] [--blur_sigma S]        # corruption sensitivity check

Examples:
  python qa_report.py _dataset/GPU_V4/apex_images/training/gsplat_prod/output/ckpts/ckpt_29999_rank0.pt \
                      _dataset/GPU_V4/apex_images/training/gsplat_prod/data --factor 2 --out qa/GPU_V4
  # corruption A/B:
  python qa_report.py <ckpt> <data> --inject_floaters 2000 --out qa/GPU_V4_floaters
  python qa_report.py <ckpt> <data> --blur_sigma 2       --out qa/GPU_V4_blur
"""
import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ensure `core` importable when run from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.qa.visibility import (
    VisibilityConfig, render_alpha, build_visibility_mask,
    per_gaussian_obs_counts, render_undersampled_map,
)
from core.qa.masked_metrics import masked_psnr, masked_ssim, masked_lpips
from core.qa.scorecard import Scorecard, ViewScore, SceneContext, write_html


# ---------------------------------------------------------------------------
# Splat loading (.pt checkpoint  or  .ply — inverse of gsplat export_splats)
# ---------------------------------------------------------------------------

def load_splats(path: str, device: str) -> dict:
    """Load gaussian params into the trainer's splats dict layout:
    means(N,3) opacities(N,) quats(N,4) scales(N,3) sh0(N,1,3) shN(N,K,3)."""
    p = str(path).lower()
    if p.endswith(".ply"):
        return _load_ply(path, device)
    if p.endswith(".spz"):                       # WP-D2: SPZ v4 (Niantic bindings)
        from core.spz_io import read_spz
        return {k: v.to(device).float() for k, v in read_spz(path).items()}
    ck = torch.load(path, map_location=device, weights_only=False)
    sp = ck["splats"] if isinstance(ck, dict) and "splats" in ck else ck
    return {k: v.to(device).float() for k, v in sp.items()}


_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _read_ply_vertex(path: str) -> dict:
    """Minimal PLY reader (pure numpy — no plyfile dependency). Reads the `vertex`
    element into {property_name: np.ndarray}. Supports binary little/big endian and
    ascii; all standard scalar property types."""
    with open(path, "rb") as f:
        raw = f.read()
    hdr_end = raw.find(b"end_header\n")
    if hdr_end < 0:
        raise ValueError(f"{path}: no PLY end_header")
    header = raw[:hdr_end].decode("ascii", "replace").splitlines()
    body = raw[hdr_end + len(b"end_header\n"):]

    fmt = "ascii"
    n = 0
    in_vertex = False
    props = []  # (name, typechar)
    for line in header:
        t = line.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            in_vertex = (t[1] == "vertex")
            if in_vertex:
                n = int(t[2])
        elif t[0] == "property" and in_vertex:
            # scalar property: "property <type> <name>"
            props.append((t[2], _PLY_TYPES[t[1]]))

    if fmt.startswith("binary"):
        endian = "<" if "little" in fmt else ">"
        dt = np.dtype([(nm, endian + tc) for nm, tc in props])
        arr = np.frombuffer(body, dtype=dt, count=n)
        return {nm: np.asarray(arr[nm]) for nm, _ in props}
    # ascii
    vals = np.array(body.decode("ascii", "replace").split(), dtype=np.float64)
    vals = vals.reshape(n, len(props))
    return {nm: vals[:, i] for i, (nm, _) in enumerate(props)}


def _load_ply(path: str, device: str) -> dict:
    """Read a standard 3DGS PLY. gsplat's exporter writes f_rest channel-major
    (shN.permute(0,2,1).reshape(N,-1)); we invert that exactly."""
    v = _read_ply_vertex(path)
    names = set(v.keys())

    def col(n):
        return np.asarray(v[n], dtype=np.float32)

    means = np.stack([col("x"), col("y"), col("z")], axis=1)
    N = means.shape[0]

    sh0 = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=1)  # (N,3)
    sh0 = sh0[:, None, :]                                                  # (N,1,3)

    rest = sorted((n for n in names if n.startswith("f_rest_")),
                  key=lambda s: int(s.split("_")[-1]))
    if rest:
        fr = np.stack([col(n) for n in rest], axis=1)                     # (N, 3K)
        K = fr.shape[1] // 3
        shN = fr.reshape(N, 3, K).transpose(0, 2, 1)                      # (N,K,3)
    else:
        shN = np.zeros((N, 0, 3), dtype=np.float32)

    opac = col("opacity")                                                 # (N,) raw logit
    scales = np.stack([col(f"scale_{i}") for i in range(3)], axis=1)      # (N,3) raw log
    rot = sorted((n for n in names if n.startswith("rot")),
                 key=lambda s: int(s.split("_")[-1]))
    quats = np.stack([col(n) for n in rot], axis=1)                       # (N,4)

    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(device)
    return {"means": to(means), "opacities": to(opac), "quats": to(quats),
            "scales": to(scales), "sh0": to(sh0), "shN": to(shN)}


# ---------------------------------------------------------------------------
# Corruption knobs (default off) — for the WP validation sensitivity check.
# Single source of truth: core/qa/corruptions.py (shared with the phase-2
# scorer clip renderers, so there is exactly one copy of the math).
# ---------------------------------------------------------------------------

from core.qa.corruptions import inject_floaters, gaussian_blur as blur_render


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_scene(data_dir: str) -> str:
    parts = Path(data_dir).resolve().parts
    if "_dataset" in parts:
        i = parts.index("_dataset")
        if i + 1 < len(parts):
            return parts[i + 1]
    return Path(data_dir).resolve().parent.name


def _thumb(img_uint8_bgr: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = img_uint8_bgr.shape[:2]
    sc = min(1.0, long_edge / max(h, w))
    if sc < 1.0:
        img_uint8_bgr = cv2.resize(img_uint8_bgr, (round(w * sc), round(h * sc)),
                                   interpolation=cv2.INTER_AREA)
    return img_uint8_bgr


def to_bgr(t: torch.Tensor) -> np.ndarray:
    rgb = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return np.ascontiguousarray(rgb[..., ::-1])   # RGB->BGR, contiguous for cv2


def _render_probe(splats, data, idx, sh_degree, device):
    """Render view `idx` and return (full-frame PSNR vs GT, mean luminance).
    Used by --normalize auto (frame pick) and the always-on sanity check."""
    from core.training.gsplat_trainer import load_image, render_splats
    b = load_image(data, idx)
    gt = b["image"].to(device)
    H, W = gt.shape[:2]
    viewmat = torch.linalg.inv(b["c2w"].to(device))
    with torch.no_grad():
        renders, _, _ = render_splats(splats, viewmat, b["K"].to(device), W, H,
                                      sh_degree, torch.zeros(3, device=device),
                                      "antialiased")
        render = renders[0].clamp(0, 1)
    mse = float(((render - gt) ** 2).mean())
    psnr = 10.0 * float(np.log10(1.0 / (mse + 1e-10)))
    return psnr, float(render.mean())


def _first_split_index(data, wedge) -> int:
    """First held-out view index for the current split (same math as main)."""
    if wedge is not None:
        n = len(data["image_names"])
        lo = int(wedge[0] * n) if wedge[0] <= 1.0 else int(wedge[0])
        return max(0, min(n - 1, lo))
    return int(sorted(int(i) for i in data["test_indices"])[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="WP-Q1 QA scorecard for a trained splat")
    ap.add_argument("ckpt", help="checkpoint .pt or .ply")
    ap.add_argument("data_dir", help="COLMAP data dir (has sparse/, images_<factor>/)")
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--split", default="every8",
                    help="every8 | wedge:START:END — fractions (wedge:0.40:0.55) "
                         "or absolute name-sorted view indices (wedge:131:170)")
    ap.add_argument("--out", default=None, help="output dir (default qa/<scene>/)")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--sh_degree", type=int, default=None,
                    help="SH degree to render at. Default: the model's own degree (from its "
                         "shN / f_rest count) — 3 for every production model, so unchanged "
                         "there; a degree-4 model was previously scored TRUNCATED to 3 "
                         "(WP-SH4). Pass a lower value to score a truncation on purpose.")
    ap.add_argument("--alpha_thresh", type=float, default=0.3)
    ap.add_argument("--lpips_mode", choices=["bbox", "tiled"], default="bbox")
    ap.add_argument("--vis_view_count", action="store_true",
                    help="secondary signal: also flag pixels dominated by "
                         "<k-training-view Gaussians (heavy; default off)")
    ap.add_argument("--view_count_k", type=int, default=3)
    ap.add_argument("--max_views", type=int, default=0, help="cap held-out views (0=all)")
    ap.add_argument("--thumb", type=int, default=512, help="thumbnail long edge (px)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--inject_floaters", type=int, default=0,
                    help="corruption: add N random floater Gaussians (default 0)")
    ap.add_argument("--floater_scale", type=float, default=0.01,
                    help="floater blob size as fraction of robust scene diagonal")
    ap.add_argument("--blur_sigma", type=float, default=0.0,
                    help="corruption: gaussian-blur renders, sigma px (default 0)")
    ap.add_argument("--floater_jitter", action="store_true",
                    help="corruption: re-randomize the injected floaters' "
                         "POSITIONS on every frame of the scorer clips (temporal "
                         "popping — positive control for raft_warp/flicker). "
                         "Requires --inject_floaters > 0.")
    ap.add_argument("--normalize", choices=["auto", "true", "false"], default="true",
                    help="pose frame for load_colmap_data. 'true' (default, "
                         "unchanged behavior) = the trainer's NORMALIZED frame — "
                         "correct for this project's own checkpoints. 'false' = "
                         "raw COLMAP frame — required for foreign PLYs (e.g. "
                         "3DGRUT exports), which render GARBAGE under normalized "
                         "poses. 'auto' = probe the first held-out view under "
                         "both frames and keep the higher-PSNR one.")
    ap.add_argument("--scorers", default="",
                    help="phase-2 external scorers to run (default OFF). Comma "
                         "list from {crossscore,nova,tsed,raft_warp,nvs_sqa,"
                         "flicker} or 'all'. Empty = phase-1 behavior unchanged.")
    ap.add_argument("--qa_python", default=None,
                    help="python.exe of the qa venv (subprocess scorers). "
                         "Default: D:/REPOS/HYPERclone/venvs/qa/Scripts/python.exe")
    args = ap.parse_args()
    if args.floater_jitter and args.inject_floaters <= 0:
        ap.error("--floater_jitter requires --inject_floaters > 0")

    # determinism
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    # split: every8 (default) or a contiguous wedge of the name-sorted trajectory.
    # Wedge bounds may be FRACTIONS (<=1.0, matches the trainer's APEX_HOLDOUT math)
    # or ABSOLUTE view indices (>1, e.g. wedge:131:170 = views [131,170) ). The
    # wedge is applied post-load here (no env round-trip); APEX_HOLDOUT is cleared
    # so a stale env var can never leak into the split.
    wedge = None
    if args.split.startswith("wedge:"):
        _, s, e = args.split.split(":")
        wedge = (float(s), float(e))
    os.environ.pop("APEX_HOLDOUT", None)
    test_every = 8

    from core.training.gsplat_trainer import load_colmap_data, load_image, render_splats

    scene = args.scene or infer_scene(args.data_dir)
    out_dir = Path(args.out) if args.out else Path("qa") / scene
    views_dir = out_dir / "views"
    views_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"[QA] scene={scene}  split={args.split}  factor={args.factor}  out={out_dir}")
    print(f"[QA] loading splats: {args.ckpt}")
    splats = load_splats(args.ckpt, device)
    print(f"[QA] {splats['means'].shape[0]:,} Gaussians")
    from core.sh_utils import sh_degree_of
    model_deg = sh_degree_of(splats)
    if args.sh_degree is None:
        args.sh_degree = model_deg
    elif args.sh_degree > model_deg:
        ap.error(f"--sh_degree {args.sh_degree} > the model's SH degree {model_deg}")
    print(f"[QA] SH degree: rendering {args.sh_degree} (model has {model_deg})")

    corruption = {}
    if args.inject_floaters > 0:
        splats = inject_floaters(splats, args.inject_floaters, seed=args.seed,
                                 scale_frac=args.floater_scale)
        # record ALL corruption params so scorecards are self-describing
        corruption["inject_floaters"] = args.inject_floaters
        corruption["floater_scale"] = args.floater_scale
        corruption["floater_seed"] = args.seed
        if args.floater_jitter:
            # phase-1 held-out views keep the STATIC injection above; the phase-2
            # clip renderers re-randomize floater positions per frame (popping).
            corruption["floater_jitter"] = True
        print(f"[QA] injected {args.inject_floaters} floaters "
              f"(scale_frac={args.floater_scale}"
              f"{', jitter per clip frame' if args.floater_jitter else ''}) -> "
              f"{splats['means'].shape[0]:,} Gaussians")
    if args.blur_sigma > 0:
        corruption["blur_sigma"] = args.blur_sigma
        print(f"[QA] renders will be gaussian-blurred (sigma={args.blur_sigma})")

    # pose frame: our own ckpts live in the trainer's NORMALIZED frame (true);
    # foreign PLYs (3DGRUT exports etc.) live in the raw COLMAP frame (false).
    # auto probes the first held-out view under both and keeps the better one.
    if args.normalize == "auto":
        best = None
        for nm in (True, False):
            d_try = load_colmap_data(args.data_dir, factor=args.factor,
                                     test_every=test_every, normalize=nm)
            p_idx = _first_split_index(d_try, wedge)
            psnr, lum = _render_probe(splats, d_try, p_idx, args.sh_degree, device)
            print(f"[QA] normalize probe: normalize={nm} -> view {p_idx} "
                  f"PSNR {psnr:.2f} dB, mean lum {lum:.3f}")
            if best is None or psnr > best[0]:
                best = (psnr, nm, d_try)
        normalize_used = best[1]
        data = best[2]
        print(f"[QA] --normalize auto resolved to {str(normalize_used).lower()}")
    else:
        normalize_used = (args.normalize == "true")
        data = load_colmap_data(args.data_dir, factor=args.factor,
                                test_every=test_every, normalize=normalize_used)
    if wedge is not None:
        n = len(data["image_names"])
        lo = int(wedge[0] * n) if wedge[0] <= 1.0 else int(wedge[0])
        hi = int(wedge[1] * n) if wedge[1] <= 1.0 else int(wedge[1])
        lo, hi = max(0, lo), min(n, hi)
        test_indices = list(range(lo, hi))
        print(f"[QA] wedge split: views [{lo}, {hi}) of {n}")
    else:
        test_indices = sorted(int(i) for i in data["test_indices"])
    if args.max_views > 0:
        test_indices = test_indices[:args.max_views]
    print(f"[QA] scoring {len(test_indices)} held-out views")

    vcfg = VisibilityConfig(alpha_thresh=args.alpha_thresh,
                            use_view_count=args.vis_view_count,
                            view_count_k=args.view_count_k)

    # secondary per-Gaussian view-count signal (optional, heavy) — computed once
    n_obs = None
    if args.vis_view_count:
        print("[QA] computing per-Gaussian training-view counts (secondary signal)...")
        n_obs = per_gaussian_obs_counts(splats, data, device)

    view_scores = []
    psnrs, ssims, lpipss, obs_pcts = [], [], [], []
    render_sanity = "ok"

    for vi, idx in enumerate(test_indices):
        b = load_image(data, idx)
        gt = b["image"].to(device)
        K = b["K"].to(device)
        c2w = b["c2w"].to(device)
        H, W = gt.shape[:2]
        viewmat = torch.linalg.inv(c2w)

        with torch.no_grad():
            renders, alphas, _ = render_splats(
                splats, viewmat, K, W, H, args.sh_degree,
                torch.zeros(3, device=device), vcfg.rasterize_mode)
            render = renders[0].clamp(0, 1)
            alpha = alphas[0, ..., 0]

        # ALWAYS-ON degenerate-render sanity check (first view, pre-blur): a
        # near-black frame or render-vs-GT PSNR < 9 dB almost always means the
        # splat lives in the OTHER pose frame (e.g. a raw-COLMAP-frame 3DGRUT
        # PLY loaded under normalize=true). Thresholds are EMPIRICAL from the
        # live footgun: the 3DGRUT wedge PLY under the wrong frame renders a
        # BRIGHT garbage blob (lum 0.45 — the black-frame arm alone misses it)
        # at 7.7-7.8 dB, while the heaviest banked intentional corruption
        # (floaters@0.01) stays at 12.5 dB -> 9.0 splits them with margin.
        # Warn LOUDLY + self-flag the scorecard; never error.
        if vi == 0:
            _mse = float(((render - gt) ** 2).mean())
            _fpsnr = 10.0 * float(np.log10(1.0 / (_mse + 1e-10)))
            _lum = float(render.mean())
            if _lum < 0.01 or _fpsnr < 9.0:
                render_sanity = "suspect"
                bang = "!" * 78
                print(f"\n{bang}\n"
                      f"[QA] !! RENDER SANITY SUSPECT: first held-out render vs GT "
                      f"PSNR {_fpsnr:.2f} dB, mean luminance {_lum:.4f}.\n"
                      f"[QA] !! The splat likely lives in the OTHER pose frame "
                      f"(current --normalize {args.normalize} -> "
                      f"{str(normalize_used).lower()}).\n"
                      f"[QA] !! For foreign PLYs (e.g. 3DGRUT exports) re-run with "
                      f"--normalize false. Scorecard is flagged "
                      f"render_sanity=suspect.\n{bang}\n", flush=True)

        secondary = None
        if n_obs is not None:
            umap = render_undersampled_map(splats, n_obs, viewmat, K, W, H, device,
                                           k=args.view_count_k)
            secondary = (umap.detach().cpu().numpy() > vcfg.view_count_frac)

        vis = build_visibility_mask(alpha, vcfg, secondary_unobserved=secondary)
        observed = torch.from_numpy(vis["observed"]).to(device)

        # corruption: blur only the RGB used for metrics (mask stays model-derived)
        render_metric = blur_render(render, args.blur_sigma) if args.blur_sigma > 0 else render

        mp = masked_psnr(render_metric, gt, observed)
        ms = masked_ssim(render_metric, gt, observed)
        ml = masked_lpips(render_metric, gt, observed, device, mode=args.lpips_mode)

        # thumbnails + overlay (red tint on unobserved)
        r_bgr = to_bgr(render_metric)
        g_bgr = to_bgr(gt)
        overlay = r_bgr.copy()
        um = vis["unobserved"]
        overlay[um] = (0.5 * overlay[um] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        name = data["image_names"][idx]
        stem = f"v{idx:04d}"
        rp = views_dir / f"{stem}_render.jpg"
        gp = views_dir / f"{stem}_gt.jpg"
        op = views_dir / f"{stem}_overlay.png"
        cv2.imwrite(str(rp), _thumb(r_bgr, args.thumb), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(gp), _thumb(g_bgr, args.thumb), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(op), _thumb(overlay, args.thumb))

        view_scores.append(ViewScore(
            index=idx, name=name, observed_pct=vis["observed_pct"],
            masked_psnr=mp, masked_ssim=ms, masked_lpips=ml,
            render=f"views/{rp.name}", gt=f"views/{gp.name}",
            overlay=f"views/{op.name}"))
        psnrs.append(mp); ssims.append(ms); lpipss.append(ml)
        obs_pcts.append(vis["observed_pct"])
        if vi % 10 == 0 or vi == len(test_indices) - 1:
            print(f"  [{vi + 1}/{len(test_indices)}] {name}  obs {vis['observed_pct']:.1f}%  "
                  f"PSNR {mp:.2f}  SSIM {ms:.4f}  LPIPS {ml:.4f}", flush=True)

    card = Scorecard(
        scene=scene, ckpt=str(args.ckpt), data_dir=str(args.data_dir),
        factor=args.factor, split=args.split, n_views=len(test_indices),
        observed_area_pct=float(np.nanmean(obs_pcts)) if obs_pcts else float("nan"),
        masked_psnr=float(np.nanmean(psnrs)) if psnrs else float("nan"),
        masked_ssim=float(np.nanmean(ssims)) if ssims else float("nan"),
        masked_lpips=float(np.nanmean(lpipss)) if lpipss else float("nan"),
        alpha_thresh=args.alpha_thresh,
        normalize=str(normalize_used).lower(), render_sanity=render_sanity,
        corruption=corruption, views=view_scores)

    # SceneContext hands the (possibly-corrupted) splats + poses to phase-2 scorers.
    _ctx = SceneContext(
        scene=scene, ckpt=str(args.ckpt), data_dir=str(args.data_dir),
        factor=args.factor, sh_degree=args.sh_degree, device=device,
        out_dir=str(out_dir), data=data, splats=splats,
        test_indices=test_indices, views=view_scores, corruption=corruption)

    # Phase-2 external scorers (default OFF behind --scorers). In-process scorers
    # (tsed/raft_warp/flicker) run here in the main env; subprocess scorers
    # (crossscore/nova/nvs_sqa) shell out to the qa venv. Each result dict gets a
    # "runtime_s"; a scorer failure is recorded, never fatal.
    if args.scorers:
        import json as _json
        from core.qa.scorecard import run_scorers
        from core.qa.scorers import build_scorers, ScorerConfig
        names = [s.strip() for s in args.scorers.split(",") if s.strip()]
        cfg = ScorerConfig()
        if args.qa_python:
            cfg.qa_python = args.qa_python
        scorers = build_scorers(names, cfg)
        # Merge: preserve previously-computed scorer values in this out_dir so a
        # targeted re-run (e.g. --scorers nvs_sqa after a fix) never wipes the
        # heavy ones — but ONLY if the prior scorecard's corruption matches this
        # run's exactly. On mismatch the prior values were measured on a different
        # model state: refuse to merge (fresh scorecard) and say why.
        prior_path = out_dir / "scorecard.json"
        if prior_path.exists():
            try:
                prior = _json.loads(prior_path.read_text(encoding="utf-8"))
                prior_corr = prior.get("corruption") or {}
                cur_corr = _json.loads(_json.dumps(corruption))  # JSON-normalized
                if prior_corr == cur_corr:
                    for k, v in (prior.get("scorers") or {}).items():
                        if v is not None and k not in names:
                            card.scorers[k] = v
                else:
                    print(f"[QA] NOT merging prior scorer values from {prior_path}: "
                          f"corruption mismatch (prior={prior_corr} vs "
                          f"current={cur_corr}) — starting a fresh scorecard.")
            except Exception as e:
                print(f"[QA] prior scorecard unreadable ({e}); fresh scorecard.")
        print(f"[QA] running phase-2 scorers: {[s.name for s in scorers]}")
        run_scorers(card, _ctx, scorers)

    json_path = out_dir / "scorecard.json"
    html_path = out_dir / "report.html"
    card.write_json(json_path)
    write_html(card, html_path)

    print(f"\n[QA] === {scene} ({args.split}) ===")
    print(f"  observed area   {card.observed_area_pct:.1f}%")
    print(f"  masked PSNR     {card.masked_psnr:.3f}")
    print(f"  masked SSIM     {card.masked_ssim:.4f}")
    print(f"  masked LPIPS    {card.masked_lpips:.4f}  (lower=better)")
    print(f"  normalize       {card.normalize}   render_sanity {card.render_sanity}")
    if corruption:
        print(f"  corruption      {corruption}")
    print(f"[QA] wrote {json_path}")
    print(f"[QA] wrote {html_path}  (open in a browser)")


if __name__ == "__main__":
    main()
