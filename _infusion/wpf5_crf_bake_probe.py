"""Bake a PPISP camera response curve (CRF) into a checkpoint's Gaussian colours -> standard PLY.

Why (WP-F5): with `--ppisp` the Gaussians learn the CANONICAL radiance that sits *before* the
learned ISP (exposure -> vignetting -> colour homography -> per-channel CRF). A viewer (Postshot,
SuperSplat) shows that radiance as-is, i.e. without the camera's tone curve, so the PLY can look
flatter/darker than the photos. Per-frame exposure and colour are mean-anchored to identity by
PPISP's own regularizer, and vignetting is a lens effect we do NOT want in a free-viewpoint asset,
so the only systematic gap is the CRF. This tool applies it per Gaussian:

  c   = 0.5 + C0 * sh0                (view-independent colour, as the rasterizer computes it)
  c'  = CRF(clamp(c, 0, 1))           (PPISP toe/shoulder/gamma curve of one camera, per channel)
  sh0'= (c' - 0.5) / C0
  shN'= shN * dCRF/dc (at c)          (first-order: view-dependent lobes scaled by the local slope)

The CRF is applied to each Gaussian's colour rather than to the composited pixel, so it is exact
for opaque single-Gaussian pixels and first-order elsewhere — measure it (WP-F5 tone eval) before
trusting it. The CRF formula is PPISP's own (nv-tlabs/ppisp tests/torch_reference.py, Apache-2.0),
re-written here without its custom dead-zone autograd (no gradients needed).

  python _infusion/wpf5_crf_bake_probe.py --ckpt <run>/ckpts/ckpt_29999_rank0.pt --out <run>/ply/baked.ply       [--out_ckpt baked.pt]   # ckpt form for _infusion/wpf5_tone_eval.py

RESULT (2026-10-18, Laptop `wpf5/pp`, `_infusion/wpf5/laptop_tone_crfbaked.json`): NEGATIVE — do not
ship. Held-out raw LPIPS 0.2805 -> 0.2863 and PSNR 22.69 -> 22.23 after baking; the baked PLY is
BRIGHTER than the photos (mean RGB 0.519 vs GT 0.507) because PPISP also learned vignetting
(alpha_0 ~ -0.28: photo corners darker), which is image-space and cannot be baked; the CRF's
mid-tone lift then overshoots. The un-baked canonical radiance (mean 0.497) is the closer one.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

C0 = 0.28209479177387814


def crf_apply(x: torch.Tensor, crf: torch.Tensor) -> torch.Tensor:
    """x [N, 3] in [0, 1]; crf [3, 4] raw (toe, shoulder, gamma, center) -> [N, 3]."""
    out = []
    for ch in range(3):
        toe = 0.3 + F.softplus(crf[ch, 0])
        shoulder = 0.3 + F.softplus(crf[ch, 1])
        gamma = 0.1 + F.softplus(crf[ch, 2])
        center = torch.sigmoid(crf[ch, 3])
        lerp = toe + center * (shoulder - toe)
        a = (shoulder * center) / lerp
        b = 1.0 - a
        v = x[:, ch]
        y_low = a * torch.pow((v / center).clamp_min(0), toe)
        y_high = 1.0 - b * torch.pow(((1.0 - v) / (1.0 - center)).clamp_min(0), shoulder)
        y = torch.where(v <= center, y_low, y_high).clamp_min(0)
        out.append(torch.pow(y, gamma))
    return torch.stack(out, -1)


def bake(splats: dict, crf: torch.Tensor, eps: float = 1e-3) -> dict:
    sh0 = splats["sh0"].float()                      # [N, 1, 3]
    c = (0.5 + C0 * sh0[:, 0]).clamp(0, 1)           # [N, 3]
    c2 = crf_apply(c, crf)
    slope = (crf_apply((c + eps).clamp(0, 1), crf) - crf_apply((c - eps).clamp(0, 1), crf)) \
        / ((c + eps).clamp(0, 1) - (c - eps).clamp(0, 1)).clamp_min(1e-6)
    out = dict(splats)
    out["sh0"] = ((c2 - 0.5) / C0)[:, None, :]
    out["shN"] = splats["shN"].float() * slope[:, None, :]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="baked PLY")
    ap.add_argument("--out_ckpt", default=None, help="baked splats as a {'step','splats'} ckpt")
    ap.add_argument("--camera", type=int, default=0, help="PPISP camera index whose CRF to bake")
    ap.add_argument("--stats_json", default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.training import gsplat_jit_shim  # noqa: F401
    from gsplat import export_splats
    p = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    if "ppisp" not in p:
        raise SystemExit(f"{a.ckpt}: no PPISP state (train with --ppisp)")
    crf = p["ppisp"]["crf_params"][a.camera].float()
    sp = {k: v.float() for k, v in p["splats"].items()}
    b = bake(sp, crf)
    if a.out:
        export_splats(means=b["means"], scales=b["scales"], quats=b["quats"],
                      opacities=b["opacities"], sh0=b["sh0"], shN=b["shN"], format="ply",
                      save_to=a.out)
    if a.out_ckpt:
        torch.save({"step": p["step"], "splats": b}, a.out_ckpt)
    grid = torch.linspace(0, 1, 11)[:, None].repeat(1, 3)
    info = {"ckpt": a.ckpt, "out": a.out, "camera": a.camera,
            "crf_on_grid_0_to_1": crf_apply(grid, crf).round(decimals=4).tolist(),
            "num_gaussians": int(sp["means"].shape[0])}
    if a.stats_json:
        Path(a.stats_json).write_text(json.dumps(info, indent=1), encoding="utf8")
    print(json.dumps({k: v for k, v in info.items() if k != "crf_on_grid_0_to_1"}))
    print("CRF(0,0.1,..,1) R:", [round(r[0], 3) for r in info["crf_on_grid_0_to_1"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
