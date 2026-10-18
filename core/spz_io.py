"""SPZ v4 writer/reader for our splats dicts (WP-D2), via Niantic's `spz` Python bindings.

SPZ (github.com/nianticlabs/spz, MIT) is the only compact splat format that carries SH degree
0..4, so it is the compressed carrier for the degree-4 master our Unreal plugin reads
(SOG v2 and KHR_gaussian_splatting cap at degree 3). Build: `tools/spz/build_spz.bat`.

Mapping (our raw-parameter dict -> spz.GaussianCloud), all verified in `tests/test_spz_io.py`:
  means (N,3)          -> positions   (24-bit fixed point, 12 fractional bits: step 2^-12,
                                       range +-2048 units)
  scales (N,3) log     -> scales      (8 bit: (s+10)*16 -> step 1/16 in log space, [-10, 5.94])
  quats (N,4) w,x,y,z  -> rotations   x,y,z,w (normalised; v3+ "smallest three", 10-bit)
  opacities (N,) logit -> alphas      (8 bit of sigmoid(alpha))
  sh0 (N,1,3) DC coeff -> colors      (8 bit: c*0.15*255+127.5 -> step ~0.026, |c| <= 3.33)
  shN (N,K-1,3)        -> sh          coefficient-major, channel-inner = shN.reshape(N, -1)
                                       as stored (NOT the channel-major PLY f_rest order);
                                       8 bit over [-1, 1), then bucketed to `sh1_bits` (band 1)
                                       / `sh_rest_bits` (bands 2..4). Values outside [-1, 1)
                                       CLIP.
Coordinates: our models are COLMAP/OpenCV = RDF (x right, y down, z forward), which is also the
PLY convention SPZ assumes. The file is stored in SPZ's canonical RUB frame, and a consumer
(our UE plugin) must convert RUB -> its own frame on load. **We do the RDF<->RUB change
ourselves** (`rdf_rub_flip`, a 180-deg rotation about X) and pack/unpack with
CoordinateSystem.UNSPECIFIED, because spz @ affd0ec has a band-4 bug in its own converter:
`splat-types.h` flipSh[23] (the m=+4, cos(4phi) coefficient) is `y` but must be `1` under
gsplat's basis parity (band 4 m=+4 is even in x and y). Measured: library RDF->RUB at degree 4
mis-signs exactly that coefficient (tests/test_spz_io.py pins it). Bands 0..3 are correct.

  python -m core.spz_io model.ply model.spz [--sh_degree 4] [--sh1_bits 5 --sh_rest_bits 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from core.sh_utils import num_rest, sh_degree_of  # noqa: E402

# spz's own PackOptions defaults are 5 bits (band 1) / 4 bits (bands 2+). WP-D2 measured them on
# a real 500k-Gaussian model: decoded-vs-original render PSNR only 30.4 dB (SH4) / 33.7 dB (SH3),
# vs 53.8 / 54.1 dB at 8/8 — the SH bucketing is the whole loss. We default to 8/8 (file still
# 7.1x / 6.4x smaller than the PLY). Pass 5/4 explicitly for the smallest file.
DEFAULT_SH1_BITS = 8
DEFAULT_SH_REST_BITS = 8
SPZ_LIB_SH1_BITS, SPZ_LIB_SH_REST_BITS = 5, 4
# 24-bit signed fixed point with 12 fractional bits: |x| must stay below 2^11. Beyond it the
# packer WRAPS (fixed32 & 0xffffff), teleporting a far floater to the other side of the world.
POSITION_LIMIT = 2047.5
# Representable ranges of the 8-bit attributes (load-spz.cc @ affd0ec); values outside CLIP:
#   log-scale  u8 of (s+10)*16            -> [-10, 255/16-10 = 5.9375]
#   DC color   u8 of c*0.15*255+127.5     -> |c| <= 127.5/(0.15*255) = 3.333
#   SH rest    u8 of round(x*128)+128     -> [-1, 127/128], i.e. outside [-1, 1) clips
SCALE_RANGE = (-10.0, 255.0 / 16.0 - 10.0)
SH0_LIMIT = 127.5 / (0.15 * 255.0)
# write_spz warns when more than this fraction of scale values clip (WP-D2 smoke models: 1.6-2.0 %
# — tiny sub-pixel Gaussians, render-neutral there, but a scene where it bites should be visible).
SCALE_CLIP_WARN_FRAC = 0.005


def clip_fractions(splats: dict) -> dict:
    """Fraction of VALUES (not Gaussians) of each 8-bit attribute outside SPZ's representable
    range. Pass the dict actually being packed (post-drop/truncate/frame-flip)."""
    sc, c0, sh = splats["scales"], splats["sh0"], splats["shN"]
    frac = lambda m: float(m.float().mean()) if m.numel() else 0.0  # noqa: E731
    return {"scales": frac((sc < SCALE_RANGE[0]) | (sc > SCALE_RANGE[1])),
            "sh0": frac(c0.abs() > SH0_LIMIT),
            "shN": frac((sh < -1.0) | (sh >= 1.0))}


def spz_warnings(rec: dict) -> list[str]:
    """Human-readable warnings for a write_spz record (drops + scale clipping)."""
    out = []
    n_drop, n = rec.get("dropped_out_of_range", 0), rec.get("n", 0)
    if n_drop:
        out.append(f"SPZ dropped {n_drop:,} of {n + n_drop:,} Gaussians "
                   f"({100.0 * n_drop / max(n + n_drop, 1):.4f}%) with |position| >= "
                   f"{rec.get('position_limit', POSITION_LIMIT)} units (24-bit fixed point "
                   f"would wrap them into the scene)")
    cf = rec.get("clip_frac", {})
    if cf.get("scales", 0.0) > SCALE_CLIP_WARN_FRAC:
        out.append(f"SPZ clipped {100.0 * cf['scales']:.2f}% of log-scale values to "
                   f"[{SCALE_RANGE[0]:g}, {SCALE_RANGE[1]:g}] (> "
                   f"{100.0 * SCALE_CLIP_WARN_FRAC:g}% threshold); sh0 clipped "
                   f"{100.0 * cf.get('sh0', 0.0):.3f}%, shN outside [-1, 1) "
                   f"{100.0 * cf.get('shN', 0.0):.3f}%")
    return out


def _spz():
    try:
        import spz  # noqa: F401
    except ImportError as e:                                    # pragma: no cover
        raise RuntimeError("Niantic `spz` Python bindings not installed — run "
                           "tools/spz/build_spz.bat (WP-D2)") from e
    return spz


def _np(t) -> np.ndarray:
    return t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, np.float32)


def _sh_parity_signs(degree: int, flip=(1.0, -1.0, -1.0)) -> torch.Tensor:
    """(K,) exact +-1 sign of each gsplat SH basis function under (x,y,z) -> flip*(x,y,z).

    Computed from gsplat's own basis (float64 torch oracle), not a hand table: for a signed
    diagonal flip every real-SH basis function maps to +- itself."""
    from gsplat.cuda._torch_impl import _eval_sh_bases_fast

    g = torch.Generator().manual_seed(0)
    u = torch.nn.functional.normalize(torch.randn(64, 3, dtype=torch.float64, generator=g),
                                      dim=-1)
    k = (degree + 1) ** 2
    a = _eval_sh_bases_fast(k, u)
    b = _eval_sh_bases_fast(k, u * torch.tensor(flip, dtype=torch.float64))
    ratio = (a * b).sum(0) / (a * a).sum(0)
    signs = torch.round(ratio)
    assert torch.all(signs.abs() == 1) and (ratio - signs).abs().max() < 1e-9, ratio
    assert (b - a * signs).abs().max() < 1e-9
    return signs.float()


def rdf_rub_flip(splats: dict) -> dict:
    """RDF <-> RUB frame change (180 deg about X; an involution). Exact sign flips only."""
    deg = sh_degree_of(splats)
    q = splats["quats"]
    sgn = _sh_parity_signs(deg)[1:].view(1, -1, 1).to(splats["shN"])
    return {"means": splats["means"] * torch.tensor([1.0, -1.0, -1.0]).to(splats["means"]),
            "scales": splats["scales"].clone(),
            # R' = F R F^T (F diagonal, so Sigma' = F Sigma F^T): q -> (w, x, -y, -z)
            "quats": q * torch.tensor([1.0, 1.0, -1.0, -1.0]).to(q),
            "opacities": splats["opacities"].clone(),
            "sh0": splats["sh0"].clone(),                 # band 0 is invariant
            "shN": splats["shN"] * sgn}


def to_cloud(splats: dict, sh_degree: int | None = None, antialiased: bool = True):
    """splats dict (raw params, in whatever frame they are) -> spz.GaussianCloud with SH
    truncated to `sh_degree`. No coordinate change here — see `write_spz`."""
    spz = _spz()
    have = sh_degree_of(splats)
    deg = have if sh_degree is None else int(sh_degree)
    if deg > have:
        raise ValueError(f"model has SH degree {have}; cannot write degree {deg}")
    n = int(splats["means"].shape[0])
    q = _np(splats["quats"]).reshape(n, 4)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    c = spz.GaussianCloud()
    c.positions = np.ascontiguousarray(_np(splats["means"]).reshape(-1))   # defines num_points
    c.sh_degree = deg
    c.antialiased = bool(antialiased)
    c.scales = np.ascontiguousarray(_np(splats["scales"]).reshape(-1))
    c.rotations = np.ascontiguousarray(q[:, [1, 2, 3, 0]].reshape(-1))     # wxyz -> xyzw
    c.alphas = np.ascontiguousarray(_np(splats["opacities"]).reshape(-1))
    c.colors = np.ascontiguousarray(_np(splats["sh0"]).reshape(-1))
    sh = _np(splats["shN"]).reshape(n, -1, 3)[:, :num_rest(deg)]
    c.sh = np.ascontiguousarray(sh.reshape(-1))
    return c


def from_cloud(c) -> dict:
    """spz.GaussianCloud -> splats dict (torch CPU float32, quats wxyz)."""
    n, deg = int(c.num_points), int(c.sh_degree)
    rot = np.asarray(c.rotations, np.float32).reshape(n, 4)
    return {"means": torch.from_numpy(np.asarray(c.positions, np.float32).reshape(n, 3).copy()),
            "scales": torch.from_numpy(np.asarray(c.scales, np.float32).reshape(n, 3).copy()),
            "quats": torch.from_numpy(rot[:, [3, 0, 1, 2]].copy()),
            "opacities": torch.from_numpy(np.asarray(c.alphas, np.float32).reshape(n).copy()),
            "sh0": torch.from_numpy(np.asarray(c.colors, np.float32).reshape(n, 1, 3).copy()),
            "shN": torch.from_numpy(np.asarray(c.sh, np.float32)
                                    .reshape(n, num_rest(deg), 3).copy())}


def write_spz(splats: dict, path, sh_degree: int | None = None,
              sh1_bits: int = DEFAULT_SH1_BITS, sh_rest_bits: int = DEFAULT_SH_REST_BITS,
              antialiased: bool = True) -> dict:
    """Write an SPZ v4 file from an RDF splats dict (stored as RUB). Returns a small record.

    Gaussians with any |coordinate| >= POSITION_LIMIT are DROPPED (counted in the record):
    the format cannot represent them and would wrap them into the scene. The record also
    carries `clip_frac` (fraction of scale / sh0 / shN values outside the 8-bit range, see
    `clip_fractions`) and `warnings` (drops, scale clip > SCALE_CLIP_WARN_FRAC); callers print
    them (core.delivery, the CLI)."""
    spz = _spz()
    cpu = {k: v.detach().float().cpu() for k, v in splats.items()}
    keep = cpu["means"].abs().amax(dim=1) < POSITION_LIMIT
    n_drop = int((~keep).sum())
    if n_drop:
        cpu = {k: v[keep].contiguous() for k, v in cpu.items()}
    if sh_degree is not None:
        from core.sh_utils import truncate_shN
        if sh_degree > sh_degree_of(cpu):
            raise ValueError(f"model has SH degree {sh_degree_of(cpu)}; cannot write degree "
                             f"{sh_degree}")
        cpu["shN"] = truncate_shN(cpu["shN"], sh_degree)
    rub = rdf_rub_flip(cpu)
    clip = clip_fractions(rub)
    cloud = to_cloud(rub, None, antialiased)
    opt = spz.PackOptions()
    opt.version = 4
    opt.from_coord = spz.CoordinateSystem.UNSPECIFIED      # already RUB (see module doc)
    opt.sh1_bits = int(sh1_bits)
    opt.sh_rest_bits = int(sh_rest_bits)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not spz.save_spz(cloud, opt, str(path)):
        raise RuntimeError(f"spz.save_spz failed for {path}")
    rec = {"path": str(path), "sh_degree": int(cloud.sh_degree), "n": int(cloud.num_points),
           "sh1_bits": int(sh1_bits), "sh_rest_bits": int(sh_rest_bits),
           "bytes": os.path.getsize(path), "coord_stored": "RUB (our own RDF->RUB flip)",
           "dropped_out_of_range": n_drop, "position_limit": POSITION_LIMIT,
           "clip_frac": clip, "clip_ranges": {"scales": list(SCALE_RANGE),
                                              "sh0_abs_max": SH0_LIMIT, "shN": [-1.0, 1.0]},
           "version": 4}
    rec["warnings"] = spz_warnings(rec)
    return rec


def read_spz(path, to_coord: str = "RDF") -> dict:
    """Decode an SPZ (stored RUB) to a splats dict in `to_coord` ("RDF" default, or "RUB" =
    the raw stored frame). Uses our own exact frame flip, never spz's converter."""
    if to_coord not in ("RDF", "RUB"):
        raise ValueError("to_coord must be 'RDF' or 'RUB'")
    spz = _spz()
    opt = spz.UnpackOptions()
    opt.to_coord = spz.CoordinateSystem.UNSPECIFIED
    cloud = spz.load_spz(str(path), opt)
    if int(cloud.num_points) == 0 and os.path.getsize(path) > 64:
        raise RuntimeError(f"spz.load_spz returned no gaussians for {path}")
    s = from_cloud(cloud)
    return rdf_rub_flip(s) if to_coord == "RDF" else s


def spz_header(path) -> dict:
    """The 32-byte v4 plaintext header: magic 'NGSP', version, numPoints, shDegree, ..."""
    import struct
    with open(path, "rb") as fh:
        b = fh.read(16)
    magic, version, num_points = struct.unpack("<4sII", b[:12])
    sh_degree, fractional_bits, flags = b[12], b[13], b[14]
    return {"magic": magic.decode("ascii", "replace"), "version": version,
            "num_points": num_points, "sh_degree": sh_degree,
            "fractional_bits": fractional_bits, "flags": flags}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m core.spz_io")
    ap.add_argument("src", help=".ply or .pt checkpoint")
    ap.add_argument("dst", help="output .spz")
    ap.add_argument("--sh_degree", type=int, default=None, choices=[0, 1, 2, 3, 4],
                    help="cap (default: the model's own degree)")
    ap.add_argument("--sh1_bits", type=int, default=DEFAULT_SH1_BITS)
    ap.add_argument("--sh_rest_bits", type=int, default=DEFAULT_SH_REST_BITS)
    args = ap.parse_args(argv)
    import qa_report
    splats = {k: v.cpu() for k, v in qa_report.load_splats(args.src, "cpu").items()}
    rec = write_spz(splats, args.dst, args.sh_degree, args.sh1_bits, args.sh_rest_bits)
    for w in rec["warnings"]:
        print(f"[spz] WARNING: {w}", file=sys.stderr)
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
