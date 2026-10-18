"""WP-D2 — SPZ v4 writer (core/spz_io.py, Niantic `spz` bindings) at SH degree 3 and 4.

Round-trip errors are asserted against the format's QUANTIZATION bounds, derived from the
packer source (spz src/cc/load-spz.cc @ affd0ec), not from tolerances picked to pass:
  position   24-bit fixed point, 12 fractional bits           |e| <= 2^-13
  scale      u8 of (s+10)*16                                  |e| <= 1/32
  alpha      u8 of sigmoid(a)                                 |e_sigmoid| <= 1/510
  color(DC)  u8 of c*0.15*255+127.5                           |e| <= 1/(2*0.15*255)
  SH         u8 of round(x*128)+128, then rounded to a bucket of 2^(8-bits)
                                                             |e| <= (0.5 + b/2)/128 (b>1)
The coordinate test packs RDF -> stored RUB and decodes the raw RUB, then compares with the
Wigner-D rotation of tools/align_ply.py (lstsq over gsplat's basis, itself checked against the
CUDA kernel) for the 180-deg-about-X frame change — independent of core.spz_io's parity table.
`test_upstream_spz_converter_band4_bug_is_pinned` records why we do not use spz's converter.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

spz = pytest.importorskip("spz")
from core.sh_utils import num_rest  # noqa: E402
from core.spz_io import read_spz, spz_header, write_spz  # noqa: E402


def sh_bound(bits: int) -> float:
    b = 1 << (8 - bits)
    return (0.5 + (b / 2 if b > 1 else 0.0)) / 128.0


def synth(n=3000, degree=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = lambda *s, lo, hi: lo + (hi - lo) * torch.rand(*s, generator=g)  # noqa: E731
    return {"means": u(n, 3, lo=-3.0, hi=3.0),
            "scales": u(n, 3, lo=-7.0, hi=-1.0),
            "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
            "opacities": u(n, lo=-4.0, hi=4.0),
            "sh0": u(n, 1, 3, lo=-3.0, hi=3.0),
            "shN": u(n, num_rest(degree), 3, lo=-0.95, hi=0.95)}


def quat_angle(q1, q2):
    d = (torch.nn.functional.normalize(q1, dim=-1)
         * torch.nn.functional.normalize(q2, dim=-1)).sum(-1).abs().clamp(max=1.0)
    return 2 * torch.acos(d)


@pytest.mark.parametrize("bits", [(5, 4), (8, 8)], ids=["spz_default_5_4", "max_8_8"])
@pytest.mark.parametrize("degree", [3, 4])
def test_roundtrip_within_quantization(tmp_path, degree, bits):
    s = synth(degree=degree)
    p = tmp_path / f"d{degree}.spz"
    rec = write_spz(s, p, sh1_bits=bits[0], sh_rest_bits=bits[1])
    h = spz_header(p)
    assert (h["magic"], h["version"], h["sh_degree"], h["num_points"]) == \
        ("NGSP", 4, degree, 3000)
    assert h["flags"] & 0x1                                # antialiased flag set
    d = read_spz(p)
    assert d["shN"].shape == s["shN"].shape
    e = {"means": (d["means"] - s["means"]).abs().max().item(),
         "scales": (d["scales"] - s["scales"]).abs().max().item(),
         "alpha": (torch.sigmoid(d["opacities"]) - torch.sigmoid(s["opacities"])
                   ).abs().max().item(),
         "sh0": (d["sh0"] - s["sh0"]).abs().max().item(),
         "sh1": (d["shN"][:, :3] - s["shN"][:, :3]).abs().max().item(),
         "sh_rest": (d["shN"][:, 3:] - s["shN"][:, 3:]).abs().max().item(),
         "quat_deg": math.degrees(quat_angle(d["quats"], s["quats"]).max().item())}
    print(f"[spz] deg {degree} bits {bits}: " + " ".join(f"{k} {v:.2e}" for k, v in e.items())
          + f"  bytes {rec['bytes']}")
    eps = 1e-6
    assert e["means"] <= 2 ** -13 + eps
    assert e["scales"] <= 1 / 32 + eps
    assert e["alpha"] <= 1 / 510 + eps
    assert e["sh0"] <= 1 / (2 * 0.15 * 255) + eps
    assert e["sh1"] <= sh_bound(bits[0]) + eps
    assert e["sh_rest"] <= sh_bound(bits[1]) + eps
    assert e["quat_deg"] < 0.25                         # 10-bit smallest-three


def test_sh_degree_cap(tmp_path):
    s = synth(500, 4)
    p = tmp_path / "cap3.spz"
    write_spz(s, p, sh_degree=3, sh1_bits=8, sh_rest_bits=8)
    d = read_spz(p)
    assert spz_header(p)["sh_degree"] == 3 and d["shN"].shape == (500, 15, 3)
    assert (d["shN"] - s["shN"][:, :15]).abs().max().item() <= sh_bound(8) + 1e-6
    with pytest.raises(ValueError):
        write_spz(synth(10, 2), tmp_path / "up.spz", sh_degree=3)


def test_sh_outside_unit_range_clips(tmp_path):
    """Documented format limit: SH coeffs are stored over [-1, 127/128] and CLIP outside it
    (sign depends on the RDF->RUB parity of each coefficient, so |decoded| <= 1)."""
    s = synth(64, 4)
    s["shN"][:] = 1.7
    p = tmp_path / "clip.spz"
    write_spz(s, p, sh1_bits=8, sh_rest_bits=8)
    d = read_spz(p)["shN"]
    assert d.abs().max().item() <= 1.0 + 1e-6 and d.abs().min().item() >= 127 / 128 - 1e-6


@pytest.mark.parametrize("degree", [3, 4])
def test_stored_frame_is_rub_with_correct_sh_rotation(tmp_path, degree):
    from tools.align_ply import rotate_sh, sh_rotation_matrix
    s = synth(800, degree, seed=2)
    p = tmp_path / "rub.spz"
    write_spz(s, p, sh1_bits=8, sh_rest_bits=8)
    rub = read_spz(p, to_coord="RUB")
    F = np.diag([1.0, -1.0, -1.0])                     # RDF -> RUB = 180 deg about X
    assert (rub["means"] - s["means"] * torch.tensor([1.0, -1.0, -1.0])
            ).abs().max().item() <= 2 ** -13 + 1e-6
    sh = torch.cat([s["sh0"], s["shN"]], 1).double()
    exp = rotate_sh(sh, sh_rotation_matrix(F, degree))[:, 1:]
    err = (rub["shN"].double() - exp).abs().max().item()
    print(f"[spz] deg {degree} RUB SH vs our Wigner-D: {err:.2e}")
    assert err <= sh_bound(8) + 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="rasterizer needs CUDA")
def test_decoded_render_close_to_original(tmp_path):
    from core.training.gsplat_trainer import render_splats
    dev = torch.device("cuda")
    s = synth(2000, 4, seed=3)
    s["scales"] = s["scales"].clamp(-4.5, -2.5)
    s["shN"] = s["shN"] * 0.3
    p = tmp_path / "r.spz"
    write_spz(s, p)
    d = read_spz(p)
    K = torch.tensor([[300.0, 0, 128.0], [0, 300.0, 128.0], [0, 0, 1.0]], device=dev)
    vm = torch.eye(4, device=dev)
    vm[2, 3] = 8.0
    bg = torch.zeros(3, device=dev)
    with torch.no_grad():
        a = render_splats({k: v.to(dev) for k, v in s.items()}, vm, K, 256, 256, 4, bg)[0]
        b = render_splats({k: v.to(dev) for k, v in d.items()}, vm, K, 256, 256, 4, bg)[0]
    mse = ((a.clamp(0, 1) - b.clamp(0, 1)) ** 2).mean().item()
    psnr = 10 * math.log10(1 / max(mse, 1e-12))
    print(f"[spz] synthetic render PSNR orig vs decoded: {psnr:.1f} dB")
    assert psnr > 35.0


@pytest.mark.parametrize("degree", [3, 4])
def test_rdf_reload_through_ply_loader_matches(tmp_path, degree):
    """Our .spz decodes back to the SAME RDF model qa_report would load from the PLY."""
    import gsplat
    import qa_report
    s = synth(300, degree, seed=5)
    ply = tmp_path / "m.ply"
    gsplat.export_splats(**s, format="ply", save_to=str(ply))
    write_spz(qa_report.load_splats(str(ply), "cpu"), tmp_path / "m.spz",
              sh1_bits=8, sh_rest_bits=8)
    d = qa_report.load_splats(str(tmp_path / "m.spz"), "cpu")      # .spz branch
    assert (d["shN"] - s["shN"]).abs().max().item() <= sh_bound(8) + 1e-6
    assert (d["means"] - s["means"]).abs().max().item() <= 2 ** -13 + 1e-6


def test_upstream_spz_converter_band4_bug_is_pinned(tmp_path):
    """spz @ affd0ec, splat-types.h flipSh[23] = y; gsplat parity says 1 (m=+4 cos(4phi) is
    even in x and y). Library RDF->RUB therefore mis-signs exactly rest-coeff 23 at degree 4
    and nothing else. If this test starts FAILING, upstream fixed it: re-evaluate whether to
    switch core.spz_io back to the library converter (and tell the UE plugin side)."""
    s = synth(200, 4, seed=6)
    ours = tmp_path / "ours.spz"
    write_spz(s, ours, sh1_bits=8, sh_rest_bits=8)
    rub_ours = read_spz(ours, to_coord="RUB")["shN"]
    from core.spz_io import to_cloud, from_cloud
    opt = spz.PackOptions()
    opt.from_coord = spz.CoordinateSystem.RDF
    opt.sh1_bits = opt.sh_rest_bits = 8
    lib = tmp_path / "lib.spz"
    assert spz.save_spz(to_cloud(s), opt, str(lib))
    uo = spz.UnpackOptions()
    uo.to_coord = spz.CoordinateSystem.UNSPECIFIED
    rub_lib = from_cloud(spz.load_spz(str(lib), uo))["shN"]
    diff = (rub_lib - rub_ours).abs().amax(dim=(0, 2))                  # per rest coeff
    bad = [int(i) for i in torch.nonzero(diff > 2 * sh_bound(8)).flatten()]
    assert bad == [23], (bad, diff)
    # and it is a pure sign flip there
    assert (rub_lib[:, 23] + rub_ours[:, 23]).abs().max().item() <= 2 * sh_bound(8)


def test_out_of_range_positions_are_dropped_not_wrapped(tmp_path):
    """24-bit fixed point WRAPS beyond +-2048 units; the writer must drop such Gaussians."""
    s = synth(100, 3, seed=7)
    s["means"][5] = torch.tensor([3000.0, 0.0, 0.0])
    s["means"][9] = torch.tensor([0.0, -5000.0, 1.0])
    rec = write_spz(s, tmp_path / "far.spz")
    assert rec["dropped_out_of_range"] == 2 and rec["n"] == 98
    d = read_spz(tmp_path / "far.spz")
    keep = torch.ones(100, dtype=torch.bool)
    keep[[5, 9]] = False
    assert (d["means"] - s["means"][keep]).abs().max().item() <= 2 ** -13 + 1e-6
