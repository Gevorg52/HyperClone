"""WP-SH4 phase A — degree-agnostic SH I/O: degree bookkeeping, PLY f_rest ordering through
gsplat `export_splats` and our readers, exact degree truncation, and the SH rotation in
tools/align_ply.py at band 4 (it was hard-coded to degree 3).

The rotation is checked against its DEFINITION (rotating the coefficients must equal
evaluating the original function at rotated directions), with the function evaluated by the
CUDA kernel — an implementation independent of the float64 torch basis used to build D.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.sh_utils import (  # noqa: E402
    num_rest, ply_sh_degree, read_ply_header, sh_degree_from_f_rest, sh_degree_from_rest,
    sh_degree_of, truncate_ply_file, truncate_shN,
)

gsplat = pytest.importorskip("gsplat")
CUDA = torch.cuda.is_available()


def random_rotation(seed: int = 0) -> np.ndarray:
    g = np.random.default_rng(seed)
    q, _ = np.linalg.qr(g.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def synth(n=500, degree=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = {"means": torch.randn(n, 3, generator=g) * 0.6,
         "scales": torch.randn(n, 3, generator=g) * 0.2 - 2.6,
         "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
         "opacities": torch.randn(n, generator=g) + 1.0,
         "sh0": torch.randn(n, 1, 3, generator=g) * 0.5,
         "shN": torch.randn(n, num_rest(degree), 3, generator=g) * 0.5}
    return s


def export(s, path):
    gsplat.export_splats(means=s["means"], scales=s["scales"], quats=s["quats"],
                         opacities=s["opacities"], sh0=s["sh0"], shN=s["shN"],
                         format="ply", save_to=str(path))


# --- degree bookkeeping -----------------------------------------------------

def test_degree_table():
    assert [num_rest(d) for d in range(5)] == [0, 3, 8, 15, 24]
    for d in range(5):
        assert sh_degree_from_rest(num_rest(d)) == d
        assert sh_degree_from_f_rest(3 * num_rest(d)) == d
    assert sh_degree_from_f_rest(72) == 4
    for bad in (1, 2, 5, 16, 17, 25, 35):
        with pytest.raises(ValueError):
            sh_degree_from_rest(bad)
    with pytest.raises(ValueError):
        sh_degree_from_f_rest(71)
    with pytest.raises(ValueError):
        num_rest(5)                       # gsplat's oracle is garbage above 25 bases


def test_callers_are_degree_agnostic():
    from core.delivery import sh_degree_of as deliv_deg
    for d in range(5):
        s = {"shN": torch.zeros(2, num_rest(d), 3)}
        assert deliv_deg(s) == d == sh_degree_of(s)
    with pytest.raises(ValueError):
        deliv_deg({"shN": torch.zeros(2, 17, 3)})   # what splat-transform makes of SH4


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_ply_header_degree(tmp_path, degree):
    from gui2.preview import _ply_sh_degree
    p = tmp_path / f"d{degree}.ply"
    export(synth(50, degree), p)
    n, props, _ = read_ply_header(p)
    assert sum(x.startswith("f_rest_") for x in props) == 3 * num_rest(degree)
    assert ply_sh_degree(p) == degree == _ply_sh_degree(str(p))


# --- f_rest ordering (channel-major INRIA) ------------------------------------

def test_f_rest_is_channel_major_through_export_and_all_readers(tmp_path):
    """Tag every coefficient with a unique value encoding (coeff k, channel c), export with
    gsplat, and check the raw column order AND both of our readers' inverse."""
    import qa_report
    from tools.prune_ply import read_gsplat_ply

    n, K = 4, 24
    s = synth(n, 4)
    k_idx = torch.arange(K).view(1, K, 1).float()
    c_idx = torch.arange(3).view(1, 1, 3).float()
    s["shN"] = (1000 * torch.arange(n).view(n, 1, 1) + 10 * k_idx + c_idx).float()
    p = tmp_path / "tag.ply"
    export(s, p)

    nn, props, hlen = read_ply_header(p)
    raw = np.fromfile(p, dtype="<f4", offset=hlen).reshape(nn, len(props))
    col = {x: i for i, x in enumerate(props)}
    for c in range(3):
        for k in range(K):
            v = raw[:, col[f"f_rest_{c * K + k}"]]
            assert np.array_equal(v, 1000 * np.arange(n) + 10 * k + c), (c, k)

    a = read_gsplat_ply(p)
    b = qa_report.load_splats(str(p), "cpu")
    assert torch.equal(a["shN"], s["shN"]) and torch.equal(b["shN"], s["shN"])


# --- truncation ---------------------------------------------------------------

@pytest.mark.parametrize("to_deg", [0, 1, 2, 3, 4])
def test_truncate_ply_is_exact(tmp_path, to_deg):
    import qa_report
    s = synth(300, 4, seed=1)
    src, dst = tmp_path / "sh4.ply", tmp_path / f"sh{to_deg}.ply"
    export(s, src)
    info = truncate_ply_file(src, dst, to_deg)
    assert info["f_rest_out"] == 3 * num_rest(to_deg)
    t = qa_report.load_splats(str(dst), "cpu")
    o = qa_report.load_splats(str(src), "cpu")
    for k in ("means", "scales", "quats", "opacities", "sh0"):
        assert torch.equal(t[k], o[k]), k                      # bit-identical
    assert torch.equal(t["shN"], truncate_shN(o["shN"], to_deg))
    if to_deg == 4:
        assert dst.read_bytes() == src.read_bytes()            # identity is byte-exact


def test_truncate_refuses_upward(tmp_path):
    p = tmp_path / "sh2.ply"
    export(synth(10, 2), p)
    with pytest.raises(ValueError):
        truncate_ply_file(p, tmp_path / "x.ply", 3)


@pytest.mark.skipif(not CUDA, reason="rasterizer needs CUDA")
def test_truncated_render_equals_low_degree_render(tmp_path):
    """The Postshot-twin claim: render(truncate(m, 3), deg 3) == render(m, deg 3)."""
    from core.training.gsplat_trainer import render_splats
    dev = torch.device("cuda")
    s = synth(400, 4, seed=2)
    t = dict(s, shN=truncate_shN(s["shN"], 3))
    K = torch.tensor([[300.0, 0, 96.0], [0, 300.0, 96.0], [0, 0, 1.0]], device=dev)
    vm = torch.eye(4, device=dev)
    vm[2, 3] = 4.0
    bg = torch.zeros(3, device=dev)
    with torch.no_grad():
        a = render_splats({k: v.to(dev) for k, v in s.items()}, vm, K, 192, 192, 3, bg)[0]
        b = render_splats({k: v.to(dev) for k, v in t.items()}, vm, K, 192, 192, 3, bg)[0]
        c = render_splats({k: v.to(dev) for k, v in s.items()}, vm, K, 192, 192, 4, bg)[0]
    assert a.abs().sum() > 0
    assert (a - b).abs().max().item() == 0.0
    assert (a - c).abs().max().item() > 1e-3       # band 4 is live at degree 4


# --- SH rotation at band 4 (tools/align_ply.py) -------------------------------

def test_rotation_matrix_degree4_structure():
    from tools.align_ply import sh_rotation_matrix
    R1, R2 = random_rotation(21), random_rotation(22)
    D1, D2 = sh_rotation_matrix(R1, 4), sh_rotation_matrix(R2, 4)
    D12 = sh_rotation_matrix(R1 @ R2, 4)
    assert D1.dtype == torch.float64 and D1.shape == (25, 25)
    for a in range(5):
        for b in range(5):
            blk = D1[a * a:(a + 1) ** 2, b * b:(b + 1) ** 2]
            if a != b:
                assert blk.abs().max().item() < 1e-12, (a, b)
            else:                                   # Wigner-D blocks are orthogonal
                eye = torch.eye(2 * a + 1, dtype=torch.float64)
                assert (blk @ blk.T - eye).abs().max().item() < 1e-10, a
    # group homomorphism D(R1 R2) = D(R1) D(R2)
    assert (D12 - D1 @ D2).abs().max().item() < 1e-10
    assert (sh_rotation_matrix(np.eye(3), 4) - torch.eye(25, dtype=torch.float64)
            ).abs().max().item() < 1e-12


def test_band1_matches_closed_form():
    """Independent convention check: gsplat band 1 = C1 * (-y, z, -x), so D_1 = P R P^T."""
    from tools.align_ply import sh_rotation_matrix
    R = random_rotation(23)
    P = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float64)
    D = sh_rotation_matrix(R, 4)[1:4, 1:4].numpy()
    assert np.abs(D - P @ R @ P.T).max() < 1e-12


@pytest.mark.skipif(not CUDA, reason="CUDA SH kernel")
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_rotation_preserves_radiance_cuda_eval(degree):
    """f'(R d) == f(d) with f evaluated by the CUDA kernel (not the basis D was built from)."""
    from tools.align_ply import fibonacci_sphere, rotate_sh, sh_rotation_matrix
    dev = torch.device("cuda")
    R = random_rotation(30 + degree)
    k = (degree + 1) ** 2
    c = torch.randn(1, k, 3, generator=torch.Generator().manual_seed(5)).double()
    c_rot = rotate_sh(c, sh_rotation_matrix(R, degree))
    d = fibonacci_sphere(4000, "cpu", torch.float64)
    Rd = d @ torch.as_tensor(R.T)
    m = d.shape[0]
    f = gsplat.spherical_harmonics(degree, d.float().to(dev),
                                   c.float().expand(m, k, 3).contiguous().to(dev))
    f_rot = gsplat.spherical_harmonics(degree, Rd.float().to(dev),
                                       c_rot.float().expand(m, k, 3).contiguous().to(dev))
    err = (f - f_rot).abs().max().item()
    print(f"[sh-rot] degree {degree}: max|f'(Rd)-f(d)| = {err:.2e} (CUDA fp32 eval)")
    assert err < 2e-5, err


@pytest.mark.skipif(not CUDA, reason="rasterizer needs CUDA")
def test_degree4_model_aligns_losslessly_and_rotation_matters():
    from core.training.gsplat_trainer import render_splats
    from tools.align_ply import apply_to_splats, transform_camera
    dev = torch.device("cuda")
    s = synth(400, 4, seed=3)
    s["shN"] = s["shN"] * 3.0                                  # strong view dependence
    T = {"R": random_rotation(40), "centre": np.array([0.2, -0.1, 0.3]), "scale": 1.7}
    good = apply_to_splats(s, T, dev)
    assert good["shN"].shape == (400, 24, 3)
    bad = dict(good, sh0=s["sh0"], shN=s["shN"])
    c2w = np.eye(4)
    c2w[:3, 3] = [0.0, 0.0, -4.0]
    K = torch.tensor([[300.0, 0, 96.0], [0, 300.0, 96.0], [0, 0, 1.0]], device=dev)
    bg = torch.zeros(3, device=dev)

    def render(m, pose):
        vm = torch.linalg.inv(torch.tensor(pose, dtype=torch.float32, device=dev))
        with torch.no_grad():
            return render_splats({k: v.to(dev) for k, v in m.items()}, vm, K, 192, 192, 4,
                                 bg)[0][0][..., :3]
    ref = render(s, c2w)
    pose = transform_camera(c2w, T)
    assert ref.abs().sum() > 0
    assert (ref - render(good, pose)).abs().max().item() < 5e-3
    assert (ref - render(bad, pose)).abs().max().item() > 5e-2


def test_roundtrip_synthetic_deg4_align_truncate_reload(tmp_path):
    """synthetic deg-4 PLY -> align -> PLY -> truncate to 3 -> reload == expected, and the
    two orders (align-then-truncate, truncate-then-align) agree because D is block-diagonal."""
    import qa_report
    from tools.align_ply import apply_to_splats
    src = tmp_path / "sh4.ply"
    export(synth(256, 4, seed=4), src)
    s = qa_report.load_splats(str(src), "cpu")
    T = {"R": random_rotation(50), "centre": np.array([0.1, 0.2, -0.3]), "scale": 2.0}
    al = apply_to_splats(s, T, torch.device("cpu"))
    al_ply = tmp_path / "sh4_aligned.ply"
    export(al, al_ply)
    assert ply_sh_degree(al_ply) == 4
    tw = tmp_path / "sh3_twin.ply"
    truncate_ply_file(al_ply, tw, 3)
    got = qa_report.load_splats(str(tw), "cpu")
    exp = dict(al, shN=truncate_shN(al["shN"], 3))
    for k in exp:
        assert torch.allclose(got[k], exp[k].float(), atol=0, rtol=0), k
    # commutation: truncate first, then align (degree-3 rotation of the low bands)
    al3 = apply_to_splats(dict(s, shN=truncate_shN(s["shN"], 3)), T, torch.device("cpu"))
    assert (al3["shN"] - got["shN"]).abs().max().item() < 1e-5
