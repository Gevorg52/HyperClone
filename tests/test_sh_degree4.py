"""WP-SH4 phase A — gsplat's spherical-harmonics kernel at degree 4 vs its torch oracle.

The installed gsplat (1.5.3, AOT sm_120) evaluates SH in CUDA for degrees 0..4. Degree 4 (K=25,
72 f_rest in a PLY) is the new lever for our Unreal plugin, so before ANY training run we prove
the CUDA forward and backward agree with gsplat's pure-torch reference
(`gsplat.cuda._torch_impl._spherical_harmonics`, which uses `_eval_sh_bases_fast`, the Sloan
JCGT-2013 closed form) to fp32 tolerance. Degrees 0..3 are the control: they are what every
production run has used, so if they pass and 4 does not, the defect is band 4 specifically.

Independent of both implementations, `test_basis_is_orthonormal_on_the_sphere` checks that the
oracle's band-4 basis really is an orthonormal real SH basis (a Fibonacci-sphere quadrature of
the Gram matrix). That property is what makes per-band SH rotation (tools/align_ply.py) exact,
so it is checked rather than assumed.
"""

import math

import pytest
import torch

gsplat = pytest.importorskip("gsplat")
CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="gsplat SH kernel needs CUDA")

from gsplat.cuda._torch_impl import _eval_sh_bases_fast, _spherical_harmonics  # noqa: E402

N = 4096
# fp32 tolerances. Coefficients ~N(0,1), |Y_l| <= ~2.6 at l=4, sum over 25 terms -> outputs O(5).
FWD_ATOL = 2e-5
GRAD_ATOL = 5e-5


def _inputs(degree: int, K: int, seed: int, unit: bool):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dirs = torch.randn(N, 3, device="cuda", generator=g)
    if unit:
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    else:                                           # the rasterizer passes UN-normalised dirs
        dirs = dirs * (0.1 + 3.0 * torch.rand(N, 1, device="cuda", generator=g))
    coeffs = torch.randn(N, K, 3, device="cuda", generator=g)
    return dirs, coeffs


def _fwd_bwd(fn, degree, dirs, coeffs, v_out):
    d = dirs.clone().requires_grad_(True)
    c = coeffs.clone().requires_grad_(True)
    out = fn(degree, d, c)
    (out * v_out).sum().backward()
    # degree 0 never touches dirs, so autograd may leave d.grad as None (torch ref) — that IS
    # a zero gradient
    gd = d.grad.detach() if d.grad is not None else torch.zeros_like(dirs)
    return out.detach(), gd, c.grad.detach()


@pytest.mark.parametrize("unit", [True, False], ids=["unit_dirs", "raw_dirs"])
@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_cuda_matches_torch_reference(degree, unit):
    K = (degree + 1) ** 2
    dirs, coeffs = _inputs(degree, K, seed=100 + degree, unit=unit)
    v_out = torch.randn(N, 3, device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(7))
    o_c, gd_c, gc_c = _fwd_bwd(gsplat.spherical_harmonics, degree, dirs, coeffs, v_out)
    o_t, gd_t, gc_t = _fwd_bwd(_spherical_harmonics, degree, dirs, coeffs, v_out)
    err = {"fwd": (o_c - o_t).abs().max().item(),
           "d_coeffs": (gc_c - gc_t).abs().max().item(),
           "d_dirs": (gd_c - gd_t).abs().max().item()}
    print(f"[sh-kernel] degree {degree} {'unit' if unit else 'raw'} dirs: "
          + "  ".join(f"{k} {v:.2e}" for k, v in err.items()))
    assert err["fwd"] < FWD_ATOL, err
    assert err["d_coeffs"] < GRAD_ATOL, err
    if degree == 0:
        assert gd_c.abs().max().item() == 0.0       # band 0 is direction-independent
    # Raw dirs: the d/d(dirs) path goes through the normalisation Jacobian, which scales like
    # 1/|d|; scale the tolerance by the largest 1/|d| in the batch.
    inv = (1.0 / dirs.norm(dim=-1)).max().item() if not unit else 1.0
    assert err["d_dirs"] < GRAD_ATOL * max(1.0, inv), err


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_lower_degree_ignores_extra_coeffs(degree):
    """Evaluating degree d on a K=25 tensor must use only the first (d+1)^2 coeffs — this is
    what happens when a deg-4 model is rendered/QA'd at a lower degree, and what makes the
    deg-3 'Postshot twin' an EXACT truncation."""
    dirs, coeffs = _inputs(4, 25, seed=200 + degree, unit=True)
    k = (degree + 1) ** 2
    full = gsplat.spherical_harmonics(degree, dirs, coeffs)
    trunc = gsplat.spherical_harmonics(degree, dirs, coeffs[:, :k].contiguous())
    assert (full - trunc).abs().max().item() == 0.0
    v = torch.randn_like(full)
    c = coeffs.clone().requires_grad_(True)
    (gsplat.spherical_harmonics(degree, dirs, c) * v).sum().backward()
    assert c.grad[:, k:].abs().max().item() == 0.0  # no gradient leaks into unused bands


def test_gradcheck_degree4_float64_reference():
    """Sanity of the ORACLE itself: torch autograd vs finite differences, float64."""
    g = torch.Generator().manual_seed(3)
    d = torch.randn(8, 3, dtype=torch.float64, generator=g).requires_grad_(True)
    c = torch.randn(8, 25, 3, dtype=torch.float64, generator=g).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a, b: _spherical_harmonics(4, a, b), (d, c))


def test_basis_is_orthonormal_on_the_sphere():
    """∫ Y_i Y_j dΩ = δ_ij for all 25 basis functions (quadrature on a Fibonacci sphere)."""
    M = 200_000
    i = torch.arange(M, dtype=torch.float64) + 0.5
    phi = torch.acos(1.0 - 2.0 * i / M)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    u = torch.stack([torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi),
                     torch.cos(phi)], dim=-1)
    Y = _eval_sh_bases_fast(25, u)                       # (M, 25), float64
    G = (4.0 * math.pi / M) * (Y.T @ Y)
    err = (G - torch.eye(25, dtype=torch.float64)).abs().max().item()
    print(f"[sh-basis] Gram max|G-I| over 25 bases: {err:.2e}")
    assert err < 1e-6, err


def test_cuda_basis_matches_oracle_basis_degree4():
    """Column-by-column: CUDA's basis (one-hot coeffs) == the oracle's basis, all 25 columns."""
    dirs, _ = _inputs(4, 25, seed=9, unit=True)
    Y_t = _eval_sh_bases_fast(25, dirs)
    worst = 0.0
    for j in range(25):
        c = torch.zeros(N, 25, 3, device="cuda")
        c[:, j, :] = 1.0
        col = gsplat.spherical_harmonics(4, dirs, c)[:, 0]
        worst = max(worst, (col - Y_t[:, j]).abs().max().item())
    assert worst < 1e-5, worst


def test_trainer_allocates_24_rest_coeffs_for_sh_degree_4():
    """Read-only check of core/training/gsplat_trainer.py::init_gaussians_from_sfm."""
    import numpy as np
    from core.training.gsplat_trainer import init_gaussians_from_sfm

    pts = np.random.default_rng(0).normal(size=(64, 3)).astype(np.float32)
    rgb = np.full((64, 3), 128, dtype=np.uint8)
    for deg, rest in [(3, 15), (4, 24)]:
        s = init_gaussians_from_sfm(pts, rgb, 1.0, torch.device("cuda"), sh_degree=deg)
        assert tuple(s["shN"].shape) == (64, rest, 3), (deg, s["shN"].shape)
        assert tuple(s["sh0"].shape) == (64, 1, 3)


def test_resize_shN_pad_and_truncate():
    """resize_shN zero-pads SH3 -> SH4 and truncates SH4 -> SH3 bit-identically."""
    from core.sh_utils import resize_shN, sh_degree_of

    sh3 = torch.randn(32, 15, 3, device="cuda")
    sh4 = resize_shN(sh3, 4)
    assert sh4.shape == (32, 24, 3)
    assert sh_degree_of({"shN": sh4}) == 4
    assert torch.equal(sh4[:, :15], sh3)
    assert float(sh4[:, 15:].abs().max()) == 0.0

    sh3_back = resize_shN(sh4, 3)
    assert torch.equal(sh3_back, sh3)
    assert torch.equal(resize_shN(sh3, 3), sh3)
