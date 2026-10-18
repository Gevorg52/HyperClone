"""gsplat version shim (WP-B2): `spherical_harmonics_dirs` must give the 1.5.3 semantics on both
gsplat 1.5.3 (system) and gsplat main 1.6.0 (venvs/gsplat_main). Oracle = gsplat's own pure-torch
SH basis `_eval_sh_bases_fast` (present in both versions)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_spherical_harmonics_dirs_matches_torch_basis(degree):
    from core.training.gsplat_compat import spherical_harmonics_dirs
    from gsplat.cuda._torch_impl import _eval_sh_bases_fast
    g = torch.Generator(device="cpu").manual_seed(0)
    n, K = 512, 16
    dirs = torch.randn(n, 3, generator=g).cuda()
    coeffs = torch.randn(n, K, 3, generator=g).cuda()
    out = spherical_harmonics_dirs(degree, dirs, coeffs)
    nb = (degree + 1) ** 2
    basis = _eval_sh_bases_fast(nb, torch.nn.functional.normalize(dirs, dim=-1))   # [n, nb]
    ref = (basis[..., None] * coeffs[:, :nb]).sum(-2)
    # gsplat adds +0.5 and clamps at 0 in rasterization only; the raw SH op does neither
    assert out.shape == (n, 3)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


def test_spherical_harmonics_dirs_keeps_leading_shape():
    from core.training.gsplat_compat import spherical_harmonics_dirs
    dirs = torch.randn(4, 5, 3).cuda()
    coeffs = torch.randn(4, 5, 16, 3).cuda()
    assert spherical_harmonics_dirs(3, dirs, coeffs).shape == (4, 5, 3)
