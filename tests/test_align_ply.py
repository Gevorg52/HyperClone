"""Tests for tools/align_ply.py — WP-O5 addendum: lossless gravity alignment.

The load-bearing claim is that the similarity transform is INVISIBLE: the transformed model
must render identically from a correspondingly transformed camera. The strongest test here is
exactly that — a real gsplat render of a tiny random model, before and after, asserted equal.
That catches every convention error at once (quaternion order, SH band rotation, scale in
log-space, camera transform), which is why it is worth the GPU.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.align_ply import (  # noqa: E402
    AXES,
    apply_to_splats,
    build_transform,
    camera_up_prior,
    distinct_centres,
    fibonacci_sphere,
    quat_left_multiply,
    rotate_sh,
    rotmat_to_quat_wxyz,
    sh_basis,
    sh_rotation_matrix,
    transform_camera,
    trajectory_plane_normal,
)

gsplat = pytest.importorskip("gsplat")
CUDA = torch.cuda.is_available()


def random_rotation(seed: int = 0) -> np.ndarray:
    g = np.random.default_rng(seed)
    q, _ = np.linalg.qr(g.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


# --- quaternions ------------------------------------------------------------

def test_rotmat_to_quat_roundtrips_through_gsplat():
    from gsplat.cuda._torch_impl import _quat_to_rotmat

    for seed in range(6):
        R = random_rotation(seed)
        q = rotmat_to_quat_wxyz(R)
        back = _quat_to_rotmat(torch.tensor(q, dtype=torch.float64)[None])[0].numpy()
        assert np.allclose(back, R, atol=1e-10), seed


def test_quat_left_multiply_composes_rotations():
    from gsplat.cuda._torch_impl import _quat_to_rotmat

    Rg, Rl = random_rotation(1), random_rotation(2)
    q = quat_left_multiply(rotmat_to_quat_wxyz(Rg),
                           torch.tensor(rotmat_to_quat_wxyz(Rl), dtype=torch.float64)[None])
    got = _quat_to_rotmat(q)[0].numpy()
    assert np.allclose(got, Rg @ Rl, atol=1e-10)


# --- spherical harmonics ----------------------------------------------------

@pytest.mark.skipif(not CUDA, reason="gsplat SH eval needs CUDA")
def test_sh_rotation_matrix_is_block_diagonal_per_band():
    D = sh_rotation_matrix(random_rotation(3), 3, samples=256).cpu().numpy()
    for a in range(4):
        for b in range(4):
            if a == b:
                continue
            block = D[a * a:(a + 1) ** 2, b * b:(b + 1) ** 2]
            assert np.abs(block).max() < 1e-9, (a, b)


@pytest.mark.skipif(not CUDA, reason="gsplat SH eval needs CUDA")
def test_sh_rotation_preserves_the_radiance_function():
    """The whole point: f'(R d) must equal f(d) for every direction d."""
    device = torch.device("cuda")
    R = random_rotation(4)
    D = sh_rotation_matrix(R, 3, samples=512, device=device)
    g = torch.Generator().manual_seed(7)
    c = torch.randn(1, 16, 3, generator=g).to(device)
    c_rot = rotate_sh(c, D)

    d = fibonacci_sphere(97, device)
    Rd = (d.double() @ torch.as_tensor(R.T, device=device, dtype=torch.float64)).float()
    f = sh_basis(d).double() @ c[0].double()
    f_rot = sh_basis(Rd).double() @ c_rot[0].double()
    assert torch.allclose(f, f_rot, atol=1e-4), (f - f_rot).abs().max().item()


@pytest.mark.skipif(not CUDA, reason="gsplat SH eval needs CUDA")
def test_identity_rotation_leaves_sh_untouched():
    D = sh_rotation_matrix(np.eye(3), 3, samples=256)
    assert torch.allclose(D, torch.eye(16, dtype=D.dtype, device=D.device), atol=1e-9)


# --- geometry helpers -------------------------------------------------------

def test_distinct_centres_collapses_rig_duplicates():
    stations = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    centres = np.repeat(stations, 12, axis=0)          # 12 crops per station, pano-major
    out = distinct_centres(centres)
    assert out.shape == (3, 3)
    assert np.allclose(out, stations)


def test_distinct_centres_keeps_a_normal_trajectory():
    c = np.stack([np.arange(20.0), np.zeros(20), np.zeros(20)], axis=1)
    assert distinct_centres(c).shape == (20, 3)


def test_trajectory_plane_normal_finds_the_walking_plane():
    g = np.random.default_rng(0)
    xy = g.uniform(-5, 5, size=(200, 2))
    stations = np.stack([xy[:, 0], g.normal(0, 0.02, 200), xy[:, 1]], axis=1)  # flat in Y
    out = trajectory_plane_normal(stations, np.array([0.0, 1.0, 0.0]))
    assert abs(abs(out["normal"][1]) - 1.0) < 1e-2
    assert out["normal"][1] > 0                     # sign disambiguated by the prior
    assert out["planarity_ratio"] > 10


def test_camera_up_prior_uses_negative_y_column():
    c2w = np.tile(np.eye(4), (5, 1, 1))
    assert np.allclose(camera_up_prior(c2w), [0.0, -1.0, 0.0])


def test_transform_camera_matches_the_point_transform():
    """A camera centre is a point: both must land in the same place."""
    T = {"R": random_rotation(5), "centre": np.array([0.3, -1.2, 2.0]), "scale": 3.7}
    c2w = np.eye(4)
    c2w[:3, :3] = random_rotation(6)
    c2w[:3, 3] = np.array([1.5, 2.5, -0.5])
    out = transform_camera(c2w, T)
    expected = T["scale"] * (T["R"] @ (c2w[:3, 3] - T["centre"]))
    assert np.allclose(out[:3, 3], expected)
    assert np.allclose(out[:3, :3], T["R"] @ c2w[:3, :3])
    assert np.allclose(out[:3, :3] @ out[:3, :3].T, np.eye(3), atol=1e-12)


def test_axes_table_is_self_consistent():
    from tools.align_ply import AXIS_INDEX

    for name, vec in AXES.items():
        i = AXIS_INDEX[name]
        assert abs(vec[i]) == 1.0 and np.count_nonzero(vec) == 1
        assert (vec[i] > 0) == name.endswith("_up")


def test_build_transform_refuses_a_degenerate_scale():
    c2w = np.tile(np.eye(4), (10, 1, 1))                 # every camera at the origin
    pts = np.random.default_rng(0).normal(size=(500, 3))
    with pytest.raises(ValueError, match="cannot derive a metric scale"):
        build_transform(c2w, pts, "y_down", 0.65, "baseline")


# --- the real thing ---------------------------------------------------------

@pytest.mark.skipif(not CUDA, reason="rasterizer needs CUDA")
def test_transformed_model_renders_identically_from_a_transformed_camera():
    from core.training.gsplat_trainer import render_splats

    device = torch.device("cuda")
    g = torch.Generator().manual_seed(11)
    n = 400
    splats = {"means": torch.randn(n, 3, generator=g) * 0.6,
              "scales": (torch.randn(n, 3, generator=g) * 0.2 - 2.6),
              "quats": torch.randn(n, 4, generator=g),
              "opacities": torch.randn(n, generator=g),
              "sh0": torch.randn(n, 1, 3, generator=g) * 0.5,
              "shN": torch.randn(n, 15, 3, generator=g) * 0.2}
    splats["quats"] = splats["quats"] / splats["quats"].norm(dim=-1, keepdim=True)

    T = {"R": random_rotation(12), "centre": np.array([0.2, -0.1, 0.35]), "scale": 2.5}
    aligned = apply_to_splats(splats, T, device)

    c2w = np.eye(4)
    c2w[:3, 3] = np.array([0.0, 0.0, -4.0])
    K = torch.tensor([[300.0, 0, 96.0], [0, 300.0, 96.0], [0, 0, 1.0]], device=device)
    bg = torch.zeros(3, device=device)
    src = {k: v.to(device) for k, v in splats.items()}
    dst = {k: v.to(device) for k, v in aligned.items()}

    def render(model, pose):
        vm = torch.linalg.inv(torch.tensor(pose, dtype=torch.float32, device=device))
        with torch.no_grad():
            out, _, _ = render_splats(model, vm, K, 192, 192, 3, bg, "antialiased")
        return out[0][..., :3]

    a = render(src, c2w)
    b = render(dst, transform_camera(c2w, T))
    assert a.abs().sum() > 0, "degenerate test: nothing was rendered"
    assert (a - b).abs().max() < 5e-3, (a - b).abs().max().item()


@pytest.mark.skipif(not CUDA, reason="rasterizer needs CUDA")
def test_sh_must_actually_be_rotated_or_the_render_changes():
    """Guard against a future 'simplification' that drops the SH rotation."""
    from core.training.gsplat_trainer import render_splats

    device = torch.device("cuda")
    g = torch.Generator().manual_seed(13)
    n = 300
    splats = {"means": torch.randn(n, 3, generator=g) * 0.6,
              "scales": (torch.randn(n, 3, generator=g) * 0.2 - 2.6),
              "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
              "opacities": torch.randn(n, generator=g) + 2.0,
              "sh0": torch.randn(n, 1, 3, generator=g) * 0.5,
              "shN": torch.randn(n, 15, 3, generator=g) * 2.0}      # strong view-dependence
    T = {"R": random_rotation(14), "centre": np.zeros(3), "scale": 1.0}
    good = apply_to_splats(splats, T, device)
    bad = {k: v.clone() for k, v in good.items()}
    bad["sh0"], bad["shN"] = splats["sh0"].clone(), splats["shN"].clone()   # SH left alone

    c2w = np.eye(4)
    c2w[:3, 3] = np.array([0.0, 0.0, -4.0])
    K = torch.tensor([[300.0, 0, 96.0], [0, 300.0, 96.0], [0, 0, 1.0]], device=device)
    bg = torch.zeros(3, device=device)

    def render(model, pose):
        vm = torch.linalg.inv(torch.tensor(pose, dtype=torch.float32, device=device))
        with torch.no_grad():
            out, _, _ = render_splats({k: v.to(device) for k, v in model.items()},
                                      vm, K, 192, 192, 3, bg, "antialiased")
        return out[0][..., :3]

    ref = render(splats, c2w)
    pose = transform_camera(c2w, T)
    assert (ref - render(good, pose)).abs().max() < 5e-3
    assert (ref - render(bad, pose)).abs().max() > 5e-2, \
        "un-rotated SH rendered the same — the test lost its teeth"
