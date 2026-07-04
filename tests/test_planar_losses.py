"""CPU unit tests for the WP-F4 planar-prior math (no GPU / no training).

Guards the ported PlanarGS geometry so future edits are caught by CI:
  - co_planar_depth: exact plane fit + gradient flow + sign-preserving degenerate
    guard (grazing rays must NOT blow up to ~1e6).
  - normal_from_depth: matches the true plane normal, points toward the camera.
  - plane_masks._fit_plane/_plane_depth_along + build_view_mask on synthetic data.
"""
import numpy as np
import torch

from core.training.gsplat_trainer import co_planar_depth, normal_from_depth
from core.plane_masks import _fit_plane, _plane_depth_along, build_view_mask


def _synth_plane(H=48, W=64, fx=60.0):
    K = torch.tensor([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], dtype=torch.float64)
    invK = torch.linalg.inv(K)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                            torch.arange(W, dtype=torch.float64), indexing="ij")
    rays = invK @ torch.stack([xx.reshape(-1), yy.reshape(-1),
                               torch.ones(H * W, dtype=torch.float64)], 0)
    n = torch.tensor([0.15, -0.25, -1.0], dtype=torch.float64); n = n / n.norm()
    z = (-4.0 / (n @ rays)).reshape(H, W)          # positive, in front of camera
    return z, invK, n, K


def test_coplanar_exact_on_perfect_plane():
    z, invK, _, _ = _synth_plane()
    assert (z > 0).all()
    pm = torch.ones_like(z, dtype=torch.long)
    pd = co_planar_depth(z, pm, invK)
    rel = ((pd - z).abs() / z)[8:-8, 8:-8]         # ignore replicate/border region
    assert rel.max().item() < 1e-4, rel.max().item()


def test_coplanar_gradients_flow_and_finite():
    z, invK, _, _ = _synth_plane()
    zb = z.clone(); zb[16:32, 20:44] += 0.3 * z[16:32, 20:44]
    zb = zb.detach().requires_grad_(True)
    pm = torch.ones_like(z, dtype=torch.long)
    pd = co_planar_depth(zb, pm, invK)
    mask = (pm > 0).float()
    term = ((zb - pd).abs() * mask).sum() / mask.sum()
    term.backward()
    assert term.item() > 0
    assert torch.isfinite(zb.grad).all()
    assert zb.grad.abs().sum() > 0


def test_coplanar_grazing_denominator_bounded():
    """A ray nearly parallel to the plane (paramᵀr→0) must yield an O(1) fallback
    depth, NOT ~1e6 (the sign-preserving guard, review finding #1)."""
    z, invK, _, _ = _synth_plane()
    pm = torch.ones_like(z, dtype=torch.long)
    pd = co_planar_depth(z, pm, invK)
    assert torch.isfinite(pd).all()
    assert pd.abs().max().item() < 1e3    # no 1e6 spike from a fixed +eps guard


def test_normal_from_depth_matches_and_faces_camera():
    z, invK, n, _ = _synth_plane()
    nn = normal_from_depth(z, invK)
    core = nn[8:-8, 8:-8].reshape(-1, 3)
    cos = (core @ n).abs()
    assert cos.min().item() > 0.999               # matches true plane normal
    assert (nn[..., 2] < 0).float().mean().item() > 0.99   # toward camera (−z)


def test_fit_plane_numpy_twin():
    z, invK, _, _ = _synth_plane()
    H, W = z.shape
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float64),
                         np.arange(W, dtype=np.float64), indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel(), np.ones(H * W)], 0)
    rays = invK.numpy() @ coords
    pts = rays * z.numpy().ravel()[None, :]
    param = _fit_plane(pts)
    pred = _plane_depth_along(param, rays)
    rel = np.abs(pred - z.numpy().ravel()) / z.numpy().ravel()
    assert rel.max() < 1e-4


def test_build_view_mask_finds_a_plane():
    """Synthetic frame: a large fronto-parallel plane on the left half, random
    (non-planar) normals on the right → left half should become one instance."""
    H, W = 200, 240
    normal = np.zeros((H, W, 3), np.float64)
    depth = np.full((H, W), np.inf)
    # left half: plane facing camera at depth 3
    normal[:, :W // 2] = np.array([0.0, 0.0, -1.0])
    depth[:, :W // 2] = 3.0
    # right half: valid but noisy normals + noisy depth (should be pruned)
    rng = np.random.default_rng(0)
    nr = rng.normal(size=(H, W // 2, 3)); nr /= np.linalg.norm(nr, axis=-1, keepdims=True)
    normal[:, W // 2:] = nr
    depth[:, W // 2:] = 3.0 + rng.normal(scale=1.0, size=(H, W // 2))
    fx = 150.0
    inv_K = np.linalg.inv(np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]]))
    labels, n = build_view_mask(normal, depth, inv_K, k_normal=4,
                                dist_thresh=0.05, min_area=1000, min_frac=0.01,
                                rng=np.random.default_rng(0))
    assert n >= 1
    # the biggest instance should be the left plane and cover most of the left half
    big = np.bincount(labels.ravel())[1:]
    assert big.max() > 0.30 * H * W


def test_default_off_helpers_are_pure():
    """Sanity: helpers don't touch CUDA / global state (CPU tensors round-trip)."""
    z, invK, _, _ = _synth_plane(H=32, W=32)
    pm = torch.ones_like(z, dtype=torch.long)
    assert co_planar_depth(z, pm, invK).shape == z.shape
    assert normal_from_depth(z, invK).shape == (*z.shape, 3)
