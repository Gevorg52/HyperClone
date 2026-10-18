"""Unit tests for WP-C3 Part 1: TIDI-GS Cross-View Evidence, Importance & 3D Isolation Pruner."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from tools.prune_ply import prune as prune_op05_only
from tools.prune_tidi import (
    TidiConfig,
    compute_knn_distances,
    compute_tidi_scores,
    compute_view_blending_weights,
    tidi_keep_mask,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for gsplat rasterization")
def test_backward_blending_weights_exactness():
    """Verify exactness of the single-backward-pass blending weight vs one-hot forward oracle,
    and verify foreground > 0, occluded behind opaque wall == 0, out-of-frustum == 0.
    """
    import gsplat

    device = torch.device("cuda:0")
    # 5 Gaussians:
    # 0: visible foreground translucent (z = 2.0, op = 0.5)
    # 1, 2: two-layer opaque wall right behind it (z = 3.0, 3.1, op = 0.99999 -> T <= 1e-4)
    # 3: occluded behind the opaque wall (z = 5.0, op = 0.8, small scale at center)
    # 4: out-of-frustum Gaussian far to the side (x = 50.0, y = 50.0, z = 2.0, op = 0.9)
    means = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 3.1],
            [0.0, 0.0, 5.0],
            [50.0, 50.0, 2.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 5, dtype=torch.float32, device=device)
    scales = torch.tensor(
        [
            [0.20, 0.20, 0.01],
            [2.00, 2.00, 0.01],
            [2.00, 2.00, 0.01],
            [0.05, 0.05, 0.01],
            [0.20, 0.20, 0.01],
        ],
        dtype=torch.float32,
        device=device,
    )
    opac = torch.tensor([0.50, 0.99999, 0.99999, 0.80, 0.90], dtype=torch.float32, device=device)
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    K = torch.tensor([[120.0, 0.0, 32.0], [0.0, 120.0, 32.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    W, H = 64, 64

    w_bwd, alpha_grad_w, ed, alpha, info = compute_view_blending_weights(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opac,
        viewmat=viewmat,
        K=K,
        width=W,
        height=H,
        rasterize_mode="classic",
        compute_alpha_grad=True,
    )

    # Brute-force forward oracle: render one-hot color per Gaussian
    w_oracle = []
    for i in range(5):
        c_oh = torch.zeros((5, 1), dtype=torch.float32, device=device)
        c_oh[i, 0] = 1.0
        r_i, _, _ = gsplat.rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opac,
            colors=c_oh,
            viewmats=viewmat[None],
            Ks=K[None],
            width=W,
            height=H,
            render_mode="RGB",
            rasterize_mode="classic",
            packed=False,
        )
        w_oracle.append(float(r_i.sum().item()))
    w_oracle_t = torch.tensor(w_oracle, dtype=torch.float32, device=device)

    # 1. Exact agreement between single backward pass and N forward one-hot passes
    assert torch.allclose(w_bwd, w_oracle_t, rtol=1e-4, atol=1e-4), f"{w_bwd} vs {w_oracle_t}"

    # 2. Visible foreground and wall Gaussians have strong positive contribution
    assert float(w_bwd[0]) > 50.0
    assert float(w_bwd[1]) > 500.0

    # 3. Occluded Gaussian behind opaque wall gets 0 contribution
    assert float(w_bwd[3]) == 0.0

    # 4. Out-of-frustum Gaussian gets strictly 0 contribution and 0 radius
    assert float(w_bwd[4]) == 0.0
    assert int(info["radii"][0, 4].max()) == 0
    assert alpha_grad_w is not None and float(alpha_grad_w[4]) == 0.0


def _build_synthetic_multi_view_scene(device: torch.device):
    """Build a synthetic 8-camera rig looking at a dense consensus surface at z = 4.0,
    plus synthetic floaters with opacity = 0.12 that survive `prune_ply.py` (op < 0.05):
      - 64 valid consensus surface Gaussians on a 8x8 grid at z = 0 (camera distance ~4.0)
      - Floater A: opacity = 0.12, seen by only 1 training camera (placed right in front of cam 0's near frustum edge)
      - Floater B: opacity = 0.12, floating in front of the consensus surface at z = -1.8 (closer to cameras),
        projecting in front of the opaque surface across multiple views (`z_i < D_exp(u_i, v_i) - delta`)
      - Floater C: opacity = 0.12, 3D spatially isolated outlier far from the surface manifold with tiny scale
    """
    # 8x8 dense consensus surface at z = 0.0 in world space
    xs = np.linspace(-0.35, 0.35, 8)
    ys = np.linspace(-0.35, 0.35, 8)
    gx, gy = np.meshgrid(xs, ys)
    surf_means = np.stack([gx.ravel(), gy.ravel(), np.zeros(64)], axis=1).astype(np.float32)
    n_surf = surf_means.shape[0]

    # Cameras at z = -4.0 looking toward +z (at the origin) with small baseline shifts
    cam_positions = [
        [-0.4, -0.2, -4.0],
        [-0.2, -0.2, -4.0],
        [0.0, -0.2, -4.0],
        [0.2, -0.2, -4.0],
        [0.4, -0.2, -4.0],
        [-0.3, 0.2, -4.0],
        [0.0, 0.2, -4.0],
        [0.3, 0.2, -4.0],
    ]
    c2ws = []
    for pos in cam_positions:
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = np.array(pos, dtype=np.float32)
        c2ws.append(c2w)
    c2ws_np = np.stack(c2ws, axis=0)

    # Floater A: opacity=0.12, placed so ONLY cam 0 sees it (far off to -x near cam 0)
    floater_single_view = np.array([[-0.92, -0.20, -3.0]], dtype=np.float32)
    # Floater B: opacity=0.12, floating in front of the consensus surface at (0, 0, -1.8)
    # Every camera sees it at depth ~2.2 in front of the opaque surface at depth ~4.0!
    floater_infront = np.array([[0.0, 0.0, -1.8]], dtype=np.float32)
    # Floater C: opacity=0.12, 3D isolated outlier at (-0.6, 0.6, -1.0) with tiny footprint
    floater_isolated = np.array([[-0.6, 0.6, -1.0]], dtype=np.float32)

    all_means = np.concatenate([surf_means, floater_single_view, floater_infront, floater_isolated], axis=0)
    N = all_means.shape[0]

    opac_vals = np.full(N, 0.98, dtype=np.float32)
    opac_vals[n_surf:] = 0.12  # All 3 floaters have opacity = 0.12 (> 0.05!)

    scale_vals = np.full((N, 3), 0.09, dtype=np.float32)
    scale_vals[n_surf + 0] = 0.02  # single-view floater
    scale_vals[n_surf + 1] = 0.03  # in-front translucent floater
    scale_vals[n_surf + 2] = 0.008  # isolated 3D floater

    splats = {
        "means": torch.from_numpy(all_means).to(device),
        "scales": torch.log(torch.from_numpy(scale_vals)).to(device),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * N, dtype=torch.float32, device=device),
        "opacities": torch.logit(torch.from_numpy(opac_vals)).to(device),
        "sh0": torch.zeros((N, 1, 3), dtype=torch.float32, device=device),
        "shN": torch.zeros((N, 15, 3), dtype=torch.float32, device=device),
    }

    W, H = 128, 128
    K = np.array([[160.0, 0.0, 64.0], [0.0, 160.0, 64.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    data = {
        "camtoworlds": c2ws_np,
        "Ks_dict": {0: K},
        "camera_ids": [0] * len(c2ws),
        "imsize_dict": {0: (W, H)},
        "train_indices": np.arange(len(c2ws)),
        "test_indices": np.array([], dtype=np.int64),
        "image_names": [f"cam_{i}.png" for i in range(len(c2ws))],
        "scene_scale": 1.5,
    }
    return splats, data, n_surf


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for gsplat rasterization")
def test_tidi_detects_floaters_missed_by_op05_and_preserves_surface():
    """Verify that:
    1. `tools/prune_ply.py` (`opacity_min=0.05`) misses all three `opacity=0.12` floaters.
    2. `tools/prune_tidi.py` detects and removes:
       - the 1-camera floater (`active_views < 2`),
       - the multi-view in-front translucent floater (`infront_views > surface_views`),
       - the 3D spatially isolated floater (`knn_dist > sor_002_threshold`).
    3. All 64 valid multi-view consensus surface Gaussians are preserved.
    """
    device = torch.device("cuda:0")
    splats, data, n_surf = _build_synthetic_multi_view_scene(device)
    splats_cpu = {k: v.cpu() for k, v in splats.items()}

    # 1. Baseline prune_ply.py (op < 0.05, bbox 3.0) keeps all 3 floaters because op = 0.12 >= 0.05
    kept_op05, reasons_op05 = prune_op05_only(splats_cpu, opacity_min=0.05, bbox_scale=0.0)
    assert reasons_op05["kept"] == n_surf + 3, "prune_op05 must miss opacity=0.12 floaters"

    # 2. Compute TIDI scores and apply TIDI prune mask
    cfg = TidiConfig(
        min_opacity=0.05,
        bbox_mult=0.0,  # disable bbox filter in this test to isolate cross-view/depth/KNN rules
        min_views=2,
        active_eps=0.02,
        min_importance=0.10,
        delta_infront=0.05,
        tau_surface=0.05,
        infront_max_importance=50.0,
        infront_max_opacity=0.35,
        sor_k=8,
        sor_scale_frac=0.02,
        sor_max_importance=5.0,
        rasterize_mode="classic",
    )
    scores = compute_tidi_scores(splats, data, cfg=cfg, device=device)

    # Check Floater A (index n_surf + 0): seen by <= 1 view
    assert int(scores["active_views"][n_surf + 0]) <= 1

    # Check Floater B (index n_surf + 1): seen by multiple views, but consistently in front of surface!
    assert int(scores["proj_views"][n_surf + 1]) >= 4
    assert int(scores["infront_views"][n_surf + 1]) >= 4
    assert int(scores["surface_views"][n_surf + 1]) == 0

    # Check Floater C (index n_surf + 2): 3D kNN distance much larger than surface grid spacing (~0.10)
    assert float(scores["knn_dist"][n_surf + 2]) > 0.40

    keep, reasons = tidi_keep_mask(scores, cfg=cfg)

    # All 64 consensus surface Gaussians must be preserved
    assert bool(keep[:n_surf].all()), f"Surface Gaussians dropped: {(~keep[:n_surf]).nonzero()}"

    # All 3 floaters (single-view, in-front depth-inconsistent, 3D isolated) must be removed
    assert not bool(keep[n_surf + 0]), "Single-view opacity=0.12 floater was not pruned"
    assert not bool(keep[n_surf + 1]), "In-front depth-inconsistent opacity=0.12 floater was not pruned"
    assert not bool(keep[n_surf + 2]), "3D isolated opacity=0.12 floater was not pruned"
    assert reasons["kept"] == n_surf
    assert reasons["dropped_total"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for gsplat rasterization")
def test_alpha_grad_weights_formula():
    """Verify mathematically that `alpha_grad_weights = opacities * opacities.grad`
    equals `sum_p alpha_{p,i} * T_{p,i} * prod_{j>i} (1 - alpha_{p,j})` = `sum_p w_{p,i} * (1 - A_p) / (1 - alpha_{p,i})`.
    For a single isolated Gaussian layer (`prod_{j>i} (1 - alpha_{p,j}) == 1`), `alpha_grad_weights`
    must match `blending_weights` identically!
    """
    device = torch.device("cuda:0")
    means = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32, device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    scales = torch.tensor([[0.25, 0.25, 0.05]], dtype=torch.float32, device=device)
    opac = torch.tensor([0.35], dtype=torch.float32, device=device)
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    K = torch.tensor([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)

    w_bwd, alpha_grad_w, _, _, _ = compute_view_blending_weights(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opac,
        viewmat=viewmat,
        K=K,
        width=64,
        height=64,
        rasterize_mode="classic",
        compute_alpha_grad=True,
    )
    assert alpha_grad_w is not None
    assert torch.allclose(w_bwd, alpha_grad_w, rtol=1e-4, atol=1e-4), f"{w_bwd} vs {alpha_grad_w}"


def test_delivery_prune_tidi_integration_and_default_off(monkeypatch, tmp_path):
    """Verify `core/delivery.py` `--prune_tidi` integration and confirm default-OFF behavior."""
    import json
    import shutil
    import sys
    from pathlib import Path

    import core.delivery as delivery
    import tools.prune_tidi as prune_tidi_mod

    g = torch.Generator().manual_seed(42)
    n = 40
    splats = {
        "means": torch.rand(n, 3, generator=g) * 2.0 - 1.0,
        "scales": torch.rand(n, 3, generator=g) * 2.0 - 5.0,
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.linspace(-4.0, 4.0, n),  # some below 0.01, some above
        "sh0": torch.zeros((n, 1, 3), dtype=torch.float32),
        "shN": torch.zeros((n, 15, 3), dtype=torch.float32),
    }
    src_ply = tmp_path / "master.ply"
    delivery.export_ply(splats, str(src_ply))

    def fake_splat_transform(in_ply, out_path, actions, gpu="0", iterations=None, summary=False):
        if str(out_path) == "null":
            return "summary"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if str(out_path).endswith(".ply"):
            shutil.copyfile(in_ply, out_path)
        else:
            Path(out_path).write_bytes(b"fake-sog")
        return ""

    c2w = np.stack([np.eye(4) for _ in range(4)])
    for i in range(4):
        c2w[i, :3, 3] = [i * 0.5, 0.0, -3.0]
    fake_colmap = {"camtoworlds": c2w, "train_indices": [0, 1, 2, 3], "test_indices": []}

    monkeypatch.setattr(delivery, "run_splat_transform", fake_splat_transform)
    monkeypatch.setattr(delivery, "load_colmap_data", lambda *a, **k: fake_colmap)
    monkeypatch.setattr(delivery, "object_center", lambda d: np.zeros(3))

    tidi_calls = []

    def fake_prune_ckpt_tidi(source_path, data_dir, factor=2, cfg=None, device="cpu", **kwargs):
        tidi_calls.append({
            "source_path": str(source_path),
            "data_dir": str(data_dir),
            "factor": int(factor),
            "cfg": cfg,
            "device": str(device),
        })
        kept = {k: v[:25].clone() for k, v in splats.items()}
        reasons = {
            "dropped_low_opacity": 5,
            "dropped_bbox_outlier": 1,
            "dropped_low_crossview_evidence": 6,
            "dropped_infront_depth_floater": 2,
            "dropped_3d_isolated_floater": 1,
            "kept": 25,
            "dropped_total": 15,
            "dropped_beyond_op05": 10,
            "kept_frac": 25 / 40,
        }
        payload = {
            "before": {"num_gaussians": 40},
            "pruned": reasons,
            "isolation_diagnostics": {"tidi_knn_gt_sor002_count": 0},
            "after": {"num_gaussians": 25},
        }
        return kept, reasons, payload

    monkeypatch.setattr(prune_tidi_mod, "prune_ckpt_tidi", fake_prune_ckpt_tidi)

    # 1. Default OFF: --prune_tidi not passed -> uses basic prune_dict, never calls prune_ckpt_tidi
    out_off = tmp_path / "out_off"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "core.delivery",
            "--ckpt",
            str(src_ply),
            "--data_dir",
            str(tmp_path),
            "--name",
            "def_off",
            "--out",
            str(out_off),
            "--device",
            "cpu",
            "--gpu",
            "cpu",
            "--no_qa",
            "--no_decimate",
            "--sor_k",
            "0",
        ],
    )
    delivery.main()
    rec_off = json.loads((out_off / "def_off_full_delivery.json").read_text(encoding="utf-8"))
    assert len(tidi_calls) == 0, "prune_ckpt_tidi must not be called when --prune_tidi is OFF"
    assert "prune_tidi" not in rec_off["stages"]
    assert "removed_low_opacity" in rec_off["stages"]["prune"]

    # 2. --prune_tidi ON with --tidi_data_dir, --tidi_data_factor 4, --tidi_mode conservative
    out_on = tmp_path / "out_on"
    tidi_dir = tmp_path / "custom_colmap"
    tidi_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "core.delivery",
            "--ckpt",
            str(src_ply),
            "--data_dir",
            str(tmp_path),
            "--name",
            "tidi_on",
            "--out",
            str(out_on),
            "--device",
            "cpu",
            "--gpu",
            "cpu",
            "--no_qa",
            "--no_decimate",
            "--prune_tidi",
            "--tidi_data_dir",
            str(tidi_dir),
            "--tidi_data_factor",
            "4",
            "--tidi_mode",
            "conservative",
        ],
    )
    delivery.main()
    rec_on = json.loads((out_on / "tidi_on_full_delivery.json").read_text(encoding="utf-8"))
    assert len(tidi_calls) == 1
    assert tidi_calls[0]["data_dir"] == str(tidi_dir)
    assert tidi_calls[0]["factor"] == 4
    assert tidi_calls[0]["cfg"].infront_max_importance == 10.0  # conservative preset
    assert rec_on["stages"]["prune"]["tidi_mode"] == "conservative"
    assert rec_on["stages"]["prune"]["n_before"] == 40
    assert rec_on["stages"]["prune"]["n_after"] == 25
    assert rec_on["stages"]["prune"]["dropped_low_crossview_evidence"] == 6
    assert rec_on["stages"]["prune_tidi"]["kept"] == 25
    assert rec_on["deliverable"]["n_gaussians"] == 25


