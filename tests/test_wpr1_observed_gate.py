"""Unit tests for WP-R1: Observed-Preservation Gate for Generative Supervision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from core.training.gsplat_backend import GsplatBackend
from core.training.gsplat_trainer import (
    append_extra_views,
    blend_extra_view_target,
    compute_extra_unobs_weight,
    load_image,
    ssim,
)


def test_compute_extra_unobs_weight_formula():
    """Verify w_unobs(p) = clamp((tau_obs - alpha_init) / tau_obs, 0, 1)^gamma."""
    alpha = torch.tensor([[[0.0], [0.475], [0.95], [0.99]]], dtype=torch.float32)
    w = compute_extra_unobs_weight(alpha, tau_obs=0.95, gamma=1.0)
    expected = torch.tensor([[[1.0], [0.5], [0.0], [0.0]]], dtype=torch.float32)
    assert torch.allclose(w, expected, atol=1e-6)

    w_gamma2 = compute_extra_unobs_weight(alpha, tau_obs=0.95, gamma=2.0)
    expected_g2 = torch.tensor([[[1.0], [0.25], [0.0], [0.0]]], dtype=torch.float32)
    assert torch.allclose(w_gamma2, expected_g2, atol=1e-6)


def test_compute_extra_unobs_weight_with_hole_mask():
    """Explicit hole_mask gates w_unobs (w * hole_mask) so semi-transparent fringes outside the hole get 0."""
    # pixel 0: inside hole (alpha=0.0, hole_mask=1.0) -> w=1.0
    # pixel 1: semi-transparent fringe outside hole (alpha=0.5, hole_mask=0.0) -> gated to 0.0
    # pixel 2: solid observed (alpha=0.99, hole_mask=0.0) -> w=0.0
    alpha = torch.tensor([[[0.0], [0.50], [0.99]]], dtype=torch.float32)
    hole_mask = torch.tensor([[[1.0], [0.0], [0.0]]], dtype=torch.float32)
    w = compute_extra_unobs_weight(alpha, tau_obs=0.95, gamma=1.0, hole_mask=hole_mask)
    expected = torch.tensor([[[1.0], [0.0], [0.0]]], dtype=torch.float32)
    assert torch.allclose(w, expected, atol=1e-6)


def test_blend_extra_view_target():
    """Verify Observed-Anchor Target Blending preserves anchor on w_unobs=0 and pseudo on w_unobs=1."""
    pseudo = torch.ones((4, 4, 3), dtype=torch.float32) * 0.8
    anchor = torch.ones((4, 4, 3), dtype=torch.float32) * 0.2
    w_unobs = torch.zeros((4, 4, 1), dtype=torch.float32)
    w_unobs[:2, :, 0] = 1.0   # top half unobserved
    w_unobs[2, :, 0] = 0.5    # row 2 half-observed
    # row 3 fully observed (w_unobs = 0)

    # Strength 0.0 -> unchanged pseudo
    out0 = blend_extra_view_target(pseudo, anchor, w_unobs, anchor_strength=0.0)
    assert torch.allclose(out0, pseudo)

    # Strength 1.0 -> top half = pseudo (0.8), row 2 = 0.5, row 3 = anchor (0.2)
    out1 = blend_extra_view_target(pseudo, anchor, w_unobs, anchor_strength=1.0)
    assert torch.allclose(out1[:2], torch.full_like(out1[:2], 0.8))
    assert torch.allclose(out1[2], torch.full_like(out1[2], 0.5))
    assert torch.allclose(out1[3], torch.full_like(out1[3], 0.2))


def test_observed_pixels_receive_zero_gradient_under_unobserved_gate():
    """Verify that observed pixels (w_unobs == 0) receive zero gradient from L1 + SSIM when gated."""
    H, W = 32, 32
    rendered = torch.full((H, W, 3), 0.25, dtype=torch.float32, requires_grad=True)
    # Corrupted pseudo-view target everywhere (0.9)
    gt_image = torch.full((H, W, 3), 0.90, dtype=torch.float32)

    # Left half is observed (loss_mask = 0), right half is unobserved hole (loss_mask = 1)
    loss_mask = torch.zeros((H, W), dtype=torch.float32)
    loss_mask[:, W // 2 :] = 1.0
    w_unobs = loss_mask.unsqueeze(-1)

    # Masked L1 + masked SSIM map + gated gt_window (exact trainer logic)
    denom = w_unobs.sum() * rendered.shape[-1] + 1e-8
    l1 = ((rendered - gt_image).abs() * w_unobs).sum() / denom
    gt_window = w_unobs * gt_image + (1.0 - w_unobs) * rendered.detach()
    smap = ssim(rendered.permute(2, 0, 1)[None], gt_window.permute(2, 0, 1)[None], return_map=True)[0]
    ssim_val = (smap * loss_mask).sum() / (loss_mask.sum() * smap.shape[0] + 1e-8)
    loss = 0.8 * l1 + 0.2 * (1.0 - ssim_val)
    loss.backward()

    # Columns well inside the observed half (away from 11x11 SSIM window boundary at W//2)
    obs_grad = rendered.grad[:, : W // 2 - 6, :]
    unobs_grad = rendered.grad[:, W // 2 + 6 :, :]
    assert torch.allclose(obs_grad, torch.zeros_like(obs_grad), atol=1e-7)
    assert unobs_grad.abs().mean().item() > 1e-4


def test_append_extra_views_and_load_image_flags(tmp_path: Path):
    """Verify append_extra_views records extra_view_indices and hole_mask paths."""
    real_img = tmp_path / "real.png"
    extra_img = tmp_path / "extra.png"
    hole_mask_img = tmp_path / "hole_mask.png"
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(real_img)
    Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(extra_img)
    Image.fromarray(np.full((16, 16), 255, dtype=np.uint8)).save(hole_mask_img)

    data = {
        "image_names": ["real.png"],
        "image_paths": [str(real_img)],
        "camtoworlds": np.eye(4, dtype=np.float32)[None],
        "camera_ids": [1],
        "Ks_dict": {1: np.eye(3, dtype=np.float32)},
        "params_dict": {1: np.empty(0, dtype=np.float32)},
        "imsize_dict": {1: (16, 16)},
        "train_indices": np.array([0], dtype=np.int64),
        "test_indices": np.array([], dtype=np.int64),
        "mask_paths": [None],
    }

    manifest = tmp_path / "extra_views.json"
    manifest.write_text(json.dumps([
        {
            "image": str(extra_img),
            "c2w": np.eye(4).tolist(),
            "K": np.eye(3).tolist(),
            "width": 16,
            "height": 16,
            "hole_mask": str(hole_mask_img),
        }
    ]))

    n = append_extra_views(data, str(manifest))
    assert n == 1
    assert data["extra_view_indices"] == {1}
    assert 0 not in data["extra_hole_masks"]
    assert data["extra_hole_masks"][1] == str(hole_mask_img)

    item_real = load_image(data, 0)
    item_extra = load_image(data, 1)
    assert item_real["is_extra_view"] is False
    assert item_extra["is_extra_view"] is True


def test_backend_defaults_and_cli_forwarding():
    """Verify WP-R1 levers are OFF by default in GsplatBackend."""
    backend = GsplatBackend()
    cfg = backend.get_default_config()
    assert cfg["extra_unobserved_only"] is False
    assert cfg["extra_anchor_observed"] == 0.0
    assert cfg["extra_loss_weight"] == 1.0
    assert cfg["extra_obs_tau"] == 0.95
    assert cfg["extra_obs_gamma"] == 1.0
