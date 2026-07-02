# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PPISP Report Export

Export PDF reports visualizing learned PPISP parameters:
- Exposure compensation (per-frame)
- Vignetting (per-camera, per-channel)
- Color correction (per-frame)
- Camera Response Function / tone mapping (per-camera)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import __version__

if TYPE_CHECKING:
    from . import PPISP


def _srgb_inverse_oetf(x: np.ndarray) -> np.ndarray:
    """Inverse sRGB OETF (EOTF): pre-compensation so that a later sRGB OETF yields identity."""
    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    out = np.empty_like(x, dtype=np.float32)
    mask = x <= 0.04045
    out[mask] = x[mask] / 12.92
    inv = x[~mask]
    out[~mask] = np.power((inv + 0.055) / 1.055, 2.4, dtype=np.float32)
    return out


def _gray_bars(size: int = 256, num: int = 16) -> np.ndarray:
    w = size // num
    img = np.zeros((size, size, 3), dtype=np.float32)
    for i in range(num):
        v = i / (num - 1)
        img[:, i * w: size if i == num - 1 else (i + 1) * w] = v
    return img


def _show_image(ax, img: np.ndarray, title: str):
    ax.imshow(np.clip(img, 0.0, 1.0))
    ax.axis("off")
    ax.set_title(title)


# =============================================================================
# Exposure Plotting
# =============================================================================


def _plot_exposure(
    fig,
    gs,
    exposure_params: torch.Tensor,
    frames_per_camera: list[int],
    cam: int,
):
    sub = gs.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    ax_plot = fig.add_subplot(sub[0])
    ax_img = fig.add_subplot(sub[1])

    start = int(sum(frames_per_camera[:cam]))
    end = start + int(frames_per_camera[cam])
    vals = exposure_params[start:end].detach().float().cpu()
    mean_val = vals.mean().item() if vals.numel() else 0.0

    ax_plot.plot(np.arange(vals.numel()), vals.numpy(), "b-")
    ax_plot.axhline(mean_val, color="b", linestyle="--", alpha=0.5)
    ax_plot.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax_plot.set_xlabel("Frame Index")
    ax_plot.set_ylabel("Exposure Offset [EV]")
    ax_plot.set_title("Exposure Offset Over Time")
    ax_plot.grid(True, alpha=0.3)

    img = _gray_bars(256)
    scale = 2.0 ** float(mean_val)
    img[img.shape[0] // 2:, :] *= scale
    _show_image(ax_img, img, "Mean Exposure Visualization")
    size = img.shape[0]
    ax_img.text(
        size * 0.5, 20, "Original", ha="center", va="top", color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )
    ax_img.text(
        size * 0.5, size - 20, f"{mean_val:+.2f} EV", ha="center", va="bottom",
        color="white", bbox=dict(facecolor="black", alpha=0.5),
    )


# =============================================================================
# Vignetting Plotting
# =============================================================================


def _vig_weight_forward(r2: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    """Exact forward falloff: 1 + sum a_i * r2^(i+1), clamped to [0,1]."""
    falloff = torch.ones_like(r2)
    r2_pow = r2
    for i in range(int(alphas.shape[-1])):
        falloff = falloff + alphas[..., i] * r2_pow
        r2_pow = r2_pow * r2
    return torch.clamp(falloff, 0.0, 1.0)


def _plot_vignetting(fig, gs, vig_params: torch.Tensor, cam: int):
    sub = gs.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    ax_plot = fig.add_subplot(sub[0])
    ax_img = fig.add_subplot(sub[1])

    device = vig_params.device
    r = torch.linspace(0, np.sqrt(2) / 2.0, 200, device=device)
    r2 = r * r

    colors = [(1, 0, 0, 0.6), (0, 1, 0, 0.6), (0, 0, 1, 0.6)]
    # Per-channel vignetting curves
    for ch in range(3):
        alphas = vig_params[cam, ch, 2:]
        w = _vig_weight_forward(r2, alphas)
        ax_plot.plot(
            r.detach().cpu().numpy(), w.detach().cpu().numpy(),
            color=colors[ch], linewidth=2.0, label=["Red", "Green", "Blue"][ch],
        )
    ax_plot.set_title("Vignetting Curves (R,G,B)")

    ax_plot.set_xlabel("Radial Distance")
    ax_plot.set_ylabel("Light Transmission")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.legend()
    ax_plot.set_ylim(bottom=0)

    # Vignette visualization on square uv grid centered at 0
    size = 256
    coords = torch.linspace(-0.5, 0.5, size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    img = torch.full((size, size, 3), 0.75, device=device)

    # Per-channel vignetting
    for ch in range(3):
        oc = vig_params[cam, ch, :2]
        alphas = vig_params[cam, ch, 2:]
        dx = xx - oc[0]
        dy = yy - oc[1]
        r2m = dx * dx + dy * dy
        w = _vig_weight_forward(r2m, alphas)
        img[..., ch] = img[..., ch] * w

    img_np = img.detach().cpu().numpy()
    ax_img.imshow(np.clip(img_np, 0.0, 1.0))
    ax_img.axis("off")
    ax_img.set_title("Vignetting Effect Visualization")
    center = size * 0.5
    ax_img.axhline(y=center, color="gray", linestyle="--", alpha=0.5)
    ax_img.axvline(x=center, color="gray", linestyle="--", alpha=0.5)
    # Overlay crosses at optical centers
    cross_size = 10
    cross_width = 2
    for ch, color in enumerate(colors):
        oc = vig_params[cam, ch, :2].detach().cpu().numpy()
        cx = (float(oc[0]) + 0.5) * size
        cy = (float(oc[1]) + 0.5) * size
        ax_img.plot([cx - cross_size, cx + cross_size],
                    [cy, cy], color=color, linewidth=cross_width)
        ax_img.plot([cx, cx], [cy - cross_size, cy + cross_size],
                    color=color, linewidth=cross_width)


# =============================================================================
# Color Correction Plotting (Standard PPISP)
# =============================================================================

# ZCA pinv blocks for color correction [Blue, Red, Green, Neutral]
_PINV_BLOCKS = torch.tensor([
    [[0.0480542, -0.0043631], [-0.0043631, 0.0481283]],
    [[0.0580570, -0.0179872], [-0.0179872, 0.0431061]],
    [[0.0433336, -0.0180537], [-0.0180537, 0.0580500]],
    [[0.0128369, -0.0034654], [-0.0034654, 0.0128158]],
])


def _color_offsets_from_params(p: torch.Tensor) -> torch.Tensor:
    """Map latent 8-dim color params per frame to real RG offsets via ZCA blocks.

    Args:
        p: [num_frames, 8] tensor ordered as pairs per chromaticity:
           [blue(dr,dg), red(dr,dg), green(dr,dg), neutral(dr,dg)]

    Returns:
        Tensor of shape [num_frames, 4, 2] with real-space RG offsets for
        [blue, red, green, neutral] in that order.
    """
    device = p.device
    dtype = p.dtype
    pinv = _PINV_BLOCKS.to(device=device, dtype=dtype)

    def _mul2(a: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            a[..., 0] * m[0, 0] + a[..., 1] * m[0, 1],
            a[..., 0] * m[1, 0] + a[..., 1] * m[1, 1],
        ], dim=-1)

    xb, xr, xg, xn = p[..., 0:2], p[..., 2:4], p[..., 4:6], p[..., 6:8]
    yb = _mul2(xb, pinv[0])
    yr = _mul2(xr, pinv[1])
    yg = _mul2(xg, pinv[2])
    yn = _mul2(xn, pinv[3])
    return torch.stack([yb, yr, yg, yn], dim=1)


def _source_chroms(device: torch.device) -> torch.Tensor:
    # RG chromaticities for [Blue, Red, Green, Neutral]
    return torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.33, 0.33]], device=device)


def _homography_from_params(p: torch.Tensor) -> torch.Tensor:
    """Construct H from RG offsets p for [blue, red, green, gray].

    p shape: [..., 8] ordered as pairs per chromaticity.
    """
    device = p.device
    dtype = p.dtype

    offsets = _color_offsets_from_params(p)  # [...,4,2]
    bd = offsets[..., 0, :]
    rd = offsets[..., 1, :]
    gd = offsets[..., 2, :]
    nd = offsets[..., 3, :]

    # Sources
    s_b = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    s_r = torch.tensor([1.0, 0.0, 1.0], device=device, dtype=dtype)
    s_g = torch.tensor([0.0, 1.0, 1.0], device=device, dtype=dtype)
    s_gray = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0],
                          device=device, dtype=dtype)

    t_b = torch.stack([s_b[0] + bd[..., 0], s_b[1] +
                      bd[..., 1], torch.ones_like(bd[..., 0])], dim=-1)
    t_r = torch.stack([s_r[0] + rd[..., 0], s_r[1] +
                      rd[..., 1], torch.ones_like(rd[..., 0])], dim=-1)
    t_g = torch.stack([s_g[0] + gd[..., 0], s_g[1] +
                      gd[..., 1], torch.ones_like(gd[..., 0])], dim=-1)
    t_gray = torch.stack([s_gray[0] + nd[..., 0], s_gray[1] +
                         nd[..., 1], torch.ones_like(nd[..., 0])], dim=-1)

    T = torch.stack([t_b, t_r, t_g], dim=-1)  # [...,3,3]

    zero = torch.zeros_like(bd[..., 0])
    skew = torch.stack([
        torch.stack([zero, -t_gray[..., 2], t_gray[..., 1]], dim=-1),
        torch.stack([t_gray[..., 2], zero, -t_gray[..., 0]], dim=-1),
        torch.stack([-t_gray[..., 1], t_gray[..., 0], zero], dim=-1),
    ], dim=-2)

    M = torch.matmul(skew, T)
    r0, r1, r2 = M[..., 0, :], M[..., 1, :], M[..., 2, :]
    lam = torch.cross(r0, r1, dim=-1)
    n2 = (lam * lam).sum(dim=-1)
    mask = n2 < 1.0e-20
    lam = torch.where(mask.unsqueeze(-1), torch.cross(r0, r2, dim=-1), lam)
    n2 = (lam * lam).sum(dim=-1)
    mask = n2 < 1.0e-20
    lam = torch.where(mask.unsqueeze(-1), torch.cross(r1, r2, dim=-1), lam)

    S_inv = torch.tensor([[-1.0, -1.0, 1.0], [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]], device=device, dtype=dtype)
    D = torch.zeros(*p.shape[:-1], 3, 3, device=device, dtype=dtype)
    D[..., 0, 0] = lam[..., 0]
    D[..., 1, 1] = lam[..., 1]
    D[..., 2, 2] = lam[..., 2]
    H = torch.matmul(T, torch.matmul(D, S_inv))
    s = H[..., 2:3, 2:3]
    denom = s + (s.abs() <= 1.0e-20).to(dtype)
    H = H / denom
    return H


def _apply_h_rg_loss(h: torch.Tensor, rg: torch.Tensor) -> torch.Tensor:
    """Loss mapping: apply H to (r,g,1) and divide xy by z."""
    r, g = rg[..., 0], rg[..., 1]
    ones = torch.ones_like(r)
    v = torch.stack([r, g, ones], dim=-1)
    vv = torch.matmul(h, v.unsqueeze(-1)).squeeze(-1)
    denom = vv[..., 2] + 1.0e-5
    return torch.stack([vv[..., 0] / denom, vv[..., 1] / denom], dim=-1)


def _apply_h_rgb_forward(h: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Forward mapping: RGB -> RGI, apply H, intensity-preserving renormalization, back to RGB."""
    intensity = rgb[..., 0] + rgb[..., 1] + rgb[..., 2]
    rgi = torch.stack([rgb[..., 0], rgb[..., 1], intensity], dim=-1)
    rgi_m = torch.matmul(h, rgi.unsqueeze(-1)).squeeze(-1)
    scale = intensity / (rgi_m[..., 2] + 1.0e-5)
    rgi_m = rgi_m * scale.unsqueeze(-1)
    r, g = rgi_m[..., 0], rgi_m[..., 1]
    b = rgi_m[..., 2] - r - g
    return torch.stack([r, g, b], dim=-1)


def _dlt_homography(src_rg: np.ndarray, dst_rg: np.ndarray) -> np.ndarray:
    """Solve for H (3x3) mapping (r,g,1)->(r',g',1) using 4 RG correspondences via DLT."""
    A = []
    for (x, y), (u, v) in zip(src_rg, dst_rg):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A, dtype=np.float64)
    _, _, vh = np.linalg.svd(A)
    h = vh[-1]
    H = h.reshape(3, 3)
    if abs(H[2, 2]) < 1e-8:
        H = H / (np.sign(H[2, 2]) + 1e-8)
    else:
        H = H / H[2, 2]
    return H


def _chrom_triangle_size(size: int) -> tuple[int, int]:
    height = int(size * np.sqrt(3.0) / 2.0)
    return size, height


def _chrom_barycentric_to_window(r: float, g: float, size: int) -> tuple[float, float]:
    width, height = _chrom_triangle_size(size)
    top = (width * 0.5, 0.0)
    bl = (0.0, float(height))
    br = (float(width), float(height))
    b = 1.0 - r - g
    x = r * bl[0] + g * br[0] + b * top[0]
    y = r * bl[1] + g * br[1] + b * top[1]
    return x, y


@lru_cache(maxsize=4)
def _create_chromaticity_triangle(size: int) -> np.ndarray:
    width, height = _chrom_triangle_size(size)
    img = np.ones((height, width, 3), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            top = np.array([width * 0.5, 0.0])
            bl = np.array([0.0, height])
            br = np.array([width, height])
            v0 = bl - top
            v1 = br - top
            v2 = np.array([x, y], dtype=np.float32) - top
            d00 = (v0 * v0).sum()
            d01 = (v0 * v1).sum()
            d11 = (v1 * v1).sum()
            d20 = (v2 * v0).sum()
            d21 = (v2 * v1).sum()
            denom = d00 * d11 - d01 * d01
            if denom == 0:
                continue
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            if (u >= 0.0) and (v >= 0.0) and (w >= 0.0):
                r_val, g_val = v, w
                b_val = max(0.0, 1.0 - r_val - g_val)
                rgb = np.array([r_val, g_val, b_val], dtype=np.float32)
                m = rgb.max()
                if m > 0:
                    rgb = rgb / m
                img[y, x] = rgb * 0.85 + 0.15
    return img


def _plot_color(
    fig,
    gs_top,
    gs_bot,
    color_params: torch.Tensor,
    frames_per_camera: list[int],
    cam: int,
):
    start = int(sum(frames_per_camera[:cam]))
    n = int(frames_per_camera[cam])
    p = color_params[start: start + n]
    H = _homography_from_params(p)
    src = _source_chroms(color_params.device)
    tgt_list = []
    for i in range(4):
        rg_in = src[i].unsqueeze(0).expand(n, -1)
        tgt_i = _apply_h_rg_loss(H, rg_in)
        tgt_list.append(tgt_i)
    tgt = torch.stack(tgt_list, dim=1)
    shifts = (tgt - src)

    names = ["Blue", "Red", "Green", "Neutral"]
    cols = ["blue", "red", "green", "gray"]

    sub_top = gs_top.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    sub_bot = gs_bot.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    ax_rc = fig.add_subplot(sub_top[0])
    ax_rgplot = fig.add_subplot(sub_top[1])
    ax_gm = fig.add_subplot(sub_bot[0])
    ax_img = fig.add_subplot(sub_bot[1])

    x = np.arange(n)
    for i in range(4):
        ax_rc.plot(x, shifts[:, i, 0].detach().cpu().numpy(),
                   color=cols[i], label=names[i], alpha=0.8)
        ax_gm.plot(x, shifts[:, i, 1].detach(
        ).cpu().numpy(), color=cols[i], alpha=0.8)
    ax_rc.set_title("Red-Cyan Shift Over Time")
    ax_rc.set_xlabel("Frame Index")
    ax_rc.set_ylabel("Red-Cyan Shift")
    ax_rc.grid(True, alpha=0.3)
    ax_rc.legend()

    ax_gm.set_title("Green-Magenta Shift Over Time")
    ax_gm.set_xlabel("Frame Index")
    ax_gm.set_ylabel("Green-Magenta Shift")
    ax_gm.grid(True, alpha=0.3)

    size = 256
    scale = 5.0
    tri_img = _create_chromaticity_triangle(size)
    ax_rgplot.imshow(tri_img)
    ax_rgplot.axis("off")
    ax_rgplot.set_title(f"Chromaticity Shifts Over Time, Scaled {scale:.1f}x")

    chroms_scaled = (src + shifts * scale).detach().cpu().numpy()
    cross_size = 7
    cross_width = 2
    for i in range(4):
        pts = chroms_scaled[:, i, :]
        traj = np.array([_chrom_barycentric_to_window(
            float(r_val), float(g_val), size) for r_val, g_val in pts])
        ax_rgplot.plot(traj[:, 0], traj[:, 1], "-",
                       color="black", linewidth=1.0, alpha=0.7)
        fx, fy = traj[-1]
        ax_rgplot.plot([fx - cross_size, fx + cross_size],
                       [fy, fy], "-", color="black", linewidth=cross_width)
        ax_rgplot.plot([fx, fx], [fy - cross_size, fy + cross_size],
                       "-", color="black", linewidth=cross_width)
        sx, sy = _chrom_barycentric_to_window(
            float(src[i, 0].item()), float(src[i, 1].item()), size)
        ax_rgplot.plot(
            [sx - cross_size * 0.75, sx + cross_size * 0.75], [sy, sy],
            "-", color="black", linewidth=cross_width / 2, alpha=0.5,
        )
        ax_rgplot.plot(
            [sx, sx], [sy - cross_size * 0.75, sy + cross_size * 0.75],
            "-", color="black", linewidth=cross_width / 2, alpha=0.5,
        )

    mean_targets = tgt.mean(dim=0).detach().cpu().numpy()
    src_np = src.detach().cpu().numpy()
    H_np = _dlt_homography(src_np, mean_targets)
    H_mean = torch.from_numpy(H_np).to(
        color_params.device, dtype=color_params.dtype)

    size = 256
    bars = np.zeros((size, size, 3), dtype=np.float32)
    w = size // 4
    bars[:, 0:w] = [0, 0, 1]
    bars[:, w: 2 * w] = [1, 0, 0]
    bars[:, 2 * w: 3 * w] = [0, 1, 0]
    bars[:, 3 * w:] = [0.5, 0.5, 0.5]
    bottom = torch.from_numpy(
        bars[size // 2:].reshape(-1, 3)).to(color_params.device)
    corrected = _apply_h_rgb_forward(H_mean, bottom)
    vis = bars.copy()
    vis[size // 2:] = corrected.reshape(size // 2,
                                        size, 3).clamp(0, 1).detach().cpu().numpy()
    _show_image(ax_img, vis, "Mean Color Correction Visualization")
    ax_img.text(
        size * 0.5, 20, "Original", ha="center", va="top", color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )
    ax_img.text(
        size * 0.5, size - 20, f"Color Corrected, Scaled {scale:.1f}x", ha="center", va="bottom",
        color="white", bbox=dict(facecolor="black", alpha=0.5),
    )


# =============================================================================
# CRF Plotting
# =============================================================================


def _softplus_with_min(x: torch.Tensor, min_value: float) -> torch.Tensor:
    return torch.tensor(min_value, device=x.device, dtype=x.dtype) + torch.log1p(torch.exp(x))


def _crf_effective_from_raw(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map raw CRF params to effective toe, shoulder, gamma, center."""
    toe = _softplus_with_min(raw[..., 0], 0.3)
    shoulder = _softplus_with_min(raw[..., 1], 0.3)
    gamma = _softplus_with_min(raw[..., 2], 0.1)
    center = torch.sigmoid(raw[..., 3])
    return toe, shoulder, gamma, center


def _apply_crf(
    toe: torch.Tensor,
    shoulder: torch.Tensor,
    gamma: torch.Tensor,
    center: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Apply CRF with center split."""
    x = torch.clamp(x, 0.0, 1.0)
    a = (shoulder * center) / \
        torch.clamp(torch.lerp(toe, shoulder, center), min=1.0e-12)
    b = 1.0 - a
    left = a * \
        torch.pow(torch.clamp(
            x / torch.clamp(center, min=1.0e-12), 0.0, 1.0), toe)
    right = 1.0 - b * torch.pow(torch.clamp((1.0 - x) /
                                torch.clamp(1.0 - center, min=1.0e-12), 0.0, 1.0), shoulder)
    y0 = torch.where(x <= center, left, right)
    y = torch.pow(torch.clamp(y0, 0.0, 1.0), gamma)
    return torch.clamp(y, 0.0, 1.0)


def _plot_crf(fig, gs, crf_params: torch.Tensor, cam: int):
    sub = gs.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    ax_plot = fig.add_subplot(sub[0])
    ax_img = fig.add_subplot(sub[1])

    cols = [(1, 0, 0, 0.6), (0, 1, 0, 0.6), (0, 0, 1, 0.6)]
    x = torch.linspace(0.0, 1.0, 256, device=crf_params.device)
    crf_cam = crf_params[cam]
    for ch in range(3):
        toe, shoulder, gamma, center = _crf_effective_from_raw(crf_cam[ch])
        y = _apply_crf(toe, shoulder, gamma, center, x)
        ax_plot.plot(
            x.detach().cpu().numpy(), y.detach().cpu().numpy(),
            color=cols[ch], linewidth=2.0, label=["Red", "Green", "Blue"][ch],
        )
        center_x = float(center.detach().cpu().item())
        center_y = float(_apply_crf(toe, shoulder, gamma,
                         center, center).detach().cpu().item())
        ax_plot.scatter([center_x], [center_y], color=cols[ch],
                        s=18, zorder=6, marker="o")
    ax_plot.axvline(1.0, color="black", linestyle="--", alpha=0.5)
    ax_plot.set_xlabel("Linear Input Intensity")
    ax_plot.set_ylabel("Output Intensity")
    ax_plot.set_title("Camera Response Function (R, G, B)")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.legend()

    img = _gray_bars(256)
    xin = torch.from_numpy(
        img[img.shape[0] // 2:, :, 0].copy()).to(crf_params.device)
    xin_lin = xin.flatten()
    for ch in range(3):
        toe, shoulder, gamma, center = _crf_effective_from_raw(crf_cam[ch])
        y = _apply_crf(toe, shoulder, gamma, center, xin_lin)
        img[img.shape[0] // 2:, :,
            ch] = y.reshape(img.shape[0] // 2, img.shape[1]).cpu().numpy()
    img = _srgb_inverse_oetf(img)
    _show_image(ax_img, img, "Tone Mapping Visualization")
    size = img.shape[0]
    ax_img.text(
        size * 0.5, 20, "Linear", ha="center", va="top", color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )
    ax_img.text(
        size * 0.5, size - 20, "Tone Mapped", ha="center", va="bottom", color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )


# =============================================================================
# Public API
# =============================================================================

@torch.no_grad()
def export_ppisp_report(
    ppisp: PPISP,
    frames_per_camera: list[int],
    output_dir: Path,
    camera_names: list[str] | None = None,
) -> list[Path]:
    """Generate PDF reports visualizing learned PPISP parameters.

    Creates one PDF per camera showing:
    - Exposure compensation over time
    - Vignetting curves and effect visualization
    - Color correction shifts over time (chromaticity diagram)
    - Camera response function curves

    Also exports a JSON file with all parameter values.

    Args:
        ppisp: Trained PPISP module instance.
        frames_per_camera: List of frame counts per camera.
        output_dir: Directory to write output files.
        camera_names: Optional list of camera names (defaults to "camera_0", etc.).

    Returns:
        List of paths to written PDF files.
    """
    matplotlib.use("Agg", force=True)

    num_cams = int(ppisp.num_cameras)

    if camera_names is None:
        camera_names = [f"camera_{i}" for i in range(num_cams)]

    assert len(camera_names) == num_cams, (
        f"camera_names length {len(camera_names)} != num_cameras {num_cams}"
    )
    assert len(frames_per_camera) == num_cams, (
        f"frames_per_camera length {len(frames_per_camera)} != num_cameras {num_cams}"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []

    for cam in range(num_cams):
        num_rows = 5
        fig = plt.figure(figsize=(20, 5 * num_rows))
        gs = fig.add_gridspec(num_rows, 1, height_ratios=[1] * num_rows)

        row_idx = 0

        # Row 0: Exposure
        _plot_exposure(fig, gs[row_idx],
                       ppisp.exposure_params, frames_per_camera, cam)
        row_idx += 1

        # Row 1: Vignetting
        _plot_vignetting(fig, gs[row_idx], ppisp.vignetting_params, cam)
        row_idx += 1

        # Rows 2-3: Color correction
        _plot_color(fig, gs[row_idx], gs[row_idx + 1],
                    ppisp.color_params, frames_per_camera, cam)
        row_idx += 2

        # Row 4: CRF
        _plot_crf(fig, gs[row_idx], ppisp.crf_params, cam)

        plt.tight_layout()
        cam_label = camera_names[cam] if camera_names[cam] else f"camera_{cam}"
        safe_label = "".join(c if c.isalnum() or c in (
            "-", "_") else "_" for c in cam_label)
        out_path = output_dir / f"{safe_label}_ppisp_report.pdf"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)

    # Export JSON with parameter values
    _export_params_json(ppisp, frames_per_camera, camera_names, output_dir)

    return outputs


def _export_params_json(
    ppisp: PPISP,
    frames_per_camera: list[int],
    camera_names: list[str],
    output_dir: Path,
) -> Path:
    """Export PPISP parameters to JSON."""
    num_cams = int(ppisp.num_cameras)

    # Exposure
    exposure_params_raw = ppisp.exposure_params.detach().float().cpu()
    exposure_vals = exposure_params_raw.tolist()

    # Vignetting
    vig = {}
    for cam in range(num_cams):
        cam_label = camera_names[cam] if camera_names[cam] else f"camera_{cam}"
        safe_label = "".join(c if c.isalnum() or c in (
            "-", "_") else "_" for c in cam_label)
        vig_cam = {}
        # Per-channel vignetting
        for ch, ch_name in enumerate(["red", "green", "blue"]):
            oc = ppisp.vignetting_params[cam, ch,
                                         :2].detach().float().cpu().tolist()
            alphas = ppisp.vignetting_params[cam,
                                             ch, 2:].detach().float().cpu().tolist()
            vig_cam[ch_name] = {"optical_center": oc, "alphas": alphas}
        vig[safe_label] = vig_cam

    # Color
    color_per_frame = []
    color_real = _color_offsets_from_params(
        ppisp.color_params).detach().float().cpu().numpy()
    for i in range(color_real.shape[0]):
        color_per_frame.append({
            "blue": color_real[i, 0].tolist(),
            "red": color_real[i, 1].tolist(),
            "green": color_real[i, 2].tolist(),
            "neutral": color_real[i, 3].tolist(),
        })

    # CRF
    crf = {}
    for cam in range(num_cams):
        cam_label = camera_names[cam] if camera_names[cam] else f"camera_{cam}"
        safe_label = "".join(c if c.isalnum() or c in (
            "-", "_") else "_" for c in cam_label)
        crf_cam = {}
        crf_cam_params = ppisp.crf_params[cam]
        for ch, ch_name in enumerate(["red", "green", "blue"]):
            raw = crf_cam_params[ch].detach().float().cpu()
            toe, shoulder, gamma, center = _crf_effective_from_raw(
                crf_cam_params[ch])
            crf_cam[ch_name] = {
                "raw": {
                    "toe_raw": float(raw[0].item()),
                    "shoulder_raw": float(raw[1].item()),
                    "gamma_raw": float(raw[2].item()),
                    "center_raw": float(raw[3].item()),
                },
                "effective": {
                    "toe": float(toe.detach().cpu().item()),
                    "shoulder": float(shoulder.detach().cpu().item()),
                    "gamma": float(gamma.detach().cpu().item()),
                    "center": float(center.detach().cpu().item()),
                },
            }
        crf[safe_label] = crf_cam

    json_obj = {
        "ppisp_version": __version__,
        "frames_per_camera": frames_per_camera,
        "exposure": {
            "per_frame": exposure_vals,
        },
        "vignetting": {"by_camera": vig},
        "color": {"per_frame": color_per_frame},
        "crf": {"by_camera": crf},
    }

    json_out = output_dir / "ppisp_params.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2)

    return json_out
