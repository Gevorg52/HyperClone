"""Smoke test: gsplat JIT compile + antialiased rasterization + MCMC API.

Run manually after gsplat upgrades:
    python tests/smoke_gsplat.py
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from core.training import gsplat_jit_shim  # noqa: F401  (before gsplat import)

import gsplat
from gsplat import export_splats  # noqa: F401  (PLY export availability)
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy


def main():
    print("gsplat", gsplat.__version__)
    print("MCMC initialize_state:", inspect.signature(MCMCStrategy.initialize_state))
    print("MCMC step_post_backward:", inspect.signature(MCMCStrategy.step_post_backward))

    device = "cuda"
    N = 1000
    means = torch.randn(N, 3, device=device)
    quats = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = torch.rand(N, 3, device=device) * 0.1
    opac = torch.rand(N, device=device)
    colors = torch.rand(N, 3, device=device)
    viewmat = torch.eye(4, device=device)[None]
    K = torch.tensor([[300.0, 0, 150], [0, 300.0, 100], [0, 0, 1]], device=device)[None]

    img, alpha, info = rasterization(
        means, quats, scales, opac, colors, viewmat, K, 300, 200,
        rasterize_mode="antialiased",
    )
    print("antialiased rasterization OK:", tuple(img.shape), img.dtype)

    img2, _, _ = rasterization(
        means, quats, scales, opac, colors, viewmat, K, 300, 200,
        rasterize_mode="classic",
    )
    print("classic rasterization OK:", tuple(img2.shape))


if __name__ == "__main__":
    main()
