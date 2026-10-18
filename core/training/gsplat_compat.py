"""Version shims so our code runs on BOTH gsplat 1.5.3 (system site, production) and gsplat main
1.6.0 (WP-B2, isolated venv `D:/REPOS/HYPERclone/venvs/gsplat_main`, commit 512d366).

API differences that matter to this repo (verified 2026-10-18 against both installs):

* ``gsplat.spherical_harmonics`` — 1.5.3: ``(degrees_to_use, dirs, coeffs, masks=None)`` with
  ``dirs [..., 3]``. main: ``(degrees_to_use, means, viewmats, coeffs, masks=None, ...)`` — the
  view direction is fused into the kernel as ``means - campos`` with ``campos = -R^T t`` taken
  from each world-to-camera matrix. Calling the old signature on main raises
  ``TypeError: missing 1 required positional argument: 'coeffs'``. ``spherical_harmonics_dirs``
  below keeps the 1.5.3 semantics on both: with an identity view matrix the camera sits at the
  origin, so ``means - campos == dirs`` exactly.

Everything else we call is source-compatible (``rasterization``, ``MCMCStrategy``/
``DefaultStrategy`` and their ``step_*`` hooks, ``export_splats``,
``cuda._wrapper.fully_fused_projection``, ``cuda._torch_impl._eval_sh_bases_fast`` /
``_quat_to_rotmat``). Behavioural (not API) changes on main: MCMC ``inject_noise_to_position`` is
a native CUDA kernel with keyword-only ``t``/``k`` (same maths as 1.5.3's ``op_sigmoid(1-o)``,
k=100, t=0.005, different RNG stream) and ``MCMCStrategy`` gained ``noise_injection_stop_iter``.
"""

from __future__ import annotations

import inspect

import torch


def gsplat_version() -> str:
    import gsplat
    return str(getattr(gsplat, "__version__", "unknown"))


def spherical_harmonics_dirs(degrees_to_use: int, dirs: torch.Tensor, coeffs: torch.Tensor,
                             masks: torch.Tensor | None = None) -> torch.Tensor:
    """1.5.3-style SH evaluation ``(degree, dirs[..., 3], coeffs[..., K, D]) -> [..., D]`` on any
    gsplat version."""
    import gsplat
    fn = gsplat.spherical_harmonics
    if "dirs" in inspect.signature(fn).parameters:          # gsplat <= 1.5.x
        return fn(degrees_to_use, dirs, coeffs, masks=masks)
    lead = dirs.shape[:-1]
    d = dirs.reshape(-1, 3).contiguous()
    c = coeffs.reshape(-1, coeffs.shape[-2], coeffs.shape[-1]).contiguous()
    eye = torch.eye(4, device=dirs.device, dtype=dirs.dtype)[None]     # camera at origin
    m = None if masks is None else masks.reshape(1, -1)
    out = fn(degrees_to_use, d, eye, c, masks=m)                        # [1, N, D]
    return out[0].reshape(*lead, coeffs.shape[-1])
