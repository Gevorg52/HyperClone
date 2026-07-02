# THIRD_PARTY.md — provenance ledger

Lightweight component → source → license ledger for the future funded rebuild
(POC stance: licenses are NOT a selection filter; this exists purely so a
commercial port knows exactly what to re-license or re-implement). See
`docs/EXECUTION_PLAN.md` §0 for the stance.

Format: what we ship · where the idea/code came from · how we adopted it · license of the source.

---

## WP-Q2 — Quick trainer wins (added 2026-07-01)

### (a) Effective-rank anti-needle loss — `core/training/gsplat_trainer.py::erank_loss`
- **Idea/source:** "Effective Rank Analysis and Regularization for Enhanced 3D
  Gaussian Splatting", Hyung et al., **NeurIPS 2024, arXiv:2406.11672**.
- **Adoption:** **RE-IMPLEMENTED from the paper (Eq. 7–10)**, cross-checked against
  the official repo snapshot in `tools/reference_sources/erank_gs/`
  (`utils/get_stats.py::get_effective_rank`, `train.py` erank_loss line). Our
  version adds `.clamp_min(1e-30)` on the eigenvalue sum and `.clamp_min(1e-12)`
  inside `log(p)` — these are OURS, needed because gsplat MCMC relocates dead
  Gaussians to log-scale ≈ −inf whose `exp` underflows to 0 and would NaN the
  entropy. `w_thin` stays 0 (MCMC `scale_reg` already shrinks size). No source
  code copied — a ~10-line pure-PyTorch function.
- **Source repo license:** the official erank_gs repo is built on the Inria
  GRAPHDECO 3DGS base (`loss_utils.py` carries the Inria "free for non-commercial,
  research and evaluation use" header). Since we re-implemented the method rather
  than vendoring the code, the funded port only needs to re-derive from the paper
  (method itself is not the licensed artifact). **Re-license/re-derive: trivial.**

### (b) SA-GS eval-time eps2d anti-aliasing rule — `compute_eps2d` + `render_splats(eps2d=)`
- **Idea/source:** "SA-GS: Scale-Adaptive Gaussian Splatting for Training-Free
  Anti-Aliasing", Song et al., **arXiv:2403.19615**.
- **Adoption:** **IDEA ONLY.** SA-GS ships a custom CUDA rasterizer; we do NOT use
  it. gsplat already exposes the exact same 2D screen-space dilation knob
  (`rasterization(eps2d=...)`, default 0.3), so we implement the scale-adaptive
  rule as a ~2-line pure-Python helper `compute_eps2d(...) = 0.3 * r²` and pass it
  through at zoom/eval render time (+ optional `eval_ss` supersampling). No SA-GS
  code copied. Reference snapshot (their CUDA + train scripts, **Apache-2.0**) at
  `tools/reference_sources/sags/` was consulted for the formula only.
- **Source repo license:** Apache-2.0. **Re-license: none needed** (we wrote our own
  code; the idea is unpatentable math over gsplat's existing API).

---

## Pre-existing components (recorded for completeness — not added by WP-Q2)
- **gsplat** v1.5.3 (AOT sm_120) — nerfstudio-project/gsplat — Apache-2.0. Core rasterizer/MCMC.
- **Mip-Splatting** antialiased mode — via gsplat `rasterize_mode="antialiased"` — used as-is.
