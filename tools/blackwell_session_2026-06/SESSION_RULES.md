# WORKING RULES (from docs/rules.txt) — reference constantly
- Don't make assumptions, always research and make educated decisions
- Break complex problems into smaller pieces
- Check answers from multiple perspectives
- Think through the code logically
- Always double check before saying you're done and it's working
- Don't be lazy. Be thorough. Be methodical. Be smart.
- Don't add complexity for marginal performance gains
- Simpler code that's slightly slower is often better
- Only optimize when profiling shows a real bottleneck

## SESSION GOAL (2026-06-28)
New hardware: Blackwell Pro 6000, 96GB VRAM (was RTX 4090 24GB). Goal: PERFECT gaussian splats.
- Research cutting-edge since June 2026; PARTICULARLY ArtiFixer/Artifix (NVIDIA) + alternatives.
- Re-examine every "ceiling" finding that was VRAM-bound on 24GB now we have 96GB.
- Output: updated, prioritized, hardware-aware plan to push splat quality.

## KEY VRAM-BOUND FINDINGS TO RE-TEST ON 96GB
- "3M cap near-optimal, 8M diverged+OOM@33GB" — divergence may be tunable; OOM no longer binding.
- VGGT OOM'd ~10 imgs on 4090 (no bf16). Blackwell HAS bf16 → larger subsets feasible.
- Train res: factor 1 @ 24MP "infeasible on 24GB" → now realistic.
- Big diffusion models (FLUX.2, larger fixers, SEVA hi-res) now fit locally.
- 3DGRUT/3DGUT Blackwell-era backends now in play.

## BIGGEST UNBUILT FAITHFUL LEVER
- Motion-deblur for 3DGS (BAD-Gaussians 2403.11831, Apache, gsplat-based). Needs real blurry orbit video to validate.
