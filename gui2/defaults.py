"""gui2 — single source of truth for the two product modes' validated defaults.

Mirrors the design doc (`docs/GUI_REDESIGN.md` §2 mode-defaults table and §4 settings
inventory) EXACTLY. The Advanced drawer in `gui2/app.py` is built by iterating `SETTINGS`
and seeding each control from `defaults_for(mode)` — so this file is the ONLY place a
default value lives. The UI-1 smoke gate asserts the drawer values equal `defaults_for`
byte-for-byte, which guards against a value being hard-coded elsewhere.

Every lever here is default-ON only because it is part of the validated WP-S1 stack
(all compose, no interactions — see EVIDENCE). Nothing here turns a *failed*-gate lever
on (WD-R, ungated LPIPS, planar-at-paper-lambdas are all OFF / hidden per house rules).

Design table (§2), the two modes:

  | Setting        | Object Scan            | Environment Scan       |
  |----------------|------------------------|------------------------|
  | data_factor    | auto -> nearest ~6 MP  | auto -> ~6 MP          |
  | Iterations/cap | 30k / MCMC 3M          | 30k / 3M               |
  | Levers ON      | dense_init, mono_depth | same                   |
  |                | 0.05, erank 0.01@7k,   |                        |
  |                | gated-LPIPS 0.2,       |                        |
  |                | pose_opt, antialiased, |                        |
  |                | c2f                    |                        |
  | Planar         | off                    | off (Advanced: flatten)|
  | Delivery       | object SOG (SH3,       | full SOG (SH2,         |
  |                | extracted) + full SOG  | decimated) + PLY master|
  | Detail toggle  | ~15 MP + pose_opt OFF  | same                   |
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# The validated stack, shared by both modes (WP-S1: all levers compose)
# ---------------------------------------------------------------------------

BASE_TRAINING: dict[str, Any] = {
    # Resolution — auto picks the data_factor whose training MP is nearest target_mp.
    "data_factor": "auto",       # "auto" | 1 | 2 | 4 | 8
    "target_mp": 6.0,            # design: validated ~6 MP training regime (WP-S1/T1)
    # Capacity / schedule
    "max_steps": 30_000,         # convergence evidence (X1)
    "strategy": "mcmc",          # MCMC-3DGS densification (proven quality base)
    "cap_max": 3_000_000,        # 3M near-optimal; more HURTS (N1/X1/§6)
    # Levers ON (WP-S1 stack)
    "dense_init": True,          # EDGS-style RoMa dense triangulation seed (WP-F2)
    "mono_depth_lambda": 0.05,   # MoGe-2 dense mono-depth prior (WP-F1)
    "erank_lambda": 0.01,        # effective-rank anti-needle hinge (2406.11672)
    "erank_start": 7_000,        # hinge starts at 7k (of a 30k schedule)
    "lpips_lambda": 0.2,         # perceptual sharpness lever (gated -> safe any res)
    "lpips_gated": True,         # gate toward under-resolved crops (§4 / DEV_NOTES §4)
    "pose_opt": True,            # SE(3) per-camera refinement
    "antialiased": True,         # Mip-Splatting
    "c2f": True,                 # coarse-to-fine: densify low-res, finish full
    # Planar losses — OFF (WP-F4: intrinsic perceptual cost at the paper's lambdas).
    # Exposed in Advanced ("flatten walls/floors") for geometry-only Environment use.
    "planar_lambda": 0.0,
    # Bookkeeping
    "seed": 0,
    "test_every": 8,             # hold out every Nth image for eval
    # Delivery (§4 exposes SH level, decimation %, extraction on/off)
    "decimate": "60%",           # splat-transform --decimate target
}

OBJECT_DEFAULTS: dict[str, Any] = {
    **BASE_TRAINING,
    "extract_object": True,      # delivery carves the object out (WP-S2 vote+refit)
    "filter_harmonics": 3,       # object SOG keeps SH3 (specular; WP-D1)
}

ENV_DEFAULTS: dict[str, Any] = {
    **BASE_TRAINING,
    "extract_object": False,     # full-scene delivery, no carve
    "filter_harmonics": 2,       # full SOG truncates to SH2 (size; within QA bound)
}

# "Detail mode" toggle (§2 last row): sharper microstructure, slower.
DETAIL_MODE_OVERRIDES: dict[str, Any] = {
    "target_mp": 15.0,           # factor -> ~15 MP
    "pose_opt": False,           # WP-N1 native recipe: pose_opt OFF at native res
}


# ---------------------------------------------------------------------------
# Plain-language help + evidence citations (design §4: one-line help + citation)
# ---------------------------------------------------------------------------

HELP: dict[str, str] = {
    "data_factor": "Training resolution. 'auto' downsamples to about the recommended "
                   "megapixel count; you almost never need to change this.",
    "target_mp": "Recommended training megapixels. ~6 MP is the validated sweet spot; "
                 "Detail mode raises it to ~15 MP.",
    "max_steps": "How long to train. 30,000 is the converged default.",
    "strategy": "Densification strategy. MCMC gives the best quality and an explicit "
                "Gaussian budget.",
    "cap_max": "Maximum number of Gaussians (MCMC budget). 3 million is near-optimal; "
               "more actively hurts quality.",
    "dense_init": "Seed textureless regions with extra points from dense matching "
                  "before training. Fewer holes on flat surfaces.",
    "mono_depth_lambda": "Weight of the AI depth prior (MoGe-2). Gently keeps geometry "
                         "honest where photos are ambiguous. 0 turns it off.",
    "erank_lambda": "Anti-'needle' regularizer: stops Gaussians collapsing into thin "
                    "spikes. Small is enough.",
    "erank_start": "Step at which the anti-needle regularizer switches on.",
    "lpips_lambda": "Perceptual-sharpness weight. Recovers fine detail; it is gated so "
                    "it only sharpens under-resolved areas.",
    "lpips_gated": "Keep the perceptual loss focused on soft/under-resolved regions so "
                   "it never over-textures already-sharp areas. Leave ON.",
    "pose_opt": "Let the trainer nudge camera poses a little to fix small calibration "
                "errors. Turn OFF for maximum microstructure at native resolution.",
    "antialiased": "Mip-Splatting anti-aliasing. Sharper at any zoom. Leave ON.",
    "c2f": "Coarse-to-fine: densify at lower resolution, finish at full. Faster and "
           "more stable.",
    "planar_lambda": "Flatten walls/floors (geometry use only). OFF by default: it "
                     "costs perceptual sharpness. Only raise for CAD-like flat scenes.",
    "seed": "Random seed for a reproducible run.",
    "test_every": "Hold out every Nth image for quality evaluation.",
    "decimate": "How much to shrink the delivered file (keeps this fraction of "
                "Gaussians). 60% is near-lossless.",
    "extract_object": "Deliver the object carved out of the scene (Object Scan) vs the "
                      "whole scene (Environment Scan).",
    "filter_harmonics": "Spherical-harmonic detail in the delivery. 3 keeps specular "
                        "highlights (objects); 2 halves the file (scenes).",
}

EVIDENCE: dict[str, str] = {
    "data_factor": "WP-S1/T1: ~6 MP validated regime.",
    "target_mp": "WP-S1/T1; Detail = WP-N1 native recipe.",
    "max_steps": "WP-X1 convergence.",
    "strategy": "MCMC-3DGS + Mip-Splatting = proven quality base (DEV_NOTES §4).",
    "cap_max": "WP-X1/N1 + DEV_NOTES §6: 3M near-optimal; >3M degrades LPIPS/PSNR.",
    "dense_init": "WP-F2 (EDGS-style RoMa dense triangulation).",
    "mono_depth_lambda": "WP-F1 (MoGe-2 mono-depth prior).",
    "erank_lambda": "arXiv 2406.11672 (effective-rank hinge).",
    "erank_start": "WP-S1 schedule (7k of 30k).",
    "lpips_lambda": "DEV_NOTES §4/§7: gated LPIPS is a safe perceptual win at any res.",
    "lpips_gated": "DEV_NOTES §7: global LPIPS wrecks well-resolved regions; gate fixes.",
    "pose_opt": "WP-S1; Detail-mode OFF per WP-N1.",
    "antialiased": "Mip-Splatting (keep on, DEV_NOTES §3).",
    "c2f": "WP-S1 coarse-to-fine schedule.",
    "planar_lambda": "WP-F4: intrinsic perceptual cost at the paper's lambdas.",
    "seed": "-",
    "test_every": "-",
    "decimate": "WP-D1: decimation near-lossless.",
    "extract_object": "WP-S2 delivery-time object extraction; WP-D1 gates.",
    "filter_harmonics": "WP-D1: SH2 truncation is the whole LPIPS regression on glossy "
                        "objects -> keep SH3 for objects.",
}


# ---------------------------------------------------------------------------
# Ordered setting specs — the Advanced drawer is built by iterating this list,
# so there is exactly one place that decides which levers are exposed and how.
# ---------------------------------------------------------------------------
# kind: "factor" | "int" | "float" | "bool" | "choice" | "text"
SETTINGS: list[dict[str, Any]] = [
    {"key": "data_factor", "label": "Training resolution",
     "kind": "factor", "choices": ["auto", "1", "2", "4", "8"]},
    {"key": "target_mp", "label": "Target megapixels", "kind": "float",
     "min": 1.0, "max": 60.0, "step": 0.5},
    {"key": "max_steps", "label": "Iterations", "kind": "int",
     "min": 1_000, "max": 60_000, "step": 1_000},
    {"key": "strategy", "label": "Densification strategy",
     "kind": "choice", "choices": ["mcmc", "default"]},
    {"key": "cap_max", "label": "Max Gaussians (budget)", "kind": "int",
     "min": 100_000, "max": 5_000_000, "step": 100_000},
    {"key": "dense_init", "label": "Dense init (fill textureless)", "kind": "bool"},
    {"key": "mono_depth_lambda", "label": "AI depth prior weight", "kind": "float",
     "min": 0.0, "max": 0.5, "step": 0.01},
    {"key": "erank_lambda", "label": "Anti-needle weight", "kind": "float",
     "min": 0.0, "max": 0.1, "step": 0.005},
    {"key": "erank_start", "label": "Anti-needle start step", "kind": "int",
     "min": 0, "max": 30_000, "step": 500},
    {"key": "lpips_lambda", "label": "Perceptual sharpness weight", "kind": "float",
     "min": 0.0, "max": 0.5, "step": 0.02},
    {"key": "lpips_gated", "label": "Gate perceptual loss", "kind": "bool"},
    {"key": "pose_opt", "label": "Pose refinement", "kind": "bool"},
    {"key": "antialiased", "label": "Anti-aliasing (Mip-Splatting)", "kind": "bool"},
    {"key": "c2f", "label": "Coarse-to-fine", "kind": "bool"},
    {"key": "planar_lambda", "label": "Flatten walls/floors", "kind": "float",
     "min": 0.0, "max": 0.5, "step": 0.05},
    {"key": "extract_object", "label": "Deliver object (carve out)", "kind": "bool"},
    {"key": "filter_harmonics", "label": "Delivery SH level", "kind": "choice",
     "choices": [1, 2, 3]},
    {"key": "decimate", "label": "Delivery decimation", "kind": "text"},
    {"key": "test_every", "label": "Eval: hold out every Nth", "kind": "int",
     "min": 1, "max": 32, "step": 1},
    {"key": "seed", "label": "Random seed", "kind": "int",
     "min": 0, "max": 2**31 - 1, "step": 1},
]

SETTING_KEYS: list[str] = [s["key"] for s in SETTINGS]


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

def defaults_for(mode: str, detail: bool = False) -> dict[str, Any]:
    """Return the full default dict for a mode, with Detail-mode overrides applied.

    `mode` is "object" or "environment". This is THE function the drawer seeds from
    and the smoke gate compares against.
    """
    base = OBJECT_DEFAULTS if mode == "object" else ENV_DEFAULTS
    d = dict(base)
    if detail:
        d.update(DETAIL_MODE_OVERRIDES)
    return d


# ---------------------------------------------------------------------------
# Resolution math
# ---------------------------------------------------------------------------

def auto_data_factor(source_mp: float, target_mp: float,
                     max_factor: int = 8) -> int:
    """Pick the integer downsample factor whose training MP is NEAREST target_mp.

    Brute-forces f in 1..max_factor minimizing |source_mp/f**2 - target_mp| (training
    MP at factor f is source_mp/f**2). Tiebreak toward the SMALLER factor (more
    resolution). This is exact — sqrt(source/target) rounding is biased (e.g. 12 MP @
    6 MP target gives factor 1 = 12 MP instead of factor 2 = 3 MP, the true nearest).
    """
    if source_mp <= 0 or target_mp <= 0:
        return 1
    best_f, best_err = 1, float("inf")
    for f in range(1, max_factor + 1):
        err = abs(source_mp / (f * f) - target_mp)
        if err < best_err - 1e-9:          # strictly better; ties keep smaller f
            best_f, best_err = f, err
    return best_f


def training_mp(source_mp: float, factor: int) -> float:
    """Megapixels the trainer actually sees at this factor."""
    factor = max(1, int(factor))
    return source_mp / (factor * factor)


def resolve_data_factor(d: dict[str, Any], source_mp: float) -> int:
    """Resolve the 'auto'/explicit data_factor in a settings dict to a concrete int."""
    val = d.get("data_factor", "auto")
    if val in ("auto", None, ""):
        return auto_data_factor(source_mp, float(d.get("target_mp", 6.0)))
    return max(1, int(val))


# ---------------------------------------------------------------------------
# Assemble the backend TrainingConfig.extra dict (keys match GsplatBackend config)
# ---------------------------------------------------------------------------

def assemble_training_extra(d: dict[str, Any], data_factor: int,
                            max_steps: int) -> dict[str, Any]:
    """Map the mode-settings dict to the flat `extra` dict GsplatBackend expects.

    Mirrors the validated GUI-1 wiring (gui/stages/training.py) for the derived
    schedule values (coarse_factor, coarse_until, lpips_start).
    """
    c2f = bool(d.get("c2f", True))
    coarse_factor = data_factor * 2 if c2f else 0
    coarse_until = min(15_000, max_steps // 2)

    lpips_lambda = float(d.get("lpips_lambda", 0.0))
    # At full res under c2f, start LPIPS only in the fine phase (avoid over-texturing
    # well-resolved content); otherwise start it at ~1/5 of the run. (GUI-1 logic.)
    if lpips_lambda > 0:
        lpips_start = (min(15_000, max_steps // 2)
                       if (c2f and data_factor <= 1) else max(1, max_steps // 5))
    else:
        lpips_start = 3_000

    extra: dict[str, Any] = {
        "strategy": str(d.get("strategy", "mcmc")),
        "cap_max": int(d.get("cap_max", 3_000_000)),
        "antialiased": bool(d.get("antialiased", True)),
        "pose_opt": bool(d.get("pose_opt", True)),
        "coarse_factor": coarse_factor,
        "coarse_until": coarse_until,
        # Levers
        "dense_init": bool(d.get("dense_init", False)),
        "mono_depth_lambda": float(d.get("mono_depth_lambda", 0.0)),
        "erank_lambda": float(d.get("erank_lambda", 0.0)),
        "erank_start": int(d.get("erank_start", 7_000)),
        "lpips_lambda": lpips_lambda,
        "lpips_gated": bool(d.get("lpips_gated", True)),
        "lpips_start": lpips_start,
        "planar_lambda": float(d.get("planar_lambda", 0.0)),
    }
    return extra


def assemble_delivery_flags(d: dict[str, Any]) -> list[str]:
    """CLI flags for `python -m core.delivery` derived from the mode settings.

    Only passes overrides; delivery already applies the correct per-mode SH/refit
    defaults when the flags are omitted.
    """
    flags: list[str] = []
    if bool(d.get("extract_object", False)):
        flags += ["--extract_object", "--gen_masks"]
    decimate = str(d.get("decimate", "60%"))
    flags += ["--decimate", decimate]
    # SH level: emit explicitly so the drawer lever is LIVE (delivery's per-mode
    # fallback only coincidentally matches these defaults).
    flags += ["--filter_harmonics", str(int(d.get("filter_harmonics", 2)))]
    return flags
