"""WP-Q1 phase-2 scorers — the external-model quality metrics that plug into the
phase-1 `Scorer` protocol (core/qa/scorecard.py).

ENV MAPPING (see qa/README.md):
  IN-PROCESS  (main env, torch 2.11 / numpy 2)   : tsed, raft_warp, flicker
  SUBPROCESS  (qa venv, D:/REPOS/HYPERclone/venvs/qa) : crossscore, nova, nvs_sqa
The subprocess scorers shell out to the qa-venv python running a driver in
`core/qa/venv_drivers/` (or CrossScore's own Hydra CLI). This keeps the pinned /
shadowing deps (transformers, lightning, hydra, timm) out of the main env while
letting the whole scorecard run from one `qa_report.py` invocation.

`build_scorers(names, cfg)` returns the scorer objects for the requested names.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

_QA_ROOT = Path(__file__).resolve().parents[1]        # core/qa
_DRIVERS = _QA_ROOT / "venv_drivers"
_REPOS = Path("D:/REPOS/HYPERclone/qa_repos")


@dataclass
class ScorerConfig:
    """Paths + knobs for the phase-2 scorers (defaults = the WP-Q1 layout)."""
    qa_python: str = str(Path("D:/REPOS/HYPERclone/venvs/qa/Scripts/python.exe"))
    # CrossScore
    crossscore_repo: str = str(_REPOS / "CrossScore")
    crossscore_ckpt: str = "ckpt/CrossScore-v1.0.0.ckpt"    # relative to repo
    # NOVA
    nova_repo: str = str(_REPOS / "NOVA")
    nova_ckpt: str = str(_REPOS / "NOVA" / "weights" / "NOVA_NVS.pt")
    # NVS-SQA — the extracted 384->1 linear head is VENDORED in apex-sfm (the
    # copy in the clone was untracked; a re-clone would silently lose it)
    nvssqa_repo: str = str(_REPOS / "NVS-SQA")
    nvssqa_model_folder: str = str(_REPOS / "NVS-SQA" / "checkpoints" / "v0.0.1" / "models")
    nvssqa_head: str = str(_QA_ROOT / "vendor" / "nvs_sqa_linear_head.npz")
    # shared clip knobs (moving trajectory for tsed/raft/nvs_sqa)
    n_anchors: int = 12
    steps: int = 3
    long_edge: int = 640
    # flicker static clip
    flicker_frames: int = 16
    flicker_jitter: float = 0.06
    flicker_static_max_frac: float = 0.02
    drivers_dir: str = str(_DRIVERS)
    timeout_s: int = 3600


def run_venv(cfg: ScorerConfig, driver: str, in_json: str, out_json: str,
             cwd: str = None) -> None:
    """Run `<qa_python> <drivers>/<driver> <in_json> <out_json>` and raise on
    non-zero exit (captured stderr in the message). Paths are resolved to ABSOLUTE
    so drivers that os.chdir (NVS-SQA) still find/write them."""
    script = str(Path(cfg.drivers_dir) / driver)
    in_json = str(Path(in_json).resolve())
    out_json = str(Path(out_json).resolve())
    cmd = [cfg.qa_python, script, in_json, out_json]
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       timeout=cfg.timeout_s)
    if p.returncode != 0:
        raise RuntimeError(
            f"{driver} failed (exit {p.returncode}).\nSTDOUT:\n{p.stdout[-2000:]}"
            f"\nSTDERR:\n{p.stderr[-3000:]}")


def build_scorers(names: List[str], cfg: ScorerConfig = None):
    """Instantiate the requested scorers. `names` is a subset of SCORER_REGISTRY
    (or ['all']). Unknown names raise."""
    from ..scorecard import SCORER_REGISTRY
    cfg = cfg or ScorerConfig()
    if names == ["all"] or names == "all":
        names = list(SCORER_REGISTRY)

    from .tsed import TSEDScorer
    from .raft_warp import RaftWarpScorer
    from .flicker import FlickerScorer
    from .crossscore import CrossScoreScorer
    from .nova import NovaScorer
    from .nvs_sqa import NvsSqaScorer

    factory = {
        "tsed": lambda: TSEDScorer(cfg.n_anchors, cfg.steps, cfg.long_edge),
        "raft_warp": lambda: RaftWarpScorer(cfg.n_anchors, cfg.steps, cfg.long_edge),
        "flicker": lambda: FlickerScorer(cfg.flicker_frames, cfg.flicker_jitter,
                                         cfg.long_edge, cfg.flicker_static_max_frac),
        "crossscore": lambda: CrossScoreScorer(cfg),
        "nova": lambda: NovaScorer(cfg),
        "nvs_sqa": lambda: NvsSqaScorer(cfg),
    }
    out = []
    for n in names:
        if n not in factory:
            raise ValueError(f"unknown scorer '{n}' (registry: {list(factory)})")
        out.append(factory[n]())
    return out
