"""NVS-SQA scorer (WP-Q1 comp.7) — SUBPROCESS to the qa venv.

Feeds the rendered SMOOTH-trajectory clip (sorted, zero-padded frame names —
fusion='last' means frame ORDER matters) to the NVS-SQA sequence encoder and
maps its feature to a JOD score via the extracted linear head. JOD is higher =
better but on a ~negative scale; it is only meaningful for WITHIN-scene deltas
(the dataset author's own caveat).

Stored under scorers["nvs_sqa"]: {jod, n_frames, feat_dim}.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import common
from . import run_venv


class NvsSqaScorer:
    name = "nvs_sqa"

    def __init__(self, cfg):
        self.cfg = cfg

    def score(self, ctx) -> dict:
        clip = common.moving_clip(ctx, n_anchors=self.cfg.n_anchors,
                                  steps=self.cfg.steps, long_edge=self.cfg.long_edge)
        # ABSOLUTE: the driver does os.chdir(repo) before globbing frames_dir.
        frames_dir = str(Path(clip["paths"][0]).parent.resolve())   # <stage>/moving
        stage = Path(ctx.out_dir) / "_scorers"
        in_json = str(stage / "nvssqa_in.json")
        out_json = str(stage / "nvssqa_out.json")
        json.dump({"repo": self.cfg.nvssqa_repo,
                   "model_folder": self.cfg.nvssqa_model_folder,
                   "linear_head": self.cfg.nvssqa_head,
                   "frames_dir": frames_dir, "epo_offset": 100},
                  open(in_json, "w"))
        run_venv(self.cfg, "nvs_sqa_driver.py", in_json, out_json)
        return json.load(open(out_json))
