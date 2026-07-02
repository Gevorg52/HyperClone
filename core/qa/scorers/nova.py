"""NOVA scorer (WP-Q1 comp.3) — SUBPROCESS to the qa venv.

For each held-out render we pick the nearest OTHER real photo by COLMAP camera
pose distance (a non-aligned reference; see common.nearest_real_index) and ask
NOVA for the cosine distance between their DINOv2 embeddings. Output distance is
in [0,2], LOWER = better. Calibrate per-scene (use within-scene deltas).

Stored under scorers["nova"]: {mean, median, per_view:[{index, ref_index,
cosine_distance}], n}.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import common
from . import run_venv


class NovaScorer:
    name = "nova"

    def __init__(self, cfg):
        self.cfg = cfg

    def score(self, ctx) -> dict:
        common.stage_heldout_renders(ctx)          # writes renders/frame_XXXXX.png
        stage = Path(ctx.out_dir) / "_scorers"
        pairs = []
        ref_map = {}
        for idx in ctx.test_indices:
            ref_idx = common.nearest_real_index(ctx, idx)
            ref_map[idx] = ref_idx
            ref_path = str(Path(common.stage_real_photo(ctx, ref_idx)).resolve())
            render_path = str((stage / "renders" / f"frame_{idx:05d}.png").resolve())
            pairs.append({"render": render_path, "ref": ref_path, "index": int(idx)})

        in_json = str(stage / "nova_in.json")
        out_json = str(stage / "nova_out.json")
        json.dump({"repo": self.cfg.nova_repo, "checkpoint": self.cfg.nova_ckpt,
                   "pairs": pairs, "resize": 518}, open(in_json, "w"))
        run_venv(self.cfg, "nova_driver.py", in_json, out_json)
        res = json.load(open(out_json))
        for v in res.get("per_view", []):
            v["ref_index"] = ref_map.get(v["index"])
        return res
