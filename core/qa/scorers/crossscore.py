"""CrossScore scorer (WP-Q1 comp.2) — SUBPROCESS to the qa venv via CrossScore's
own Hydra CLI (task/predict.py).

CrossScore predicts a per-pixel SSIM map for each rendered (query) view from
cross-view evidence in a pool of REAL reference photos — NO ground-truth needed.
We stage the renders + a pool of real train photos into the DEEP directory layout
CrossScore's summariser parses (`<method>/<dataset>/<res>/<scene>/test/ours_X/
renders/frame_*.png`), run predict.py with the determinism-critical overrides,
and read the per-frame mean predicted-SSIM the summariser writes to CSV.

CRITICAL overrides (from EXECUTION_PLAN):
  this_main.force_batch_size=True                 (skip the interactive prompt)
  data.neighbour_config.deterministic=True        (else 5 RANDOM refs per query)
  ...score_map_colour_mode=gray                    (machine-readable maps)
Output predicted SSIM in [0,1], HIGHER = better. Stored under scorers["crossscore"]:
  {mean, median, per_view:[{index, pred_ssim}], n, csv}.
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np

from . import common


class CrossScoreScorer:
    name = "crossscore"

    def __init__(self, cfg, max_refs=30):
        self.cfg = cfg
        self.max_refs = max_refs

    def _stage_inputs(self, ctx):
        import shutil
        base = Path(ctx.out_dir) / "_scorers" / "crossscore"
        scene = "".join(c for c in ctx.scene if c.isalnum() or c in "-_") or "scene"
        root = base / "gaussian" / "apex" / "res_540" / scene
        # ALWAYS FRESH: dirs cleared + files copied unconditionally (stale staged
        # inputs were the reviewer-reproduced HIGH bug; CrossScore GLOBS these
        # dirs, so leftover extra frames would silently join the query set).
        qdir = common._fresh_dir(root / "test" / "ours_00000" / "renders")
        rdir = common._fresh_dir(root / "train" / "ours_00000" / "gt")

        # query = this run's held-out renders (freshly rendered, run-memoized)
        common.stage_heldout_renders(ctx)          # <stage>/renders/frame_*.png
        for idx in ctx.test_indices:
            src = Path(ctx.out_dir) / "_scorers" / "renders" / f"frame_{idx:05d}.png"
            shutil.copyfile(src, qdir / f"frame_{idx:05d}.png")

        # references = a pool of real train photos (evenly spaced), undistorted
        train = sorted(int(i) for i in ctx.data["train_indices"])
        if len(train) > self.max_refs:
            sel = np.linspace(0, len(train) - 1, self.max_refs).round().astype(int)
            train = [train[i] for i in sel]
        for j in train:
            rp = common.stage_real_photo(ctx, j)
            shutil.copyfile(rp, rdir / f"frame_{j:05d}.png")
        return qdir, rdir, base

    def score(self, ctx) -> dict:
        import shutil
        qdir, rdir, base = self._stage_inputs(ctx)
        out_dir = base.parent / "crossscore_out"
        # clear THIS run's output (incl. the alias-suffixed sibling predict.py
        # writes to) so the CSV read below can never pick up a stale summary
        for cand in sorted(out_dir.parent.glob(out_dir.name + "*")):
            shutil.rmtree(cand, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        overrides = [
            "trainer.devices=[0]",
            f"trainer.ckpt_path_to_load={self.cfg.crossscore_ckpt}",
            "trainer.precision=32-true",
            # absolute paths: predict.py runs with cwd=CrossScore repo
            f"data.dataset.query_dir={qdir.resolve().as_posix()}",
            f"data.dataset.reference_dir={rdir.resolve().as_posix()}",
            "this_main.force_batch_size=True",
            "data.neighbour_config.deterministic=True",
            f"logger.predict.out_dir={out_dir.resolve().as_posix()}",
            "logger.predict.write.config.score_map_colour_mode=gray",
            # We read the machine-readable per-frame mean SSIM from the summariser
            # CSV (runs unconditionally). The batch image/map writer mirrors the
            # query subpath into many leaf dirs it never mkdir's; disable it for
            # robustness (the numeric metric is unaffected). Re-enable + finish the
            # mkdir fixes if per-pixel gray maps are wanted for eyeballing.
            "logger.predict.write.flag.batch=False",
            "logger.predict.write.config.vis_img_every_n_steps=-1",
            "alias=apex",
        ]
        cmd = [self.cfg.qa_python, "task/predict.py"] + overrides
        p = subprocess.run(cmd, cwd=self.cfg.crossscore_repo,
                           capture_output=True, text=True, timeout=self.cfg.timeout_s)
        if p.returncode != 0:
            raise RuntimeError(
                f"CrossScore predict.py failed (exit {p.returncode}).\n"
                f"STDOUT:\n{p.stdout[-2000:]}\nSTDERR:\n{p.stderr[-3000:]}")

        # summariser writes <out>/score_summary/<dataset_type>/<method>.csv, where
        # predict.py may append "_{alias}" to out_dir -> search out_dir and any
        # sibling dir sharing its name prefix (e.g. crossscore_out_apex).
        csvs = []
        for cand in sorted(out_dir.parent.glob(out_dir.name + "*")):
            csvs += list((cand / "score_summary").rglob("*.csv"))
        if not csvs:
            raise RuntimeError(f"CrossScore wrote no score_summary CSV under {out_dir}*."
                               f"\nSTDOUT tail:\n{p.stdout[-1500:]}")
        per_view = []
        for cpath in csvs:
            with open(cpath) as f:
                rd = csv.DictReader(f)
                score_col = next((c for c in rd.fieldnames if c.startswith("pred_")), None)
                if score_col is None:
                    raise RuntimeError(
                        f"CrossScore CSV {cpath} has no 'pred_*' score column "
                        f"(columns: {rd.fieldnames}) — upstream summariser "
                        f"format changed?")
                for row in rd:
                    name = row["image_name"]           # e.g. "00042.png"
                    try:
                        idx = int(Path(name).stem)
                    except ValueError:
                        idx = -1
                    per_view.append({"index": idx, "pred_ssim": float(row[score_col])})
        vals = [v["pred_ssim"] for v in per_view]
        return {
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "median": float(np.median(vals)) if vals else float("nan"),
            "n": len(per_view),
            "per_view": sorted(per_view, key=lambda v: v["index"]),
            "csv": str(csvs[0]),
            "maps_dir": str(out_dir),
        }
