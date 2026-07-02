"""TSED scorer (WP-Q1 comp.4) — runs IN-PROCESS in the main env (numpy+opencv
only). Median symmetric epipolar distance between consecutive frames of the
rendered smooth trajectory, thresholded per the vendored spec.

Output stored under scorers["tsed"]:
  ratio           fraction of consecutive pairs that are geometry-consistent
  n_pairs         number of consecutive pairs
  median_epi_med  median over pairs of the per-pair median epipolar distance (px)
  mean_matches    mean SIFT match count per pair (textureless != geom failure)
  per_pair        [{n_matches, median_epi, thresh, consistent, reason}, ...]
Higher `ratio` = more multi-view-consistent geometry.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import common
from ..vendor import tsed_epipolar as te


class TSEDScorer:
    name = "tsed"

    def __init__(self, n_anchors=12, steps=3, long_edge=640,
                 min_matches=10, thresh_px=2.0, ratio=0.75):
        self.n_anchors = n_anchors
        self.steps = steps
        self.long_edge = long_edge
        self.min_matches = min_matches
        self.thresh_px = thresh_px
        self.ratio = ratio

    def score(self, ctx) -> dict:
        clip = common.moving_clip(ctx, n_anchors=self.n_anchors, steps=self.steps,
                                  long_edge=self.long_edge)
        frames = clip["frames_u8"]
        poses = clip["poses"]
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

        per_pair = []
        for i in range(len(frames) - 1):
            a, b = poses[i], poses[i + 1]
            res = te.tsed_pair(
                grays[i], grays[i + 1],
                a["R"], a["t"], a["K"], b["R"], b["t"], b["K"], H=a["H"],
                min_matches=self.min_matches, thresh_px=self.thresh_px,
                ratio=self.ratio)
            per_pair.append(res)

        if not per_pair:
            return {"ratio": float("nan"), "n_pairs": 0}
        consistent = [p["consistent"] for p in per_pair]
        med_epis = [p["median_epi"] for p in per_pair if not np.isnan(p["median_epi"])]
        matches = [p["n_matches"] for p in per_pair]
        return {
            "ratio": float(np.mean(consistent)),
            "n_pairs": len(per_pair),
            "median_epi_med": float(np.median(med_epis)) if med_epis else float("nan"),
            "mean_matches": float(np.mean(matches)),
            "thresh_px": self.thresh_px,
            "per_pair": per_pair,
        }
