"""Temporal-flickering scorer (WP-Q1 comp.6) — IN-PROCESS, main env. Vendored
VBench metric on a NEAR-STATIC micro-jitter clip, with a hard guard that refuses
genuinely moving-camera sequences (the metric is only meaningful when the camera
is ~static — otherwise parallax masquerades as flicker).

Stored under scorers["flicker"]:
  score            (255 - mean consecutive-frame MAE)/255  (higher = less flicker)
  mean_mae, n_pairs
  max_motion_frac  max inter-frame camera-center motion / scene_scale
  static           True iff max_motion_frac < static_max_frac (else score is null
                   and skipped="moving_camera")
"""
from __future__ import annotations

from . import common
from ..vendor.vbench_flicker import flicker_score


class FlickerScorer:
    name = "flicker"

    def __init__(self, n_frames=16, jitter_frac=0.06, long_edge=640,
                 static_max_frac=0.02):
        self.n_frames = n_frames
        self.jitter_frac = jitter_frac
        self.long_edge = long_edge
        self.static_max_frac = static_max_frac

    def score(self, ctx) -> dict:
        clip = common.static_clip(ctx, n_frames=self.n_frames,
                                  jitter_frac=self.jitter_frac,
                                  long_edge=self.long_edge)
        mmf = clip["max_motion_frac"]
        if mmf >= self.static_max_frac:
            # guard: refuse moving-camera sequences
            return {"score": None, "skipped": "moving_camera",
                    "max_motion_frac": mmf, "static_max_frac": self.static_max_frac,
                    "static": False}
        res = flicker_score(clip["frames_u8"])
        res.update({"max_motion_frac": mmf, "static": True,
                    "static_max_frac": self.static_max_frac})
        return res
