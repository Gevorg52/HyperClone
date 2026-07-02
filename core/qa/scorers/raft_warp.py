"""RAFT warp-error scorer (WP-Q1 comp.5) — IN-PROCESS, main env (torchvision
raft_large). Primary temporal metric for MOVING cameras.

For each consecutive pair (t, t+1) of the rendered smooth trajectory:
  * forward flow  f = RAFT(t -> t+1)[-1]        (last iterate)
  * backward flow b = RAFT(t+1 -> t)[-1]
  * backward-warp frame t+1 into frame t via grid_sample using f
  * forward-backward occlusion mask: a pixel is occluded where
        |f + b(x+f)|^2  >  0.01*(|f|^2 + |b(x+f)|^2) + 0.5
    (valid = not occluded)
  * masked L1 between warped(t+1) and frame t over valid pixels
Average over pairs. Higher warp error = more temporal inconsistency (floaters /
popping). Stored under scorers["raft_warp"]: {mean, per_pair, n_pairs, ...}.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import common

_model = None
_transforms = None


def _get_raft(device):
    global _model, _transforms
    if _model is None:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        w = Raft_Large_Weights.DEFAULT
        _model = raft_large(weights=w, progress=False).eval().to(device)
        _transforms = w.transforms()
    return _model, _transforms


def _flow(model, img1, img2):
    """img1,img2: (B,3,H,W) already preprocessed. Returns (B,2,H,W) last iterate."""
    with torch.no_grad():
        return model(img1, img2)[-1]


def _warp(img, flow):
    """Backward-warp img (B,3,H,W) by flow (B,2,H,W): out[x] = img[x + flow[x]]."""
    B, _, H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device),
                            torch.arange(W, device=img.device), indexing="ij")
    grid = torch.stack((xx, yy), dim=0).float()[None].expand(B, -1, -1, -1)
    vgrid = grid + flow
    vx = 2.0 * vgrid[:, 0] / max(W - 1, 1) - 1.0
    vy = 2.0 * vgrid[:, 1] / max(H - 1, 1) - 1.0
    g = torch.stack((vx, vy), dim=-1)
    return F.grid_sample(img, g, mode="bilinear", padding_mode="border",
                         align_corners=True)


class RaftWarpScorer:
    name = "raft_warp"

    def __init__(self, n_anchors=12, steps=3, long_edge=640, batch=6):
        self.n_anchors = n_anchors
        self.steps = steps
        self.long_edge = long_edge
        self.batch = batch

    def score(self, ctx) -> dict:
        clip = common.moving_clip(ctx, n_anchors=self.n_anchors, steps=self.steps,
                                  long_edge=self.long_edge)
        frames = clip["frames_u8"]
        if len(frames) < 2:
            return {"mean": float("nan"), "n_pairs": 0}
        device = ctx.device
        model, tf = _get_raft(device)

        # float [0,1] CHW tensors, cropped to a multiple of 8 (RAFT's transform
        # validates H,W % 8 == 0 but does not resize).
        H8 = frames[0].shape[0] - frames[0].shape[0] % 8
        W8 = frames[0].shape[1] - frames[0].shape[1] % 8
        ten = [torch.from_numpy(f[:H8, :W8]).permute(2, 0, 1).float().to(device) / 255.0
               for f in frames]

        per_pair = []
        n = len(ten) - 1                                    # number of pairs
        for s in range(0, n, self.batch):
            cnt = min(self.batch, n - s)
            a = torch.stack(ten[s:s + cnt])                 # frame t
            b = torch.stack(ten[s + 1:s + 1 + cnt])         # frame t+1
            a, b = tf(a, b)
            fwd = _flow(model, a, b)      # t -> t+1
            bwd = _flow(model, b, a)      # t+1 -> t
            warped = _warp(b, fwd)        # frame t+1 sampled into frame t
            bwd_w = _warp(bwd, fwd)       # backward flow sampled at x+f
            # forward-backward occlusion
            fb = fwd + bwd_w
            mag_fb = (fb ** 2).sum(1)
            thresh = 0.01 * ((fwd ** 2).sum(1) + (bwd_w ** 2).sum(1)) + 0.5
            valid = (mag_fb <= thresh).float()              # (b,H,W)
            l1 = (warped - a).abs().mean(1)                 # (b,H,W)
            for i in range(a.shape[0]):
                v = valid[i]
                denom = v.sum().clamp_min(1.0)
                err = float((l1[i] * v).sum() / denom)
                per_pair.append({"warp_l1": err,
                                 "valid_frac": float(v.mean())})
        errs = [p["warp_l1"] for p in per_pair]
        return {
            "mean": float(np.mean(errs)),
            "median": float(np.median(errs)),
            "n_pairs": len(per_pair),
            "mean_valid_frac": float(np.mean([p["valid_frac"] for p in per_pair])),
            "per_pair": per_pair,
        }
