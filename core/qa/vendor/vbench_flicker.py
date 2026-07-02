"""Temporal flickering — vendored from VBench (github.com/Vchitect/VBench,
`vbench/temporal_flickering.py`), Apache-2.0.

VBench itself is NEVER pip-installed (it pins numpy<2 + transformers 4.33, which
would poison the torch-2.11 / numpy-2 env). We copy the ONE relevant function
verbatim in spirit: the mean-absolute-error between consecutive frames, mapped
to a [0,1] "no-flicker" score:

    flicker = (255 - mean_over_pairs(MAE(frame_i, frame_{i+1}))) / 255

Higher = less flicker = better. This is a STATIC-CAMERA metric: with a moving
camera, legitimate parallax/content change inflates the MAE and the number is
meaningless (the VBench docstring literally says "please ensure the video is
static"). The caller (core/qa/scorers/flicker.py) enforces that guard.

Input frames are uint8 RGB (H,W,3). Deterministic (pure numpy absdiff).
"""
import cv2
import numpy as np


def calculate_mae(img1, img2) -> float:
    """Mean absolute error between two same-shape images (VBench verbatim)."""
    if img1.shape != img2.shape:
        raise ValueError("Images don't have the same shape.")
    return float(np.mean(cv2.absdiff(
        np.asarray(img1, dtype=np.float32), np.asarray(img2, dtype=np.float32))))


def mae_seq(frames) -> np.ndarray:
    """MAE between each consecutive frame pair (VBench verbatim)."""
    return np.array([calculate_mae(frames[i], frames[i + 1])
                     for i in range(len(frames) - 1)])


def flicker_score(frames) -> dict:
    """VBench temporal_flickering on a STATIC-camera clip.

    frames: list of uint8 RGB (H,W,3), length >= 2.
    Returns {"score": (255-meanMAE)/255 in [0,1], "mean_mae": float,
             "n_pairs": int}. Higher score = less flicker.
    """
    if len(frames) < 2:
        return {"score": float("nan"), "mean_mae": float("nan"), "n_pairs": 0}
    seq = mae_seq(frames)
    mean_mae = float(np.mean(seq))
    return {"score": (255.0 - mean_mae) / 255.0,
            "mean_mae": mean_mae, "n_pairs": int(len(seq))}
