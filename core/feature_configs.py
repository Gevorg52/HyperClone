"""Build HLoc-compatible feature extraction config dicts.

Each extractor has slightly different field names and preprocessing requirements.
This module centralizes all that mapping so the GUI just passes user-friendly params.
"""


def build_extraction_config(
    extractor: str = "superpoint",
    max_keypoints: int = 8192,
    nms_radius: int = 3,
    resize_max: int | None = None,
    as_half: bool = False,
    keypoint_threshold: float = 0.001,
) -> dict:
    """Build an HLoc-compatible extraction config dict.

    Args:
        extractor: One of "superpoint", "disk", "aliked".
        max_keypoints: Maximum keypoints per image.
        nms_radius: NMS radius (SuperPoint only).
        resize_max: Maximum image dimension. None = full resolution.
        as_half: If True, store descriptors as float16.
        keypoint_threshold: Detection threshold (SuperPoint only).

    Returns:
        Dict with "output", "model", and "preprocessing" keys for HLoc.
    """
    # Handle resize_max: HLoc expects an integer; use large sentinel for "no resize"
    effective_resize = resize_max if resize_max is not None else 99999

    if extractor == "superpoint":
        return {
            "output": "feats-superpoint-n{}-r{}".format(max_keypoints, effective_resize),
            "model": {
                "name": "superpoint",
                "nms_radius": nms_radius,
                "max_keypoints": max_keypoints,  # CRITICAL: HLoc uses max_keypoints
                "keypoint_threshold": keypoint_threshold,
            },
            "preprocessing": {
                "grayscale": True,  # SuperPoint requires grayscale
                "resize_max": effective_resize,
                "as_half": as_half,
            },
        }

    elif extractor == "disk":
        return {
            "output": "feats-disk-n{}-r{}".format(max_keypoints, effective_resize),
            "model": {
                "name": "disk",
                "max_keypoints": max_keypoints,
            },
            "preprocessing": {
                "grayscale": False,  # DISK needs RGB
                "resize_max": effective_resize,
                "as_half": as_half,
            },
        }

    elif extractor == "aliked":
        return {
            "output": "feats-aliked-n{}-r{}".format(max_keypoints, effective_resize),
            "model": {
                "name": "aliked",
                "max_num_keypoints": max_keypoints,  # ALIKED uses max_num_keypoints
            },
            "preprocessing": {
                "grayscale": False,  # ALIKED needs RGB
                "resize_max": effective_resize,
                "as_half": as_half,
            },
        }

    else:
        raise ValueError(f"Unknown extractor: {extractor}. Choose from: superpoint, disk, aliked")
