"""Build HLoc-compatible feature matching config dicts.

Centralizes LightGlue configuration so the GUI passes user-friendly params.
"""


def build_matching_config(
    matcher: str = "lightglue",
    features: str = "superpoint",
    depth_confidence: float = -1,
    width_confidence: float = -1,
    filter_threshold: float = 0.1,
) -> dict:
    """Build an HLoc-compatible matching config dict.

    Args:
        matcher: Matcher name (currently only "lightglue").
        features: Feature type to match against ("superpoint", "disk", "aliked").
        depth_confidence: Set to -1 to disable adaptive depth (max quality, slower).
        width_confidence: Set to -1 to disable adaptive width (max quality, slower).
        filter_threshold: Match confidence threshold.

    Returns:
        Dict with "output", "model", and "preprocessing" keys for HLoc.
    """
    if matcher == "lightglue":
        return {
            "output": f"matches-lightglue-{features}",
            "model": {
                "name": f"lightglue",
                "features": features,
                "depth_confidence": depth_confidence,
                "width_confidence": width_confidence,
                "filter_threshold": filter_threshold,
            },
        }
    else:
        raise ValueError(f"Unknown matcher: {matcher}. Currently only 'lightglue' is supported.")
