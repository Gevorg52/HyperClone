"""RoMa v2 rescue matching for weak pairs.

After LightGlue matching, pairs with few matches (low-texture, glossy, wide
baseline) can be re-matched with the RoMa v2 dense matcher (SOTA two-view
accuracy, arXiv 2511.15706).  Dense matches are snapped to EXISTING ALIKED
keypoints so multi-view track consistency is preserved — COLMAP sees extra
correspondences between keypoints it already knows, no new detections.

RoMa v2 is an optional dependency (MIT code; its DINOv3 backbone has Meta's
custom license — review before commercial use):

    pip install git+https://github.com/Parskatt/RoMaV2

Note: upstream targets Python 3.12; on this project's 3.11 env it may need
a source checkout. The module degrades gracefully when unavailable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np

ProgressCallback = Callable[[int, int, str], None]

_roma_model = None


def roma_available() -> tuple[bool, str]:
    """Check whether the RoMa v2 package can be imported."""
    try:
        import romav2  # noqa: F401
        return True, "RoMa v2 is available."
    except ImportError as e:
        return False, (
            f"RoMa v2 not installed ({e}).\n"
            "Install with:\n"
            "  pip install git+https://github.com/Parskatt/RoMaV2\n"
            "(MIT code; DINOv3 backbone has Meta's custom license — "
            "review for commercial use.)"
        )


def _load_roma():
    global _roma_model
    if _roma_model is None:
        from romav2 import RoMaV2
        _roma_model = RoMaV2()
    return _roma_model


def unload_roma() -> None:
    """Free RoMa VRAM (e.g. before reconstruction)."""
    global _roma_model
    if _roma_model is not None:
        _roma_model = None
        import torch
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Match-file inspection
# ---------------------------------------------------------------------------

def _pair_key(h5: h5py.File, name0: str, name1: str):
    """Locate a pair group in an HLoc matches file. Returns (key0, key1, reversed)."""
    from hloc.utils.io import find_pair
    pair, reverse = find_pair(h5, name0, name1)
    return pair, reverse


def find_weak_pairs(
    match_path: str,
    pairs_path: str,
    min_matches: int = 100,
) -> list[tuple[str, str, int]]:
    """Return (nameA, nameB, num_matches) for pairs below *min_matches*."""
    from hloc.utils.parsers import parse_retrieval

    pairs = []
    for line in Path(pairs_path).read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))

    weak = []
    with h5py.File(str(match_path), "r") as h5:
        for name0, name1 in pairs:
            try:
                pair, _ = _pair_key(h5, name0, name1)
            except ValueError:
                weak.append((name0, name1, 0))
                continue
            matches0 = h5[pair]["matches0"].__array__()
            n = int((matches0 != -1).sum())
            if n < min_matches:
                weak.append((name0, name1, n))
    return weak


# ---------------------------------------------------------------------------
# Rescue
# ---------------------------------------------------------------------------

def rescue_weak_pairs(
    image_dir: str,
    feature_path: str,
    match_path: str,
    pairs_path: str,
    min_matches: int = 100,
    snap_radius: float = 8.0,
    samples_per_pair: int = 2000,
    progress_cb: Optional[ProgressCallback] = None,
) -> dict:
    """Re-match weak pairs with RoMa v2 and merge snapped matches in place.

    For each weak pair: run dense matching, sample correspondences, snap both
    endpoints to existing ALIKED keypoints within *snap_radius* px (original
    image resolution), and add non-conflicting matches to the match file.

    Returns stats: {"weak_pairs": N, "rescued_pairs": M, "added_matches": K}.
    """
    from PIL import Image
    from scipy.spatial import cKDTree

    from hloc.utils.io import get_keypoints

    ok, msg = roma_available()
    if not ok:
        raise RuntimeError(msg)

    weak = find_weak_pairs(match_path, pairs_path, min_matches)
    stats = {"weak_pairs": len(weak), "rescued_pairs": 0, "added_matches": 0}
    if not weak:
        return stats

    model = _load_roma()
    img_root = Path(image_dir)

    # Cache keypoints + KD-trees per image
    kpts_cache: dict[str, np.ndarray] = {}
    tree_cache: dict[str, cKDTree] = {}

    def _kpts(name: str):
        if name not in kpts_cache:
            k = get_keypoints(Path(feature_path), name)
            kpts_cache[name] = k
            tree_cache[name] = cKDTree(k)
        return kpts_cache[name], tree_cache[name]

    for i, (name0, name1, n_before) in enumerate(weak):
        if progress_cb:
            progress_cb(i, len(weak), f"RoMa rescue {name0} ↔ {name1} "
                                      f"({n_before} matches)")

        path0, path1 = img_root / name0, img_root / name1
        with Image.open(path0) as im:
            W0, H0 = im.size
        with Image.open(path1) as im:
            W1, H1 = im.size

        preds = model.match(str(path0), str(path1))
        matches, *_ = model.sample(preds, samples_per_pair)
        pts0, pts1 = model.to_pixel_coordinates(matches, H0, W0, H1, W1)
        pts0 = pts0.cpu().numpy()
        pts1 = pts1.cpu().numpy()

        k0, t0 = _kpts(name0)
        k1, t1 = _kpts(name1)

        d0, i0 = t0.query(pts0, distance_upper_bound=snap_radius)
        d1, i1 = t1.query(pts1, distance_upper_bound=snap_radius)
        valid = (d0 < snap_radius) & (d1 < snap_radius)
        cand = np.stack([i0[valid], i1[valid]], axis=1)
        if len(cand) == 0:
            continue
        # Dedupe by keypoint index on either side
        _, uniq0 = np.unique(cand[:, 0], return_index=True)
        cand = cand[uniq0]
        _, uniq1 = np.unique(cand[:, 1], return_index=True)
        cand = cand[uniq1]

        # Merge into the matches file (respect existing assignments)
        added = 0
        with h5py.File(str(match_path), "r+") as h5:
            pair, reverse = _pair_key(h5, name0, name1)
            grp = h5[pair]
            matches0 = grp["matches0"].__array__()
            scores0 = grp["matching_scores0"].__array__()
            for a, b in cand:
                # matches0 is indexed by the FIRST image of the stored pair
                src, dst = (int(a), int(b)) if not reverse else (int(b), int(a))
                if src < len(matches0) and matches0[src] == -1 \
                        and dst not in matches0:
                    matches0[src] = dst
                    scores0[src] = 0.5  # nominal confidence for rescued matches
                    added += 1
            if added:
                del grp["matches0"]
                del grp["matching_scores0"]
                grp.create_dataset("matches0", data=matches0)
                grp.create_dataset("matching_scores0", data=scores0)

        if added:
            stats["rescued_pairs"] += 1
            stats["added_matches"] += added

    if progress_cb:
        progress_cb(len(weak), len(weak),
                    f"Rescued {stats['rescued_pairs']}/{len(weak)} pairs, "
                    f"+{stats['added_matches']} matches")
    return stats
