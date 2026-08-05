"""WP-O5 — GPU feature backend for the 360 crop-rig SfM lane (ALIKED + LightGlue via HLoc).

`core.sfm_360 --features sift` (the default) is untouched: pycolmap's CPU SIFT extraction +
`match_sequential`. `--features aliked` swaps ONLY the "put keypoints and verified matches into
the COLMAP database" step for HLoc's GPU stack, and hands the *same* database to the *same*
`pycolmap.incremental_mapping` call. Everything that defines the reconstruction problem — the
12-crop rig geometry, the fixed SIMPLE_PINHOLE intrinsics, `apply_rig_config`, the WP-O1
masks, `rig_verification`, `skip_image_pairs_in_same_frame`, the mapper options — is shared,
so the two feature stacks are a controlled A/B.

Why this exists: WP-O2 measured the 151-pano crop-rig run at 4 h 14 m, of which SIFT
*extraction* was only 92 s — the cost is CPU **matching** (95 min) and mapping (157 min).
8K panos are 4x the pixels per crop, so the CPU path scales the wrong way. ALIKED+LightGlue
run on the GPU and produce far fewer, far cleaner matches, which shortens mapping too.

Pipeline (crop_rig only; the spherical arm has no rig and is not wired up):

  1. `pycolmap.import_images` PER_FOLDER with the rig's SIMPLE_PINHOLE model+params
     -> cameras + images, no features. (The SIFT path gets these from `extract_features`.)
  2. `pycolmap.apply_rig_config` -> rigs + frames. Must precede pair generation because
     `SequentialPairingOptions.expand_rig_images` needs frames to expand.
  3. HLoc `extract_features` (ALIKED) on the crop images -> features.h5.
  4. **Mask filter**: HLoc has no notion of an input mask, so every keypoint that lands on a
     0 pixel of the crop mask (`<mask_dir>/pano_cameraN/<image>.png`, 255 = VALID) is dropped
     from keypoints/descriptors/scores BEFORE matching. This reproduces what COLMAP's
     `MaskFeatures` does for SIFT (feature_extraction.cc), and the drop counts are reported so
     the WP-O2 "0 keypoints on masked-out pixels" check can be repeated.
  5. Pairs: `pycolmap.SequentialPairGenerator` on the same `SequentialPairingOptions` the SIFT
     path uses, so the pairing is IDENTICAL; same-frame pairs are dropped here because the
     SIFT path drops them at match time via `skip_image_pairs_in_same_frame` (the 12 crops of
     one pano share no pixel, so they cannot match). Optional NetVLAD retrieval pairs
     (`--retrieval_num`) are unioned in as the loop-closure equivalent of COLMAP's
     `--loop_detection 1`, which WP-O2 ran with.
  6. HLoc `match_features` (LightGlue) over those pairs -> matches.h5.
  7. `hloc.triangulation.import_features` / `import_matches` (raw, unverified).
  8. `pycolmap.geometric_verification(..., GeometricVerifierOptions(rig_verification=True))`
     — the rig-aware verifier, i.e. the same `rig_verification=True` the SIFT path sets on
     `FeatureMatchingOptions`. `hloc`'s own `estimation_and_geometric_verification` calls
     `pycolmap.verify_matches`, which takes no rig options, so it is deliberately NOT used.

PROVENANCE (POC stance — log, do not filter): HLoc v1.5 (Apache-2.0, editable clone at
../Hierarchical-Localization), ALIKED + LightGlue weights via the `lightglue` package, NetVLAD
weights fetched by HLoc on first use. Ledger entries in THIRD_PARTY.md.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

# h5 datasets written per KEYPOINT by hloc's extractors, i.e. the ones the mask filter must
# subset. Everything else in the group (notably `image_size`, shape (2,)) is per IMAGE and
# must be copied through untouched — matching on "1-D and as long as the keypoint array"
# would silently corrupt `image_size` on an image that happens to have exactly 2 keypoints.
PER_KEYPOINT_KEYS = ("keypoints", "keypoint_scores", "scores", "scales", "oris",
                     "descriptors")


# ---------------------------------------------------------------------------
# feature configs — production values from core/feature_configs.py
# ---------------------------------------------------------------------------

def aliked_conf(max_keypoints: int, resize_max: int) -> dict:
    """HLoc extractor conf. `resize_max` <= 0 means "no resize" (native crop resolution)."""
    eff = int(resize_max) if resize_max and resize_max > 0 else 99999
    return {
        "output": f"feats-aliked-n{max_keypoints}-r{eff}",
        "model": {"name": "aliked", "max_num_keypoints": int(max_keypoints)},
        "preprocessing": {"grayscale": False, "resize_max": eff, "as_half": False},
    }


def lightglue_conf() -> dict:
    return {"output": "matches-aliked-lightglue", "model": {"name": "lightglue",
                                                            "features": "aliked"}}


def netvlad_conf() -> dict:
    from hloc import extract_features
    return extract_features.confs["netvlad"]


# ---------------------------------------------------------------------------
# 1. mask filtering — HLoc has no mask support; COLMAP's SIFT path does
# ---------------------------------------------------------------------------

def filter_features_by_mask(features_path: Path, mask_dir: Path, image_names,
                            progress: int = 200) -> dict:
    """Drop every keypoint whose crop-mask pixel is 0. Rewrites the h5 groups in place.

    Mask lookup mirrors `PanoProcessor.process`: the mask for image `pano_cameraN/x.jpg` is
    `<mask_dir>/pano_cameraN/x.jpg.png` (COLMAP's extension-append convention), uint8 with
    255 = VALID. HLoc keypoints are already rescaled to the ORIGINAL crop resolution and use
    the OpenCV origin (pixel centre at integer), so `round()` is the right sampling rule.
    """
    import h5py

    stats = {"images": 0, "kept": 0, "dropped": 0, "empty_images": [],
             "min_kept": None, "missing_masks": 0}
    t0 = time.time()
    with h5py.File(str(features_path), "r+", libver="latest") as fd:
        for n, name in enumerate(image_names):
            grp = fd[name]
            kpts = grp["keypoints"].__array__()
            mp = mask_dir / f"{name}.png"
            if not mp.is_file():
                stats["missing_masks"] += 1
                continue
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                stats["missing_masks"] += 1
                continue
            h, w = mask.shape[:2]
            xi = np.clip(np.round(kpts[:, 0]).astype(np.int64), 0, w - 1)
            yi = np.clip(np.round(kpts[:, 1]).astype(np.int64), 0, h - 1)
            keep = mask[yi, xi] > 0
            n_keep = int(keep.sum())
            stats["images"] += 1
            stats["kept"] += n_keep
            stats["dropped"] += int(kpts.shape[0] - n_keep)
            stats["min_kept"] = n_keep if stats["min_kept"] is None \
                else min(stats["min_kept"], n_keep)
            if n_keep == 0:
                stats["empty_images"].append(name)
            if n_keep == kpts.shape[0]:
                continue
            uncertainty = grp["keypoints"].attrs.get("uncertainty")
            payload = {}
            for k in list(grp.keys()):
                v = grp[k].__array__()
                if k not in PER_KEYPOINT_KEYS:
                    payload[k] = v                       # image_size etc. — untouched
                elif k == "descriptors":
                    # hloc stores ALIKED descriptors transposed: (D, N)
                    payload[k] = v[:, keep] if v.shape[-1] == keep.shape[0] else v[keep]
                else:
                    payload[k] = v[keep]
            del fd[name]
            g2 = fd.create_group(name)
            for k, v in payload.items():
                g2.create_dataset(k, data=v)
            if uncertainty is not None:
                g2["keypoints"].attrs["uncertainty"] = uncertainty
            if progress and (n + 1) % progress == 0:
                print(f"[mask-filter] {n + 1}/{len(image_names)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    total = stats["kept"] + stats["dropped"]
    stats["dropped_frac"] = round(stats["dropped"] / total, 6) if total else 0.0
    stats["seconds"] = round(time.time() - t0, 2)
    print(f"[mask-filter] {stats['images']} images: kept {stats['kept']}, dropped "
          f"{stats['dropped']} ({stats['dropped_frac'] * 100:.1f}%), "
          f"min/image {stats['min_kept']}, {stats['seconds']:.0f}s", flush=True)
    return stats


def verify_mask_integrity(features_path: Path, mask_dir: Path, image_names,
                          sample: int = 60, seed: int = 0) -> dict:
    """Independent re-check: no surviving keypoint may sit on a masked-out pixel."""
    import cv2
    import h5py

    rng = np.random.default_rng(seed)
    names = list(image_names)
    pick = names if len(names) <= sample else \
        [names[i] for i in rng.choice(len(names), sample, replace=False)]
    checked = violations = 0
    with h5py.File(str(features_path), "r", libver="latest") as fd:
        for name in pick:
            kpts = fd[name]["keypoints"].__array__()
            mask = cv2.imread(str(mask_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None or kpts.shape[0] == 0:
                continue
            h, w = mask.shape[:2]
            xi = np.clip(np.round(kpts[:, 0]).astype(np.int64), 0, w - 1)
            yi = np.clip(np.round(kpts[:, 1]).astype(np.int64), 0, h - 1)
            checked += int(kpts.shape[0])
            violations += int((mask[yi, xi] == 0).sum())
    out = {"images_checked": len(pick), "keypoints_checked": checked,
           "keypoints_on_masked_pixels": violations}
    print(f"[mask-check] {violations} keypoints on masked-out pixels out of {checked} "
          f"checked ({len(pick)} images)", flush=True)
    return out


# ---------------------------------------------------------------------------
# 2. pair generation — same options object the SIFT path matches with
# ---------------------------------------------------------------------------

def sequential_pairs(database_path: Path, pairing_options, skip_same_frame: bool = True) -> list:
    """(name0, name1) pairs from COLMAP's own sequential generator, run on our database."""
    import pycolmap

    with pycolmap.Database.open(database_path) as db:
        images = {im.image_id: im for im in db.read_all_images()}
        gen = pycolmap.SequentialPairGenerator(pairing_options, db)
        raw = gen.all_pairs()
    frame_of = {iid: im.frame_id for iid, im in images.items()}
    out, seen = [], set()
    dropped_same_frame = 0
    for a, b in raw:
        if a == b:
            continue
        if skip_same_frame and frame_of.get(a, -1) == frame_of.get(b, -2):
            dropped_same_frame += 1
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        out.append((images[key[0]].name, images[key[1]].name))
    print(f"[pairs] sequential: {len(out)} pairs "
          f"({dropped_same_frame} same-frame pairs dropped)", flush=True)
    return out


def retrieval_pairs(image_dir: Path, workdir: Path, image_names, num_matched: int) -> list:
    """NetVLAD top-k global pairs — the loop-closure equivalent of --loop_detection 1."""
    from hloc import extract_features, pairs_from_retrieval

    t0 = time.time()
    desc = extract_features.main(netvlad_conf(), image_dir, export_dir=workdir,
                                 image_list=list(image_names))
    out_txt = workdir / "pairs_retrieval.txt"
    pairs_from_retrieval.main(desc, out_txt, num_matched=int(num_matched))
    pairs = [tuple(line.split()) for line in
             out_txt.read_text(encoding="utf8").splitlines() if line.strip()]
    print(f"[pairs] netvlad top-{num_matched}: {len(pairs)} pairs "
          f"({time.time() - t0:.0f}s)", flush=True)
    return pairs


def union_pairs(*groups) -> list:
    seen, out = set(), []
    for g in groups:
        for a, b in g:
            key = (a, b) if a < b else (b, a)
            if key in seen or a == b:
                continue
            seen.add(key)
            out.append(key)
    return out


def write_pairs(pairs, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{a} {b}\n" for a, b in pairs), encoding="utf8")
    return path


# ---------------------------------------------------------------------------
# 3. the whole thing
# ---------------------------------------------------------------------------

def build_database(args, database_path: Path, image_dir: Path, mask_dir: Path | None,
                   camera, rig_config, pairing_options, workdir: Path,
                   stage=lambda name: None) -> dict:
    """Fill `database_path` with ALIKED keypoints + rig-verified LightGlue matches.

    Returns a report dict. On return the database holds cameras, images, rigs, frames,
    keypoints, matches and two_view_geometries — exactly what `incremental_mapping` wants.
    """
    import pycolmap
    from hloc import extract_features, match_features
    from hloc.reconstruction import get_image_ids
    from hloc.triangulation import import_features, import_matches

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    report = {"backend": "hloc-aliked-lightglue",
              "aliked_max_keypoints": int(args.aliked_max_keypoints),
              "aliked_resize_max": int(args.aliked_resize_max),
              "retrieval_num": int(args.retrieval_num),
              "workdir": str(workdir)}
    timings = {}

    # -- 1. images + cameras ------------------------------------------------
    stage("hloc_import_images")
    t0 = time.time()
    # pycolmap.import_images requires the file to exist (images.cc:30 CHECK); the SIFT path
    # gets it for free because extract_features creates it.
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    with pycolmap.Database.open(database_path):
        pass
    reader = pycolmap.ImageReaderOptions(camera_model=camera.model_name,
                                         camera_params=camera.params_to_string())
    pycolmap.import_images(database_path, image_dir,
                           pycolmap.CameraMode.PER_FOLDER, options=reader)
    image_ids = get_image_ids(database_path)
    names = sorted(image_ids)
    timings["import_images"] = round(time.time() - t0, 2)
    print(f"[hloc] imported {len(names)} images "
          f"({timings['import_images']:.0f}s)", flush=True)

    # -- 2. rig config (before pairing: expand_rig_images needs frames) -----
    stage("hloc_apply_rig_config")
    with pycolmap.Database.open(database_path) as db:
        pycolmap.apply_rig_config([rig_config], db)

    # -- 3. ALIKED ----------------------------------------------------------
    stage("hloc_extract_aliked")
    t0 = time.time()
    conf = aliked_conf(args.aliked_max_keypoints, args.aliked_resize_max)
    features_path = extract_features.main(conf, image_dir, export_dir=workdir,
                                          image_list=names)
    timings["extract"] = round(time.time() - t0, 2)
    report["features_h5"] = str(features_path)
    print(f"[hloc] ALIKED on {len(names)} crops in {timings['extract']:.0f}s", flush=True)

    # -- 4. mask filter -----------------------------------------------------
    if mask_dir is not None:
        stage("hloc_mask_filter")
        report["mask_filter"] = filter_features_by_mask(features_path, Path(mask_dir), names,
                                                        progress=args.progress * 12)
        report["mask_check"] = verify_mask_integrity(features_path, Path(mask_dir), names)
    else:
        report["mask_filter"] = None

    # -- 5. pairs -----------------------------------------------------------
    stage("hloc_pairs")
    t0 = time.time()
    seq = sequential_pairs(database_path, pairing_options, skip_same_frame=True)
    ret = retrieval_pairs(image_dir, workdir, names, args.retrieval_num) \
        if args.retrieval_num else []
    pairs = union_pairs(seq, ret)
    pairs_path = write_pairs(pairs, workdir / "pairs.txt")
    timings["pairs"] = round(time.time() - t0, 2)
    report["pairs"] = {"sequential": len(seq), "retrieval": len(ret), "union": len(pairs),
                       "path": str(pairs_path)}
    print(f"[hloc] {len(pairs)} unique pairs "
          f"(seq {len(seq)} + netvlad {len(ret)}), {timings['pairs']:.0f}s", flush=True)

    # -- 6. LightGlue -------------------------------------------------------
    stage("hloc_match_lightglue")
    t0 = time.time()
    matches_path = match_features.main(lightglue_conf(), pairs_path, features_path.stem,
                                       export_dir=workdir)
    timings["match"] = round(time.time() - t0, 2)
    report["matches_h5"] = str(matches_path)
    print(f"[hloc] LightGlue {len(pairs)} pairs in {timings['match']:.0f}s "
          f"({len(pairs) / max(timings['match'], 1e-9):.1f} pairs/s)", flush=True)

    # -- 7. into the database ----------------------------------------------
    stage("hloc_import_features")
    t0 = time.time()
    with pycolmap.Database.open(database_path) as db:
        import_features(image_ids, db, features_path)
        import_matches(image_ids, db, pairs_path, matches_path,
                       min_match_score=None, skip_geometric_verification=True)
    timings["import"] = round(time.time() - t0, 2)

    # -- 8. rig-aware geometric verification --------------------------------
    stage("hloc_geometric_verification")
    t0 = time.time()
    verifier = pycolmap.GeometricVerifierOptions(num_threads=int(args.num_threads),
                                                 rig_verification=True)
    two_view = pycolmap.TwoViewGeometryOptions()
    two_view.ransac.max_num_trials = 20000
    two_view.ransac.min_inlier_ratio = 0.1
    pycolmap.geometric_verification(
        database_path, verifier_options=verifier,
        pairing_options=pycolmap.ExistingMatchedPairingOptions(),
        two_view_geometry_options=two_view)
    timings["verify"] = round(time.time() - t0, 2)

    with pycolmap.Database.open(database_path) as db:
        report["database"] = {
            "num_images": db.num_images(), "num_cameras": db.num_cameras(),
            "num_rigs": db.num_rigs(), "num_frames": db.num_frames(),
            "num_keypoints": db.num_keypoints(),
            "num_matched_image_pairs": db.num_matched_image_pairs(),
            "num_verified_image_pairs": db.num_verified_image_pairs(),
            "num_inlier_matches": db.num_inlier_matches()}
    report["rig_verification"] = True
    report["timings_s"] = timings
    report["seconds"] = round(sum(timings.values()), 2)
    print(f"[hloc] verified {report['database']['num_verified_image_pairs']} / "
          f"{report['database']['num_matched_image_pairs']} pairs, "
          f"{report['database']['num_inlier_matches']} inlier matches "
          f"({timings['verify']:.0f}s)", flush=True)
    return report
