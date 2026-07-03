"""EDGS-style dense-matcher triangulation init for 3DGS (apex-sfm WP-F2).

Motivation (DEVELOPER_NOTES §4 / quality-roadmap density profiling): textureless
surfaces (the Laptop "deck"/palm-rest) are Gaussian-STARVED — the deck carries only
~0.28x the SfM-init point density of the textured keyboard, and MCMC drains it
further during training (0.16x trained). ALIKED+LightGlue simply finds few
keypoints on flat/dark/semi-gloss plastic. This module adds a DENSE point layer on
those regions by matching every covisible image pair with the RoMa dense matcher
(EDGS's matcher) and triangulating the correspondences against the EXISTING COLMAP
poses — no pose change, points only.

Pipeline (all via our editable HLoc stack — EDGS-as-shipped is a poor technical fit;
see docs/EXECUTION_PLAN.md WP-F2):
  1. pairs_from_covisibility  -> covisible image pairs (num_matched per image)
  2. match_dense (roma conf)  -> dense warp-sampled correspondences (max_kps=None:
        the 8192 per-image default would silently gut the dense init)
  3. hloc.triangulation.main  -> triangulate vs the reference poses (Python API only;
        the CLI __main__ is broken). Poses PRESERVED, points only; epipolar
        verification vs reference poses removes outliers.
  4. merge + voxel-dedup      -> concat the new dense points with the original sparse
        cloud, deduped on a voxel grid sized ~ the sparse cloud's median-NN spacing.
  5. write augmented COLMAP model (cameras/images/poses identical to the reference;
        points3D = original tracks + new dense points with empty tracks).

Selectivity (CoMapGS finding — random point injection HURTS -0.2..-0.7 dB): we keep
ALL triangulated points but REPORT per-region density, and expose `low_density_only`
to restrict the EXTRA seeding to under-dense voxels if a global A/B regresses.

The augmented model drops straight into the trainer's existing
`init_gaussians_from_sfm` (color from COLMAP track mean -> SH DC, KNN scales) — no
trainer change needed; a `--dense_init` flag on prepare_data swaps it in.

romatch is MIT (pip); RoMa weights auto-download via torch.hub. See THIRD_PARTY.md.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# DataLoader worker override (Windows + 4K images crash hloc's hardcoded 16
# workers). RoMa GPU matching dominates runtime, so serial loading is fine.
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _force_dataloader_workers(num_workers: int):
    import torch.utils.data as tud

    orig = tud.DataLoader

    class _Patched(orig):
        def __init__(self, *args, **kwargs):
            kwargs["num_workers"] = num_workers
            if num_workers == 0:
                kwargs.pop("persistent_workers", None)
            super().__init__(*args, **kwargs)

    tud.DataLoader = _Patched
    try:
        yield
    finally:
        tud.DataLoader = orig


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def _median_nn_distance(xyz: np.ndarray, sample: int = 20000, seed: int = 0) -> float:
    """Median nearest-neighbour distance of a point set (voxel-size heuristic)."""
    from scipy.spatial import cKDTree

    if len(xyz) < 2:
        return 0.0
    if len(xyz) > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(xyz), size=sample, replace=False)
        q = xyz[idx]
    else:
        q = xyz
    tree = cKDTree(xyz)
    d, _ = tree.query(q, k=2)  # [:,0]=self(0), [:,1]=nearest other
    return float(np.median(d[:, 1]))


def _voxel_keys(xyz: np.ndarray, voxel: float, origin: np.ndarray) -> np.ndarray:
    return np.floor((xyz - origin) / voxel).astype(np.int64)


def _pose_delta(rec_a, rec_b) -> dict:
    """Max pose deltas between two reconstructions sharing image names.

    Returns {"max_dt": max |Δtranslation| (world units),
             "max_dR_fro": max Frobenius norm of (Ra − Rb) — EXACTLY 0 for
                 bitwise-identical rotations and well-conditioned (use for
                 asserts),
             "max_drot_deg": max rotation-angle delta (degrees) — human-readable
                 but arccos is ill-conditioned near identity (float64 matmul
                 noise alone reads ~1e-6 deg; NOT a drift signal below ~1e-4),
             "n_images_compared": N}.
    Used to PERSIST the poses-preserved claim (§0 gate-evidence rule) —
    triangulation must never move cameras."""
    by_name = {im.name: im for im in rec_b.images.values()}
    max_dt = 0.0
    max_dR = 0.0
    max_drot = 0.0
    n = 0
    for im in rec_a.images.values():
        other = by_name.get(im.name)
        if other is None:
            continue
        ca, cb = im.cam_from_world(), other.cam_from_world()
        dt = float(np.linalg.norm(np.asarray(ca.translation)
                                  - np.asarray(cb.translation)))
        Ra = np.asarray(ca.rotation.matrix())
        Rb = np.asarray(cb.rotation.matrix())
        dR = float(np.linalg.norm(Ra - Rb))
        cos = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
        drot = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        max_dt = max(max_dt, dt)
        max_dR = max(max_dR, dR)
        max_drot = max(max_drot, drot)
        n += 1
    return {"max_dt": max_dt, "max_dR_fro": max_dR, "max_drot_deg": max_drot,
            "n_images_compared": n}


def _ensure_feature_coverage(features_path, reference_model) -> int:
    """hloc.triangulation.import_features iterates over EVERY reference image, so an
    image that never appeared in a covisible pair (no features written) would crash
    it with a KeyError. Stub empty keypoints for any uncovered reference image so
    triangulation proceeds (import_matches skips its pairs; 0 kpts is handled)."""
    import h5py
    import pycolmap

    ref = pycolmap.Reconstruction(str(reference_model))
    names = [im.name for im in ref.images.values()]
    added = 0
    with h5py.File(str(features_path), "a") as fd:
        for name in names:
            if name in fd and "keypoints" in fd[name]:
                continue
            grp = fd.require_group(name)
            if "keypoints" not in grp:
                grp.create_dataset("keypoints", data=np.zeros((0, 2), dtype=np.float32))
                added += 1
    return added


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def build_dense_init(
    reference_model: str,
    image_dir: str,
    out_model_dir: str,
    work_dir: Optional[str] = None,
    num_matched: int = 5,
    matcher_weights: str = "outdoor",
    max_num_matches: int = 15000,
    min_certainty: float = 0.05,
    voxel_scale: float = 1.0,
    low_density_only: bool = False,
    low_density_cell: Optional[float] = None,
    low_density_thresh: int = 3,
    num_workers: int = 0,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Build an augmented COLMAP model with EDGS-style dense triangulated points.

    Args:
        reference_model: dir with the reference sparse model (cameras/images/points3D).
        image_dir: dir with the (COLMAP-named) images to match on.
        out_model_dir: dir to write the augmented model into.
        work_dir: scratch dir for pairs/features/matches/triangulated (default:
            <out_model_dir>/_dense_init_work).
        num_matched: covisible pairs per image (limit for >300-image scenes).
        matcher_weights: "outdoor" | "indoor" RoMa weights.
        max_num_matches / min_certainty: RoMa sampling (per-pair count / cert floor).
        voxel_scale: multiplies the median-NN voxel size used for dedup.
        low_density_only: if True, only ADD dense points that fall in coarse cells
            whose original-sparse count is < low_density_thresh (CoMapGS selectivity).
        low_density_cell: side length of the coarse selectivity cell (world units);
            default = 8 x the dedup voxel.
        low_density_thresh: original-point count below which a cell is "low density".
        num_workers: DataLoader workers for dense matching (0 = safe on Windows/4K).
        seed: RNG seed applied (torch CPU+CUDA) before the matching phase — RoMa's
            sample() draws matches via torch.multinomial, which is otherwise
            unseeded. See wpf2_RECORD.md "Determinism" for the verified status.

    Returns a stats dict (also written to <work_dir>/dense_init_stats.json).
    """
    import pycolmap
    from hloc import (  # editable install
        match_dense,
        pairs_from_covisibility,
        triangulation,
    )

    t_start = time.time()
    reference_model = Path(reference_model)
    image_dir = Path(image_dir)
    out_model_dir = Path(out_model_dir)
    out_model_dir.mkdir(parents=True, exist_ok=True)
    work = Path(work_dir) if work_dir else out_model_dir / "_dense_init_work"
    work.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if verbose:
            print(f"[dense_init] {msg}", flush=True)

    # --- 1. covisibility pairs --------------------------------------------
    pairs_path = work / "pairs-dense-covis.txt"
    log(f"pairs_from_covisibility (num_matched={num_matched}) -> {pairs_path.name}")
    pairs_from_covisibility.main(reference_model, pairs_path, num_matched)
    n_pairs = sum(1 for ln in pairs_path.read_text().splitlines() if ln.strip())
    log(f"  {n_pairs} covisible pairs")

    # --- 2. dense matching (RoMa) -----------------------------------------
    conf = {k: (dict(v) if isinstance(v, dict) else v)
            for k, v in match_dense.confs["roma"].items()}
    conf["model"]["weights"] = matcher_weights
    conf["model"]["max_num_matches"] = int(max_num_matches)
    conf["model"]["min_certainty"] = float(min_certainty)
    log(f"match_dense[roma weights={matcher_weights} num={max_num_matches} "
        f"cert>={min_certainty} seed={seed}] (max_kps=None) ...")
    # Seed torch RNG (CPU + CUDA): RoMa's sample() draws correspondences with an
    # otherwise-unseeded torch.multinomial. This pins the sampling stream; note
    # that full bitwise reproducibility additionally depends on cuDNN/CUDA
    # non-determinism in the network forward (recorded in wpf2_RECORD.md).
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    t0 = time.time()
    with _force_dataloader_workers(num_workers):
        # max_kps=None: do NOT cap keypoints/image (the 8192 default guts density).
        features, matches = match_dense.main(
            conf, pairs_path, image_dir, export_dir=work, max_kps=None,
        )
    t_match = time.time() - t0
    log(f"  dense matching done in {t_match/60:.1f} min "
        f"-> {Path(features).name}, {Path(matches).name}")

    # --- 3. triangulation vs reference poses (Python API only) ------------
    n_stub = _ensure_feature_coverage(features, reference_model)
    if n_stub:
        log(f"  stubbed empty keypoints for {n_stub} uncovered reference image(s)")
    sfm_dir = work / "triangulated"
    log("hloc.triangulation.main (poses preserved, points only) ...")
    t0 = time.time()
    tri_rec = triangulation.main(
        sfm_dir, reference_model, image_dir, pairs_path, Path(features),
        Path(matches), verbose=verbose,
    )
    t_tri = time.time() - t0
    log(f"  triangulated {tri_rec.num_points3D()} points in {t_tri/60:.1f} min")

    db_path = sfm_dir / "database.db"
    db_size_mb = db_path.stat().st_size / 1e6 if db_path.exists() else 0.0

    # --- 4. merge with original sparse (voxel dedup) ----------------------
    ref_rec = pycolmap.Reconstruction(str(reference_model))

    # Poses-preserved check (persisted per the §0 gate-evidence rule): the
    # triangulated model must carry EXACTLY the reference poses — triangulation
    # adds points, never moves cameras. Hard-fail if anything drifted.
    pose_check_tri = _pose_delta(ref_rec, tri_rec)
    log(f"pose check (ref vs triangulated): max|dt|={pose_check_tri['max_dt']:.3e} "
        f"max|dR|_F={pose_check_tri['max_dR_fro']:.3e} "
        f"(angle {pose_check_tri['max_drot_deg']:.3e} deg) over "
        f"{pose_check_tri['n_images_compared']} images")
    assert pose_check_tri["max_dt"] < 1e-9 and pose_check_tri["max_dR_fro"] < 1e-9, \
        f"triangulation moved poses: {pose_check_tri}"

    orig_ids = sorted(ref_rec.points3D.keys())
    orig_xyz = np.array([ref_rec.points3D[i].xyz for i in orig_ids], dtype=np.float64)
    n_orig = len(orig_xyz)

    tri_ids = sorted(tri_rec.points3D.keys())
    tri_xyz = np.array([tri_rec.points3D[i].xyz for i in tri_ids], dtype=np.float64)
    tri_rgb = np.array([tri_rec.points3D[i].color for i in tri_ids], dtype=np.uint8)
    tri_err = np.array([float(tri_rec.points3D[i].error) for i in tri_ids],
                       dtype=np.float64)
    n_tri = len(tri_xyz)
    mean_reproj = float(np.mean(tri_err)) if n_tri else 0.0
    med_reproj = float(np.median(tri_err)) if n_tri else 0.0

    voxel = voxel_scale * _median_nn_distance(orig_xyz)
    log(f"original {n_orig} pts, median-NN {voxel/voxel_scale:.5f} -> "
        f"voxel {voxel:.5f} (scale {voxel_scale})")

    # Build augmented model on top of the reference (keeps original tracks intact).
    aug = pycolmap.Reconstruction(str(reference_model))
    added = 0
    dropped_dup = 0
    dropped_sel = 0
    if n_tri and voxel > 0:
        origin = orig_xyz.min(axis=0) if n_orig else tri_xyz.min(axis=0)

        # Occupied voxels from the ORIGINAL cloud (dedup dense-vs-sparse).
        occ = set(map(tuple, _voxel_keys(orig_xyz, voxel, origin))) if n_orig else set()

        # Optional CoMapGS selectivity: only seed dense points in coarse cells that
        # are under-dense in the ORIGINAL sparse cloud.
        allow_cell = None
        if low_density_only:
            cell = low_density_cell if low_density_cell else 8.0 * voxel
            from collections import Counter
            ck = Counter(map(tuple, _voxel_keys(orig_xyz, cell, origin)))
            allow_cell = (cell, ck)

        empty_track = pycolmap.Track()
        tri_keys = _voxel_keys(tri_xyz, voxel, origin)
        for j in range(n_tri):
            key = tuple(tri_keys[j])
            if key in occ:  # dedups dense-vs-sparse AND dense-vs-dense
                dropped_dup += 1
                continue
            if allow_cell is not None:
                cell, ck = allow_cell
                cellkey = tuple(np.floor((tri_xyz[j] - origin) / cell).astype(np.int64))
                if ck.get(cellkey, 0) >= low_density_thresh:
                    dropped_sel += 1
                    continue
            aug.add_point3D(tri_xyz[j], empty_track, tri_rgb[j])
            occ.add(key)
            added += 1

    n_final = aug.num_points3D()
    log(f"merged: +{added} dense pts (dropped {dropped_dup} dup, {dropped_sel} "
        f"selectivity) -> {n_final} total ({n_final/max(n_orig,1):.2f}x sparse)")

    # --- 5. write augmented model -----------------------------------------
    aug.write(str(out_model_dir))
    log(f"wrote augmented model -> {out_model_dir}")

    # Poses-preserved check #2 on the WRITTEN artifact (reload round-trip): the
    # augmented model a trainer will actually consume must match the reference.
    aug_reloaded = pycolmap.Reconstruction(str(out_model_dir))
    pose_check_aug = _pose_delta(ref_rec, aug_reloaded)
    log(f"pose check (ref vs written augmented): "
        f"max|dt|={pose_check_aug['max_dt']:.3e} "
        f"max|dR|_F={pose_check_aug['max_dR_fro']:.3e} "
        f"(angle {pose_check_aug['max_drot_deg']:.3e} deg)")
    assert pose_check_aug["max_dt"] < 1e-9 and pose_check_aug["max_dR_fro"] < 1e-9, \
        f"augmented model poses drifted: {pose_check_aug}"

    stats = {
        "reference_model": str(reference_model),
        "image_dir": str(image_dir),
        "out_model_dir": str(out_model_dir),
        "num_matched": num_matched,
        "matcher": f"roma_{matcher_weights}",
        "max_num_matches": max_num_matches,
        "min_certainty": min_certainty,
        "n_pairs": n_pairs,
        "n_orig_points": n_orig,
        "n_triangulated_points": n_tri,
        "n_added_points": added,
        "n_dropped_duplicate": dropped_dup,
        "n_dropped_selectivity": dropped_sel,
        "n_final_points": n_final,
        "final_over_orig": n_final / max(n_orig, 1),
        "new_points_mean_reproj_px": mean_reproj,
        "new_points_median_reproj_px": med_reproj,
        "voxel_size": voxel,
        "voxel_scale": voxel_scale,
        "low_density_only": low_density_only,
        "seed": seed,
        "pose_check_ref_vs_triangulated": pose_check_tri,
        "pose_check_ref_vs_augmented": pose_check_aug,
        "database_mb": db_size_mb,
        "match_minutes": t_match / 60.0,
        "triangulation_minutes": t_tri / 60.0,
        "total_minutes": (time.time() - t_start) / 60.0,
    }
    (work / "dense_init_stats.json").write_text(json.dumps(stats, indent=2),
                                                encoding="utf-8")
    log(f"stats -> {work/'dense_init_stats.json'}")
    return stats


def _cli():
    import argparse

    ap = argparse.ArgumentParser(description="EDGS-style dense triangulation init (WP-F2)")
    ap.add_argument("--reference_model", required=True)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_model_dir", required=True)
    ap.add_argument("--work_dir", default=None)
    ap.add_argument("--num_matched", type=int, default=5)
    ap.add_argument("--matcher_weights", default="outdoor", choices=["outdoor", "indoor"])
    ap.add_argument("--max_num_matches", type=int, default=15000)
    ap.add_argument("--min_certainty", type=float, default=0.05)
    ap.add_argument("--voxel_scale", type=float, default=1.0)
    ap.add_argument("--low_density_only", action="store_true")
    ap.add_argument("--low_density_thresh", type=int, default=3)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0,
                    help="torch RNG seed for RoMa's multinomial match sampling")
    args = ap.parse_args()

    stats = build_dense_init(
        reference_model=args.reference_model,
        image_dir=args.image_dir,
        out_model_dir=args.out_model_dir,
        work_dir=args.work_dir,
        num_matched=args.num_matched,
        matcher_weights=args.matcher_weights,
        max_num_matches=args.max_num_matches,
        min_certainty=args.min_certainty,
        voxel_scale=args.voxel_scale,
        low_density_only=args.low_density_only,
        low_density_thresh=args.low_density_thresh,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _cli()
