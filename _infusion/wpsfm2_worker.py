"""WP-SfM2 worker script — executed under either system Python (pycolmap 4.1.0)
or D:\\REPOS\\HYPERclone\\venvs\\colmap421\\Scripts\\python.exe (pycolmap 4.2.1).

Subcommands:
  inspect_api   Dump pycolmap version, build flags, enums, and option schemas.
  run_erp       Run ERP (EQUIRECTANGULAR) SfM: extract+match (optional) + incremental/global mapper.
  run_rig_map   Run incremental or global mapper on a pre-built crop-rig database copy,
                then fold back to equirectangular via PanoProcessor.convert_to_equirectangular().
  prep_loma_rig Apply RigConfig and camera prior focal lengths to a database after colmap.exe
                feature_extractor, before colmap.exe sequential_matcher.
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _f(val, nd: int = 6) -> float:
    return round(float(val), nd)


def db_summary(db_path: Path) -> dict:
    with pycolmap.Database.open(str(db_path)) as db:
        imgs = db.read_all_images()
        ks = [db.num_keypoints_for_image(im.image_id) for im in imgs]
        pair_ids, inliers = db.read_two_view_geometry_num_inliers()
        cfg_counts: dict[str, int] = collections.Counter()
        for pid in pair_ids:
            id1, id2 = pycolmap.pair_id_to_image_pair(pid)
            tvg = db.read_two_view_geometry(id1, id2)
            cfg_counts[str(tvg.config).split(".")[-1]] += 1
        return {
            "num_images": int(db.num_images()),
            "keypoints_total": int(sum(ks)),
            "keypoints_per_image_mean": _f(np.mean(ks), 2) if ks else 0.0,
            "keypoints_per_image_min": int(min(ks)) if ks else 0,
            "keypoints_per_image_max": int(max(ks)) if ks else 0,
            "matched_image_pairs": int(db.num_matched_image_pairs()),
            "verified_image_pairs": int(db.num_verified_image_pairs()),
            "inlier_matches": int(db.num_inlier_matches()),
            "inliers_per_verified_pair": _f(
                db.num_inlier_matches() / max(1, db.num_verified_image_pairs()), 2
            ),
            "two_view_config_counts": dict(sorted(cfg_counts.items())),
        }


def rec_detailed_metrics(
    rec: pycolmap.Reconstruction,
    n_input_panos: int,
    pano_of_image=None,
    images_per_pano: int = 1,
) -> dict:
    rec.update_point_3d_errors()
    per_pano: dict[str, int] = {}
    for image_id in rec.reg_image_ids():
        name = rec.images[image_id].name
        key = pano_of_image(name) if pano_of_image else name
        per_pano[key] = per_pano.get(key, 0) + 1
    reg_panos = set(per_pano)
    n_full = sum(1 for c in per_pano.values() if c >= int(images_per_pano))

    pts = list(rec.points3D.values())
    if pts:
        errs = np.array([p.error for p in pts], dtype=np.float64)
        tracks = np.array([len(p.track.elements) for p in pts], dtype=np.float64)
        mean_err = float(rec.compute_mean_reprojection_error())
        med_err = float(np.median(errs))
        p90_err = float(np.percentile(errs, 90))
        obs_weighted_err = float((errs * tracks).sum() / max(1.0, tracks.sum()))
        med_track = float(np.median(tracks))
        p90_track = float(np.percentile(tracks, 90))
    else:
        mean_err = med_err = p90_err = obs_weighted_err = med_track = p90_track = 0.0

    return {
        "num_input_panos": int(n_input_panos),
        "num_registered_panos": int(len(reg_panos)),
        "registered_pct": _f(100.0 * len(reg_panos) / max(1, n_input_panos), 2),
        "images_per_pano": int(images_per_pano),
        "panos_fully_registered": int(n_full),
        "panos_fully_registered_pct": _f(100.0 * n_full / max(1, n_input_panos), 2),
        "num_images": int(rec.num_images()),
        "num_reg_images": int(rec.num_reg_images()),
        "num_frames": int(rec.num_frames()),
        "num_reg_frames": int(rec.num_reg_frames()),
        "num_points3D": int(rec.num_points3D()),
        "num_observations": int(rec.compute_num_observations()),
        "mean_reproj_error_px": _f(mean_err, 6),
        "median_reproj_error_px": _f(med_err, 6),
        "p90_reproj_error_px": _f(p90_err, 6),
        "obs_weighted_reproj_error_px": _f(obs_weighted_err, 6),
        "mean_track_length": _f(rec.compute_mean_track_length(), 6),
        "median_track_length": _f(med_track, 2),
        "p90_track_length": _f(p90_track, 2),
        "mean_obs_per_reg_image": _f(rec.compute_mean_observations_per_reg_image(), 2),
    }


def extract_poses(rec: pycolmap.Reconstruction) -> dict[str, dict]:
    """Return {image_name: {'center': [x,y,z], 'R_c2w': [[3x3]], 'quat_wxyz': [w,x,y,z]}}."""
    out = {}
    for image_id in sorted(rec.reg_image_ids(), key=lambda i: rec.images[i].name):
        im = rec.images[image_id]
        cfw = im.cam_from_world()
        R_w2c = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
        R_c2w = R_w2c.T
        center = np.asarray(im.projection_center(), dtype=np.float64)
        q = cfw.rotation.quat  # xyzw
        out[im.name] = {
            "center": [float(v) for v in center],
            "R_c2w": [[float(v) for v in row] for row in R_c2w],
            "quat_wxyz": [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
        }
    return out


def cmd_inspect_api(args) -> int:
    fe_types = [x for x in dir(pycolmap.FeatureExtractorType) if not x.startswith("_") and x not in ("name", "value")]
    fm_types = [x for x in dir(pycolmap.FeatureMatcherType) if not x.startswith("_") and x not in ("name", "value")]
    ba_backends = [x for x in dir(pycolmap.BundleAdjustmentBackend) if not x.startswith("_") and x not in ("name", "value")]

    probes = {}
    for name in fe_types:
        if name == "UNDEFINED":
            continue
        fe = pycolmap.FeatureExtractionOptions()
        fe.type = getattr(pycolmap.FeatureExtractorType, name)
        try:
            pycolmap.FeatureExtractor.create(fe)
            probes[f"extractor_{name}"] = {"supported_in_wheel": True, "error": None}
        except Exception as e:
            probes[f"extractor_{name}"] = {"supported_in_wheel": False, "error": str(e)}

    for name in ba_backends:
        ba = pycolmap.BundleAdjustmentOptions()
        ba.backend = getattr(pycolmap.BundleAdjustmentBackend, name)
        rec = pycolmap.Reconstruction()
        cfg = pycolmap.BundleAdjustmentConfig()
        try:
            pycolmap.create_default_bundle_adjuster(ba, cfg, rec)
            probes[f"ba_backend_{name}"] = {"supported_in_wheel": True, "error": None}
        except Exception as e:
            probes[f"ba_backend_{name}"] = {"supported_in_wheel": False, "error": str(e)}

    info = {
        "pycolmap_version": pycolmap.__version__,
        "pycolmap_file": pycolmap.__file__,
        "has_cuda": bool(getattr(pycolmap, "has_cuda", False)),
        "hash_map_backend": getattr(pycolmap, "__hash_map_backend__", "unknown"),
        "has_panorama_module": (Path(pycolmap.__file__).parent / "panorama.py").is_file(),
        "has_global_mapping": hasattr(pycolmap, "global_mapping"),
        "has_hierarchical_mapping": hasattr(pycolmap, "hierarchical_mapping"),
        "feature_extractor_types": fe_types,
        "feature_matcher_types": fm_types,
        "ba_backends": ba_backends,
        "global_pipeline_options": sorted(
            x for x in dir(pycolmap.GlobalPipelineOptions()) if not x.startswith("_")
        ),
        "global_mapper_options": sorted(
            x for x in dir(pycolmap.GlobalMapperOptions()) if not x.startswith("_")
        ),
        "incremental_pipeline_options": sorted(
            x for x in dir(pycolmap.IncrementalPipelineOptions()) if not x.startswith("_")
        ),
        "wheel_probes": probes,
        "all_symbols": sorted(x for x in dir(pycolmap) if not x.startswith("_")),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(info, indent=1), encoding="utf8")
    return 0


def cmd_run_erp(args) -> int:
    from core.sfm_360 import estimate_up, pano_centers, trajectory_stats, upright_stats

    pycolmap.set_random_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = out_dir / "sparse"
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    db_path = out_dir / "database.db"
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    n_input = len([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    t_extract = 0.0
    t_match = 0.0
    if args.source_db:
        shutil.copyfile(args.source_db, db_path)
    else:
        if db_path.exists():
            db_path.unlink()
        t0 = time.time()
        reader = pycolmap.ImageReaderOptions(camera_model="EQUIRECTANGULAR")
        if mask_dir is not None:
            reader.mask_path = mask_dir
        ext_opts = pycolmap.FeatureExtractionOptions(
            use_gpu=False, num_threads=int(args.num_threads)
        )
        pycolmap.extract_features(
            db_path, image_dir,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader,
            extraction_options=ext_opts,
        )
        t_extract = time.time() - t0

        t0 = time.time()
        match_opts = pycolmap.FeatureMatchingOptions(
            use_gpu=False,
            num_threads=int(args.num_threads),
            guided_matching=bool(args.guided_matching),
        )
        pycolmap.match_exhaustive(db_path, matching_options=match_opts)
        t_match = time.time() - t0

    # Ensure prior focal length flag is set so global_mapping doesn't warn on EQUIRECTANGULAR
    with pycolmap.Database.open(str(db_path)) as db:
        for cam in db.read_all_cameras():
            if not cam.has_prior_focal_length:
                cam.has_prior_focal_length = True
                db.update_camera(cam)

    dbs = db_summary(db_path)

    t0 = time.time()
    if args.mapper == "incremental":
        inc_opts = pycolmap.IncrementalPipelineOptions(
            num_threads=int(args.num_threads),
            random_seed=int(args.seed),
        )
        recs = pycolmap.incremental_mapping(db_path, image_dir, sparse_dir, inc_opts)
    elif args.mapper == "global":
        glo_opts = pycolmap.GlobalPipelineOptions(
            num_threads=int(args.num_threads),
            random_seed=int(args.seed),
            mapper=pycolmap.GlobalMapperOptions(
                num_threads=int(args.num_threads),
                random_seed=int(args.seed),
            ),
        )
        recs = pycolmap.global_mapping(db_path, image_dir, sparse_dir, glo_opts)
    else:
        raise ValueError(f"Unknown mapper {args.mapper}")
    t_map = time.time() - t0

    if not recs:
        res = {
            "status": "failed_no_reconstruction",
            "pycolmap_version": pycolmap.__version__,
            "mapper": args.mapper,
            "guided_matching": bool(args.guided_matching),
            "timings_s": {
                "extract": _f(t_extract, 2),
                "match": _f(t_match, 2),
                "mapping": _f(t_map, 2),
                "total": _f(t_extract + t_match + t_map, 2),
            },
            "database": dbs,
        }
        (out_dir / "result.json").write_text(json.dumps(res, indent=1), encoding="utf8")
        return 1

    best_idx = max(recs, key=lambda i: recs[i].num_reg_images())
    rec = recs[best_idx]
    metrics = rec_detailed_metrics(rec, n_input)
    _, centers = pano_centers(rec)
    up = estimate_up(rec)
    traj = trajectory_stats(centers, up)
    traj["upright"] = upright_stats(rec)
    poses = extract_poses(rec)

    res = {
        "status": "ok",
        "pycolmap_version": pycolmap.__version__,
        "mapper": args.mapper,
        "guided_matching": bool(args.guided_matching),
        "used_masks": mask_dir is not None,
        "source_db": str(args.source_db) if args.source_db else None,
        "num_reconstructions": len(recs),
        "best_reconstruction": int(best_idx),
        "timings_s": {
            "extract": _f(t_extract, 2),
            "match": _f(t_match, 2),
            "mapping": _f(t_map, 2),
            "total": _f(t_extract + t_match + t_map, 2),
        },
        "database": dbs,
        "metrics_equirect": metrics,
        "trajectory": traj,
        "poses_equirect": poses,
    }
    (out_dir / "result.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    return 0


def cmd_run_rig_map(args) -> int:
    from core.sfm_360 import (
        PanoProcessor,
        RENDER_PRESETS,
        estimate_up,
        pano_centers,
        trajectory_stats,
        upright_stats,
    )

    pycolmap.set_random_seed(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = out_dir / "sparse"
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    db_path = out_dir / "database.db"
    shutil.copyfile(args.source_db, db_path)

    # Ensure prior focal length is marked True on all virtual cameras in the DB copy
    with pycolmap.Database.open(str(db_path)) as db:
        for cam in db.read_all_cameras():
            if not cam.has_prior_focal_length:
                cam.has_prior_focal_length = True
                db.update_camera(cam)

    dbs = db_summary(db_path)
    image_dir = Path(args.crop_image_dir)
    pano_w, pano_h = [int(x) for x in args.pano_size.split("x")]
    n_input_panos = int(args.n_input_panos)

    processor = PanoProcessor(
        pano_image_dir=image_dir,
        pano_mask_dir=None,
        out_image_dir=image_dir,
        out_mask_dir=out_dir / "masks_dummy",
        render_options=RENDER_PRESETS["perspective_overlapping"],
    )
    processor.prepare(pano_w, pano_h)
    n_cams = len(processor.cams_from_pano_rotation)

    t0 = time.time()
    if args.mapper == "incremental":
        inc_opts = pycolmap.IncrementalPipelineOptions(
            num_threads=int(args.num_threads),
            random_seed=int(args.seed),
            ba_refine_sensor_from_rig=False,
            ba_refine_focal_length=False,
            ba_refine_principal_point=False,
            ba_refine_extra_params=False,
        )
        recs = pycolmap.incremental_mapping(db_path, image_dir, sparse_dir, inc_opts)
    elif args.mapper == "global":
        glo_opts = pycolmap.GlobalPipelineOptions(
            num_threads=int(args.num_threads),
            random_seed=int(args.seed),
            mapper=pycolmap.GlobalMapperOptions(
                num_threads=int(args.num_threads),
                random_seed=int(args.seed),
                refine_sensor_from_rig=False,
            ),
        )
        glo_opts.mapper.bundle_adjustment.refine_focal_length = False
        glo_opts.mapper.bundle_adjustment.refine_principal_point = False
        glo_opts.mapper.bundle_adjustment.refine_extra_params = False
        recs = pycolmap.global_mapping(db_path, image_dir, sparse_dir, glo_opts)
    else:
        raise ValueError(f"Unknown mapper {args.mapper}")
    t_map = time.time() - t0

    if not recs:
        res = {
            "status": "failed_no_reconstruction",
            "pycolmap_version": pycolmap.__version__,
            "mapper": args.mapper,
            "timings_s": {"mapping": _f(t_map, 2)},
            "database": dbs,
        }
        (out_dir / "result.json").write_text(json.dumps(res, indent=1), encoding="utf8")
        return 1

    best_idx = max(recs, key=lambda i: recs[i].num_reg_images())
    rec = recs[best_idx]
    metrics_crop = rec_detailed_metrics(
        rec,
        n_input_panos,
        pano_of_image=lambda n: processor.split_image_name(n)[1],
        images_per_pano=n_cams,
    )

    t_conv0 = time.time()
    eq = processor.convert_to_equirectangular(rec)
    eq.update_point_3d_errors()
    eq_dir = out_dir / "sparse_equirectangular" / str(best_idx)
    eq_dir.mkdir(parents=True, exist_ok=True)
    eq.write(eq_dir)
    t_conv = time.time() - t_conv0

    metrics_eq = rec_detailed_metrics(eq, n_input_panos)
    _, centers = pano_centers(eq)
    up = estimate_up(eq)
    traj = trajectory_stats(centers, up)
    traj["upright"] = upright_stats(eq)

    res = {
        "status": "ok",
        "pycolmap_version": pycolmap.__version__,
        "mapper": args.mapper,
        "source_db": str(args.source_db),
        "num_reconstructions": len(recs),
        "best_reconstruction": int(best_idx),
        "timings_s": {
            "mapping": _f(t_map, 2),
            "convert_to_equirect": _f(t_conv, 2),
        },
        "database": dbs,
        "metrics_crop": metrics_crop,
        "metrics_equirect": metrics_eq,
        "trajectory": traj,
        "poses_equirect": extract_poses(eq),
        "poses_crop": extract_poses(rec),
    }
    (out_dir / "result.json").write_text(json.dumps(res, indent=1), encoding="utf8")
    return 0


def cmd_inspect_existing_rig(args) -> int:
    """Read an existing crop-rig + equirectangular reconstruction directory and emit result.json."""
    from core.sfm_360 import (
        PanoProcessor,
        RENDER_PRESETS,
        estimate_up,
        pano_centers,
        trajectory_stats,
        upright_stats,
    )

    run_dir = Path(args.run_dir)
    rep = json.loads((run_dir / "sfm_report.json").read_text(encoding="utf8"))
    pano_w, pano_h = rep["pano_size"]
    n_input_panos = rep["metrics_crop"]["num_input_panos"]
    processor = PanoProcessor(
        pano_image_dir=run_dir / "images",
        pano_mask_dir=None,
        out_image_dir=run_dir / "images",
        out_mask_dir=run_dir / "masks",
        render_options=RENDER_PRESETS["perspective_overlapping"],
    )
    processor.prepare(pano_w, pano_h)
    n_cams = len(processor.cams_from_pano_rotation)

    rec = pycolmap.Reconstruction(str(run_dir / "sparse" / "0"))
    eq = pycolmap.Reconstruction(str(run_dir / "sparse_equirectangular" / "0"))
    dbs = db_summary(run_dir / "database.db")

    metrics_crop = rec_detailed_metrics(
        rec,
        n_input_panos,
        pano_of_image=lambda n: processor.split_image_name(n)[1],
        images_per_pano=n_cams,
    )
    metrics_eq = rec_detailed_metrics(eq, n_input_panos)
    _, centers = pano_centers(eq)
    up = estimate_up(eq)
    traj = trajectory_stats(centers, up)
    traj["upright"] = upright_stats(eq)

    res = {
        "status": "ok",
        "pycolmap_version": rep["colmap"]["version"],
        "mapper": "incremental (banked)",
        "source_db": str(run_dir / "database.db"),
        "num_reconstructions": rep["num_reconstructions"],
        "best_reconstruction": rep["best_reconstruction"],
        "timings_s": rep["timings_s"],
        "database": dbs,
        "metrics_crop": metrics_crop,
        "metrics_equirect": metrics_eq,
        "trajectory": traj,
        "poses_equirect": extract_poses(eq),
        "poses_crop": extract_poses(rec),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf8")
    return 0


def cmd_inspect_existing_erp(args) -> int:
    from core.sfm_360 import estimate_up, pano_centers, trajectory_stats, upright_stats

    run_dir = Path(args.run_dir)
    rep = json.loads((run_dir / "sfm_report.json").read_text(encoding="utf8"))
    n_input_panos = rep["metrics_equirect"]["num_input_panos"]
    eq = pycolmap.Reconstruction(str(run_dir / "sparse" / "0"))
    dbs = db_summary(run_dir / "database.db")
    metrics_eq = rec_detailed_metrics(eq, n_input_panos)
    _, centers = pano_centers(eq)
    up = estimate_up(eq)
    traj = trajectory_stats(centers, up)
    traj["upright"] = upright_stats(eq)
    res = {
        "status": "ok",
        "pycolmap_version": rep["colmap"]["version"],
        "mapper": "incremental (banked)",
        "source_db": str(run_dir / "database.db"),
        "num_reconstructions": rep["num_reconstructions"],
        "best_reconstruction": rep["best_reconstruction"],
        "timings_s": rep["timings_s"],
        "database": dbs,
        "metrics_equirect": metrics_eq,
        "trajectory": traj,
        "poses_equirect": extract_poses(eq),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf8")
    return 0


def cmd_prep_loma_rig(args) -> int:
    import sqlite3
    from core.sfm_360 import PanoProcessor, RENDER_PRESETS

    pano_w, pano_h = [int(x) for x in args.pano_size.split("x")]
    processor = PanoProcessor(
        pano_image_dir=Path(args.crop_image_dir),
        pano_mask_dir=None,
        out_image_dir=Path(args.crop_image_dir),
        out_mask_dir=Path(args.crop_image_dir),
        render_options=RENDER_PRESETS["perspective_overlapping"],
    )
    processor.prepare(pano_w, pano_h)
    n_cams = len(processor.cams_from_pano_rotation)
    with pycolmap.Database.open(str(args.db_path)) as db:
        existing_cams = {c.camera_id: c for c in db.read_all_cameras()}
        for cid in range(1, n_cams + 1):
            if cid not in existing_cams:
                cam = pycolmap.Camera.create_from_model_id(
                    camera_id=cid,
                    model=processor.camera.model,
                    focal_length=float(processor.camera.params[0]),
                    width=int(processor.camera.width),
                    height=int(processor.camera.height),
                )
                cam.has_prior_focal_length = True
                db.write_camera(cam, use_camera_id=True)
    conn = sqlite3.connect(str(args.db_path))
    conn.execute("DELETE FROM rigs")
    conn.execute("DELETE FROM rig_sensors")
    conn.execute("DELETE FROM frames")
    conn.execute("DELETE FROM frame_data")
    for idx in range(n_cams):
        conn.execute(
            "UPDATE images SET camera_id = ? WHERE name LIKE ?",
            (idx + 1, f"pano_camera{idx}/%"),
        )
    conn.commit()
    conn.close()
    with pycolmap.Database.open(str(args.db_path)) as db:
        pycolmap.apply_rig_config([processor.rig_config], db)
        for cam in db.read_all_cameras():
            cam.has_prior_focal_length = True
            db.update_camera(cam)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_api = sub.add_parser("inspect_api")
    p_api.add_argument("--out", required=True)

    p_erp = sub.add_parser("run_erp")
    p_erp.add_argument("--image_dir", required=True)
    p_erp.add_argument("--mask_dir", default="")
    p_erp.add_argument("--source_db", default="")
    p_erp.add_argument("--out_dir", required=True)
    p_erp.add_argument("--mapper", choices=("incremental", "global"), default="incremental")
    p_erp.add_argument("--guided_matching", type=int, default=0)
    p_erp.add_argument("--num_threads", type=int, default=-1)
    p_erp.add_argument("--seed", type=int, default=0)

    p_rig = sub.add_parser("run_rig_map")
    p_rig.add_argument("--source_db", required=True)
    p_rig.add_argument("--crop_image_dir", required=True)
    p_rig.add_argument("--pano_size", required=True, help="WxH e.g. 7680x3840")
    p_rig.add_argument("--n_input_panos", type=int, required=True)
    p_rig.add_argument("--out_dir", required=True)
    p_rig.add_argument("--mapper", choices=("incremental", "global"), default="incremental")
    p_rig.add_argument("--num_threads", type=int, default=-1)
    p_rig.add_argument("--seed", type=int, default=0)

    p_ex = sub.add_parser("inspect_existing_rig")
    p_ex.add_argument("--run_dir", required=True)
    p_ex.add_argument("--out", required=True)

    p_ex_erp = sub.add_parser("inspect_existing_erp")
    p_ex_erp.add_argument("--run_dir", required=True)
    p_ex_erp.add_argument("--out", required=True)

    p_loma = sub.add_parser("prep_loma_rig")
    p_loma.add_argument("--db_path", required=True)
    p_loma.add_argument("--crop_image_dir", required=True)
    p_loma.add_argument("--pano_size", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "inspect_api":
        return cmd_inspect_api(args)
    if args.cmd == "run_erp":
        return cmd_run_erp(args)
    if args.cmd == "run_rig_map":
        return cmd_run_rig_map(args)
    if args.cmd == "inspect_existing_rig":
        return cmd_inspect_existing_rig(args)
    if args.cmd == "inspect_existing_erp":
        return cmd_inspect_existing_erp(args)
    if args.cmd == "prep_loma_rig":
        return cmd_prep_loma_rig(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
