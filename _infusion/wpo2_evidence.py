"""WP-O2 — evidence collector: turns the two arm run dirs into committable gate evidence.

The heavy outputs (crops, database.db, sparse/*.bin) live under the gitignored
`_dataset/360/sfm/*` and are reproducible from the argv recorded in every `sfm_report.json`.
This script re-derives everything a reviewer needs from repo state alone into
`_infusion/wpo2/`:

    ab_scorecard.json        both arms' metrics + argv + seed, side by side
    AB_TABLE.md              the same as a markdown table
    trajectory_<arm>.jpg     top-down camera track + step histogram (re-rendered as JPG,
                             `_infusion/.gitignore` drops *.png)
    poses_<arm>.csv          one row per registered pano: quaternion + translation + centre
    cameras_<arm>.txt        the reconstruction's camera model(s)
    sfm_report_<arm>.json    verbatim copy of the run report
    mask_integrity.json      INDEPENDENT re-check that no keypoint in either database sits
                             on a masked-out pixel

    python _infusion/wpo2_evidence.py --arm crop_rig=<dir> --arm spherical=<dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "_infusion" / "wpo2"


ARM_LABELS = {
    "crop_rig": "ARM A crop-rig",
    "spherical": "ARM B spherical",
    "spherical_ihfov200": "ARM B control (ih_fov 200)",
}


def label_of(arm: str) -> str:
    return ARM_LABELS.get(arm, arm)


def dump_poses(model_dir: Path, csv_path: Path) -> int:
    import pycolmap

    rec = pycolmap.Reconstruction(str(model_dir))
    rows = []
    for image_id in sorted(rec.reg_image_ids(), key=lambda i: rec.images[i].name):
        im = rec.images[image_id]
        p = im.cam_from_world()
        q = p.rotation.quat            # (x, y, z, w)
        c = np.asarray(im.projection_center())
        rows.append([im.name, *[f"{v:.9f}" for v in (q[3], q[0], q[1], q[2])],
                     *[f"{v:.9f}" for v in p.translation],
                     *[f"{v:.9f}" for v in c]])
    with csv_path.open("w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "qw", "qx", "qy", "qz", "tx", "ty", "tz", "cx", "cy", "cz"])
        w.writerows(rows)
    return len(rows)


def check_masks(run_dir: Path, image_root: Path, mask_root: Path, suffix_png: bool,
                sample: int, seed: int = 0) -> dict:
    """Re-open the COLMAP database and verify every sampled keypoint sits on mask != 0."""
    import pycolmap

    db_path = run_dir / "database.db"
    if not db_path.exists():
        return {"status": "no database"}
    rng = np.random.default_rng(seed)
    with pycolmap.Database.open(db_path) as db:
        images = db.read_all_images()
        idx = rng.choice(len(images), size=min(sample, len(images)), replace=False)
        n_kp = 0
        n_bad = 0
        n_img = 0
        missing = []
        for i in sorted(idx.tolist()):
            im = images[i]
            mp = mask_root / (im.name + ".png" if suffix_png else Path(im.name).stem + ".png")
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                missing.append(str(mp))
                continue
            kp = np.asarray(db.read_keypoints(im.image_id))
            if kp.size == 0:
                n_img += 1
                continue
            ix = np.clip(kp[:, 0].astype(int), 0, m.shape[1] - 1)
            iy = np.clip(kp[:, 1].astype(int), 0, m.shape[0] - 1)
            v = m[iy, ix]
            n_kp += len(kp)
            n_bad += int((v == 0).sum())
            n_img += 1
    return {"images_checked": n_img, "keypoints_checked": int(n_kp),
            "keypoints_on_masked_out_pixels": int(n_bad),
            "missing_masks": missing,
            "image_root": str(image_root), "mask_root": str(mask_root)}


def trajectory(run_dir: Path, arm: str, model_dir: Path) -> tuple:
    """Re-derive the trajectory stats + plot from the persisted camera centres.

    The stats live here rather than being copied out of `sfm_report.json` so that BOTH arms
    are described by the SAME code even when their runs happened at different times (ARM A
    was reconstructed before the PCA / up-vector descriptors were added). Nothing is
    re-reconstructed: only the poses already on disk are read.
    """
    import pycolmap

    from core.sfm_360 import (estimate_up, pano_centers, plot_trajectory, trajectory_stats,
                              upright_stats)

    rec = pycolmap.Reconstruction(str(model_dir))
    centers = pano_centers(rec)[1]
    up = estimate_up(rec)
    label = label_of(arm)
    dst = OUT / f"trajectory_{arm}.jpg"
    plot_trajectory(centers, dst, label, up)
    stats = trajectory_stats(centers, up)
    stats["upright"] = upright_stats(rec)
    # Planarity of the walk: how far the camera centres stray from their own best-fit plane.
    x = centers - centers.mean(0)
    normal = np.linalg.svd(x, full_matrices=False)[2][2]
    d = x @ normal
    stats["plane_residual_rms"] = round(float(d.std()), 6)
    stats["plane_residual_max"] = round(float(np.abs(d).max()), 6)
    stats["plane_normal_vs_up_deg"] = round(
        float(np.degrees(np.arccos(min(1.0, abs(float(normal @ up)))))), 4)
    return stats, dst.name


def dump_crop_rig(sparse_dir: Path, out_json: Path) -> dict:
    """The 12 virtual cameras + their fixed rig extrinsics — what WP-O3 needs to build the
    training dataset (pinhole intrinsics + per-crop pose = rig_from_world composed with
    cam_from_rig). Small enough to commit; the .bin model stays in _dataset/."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(sparse_dir))
    rig = next(iter(rec.rigs.values()))
    cams = []
    for cam_id, cam in sorted(rec.cameras.items()):
        entry = {"camera_id": int(cam_id), "model": cam.model_name,
                 "width": int(cam.width), "height": int(cam.height),
                 "params": [float(v) for v in cam.params]}
        sensor = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, cam_id)
        if rig.is_ref_sensor(sensor):
            entry["ref_sensor"] = True
            entry["cam_from_rig"] = None
        else:
            t = rig.sensor_from_rig(sensor)
            entry["ref_sensor"] = False
            entry["cam_from_rig"] = {
                "quat_xyzw": [float(v) for v in t.rotation.quat],
                "translation": [float(v) for v in t.translation]}
        cams.append(entry)
    data = {"rig_id": int(rig.rig_id), "num_sensors": len(cams), "cameras": cams,
            "note": "cam_from_rig is pure rotation (zero translation): the 12 virtual "
                    "cameras share the panorama's optical centre. Held FIXED in BA."}
    out_json.write_text(json.dumps(data, indent=1), encoding="utf8")
    return data


def fmt(v, nd=4):
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="arm=<run dir>, e.g. crop_rig=_dataset/360/sfm/wpo2_crop_rig")
    ap.add_argument("--log", action="append", default=[],
                    help="arm=<colmap log>, scanned for ceres/CHOLMOD warnings")
    ap.add_argument("--mask_sample", type=int, default=40)
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    logs = {}
    for spec in args.log:
        name, _, path = spec.partition("=")
        text = Path(path).read_text(encoding="utf8", errors="replace").splitlines()
        warn = [ln for ln in text if ln.startswith("W")]
        chol = [ln for ln in warn if "CHOLMOD warning: Matrix not positive definite" in ln]
        logs[name] = {
            "log": path, "warning_lines": len(warn),
            "ceres_cholmod_not_positive_definite": len(chol),
            "other_warnings": sorted({ln.split("] ", 1)[-1][:120]
                                      for ln in warn if ln not in chol}),
        }
    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        run_dir = (REPO / path).resolve() if not Path(path).is_absolute() else Path(path)
        rep = json.loads((run_dir / "sfm_report.json").read_text(encoding="utf8"))
        shutil.copyfile(run_dir / "sfm_report.json", OUT / f"sfm_report_{name}.json")

        model = Path(rep["equirect_dir"])
        n_poses = dump_poses(model, OUT / f"poses_{name}.csv")
        cam_txt = model / "cameras.txt"
        if not cam_txt.exists():          # models are written binary; re-emit text cameras
            import pycolmap
            r = pycolmap.Reconstruction(str(model))
            cam_txt = OUT / f"cameras_{name}.txt"
            cam_txt.write_text("\n".join(
                f"{c.camera_id} {c.model_name} {c.width} {c.height} {c.params_to_string()}"
                for c in r.cameras.values()) + "\n", encoding="utf8")
        else:
            shutil.copyfile(cam_txt, OUT / f"cameras_{name}.txt")

        if name == "crop_rig":
            integrity = check_masks(run_dir, run_dir / "images", run_dir / "masks",
                                    suffix_png=True, sample=args.mask_sample)
        else:
            integrity = check_masks(run_dir, Path(rep["image_dir"]), Path(rep["mask_dir"]),
                                    suffix_png=False, sample=args.mask_sample)
        if name == "crop_rig":
            dump_crop_rig(Path(rep["sparse_dir"]), OUT / "rig_crop_rig.json")
        traj, traj_jpg = trajectory(run_dir, name, model)
        arms[name] = {"report": rep, "run_dir": str(run_dir), "n_poses_exported": n_poses,
                      "trajectory": traj, "trajectory_jpg": traj_jpg,
                      "mask_integrity": integrity}

    scorecard = {
        "wp": "WP-O2", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arms": {k: {
            "argv": v["report"]["argv"],
            "seed": v["report"]["seed"],
            "run_dir": v["run_dir"],
            "sparse_dir": v["report"]["sparse_dir"],
            "equirect_dir": v["report"]["equirect_dir"],
            "colmap": v["report"]["colmap"],
            "options": v["report"]["options"],
            "matching": v["report"]["matching"],
            "timings_s": v["report"]["timings_s"],
            "metrics_equirect": v["report"]["metrics_equirect"],
            "metrics_crop": v["report"].get("metrics_crop"),
            "trajectory": v["trajectory"],
            "trajectory_in_run_report": v["report"]["trajectory"],
            "mask_integrity": v["mask_integrity"],
            "poses_csv": f"poses_{k}.csv",
            "trajectory_jpg": v["trajectory_jpg"],
            "ceres_warnings": logs.get(k),
            "sphere_coverage_frac": v["report"].get("sphere_coverage_frac"),
            "valid_frac_mean": v["report"].get("valid_frac_mean"),
        } for k, v in arms.items()},
    }
    (OUT / "ab_scorecard.json").write_text(json.dumps(scorecard, indent=1), encoding="utf8")

    names = list(arms)
    def row(label, get):
        return "| " + label + " | " + " | ".join(str(get(arms[n]["report"])) for n in names) + " |"

    def trow(label, get):
        return "| " + label + " | " + " | ".join(str(get(arms[n]["trajectory"])) for n in names) + " |"
    lines = [
        "| metric | " + " | ".join(label_of(n) for n in names) + " |",
        "|---|" + "---|" * len(names),
        row("panos registered", lambda r: f"{r['metrics_equirect']['num_registered_panos']}/"
                                          f"{r['metrics_equirect']['num_input_panos']}"),
        row("registered %", lambda r: f"{r['metrics_equirect']['registered_pct']:.2f}"),
        row("mean reproj err (ERP px)",
            lambda r: f"{r['metrics_equirect']['mean_reproj_error_px']:.4f}"),
        row("mean reproj err (native px)",
            lambda r: f"{(r['metrics_crop'] or r['metrics_equirect'])['mean_reproj_error_px']:.4f}"),
        row("3D points", lambda r: r["metrics_equirect"]["num_points3D"]),
        row("observations", lambda r: r["metrics_equirect"]["num_observations"]),
        row("mean track length", lambda r: f"{r['metrics_equirect']['mean_track_length']:.3f}"),
        row("obs / registered pano",
            lambda r: f"{r['metrics_equirect']['mean_obs_per_reg_image']:.0f}"),
        row("num reconstructions", lambda r: r["num_reconstructions"]),
        row("matcher", lambda r: r["matching"]["matcher"]),
        row("images fed to COLMAP",
            lambda r: r.get("num_crops") or r["metrics_equirect"]["num_input_panos"]),
        row("wall clock total (s)", lambda r: f"{r['timings_s']['total']:.0f}"),
        row("  render (s)", lambda r: f"{r['timings_s']['render']:.0f}"),
        row("  extract (s)", lambda r: f"{r['timings_s']['extract']:.0f}"),
        row("  match (s)", lambda r: f"{r['timings_s']['match']:.0f}"),
        row("  mapping (s)", lambda r: f"{r['timings_s']['mapping']:.0f}"),
        trow("floor-plan extent (up removed)",
             lambda t: " x ".join(f"{v:.2f}" for v in t["floorplan_extent_sorted"])),
        trow("floor-plan aspect (short/long)", lambda t: f"{t['floorplan_aspect']:.3f}"),
        trow("height extent / height std",
             lambda t: f"{t['height_extent']:.2f} / {t['height_std']:.3f}"),
        trow("height / floor-plan extent",
             lambda t: f"{t['height_over_floorplan_extent']:.3f}"),
        trow("PCA extent (sorted)",
             lambda t: " x ".join(f"{v:.2f}" for v in t["pca_extent_sorted"])),
        trow("median step", lambda t: f"{t['step_median']:.4f}"),
        trow("step p90/median", lambda t: f"{t['step_p90_over_median']:.2f}"),
        trow("step max/median", lambda t: f"{t['step_max_over_median']:.2f}"),
        trow("steps > 5x median", lambda t: t["num_steps_gt_5x_median"]),
        trow("path / end-to-end displacement",
             lambda t: f"{t['path_over_displacement']:.2f}"),
        trow("long axis / median step", lambda t: f"{t['extent_over_median_step']:.1f}"),
        trow("best-fit-plane residual rms / max",
             lambda t: f"{t['plane_residual_rms']:.3f} / {t['plane_residual_max']:.3f}"),
        trow("pano tilt vs mean up (med/p90/max deg)",
             lambda t: f"{t['upright']['tilt_deg_median']:.1f} / "
                       f"{t['upright']['tilt_deg_p90']:.1f} / {t['upright']['tilt_deg_max']:.1f}"),
    ]
    md = ["# WP-O2 A/B — crop-rig vs spherical", "", *lines, "", "## argv", ""]
    for n in names:
        md += [f"**{n}** (seed {arms[n]['report']['seed']}):", "",
               "```", " ".join(arms[n]["report"]["argv"]), "```", ""]
    if logs:
        md += ["## ceres / CHOLMOD warnings during BA "
               "(PLAN_360 §2 flagged these on the ERP probe - watched at scale)", ""]
        for n in names:
            lg = logs.get(n)
            if lg:
                md.append(f"- **{n}**: {lg['ceres_cholmod_not_positive_definite']} "
                          f"\"Linear solver failure ... CHOLMOD warning: Matrix not positive "
                          f"definite\" lines in `{lg['log']}`")
        md.append("")
    md += ["## mask integrity (independent re-check of the COLMAP databases)", ""]
    for n in names:
        mi = arms[n]["mask_integrity"]
        md.append(f"- **{n}**: {mi.get('keypoints_checked')} keypoints across "
                  f"{mi.get('images_checked')} sampled images, "
                  f"**{mi.get('keypoints_on_masked_out_pixels')}** on masked-out pixels")
    (OUT / "AB_TABLE.md").write_text("\n".join(md) + "\n", encoding="utf8")
    print("\n".join(lines))
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(REPO))
    raise SystemExit(main())
