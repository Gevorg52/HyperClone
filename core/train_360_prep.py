"""WP-O3 — package a WP-O2 crop-rig reconstruction as a gsplat-trainer dataset.

DEFAULT-OFF: `main()` returns 2 and writes nothing unless `--capture-type 360` is passed.
Nothing in the production pipeline imports this module.

WP-O2 (`core/sfm_360.py`, ARM A) leaves its 12-crops-per-panorama reconstruction in the
shape COLMAP's panorama pipeline needs:

    <sfm>/images/pano_cameraN/pano_XXXXX.jpg      (N = 0..11, one dir per virtual camera)
    <sfm>/masks/pano_cameraN/pano_XXXXX.jpg.png   (255 = VALID, 0 = EXCLUDED)
    <sfm>/sparse/0/{cameras,images,points3D,rigs,frames}.bin

`core/training/gsplat_trainer.py::load_colmap_data` wants a different shape:

    <out>/images/<name>            <- os.path.join(data_dir, "images", colmap_image_name)
    <out>/masks/<stem>.png         <- os.path.splitext(colmap_image_name)[0] + ".png"
    <out>/sparse/0/*.bin

Two mismatches, both handled here:

1. **mask filename.** WP-O2 writes `<image_name>.png` (COLMAP's own extension-fallback
   convention, `image_reader.cc:148-171`); the trainer wants `<stem>.png`. Renamed.
2. **per-camera subdirectories.** `os.path.join` would in fact resolve
   `images/pano_camera0/pano_00000.jpg` correctly, but every downstream tool that keys on
   `os.path.basename` (prior caches, render/QA scripts) would then collide 12 ways on one
   stem, and the trainer's name-sort would order the views camera-major instead of
   pano-major. So the names are FLATTENED to `pano_XXXXX_camNN.jpg` in both the file layout
   and the COLMAP model (`pycolmap.Image.name` is writable; the model is re-written).
   Flat names sort pano-major, which is also what the held-out split wants.

Held-out split: every `--holdout_every`-th PANORAMA by rank, **all 12 of its crops
together**. The 12 crops of one panorama share an optical centre, so splitting a panorama
would leak the held-out view straight into training. The trainer's own `--test_every` rule
cannot express that (it is periodic over images, and 12 crops/pano vs every-8th-image never
aligns), which is why the packaged dataset ships explicit name lists consumed by the
trainer's `--holdout_list` flag.

Only panoramas whose **all** 12 crops are registered are eligible — the strict
`panos_fully_registered` accounting from `core.sfm_360.reconstruction_metrics`.

Usage:
    python -m core.train_360_prep --capture-type 360 \
        --sfm _dataset/360/sfm/wpo2_crop_rig \
        --out_dir _dataset/360/train/wpo3_flowershop_077 --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------------------
# naming
# ---------------------------------------------------------------------------

def split_rig_name(name: str) -> tuple:
    """'pano_camera7/pano_00042.jpg' -> (7, 'pano_00042.jpg'). Raises on anything else."""
    head, sep, tail = name.replace("\\", "/").partition("/")
    if not sep or not head.startswith("pano_camera"):
        raise ValueError(f"not a WP-O2 crop image name: {name!r}")
    try:
        cam_idx = int(head[len("pano_camera"):])
    except ValueError:
        raise ValueError(f"cannot parse camera index from {name!r}") from None
    return cam_idx, tail


def flat_name(cam_idx: int, pano_name: str) -> str:
    """'pano_00042.jpg' + cam 7 -> 'pano_00042_cam07.jpg' (sorts pano-major)."""
    stem = Path(pano_name).stem
    return f"{stem}_cam{cam_idx:02d}{Path(pano_name).suffix}"


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

def pano_split(pano_names: list, holdout_every: int, holdout_offset: int = 0) -> tuple:
    """(train_panos, test_panos): every `holdout_every`-th panorama by rank is held out.

    Offset 0 matches the trainer's own `index % test_every == 0` convention.
    holdout_every <= 0 disables the split (everything trains, nothing is held out).
    """
    panos = sorted(pano_names)
    if holdout_every <= 0:
        return panos, []
    test = [p for i, p in enumerate(panos) if (i - holdout_offset) % holdout_every == 0
            and i >= holdout_offset]
    test_set = set(test)
    return [p for p in panos if p not in test_set], test


# ---------------------------------------------------------------------------
# packaging
# ---------------------------------------------------------------------------

def package(args) -> dict:
    import pycolmap

    t0 = time.time()
    sfm = Path(args.sfm)
    sparse_in = Path(args.sparse) if args.sparse else sfm / "sparse" / "0"
    images_in = Path(args.images) if args.images else sfm / "images"
    masks_in = None if args.no_masks else (Path(args.masks) if args.masks
                                           else sfm / "masks")
    for p in (sparse_in, images_in):
        if not p.is_dir():
            raise FileNotFoundError(f"{p} does not exist")
    if masks_in is not None and not masks_in.is_dir():
        raise FileNotFoundError(f"mask dir {masks_in} does not exist "
                                f"(pass --no_masks to package without them)")

    out_dir = Path(args.out_dir)
    existing = [r for r in ("sparse", "images", "masks", "prep_report.json")
                if (out_dir / r).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"out_dir contains prior output ({', '.join(existing)}); "
                              f"pass --overwrite")
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    rec = pycolmap.Reconstruction(str(sparse_in))
    n_images_in = int(rec.num_images())
    reg_ids = set(rec.reg_image_ids())
    print(f"[prep] {sparse_in}: {n_images_in} images, {len(reg_ids)} registered, "
          f"{rec.num_points3D()} points, {len(rec.cameras)} cameras", flush=True)

    # pycolmap 4.1.0's Reconstruction has no image-removal API, and an un-posed image
    # would blow up the trainer's cam_from_world(). COLMAP only writes registered images
    # anyway, so this is a hard error rather than a silent surgery.
    if len(reg_ids) != n_images_in:
        raise RuntimeError(
            f"{n_images_in - len(reg_ids)} of {n_images_in} images in {sparse_in} have no "
            f"pose. This packager cannot remove images from a pycolmap Reconstruction; "
            f"re-run the SfM or hand it a model containing only registered images.")

    # -- group crops by panorama -------------------------------------------
    by_pano: dict = defaultdict(dict)          # pano_name -> {cam_idx: image_id}
    for image_id in sorted(rec.images):
        cam_idx, pano = split_rig_name(rec.images[image_id].name)
        by_pano[pano][cam_idx] = image_id

    crops_per_pano = int(args.crops_per_pano) if args.crops_per_pano else max(
        (len(v) for v in by_pano.values()), default=0)
    full = sorted(p for p, v in by_pano.items() if len(v) == crops_per_pano)
    partial = sorted(p for p, v in by_pano.items() if len(v) != crops_per_pano)
    if not full:
        raise RuntimeError("no panorama has all its crops registered — nothing to package")
    if partial:
        # Their crops still TRAIN (they are posed and real); they are simply never
        # eligible for the held-out set, because a partial panorama cannot be withheld
        # without leaving some of its own crops in training.
        print(f"[prep] {len(partial)} partially-registered panorama(s) are train-only: "
              f"{partial[:5]}{' ...' if len(partial) > 5 else ''}", flush=True)

    holdout_train, test_panos = pano_split(full, args.holdout_every, args.holdout_offset)
    train_panos = sorted(set(holdout_train) | set(partial))
    print(f"[prep] {len(by_pano)} panos ({len(full)} full x {crops_per_pano} crops) · "
          f"train {len(train_panos)} panos / test {len(test_panos)} panos "
          f"(every {args.holdout_every}th)", flush=True)

    # -- materialise images + masks under flat names ------------------------
    out_images = out_dir / "images"
    out_masks = out_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    if masks_in is not None:
        out_masks.mkdir(parents=True, exist_ok=True)

    rename: dict = {}                       # image_id -> flat name
    per_pano_flat: dict = defaultdict(list)
    n_img = n_mask = 0
    for pano in sorted(by_pano):
        for cam_idx, image_id in sorted(by_pano[pano].items()):
            src_name = rec.images[image_id].name
            dst_name = flat_name(cam_idx, pano)
            rename[image_id] = dst_name
            per_pano_flat[pano].append(dst_name)

            src = images_in / src_name
            if not src.is_file():
                raise FileNotFoundError(f"missing crop image {src}")
            shutil.copyfile(src, out_images / dst_name)
            n_img += 1

            if masks_in is not None:
                # WP-O2 writes '<image_name>.png'; tolerate '<stem>.png' too.
                cand = [masks_in / f"{src_name}.png",
                        masks_in / f"{Path(src_name).with_suffix('.png')}"]
                mp = next((c for c in cand if c.is_file()), None)
                if mp is None:
                    raise FileNotFoundError(
                        f"missing mask for {src_name} (looked for {[str(c) for c in cand]})")
                shutil.copyfile(mp, out_masks / f"{Path(dst_name).stem}.png")
                n_mask += 1
        if args.progress and len(per_pano_flat) % args.progress == 0:
            print(f"[prep] {len(per_pano_flat)}/{len(full)} panos copied", flush=True)

    # -- rewrite the COLMAP model ------------------------------------------
    for image_id, dst_name in rename.items():
        rec.images[image_id].name = dst_name

    out_sparse = out_dir / "sparse" / "0"
    out_sparse.mkdir(parents=True, exist_ok=True)
    rec.write(str(out_sparse))

    # Re-open what we wrote and verify the trainer's own view of it.
    check = pycolmap.Reconstruction(str(out_sparse))
    check_names = sorted(im.name for im in check.images.values())
    missing_files = [n for n in check_names if not (out_images / n).is_file()]
    if missing_files:
        raise RuntimeError(f"{len(missing_files)} model images have no file, "
                           f"e.g. {missing_files[:3]}")
    if masks_in is not None:
        missing_masks = [n for n in check_names
                         if not (out_masks / f"{Path(n).stem}.png").is_file()]
        if missing_masks:
            raise RuntimeError(f"{len(missing_masks)} model images have no mask, "
                               f"e.g. {missing_masks[:3]}")

    # -- split lists --------------------------------------------------------
    test_names = sorted(n for p in test_panos for n in per_pano_flat[p])
    train_names = sorted(n for p in train_panos for n in per_pano_flat[p])
    (out_dir / "holdout_test.txt").write_text(
        "# WP-O3 held-out crops: every %dth panorama, all %d crops together.\n"
        "# Consumed by gsplat_trainer.py --holdout_list\n%s\n"
        % (args.holdout_every, crops_per_pano, "\n".join(test_names)), encoding="utf8")
    (out_dir / "holdout_train.txt").write_text("\n".join(train_names) + "\n", encoding="utf8")
    (out_dir / "split_panos.json").write_text(json.dumps(
        {"crops_per_pano": crops_per_pano,
         "holdout_every": int(args.holdout_every),
         "holdout_offset": int(args.holdout_offset),
         "train_panos": train_panos, "test_panos": test_panos,
         "pano_to_crops": {p: sorted(v) for p, v in per_pano_flat.items()}},
        indent=1), encoding="utf8")

    cams = {int(cid): {"model": c.model.name, "width": int(c.width),
                       "height": int(c.height), "params": [float(x) for x in c.params]}
            for cid, c in sorted(check.cameras.items())}
    report = {
        "wp": "WP-O3",
        "step": "dataset packaging",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "argv": [sys.executable, "-m", "core.train_360_prep"] + list(sys.argv[1:]),
        "source": {"sfm_dir": str(sfm), "sparse": str(sparse_in),
                   "images": str(images_in),
                   "masks": str(masks_in) if masks_in else None,
                   "sfm_report": str(sfm / "sfm_report.json")
                   if (sfm / "sfm_report.json").is_file() else None},
        "out_dir": str(out_dir),
        "transformations": [
            "image names flattened 'pano_cameraN/pano_XXXXX.jpg' -> "
            "'pano_XXXXX_camNN.jpg' in BOTH the file layout and the COLMAP model "
            "(pycolmap.Image.name rewritten, model re-written to sparse/0)",
            "mask files renamed '<image_name>.png' (COLMAP convention) -> "
            "'<stem>.png' (gsplat_trainer convention) and flattened the same way",
            "images and masks COPIED (source left untouched); pixels unmodified, "
            "no resize, no re-encode",
            "panoramas whose crops are not ALL registered still train, but are never "
            "eligible for the held-out set",
            "camera intrinsics, poses, rig, frames and points3D carried over verbatim",
        ],
        "counts": {
            "images_in_source_model": n_images_in,
            "registered_in_source_model": len(reg_ids),
            "panos_seen": len(by_pano),
            "crops_per_pano": crops_per_pano,
            "panos_fully_registered": len(full),
            "panos_partial_train_only": len(partial),
            "panos_train": len(train_panos),
            "panos_test": len(test_panos),
            "images_written": n_img,
            "masks_written": n_mask,
            "images_train": len(train_names),
            "images_test": len(test_names),
            "model_images_after_write": int(check.num_images()),
            "model_points3D": int(check.num_points3D()),
            "model_mean_reproj_error_px": round(
                float(check.compute_mean_reprojection_error()), 6),
        },
        "cameras": cams,
        "split": {"holdout_every": int(args.holdout_every),
                  "holdout_offset": int(args.holdout_offset),
                  "test_list": "holdout_test.txt", "train_list": "holdout_train.txt",
                  "rule": "every Nth panorama by sorted-name rank, all crops together"},
        "partial_panos_train_only": partial,
        "seconds": round(time.time() - t0, 2),
    }
    (out_dir / "prep_report.json").write_text(json.dumps(report, indent=1), encoding="utf8")
    print(f"\n[done] {n_img} images + {n_mask} masks -> {out_dir}\n"
          f"       train {len(train_names)} crops / test {len(test_names)} crops "
          f"({len(test_panos)} panos)\n"
          f"       report: {out_dir / 'prep_report.json'}", flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m core.train_360_prep",
        description="WP-O3: package a WP-O2 crop-rig reconstruction as a gsplat-trainer "
                    "COLMAP dataset (flat names, trainer mask convention, pano-grouped "
                    "held-out split). DEFAULT-OFF: does nothing without --capture-type 360.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--capture-type", dest="capture_type", default="off",
                   choices=["off", "360"],
                   help="OFF by default. '360' opts into the 360 training-prep lane.")
    p.add_argument("--sfm", help="WP-O2 ARM A out_dir (expects sparse/0, images/, masks/)")
    p.add_argument("--sparse", help="override <sfm>/sparse/0")
    p.add_argument("--images", help="override <sfm>/images")
    p.add_argument("--masks", help="override <sfm>/masks")
    p.add_argument("--no_masks", action="store_true",
                   help="package without masks (the no-priors A/B path)")
    p.add_argument("--out_dir", help="dataset dir to create")
    p.add_argument("--overwrite", action="store_true",
                   help="delete out_dir first; REQUIRED to re-use a non-empty dir")

    g = p.add_argument_group("held-out split")
    g.add_argument("--holdout_every", type=int, default=8,
                   help="hold out every Nth PANORAMA (all its crops together); 0 = none")
    g.add_argument("--holdout_offset", type=int, default=0,
                   help="rank of the first held-out panorama")
    g.add_argument("--crops_per_pano", type=int, default=0,
                   help="expected crops per panorama (0 = infer from the model)")
    p.add_argument("--progress", type=int, default=25, help="print every N panos")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.capture_type == "off":
        print("360 training prep is OFF (default). Pass --capture-type 360 to enable it.")
        return 2
    if not args.sfm and not (args.sparse and args.images):
        print("--sfm (or --sparse and --images) is required with --capture-type 360")
        return 2
    if not args.out_dir:
        print("--out_dir is required with --capture-type 360")
        return 2
    try:
        package(args)
    except FileExistsError as e:
        print(f"ERROR: {e}")
        return 2
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
