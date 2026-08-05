"""ERP-direct SfM probe: pycolmap with EQUIRECTANGULAR camera model on 40 stitched frames."""
import pathlib
import shutil

import pycolmap

root = pathlib.Path(__file__).parent
db = root / "db.db"
out = root / "sparse"
if db.exists():
    db.unlink()
if out.exists():
    shutil.rmtree(out)
out.mkdir()

reader = pycolmap.ImageReaderOptions()
reader.camera_model = "EQUIRECTANGULAR"
print("extracting features...")
pycolmap.extract_features(
    db, root / "images",
    camera_mode=pycolmap.CameraMode.SINGLE,
    reader_options=reader,
)
print("matching (exhaustive, 40 imgs)...")
pycolmap.match_exhaustive(db)

d = pycolmap.Database(db)
print(f"DB: {d.num_images} images, {d.num_keypoints} keypoints, "
      f"{d.num_matched_image_pairs} matched pairs, "
      f"{d.num_verified_image_pairs} verified pairs")
d.close()

print("incremental mapping...")
recs = pycolmap.incremental_mapping(db, root / "images", out)
for i, r in recs.items():
    print(f"model {i}: {r.num_reg_images()}/40 registered, "
          f"{r.num_points3D()} points, "
          f"mean reproj err {r.compute_mean_reprojection_error():.2f}px, "
          f"mean track {r.compute_mean_track_length():.1f}")
