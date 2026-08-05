"""Continue ERP probe: DB stats + incremental mapping (db.db already built)."""
import pathlib
import shutil

import pycolmap

root = pathlib.Path(__file__).parent
db = root / "db.db"
out = root / "sparse"
if out.exists():
    shutil.rmtree(out)
out.mkdir()

d = pycolmap.Database.open(db) if hasattr(pycolmap.Database, "open") else None
if d is not None:
    print(f"DB: {d.num_images} images, {d.num_keypoints} keypoints, "
          f"{d.num_matched_image_pairs} matched pairs, "
          f"{d.num_verified_image_pairs} verified pairs")
    cams = d.read_all_cameras()
    print("cameras:", [(c.camera_id, c.model.name, c.width, c.height) for c in cams])
    d.close()

print("incremental mapping...")
recs = pycolmap.incremental_mapping(db, root / "images", out)
for i, r in recs.items():
    print(f"model {i}: {r.num_reg_images()}/40 registered, "
          f"{r.num_points3D()} points, "
          f"mean reproj err {r.compute_mean_reprojection_error():.2f}px, "
          f"mean track {r.compute_mean_track_length():.1f}")
