# COLMAP panorama SfM — upstream reference snapshot (fetched 2026-08-04, WP-O2)

Source: <https://github.com/colmap/colmap>, BSD-3-Clause. Reference only — nothing here is
imported by apex-sfm; `core/sfm_360.py` is our adaptation (see its header for the diff list).

| file | upstream path | ref |
|---|---|---|
| `panorama_sfm_v4.1.0.py` | `python/examples/panorama_sfm.py` | tag **`4.1.0`** — the self-contained version that matches our installed `pycolmap 4.1.0` (`COLMAP 4.1.0 Commit fa8e3b3 on 2026-06-26 without CUDA`) |
| `panorama_main.py` | `python/pycolmap/panorama.py` | `main` @ `319c6e6` (2026-08-01) — same logic split into a library; **not present at tag 4.1.0 and not in our wheel** |
| `panorama_sfm_main.py` | `python/examples/panorama_sfm.py` | `main` @ `319c6e6` — thin CLI over the library above |
| `image_reader_v4.1.0.cc` | `src/colmap/controllers/image_reader.cc` | tag `4.1.0` — mask file resolution (`<mask_dir>/<image_name>.png`, falling back to `<mask_dir>/<stem>.png`) |
| `pairing_v4.1.0.cc` | `src/colmap/controllers/pairing.cc` | tag `4.1.0` — `SequentialPairGenerator` (`expand_rig_images`, quadratic overlap, loop detection) |

Facts read out of these files that `core/sfm_360.py` relies on:

- **Mask convention** (`image_reader.cc:148-171` + `feature_extraction.cc:60-83` `MaskFeatures`):
  a keypoint is KEPT iff the mask pixel's red channel is non-zero, i.e. **0 = drop, 255 = keep**
  — identical to the WP-O1 mask convention, no inversion needed. The mask is looked up as
  `<mask_path>/<image_name>.png` first and `<mask_path>/<image_name_without_extension>.png`
  second, so WP-O1's `masks/pano_00000.png` pairs with `images/pano_00000.jpg` as-is.
  Masks are applied AFTER keypoints are rescaled back to the full camera resolution, so the
  mask must have the camera's dimensions (ours do).
- **Sequential pairing is rig-aware** (`pairing.cc:422-540`): with `expand_rig_images=True`
  (the default) every sequential neighbour pair is expanded to all images of the neighbour's
  frame, so crop *k* of pano *n* is matched against all 12 crops of pano *n±2^i* — required
  here because the operator turns while walking, so the same scene point lands in different
  crop indices between neighbouring panos.
- **`vocab_tree_path` defaults to an empty path** (pybind shows it as `WindowsPath('.')`),
  which makes COLMAP resolve a built-in vocab-tree URI and download+cache it
  (`pairing.cc:372-375`).
- The 4.1.0 example enables `loop_detection=True` for the sequential matcher; `main` dropped
  that back to the COLMAP default (`False`). We expose it as a flag.
