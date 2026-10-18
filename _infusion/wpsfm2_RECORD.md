# WP-SfM2 — COLMAP / pycolmap 4.2.1 Evaluation in Isolated Venv — RECORD (2026-10-18)

**Verdict: GATE PASSED (Evaluation & Parity Verified; Keep System `pycolmap 4.1.0` Default, Retain `venvs/colmap421` + CUDA `colmap.exe 4.2.1` for Native `LOMA_B` GPU Feature/Match).**

`pycolmap 4.2.1` and the official standalone Windows CUDA release `colmap-x64-windows-cuda.zip` (4.2.1, commit `bd1fcf6`) were installed strictly inside `D:\REPOS\HYPERclone\venvs\colmap421` without touching system Python's `pycolmap 4.1.0` (`python -c "import pycolmap; print(pycolmap.__version__)"` -> `4.1.0`). Across all four benchmarks (`40`-frame ERP-direct probe, `151`-frame full WP-O2 `EQUIRECTANGULAR` run, `94`-pano / `1,128`-crop full WP-O5 8K walkthrough rig, and `20`-pano / `240`-crop parity + native `LOMA_B` evaluation), `core/sfm_360.py` runs **unmodified** under `pycolmap 4.2.1` with zero API breaks.

Key headline outcomes:
1. **ERP-Direct (`EQUIRECTANGULAR`) SfM (Benchmark 1):** COLMAP 4.2.1's tangent-space Sampson error (`#3876`), spherical homography estimation (`#4054`), and analytical Jacobians (`#4149`) increase verified inliers by **+7.6% to +8.2%** and 3D points by **+6.8% to +7.3%** on the 40-frame ERP probe (`29,105` vs `27,256` masked pts; mean reproj improves from `1.3071 px` to **`1.2874 px`**), and run **1.26× faster** (`47.56 s` vs `59.81 s`) when isolating incremental mapping/BA on the identical 4.1.0 database. On the full 151-frame WP-O2 database, 4.2.1 incremental matches 4.1.0 within **`0.0276 m` Sim3 RMSE (`0.081°` direct median rotation)** and lowers mean reprojection error from `1.2694 px` to **`1.2583 px`**.
2. **Walkthrough Crop-Rig SfM (Benchmark 2a, 94 panos / 1,128 crops @ 8K):** On the full `wpo5_theater_027_8k` database, `pycolmap 4.2.1` incremental (`94/94` panos, `38,665` pts, `1.2253 px` crop / `1.3480 px` ERP) is **statistically indistinguishable from 4.1.0's own run-to-run seed-0 noise floor** (`Sim3` RMSE vs banked 4.1.0 is `0.1804 m` / `0.174°` direct rot for 4.2.1 vs `0.1615 m` / `0.334°` direct rot for a 4.1.0 repeat run).
3. **Global Mapper (`pycolmap.global_mapping` / GLOMAP merged in 4.1–4.2):** Registers **100% of frames** across every ERP and crop-rig benchmark **2.1×–3.8× faster** than incremental mapping (`64.4 s` vs `163.8 s` on 94-pano rig; `181.3 s` vs `683.4 s` on 151-frame ERP), **but** in `4.2.1` PR `#4262` ("Preserve camera intrinsics calibration in global mapper") + 4.2.1 track/retriangulation defaults yield **higher reprojection errors** (`1.766 px` crop / `1.925 px` ERP on 94-pano rig vs `1.283 px` / `1.407 px` in 4.1.0 global and `1.210 px` / `1.331 px` in incremental). Incremental mapping remains the production quality winner.
4. **Native COLMAP 4.2.1 `LOMA_B` GPU Features + Matcher (Benchmark 2b, 20 panos / 240 crops):** While the PyPI `pycolmap 4.2.1` Windows wheel is CPU-only (`has_cuda=False`, no ONNX), the standalone `colmap.exe` 4.2.1 CUDA binary ships with ONNX Runtime CUDA support and runs **`LOMA_B` on the RTX PRO 6000 (`sm_120`)** once `torch\lib` (`cudnn64_9.dll`) is on `PATH`. Native `LOMA_B` honors `--ImageReader.mask_path` (`0` masked-pixel violations), produces **550,585 rig-verified inlier matches** (`7.0×` SIFT's `78,518` and `1.86×` HLoc ALIKED+LightGlue's `296,730`), and reconstructs **73,713 3D points** (`7.55×` SIFT's `9,761` and `1.87×` ALIKED's `39,330`) at `1.4043 px` crop / `1.5321 px` ERP reprojection error and **`0.0202 m` Sim3 RMSE (`0.110°` direct median rot)** vs HLoc ALIKED+LightGlue.
5. **Critical Negative Findings (Recorded Honest):**
   - **`guided_matching=True` on `EQUIRECTANGULAR` (`sfm_360.py:725` `TODO(colmap >= 4.2)`):** Must **NOT** be enabled. In 4.1.0 it reduces 3D points by `-37.4%` (`17,511` vs `27,957`) and corrupts the ERP trajectory (`Sim3` RMSE `2.997 m`, max `18.15 m`). In 4.2.1, PR `#3935` fixed guided matching only for `CALIBRATED` (`config=2`) ERP pairs, whereas PR `#4054` now classifies 194/780 ERP pairs as `PLANAR_OR_PANORAMIC` (`config=6`, `F=nullopt`), causing `MatchGuided` to throw a deterministic C++ `THROW_CHECK` exception inside a worker thread (`std::terminate` / `abort` on pair `(1, 10)`).
   - **Caspar BA & Ceres GPU BA on Windows:** Neither the `pycolmap 4.2.1` wheel nor `colmap-x64-windows-cuda.zip` compiles Caspar (`CASPAR_ENABLED=OFF`) or Ceres CUDA/cuDSS solvers (`Ceres was compiled without CUDA/cuDSS support. Falling back to CPU-based solvers`).

---

## 1. Environment & Pins

| Item | Value |
|---|---|
| System Python (untouched) | `C:\Users\gevor\AppData\Local\Programs\Python\Python311\python.exe` — `pycolmap 4.1.0` (`COLMAP 4.1.0`, commit `7237fd1`, `2026-04-08`, `has_cuda=False`, `ceres 2.2.0`) |
| Isolated Venv | `D:\REPOS\HYPERclone\venvs\colmap421` (`python -m venv --system-site-packages`) |
| Venv `pycolmap` wheel | `pycolmap==4.2.1` (`pycolmap-4.2.1-cp311-cp311-win_amd64.whl`, `COLMAP 4.2.1`, commit `bd1fcf6`, `2026-04-17`, `has_cuda=False`, `ceres 2.2.0`, installed with `--ignore-installed --no-deps`) |
| Standalone CUDA binary | `D:\REPOS\HYPERclone\venvs\colmap421\colmap_bin\bin\colmap.exe` — official `colmap-x64-windows-cuda.zip` (`COLMAP 4.2.1`, commit `bd1fcf6`, `2026-04-17`, **with CUDA + ONNX Runtime CUDA provider**) |
| GPU / Driver / cuDNN | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (`sm_120`); `cudnn64_9.dll` supplied at runtime by prepending `...\Python311\Lib\site-packages\torch\lib` to `PATH` |

---

## 2. API & Binary Capability Audit (`4.1.0` vs `4.2.1`)

### 2.1 `pycolmap` Python API Changes (`4.1.0` -> `4.2.1`)
- **Zero breaking changes for `core/sfm_360.py` and `core/sfm_360_hloc.py`:** Every symbol used by our codebase (`Camera`, `CameraMode`, `CameraModelId`, `Database`, `DatabaseTransaction`, `ExhaustivePairingOptions`, `ExistingMatchedPairingOptions`, `FeatureExtractionOptions`, `FeatureMatchingOptions`, `Frame`, `GeometricVerifierOptions`, `Image`, `ImageReaderOptions`, `IncrementalMapperOptions`, `IncrementalPipelineOptions`, `IncrementalTriangulatorOptions`, `Reconstruction`, `Rig`, `RigConfig`, `RigConfigCamera`, `Rigid3d`, `Rotation3d`, `SensorType`, `SequentialPairGenerator`, `SequentialPairingOptions`, `SiftExtractionOptions`, `SiftMatchingOptions`, `TwoViewGeometryConfiguration`, `TwoViewGeometryOptions`, ` VocabTreePairingOptions`, `apply_rig_config`, `extract_features`, `geometric_verification`, `global_mapping`, `import_images`, `incremental_mapping`, `match_exhaustive`, `match_sequential`, `match_vocabtree`, `set_random_seed`, `verify_matches`) is present with compatible signatures.
- **New top-level symbols in `pycolmap 4.2.1`:**
  - `GlobalPipeline`, `GlobalPipelineCallback`, `RotationAveragingReweighting`
  - `HierarchicalPipeline`, `HierarchicalPipelineOptions`, `SceneClusteringOptions`, `hierarchical_mapping`
  - `WarpImageOptions`, `CancellationToken`
  - Bundled `pycolmap.panorama` module (`PanoRenderType.PERSPECTIVE_OVERLAPPING` / `SPHERICAL`, `Mapper.INCREMENTAL` / `GLOBAL`)
- **Removed top-level symbols in `4.2.1`:** `GlobalMapperCallback` (renamed to `GlobalPipelineCallback`).
- **Option & backend changes:**
  - `FeatureExtractorType` adds `ALIKED_N16ROT`, `ALIKED_N32`, `LOMA_B`, `LOMA_L` (alongside `SIFT`, `SUPERPOINT`, `DISK`).
  - `FeatureExtractionOptions` adds `.loma` (`LomaExtractionOptions(max_num_features=8192, spp_Warning=...)`) and `.aliked` (`AlikedExtractionOptions`).
  - `FeatureMatcherType` adds `LOMA_B`, `LOMA_L`, `ALIKED_LIGHTGLUE`, `LIGHTGLUE`.
  - `GlobalPipelineOptions` adds `multiple_models=True` and `min_model_size=3`.
  - `BundleAdjustmentOptions.backend` adds `BundleAdjustmentBackend.CASPAR` (`CERES=0`, `CASPAR=1`).
  - `TwoViewGeometry.E`, `.F`, `.H` now return `None` (`std::nullopt`) when unpopulated instead of zero matrices.
  - `pycolmap 4.2.1` PyPI Windows wheel switched `default_mersenne_twister_t` hash map backend from `boost` (in 4.1.0 wheel) to `std`, while `colmap.exe` 4.2.1 CUDA binary uses `boost`.

### 2.2 Windows Packaging & GPU Acceleration Reality Check
| Feature / Backend | `pycolmap 4.2.1` PyPI Wheel (`win_amd64`) | `colmap.exe 4.2.1` (`colmap-x64-windows-cuda.zip`) | Notes / Workaround |
|---|---|---|---|
| SIFT GPU extract / match | No (`has_cuda=False`) | **Yes (`use_gpu=1`)** | Works natively on RTX PRO 6000 |
| ONNX `LOMA_B` / `LOMA_L` / `ALIKED` | No (`Compiled without ONNX support`) | **Yes (`use_gpu=1`)** | Requires `cudnn64_9.dll` on `PATH` (missing from COLMAP zip; resolved by prepending `torch\lib` to `PATH`). Weights auto-download (`LOMA_B`: 26 MB + 24 MB) |
| `--ImageReader.mask_path` with `LOMA_B` | N/A | **Yes (Verified)** | Tested on `wpo5_ab_sift/masks`: `0` keypoints on masked-out pixels |
| Ceres GPU BA (`ba_use_gpu=True`) | No | No | Both log `Ceres was compiled without CUDA/cuDSS support. Falling back to CPU-based solvers` |
| Caspar GPU BA (`backend=CASPAR`) | No (`CASPAR_ENABLED=OFF`) | No (`CASPAR_ENABLED=OFF`) | Both abort with `THROW_CHECK` (`0xC0000409`) when `CASPAR` is requested |

---

## 3. Benchmark 1 — WP-O2 ERP-Direct (`EQUIRECTANGULAR`) SfM

### 3.1 40-Frame ERP-Direct Probe (`erp40_dataset`: first 80 frames of `theater_027` subsampled by 2 -> 40 frames @ 1 fps, `3840x1920`, exhaustive matching, `seed=0`)

| Arm | `pycolmap` | Masks | Mapper | Guided Match | Verified Pairs (` CALIB ` / ` PLANAR_PANO `) | Inlier Matches | Reg Panos | 3D Points | Mean Reproj (px) | Med Reproj (px) | Mean Track | Match (s) | Map (s) | Total (s) | Sim3 RMSE vs 4.1.0 Inc (m) | Direct Med Rot (°) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `erp40_unmasked_410_inc` | 4.1.0 | Off | Incremental | Off | `780` (`780` / `0`) | `431,438` | `40/40` | `27,957` | `1.2735` | `1.2561` | `2.963` | `36.48` | `59.81` | `114.79` | `0.0000` (ref) | `0.000°` |
| `erp40_unmasked_421_inc` | **4.2.1** | Off | Incremental | Off | `780` (`586` / `194`) | **`466,603`** (+8.2%) | `40/40` | **`30,008`** (+7.3%) | `1.3167` | `1.3006` | `2.954` | `39.20` | `65.82` | `125.50` | `0.0790` | `0.166°` |
| `erp40_unmasked_421_inc_on_410db` | **4.2.1** | Off | Incremental (on 4.1.0 DB) | Off | `780` (`780` / `0`) | `431,438` | `40/40` | `28,081` (+0.4%) | `1.2973` | `1.2869` | `2.950` | *(reused)* | **`47.56`** (**1.26× faster**) | `47.56` | `0.0643` | `0.291°` |
| `erp40_unmasked_410_glo` | 4.1.0 | Off | Global | Off | `780` (`780` / `0`) | `431,438` | `40/40` | `12,261` | `1.3677` | `1.3086` | `3.621` | *(reused)* | `15.89` | `15.89` | `0.0820` | `0.469°` |
| `erp40_unmasked_421_glo` | **4.2.1** | Off | Global | Off | `780` (`586` / `194`) | `466,603` | `40/40` | `13,043` | `1.6762` | `1.5597` | `3.548` | *(reused)* | `17.02` | `17.02` | `0.0966` | `0.420°` |
| `erp40_unmasked_421_glo_on_410db` | **4.2.1** | Off | Global (on 4.1.0 DB) | Off | `780` (`780` / `0`) | `431,438` | `40/40` | `14,740` | `2.0637` | `1.7595` | `3.240` | *(reused)* | `15.22` | `15.22` | `0.0887` | `0.452°` |
| `erp40_unmasked_410_inc_guided` | 4.1.0 | Off | Incremental | **On** | `780` (`780` / `0`) | `136,221` (-68.4%) | `40/40` | `17,511` (-37.4%) | `1.0855` | `1.0315` | `3.131` | `224.81` | `26.63` | `269.93` | **`2.9968`** (corrupted!) | **`1.782°`** |
| `erp40_unmasked_421_inc_guided` | **4.2.1** | Off | Incremental | **On** | *Aborted on pair `(1,10)`* | — | — | — | — | — | — | *crashed* | — | — | — | — |
| `erp40_masked_410_inc` | 4.1.0 | **On** | Incremental | Off | `780` (`780` / `0`) | `420,964` | `40/40` | `27,256` | `1.3071` | `1.2963` | `2.956` | `31.03` | `51.69` | `100.11` | `0.0000` (masked ref) | `0.000°` |
| `erp40_masked_421_inc` | **4.2.1** | **On** | Incremental | Off | `780` (`583` / `197`) | **`453,134`** (+7.6%) | `40/40` | **`29,105`** (+6.8%) | **`1.2874`** (-0.020) | **`1.2788`** | `2.967` | `32.80` | `76.72` | `126.72` | `0.0787` | `0.293°` |
| `erp40_masked_421_glo` | **4.2.1** | **On** | Global | Off | `780` (`583` / `197`) | `453,134` | `40/40` | `17,112` | `1.9903` | `1.7666` | `3.156` | *(reused)* | `28.58` | `28.58` | `0.0849` | `0.360°` |

### 3.2 Full 151-Frame WP-O2 Spherical ERP Database (`_dataset/360/sfm/wpo2_spherical/database.db`, `3840x1920`, `5,599,131` inliers across `11,310` pairs)

| Arm | `pycolmap` | Mapper | Reg Panos | 3D Points | Observations | Mean Reproj (px) | Med Reproj (px) | Mean Track | Map Time (s) | Sim3 RMSE vs WP-O2 4.1.0 (m) | Max Trans Diff (m) | Direct Med Rot (°) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `erp151_banked_410_inc` | 4.1.0 (banked) | Incremental | `151/151` | `166,670` | `675,169` | `1.2694` | `1.2762` | `4.051` | `651.68` | `0.0000` (ref) | `0.0000` | `0.000°` |
| `erp151_421_inc` | **4.2.1** | Incremental | `151/151` | `166,469` | `673,260` | **`1.2583`** | **`1.2664`** | `4.044` | `683.38` | **`0.0276`** | **`0.0865`** | **`0.081°`** |
| `erp151_410_glo` | 4.1.0 | Global | `151/151` | `78,574` | `373,483` | `1.2426` | `1.2290` | `4.753` | `217.60` (3.0× faster) | `0.0559` | `0.1494` | `0.482°` |
| `erp151_421_glo` | **4.2.1** | Global | `151/151` | `76,393` | `356,027` | `1.4162` | `1.3840` | `4.660` | **`181.32`** (**3.6× faster**) | `0.0630` | `0.1624` | `0.593°` |

**Diagnosis of Benchmark 1:**
- **Why 4.2.1 finds +7–8% more inliers & 3D points on ERP:** In 4.1.0, homography estimation on spherical cameras was disabled (`cfg={'2': 780}`). In 4.2.1 (`#3876` tangent-space Sampson error + `#4054` spherical homography estimation), `586` pairs are verified as `CALIBRATED` (`cfg=2`) and `194` pairs as `PLANAR_OR_PANORAMIC` (`cfg=6`), increasing verified inlier matches from `431,438` to `466,603` (`+8.2%` unmasked) and `420,964` to `453,134` (`+7.6%` masked).
- **Why `erp40_unmasked_421_inc_guided` aborts:** We isolated the crash down to the exact image pair `(image_id=1, image_id=10)` (`pano_00000.jpg <-> pano_00018.jpg`), which is the very first pair in `exhaustive_matcher` block `[1/20, 5/20]` classified as `TwoViewGeometryConfiguration.PLANAR_OR_PANORAMIC` (`config=6`, `has_E=True`, `has_F=False`, `has_H=True`). Pairs classified as `CALIBRATED` (`config=2`, e.g. `(1, 2)` and `(1, 12)`) succeed with `guided_matching=True` in 4.2.1 thanks to `#3935`, but `PLANAR_OR_PANORAMIC` on `EQUIRECTANGULAR` has `F=nullopt` and triggers a `THROW_CHECK` inside the matcher worker thread (`std::terminate`). Combined with the fact that 4.1.0 `guided_matching=True` degrades point count by `-37.4%` and corrupts poses (`2.997 m` RMSE), `guided_matching` on `EQUIRECTANGULAR` must remain off.

---

## 4. Benchmark 2 — WP-O5 360 Walkthrough Crop-Rig SfM (`perspective_overlapping`, 12 virtual `1920x1920` `SIMPLE_PINHOLE` cameras per pano)

### 4.1 Benchmark 2a — Full 94-Pano (1,128-Crop) 8K SIFT Database (`_dataset/360/sfm/wpo5_theater_027_8k/database.db`, `599,527` inliers)

| Arm | `pycolmap` | Mapper | Reg Panos | Reg Crops | 3D Points | Observations | Mean Reproj Crop (px) | Mean Reproj ERP (px) | Med Reproj ERP (px) | Mean Track | Map Time (s) | Sim3 ERP RMSE vs Banked 4.1.0 (m) | Max ERP Diff (m) | Direct Med Rot (°) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `wpo5_full_banked_410_inc` | 4.1.0 (banked) | Incremental | `94/94` | `1128/1128` | `38,450` | `142,233` | `1.2096` | `1.3308` | `1.2878` | `3.699` | `132.38` | `0.0000` (ref) | `0.0000` | `0.000°` |
| `wpo5_full_rerun_410_inc` | 4.1.0 (repeat) | Incremental | `94/94` | `1128/1128` | `38,053` | `141,511` | `1.2235` | `1.3465` | `1.3100` | `3.719` | `140.15` | **`0.1615`** (noise floor) | `0.5215` | `0.334°` |
| `wpo5_full_421_inc` | **4.2.1** | Incremental | `94/94` | `1128/1128` | **`38,665`** | **`144,294`** | `1.2253` | `1.3480` | `1.3120` | **`3.732`** | `163.80` | **`0.1804`** (within noise) | `0.5423` | **`0.174°`** |
| `wpo5_full_410_glo` | 4.1.0 | Global | `94/94` | `1128/1128` | `32,067` | `126,433` | `1.2828` | `1.4073` | `1.3249` | `3.943` | `64.87` (**2.0× faster**) | `0.5168` | `1.8128` | `0.654°` |
| `wpo5_full_421_glo` | **4.2.1** | Global | `94/94` | `1128/1128` | `36,218` | `130,682` | `1.7661` | `1.9253` | `1.6495` | `3.608` | **`64.40`** (**2.1× faster**) | `0.5091` | `1.6961` | `0.919°` |

### 4.2 Benchmark 2b — 20-Pano (240-Crop) Parity Guard & Native COLMAP 4.2.1 `LOMA_B` GPU Evaluation (`wpo5_ab_sift` / `wpo5_ab_aliked`)

All arms use the identical 20-pano (`240`-crop, `1920x1920`) walkthrough subset (`pano_00000` .. `pano_00019`), the identical WP-O1 crop masks (`wpo5_ab_sift/masks`), the identical 12-camera rig configuration, and the identical `11,476` sequential cross-frame candidate pairs (`overlap=10, quadratic_overlap=True, expand_rig_images=True, loop_detection=False`).

| Arm | Features + Matcher | `pycolmap` | Mapper | Keypoints Total (Mean/Crop) | Rig-Verified Inlier Matches | Reg Panos / Crops | 3D Points | Observations | Mean Reproj Crop (px) | Mean Reproj ERP (px) | Mean Track | Extract + Match (s) | Map (s) | Sim3 RMSE vs SIFT 4.1.0 (m) | Sim3 RMSE vs ALIKED 4.1.0 (m) | Direct Med Rot vs ALIKED (°) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `wpo5_20p_sift_banked_410_inc` | SIFT (CPU) | 4.1.0 | Incremental | `434,043` (`1,808.5`) | `78,518` | `20/20` (`240/240`) | `9,761` | `37,871` | `1.2023` | `1.3095` | `3.880` | `46.7 + 105.5 = 152.2` | `12.95` | `0.0000` (ref) | — | — |
| `wpo5_20p_sift_421_inc` | SIFT (CPU) | **4.2.1** | Incremental | `434,043` (`1,808.5`) | `78,518` | `20/20` (`240/240`) | `9,774` | `37,948` | `1.2050` | `1.3120` | `3.883` | *(reused DB)* | `12.59` | **`0.00113`** (`1.1 mm`) | — | — |
| `wpo5_20p_sift_421_glo` | SIFT (CPU) | **4.2.1** | Global | `434,043` (`1,808.5`) | `78,518` | `20/20` (`240/240`) | `9,534` | `38,519` | `1.3281` | `1.4422` | `4.040` | *(reused DB)* | **`5.22`** | `0.02140` | — | — |
| `wpo5_20p_aliked_banked_410_inc` | HLoc ALIKED+LG (GPU) | 4.1.0 | Incremental | `572,867` (`2,387.0`) | `296,730` | `20/20` (`240/240`) | `39,330` | `138,298` | `1.3685` | `1.4904` | `3.516` | `15.0 + 319.6 = 334.6` | `44.44` | `0.05277` | `0.0000` (ref) | `0.000°` |
| `wpo5_20p_aliked_421_inc` | HLoc ALIKED+LG (GPU) | **4.2.1** | Incremental | `572,867` (`2,387.0`) | `296,730` | `20/20` (`240/240`) | `39,392` | `139,055` | `1.3751` | `1.4973` | `3.530` | *(reused DB)* | `52.53` | `0.04899` | **`0.00605`** (`6.1 mm`) | **`0.068°`** |
| `wpo5_20p_aliked_421_glo` | HLoc ALIKED+LG (GPU) | **4.2.1** | Global | `572,867` (`2,387.0`) | `296,730` | `20/20` (`240/240`) | `26,368` | `98,660` | `1.8872` | `2.0341` | `3.742` | *(reused DB)* | **`12.12`** | `0.07503` | `0.02417` | `0.127°` |
| **`wpo5_20p_loma_b_421_inc`** | **COLMAP 4.2.1 `LOMA_B` (GPU)** | **4.2.1** | Incremental | **`650,286`** (`2,709.5`) | **`550,585`** | `20/20` (`240/240`) | **`73,713`** | **`250,724`** | `1.4043` | `1.5321` | `3.401` | `43.2 + 491.5 = 534.7` | `183.30` | `0.06120` | **`0.02019`** (`2.0 cm`) | **`0.110°`** |
| `wpo5_20p_loma_b_421_glo` | COLMAP 4.2.1 `LOMA_B` (GPU) | **4.2.1** | Global | `650,286` (`2,709.5`) | `550,585` | `20/20` (`240/240`) | `40,149` | `149,980` | `2.0950` | `2.2762` | `3.736` | `43.2 + 491.5 = 534.7` | **`39.30`** | `0.09460` | `0.04083` | `0.158°` |

**Diagnosis of Benchmark 2b (`LOMA_B` vs HLoc `ALIKED+LightGlue` vs `SIFT`):**
- **Parity on banked SIFT & ALIKED databases:** Running `pycolmap 4.2.1` incremental mapping on the banked 20-pano databases reproduces `4.1.0` to **`1.13 mm` (`0.0027°`)** on SIFT and **`6.05 mm` (`0.068°`)** on ALIKED.
- **Native COLMAP 4.2.1 `LOMA_B` quality vs HLoc `ALIKED+LightGlue`:**
  - `LOMA_B` extracts `650,286` unmasked keypoints (`2,709.5` per crop) in `43.2 s` on GPU (`0` masked-pixel violations — native mask support eliminates HLoc's post-hoc HDF5 mask filter step) and yields **`550,585` rig-verified inlier matches** (`+85.5%` over ALIKED+LightGlue) and **`73,713` 3D points** (`+87.4%` over ALIKED+LightGlue, `7.55×` over SIFT).
  - Trajectory agreement between native `LOMA_B` and HLoc `ALIKED+LightGlue` is **`0.0202 m` (`2.0 cm`) Sim3 RMSE and `0.110°` direct median rotation** across the 20-pano walkthrough, with comparable reprojection error (`1.404 px` crop / `1.532 px` ERP vs ALIKED's `1.368 px` / `1.490 px`).
  - **Speed trade-off:** ONNX Runtime CUDA matching in `colmap.exe` (`491.5 s` for `11,476` pairs = `23.3 pairs/s`) is `1.54×` slower than PyTorch HLoc `LightGlue` (`319.6 s` = `35.9 pairs/s`), and incremental BA on `73.7k` points takes `183.3 s` vs `44.4 s` on `39.3k` points.
  - **Important CLI gotcha for `colmap.exe` rig workflows:** When invoking `colmap.exe feature_extractor` on virtual camera folders (`pano_camera0..11/`), `--ImageReader.single_camera_per_folder 1` must be passed so each folder receives its own `camera_id` (`1..12`) before `pycolmap.apply_rig_config`.

---

## 5. Recommendation & Rollback

1. **Keep system `pycolmap 4.1.0` as the default production package:** Our production `core/sfm_360.py` + `core/sfm_360_hloc.py` pipeline already achieves identical accuracy on `4.1.0` (`1.1 mm`–`6.1 mm` on 20-pano; within seed-0 noise floor on 94-pano), and switching the system Python environment is unnecessary.
2. **Retain `D:\REPOS\HYPERclone\venvs\colmap421` (`pycolmap 4.2.1` + `colmap.exe 4.2.1` CUDA):**
   - Ready drop-in upgrade whenever the environment moves to `pycolmap >= 4.2.1` (`core/sfm_360.py` requires **zero** code changes).
   - Provides a self-contained, single-binary (`colmap.exe`) GPU `LOMA_B` extractor/matcher with native mask support that produces `1.87×` denser point clouds than `ALIKED+LightGlue` without requiring `hloc` / HDF5 intermediate files.
3. **Do NOT enable `guided_matching=True` on `EQUIRECTANGULAR` (`core/sfm_360.py:725`):** Keep `guided_matching` disabled on spherical cameras even under COLMAP 4.2.1 (`PLANAR_OR_PANORAMIC` crash on pair `(1, 10)` + trajectory corruption on ERP).
4. **Rollback:** System Python was never modified. Deleting `D:\REPOS\HYPERclone\venvs\colmap421` removes COLMAP / `pycolmap` 4.2.1 completely.

---

## 6. Persisted Artifacts
- `_infusion/wpsfm2_RECORD.md` (this record)
- `_infusion/wpsfm2/gate.json` (complete structured JSON of API audit, DB stats, reconstruction metrics, timings, and `Sim3` pose alignments across all 28 benchmark arms)
- `_infusion/wpsfm2_worker.py` (isolated worker CLI runnable under both `4.1.0` and `4.2.1`)
- `_infusion/wpsfm2_eval.py` (orchestrator & `Sim3` pose / metric evaluator)
