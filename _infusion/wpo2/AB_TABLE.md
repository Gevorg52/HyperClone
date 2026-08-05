# WP-O2 A/B — crop-rig vs spherical

| metric | ARM A crop-rig | ARM B spherical | ARM B control (ih_fov 200) |
|---|---|---|---|
| panos registered | 151/151 | 151/151 | 151/151 |
| registered % | 100.00 | 100.00 | 100.00 |
| mean reproj err (ERP px) | 1.3419 | 1.2694 | 1.2588 |
| mean reproj err (native px) | 1.2313 | 1.2694 | 1.2588 |
| 3D points | 622325 | 166670 | 166459 |
| observations | 2525083 | 675169 | 662428 |
| mean track length | 4.057 | 4.051 | 3.980 |
| obs / registered pano | 16722 | 4471 | 4387 |
| num reconstructions | 1 | 1 | 1 |
| matcher | sequential | exhaustive | exhaustive |
| images fed to COLMAP | 1812 | 151 | 151 |
| wall clock total (s) | 15213 | 1244 | 1233 |
|   render (s) | 4 | 0 | 0 |
|   extract (s) | 92 | 47 | 47 |
|   match (s) | 5696 | 544 | 542 |
|   mapping (s) | 9393 | 652 | 643 |
| floor-plan extent (up removed) | 11.93 x 5.41 | 11.70 x 5.32 | 11.63 x 5.33 |
| floor-plan aspect (short/long) | 0.454 | 0.455 | 0.458 |
| height extent / height std | 3.23 / 0.926 | 3.11 / 0.912 | 3.10 / 0.913 |
| height / floor-plan extent | 0.271 | 0.266 | 0.266 |
| PCA extent (sorted) | 11.88 x 5.66 x 3.71 | 11.65 x 5.57 x 3.60 | 11.59 x 5.57 x 3.60 |
| median step | 0.4079 | 0.4065 | 0.4172 |
| step p90/median | 2.33 | 2.29 | 2.13 |
| step max/median | 4.17 | 4.07 | 3.93 |
| steps > 5x median | 0 | 0 | 0 |
| path / end-to-end displacement | 23.41 | 22.55 | 22.90 |
| long axis / median step | 29.1 | 28.7 | 27.8 |
| best-fit-plane residual rms / max | 0.882 / 2.430 | 0.864 / 2.349 | 0.862 / 2.349 |
| pano tilt vs mean up (med/p90/max deg) | 11.0 / 27.0 / 48.6 | 11.2 / 26.9 / 48.3 | 11.2 / 27.2 / 48.7 |

## argv

**crop_rig** (seed 0):

```
C:\Users\gevor\AppData\Local\Programs\Python\Python311\python.exe -m core.sfm_360 --capture-type 360 --arm crop_rig --dataset _dataset/360/flowershop_077_ihfov198 --out_dir _dataset/360/sfm/wpo2_crop_rig --overwrite --matcher sequential --overlap 10 --quadratic_overlap 1 --loop_detection 1 --seed 0 --timeout_s 21600 --heartbeat_s 300 --progress 50
```

**spherical** (seed 0):

```
C:\Users\gevor\AppData\Local\Programs\Python\Python311\python.exe -m core.sfm_360 --capture-type 360 --arm spherical --dataset _dataset/360/flowershop_077_ihfov198 --out_dir _dataset/360/sfm/wpo2_spherical --overwrite --matcher exhaustive --seed 0 --timeout_s 21600 --heartbeat_s 300
```

**spherical_ihfov200** (seed 0):

```
C:\Users\gevor\AppData\Local\Programs\Python\Python311\python.exe -m core.sfm_360 --capture-type 360 --arm spherical --dataset _dataset/360/flowershop_077 --out_dir _dataset/360/sfm/wpo2_spherical_ihfov200 --overwrite --matcher exhaustive --seed 0 --timeout_s 21600 --heartbeat_s 300
```

## ceres / CHOLMOD warnings during BA (PLAN_360 §2 flagged these on the ERP probe - watched at scale)

- **crop_rig**: 36 "Linear solver failure ... CHOLMOD warning: Matrix not positive definite" lines in `_dataset/360/sfm_armA.err`
- **spherical**: 157 "Linear solver failure ... CHOLMOD warning: Matrix not positive definite" lines in `_dataset/360/log_wpo2_spherical.log`
- **spherical_ihfov200**: 220 "Linear solver failure ... CHOLMOD warning: Matrix not positive definite" lines in `_dataset/360/log_wpo2_spherical_ihfov200.log`

## mask integrity (independent re-check of the COLMAP databases)

- **crop_rig**: 138102 keypoints across 40 sampled images, **0** on masked-out pixels
- **spherical**: 454287 keypoints across 40 sampled images, **0** on masked-out pixels
- **spherical_ihfov200**: 462963 keypoints across 40 sampled images, **0** on masked-out pixels
