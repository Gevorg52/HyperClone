@echo off
REM WP-O3 — first 360-derived Gaussian splat. Baseline arm.
REM Dataset: _dataset/360/train/wpo3_flowershop_077 (packaged by core.train_360_prep from
REM WP-O2 ARM A crop-rig, which is itself on the ih_fov-198 re-ingest).
REM
REM Config = production get_default_config() (core/training/gsplat_backend.py) with three
REM recorded deviations, all argued in _infusion/wpo3_RECORD.md:
REM   data_factor  2 -> 1   crops are already 960x960 (0.92 MP); do not throw pixels away
REM   cap_max      1M -> 3M the validated sweet spot (DEVELOPER_NOTES section 6)
REM   coarse_factor 4 -> 0  C2F needs a pre-built images_4/; at 960^2 it would densify at
REM                         240^2, and the dir does not exist so it would silently no-op
REM   pose_opt     on -> off the 12 crops of a pano share an exact optical centre by
REM                         construction (BA fixed the rig); per-crop SE(3) freedom breaks
REM                         that and is the prime suspect for seam artifacts. ON is the
REM                         first follow-up A/B.
REM Masks ACTIVE via --rgb_object_masked (the only mask->photometric path in the trainer).
REM alpha_loss/alpha_obj stay 0: masked-out pixels here are duplicated-coverage / operator /
REM seam, NOT background, so alpha must not be pushed to 0 there.

setlocal
cd /d D:\REPOS\HYPERclone\apex-sfm

python core\training\gsplat_trainer.py ^
  --data_dir _dataset\360\train\wpo3_flowershop_077 ^
  --result_dir _dataset\360\train\wpo3_run_baseline ^
  --holdout_list _dataset\360\train\wpo3_flowershop_077\holdout_test.txt ^
  --data_factor 1 ^
  --max_steps 30000 ^
  --save_steps 7000 30000 ^
  --seed 0 ^
  --strategy mcmc ^
  --cap_max 3000000 ^
  --antialiased ^
  --opacity_reg 0.01 ^
  --scale_reg 0.01 ^
  --ssim_lambda 0.2 ^
  --sh_degree 3 ^
  --rgb_object_masked ^
  --coarse_factor 0
