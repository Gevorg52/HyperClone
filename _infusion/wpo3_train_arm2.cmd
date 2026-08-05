@echo off
REM WP-O3 ARM 2 — the corrected run, after ARM 1 (production config) diverged.
REM Evidence for every deviation: _infusion/wpo3/diag_collapse{,2,3}.json (8 short arms),
REM written up in _infusion/wpo3_RECORD.md section 4.
REM
REM   --opacity_reg 0        THE FIX. At the production 0.01, arms A/A2/C/F/H all collapse
REM                          to mean opacity ~5e-4 (99.9% dead) and gsplat MCMC then injects
REM                          MAXIMUM noise into zero-opacity gaussians
REM                          (op_sigmoid(1-opacity), k=100, x0=0.995) -> positions random-walk
REM                          to |mean| ~1e3-1e6 -> flat renders / CUDA OOM-crash. Arm H
REM                          (opacity_reg 0.01, scale_reg 0) collapses, arm G (opacity_reg 0,
REM                          scale_reg 0.01) is healthy => opacity_reg alone is the cause.
REM   --scale_reg 0.01       KEPT at production. Arm G shows it is harmless here (and 0.3 dB
REM                          better than zeroing both).
REM   --loss_mask            masks active with the CORRECT semantics: L1 and the SSIM map are
REM                          averaged over the VALID pixels only. --rgb_object_masked would
REM                          composite over black, which on these ~33%-valid crops divides the
REM                          photometric term by 3 and manufactures mask-boundary edges.
REM   --max_steps 200000     1584 training views. 30k steps = 19 epochs; production runs get
REM                          100-250. Also ~6.5x fewer pixel-gradients than a GPU_V4 run
REM                          (30k x 0.92 MP vs 30k x 6 MP). 200k = ~126 epochs.
REM   --coarse_factor 0      C2F needs a pre-built images_4/; at 960^2 it is the wrong lever.
REM   pose_opt off           the 12 crops of a pano share an exact optical centre by
REM                          construction (BA fixed the rig); per-crop SE(3) freedom breaks it.

setlocal
cd /d D:\REPOS\HYPERclone\apex-sfm

python core\training\gsplat_trainer.py ^
  --data_dir _dataset\360\train\wpo3_flowershop_077 ^
  --result_dir _dataset\360\train\wpo3_run_arm2 ^
  --holdout_list _dataset\360\train\wpo3_flowershop_077\holdout_test.txt ^
  --data_factor 1 ^
  --max_steps 200000 ^
  --save_steps 30000 60000 100000 150000 200000 ^
  --seed 0 ^
  --strategy mcmc ^
  --cap_max 3000000 ^
  --antialiased ^
  --opacity_reg 0 ^
  --scale_reg 0.01 ^
  --ssim_lambda 0.2 ^
  --sh_degree 3 ^
  --loss_mask ^
  --coarse_factor 0
