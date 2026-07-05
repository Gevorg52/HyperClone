#!/usr/bin/env bash
# WP-N1 native-resolution isolation experiment — training chain (GPU_V4, the
# historical June native-res failure scene, 24 MP source).
#
# Two new arms, both = the WP-S1 stacked recipe at NATIVE factor-1 with
# coarse-to-fine (factor 2 -> factor 1 at step 15000). Only difference:
#   N-a : pose_opt OFF   (the key isolation: June root-caused the regression to
#                         pose_opt overfitting train poses at 4x sharper photo grad)
#   N-b : pose_opt ON    (tests whether the stack alone fixed the interaction)
#
# Reuses (does NOT retrain): the banked WP-S1 GPU stack ckpt as the F2 baseline.
# Dense-init: data_dir sparse/0 = the 882,902-pt dense model (WP-S1). Priors:
# data_dir/priors = a FRESH factor-1 depth-only MoGe cache (core.priors
# --depth_only, built against the original ALIKED sparse exactly as the factor-2
# WP-S1 priors were, so the ONLY prior difference is resolution).
#
# Run sequentially (we own the long runs; other agents run bursty jobs). Seed 0,
# 30k, cap 3M. Watch peak VRAM (logged every 100 steps in train.log).
set -uo pipefail
cd "$(dirname "$0")/.."
unset APEX_HOLDOUT || true

DATA=_dataset/GPU_V4/apex_images/training/wps1/n1/data
NA_OUT=_dataset/GPU_V4/apex_images/training/wps1/n1/na/output
NB_OUT=_dataset/GPU_V4/apex_images/training/wps1/n1/nb/output
LOGD=_dataset/GPU_V4/apex_images/training/wps1/n1/logs
mkdir -p "$NA_OUT" "$NB_OUT" "$LOGD"

# Common argv = WP-S1 stacked recipe, but data_factor 1 + coarse_factor 2.
common=(
  --data_dir "$DATA"
  --data_factor 1 --max_steps 30000 --test_every 8 --seed 0 --sh_degree 3 --ssim_lambda 0.2
  --strategy mcmc --cap_max 3000000 --opacity_reg 0.01 --scale_reg 0.01
  --antialiased --normalize_world_space
  --coarse_factor 2 --coarse_until 15000
  --lpips_lambda 0.2 --lpips_patch 512 --lpips_start 3000 --lpips_gated
  --save_steps 7000 30000
  --mono_depth_lambda 0.05 --mono_depth_start 2000 --mono_depth_use_weight
  --erank_lambda 0.01 --erank_start 7000
)

run_arm () {
  local name="$1"; local out="$2"; shift 2
  echo "=== WP-N1 arm $name START $(date -Iseconds) ===" | tee -a "$LOGD/${name}.log"
  python -m core.training.gsplat_trainer "${common[@]}" --result_dir "$out" "$@" \
    >> "$LOGD/${name}.log" 2>&1
  local rc=$?
  echo "=== WP-N1 arm $name END rc=$rc $(date -Iseconds) ===" | tee -a "$LOGD/${name}.log"
  return $rc
}

# N-a: pose_opt OFF (omit --pose_opt). N-b: pose_opt ON.
run_arm na "$NA_OUT"
run_arm nb "$NB_OUT" --pose_opt --pose_opt_lr 1e-5

echo "WPN1_TRAIN_ALL_DONE $(date -Iseconds)"
