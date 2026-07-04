#!/usr/bin/env bash
# WP-S1 stacked-levers production run driver.
# Usage:  bash _infusion/wps1_run.sh <Laptop|GPU_V4> <ctrl|stack>
#
# Production recipe (matches run_train_laptop.py + banked GPU_V4 config, at factor 2):
#   mcmc, cap_max 3M, opacity/scale_reg 0.01, antialiased, pose_opt (lr 1e-5),
#   coarse-to-fine factor 4 -> 2 (coarse_until 15000), gated LPIPS 0.2 (patch 512,
#   start 3000), ssim_lambda 0.2, sh_degree 3, seed 0, 30k steps.
# STACK arm additionally enables the three validated faithful levers:
#   dense init (WP-F2): the stack data dir's sparse/0 IS the RoMa-augmented COLMAP
#     model (no trainer flag — the trainer loads whatever sparse/0 it is given),
#   --mono_depth_lambda 0.05 --mono_depth_start 2000 (WP-F1; effective start is
#     raised to coarse_until+1=15001 by the trainer since priors are fine-res),
#   --erank_lambda 0.01 --erank_start 7000 (WP-Q2).
# CTRL arm = identical WITHOUT the three levers (original sparse/0) -> reproduces
# the production recipe under seed 0.
set -euo pipefail
cd "$(dirname "$0")/.."     # repo root
unset APEX_HOLDOUT || true  # never let a stale holdout env leak into the split

SCENE="${1:?scene}"; ARM="${2:?arm}"
BASE="_dataset/${SCENE}/apex_images/training/wps1/${ARM}"
DATA="${BASE}/data"
OUT="${BASE}/output"
LOG="${BASE}/train.log"
mkdir -p "$OUT"

COMMON=( --data_dir "$DATA" --result_dir "$OUT"
  --data_factor 2 --max_steps 30000 --test_every 8 --seed 0
  --sh_degree 3 --ssim_lambda 0.2
  --strategy mcmc --cap_max 3000000 --opacity_reg 0.01 --scale_reg 0.01
  --antialiased --normalize_world_space
  --pose_opt --pose_opt_lr 1e-5
  --coarse_factor 4 --coarse_until 15000
  --lpips_lambda 0.2 --lpips_patch 512 --lpips_start 3000 --lpips_gated
  --save_steps 7000 30000 )

LEVERS=()
if [[ "$ARM" == "stack" ]]; then
  LEVERS=( --mono_depth_lambda 0.05 --mono_depth_start 2000 --mono_depth_use_weight
           --erank_lambda 0.01 --erank_start 7000 )
fi

echo "=== WP-S1 ${SCENE} ${ARM} ===" | tee "$LOG"
echo "argv: python -m core.training.gsplat_trainer ${COMMON[*]} ${LEVERS[*]}" | tee -a "$LOG"
python -m core.training.gsplat_trainer "${COMMON[@]}" "${LEVERS[@]}" 2>&1 | tee -a "$LOG"
echo "=== DONE ${SCENE} ${ARM} exit ${PIPESTATUS[0]} ===" | tee -a "$LOG"
