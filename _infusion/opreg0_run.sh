#!/usr/bin/env bash
# STEP 0 (2026-10-17) — opacity_reg default defect regression.
# Usage:  bash _infusion/opreg0_run.sh <arm> [python]
#   arm  or0   --opacity_reg 0     (the NEW default)
#        or01  --opacity_reg 0.01  (the OLD default; also a seed-0 repeat of the banked
#              WP-S1 Laptop STACK run -> gives a run-to-run noise floor vs the bank)
#        b2    --opacity_reg 0 under the gsplat-main venv (STEP 1 / WP-B2 reproduction)
#   python  interpreter to use (default: system python = gsplat 1.5.3)
#
# Config = WP-S1 Laptop STACK, i.e. the gui2 production stack (_infusion/wps1_run.sh,
# wps1_RECORD.md): factor 2, 30k, mcmc cap 3M, scale_reg 0.01, antialiased, pose_opt
# 1e-5, C2F 4->2 @15k, gated LPIPS 0.2, dense-init sparse (wps1/stack/data),
# mono-depth 0.05, erank 0.01@7k, seed 0. ONLY --opacity_reg (and the interpreter) vary.
set -euo pipefail
cd "$(dirname "$0")/.."     # repo root
unset APEX_HOLDOUT || true

ARM="${1:?arm}"; PY="${2:-python}"
case "$ARM" in
  or0|or0r)  OREG=0 ;;   # or0r = seed-0 REPEAT of or0 (noise floor at opacity_reg 0; runs concurrently with b2)
  or01) OREG=0.01 ;;
  b2)   OREG=0 ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
DATA="_dataset/Laptop/apex_images/training/wps1/stack/data"
BASE="_dataset/Laptop/apex_images/training/opreg0/${ARM}"
OUT="${BASE}/output"; LOG="${BASE}/train.log"
mkdir -p "$OUT"

ARGV=( --data_dir "$DATA" --result_dir "$OUT"
  --data_factor 2 --max_steps 30000 --test_every 8 --seed 0
  --sh_degree 3 --ssim_lambda 0.2
  --strategy mcmc --cap_max 3000000 --opacity_reg "$OREG" --scale_reg 0.01
  --antialiased --normalize_world_space
  --pose_opt --pose_opt_lr 1e-5
  --coarse_factor 4 --coarse_until 15000
  --lpips_lambda 0.2 --lpips_patch 512 --lpips_start 3000 --lpips_gated
  --mono_depth_lambda 0.05 --mono_depth_start 2000 --mono_depth_use_weight
  --erank_lambda 0.01 --erank_start 7000
  --save_steps 7000 30000 )

echo "=== opreg0 ${ARM} ===" | tee "$LOG"
"$PY" -c "import sys,torch,gsplat;print('python',sys.executable,'torch',torch.__version__,'gsplat',gsplat.__version__,gsplat.__file__)" | tee -a "$LOG"
echo "argv: $PY -m core.training.gsplat_trainer ${ARGV[*]}" | tee -a "$LOG"
"$PY" -m core.training.gsplat_trainer "${ARGV[@]}" 2>&1 | tee -a "$LOG"
echo "=== DONE ${ARM} exit ${PIPESTATUS[0]} ===" | tee -a "$LOG"
