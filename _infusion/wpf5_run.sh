#!/usr/bin/env bash
# WP-F5 (PPISP) A/B driver. ALL arms run in the gsplat-main venv (ppisp is installed only there),
# seed 0, identical argv except the compensation flags.
# Usage:  bash _infusion/wpf5_run.sh <laptop|gpu|360> <base|bil|pp|ppc>
#   base  production recipe, no photometric compensation (production has bilateral_grid OFF)
#   bil   + --bilateral_grid             (our vendored BilaRF grid, the existing alternative)
#   pp    + --ppisp                      (exposure/WB per frame, vignetting/CRF per camera)
#   ppc   + --ppisp --ppisp_controller   (controller distilled from 80 %: splats frozen 24k->30k)
# Recipes:
#   laptop / gpu = WP-S1 STACK (gui2 production): factor 2, 30k, mcmc 3M, opacity_reg 0, scale_reg
#                  0.01, antialiased, pose_opt 1e-5, C2F 4->2 @15k, gated LPIPS 0.2, dense-init sparse,
#                  mono-depth 0.05, erank 0.01@7k.   (Laptop base == the WP-B2 run `opreg0/b2`.)
#   360          = WP-O5 arm D (best 360 recipe): theater_027_8k crop rig, factor 1, 30k, mcmc 3M,
#                  opacity_reg 0, scale_reg 0.01, loss_mask, no C2F, erank 0.01@7k, pano holdout list.
set -euo pipefail
cd "$(dirname "$0")/.."
unset APEX_HOLDOUT || true
PY="D:/REPOS/HYPERclone/venvs/gsplat_main/Scripts/python.exe"
SCENE="${1:?scene}"; ARM="${2:?arm}"

case "$ARM" in
  base) EXTRA=() ;;
  bil)  EXTRA=( --bilateral_grid ) ;;
  pp)   EXTRA=( --ppisp ) ;;
  ppc)  EXTRA=( --ppisp --ppisp_controller --ppisp_controller_start 0.8 ) ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac

STACK=( --data_factor 2 --max_steps 30000 --test_every 8 --seed 0
  --sh_degree 3 --ssim_lambda 0.2
  --strategy mcmc --cap_max 3000000 --opacity_reg 0 --scale_reg 0.01
  --antialiased --normalize_world_space --pose_opt --pose_opt_lr 1e-5
  --coarse_factor 4 --coarse_until 15000
  --lpips_lambda 0.2 --lpips_patch 512 --lpips_start 3000 --lpips_gated
  --mono_depth_lambda 0.05 --mono_depth_start 2000 --mono_depth_use_weight
  --erank_lambda 0.01 --erank_start 7000 --save_steps 7000 30000 )

case "$SCENE" in
  laptop) DATA=_dataset/Laptop/apex_images/training/wps1/stack/data
          BASE=_dataset/Laptop/apex_images/training/wpf5/${ARM}; RECIPE=( "${STACK[@]}" ) ;;
  gpu)    DATA=_dataset/GPU_V4/apex_images/training/wps1/stack/data
          BASE=_dataset/GPU_V4/apex_images/training/wpf5/${ARM}; RECIPE=( "${STACK[@]}" ) ;;
  360)    DATA=_dataset/360/train/wpo5_theater_027_8k
          BASE=_dataset/360/train/wpf5_${ARM}
          RECIPE=( --holdout_list "$DATA/holdout_test.txt" --max_steps 30000 --save_steps 10000 30000
                   --seed 0 --data_factor 1 --strategy mcmc --cap_max 3000000 --antialiased
                   --opacity_reg 0 --scale_reg 0.01 --ssim_lambda 0.2 --sh_degree 3 --loss_mask
                   --coarse_factor 0 --erank_lambda 0.01 --erank_start 7000 ) ;;
  *) echo "unknown scene $SCENE"; exit 2 ;;
esac
OUT="${BASE}/output"; LOG="${BASE}/train.log"
mkdir -p "$OUT"
ARGV=( --data_dir "$DATA" --result_dir "$OUT" "${RECIPE[@]}" "${EXTRA[@]}" )
echo "=== WP-F5 ${SCENE} ${ARM} ===" | tee "$LOG"
"$PY" -c "import sys,torch,gsplat;print('python',sys.executable,'torch',torch.__version__,'gsplat',gsplat.__version__)" | tee -a "$LOG"
echo "argv: $PY -m core.training.gsplat_trainer ${ARGV[*]}" | tee -a "$LOG"
"$PY" -m core.training.gsplat_trainer "${ARGV[@]}" 2>&1 | tee -a "$LOG"
echo "=== DONE ${SCENE} ${ARM} exit ${PIPESTATUS[0]} ===" | tee -a "$LOG"
