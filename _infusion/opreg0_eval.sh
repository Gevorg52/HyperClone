#!/usr/bin/env bash
# STEP 0 / WP-B2 gate evaluation on the Laptop WP-S1 STACK config.
# Usage: bash _infusion/opreg0_eval.sh <arm> [<arm> ...]     arms: or0 or01 b2 wps1 (= banked WP-S1 stack)
# Per arm: full 6-scorer WP-Q1 scorecard (masked PSNR/SSIM/LPIPS ...), deck region LPIPS,
# floater report, and a pruned (opacity < 0.05 removed) PLY + prune stats.
# Off-path renders + fog stats for ALL arms together: _infusion/offpath_orbit.py (see bottom).
set -uo pipefail
cd "$(dirname "$0")/.."
unset APEX_HOLDOUT || true
QAPY="D:/REPOS/HYPERclone/venvs/qa/Scripts/python.exe"
DATA=_dataset/Laptop/apex_images/training/gsplat_prod/data   # identical to the WP-S1 eval

ckpt_of() {
  case "$1" in
    wps1) echo _dataset/Laptop/apex_images/training/wps1/stack/output/ckpts/ckpt_29999_rank0.pt ;;
    *)    echo _dataset/Laptop/apex_images/training/opreg0/$1/output/ckpts/ckpt_29999_rank0.pt ;;
  esac
}

ARGS=()
for ARM in "$@"; do
  CK=$(ckpt_of "$ARM")
  echo "### $ARM : $CK"
  python qa_report.py "$CK" "$DATA" --factor 2 --scene "opreg0_laptop_${ARM}" \
      --out "qa/opreg0_laptop_${ARM}" --scorers all --qa_python "$QAPY"
  python _infusion/wps1_region_eval.py --data_dir "$DATA" \
      --box 0.30 0.72 0.55 0.95 --region_name deck --arms "${ARM}=${CK}" \
      --out "qa/opreg0_laptop_${ARM}/region_deck.json"
  python floater_report.py "${ARM}=${CK}" | tee "qa/opreg0_laptop_${ARM}/floaters.txt"
  PLYDIR=$(dirname "$(dirname "$CK")")/ply
  python tools/prune_ply.py --ckpt "$CK" --opacity_min 0.05 \
      --out "${PLYDIR}/point_cloud_29999_pruned_op05.ply" \
      --stats_json "qa/opreg0_laptop_${ARM}/prune_stats.json"
  ARGS+=( --arm "${ARM}=${CK}" )
done
python _infusion/offpath_orbit.py --data_dir "$DATA" --factor 2 \
    --out "qa/opreg0_laptop_offpath_$(IFS=_; echo "$*")" "${ARGS[@]}"
echo "### EVAL DONE $*"
