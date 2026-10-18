#!/usr/bin/env bash
# WP-F5 gate evaluation. Usage: bash _infusion/wpf5_eval.sh <laptop|gpu|360>
# Arms: base bil pp ppc. All metrics are on the RAW checkpoint/PLY render (what Postshot shows),
# evaluated with the same system-python harness as every banked scorecard; the PPISP-applied
# render and the tone-vs-structure split come from _infusion/wpf5_tone_eval.py (venv).
set -uo pipefail
cd "$(dirname "$0")/.."
unset APEX_HOLDOUT || true
QAPY="D:/REPOS/HYPERclone/venvs/qa/Scripts/python.exe"
VPY="D:/REPOS/HYPERclone/venvs/gsplat_main/Scripts/python.exe"
SCENE="${1:?scene}"
ARMS=(${WPF5_ARMS:-base bil pp ppc})   # WPF5_ARMS='base bil' WPF5_PER_ARM_ONLY=1 -> per-arm steps only
mkdir -p _infusion/wpf5

ck() {  # scene arm -> ckpt
  case "$1" in
    laptop) [[ "$2" == base ]] && echo _dataset/Laptop/apex_images/training/opreg0/b2/output/ckpts/ckpt_29999_rank0.pt \
                               || echo _dataset/Laptop/apex_images/training/wpf5/$2/output/ckpts/ckpt_29999_rank0.pt ;;
    gpu)    echo _dataset/GPU_V4/apex_images/training/wpf5/$2/output/ckpts/ckpt_29999_rank0.pt ;;
    360)    echo _dataset/360/train/wpf5_$2/output/ckpts/ckpt_29999_rank0.pt ;;
  esac
}

TONE=(); OFF=()
for A in "${ARMS[@]}"; do
  C=$(ck "$SCENE" "$A"); [[ -f "$C" ]] || { echo "MISSING $C"; exit 3; }
  TONE+=( --arm "$A=$C" ); OFF+=( --arm "$A=$C" )
  PLYDIR=$(dirname "$(dirname "$C")")/ply
  OUTQ=qa/wpf5_${SCENE}_${A}; mkdir -p "$OUTQ"
  if [[ "$SCENE" != 360 ]]; then
    DATA=$([[ "$SCENE" == laptop ]] && echo _dataset/Laptop/apex_images/training/gsplat_prod/data                                     || echo _dataset/GPU_V4/apex_images/training/gsplat_prod/data)
    if [[ "$SCENE" == laptop ]]; then BOX=(0.30 0.72 0.55 0.95); RN=deck; else BOX=(0.28 0.72 0.30 0.55); RN=shiny; fi
  fi
  [[ -n "${WPF5_JOINT_ONLY:-}" ]] && continue   # per-arm outputs already exist
  python tools/prune_ply.py --ckpt "$C" --opacity_min 0.05 \
      --out "${PLYDIR}/point_cloud_29999_pruned_op05.ply" --stats_json "$OUTQ/prune_stats.json"
  if [[ "$SCENE" != 360 ]]; then
    DATA=$([[ "$SCENE" == laptop ]] && echo _dataset/Laptop/apex_images/training/gsplat_prod/data \
                                    || echo _dataset/GPU_V4/apex_images/training/gsplat_prod/data)
    python qa_report.py "$C" "$DATA" --factor 2 --scene "wpf5_${SCENE}_${A}" --out "$OUTQ" \
        --scorers all --qa_python "$QAPY"
    if [[ "$SCENE" == laptop ]]; then BOX=(0.30 0.72 0.55 0.95); RN=deck; else BOX=(0.28 0.72 0.30 0.55); RN=shiny; fi
    python _infusion/wps1_region_eval.py --data_dir "$DATA" --box "${BOX[@]}" --region_name "$RN" \
        --arms "${A}=${C}" --out "$OUTQ/region_${RN}.json"
    python floater_report.py "${A}=${C}" | tee "$OUTQ/floaters.txt"
  fi
done

[[ -n "${WPF5_PER_ARM_ONLY:-}" ]] && { echo "### per-arm only: done ${ARMS[*]}"; exit 0; }
if [[ "$SCENE" == 360 ]]; then
  DATA=_dataset/360/train/wpo5_theater_027_8k
  python _infusion/wpo5_eval.py --data_dir "$DATA" --holdout_list "$DATA/holdout_test.txt" \
      --skip_existing_metrics "${OFF[@]}" --out _infusion/wpf5/eval360 --do metrics,offpath,fog --frame_cam 6
  "$VPY" _infusion/wpf5_tone_eval.py --data_dir "$DATA" --factor 1 \
      --holdout_list "$DATA/holdout_test.txt" --max_views 48 "${TONE[@]}" --out _infusion/wpf5/360_tone.json
else
  python _infusion/offpath_orbit.py --data_dir "$DATA" --factor 2 --out "qa/wpf5_${SCENE}_offpath" "${OFF[@]}"
  "$VPY" _infusion/wpf5_tone_eval.py --data_dir "$DATA" --factor 2 "${TONE[@]}" \
      --out "_infusion/wpf5/${SCENE}_tone.json"
  python _infusion/gate_summarize.py --prefix "qa/wpf5_${SCENE}_" --arms "${ARMS[@]}" \
      --region "region_${RN}.json" --offpath "qa/wpf5_${SCENE}_offpath/offpath.json" \
      --out "_infusion/wpf5/${SCENE}_gate.json"
fi
echo "### WPF5 EVAL DONE $SCENE"
