#!/usr/bin/env bash
# WP-S1 evaluation: run AFTER all 4 ckpts (2 scenes x ctrl/stack) exist.
# Produces: full 6-scorer scorecards, floater reports, region-crop metrics +
# zoom strips on the flat/shiny pain surfaces. GPU-bound; run when the GPU is free.
set -uo pipefail
cd "$(dirname "$0")/.."
unset APEX_HOLDOUT || true

QAPY="D:/REPOS/HYPERclone/venvs/qa/Scripts/python.exe"

LAP_DATA=_dataset/Laptop/apex_images/training/gsplat_prod/data
GPU_DATA=_dataset/GPU_V4/apex_images/training/gsplat_prod/data
declare -A CK=(
  [laptop_ctrl]=_dataset/Laptop/apex_images/training/wps1/ctrl/output/ckpts/ckpt_29999_rank0.pt
  [laptop_stack]=_dataset/Laptop/apex_images/training/wps1/stack/output/ckpts/ckpt_29999_rank0.pt
  [gpu_ctrl]=_dataset/GPU_V4/apex_images/training/wps1/ctrl/output/ckpts/ckpt_29999_rank0.pt
  [gpu_stack]=_dataset/GPU_V4/apex_images/training/wps1/stack/output/ckpts/ckpt_29999_rank0.pt
)

echo "### 1. Full 6-scorer scorecards ###"
python qa_report.py "${CK[laptop_ctrl]}"  "$LAP_DATA" --factor 2 --scene wps1_laptop_ctrl  --out qa/wps1_laptop_ctrl  --scorers all --qa_python "$QAPY"
python qa_report.py "${CK[laptop_stack]}" "$LAP_DATA" --factor 2 --scene wps1_laptop_stack --out qa/wps1_laptop_stack --scorers all --qa_python "$QAPY"
python qa_report.py "${CK[gpu_ctrl]}"     "$GPU_DATA" --factor 2 --scene wps1_gpu_ctrl     --out qa/wps1_gpu_ctrl     --scorers all --qa_python "$QAPY"
python qa_report.py "${CK[gpu_stack]}"    "$GPU_DATA" --factor 2 --scene wps1_gpu_stack    --out qa/wps1_gpu_stack    --scorers all --qa_python "$QAPY"

echo "### 2. Floater reports ###"
python floater_report.py laptop_ctrl="${CK[laptop_ctrl]}" laptop_stack="${CK[laptop_stack]}" | tee _infusion/wps1_laptop_floaters.txt
python floater_report.py gpu_ctrl="${CK[gpu_ctrl]}"       gpu_stack="${CK[gpu_stack]}"       | tee _infusion/wps1_gpu_floaters.txt

echo "### 3. Region-crop metrics + zoom strips ###"
# Laptop deck (semi-gloss black plastic) — box identical to WP-F1/F2 deck eval
python _infusion/wps1_region_eval.py --data_dir "$LAP_DATA" \
  --box 0.30 0.72 0.55 0.95 --region_name deck \
  --arms ctrl="${CK[laptop_ctrl]}" stack="${CK[laptop_stack]}" \
  --out _infusion/wps1_laptop_region.json \
  --strip_out qa/wps1_laptop_strips --strip_views 0 4 8 12
# GPU_V4 shiny (semi-gloss black metal backplate) — box recorded in wps1_RECORD.md
python _infusion/wps1_region_eval.py --data_dir "$GPU_DATA" \
  --box 0.28 0.72 0.30 0.55 --region_name shiny \
  --arms ctrl="${CK[gpu_ctrl]}" stack="${CK[gpu_stack]}" \
  --out _infusion/wps1_gpu_region.json \
  --strip_out qa/wps1_gpu_strips --strip_views 0 12 24 48

echo "### WP-S1 EVAL DONE ###"
