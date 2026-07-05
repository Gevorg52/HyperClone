#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
unset APEX_HOLDOUT || true
DATA=_dataset/GPU_V4/apex_images/training/wps1/n1/data
F2=_dataset/GPU_V4/apex_images/training/wps1/stack/output/ckpts/ckpt_29999_rank0.pt
NA=_dataset/GPU_V4/apex_images/training/wps1/n1/na/output/ckpts/ckpt_29999_rank0.pt
NB=_dataset/GPU_V4/apex_images/training/wps1/n1/nb/output/ckpts/ckpt_29999_rank0.pt
echo "### 2. Common-res-B + native masked metrics + train/test gap ###"
python _infusion/wpn1_eval.py --data_dir "$DATA" --arms F2="$F2" N-a="$NA" N-b="$NB" --common_factor 2 --train_sample 72 --out _infusion/wpn1_gpu_masked.json
echo "### 3. Microstructure region-LPIPS (native factor-1) + zoom strips ###"
python _infusion/wps1_region_eval.py --data_dir "$DATA" --factor 1 --box 0.28 0.72 0.30 0.55 --region_name shiny --arms F2="$F2" N-a="$NA" N-b="$NB" --out _infusion/wpn1_gpu_shiny_region.json --strip_out qa/wpn1_gpu_strips_shiny --strip_views 0 12 24 48
python _infusion/wps1_region_eval.py --data_dir "$DATA" --factor 1 --box 0.78 0.92 0.13 0.27 --region_name fins48 --arms F2="$F2" N-a="$NA" N-b="$NB" --out _infusion/wpn1_gpu_fins48_region.json --strip_out qa/wpn1_gpu_strips_fins48 --strip_views 48
python _infusion/wps1_region_eval.py --data_dir "$DATA" --factor 1 --box 0.60 0.78 0.55 0.74 --region_name fins24 --arms F2="$F2" N-a="$NA" N-b="$NB" --out _infusion/wpn1_gpu_fins24_region.json --strip_out qa/wpn1_gpu_strips_fins24 --strip_views 24
echo "### WP-N1 EVAL DONE ###"
