#!/usr/bin/env bash
# WP-N1 evaluation — run AFTER both N ckpts (na, nb ckpt_29999) exist.
# F2 baseline = the banked WP-S1 GPU stack ckpt (factor-2, pose_opt ON), REUSED.
#
# Produces:
#  1. COMMON-RES-A masked scorecards (qa_report --factor 2) on N-a, N-b — the exact
#     validated harness that produced the banked F2 stack scorecard (masked-LPIPS
#     0.4071 @ factor 2). Directly comparable to the bank.
#  2. COMMON-RES-B (render@f1 -> downsample to f2) + NATIVE-res masked metrics +
#     train/test gap (June ~6 dB signature) for ALL THREE arms (wpn1_eval.py).
#  3. MICROSTRUCTURE region-LPIPS (native factor-1) + zoom strips GT|F2|N-a|N-b on
#     the WP-S1 shiny/microtext box and dedicated fin-grille boxes (region_eval).
set -uo pipefail
cd "$(dirname "$0")/.."
unset APEX_HOLDOUT || true

QAPY="D:/REPOS/HYPERclone/venvs/qa/Scripts/python.exe"
DATA=_dataset/GPU_V4/apex_images/training/wps1/n1/data
F2=_dataset/GPU_V4/apex_images/training/wps1/stack/output/ckpts/ckpt_29999_rank0.pt
NA=_dataset/GPU_V4/apex_images/training/wps1/n1/na/output/ckpts/ckpt_29999_rank0.pt
NB=_dataset/GPU_V4/apex_images/training/wps1/n1/nb/output/ckpts/ckpt_29999_rank0.pt

echo "### 1. Common-res-A masked scorecards (qa_report --factor 2) ###"
# F2 reused from qa/wps1_gpu_stack (masked-LPIPS 0.4071). N-a / N-b scored the SAME
# way. --scorers OFF (default): the core masked PSNR/SSIM/LPIPS is always computed
# and is identical whether external scorers run or not -> comparable to the bank.
# F2 re-scored via the SAME data_dir/harness as N-a/N-b (removes any data-dir
# confound vs the banked qa/wps1_gpu_stack; should reproduce masked-LPIPS ~0.4071).
python qa_report.py "$F2" "$DATA" --factor 2 --scene wpn1_gpu_f2_f2 --out qa/wpn1_gpu_f2_f2
python qa_report.py "$NA" "$DATA" --factor 2 --scene wpn1_gpu_na_f2 --out qa/wpn1_gpu_na_f2
python qa_report.py "$NB" "$DATA" --factor 2 --scene wpn1_gpu_nb_f2 --out qa/wpn1_gpu_nb_f2

echo "### 2. Common-res-B + native masked metrics + train/test gap (all 3 arms) ###"
python _infusion/wpn1_eval.py --data_dir "$DATA" \
  --arms F2="$F2" N-a="$NA" N-b="$NB" \
  --common_factor 2 --train_sample 72 \
  --out _infusion/wpn1_gpu_masked.json

echo "### 3. Microstructure region-LPIPS (native factor-1) + zoom strips ###"
# 3a. WP-S1 shiny/microtext box (etched traces + ZOTAC logo + LIVE-TO-GAME microtext)
python _infusion/wps1_region_eval.py --data_dir "$DATA" --factor 1 \
  --box 0.28 0.72 0.30 0.55 --region_name shiny \
  --arms F2="$F2" N-a="$NA" N-b="$NB" \
  --out _infusion/wpn1_gpu_shiny_region.json \
  --strip_out qa/wpn1_gpu_strips_shiny --strip_views 0 12 24 48
# 3b. fin-grille zoom, view48 (vertical fin stack top-right of the backplate)
python _infusion/wps1_region_eval.py --data_dir "$DATA" --factor 1 \
  --box 0.78 0.92 0.13 0.27 --region_name fins48 \
  --arms F2="$F2" N-a="$NA" N-b="$NB" \
  --out _infusion/wpn1_gpu_fins48_region.json \
  --strip_out qa/wpn1_gpu_strips_fins48 --strip_views 48
# 3c. fin-grille + ridge zoom, view24 (card end fin mesh + top-edge ridges)
python _infusion/wps1_region_eval.py --data_dir "$DATA" --factor 1 \
  --box 0.60 0.78 0.55 0.74 --region_name fins24 \
  --arms F2="$F2" N-a="$NA" N-b="$NB" \
  --out _infusion/wpn1_gpu_fins24_region.json \
  --strip_out qa/wpn1_gpu_strips_fins24 --strip_views 24

echo "### WP-N1 EVAL DONE ###"
