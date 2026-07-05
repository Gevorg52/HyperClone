#!/usr/bin/env bash
# WP-F4 recalibration eval: recal-ON vs the banked OFF arm (same base/seed).
# Masks are now FG60-filtered; deck/tex boxes are fixed so deck flatness + region
# LPIPS stay comparable to the failed run; all-plane RMS is re-derived on FG masks.
set -e
cd "$(dirname "$0")/.."
BASE=_dataset/Laptop/apex_images/training/wps1/stack/data
OFF=_dataset/Laptop/apex_images/training/wpf4_ab/off/output/ckpts/ckpt_14999_rank0.pt
ON=_dataset/Laptop/apex_images/training/wpf4_ab/on_recal/output/ckpts/ckpt_14999_rank0.pt
python qa_report.py $ON $BASE --factor 2 --lpips_mode tiled --out qa/wpf4_ab_on_recal
python _infusion/wpf4_flatness_eval.py --data_dir $BASE \
  --arms off=$OFF on_recal=$ON --out _infusion/wpf4_ab_flatness_recal.json
python floater_report.py base_off=$OFF planar_recal=$ON | tee _infusion/wpf4_ab_floaters_recal.txt
echo "RECAL_EVAL_DONE"
