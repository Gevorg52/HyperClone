#!/usr/bin/env bash
# WP-S2 eval: object-region metrics (primary gate), floater/pose analysis, standard
# floater indicators, and temporal scorers (tsed/raft/flicker). Run after wps2_chain.sh.
#   bash _infusion/wps2_eval.sh 2>&1 | tee _infusion/wps2_eval.log
set -u
cd "$(dirname "$0")/.."
PY="C:/Users/gevor/AppData/Local/Programs/Python/Python311/python.exe"
T=_dataset/GPU_V4/apex_images/training
EVAL=$T/wps2/eval/data
SPARSE=$T/wps1/stack/data/sparse/0
MASKS=$T/wps2/masks_2
A=$T/wps2/a_full/output/ckpts/ckpt_14999_rank0.pt
B=$T/wps2/b_obj/output/ckpts/ckpt_14999_rank0.pt
C=$T/wps2/c_obj_alpha/output/ckpts/ckpt_14999_rank0.pt

echo "############ (1) OBJECT-REGION METRICS (primary gate) + strips ############"
$PY _infusion/wps2_obj_eval.py --eval_dir $EVAL \
  --arms a_full=$A b_obj=$B c_obj_alpha=$C \
  --out _infusion/wps2_obj_metrics.json \
  --strip_out qa/wps2_strips --strip_views 0 9 18 27 36 45 54 63

echo "############ (2) FLOATER + POSE-DRIFT ANALYSIS ############"
$PY _infusion/wps2_analysis.py --eval_dir $EVAL --sparse $SPARSE --masks $MASKS \
  --arms a_full=$A b_obj=$B c_obj_alpha=$C \
  --out _infusion/wps2_floater_analysis.json

echo "############ (3) STANDARD FLOATER INDICATORS ############"
$PY floater_report.py a_full=$A b_obj=$B c_obj_alpha=$C | tee _infusion/wps2_floaters.txt

echo "############ (4) TEMPORAL SCORERS (tsed/raft/flicker), object-framed ############"
for arm in a_full b_obj c_obj_alpha; do
  case $arm in a_full) CK=$A;; b_obj) CK=$B;; c_obj_alpha) CK=$C;; esac
  echo "---- qa scorers: $arm ----"
  $PY qa_report.py $CK $EVAL --factor 2 --lpips_mode tiled \
    --scorers tsed,raft_warp,flicker --out qa/wps2_$arm || true
done

echo "############ WPS2_EVAL_DONE ############"
