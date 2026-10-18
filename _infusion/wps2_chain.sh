#!/usr/bin/env bash
# WP-S2 object-mode background study — 3-arm A/B, 15k, seed 0, cap 1M, factor 2.
# WP-S1 STACKED config: dense_init (baked into sparse/0) + mono_depth 0.05 + erank
# 0.01@7k + gated-LPIPS 0.2 + pose_opt. Established 15k A/B protocol (WP-F4): no C2F,
# factor 2 throughout, mono_depth_start 2000 (coarse_until->0 when coarse_factor unset).
# Runs SEQUENTIAL (one GPU). Robust to harness job-tracking severs (WP-S1 orphan note).
#   bash _infusion/wps2_chain.sh 2>&1 | tee _infusion/wps2_chain.log
set -u
cd "$(dirname "$0")/.."
PY="C:/Users/gevor/AppData/Local/Programs/Python/Python311/python.exe"
T=_dataset/GPU_V4/apex_images/training

COMMON="--data_factor 2 --max_steps 15000 --test_every 8 --seed 0 --sh_degree 3 \
  --ssim_lambda 0.2 --strategy mcmc --cap_max 1000000 --opacity_reg 0.01 --scale_reg 0.01 \
  --antialiased --normalize_world_space --pose_opt --pose_opt_lr 1e-5 \
  --mono_depth_lambda 0.05 --mono_depth_start 2000 --mono_depth_use_weight \
  --erank_lambda 0.01 --erank_start 7000 \
  --lpips_lambda 0.2 --lpips_patch 512 --lpips_start 3000 --lpips_gated \
  --save_steps 15000"

echo "############ ARM (a) full-scene = CURRENT PROD (control) ############"
$PY -m core.training.gsplat_trainer $COMMON \
  --data_dir $T/wps1/stack/data \
  --result_dir $T/wps2/a_full/output 2>&1 | tee _infusion/wps2_a_full.log \
  | grep -E "Mono-depth|alpha|Pose|loss ON|step=[0-9]+000 |Eval|Checkpoint|Done|ERROR" || true

echo "############ ARM (b) object-only, NO alpha ############"
$PY -m core.training.gsplat_trainer $COMMON \
  --data_dir $T/wps2/obj/data --mono_depth_object_masked \
  --result_dir $T/wps2/b_obj/output 2>&1 | tee _infusion/wps2_b_obj.log \
  | grep -E "Mono-depth|alpha|Pose|loss ON|step=[0-9]+000 |Eval|Checkpoint|Done|ERROR" || true

echo "############ ARM (c) object-only + alpha_loss 0.1 ############"
$PY -m core.training.gsplat_trainer $COMMON \
  --data_dir $T/wps2/obj/data --mono_depth_object_masked \
  --alpha_loss_lambda 0.1 \
  --result_dir $T/wps2/c_obj_alpha/output 2>&1 | tee _infusion/wps2_c_obj_alpha.log \
  | grep -E "Mono-depth|alpha|Pose|loss ON|step=[0-9]+000 |Eval|Checkpoint|Done|ERROR" || true

echo "############ WPS2_CHAIN_DONE ############"
