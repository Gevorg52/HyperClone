#!/usr/bin/env bash
# WP-SH4 phase A smoke train (readiness only, NOT a quality A/B).
# Arm sh4 = the deliverable smoke (--sh_degree 4). Arm sh3 = identical argv at --sh_degree 3,
# used ONLY for time / VRAM / PLY-size comparison (GPU is shared with another agent -> timing
# is indicative). Explicit --opacity_reg/--scale_reg so a concurrent default change cannot
# alter the run (WP-F1 Laptop recipe values).
set -u
cd /d/REPOS/HYPERclone/apex-sfm
OUT=_dataset/Laptop/apex_images/training/wpsh4_smoke
mkdir -p "$OUT"
sha256sum core/training/gsplat_trainer.py > "$OUT/trainer_sha256.txt"
git diff core/training/gsplat_trainer.py > "$OUT/trainer_uncommitted.diff"
for DEG in 4 3; do
  ARGS=(-m core.training.gsplat_trainer
    --data_dir _dataset/Laptop/apex_images/training/gsplat_prod/data
    --data_factor 4 --max_steps 5000 --test_every 8
    --strategy mcmc --cap_max 500000 --opacity_reg 0.01 --scale_reg 0.01
    --antialiased --seed 0 --save_steps 5000 --sh_degree $DEG
    --result_dir "$OUT/sh$DEG")
  echo "python ${ARGS[*]}" > "$OUT/sh$DEG.argv.txt"
  T0=$(date +%s)
  python "${ARGS[@]}" > "$OUT/sh$DEG.log" 2>&1
  RC=$?
  echo "rc=$RC wall_s=$(( $(date +%s) - T0 ))" > "$OUT/sh$DEG.exit.txt"
done
