#!/usr/bin/env bash
# WP-G1 phase-1: 3DGS-DR (deferred reflection) UPSTREAM trainer on our two scenes.
# Ceiling probe (NOT a controlled A/B): different rasterizer (c7 deferred + cubemap
# env map), Inria adaptive densify (NOT MCMC), no pose_opt, no mono-depth/erank/
# gated-LPIPS levers, sparse ALIKED init (paper's regime). Documented deltas vs WP-S1.
#
# Recipe rationale:
#  - real-scene lineage (garden/sedan/toycar): BLACK bg (no --white_background;
#    matches our premultiplied black-bg frames), normal-propagation via longer_prop.
#  - iterations 30000 (== WP-S1 base, for comparability) + longer_prop 24000
#    (TOT 54001; propagation window iter 3000->48000). Reduced from the authors'
#    real-scene 61000/36000 for GPU coordination (WP-S2 queued) — core geometry is
#    done by 30k (position_lr_max_steps=30000); the tail is refl/normal refinement.
#  - NO --use_env_scope: our bg is masked black so there is little background to
#    suppress; a mis-sized sphere would wrongly kill object specular (false negative).
#  - factor-2 undistorted PINHOLE frames (exported by wpg1_export_colmap.py) so GT
#    is byte-identical to the WP-S1 stack eval.
set -e
DR=/d/REPOS/HYPERclone/3DGS-DR
PY=/c/Users/gevor/AppData/Local/Programs/Python/Python311/python.exe
BASE=/d/REPOS/HYPERclone/apex-sfm/_dataset
ITERS=30000
LP=24000

run() {  # $1 = scene subdir under _dataset ; $2+ = extra train.py flags (e.g. env_scope)
  local scene="$1"; shift
  local src="$BASE/$scene/apex_images/training/wpg1/3dgsdr_data"
  local out="$BASE/$scene/apex_images/training/wpg1/3dgsdr_out"
  echo "=== [$scene] 3DGS-DR train: iters=$ITERS longer_prop=$LP (TOT $((ITERS+LP+1))) extra=[$*] ==="
  cd "$DR"
  "$PY" -u train.py -s "$src" -m "$out" --eval \
    --iterations $ITERS --longer_prop_iter $LP \
    --test_iterations $ITERS $((ITERS+LP)) \
    --save_iterations $((ITERS+LP)) "$@"
  echo "=== [$scene] DONE ==="
}

# Laptop: ran WITHOUT env_scope (full outdoor scene; sparse ALIKED points too spread
#   to define a clean object sphere). GPU_V4: WITH env_scope (compact object, well-defined
#   sphere) = authors' real-scene recipe, DR's best shot.
#   Laptop:  bash wpg1_train.sh Laptop
#   GPU_V4:  bash wpg1_train.sh GPU_V4 --use_env_scope --env_scope_center 0.133 0.37 0.107 --env_scope_radius 1.0
run "$@"
