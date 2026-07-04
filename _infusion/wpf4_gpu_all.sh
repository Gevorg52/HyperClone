#!/usr/bin/env bash
# WP-F4 GPU pipeline: (1) per-loss mechanism checks (600 steps each, one loss at a
# time, with a NaN-abort gate), (2) the planar-package A/B (15k, seed 0), (3) evals.
# Run from repo root ONLY when the GPU is free (WP-S1 finished). Deterministic (seed 0).
#   bash _infusion/wpf4_gpu_all.sh 2>&1 | tee _infusion/wpf4_gpu_all.log
set -e
cd "$(dirname "$0")/.."
PY=python
BASE=_dataset/Laptop/apex_images/training/wps1/stack/data   # dense_init(707k)+priors+plane_mask
ABDIR=_dataset/Laptop/apex_images/training/wpf4_ab
MECH=_dataset/Laptop/apex_images/training/wpf4_mech
mkdir -p "$MECH" "$ABDIR"
COMMON="--data_dir $BASE --data_factor 2 --test_every 8 --strategy mcmc \
  --cap_max 1000000 --opacity_reg 0.01 --scale_reg 0.01 --antialiased --seed 0"

run_mech () {  # name  <extra flags...>
  local name="$1"; shift
  echo "---- mechanism: $name ----"
  $PY -m core.training.gsplat_trainer $COMMON --max_steps 600 --save_steps 600 \
    "$@" --result_dir "$MECH/$name/output" > "$MECH/$name.log" 2>&1 || true
  grep -E "Planar losses ON|step=(100|300|500|600) " "$MECH/$name.log" | tail -6 || true
}

echo "############ (1) MECHANISM CHECKS — one loss at a time, 600 steps ############"
run_mech lrd --prior_depth_lambda 0.05 --prior_depth_start 50
run_mech lp  --planar_lambda 0.5      --planar_start 50
run_mech lrn --prior_normal_lambda 0.2 --prior_normal_start 50

# NaN-abort gate: fail loudly BEFORE spending 2x15k on the A/B if any loss NaN'd
# or never fired (EMA term missing / not finite).
$PY - "$MECH" <<'PYEOF'
import re, sys
from pathlib import Path
mech = Path(sys.argv[1]); tags = {"lrd": "Lrd", "lp": "Lp", "lrn": "Lrn"}
bad = []
for name, tok in tags.items():
    log = (mech / f"{name}.log").read_text(encoding="utf-8", errors="ignore")
    steps = re.findall(rf"step=(\d+).*?{tok}=([-\d.eE+naif]+)", log)
    if "nan" in log.lower() or " inf" in log.lower():
        bad.append(f"{name}: nan/inf in log")
    if not steps:
        bad.append(f"{name}: EMA term {tok}= never logged (loss did not fire)")
        continue
    last = steps[-1][1]
    try:
        v = float(last)
        if v != v or v in (float("inf"), float("-inf")):
            bad.append(f"{name}: {tok} not finite ({last})")
        elif v == 0.0:
            bad.append(f"{name}: {tok} is exactly 0 (loss inert)")
    except ValueError:
        bad.append(f"{name}: {tok}={last} unparueable")
    print(f"  mech {name}: last {tok}={last}  ({len(steps)} logged)")
if bad:
    print("MECHANISM GATE FAILED:\n  " + "\n  ".join(bad)); sys.exit(3)
print("MECHANISM GATE PASSED — all three losses fired, finite, nonzero.")
PYEOF

echo "############ (2) A/B — planar package ON vs OFF, 15k, on dense_init+mono_depth ############"
# OFF: dense_init + mono_depth (the stacked base going forward)
$PY -m core.training.gsplat_trainer $COMMON --max_steps 15000 --save_steps 15000 \
  --mono_depth_lambda 0.05 --mono_depth_start 2000 \
  --result_dir $ABDIR/off/output 2>&1 | tee $ABDIR/off.log \
  | grep -E "loss ON|Planar|step=[0-9]+000 |Eval|Done" || true
# ON: + planar package (L_rd@3.5k, L_p@7k, L_rn@10k = PlanarGS 7k/14k/20k scaled to 15k)
$PY -m core.training.gsplat_trainer $COMMON --max_steps 15000 --save_steps 15000 \
  --mono_depth_lambda 0.05 --mono_depth_start 2000 \
  --prior_depth_lambda 0.05 --prior_depth_start 3500 \
  --planar_lambda 0.5 --planar_start 7000 \
  --prior_normal_lambda 0.2 --prior_normal_start 10000 \
  --result_dir $ABDIR/on/output 2>&1 | tee $ABDIR/on.log \
  | grep -E "loss ON|Planar|step=[0-9]+000 |Eval|Done" || true

OFF=$ABDIR/off/output/ckpts/ckpt_14999_rank0.pt
ON=$ABDIR/on/output/ckpts/ckpt_14999_rank0.pt

echo "############ (3) EVALS ############"
$PY qa_report.py $OFF $BASE --factor 2 --lpips_mode tiled --out qa/wpf4_ab_off
$PY qa_report.py $ON  $BASE --factor 2 --lpips_mode tiled --out qa/wpf4_ab_on
$PY _infusion/wpf4_flatness_eval.py --data_dir $BASE \
  --arms off=$OFF on=$ON --out _infusion/wpf4_ab_flatness.json
$PY floater_report.py base_off=$OFF planar_on=$ON | tee _infusion/wpf4_ab_floaters.txt
echo "############ WPF4_PIPELINE_DONE ############"
