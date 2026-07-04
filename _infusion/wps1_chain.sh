#!/usr/bin/env bash
# WP-S1 chained sequential trainer for the remaining 3 arms.
# Robust: NO `set -e` at the chain level, so a single run's failure is recorded
# (via its exit code) but does not abort the remaining runs. Each run's full
# stdout goes to <scene>/wps1/<arm>/train.log (via wps1_run.sh's tee); this
# chain writes only START/END/ALL_DONE markers to the chain log for monitoring.
cd "$(dirname "$0")/.."
CHAINLOG=_dataset/wps1_chain.log
: > "$CHAINLOG"

run () {
  local scene=$1 arm=$2
  echo "[chain] START $scene $arm $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$CHAINLOG"
  bash _infusion/wps1_run.sh "$scene" "$arm"
  local rc=${PIPESTATUS[0]}
  echo "[chain] END $scene $arm exit=$rc $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$CHAINLOG"
}

run Laptop stack
run GPU_V4 ctrl
run GPU_V4 stack
echo "[chain] ALL_DONE $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$CHAINLOG"
