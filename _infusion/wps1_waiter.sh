#!/usr/bin/env bash
# Lightweight terminal-signal waiter for the WP-S1 chain. Emits exactly one line
# and exits: WPS1_CHAIN_DONE (chain reached ALL_DONE or final ckpt present),
# WPS1_CHAIN_DIED (no GPU compute process for ~3 min while not done -> the chain
# process was killed), or WPS1_WAITER_TIMEOUT. Uses the same run_in_background
# mechanism as the chain (proven to coexist with a sibling bg job).
cd "$(dirname "$0")/.."
LOG=_dataset/wps1_chain.log
FINAL=_dataset/GPU_V4/apex_images/training/wps1/stack/output/ckpts/ckpt_29999_rank0.pt
dead=0
for i in $(seq 1 900); do   # ~7.5h max at 30s
  if grep -qa ALL_DONE "$LOG" 2>/dev/null || [ -f "$FINAL" ]; then
    echo "WPS1_CHAIN_DONE i=$i"; tail -n 8 "$LOG" 2>/dev/null; exit 0
  fi
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -qE '[0-9]'; then
    dead=0
  else
    dead=$((dead+1))
  fi
  if [ "$dead" -ge 6 ]; then
    echo "WPS1_CHAIN_DIED (no GPU compute ~3min, not done) i=$i"; tail -n 8 "$LOG" 2>/dev/null; exit 2
  fi
  sleep 30
done
echo "WPS1_WAITER_TIMEOUT"; exit 3
