#!/usr/bin/env bash
# WP-S1 heartbeat: polls for the FINAL ckpt (GPU_V4 stack). Exits 0 when the whole
# sequence is done, else runs until the harness reaps it (~hourly) which re-engages
# the agent. On exit it prints a full census (all 4 ckpts + chain-log tail + live
# trainer PIDs) so the agent gets a status snapshot each time it fires.
cd "$(dirname "$0")/.."
LAP_C=_dataset/Laptop/apex_images/training/wps1/ctrl/output/ckpts/ckpt_29999_rank0.pt
LAP_S=_dataset/Laptop/apex_images/training/wps1/stack/output/ckpts/ckpt_29999_rank0.pt
GPU_C=_dataset/GPU_V4/apex_images/training/wps1/ctrl/output/ckpts/ckpt_29999_rank0.pt
GPU_S=_dataset/GPU_V4/apex_images/training/wps1/stack/output/ckpts/ckpt_29999_rank0.pt

census () {
  echo "=== WPS1 CENSUS $(date '+%H:%M:%S') ==="
  for name in "LAP_ctrl:$LAP_C" "LAP_stack:$LAP_S" "GPU_ctrl:$GPU_C" "GPU_stack:$GPU_S"; do
    lbl=${name%%:*}; p=${name#*:}
    if [ -f "$p" ]; then echo "  [x] $lbl  ($(date -r "$p" '+%H:%M:%S'))"; else echo "  [ ] $lbl"; fi
  done
  echo "  chain-log tail:"; tail -n 4 _dataset/wps1_chain.log 2>/dev/null | sed 's/^/    /'
  echo "  live trainers:"; wmic process where "name='python.exe'" get processid,commandline 2>/dev/null | grep -o "GPU_V4/apex[^ ]*stack\|GPU_V4/apex[^ ]*ctrl\|Laptop/apex[^ ]*stack\|Laptop/apex[^ ]*ctrl" | sort | uniq -c | sed 's/^/    /'
}

for i in $(seq 1 90); do
  if [ -f "$GPU_S" ]; then echo "WPS1_ALL_DONE (GPU_V4 stack final ckpt present) i=$i"; census; exit 0; fi
  sleep 60
done
echo "WPS1_HB_CYCLE_END i=90"; census
