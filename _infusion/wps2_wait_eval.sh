#!/usr/bin/env bash
# Robust wait-then-eval: block until arm-c ckpt exists AND its trainer has exited
# (harness severs job tracking ~hourly but orphans complete), then run the full eval.
set -u
cd "$(dirname "$0")/.."
PY="C:/Users/gevor/AppData/Local/Programs/Python/Python311/python.exe"
T=_dataset/GPU_V4/apex_images/training
CKC=$T/wps2/c_obj_alpha/output/ckpts/ckpt_14999_rank0.pt

echo "[wait_eval] waiting for arm-c ckpt + trainer exit ..."
while true; do
  if [ -f "$CKC" ]; then
    running=$("$PY" -c "
import psutil
n=0
for p in psutil.process_iter(['cmdline']):
    cl=p.info['cmdline'] or []
    s=' '.join(cl)
    if 'gsplat_trainer' in s and 'c_obj_alpha' in s and len(cl)>10:
        n+=1
print(n)
" 2>/dev/null)
    if [ "$running" = "0" ]; then echo "[wait_eval] arm-c done"; break; fi
  fi
  sleep 30
done

echo "[wait_eval] launching eval pipeline"
bash _infusion/wps2_eval.sh 2>&1 | tee _infusion/wps2_eval.log
echo "WPS2_WAIT_EVAL_DONE"
