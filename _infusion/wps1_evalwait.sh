#!/usr/bin/env bash
# Polls for WP-S1 eval completion, then dumps every number needed for the record.
cd "$(dirname "$0")/.."
LOG=_dataset/wps1_eval_logs/eval.log
for i in $(seq 1 80); do
  grep -qa "WP-S1 EVAL DONE" "$LOG" 2>/dev/null && break
  sleep 30
done
echo "=== WPS1 EVAL WAIT (i=$i) ==="
grep -qa "WP-S1 EVAL DONE" "$LOG" 2>/dev/null && echo "STATUS: COMPLETE" || echo "STATUS: NOT-DONE-YET"
python - <<'PY'
import json
def g(d,*ks):
    for k in ks: d=d.get(k,{}) if isinstance(d,dict) else {}
    return d
print("---- SCORECARDS (masked metrics + 6 scorers) ----")
for d in ['wps1_laptop_ctrl','wps1_laptop_stack','wps1_gpu_ctrl','wps1_gpu_stack']:
    try:
        c=json.load(open(f'qa/{d}/scorecard.json')); sc=c.get('scorers',{})
        print(f"{d:20s} mLPIPS {c['masked_lpips']:.4f}  mPSNR {c['masked_psnr']:.3f}  mSSIM {c['masked_ssim']:.4f}  obs% {c.get('observed_area_pct',0):.1f}"
              f" | cross {g(sc,'crossscore','mean'):.4f} nova {g(sc,'nova','mean'):.4f} tsed {g(sc,'tsed','ratio'):.3f}"
              f" raft {g(sc,'raft_warp','mean'):.4f} nvssqa {g(sc,'nvs_sqa','jod'):.3f} flick {g(sc,'flicker','score'):.4f}")
    except Exception as e: print(f"{d:20s} MISSING ({e})")
print("---- REGION (deck / shiny crops) ----")
for r in ['wps1_laptop_region','wps1_gpu_region']:
    try:
        j=json.load(open(f'_infusion/{r}.json'))
        print(r, 'box', j['box_frac'])
        for a,v in j['arms'].items():
            print(f"   {a:6s} full_lpips {v['full_lpips']:.4f}  region_lpips {v['region_lpips']:.4f}  region_psnr {v['region_psnr']:.3f}")
        dl=j.get('delta',{})
        print('   delta', {k:(round(dl[k]['abs'],4),f"{dl[k]['pct']:+.1f}%") for k in dl})
    except Exception as e: print(f"{r} MISSING ({e})")
PY
echo "---- FLOATERS ----"
cat _infusion/wps1_laptop_floaters.txt 2>/dev/null
cat _infusion/wps1_gpu_floaters.txt 2>/dev/null
echo "---- STRIPS ----"
ls qa/wps1_laptop_strips qa/wps1_gpu_strips 2>/dev/null
