#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
S=/mnt/c/Users/gevor/AppData/Local/Temp/claude/D--REPOS-HYPERclone/b3f13a5a-7d67-4cbe-8a6a-d862db9a7169/scratchpad/export_ply.py
OUT=/mnt/d/REPOS/HYPERclone/artifixer-checkpoints/ply
mkdir -p "$OUT"
echo "=== base 3DGRUT (no ArtiFixer) ==="
python "$S" ~/artifixer-prep/laptop720/3dgrut_runs/laptop720/laptop720/ours_10000/ckpt_10000.pt "$OUT/laptop_base_3dgrut.ply"
echo "=== ArtiFixer3D (easy case, -10.8%) ==="
python "$S" ~/artifixer-prep/laptop720/artifixer3d/runs/laptop720/laptop720/ours_30000/ckpt_30000.pt "$OUT/laptop_artifixer3d_easy.ply"
ls -lah "$OUT"/*.ply | awk '{print $5, $NF}'
