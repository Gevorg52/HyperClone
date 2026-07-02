#!/usr/bin/env bash
R3D="$HOME/artifixer-prep/laptop720/artifixer3d/recon_results/laptop720/artifixer3d/laptop720/ours_30000/renders"
SAMP="/mnt/d/REPOS/HYPERclone/artifixer-checkpoints/laptop720_bidir_sample"
mkdir -p "$SAMP/artifixer3d"
n=0
for f in "$SAMP"/gt/*.png; do
  idx=$(basename "$f" .png)
  if [ -f "$R3D/${idx}.png" ]; then
    cp "$R3D/${idx}.png" "$SAMP/artifixer3d/${idx}.png"
    n=$((n+1))
  fi
done
echo "copied $n ArtiFixer3D val renders"
ls "$SAMP/artifixer3d" | head -3
echo "total in artifixer3d: $(ls "$SAMP/artifixer3d" | wc -l)"
