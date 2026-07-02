#!/usr/bin/env bash
R3D="$HOME/artifixer-prep/laptop720/artifixer3d/recon_results/laptop720/artifixer3d/laptop720/ours_30000/renders"
SAMP="/mnt/d/REPOS/HYPERclone/artifixer-checkpoints/laptop720_bidir_sample"
echo "render dir exists: $(test -d "$R3D" && echo yes) ; n=$(ls "$R3D" 2>/dev/null | wc -l)"
ls "$R3D" | head -3
mkdir -p "$SAMP/artifixer3d"
n=0
for f in "$SAMP"/gt/*.png; do
  idx=$(basename "$f" .png)          # e.g. 00008
  src="$R3D/frame_${idx}.png"
  if [ -f "$src" ]; then cp "$src" "$SAMP/artifixer3d/${idx}.png"; n=$((n+1)); fi
done
echo "copied $n matched ArtiFixer3D val renders -> $SAMP/artifixer3d/"
