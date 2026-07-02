#!/usr/bin/env bash
BIDIR=$(find "$HOME/artifixer-prep/laptop720_wedge_bidir" -type d -name batch_0000 | head -1)
R3D="$HOME/artifixer-prep/laptop720_wedge/artifixer3d/recon_results/laptop720_wedge/artifixer3d/laptop720_wedge/ours_30000/renders"
DST="/mnt/d/REPOS/HYPERclone/artifixer-checkpoints/laptop720_wedge_sample"
echo "bidir: $BIDIR"
echo "r3d:   $R3D (exists=$(test -d "$R3D" && echo yes), n=$(ls "$R3D" 2>/dev/null|wc -l))"
rm -rf "$DST"; mkdir -p "$DST/gt" "$DST/rendered" "$DST/pred" "$DST/artifixer3d"
cp "$BIDIR/gt/"*.png "$DST/gt/" 2>/dev/null
cp "$BIDIR/rendered/"*.png "$DST/rendered/" 2>/dev/null
cp "$BIDIR/pred/"*.png "$DST/pred/" 2>/dev/null
n=0
for f in "$DST"/gt/*.png; do
  idx=$(basename "$f" .png)
  if [ -f "$R3D/${idx}.png" ]; then cp "$R3D/${idx}.png" "$DST/artifixer3d/${idx}.png"; n=$((n+1)); fi
done
echo "gt:$(ls "$DST/gt"|wc -l) base:$(ls "$DST/rendered"|wc -l) 2D:$(ls "$DST/pred"|wc -l) 3D:$n"
