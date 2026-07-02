#!/usr/bin/env bash
SRC=$(find "$HOME/artifixer-prep/laptop720_3dplus" -type d -name batch_0000 | head -1)
DST="/mnt/d/REPOS/HYPERclone/artifixer-checkpoints/laptop720_3dplus_sample"
echo "src: $SRC"
rm -rf "$DST"; mkdir -p "$DST/gt" "$DST/rendered" "$DST/pred"
cp "$SRC/gt/"*.png "$DST/gt/" 2>/dev/null
cp "$SRC/rendered/"*.png "$DST/rendered/" 2>/dev/null   # = ArtiFixer3D renders (input to 3D+)
cp "$SRC/pred/"*.png "$DST/pred/" 2>/dev/null            # = ArtiFixer3D+ output
echo "gt:$(ls "$DST/gt"|wc -l) rendered(=3D):$(ls "$DST/rendered"|wc -l) pred(=3D+):$(ls "$DST/pred"|wc -l)"
