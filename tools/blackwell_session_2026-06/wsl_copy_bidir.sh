#!/usr/bin/env bash
SRC=$(find ~/artifixer-prep/laptop720_bidir -type d -name batch_0000 | head -1)
DST="/mnt/d/REPOS/HYPERclone/artifixer-checkpoints/laptop720_bidir_sample"
echo "src: $SRC"
rm -rf "$DST"; mkdir -p "$DST/pred" "$DST/rendered" "$DST/gt"
cp "$SRC/pred/"*.png "$DST/pred/" 2>/dev/null
cp "$SRC/rendered/"*.png "$DST/rendered/" 2>/dev/null
cp "$SRC/gt/"*.png "$DST/gt/" 2>/dev/null
echo "pred:$(ls "$DST/pred"|wc -l) rendered:$(ls "$DST/rendered"|wc -l) gt:$(ls "$DST/gt"|wc -l)"
