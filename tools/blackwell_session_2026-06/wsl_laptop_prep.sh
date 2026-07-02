#!/usr/bin/env bash
set -e
source ~/artifixer-venv/bin/activate
cd ~/artifixer
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=~/artifixer
export HF_HUB_DISABLE_TELEMETRY=1
SCENE=/mnt/d/REPOS/HYPERclone/apex-sfm/_dataset/Laptop/train/data
ANCHORS=/mnt/d/REPOS/HYPERclone/artifixer-checkpoints/laptop_train.txt
echo "anchors: $(wc -l < $ANCHORS) (from COLMAP-registered; val = 38 held-out idx%8==0)"

echo "=== ArtiFixer FULL prepare on Laptop (10k 3DGRUT recon + render + MoGe scale + captions) ==="
python -m data_processing.prepare_colmap_artifixer_inputs \
  --colmap_dir "$SCENE" \
  --output_root ~/artifixer-prep/laptop \
  --selected_image_names_file "$ANCHORS" \
  --reconstruction_steps 10000 \
  --replace
echo "LAPTOP_PREP_DONE rc=$?"
