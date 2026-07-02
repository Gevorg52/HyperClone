#!/usr/bin/env bash
set -e
source ~/artifixer-venv/bin/activate
cd ~/artifixer
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=~/artifixer
export HF_HUB_DISABLE_TELEMETRY=1
ANCHORS=/mnt/d/REPOS/HYPERclone/artifixer-checkpoints/laptop_train.txt
echo "=== ArtiFixer prepare on 720p Laptop scene (recon+render+scale+caption) ==="
python -m data_processing.prepare_colmap_artifixer_inputs \
  --colmap_dir ~/laptop_720 \
  --output_root ~/artifixer-prep/laptop720 \
  --selected_image_names_file "$ANCHORS" \
  --reconstruction_steps 10000 \
  --replace
echo "LAPTOP720_PREP_DONE rc=$?"
