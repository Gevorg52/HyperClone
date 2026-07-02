#!/usr/bin/env bash
set -e
source ~/artifixer-venv/bin/activate
cd ~/artifixer
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=~/artifixer
export HF_HUB_DISABLE_TELEMETRY=1
echo "=== ArtiFixer run_inference on 720p Laptop held-out (val_frames) ==="
python -m model_eval.run_inference \
  --evalset reconstructed_colmap \
  --checkpoint_pt /mnt/d/REPOS/HYPERclone/artifixer-checkpoints/artifixer-14b.pt \
  --save_dir ~/artifixer-prep/laptop720_artifixer_out \
  --split_path ~/artifixer-prep/laptop720/split.json \
  --render_trajectory val_frames \
  --max_neighbors_per_encode 1 \
  --save_frame_outputs_only
echo "ARTIFIXER720_INFERENCE_DONE rc=$?"
