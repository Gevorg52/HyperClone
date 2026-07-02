#!/usr/bin/env bash
set -e
source ~/artifixer-venv/bin/activate
cd ~/artifixer
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=~/artifixer
export HF_HUB_DISABLE_TELEMETRY=1
# NOTE: do NOT use PYTORCH_CUDA_ALLOC_CONF=expandable_segments on WSL2 — its CUDA VMM
# path triggers "CUDA driver error: unknown error" under WSL GPU paravirtualization.
# Control peak memory via --max_neighbors_per_encode instead.
echo "=== ArtiFixer run_inference on Laptop held-out (val_frames), memory-safe (no expandable_segments) ==="
python -m model_eval.run_inference \
  --evalset reconstructed_colmap \
  --checkpoint_pt /mnt/d/REPOS/HYPERclone/artifixer-checkpoints/artifixer-14b.pt \
  --save_dir ~/artifixer-prep/laptop_artifixer_out \
  --split_path ~/artifixer-prep/laptop/split.json \
  --render_trajectory val_frames \
  --max_neighbors_per_encode 1 \
  --save_frame_outputs_only
echo "ARTIFIXER_INFERENCE_DONE rc=$?"
