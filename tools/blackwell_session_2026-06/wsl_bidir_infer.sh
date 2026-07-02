#!/usr/bin/env bash
set -e
source ~/artifixer-venv/bin/activate
cd ~/artifixer
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=~/artifixer
export HF_HUB_DISABLE_TELEMETRY=1
echo "=== ArtiFixer BIDIRECTIONAL inference (SDPA, no block-sparse) on 720p Laptop held-out ==="
python -m model_eval.run_inference \
  --evalset reconstructed_colmap \
  --checkpoint_pt /mnt/d/REPOS/HYPERclone/artifixer-checkpoints/artifixer-14b.pt \
  --save_dir ~/artifixer-prep/laptop720_bidir \
  --split_path ~/artifixer-prep/laptop720/split.json \
  --render_trajectory val_frames \
  --inference_pipeline bidirectional \
  --bidirectional_chunk_size 8 \
  --max_neighbors_per_encode 1 \
  --save_frame_outputs_only
echo "BIDIR_DONE rc=$?"
