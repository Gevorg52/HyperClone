#!/usr/bin/env bash
set -e
source ~/artifixer-venv/bin/activate
cd ~/artifixer
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=~/artifixer
export HF_HUB_DISABLE_TELEMETRY=1
CKPT=/mnt/d/REPOS/HYPERclone/artifixer-checkpoints/artifixer-14b.pt
echo "=== ArtiFixer3D+ : re-apply ArtiFixer (bidir) on the ArtiFixer3D renders (easy case) ==="
ls ~/artifixer-prep/laptop720/split_artifixer3d_plus.json
python -m model_eval.run_inference \
  --evalset reconstructed_colmap --checkpoint_pt "$CKPT" \
  --save_dir ~/artifixer-prep/laptop720_3dplus \
  --split_path ~/artifixer-prep/laptop720/split_artifixer3d_plus.json \
  --render_trajectory val_frames \
  --inference_pipeline bidirectional --bidirectional_chunk_size 8 \
  --max_neighbors_per_encode 1 --save_frame_outputs_only
echo "ARTIFIXER3DPLUS_DONE rc=$?"
