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
ANCHORS=/mnt/d/REPOS/HYPERclone/artifixer-checkpoints/laptop_wedge_anchors.txt

echo "########## STAGE 1: prepare (264 anchors, 40-frame contiguous wedge held out) ##########"
python -m data_processing.prepare_colmap_artifixer_inputs \
  --colmap_dir ~/laptop_720 \
  --output_root ~/artifixer-prep/laptop720_wedge \
  --selected_image_names_file "$ANCHORS" \
  --reconstruction_steps 10000 --replace

echo "########## STAGE 2: bidirectional inference on the 40 wedge views ##########"
python -m model_eval.run_inference \
  --evalset reconstructed_colmap --checkpoint_pt "$CKPT" \
  --save_dir ~/artifixer-prep/laptop720_wedge_bidir \
  --split_path ~/artifixer-prep/laptop720_wedge/split.json \
  --render_trajectory val_frames \
  --inference_pipeline bidirectional --bidirectional_chunk_size 8 \
  --max_neighbors_per_encode 1 --save_frame_outputs_only

echo "########## STAGE 3: ArtiFixer3D distill ##########"
FRAMES=$(find ~/artifixer-prep/laptop720_wedge_bidir -type d -name pred | head -1)
echo "frames: $FRAMES ($(ls "$FRAMES"|wc -l))"
python -m data_processing.run_artifixer3d \
  --scene_root ~/artifixer-prep/laptop720_wedge \
  --artifixer_frames_dir "$FRAMES"
echo "WEDGE_PIPELINE_DONE rc=$?"
