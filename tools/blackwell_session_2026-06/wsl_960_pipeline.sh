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
ANCHORS=/mnt/d/REPOS/HYPERclone/artifixer-checkpoints/laptop_train.txt   # 266 anchors, every-8th held out (SAME as 720p easy case)

echo "########## STAGE 1: prepare @960p (266 anchors, 38 every-8th held out) ##########"
python -m data_processing.prepare_colmap_artifixer_inputs \
  --colmap_dir ~/laptop_960 --output_root ~/artifixer-prep/laptop960 \
  --selected_image_names_file "$ANCHORS" --reconstruction_steps 10000 --replace

echo "########## STAGE 2: bidirectional inference @960p ##########"
python -m model_eval.run_inference \
  --evalset reconstructed_colmap --checkpoint_pt "$CKPT" \
  --save_dir ~/artifixer-prep/laptop960_bidir \
  --split_path ~/artifixer-prep/laptop960/split.json \
  --render_trajectory val_frames \
  --inference_pipeline bidirectional --bidirectional_chunk_size 8 \
  --max_neighbors_per_encode 1 --save_frame_outputs_only

echo "########## STAGE 3: ArtiFixer3D distill @960p ##########"
FRAMES=$(find ~/artifixer-prep/laptop960_bidir -type d -name pred | head -1)
echo "frames: $FRAMES ($(ls "$FRAMES"|wc -l))"
python -m data_processing.run_artifixer3d \
  --scene_root ~/artifixer-prep/laptop960 --artifixer_frames_dir "$FRAMES"
echo "PIPELINE960_DONE rc=$?"
