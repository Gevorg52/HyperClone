#!/usr/bin/env bash
set -e
source ~/artifixer-venv/bin/activate
cd ~/artifixer
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=~/artifixer
export HF_HUB_DISABLE_TELEMETRY=1
SCENE_ROOT=~/artifixer-prep/laptop720
FRAMES=$(find ~/artifixer-prep/laptop720_bidir -type d -name pred | head -1)
echo "scene_root: $SCENE_ROOT"
echo "frames_dir: $FRAMES  ($(ls "$FRAMES"|wc -l) frames)"
echo "=== ArtiFixer3D: distill bidir pred frames into fresh 3DGRUT (sm_120 OK) ==="
python -m data_processing.run_artifixer3d \
  --scene_root "$SCENE_ROOT" \
  --artifixer_frames_dir "$FRAMES"
echo "ARTIFIXER3D_DONE rc=$?"
echo "=== outputs ==="; find "$SCENE_ROOT/artifixer3d" -maxdepth 2 -type d 2>/dev/null | head
