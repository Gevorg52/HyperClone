#!/usr/bin/env bash
set -e
source ~/artifixer-venv/bin/activate
cd ~/artifixer
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH=~/artifixer
SCENE=/mnt/d/REPOS/HYPERclone/apex-sfm/_dataset/GPU_V4/apex_images/export

echo "=== generate train-anchor subset (all but every 8th image -> val non-empty) ==="
ls "$SCENE/images" | grep -iE '\.(jpg|jpeg|png)$' | awk 'NR%8!=0' > ~/selected_train.txt
echo "train anchors: $(wc -l < ~/selected_train.txt)  (val = remainder)"

echo "=== 3DGRUT sm_120 smoke: prepare + 200-iter recon ==="
python -m data_processing.prepare_colmap_artifixer_inputs \
  --colmap_dir "$SCENE" \
  --output_root ~/artifixer-prep/gpu_v4_smoke \
  --selected_image_names_file ~/selected_train.txt \
  --phases prepare,reconstruct \
  --reconstruction_steps 200 \
  --replace
echo "ARTIFIXER_3DGRUT_SMOKE_DONE rc=$?"
