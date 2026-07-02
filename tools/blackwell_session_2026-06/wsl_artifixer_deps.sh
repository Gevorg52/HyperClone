#!/usr/bin/env bash
set -e
source ~/artifixer-venv/bin/activate
cd ~
echo "=== clone artifixer + 3DGRUT submodule into ext4 ==="
if [ ! -d ~/artifixer/.git ]; then
  git clone --recurse-submodules https://github.com/nv-tlabs/artifixer.git ~/artifixer
else
  cd ~/artifixer && git submodule update --init --recursive && cd ~
fi
cd ~/artifixer
echo "=== ArtiFixer python deps (Dockerfile set; flash-attn OMITTED — optional on sm_120) ==="
pip install accelerate==1.13.0 diffusers==0.37.1 transformers==5.5.0 ftfy
pip install einops scipy wandb tqdm Pillow matplotlib opencv-python-headless pyyaml torchmetrics imageio-ffmpeg h5py av torch-fidelity
pip install "git+https://github.com/microsoft/MoGe.git"
echo "ARTIFIXER_DEPS_DONE"
python -c "import diffusers, transformers; print('diffusers', diffusers.__version__, 'transformers', transformers.__version__)"
echo "=== import ArtiFixer transformer (runs _select_attention_config on the real sm_120 card) ==="
PYTHONPATH=~/artifixer python -c "from model_training.net.transformer import _DEFAULT_ATTENTION_BACKEND, _FLEX_BLOCK_SIZE; print('resolved_attn_backend=', _DEFAULT_ATTENTION_BACKEND, 'flex_block=', _FLEX_BLOCK_SIZE)"
echo "ARTIFIXER_IMPORT_OK"
