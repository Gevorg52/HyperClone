#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
echo "=== remove kernels lib (transformers tries to use it + fails) ==="
pip uninstall -y kernels 2>&1 | grep -iE "Successfully|not installed" | tail -2
echo "=== verify torch + model import clean ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
cd ~/artifixer
PYTHONPATH=~/artifixer python -c "from model_training.net.transformer import _DEFAULT_ATTENTION_BACKEND, _FLEX_KERNEL_OPTIONS; print('model attn resolved:', _DEFAULT_ATTENTION_BACKEND, _FLEX_KERNEL_OPTIONS, '=> clean Triton+SDPA, env working')"
echo "CLEANUP2_DONE"
