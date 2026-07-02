#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
SP=$(python -c "import site; print(site.getsitepackages()[0])")
pip uninstall -y flash-attn-4 nvidia-cutlass-dsl quack-kernels apache-tvm-ffi 2>&1 | grep -iE "Successfully|not installed" | tail -4
rm -rf "$SP"/flash_attn "$SP"/flash_attn_interface 2>/dev/null
echo "=== verify clean (SDPA native, no FA4 poison) ==="
cd ~/artifixer
PYTHONPATH=~/artifixer python -c "import torch; from model_training.net.transformer import _DEFAULT_ATTENTION_BACKEND,_FLEX_KERNEL_OPTIONS; print('torch',torch.__version__,'attn',_DEFAULT_ATTENTION_BACKEND,_FLEX_KERNEL_OPTIONS)"
echo "CLEAN3_DONE"
