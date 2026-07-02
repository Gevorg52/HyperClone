#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
SP=$(python -c "import site; print(site.getsitepackages()[0])")
echo "=== remove fork-poisoned flash_attn + FA4 helpers (restore clean Triton+SDPA) ==="
pip uninstall -y flash-attn-4 nvidia-cutlass-dsl quack-kernels apache-tvm-ffi torch-c-dlpack-ext 2>&1 | grep -iE "Successfully|not installed" | tail -5
rm -rf "$SP"/flash_attn "$SP"/flash_attn_interface 2>/dev/null
echo "=== verify clean working state ==="
python -c "import torch; from torch.nn.attention import list_flash_attention_impls; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
cd ~/artifixer
PYTHONPATH=~/artifixer python -c "from model_training.net.transformer import _DEFAULT_ATTENTION_BACKEND, _FLEX_KERNEL_OPTIONS; print('resolved:', _DEFAULT_ATTENTION_BACKEND, _FLEX_KERNEL_OPTIONS, '=> clean Triton+SDPA')"
echo "CLEANUP_DONE"
