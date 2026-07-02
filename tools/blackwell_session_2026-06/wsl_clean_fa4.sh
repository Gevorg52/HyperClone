#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
SP=$(python -c "import site; print(site.getsitepackages()[0])")
echo "site-packages: $SP"
echo "=== remove leftover flash_attn dirs (FA4 poisons sm_120; force clean Triton+SDPA) ==="
rm -rf "$SP"/flash_attn "$SP"/flash_attn_interface "$SP"/flash_attn_3 "$SP"/flash_attn_4* 2>/dev/null
# also drop FA4 helper pkgs so nothing re-registers it
pip uninstall -y nvidia-cutlass-dsl quack-kernels apache-tvm-ffi torch-c-dlpack-ext 2>&1 | grep -iE "Successfully|not installed" | tail -4
echo "=== verify clean import + attention config ==="
cd ~/artifixer
PYTHONPATH=~/artifixer python - <<'PY'
import torch
from torch.nn.attention import list_flash_attention_impls
print("FA impls:", list_flash_attention_impls(), "(FA4 should be gone)")
from model_training.net.transformer import _DEFAULT_ATTENTION_BACKEND, _FLEX_KERNEL_OPTIONS
print("resolved backend:", _DEFAULT_ATTENTION_BACKEND, "| flex_opts:", _FLEX_KERNEL_OPTIONS, "(want None / {} => Triton+SDPA)")
print("CLEAN_OK")
PY
