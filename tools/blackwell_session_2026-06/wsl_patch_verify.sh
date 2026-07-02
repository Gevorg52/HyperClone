#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
python /mnt/c/Users/gevor/AppData/Local/Temp/claude/D--REPOS-HYPERclone/b3f13a5a-7d67-4cbe-8a6a-d862db9a7169/scratchpad/patch_flex.py
echo "--- verify import + resolved config on the card ---"
cd ~/artifixer
PYTHONPATH=~/artifixer python - <<'PY'
from model_training.net.transformer import _DEFAULT_ATTENTION_BACKEND, _FLEX_KERNEL_OPTIONS, _FLEX_BLOCK_SIZE
print("resolved_backend =", _DEFAULT_ATTENTION_BACKEND)
print("flex_kernel_opts =", _FLEX_KERNEL_OPTIONS)
print("flex_block_size  =", _FLEX_BLOCK_SIZE)
assert _FLEX_KERNEL_OPTIONS.get("BACKEND") != "FLASH", "FLASH still set without FA4!"
print("OK: flex stays on Triton (no FA4) -> inference path safe on sm_120")
PY
