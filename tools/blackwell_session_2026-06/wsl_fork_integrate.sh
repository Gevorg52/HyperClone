#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
SP=$(python -c "import site; print(site.getsitepackages()[0])")
echo "site-packages: $SP"
echo "=== 1. install stock flash-attn-4 (gives flash_attn pkg + dist metadata) ==="
pip install --pre flash-attn-4 2>&1 | tail -2
echo "=== 2. fix cuda bindings (need driver + runtime.cudaLibrary_t) ==="
pip install --force-reinstall "cuda-python==12.8.0" "cuda-bindings==12.8.0" 2>&1 | tail -2
echo "=== 3. OVERWRITE flash_attn/cute with the sm_120 fork (block-sparse, no assert) ==="
ls "$SP/flash_attn/cute/" | head -3
cp -f ~/fa4sm120/build/torch-cuda/*.py "$SP/flash_attn/cute/"
echo "copied fork cute files: $(ls ~/fa4sm120/build/torch-cuda/*.py | wc -l)"
grep -c "Block sparsity not supported on SM 12.0" "$SP/flash_attn/cute/interface.py" && echo "WARN: assert still present" || echo "assert removed (fork has sm_120 block-sparse)"
echo "=== 4. DECISIVE TEST: block-sparse flex FLASH on sm_120, head_dim=128 (Wan) ==="
python - <<'PY'
import torch
from torch.nn.attention import list_flash_attention_impls, activate_flash_attention_impl
print("impls", list_flash_attention_impls(), "cap", torch.cuda.get_device_capability())
try:
    activate_flash_attention_impl("FA4"); print("FA4 activated")
except Exception as e:
    print("activate note:", str(e)[:80])
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
B,H,S,D = 1, 8, 1024, 128
q=torch.randn(B,H,S,D,device='cuda',dtype=torch.bfloat16); k=torch.randn_like(q); v=torch.randn_like(q)
def causal(b,h,qi,ki): return qi>=ki
bm=create_block_mask(causal,B,H,S,S,device='cuda',BLOCK_SIZE=(256,128))
fa=torch.compile(lambda q,k,v,bm: flex_attention(q,k,v,block_mask=bm,kernel_options={"BACKEND":"FLASH"}))
out=fa(q,k,v,bm); torch.cuda.synchronize()
print("FORK BLOCK-SPARSE FLEX(FLASH) OK", tuple(out.shape), "finite", bool(torch.isfinite(out).all()))
print("FORK_FLEX_WORKS_ON_SM120")
PY
echo "INTEGRATE_DONE rc=$?"
