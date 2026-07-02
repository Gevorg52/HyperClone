#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
echo "=== install cuda-python 12.8.0 (full bindings: driver + runtime) ==="
pip install --force-reinstall "cuda-python==12.8.0" "cuda-bindings==12.8.0" 2>&1 | tail -5
echo "=== verify BOTH submodules ==="
python -c "import cuda.bindings.runtime as r; print('cudaLibrary_t:', 'cudaLibrary_t' in dir(r)); import cuda.bindings.driver as d; print('driver: ok')"
echo "=== RETEST FA4 on sm_120 (first run JIT-compiles CuTeDSL kernels) ==="
python - <<'PY'
import torch
from torch.nn.attention import list_flash_attention_impls, activate_flash_attention_impl
print('cap', torch.cuda.get_device_capability(), 'impls', list_flash_attention_impls())
activate_flash_attention_impl('FA4'); print('FA4 activated OK')
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
B,H,S,D=1,8,512,64
q=torch.randn(B,H,S,D,device='cuda',dtype=torch.bfloat16); k=torch.randn_like(q); v=torch.randn_like(q)
def causal(b,h,qi,ki): return qi>=ki
bm=create_block_mask(causal,B,H,S,S,device='cuda',BLOCK_SIZE=(256,128))
fa=torch.compile(flex_attention)
out=fa(q,k,v,block_mask=bm,kernel_options={'BACKEND':'FLASH'}); torch.cuda.synchronize()
print('FA4 flex(FLASH) OK', tuple(out.shape), 'finite', bool(torch.isfinite(out).all()))
print('FA4_WORKS_ON_SM120')
PY
echo "FA4_CUDA128_DONE rc=$?"
