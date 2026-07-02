#!/usr/bin/env bash
set -e
source ~/artifixer-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
CONSTR=/mnt/c/Users/gevor/AppData/Local/Temp/claude/D--REPOS-HYPERclone/b3f13a5a-7d67-4cbe-8a6a-d862db9a7169/scratchpad/constraints.txt
echo "=== install flash-attn-4 (FA4 CuTeDSL) ==="
pip install -c "$CONSTR" --pre flash-attn-4 2>&1 | tail -6
echo "=== pin cuda-python (per Dockerfile) ==="
pip install --force-reinstall --no-deps "cuda-python==12.6.2.post1" 2>&1 | tail -3
echo "=== TEST FA4 on sm_120 ==="
python - <<'PY'
import torch
from torch.nn.attention import list_flash_attention_impls, activate_flash_attention_impl
print("cap", torch.cuda.get_device_capability(), "impls", list_flash_attention_impls())
activate_flash_attention_impl("FA4")
print("FA4 activated OK")
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
B,H,S,D = 1,8,512,64
q=torch.randn(B,H,S,D,device='cuda',dtype=torch.bfloat16)
k=torch.randn(B,H,S,D,device='cuda',dtype=torch.bfloat16)
v=torch.randn(B,H,S,D,device='cuda',dtype=torch.bfloat16)
def causal(b,h,qi,ki): return qi>=ki
bm=create_block_mask(causal,B,H,S,S,device='cuda',BLOCK_SIZE=(256,128))
fa=torch.compile(flex_attention)
out=fa(q,k,v,block_mask=bm,kernel_options={"BACKEND":"FLASH"})
torch.cuda.synchronize()
print("FA4 flex_attention(FLASH) OK", tuple(out.shape), "finite", bool(torch.isfinite(out).all()))
print("FA4_WORKS_ON_SM120")
PY
echo "FA4_TEST_DONE rc=$?"
