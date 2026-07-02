#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
echo "=== restore torch 2.11.0+cu128 (flash-attn-4 downgraded it to 2.9.1) ==="
pip install --force-reinstall torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
echo "=== verify torch 2.11 + sees card + has list_flash_attention_impls ==="
python -c "import torch; from torch.nn.attention import list_flash_attention_impls; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'impls', list_flash_attention_impls())"
echo "=== confirm fork cute still in place (no assert) ==="
SP=$(python -c "import site; print(site.getsitepackages()[0])")
grep -c "Block sparsity not supported on SM 12.0" "$SP/flash_attn/cute/interface.py" || echo "(0 = fork in place)"
echo "=== DECISIVE TEST: block-sparse flex FLASH on sm_120, head_dim=128 ==="
python - <<'PY'
import torch
from torch.nn.attention import activate_flash_attention_impl
try:
    activate_flash_attention_impl("FA4"); print("FA4 activated")
except Exception as e:
    print("activate note:", str(e)[:90])
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
echo "RESTORE_TEST_DONE rc=$?"
