#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
CONSTR=/mnt/c/Users/gevor/AppData/Local/Temp/claude/D--REPOS-HYPERclone/b3f13a5a-7d67-4cbe-8a6a-d862db9a7169/scratchpad/constraints.txt
SP=$(python -c "import site; print(site.getsitepackages()[0])")
echo "=== install flash_attn skeleton + deps WITH --no-deps + constraints (NO torch downgrade) ==="
pip install --no-deps --force-reinstall -c "$CONSTR" "flash-attn-4==4.0.0b19" 2>&1 | tail -1
pip install --no-deps --force-reinstall "nvidia-cutlass-dsl==4.5.2" apache-tvm-ffi quack-kernels 2>&1 | tail -1
python -c "import torch; print('torch', torch.__version__, '(must be 2.11)')"
echo "=== overwrite flash_attn/cute with fork + PATCH cute.core.ThrMma -> cute.ThrMma ==="
cp -f ~/fa4sm120/build/torch-cuda/*.py "$SP/flash_attn/cute/"
sed -i 's/cute\.core\.ThrMma/cute.ThrMma/g' "$SP/flash_attn/cute/"*.py
echo "remaining cute.core.ThrMma refs: $(grep -rho 'cute\.core\.ThrMma' "$SP/flash_attn/cute/" | wc -l) (want 0)"
echo "=== DECISIVE: import + block-sparse flex FLASH on sm_120, head_dim=128 ==="
python - <<'PY'
import torch, time
import flash_attn.cute.interface as I
print("fork interface imported OK; has flash_attn_func:", hasattr(I,"flash_attn_func"))
from torch.nn.attention import activate_flash_attention_impl
try: activate_flash_attention_impl("FA4"); print("FA4 activated")
except Exception as e: print("activate note:", str(e)[:90])
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
B,H,S,D = 1, 8, 1024, 128
q=torch.randn(B,H,S,D,device='cuda',dtype=torch.bfloat16); k=torch.randn_like(q); v=torch.randn_like(q)
def causal(b,h,qi,ki): return qi>=ki
bm=create_block_mask(causal,B,H,S,S,device='cuda',BLOCK_SIZE=(256,128))
fa=torch.compile(lambda q,k,v,bm: flex_attention(q,k,v,block_mask=bm,kernel_options={"BACKEND":"FLASH"}))
t=time.time(); out=fa(q,k,v,bm); torch.cuda.synchronize()
print(f"compile+run {time.time()-t:.1f}s shape {tuple(out.shape)} finite {bool(torch.isfinite(out).all())}")
print("FORK_FLEX_WORKS_ON_SM120")
PY
echo "THRMMA_FIX_DONE rc=$?"
