#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
for V in 4.4.2 4.4.1 4.4.0; do
  echo "=== try nvidia-cutlass-dsl==$V ==="
  pip install --force-reinstall --no-deps "nvidia-cutlass-dsl==$V" 2>&1 | tail -1
  if python -c "import cutlass.cute as c; assert hasattr(c.core,'ThrMma')" 2>/dev/null; then
    echo "  -> $V HAS cute.core.ThrMma  ✓"; CHOSEN=$V; break
  else
    echo "  -> $V does NOT have cute.core.ThrMma"
  fi
done
echo "chosen: ${CHOSEN:-NONE}"
[ -z "$CHOSEN" ] && { echo "no version has cute.core.ThrMma"; exit 2; }
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
echo "=== DECISIVE TEST: block-sparse flex FLASH on sm_120, head_dim=128 ==="
python - <<'PY'
import torch, time
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
print(f"compile+run {time.time()-t:.1f}s", "shape", tuple(out.shape), "finite", bool(torch.isfinite(out).all()))
print("FORK_FLEX_WORKS_ON_SM120")
PY
echo "CUTLASS442_DONE rc=$?"
