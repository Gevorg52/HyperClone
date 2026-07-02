#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
echo "=== sweep BLOCK_SIZE to find sm_120 block-sparse tile (head_dim=128) ==="
python - <<'PY'
import torch, time
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
B,H,S,D = 1, 8, 1024, 128
q=torch.randn(B,H,S,D,device='cuda',dtype=torch.bfloat16); k=torch.randn_like(q); v=torch.randn_like(q)
def causal(b,h,qi,ki): return qi>=ki
for bs in [(256,64),(128,64),(64,64),(256,128)]:
    try:
        torch._dynamo.reset()
        bm=create_block_mask(causal,B,H,S,S,device='cuda',BLOCK_SIZE=bs)
        fa=torch.compile(lambda q,k,v,bm: flex_attention(q,k,v,block_mask=bm,kernel_options={"BACKEND":"FLASH"}))
        t=time.time(); out=fa(q,k,v,bm); torch.cuda.synchronize()
        print(f"  BLOCK_SIZE={bs}: OK  {time.time()-t:.1f}s  finite={bool(torch.isfinite(out).all())}  <<< WORKS")
    except Exception as e:
        print(f"  BLOCK_SIZE={bs}: FAIL {type(e).__name__}: {str(e).splitlines()[-1][:90]}")
PY
echo "SWEEP_DONE rc=$?"
