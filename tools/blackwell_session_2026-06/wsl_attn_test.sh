#!/usr/bin/env bash
set -e
source ~/artifixer-venv/bin/activate
python - <<'PY'
import torch
print("torch", torch.__version__, "cap", torch.cuda.get_device_capability(), torch.cuda.get_device_name(0))
B,H,S,D=1,8,512,64
dt=torch.bfloat16
q=torch.randn(B,H,S,D,device='cuda',dtype=dt)
k=torch.randn(B,H,S,D,device='cuda',dtype=dt)
v=torch.randn(B,H,S,D,device='cuda',dtype=dt)

# 1) SDPA (the Blackwell dispatch fallback ArtiFixer uses: _select_attention_config returns None -> auto SDPA)
o=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
torch.cuda.synchronize()
print("SDPA OK", tuple(o.shape), o.dtype, "finite", bool(torch.isfinite(o).all()))

# Which SDPA backends are usable on sm_120?
from torch.nn.attention import sdpa_kernel, SDPBackend
for name,b in [("FLASH",SDPBackend.FLASH_ATTENTION),("CUDNN",SDPBackend.CUDNN_ATTENTION),("MEM_EFF",SDPBackend.EFFICIENT_ATTENTION),("MATH",SDPBackend.MATH)]:
    try:
        with sdpa_kernel(b):
            oo=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
            torch.cuda.synchronize()
        print(f"  SDPA[{name}] OK")
    except Exception as e:
        print(f"  SDPA[{name}] FAIL: {type(e).__name__}: {str(e)[:80]}")

# 2) flex_attention with Triton backend (the Blackwell flex fallback when FA4 absent)
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
def causal(b,h,qi,ki): return qi>=ki
bm=create_block_mask(causal,B,H,S,S,device='cuda')
fa=torch.compile(flex_attention)
of=fa(q,k,v,block_mask=bm)
torch.cuda.synchronize()
print("flex_attention(Triton, compiled) OK", tuple(of.shape), "finite", bool(torch.isfinite(of).all()))
print("ALL_FALLBACK_ATTN_OK")
PY
