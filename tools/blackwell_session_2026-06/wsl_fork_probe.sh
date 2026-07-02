#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HUB_DISABLE_TELEMETRY=1
echo "=== fetch the sm120 FA4 fork kernel (trust_remote_code) + inspect API ==="
python - <<'PY'
import torch
print("cap", torch.cuda.get_device_capability())
from kernels import get_kernel
k = get_kernel("SecondNatureComputing/flash-attn-4-sm120", trust_remote_code=True, revision="main")
print("kernel loaded:", type(k))
print("attrs:", [a for a in dir(k) if not a.startswith('__')])
# does it expose a flash_attn-style interface / block-sparse?
for a in ("flash_attn_func","fwd","cute","interface","flash_attn_varlen_func","block_sparse"):
    print("  has", a, "=", hasattr(k, a))
PY
echo "FORK_PROBE_DONE rc=$?"
