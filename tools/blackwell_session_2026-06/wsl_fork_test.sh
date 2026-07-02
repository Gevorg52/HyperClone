#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
echo "=== build.toml (package name) ==="; cat ~/fa4sm120/build.toml 2>/dev/null | head -20
echo "=== public funcs in interface.py ==="; grep -nE "^def |^    def |flash_attn_func|block_sparse|def fwd|def flash_attn" ~/fa4sm120/build/torch-cuda/interface.py 2>/dev/null | head -25
echo "=== load via get_local_kernel + inspect ==="
python - <<'PY'
from pathlib import Path
import torch
from kernels import get_local_kernel
import os
repo = Path(os.path.expanduser("~/fa4sm120"))
# package name is usually the build.toml [torch] name or the dir under build/<variant>/<name>
for pkg in ("flash_attn","flash_attn_4","flash_attn4","flashattn"):
    try:
        k = get_local_kernel(repo, pkg)
        print("LOADED with package_name=", pkg, "->", type(k))
        print("attrs:", [a for a in dir(k) if not a.startswith('_')][:40])
        break
    except Exception as e:
        print("pkg", pkg, "->", type(e).__name__, str(e)[:120])
PY
echo "FORK_TEST_DONE rc=$?"
