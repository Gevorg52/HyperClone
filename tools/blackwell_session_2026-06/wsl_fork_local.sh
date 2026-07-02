#!/usr/bin/env bash
source ~/artifixer-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
which git-lfs >/dev/null 2>&1 || sudo apt-get install -y -q git-lfs >/dev/null 2>&1
git lfs install >/dev/null 2>&1
if [ ! -d ~/fa4sm120/.git ]; then
  echo "=== cloning fork (git-lfs) ==="
  git clone --depth 1 https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120 ~/fa4sm120 2>&1 | tail -4
fi
echo "=== top-level ==="; ls ~/fa4sm120
echo "=== build/ layout (torch-cuda variant dirs) ==="; find ~/fa4sm120/build -maxdepth 3 2>/dev/null | head -25
echo "=== kernels local-load API ==="
python - <<'PY'
import kernels, torch
print("kernels", getattr(kernels,'__version__','?'), "cap", torch.cuda.get_device_capability())
print("APIs:", [a for a in dir(kernels) if 'kernel' in a.lower()])
PY
echo "FORK_LOCAL_DONE rc=$?"
