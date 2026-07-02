#!/usr/bin/env bash
set -e
cd ~
echo "=== creating venv (python3.12) at ~/artifixer-venv ==="
python3 -m venv artifixer-venv
source artifixer-venv/bin/activate
pip install --upgrade pip wheel setuptools >/dev/null
echo "=== installing torch 2.11.0 + torchvision (cu128) ==="
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
echo "=== verifying torch on Blackwell (sm_120) ==="
python -c "import torch; print('torch', torch.__version__, 'cudabuild', torch.version.cuda, 'avail', torch.cuda.is_available()); print('arch', torch.cuda.get_arch_list()); print('dev', torch.cuda.get_device_name(0)); print('cap', torch.cuda.get_device_capability(0)); x=torch.randn(1024,1024,device='cuda'); print('matmul', float((x@x).sum().abs()>=0))"
echo "WSL_TORCH_SETUP_DONE"
