#!/usr/bin/env bash
set +e
echo "user=$(whoami)"
echo "--- passwordless sudo? ---"
if sudo -n true 2>/dev/null; then echo "SUDO_NOPASSWD_OK"; else echo "SUDO_NEEDS_PASSWORD"; fi
echo "--- python ---"
python3 --version
python3 -c 'import venv; print("venv module OK")' 2>&1
which pip3 2>/dev/null || echo "no pip3"
echo "--- nvcc / cuda ---"
ls -d /usr/local/cuda*/bin/nvcc 2>/dev/null || echo "no nvcc binary found"
/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -2 || echo "nvcc not runnable"
echo "--- gcc/g++ ---"
gcc --version 2>/dev/null | head -1 || echo "no gcc"
echo "--- disk (ext4 root) ---"
df -h / | tail -1
echo "--- nvidia-smi in wsl ---"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
