#!/usr/bin/env bash
set -e
source ~/artifixer-venv/bin/activate
cd ~/artifixer/thirdparty/3DGRUT-ArtiFixer
export TORCH_CUDA_ARCH_LIST="12.0"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
CONSTR="/mnt/c/Users/gevor/AppData/Local/Temp/claude/D--REPOS-HYPERclone/b3f13a5a-7d67-4cbe-8a6a-d862db9a7169/scratchpad/constraints.txt"

echo "=== nvcc ==="; nvcc --version | tail -2
echo "=== ensure slangc (idempotent) ==="
sudo bash scripts/install_slangc.sh /usr/local >/dev/null 2>&1 || true
slangc -version 2>/dev/null || true

echo "=== build deps present in venv (for --no-build-isolation) ==="
pip install -c "$CONSTR" ninja wheel "setuptools<72.1.0" pybind11 cmake

echo "=== 3DGRUT requirements WITH --no-build-isolation (fused-ssim sees torch) ==="
pip install -c "$CONSTR" --no-build-isolation -r requirements.txt

echo "=== build/install 3DGRUT (-e, compiles CUDA for sm_120) ==="
pip install -c "$CONSTR" --no-build-isolation -e .

echo "=== import check ==="
python -c "import threedgrut; print('threedgrut import OK')" 2>&1 | tail -8 || true
echo "3DGRUT_BUILD_DONE rc=$?"
