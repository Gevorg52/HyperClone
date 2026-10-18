@echo off
REM === Apex Reality WP-F5: AOT-build PPISP (nv-tlabs/ppisp, Apache-2.0) for sm_120 into the gsplat_main venv ===
REM Pinned commit: a247368047651db2e03e6f97a36e4f5bebbe5c5f (2026-09-23; v1.3.0 + 5 fixes incl. misaligned
REM pixel_coords copy). Source clone: D:\REPOS\HYPERclone\ppisp. Installed into the ISOLATED venv only.
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=%CUDA_HOME%"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "DISTUTILS_USE_SDK=1"
set "MAX_JOBS=8"
set "NVCC_PREPEND_FLAGS=-ID:\REPOS\HYPERclone\apex-sfm\tools\patches\cuda129_win_overlay"
set "PY=D:\REPOS\HYPERclone\venvs\gsplat_main\Scripts\python.exe"
set "SRC=D:\REPOS\HYPERclone\ppisp"
set "CONSTRAINTS=D:\REPOS\HYPERclone\apex-sfm\tools\blackwell_session_2026-06\constraints.txt"

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 ( echo VCVARS FAILED & exit /b 1 )
cd /d "%SRC%"
git rev-parse HEAD
"%PY%" -m pip install -v --no-deps --ignore-installed --no-build-isolation --no-cache-dir -c "%CONSTRAINTS%" "%SRC%"
if errorlevel 1 ( echo PPISP BUILD FAILED & exit /b 1 )
cd /d D:\
"%PY%" -c "import ppisp, torch; print('ppisp', ppisp.__file__); from ppisp import PPISP; m = PPISP(num_cameras=1, num_frames=2); x = torch.rand(8, 8, 3, device='cuda', requires_grad=True); y = m(x, resolution=(8, 8), camera_idx=0, frame_idx=1); y.sum().backward(); print('ppisp fwd/bwd OK', tuple(y.shape))"
if errorlevel 1 ( echo PPISP IMPORT FAILED & exit /b 1 )
echo === PPISP BUILD COMPLETE ===
