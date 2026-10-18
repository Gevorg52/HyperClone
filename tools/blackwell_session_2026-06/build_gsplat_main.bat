@echo off
REM === Apex Reality WP-B2: AOT-build gsplat MAIN (1.6.0 unreleased) for sm_120, ISOLATED venv ===
REM Pinned commit: 512d366b67073d77ca099ede742683c165dfc23b (2026-09-19, "[perf] 3DGS training ... #1062")
REM Source clone : D:\REPOS\HYPERclone\gsplat_main   (git clone --recursive; glm + googletest submodules)
REM Target venv  : D:\REPOS\HYPERclone\venvs\gsplat_main  (--system-site-packages -> inherits torch 2.11.0+cu128)
REM The SYSTEM-site gsplat 1.5.3 is NEVER touched: --ignore-installed + --no-deps install into the venv only.
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=%CUDA_HOME%"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "DISTUTILS_USE_SDK=1"
set "MAX_JOBS=12"
REM Skip the experimental HiGS inference extension (Linux-only code: GCC builtins in its Utils.h; unused here)
set "BUILD_EXPERIMENTAL=0"
set "PY=D:\REPOS\HYPERclone\venvs\gsplat_main\Scripts\python.exe"
set "SRC=D:\REPOS\HYPERclone\gsplat_main"
REM CUDA 12.9 / CCCL 2.8.2 Windows bug: cuda/__ptx/.../clusterlaunchcontrol.h binds 'long2' (32-bit long
REM on LLP64 Windows) to a 64-bit "l" asm constraint -> "asm operand type size(4) does not match" in every
REM sm_100+/sm_120 TU that pulls <cuda/ptx> (cub.cuh). The toolkit is NOT modified: a patched copy of that
REM ONE header (long2 -> longlong2, the upstream CCCL fix) is put first on nvcc's include path.
set "NVCC_PREPEND_FLAGS=-ID:\REPOS\HYPERclone\apex-sfm\tools\patches\cuda129_win_overlay"
set "CONSTRAINTS=D:\REPOS\HYPERclone\apex-sfm\tools\blackwell_session_2026-06\constraints.txt"

echo === Activating VS2022 x64 dev environment ===
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 ( echo VCVARS FAILED & exit /b 1 )

"%CUDA_HOME%\bin\nvcc.exe" --version
if errorlevel 1 ( echo NVCC NOT FOUND & exit /b 1 )

cd /d "%SRC%"
git rev-parse HEAD

echo === Building gsplat main AOT for sm_120 into the venv ===
"%PY%" -m pip install -v --no-deps --ignore-installed --no-build-isolation --no-cache-dir -c "%CONSTRAINTS%" "%SRC%"
if errorlevel 1 ( echo GSPLAT BUILD FAILED & exit /b 1 )

echo === Verifying AOT compile ===
cd /d D:\
"%PY%" -c "import gsplat, torch; print('gsplat', gsplat.__version__, gsplat.__file__); from gsplat import csrc; print('csrc OK', csrc.__file__); print('torch', torch.__version__)"
if errorlevel 1 ( echo GSPLAT CSRC IMPORT FAILED & exit /b 1 )
echo === GSPLAT MAIN BUILD COMPLETE ===
