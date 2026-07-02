@echo off
REM === Apex Reality: AOT-build gsplat 1.5.3 for Blackwell sm_120 ===
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=%CUDA_HOME%"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "DISTUTILS_USE_SDK=1"
set "MAX_JOBS=16"
set "PY=C:\Users\gevor\AppData\Local\Programs\Python\Python311\python.exe"
set "CONSTRAINTS=C:\Users\gevor\AppData\Local\Temp\claude\D--REPOS-HYPERclone\b3f13a5a-7d67-4cbe-8a6a-d862db9a7169\scratchpad\constraints.txt"

echo === Activating VS2022 x64 dev environment (MSVC 14.44, no toolset pin needed on CUDA 12.9) ===
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 ( echo VCVARS FAILED & exit /b 1 )

echo === nvcc version ===
"%CUDA_HOME%\bin\nvcc.exe" --version
if errorlevel 1 ( echo NVCC NOT FOUND & exit /b 1 )

echo === Building gsplat v1.5.3 AOT for sm_120 (this takes ~10-20 min) ===
"%PY%" -m pip install --force-reinstall --no-deps --no-build-isolation --no-cache-dir -c "%CONSTRAINTS%" "git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3"
if errorlevel 1 ( echo GSPLAT BUILD FAILED & exit /b 1 )

echo === Verifying AOT compile (from gsplat import csrc) ===
"%PY%" -c "import gsplat; print('gsplat', gsplat.__version__); from gsplat import csrc; print('csrc OK (AOT compiled)')"
if errorlevel 1 ( echo GSPLAT CSRC IMPORT FAILED & exit /b 1 )
echo === GSPLAT BUILD COMPLETE ===
