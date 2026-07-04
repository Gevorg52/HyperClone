@echo off
REM === WP-G1: AOT-build 3DGS-DR CUDA submodules for Blackwell sm_120 ===
REM House recipe: VS2022 vcvars64 + CUDA v12.9 + TORCH_CUDA_ARCH_LIST=12.0 +
REM DISTUTILS_USE_SDK=1 + --no-build-isolation + constraints (never touch torch 2.11).
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set "CUDA_PATH=%CUDA_HOME%"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "DISTUTILS_USE_SDK=1"
set "MAX_JOBS=16"
set "PY=C:\Users\gevor\AppData\Local\Programs\Python\Python311\python.exe"
set "CONSTRAINTS=D:\REPOS\HYPERclone\apex-sfm\tools\blackwell_session_2026-06\constraints.txt"
set "REPO=D:\REPOS\HYPERclone\3DGS-DR"

echo === Activating VS2022 x64 dev environment ===
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 ( echo VCVARS FAILED & exit /b 1 )

echo === nvcc version ===
"%CUDA_HOME%\bin\nvcc.exe" --version
if errorlevel 1 ( echo NVCC NOT FOUND & exit /b 1 )

echo === Python dep: plyfile ===
"%PY%" -m pip install --no-build-isolation --no-deps -c "%CONSTRAINTS%" plyfile
if errorlevel 1 ( echo PLYFILE FAILED & exit /b 1 )

for %%M in (simple-knn diff-gaussian-rasterization_c3 diff-gaussian-rasterization_c7 cubemapencoder) do (
  echo === Building %%M ===
  "%PY%" -m pip install --force-reinstall --no-build-isolation --no-deps --no-cache-dir -c "%CONSTRAINTS%" "%REPO%\submodules\%%M"
  if errorlevel 1 ( echo BUILD %%M FAILED & exit /b 1 )
)

echo === Verifying imports ===
"%PY%" -c "import torch; import simple_knn._C; import diff_gaussian_rasterization_c3; import diff_gaussian_rasterization_c7; import cubemapencoder; print('ALL 3DGS-DR SUBMODULE IMPORTS OK')"
if errorlevel 1 ( echo IMPORT VERIFY FAILED & exit /b 1 )
echo === WPG1 BUILD COMPLETE ===
