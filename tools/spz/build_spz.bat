@echo off
REM WP-D2: build Niantic spz Python bindings (nanobind + scikit-build-core) on Windows.
REM Source: D:\REPOS\HYPERclone\spz  (github.com/nianticlabs/spz @ affd0ec, MIT, lib v1.1.0)
REM No torch involvement. Build deps installed with the constraints file:
REM   python -m pip install -c tools\blackwell_session_2026-06\constraints.txt "scikit-build-core>=0.10" "nanobind>=1.3.2"
REM --no-build-isolation skips pyproject's extra "python_devtools" build requirement (unused by the build).
REM zlib + zstd are fetched by CMake FetchContent (hash-pinned in spz's CMakeLists.txt).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PATH%"
cd /d D:\REPOS\HYPERclone\spz || exit /b 1
"C:\Users\gevor\AppData\Local\Programs\Python\Python311\python.exe" -m pip install --no-build-isolation --no-deps -v . || exit /b 1
REM spz's CMake links the FetchContent zlib as a SHARED lib (z.dll) and the wheel does not bundle
REM it -> "DLL load failed while importing spz". Ship it next to the .pyd (Python's extension
REM loader searches the module's own directory).
copy /y "build\cp311-cp311-win_amd64\_deps\zlib-build\Release\z.dll" "C:\Users\gevor\AppData\Local\Programs\Python\Python311\Lib\site-packages\spz\" || exit /b 1
"C:\Users\gevor\AppData\Local\Programs\Python\Python311\python.exe" -c "import spz; print('spz OK', spz.__file__)"
