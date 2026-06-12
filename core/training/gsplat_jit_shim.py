"""Windows shim for gsplat's JIT kernel compilation.

gsplat (<= 1.5.3) passes GCC-style flags ("-O3 -Wno-attributes") to the host
C++ compiler unconditionally; on Windows cl.exe rejects them (error D8021).
This wraps torch's ``_jit_compile`` to translate the flags for MSVC.

MUST be imported before the first ``import gsplat`` in any process that may
trigger JIT compilation (gsplat binds ``_jit_compile`` at import time).
Once the compiled kernels are cached in ``%LOCALAPPDATA%/torch_extensions``,
imports load the cache and no compiler is invoked.
"""

from __future__ import annotations

import os

_MSVC_FLAG_MAP = {"-O3": "/O2", "-O0": "/Od"}
_MSVC_DROP = {"-Wno-attributes"}


def install() -> None:
    if os.name != "nt":
        return

    import torch.utils.cpp_extension as cpp_ext

    orig = cpp_ext._jit_compile
    if getattr(orig, "_apex_msvc_shim", False):
        return

    def patched(name, sources, extra_cflags, extra_cuda_cflags, *args, **kwargs):
        if extra_cflags:
            extra_cflags = [
                _MSVC_FLAG_MAP.get(f, f)
                for f in extra_cflags
                if f not in _MSVC_DROP
            ]
        return orig(name, sources, extra_cflags, extra_cuda_cflags, *args, **kwargs)

    patched._apex_msvc_shim = True
    cpp_ext._jit_compile = patched


install()
