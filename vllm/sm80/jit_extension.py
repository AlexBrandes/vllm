# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""JIT builds of the plugin's CUDA extensions.

The C++ standard is torch's: ``torch.utils.cpp_extension`` already adds the
``-std`` its own headers require to both the host and the nvcc command lines,
so callers pass none. A ``-std`` from a caller would come later on the command
line and override torch's, which breaks the build as soon as torch raises its
minimum (torch 2.14's headers refuse anything below C++20).
"""

import hashlib
import os

import torch


def load_cuda_extension(
    prefix: str,
    source: str,
    build_dir: str,
    *,
    extra_cflags: list[str],
    extra_cuda_cflags: list[str],
):
    """Compile (or reuse) the CUDA ``source`` as a torch extension.

    The build name hashes the source, the flags and the torch version, so a
    cached build made with other flags or for another torch is never loaded.
    """
    for flag in (*extra_cflags, *extra_cuda_cflags):
        if flag.startswith("-std"):
            raise ValueError(
                f"{prefix}: {flag!r} would override the C++ standard torch "
                "requires; torch.utils.cpp_extension sets it")
    from torch.utils.cpp_extension import load

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    key = "\0".join(
        [source, *extra_cflags, "", *extra_cuda_cflags, "", torch.__version__])
    name = f"{prefix}_{hashlib.sha1(key.encode()).hexdigest()[:10]}"
    src = os.path.join(build_dir, f"{name}.cu")
    if not os.path.exists(src):
        tmp = src + f".{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(source)
        os.replace(tmp, src)
    return load(
        name=name,
        sources=[src],
        build_directory=build_dir,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )
