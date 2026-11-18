# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host file-read helpers of the native table readers (the DeepSeek-V4.1
Engram SSD reader, ``models/deepseek_v41/sm80_engram.py``).

Carried from vllm-sm80 plugin-dev b2c9487dd5 ``models/qwen4_exp/exl3_ple.py``
(the Qwen4Exp PLE path is not part of this plugin).
"""

import ctypes
import functools
import mmap
import os

import numpy as np

_RESIDENCY_WINDOW = 1 << 30


@functools.cache
def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = (
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int64,
    )
    libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
    return libc


def page_cache_residency(fd: int, start: int, length: int) -> tuple[int, int]:
    """Return (resident, total) page-cache pages of a file extent.

    Maps the extent without touching it and asks mincore, so sampling does
    not change residency.
    """
    libc = _libc()
    page = mmap.PAGESIZE
    first = start // page * page
    end = start + length
    resident = 0
    for window in range(first, end, _RESIDENCY_WINDOW):
        size = min(_RESIDENCY_WINDOW, end - window)
        address = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, window)
        if address is None or address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "mmap of the table file failed")
        try:
            vector = (ctypes.c_ubyte * ((size + page - 1) // page))()
            if libc.mincore(address, size, vector) != 0:
                raise OSError(ctypes.get_errno(), "mincore of the table file failed")
            resident += int((np.frombuffer(vector, dtype=np.uint8) & 1).sum())
        finally:
            libc.munmap(address, size)
    return resident, (end - first + page - 1) // page


def _read_error(code: int, offset: int) -> BaseException:
    """The error a native reader's reported failure stands for."""
    if code == 0:
        return EOFError(f"Truncated table data at byte {offset}")
    return OSError(code, os.strerror(code))
