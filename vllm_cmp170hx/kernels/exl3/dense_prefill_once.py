# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lab (default off): decode each dense EXL3 weight once per prefill call.

VLLM_EXL3_DENSE_PREFILL_ONCE = "i8" | "f16" (unset: off). Requires the dense
tensor-core prefill gate (VLLM_EXL3_DENSE_PREFILL_TC) to be open; covers the
same validated matrices and the same prefill-step launches, at
VLLM_EXL3_DENSE_PREFILL_ONCE_MIN_ROWS (default 512) rows or more.

EXL3 linear: out = svh * H128_out(H128_in(x * suh) @ W~). The output side is
folded into the weight, W' = W~ H128_out diag(svh), rebuilt from the trellis
per call in column chunks of at most VLLM_EXL3_DENSE_PREFILL_ONCE_CHUNK_MB
(default 32) MB:
- i8: W' int8 with one scale per output column (column max over K, computed
  once per matrix on first use), xin = H128_in(x * suh) int8 per token, then
  an int8 x int8 -> bf16 CUTLASS GEMM with the per-token x per-channel
  epilogue (cmp_i8mm.scaled_mm_out, 128x256x64, 3 stages, swizzle 8).
- f16: W' and xin in FP16, cuBLAS, cast to bf16.
Numerics differ from linear_out_prefill (lab; TC admission policy applies).

The kernels live in two lab libraries loaded from VLLM_EXL3_ONCE_LIB_DIR
(default /opt/exl3-once): exl3_i8_lab.so (colmax, rebuild_int8/fp16,
act_int8/fp16) and, for i8, cmp_i8mm.so.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

MODE = os.environ.get("VLLM_EXL3_DENSE_PREFILL_ONCE", "").strip().lower()
LIB_DIR = os.environ.get("VLLM_EXL3_ONCE_LIB_DIR", "/opt/exl3-once")
MIN_ROWS = int(os.environ.get("VLLM_EXL3_DENSE_PREFILL_ONCE_MIN_ROWS", "512"))
CHUNK_BYTES = int(os.environ.get("VLLM_EXL3_DENSE_PREFILL_ONCE_CHUNK_MB", "32")) << 20
# Activation buffer: max_num_batched_tokens (2048) x the widest covered K (16384).
MAX_ROWS = int(os.environ.get("VLLM_EXL3_DENSE_PREFILL_ONCE_MAX_ROWS", "2048"))
MAX_K = 16384

_ops = None
_mm = None
_scales: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_buffers: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_logged: set[tuple[int, int]] = set()


def enabled() -> bool:
    return MODE in ("i8", "f16")


def load() -> None:
    """Load the lab libraries (called once when the prefill gate opens)."""
    global _ops, _mm
    if _ops is not None or not enabled():
        return
    torch.ops.load_library(os.path.join(LIB_DIR, "exl3_i8_lab.so"))
    _ops = torch.ops.exl3_i8_lab
    if MODE == "i8":
        torch.ops.load_library(os.path.join(LIB_DIR, "cmp_i8mm.so"))
        _mm = torch.ops.cmp_i8mm.scaled_mm_out
    logger.info_once(
        "EXL3 dense prefill decode-once lab path: mode %s, launches of >= %d rows "
        "on the covered matrices, weight chunks of %d MB",
        MODE,
        MIN_ROWS,
        CHUNK_BYTES >> 20,
        scope="process",
    )


# The shared-expert aux stream only runs the shared-expert projections (K <= 4096,
# N x K <= 2048 x 4096), so its buffers are sized for those.
AUX_MAX_K = 4096
AUX_CHUNK_BYTES = 16 << 20


def _stream_buffers(device: torch.device, *, max_k: int = MAX_K,
                    chunk_bytes: int = CHUNK_BYTES, create: bool = False):
    """Per-stream weight chunk, activation and per-token scale buffers (the aux
    stream overlaps the compute stream, so each has its own). Created only by
    preallocate, before memory profiling; None for a stream without buffers."""
    key = torch.cuda.current_stream(device).cuda_stream
    bufs = _buffers.get(key)
    if bufs is None and create:
        act_bytes = MAX_ROWS * max_k * (1 if MODE == "i8" else 2)
        bufs = (
            torch.empty(chunk_bytes, dtype=torch.int8, device=device),
            torch.empty(act_bytes, dtype=torch.int8, device=device),
            torch.empty(MAX_ROWS, dtype=torch.float32, device=device),
        )
        _buffers[key] = bufs
    return bufs


def preallocate(device: torch.device) -> None:
    """Allocate the compute and shared-expert aux streams' buffers before
    memory profiling, so the KV cache is sized with them in place."""
    if not enabled():
        return
    load()
    _stream_buffers(device, create=True)
    try:
        from vllm.utils.torch_utils import aux_stream

        aux = aux_stream()
    except Exception:  # noqa: BLE001 - no aux stream in this build
        aux = None
    if aux is not None:
        with torch.cuda.stream(aux):
            _stream_buffers(device, max_k=AUX_MAX_K, chunk_bytes=AUX_CHUNK_BYTES,
                            create=True)


def takes(rows: int, in_features: int, trellis: torch.Tensor, logical_out: int) -> bool:
    """Shape-side conditions; the caller has already checked the phase gate."""
    padded_in = trellis.shape[0] * 16
    padded_out = trellis.shape[1] * 16
    return (
        _ops is not None
        and MIN_ROWS <= rows <= MAX_ROWS
        and in_features == padded_in <= MAX_K
        and logical_out == padded_out
        and trellis.data_ptr() % 16 == 0
    )


def _column_scales(trellis, svh, bits_x2, codebook):
    key = trellis.data_ptr()
    sc = _scales.get(key)
    if sc is None:
        cm = _ops.colmax(trellis, svh, bits_x2, codebook)
        sc = (
            (cm / 127.0).contiguous(),
            torch.where(cm > 0, 127.0 / cm, torch.zeros_like(cm)).contiguous(),
        )
        _scales[key] = sc
    return sc


def linear(x, trellis, suh, svh, bits_x2: int, codebook: int, logical_out: int):
    """The decode-once result, or None when this stream's buffers cannot hold
    the call (the caller then runs the shipped path)."""
    rows, k = x.shape
    n = trellis.shape[1] * 16
    bufs = _stream_buffers(x.device)
    elem = 1 if MODE == "i8" else 2
    if bufs is None or rows * k * elem > bufs[1].numel():
        return None
    wbuf, abuf, sbuf = bufs
    chunk = (wbuf.numel() // (k * elem)) // 128 * 128
    if chunk < 128:
        return None
    out = torch.empty((rows, n), dtype=x.dtype, device=x.device)
    if MODE == "i8":
        sw, inv = _column_scales(trellis, svh, bits_x2, codebook)
        xq = abuf[: rows * k].view(rows, k)
        sx = sbuf[:rows]
        _ops.act_int8(x, suh, xq, sx)
        for n0 in range(0, n, chunk):
            nc = min(chunk, n - n0)
            w8 = wbuf[: nc * k].view(nc, k)
            _ops.rebuild_int8(trellis, svh, inv, bits_x2, codebook, n0, w8)
            _mm(out[:, n0 : n0 + nc], xq, w8, sx, sw[n0 : n0 + nc])
    else:
        xr = abuf[: rows * k * 2].view(torch.float16).view(rows, k)
        _ops.act_fp16(x, suh, xr)
        for n0 in range(0, n, chunk):
            nc = min(chunk, n - n0)
            w16 = wbuf[: nc * k * 2].view(torch.float16).view(nc, k)
            _ops.rebuild_fp16(trellis, svh, bits_x2, codebook, n0, w16)
            out[:, n0 : n0 + nc].copy_(torch.mm(xr, w16.t()))
    shape = (k, n)
    if shape not in _logged:
        _logged.add(shape)
        logger.info(
            "EXL3 dense prefill decode-once (%s) dispatched: rows=%d, in=%d, out=%d",
            MODE,
            rows,
            k,
            n,
        )
    return out
