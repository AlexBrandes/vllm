# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gluon thin GEMM for the Qwen3.8 decode shapes (``VLLM_THIN_GEMM_GLUON``).

``thin_gemm_gluon`` carries a schedule measured on the 74-SM sm_80 part for
the seven unquantized (N, K) of Qwen3.8-Flash-Next at tensor-parallel size 2
(``SHAPES``), at every row count up to 32: split-K Gluon kernels with the
reduction in the last-arriving CTA, a K-sliced variant, a persistent
balanced-column kernel, the Triton thin GEMM for the remaining rows and a GEMV
for N = 1. Against cuBLAS it is ~1.41x on the served mix of these calls.

This module wraps the ``VLLM_THIN_GEMM`` dispatch (``ampere_thin_gemm``): a
call goes to ``thin_gemm_gluon.thin_gemm`` when the gate is open, its (N, K)
is in ``SHAPES``, it has at most ``MAX_M`` rows and ``thin_gemm_supported``
admits it, and its operands are in the layout the warmup compiled for
(``gluon_layout_ok``: unit K stride, row stride a multiple of 16 elements,
16-byte aligned base); every other call takes the wrapped function unchanged,
except a table call in another layout, which takes cuBLAS (``F.linear``): the
kernels specialize on those properties, so a different layout would compile
a new variant at serve time, and the cp.async loads of the Gluon kernels do
not compile for a row stride that is not a multiple of 16 elements. The row
count decides, never the phase: prefill and mixed steps with at most 32 rows
run the same schedule as decode steps of that size.

The gate is decided once, before CUDA-graph capture, by ``warmup_table``
(called from ``warmup.py``): sm_80 with 74 SMs (the schedule is measured
there), compilation mode NONE (no torch.compile tracing through the Gluon
launches), CUDA graphs for every step (no ``enforce_eager``, no graph mode
that runs decode or mixed steps eagerly: those send unpadded row counts) and
the Triton Gluon ampere API importable. Until then, and whenever it stays
closed, the wrapped dispatch serves every call; a closed gate logs one warning
with the reason. Once those hold, the warmup compiles every kernel variant
and allocates its workspace; a failure there fails the boot rather than
leaving this process on a different kernel from the other ranks.

``VLLM_THIN_GEMM_CONCAT`` pairs only the shipped thin GEMM's calls, so the
two flags do not combine: setting both raises.

Arithmetic: BF16 operands, FP32 accumulation, one BF16 rounding, bitwise
reproducible run to run; the summation order differs from cuBLAS's and from
``thin_gemm_v74``'s.
"""

import functools

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.sm80.kernels.thin_gemm import impl as shipped_impl
from vllm.sm80.kernels.thin_gemm import thin_gemm_supported

logger = init_logger(__name__)

# (N, K) of the schedule table: hyper-connection down, final mixer down,
# hyper-connection up, MoE router, shared-expert gate, GDN in_proj_ba (TP2
# shard), PLE kv_proj.
SHAPES = frozenset({
    (336, 10240), (320, 10240), (10240, 320), (512, 2560), (1, 2560),
    (48, 2560), (12800, 2560),
})
# Largest row count of the table; more rows keep the wrapped dispatch.
MAX_M = 32
# The SM count the schedule was measured on.
SM_COUNT = 74
# Row counts with a schedule row (the forced-schedule diagnostic's values).
SERVED_M = (1, 2, 4, 8, 16, 24, 32)

# Set by warmup_table once every check passed and every variant compiled.
_LIVE = [False]


def set_but_off(reason: str) -> None:
    logger.warning_once(
        "VLLM_THIN_GEMM_GLUON set but off: %s; the VLLM_THIN_GEMM path serves "
        "every call", reason)


def gluon_api_missing() -> str:
    """'' when the Triton Gluon ampere API the kernels use imports, else why."""
    try:
        from triton.experimental import gluon  # noqa: F401
        from triton.experimental.gluon import language as gl
        from triton.experimental.gluon.language.nvidia.ampere import (  # noqa: F401
            async_copy,
            mma_v2,
        )
        for name in ("NVMMADistributedLayout", "DotOperandLayout",
                     "SwizzledSharedLayout", "allocate_shared_memory"):
            getattr(gl, name)
    except Exception as exc:  # noqa: BLE001 - any failure closes the gate
        return f"Triton Gluon ampere API missing ({type(exc).__name__}: {exc})"
    return ""


def force_m() -> int:
    """VLLM_THIN_GEMM_GLUON_FORCE_M if it names a served row count, else 0."""
    value = envs.VLLM_THIN_GEMM_GLUON_FORCE_M
    if value and value not in SERVED_M:
        logger.warning_once(
            "VLLM_THIN_GEMM_GLUON_FORCE_M=%d ignored: not one of %s",
            value, SERVED_M)
        return 0
    return value


@functools.cache
def impl():
    """``thin_gemm_gluon``, sharing the shipped module's workspace lane (a
    caller that selects a lane for a second stream selects it for both)."""
    from vllm.sm80.kernels.thin_gemm import thin_gemm_gluon

    thin_gemm_gluon._LANE = shipped_impl()._LANE
    thin_gemm_gluon._FORCE_M[0] = force_m()
    return thin_gemm_gluon


def closed_reason(device, vllm_config) -> str:
    """'' when this process may open the gate, else the reason it may not.

    The same on every rank of a launch (device model, config, venv)."""
    from vllm.config import CompilationMode, CUDAGraphMode

    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if sms != SM_COUNT:
        return f"{sms} SMs, the schedule is measured at {SM_COUNT}"
    compilation = vllm_config.compilation_config
    if compilation.mode != CompilationMode.NONE:
        return "compilation mode is not NONE"
    if vllm_config.model_config.enforce_eager:
        return "enforce_eager runs every step eagerly (unpadded row counts)"
    graphs = compilation.cudagraph_mode
    if graphs is None or CUDAGraphMode.NONE in (graphs.decode_mode(),
                                                graphs.mixed_mode()):
        name = getattr(graphs, "name", graphs)
        return (f"cudagraph_mode {name} runs steps eagerly (unpadded row "
                "counts)")
    return gluon_api_missing()


def warmup_table(worker, ms) -> frozenset:
    """Decide the gate and, if it opens, compile and allocate every variant.

    -> the (N, K) this module now serves at M <= MAX_M (empty when closed).
    The shipped thin GEMM is warmed here for those shapes at any M above
    MAX_M that ``ms`` holds; the caller warms it for everything else. A
    warmup failure raises: one rank must not serve a different kernel from
    the others (the replicated router and hyper-connection GEMMs would
    differ between ranks).
    """
    _LIVE[0] = False
    reason = closed_reason(worker.device, worker.vllm_config)
    if reason:
        set_but_off(reason)
        return frozenset()
    table_ms = [m for m in ms if m <= MAX_M]
    try:
        impl().warmup(sorted(SHAPES), table_ms, device=worker.device)
    except Exception as exc:
        raise RuntimeError(
            "VLLM_THIN_GEMM_GLUON=1: the Gluon thin GEMM warmup failed "
            f"({type(exc).__name__}: {exc}); unset VLLM_THIN_GEMM_GLUON to "
            "serve without it") from exc
    over = [m for m in ms if m > MAX_M]
    if over:
        shipped_impl().warmup(sorted(SHAPES), over, device=worker.device)
    _LIVE[0] = True
    pinned = impl()._FORCE_M[0]
    logger.info_once(
        "Gluon thin GEMM active [VLLM_THIN_GEMM_GLUON=1]: %d (N, K) x M in %s "
        "(up to %d rows)%s", len(SHAPES), tuple(table_ms), MAX_M,
        f", schedule pinned to M={pinned} [VLLM_THIN_GEMM_GLUON_FORCE_M]"
        if pinned else "")
    return SHAPES


def gluon_layout_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether ``x`` and ``weight`` have the layout the warmup compiled for.

    Triton specializes a kernel on each pointer's 16-byte alignment, on an
    integer argument being 1 or a multiple of 16, so a contiguous operand of
    the table's K (all multiples of 16) and any view with a unit K stride, a
    row stride that is a multiple of 16 elements and a 16-byte aligned base
    run the warmed kernels. Host-side reads only, no allocation."""
    return (
        x.stride(1) == 1
        and x.stride(0) % 16 == 0
        and x.data_ptr() % 16 == 0
        and weight.stride(0) % 16 == 0
        and weight.data_ptr() % 16 == 0
    )


def wrap_dispatch(shipped):
    """``shipped`` (an ``ampere_thin_gemm``) with the table's calls sent to
    ``thin_gemm_gluon`` while the gate is open. Same signature."""
    if envs.VLLM_THIN_GEMM_CONCAT:
        raise ValueError(
            "VLLM_THIN_GEMM_GLUON and VLLM_THIN_GEMM_CONCAT are both set: the "
            "concat groups pair only the shipped thin GEMM's calls; set one "
            "of them")

    def ampere_thin_gemm(
        layer: torch.nn.Module,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            _LIVE[0]
            and x.shape[0] <= MAX_M
            and tuple(weight.shape) in SHAPES
            and thin_gemm_supported(x, weight, bias)
        ):
            if gluon_layout_ok(x, weight):
                return impl().thin_gemm(x, weight)
            logger.warning_once(
                "Gluon thin GEMM: a (N, K) = %s call with x strides %s and "
                "base offset %d mod 16 bytes is not in the layout its kernels "
                "were compiled for; such calls take cuBLAS",
                tuple(weight.shape), tuple(x.stride()), x.data_ptr() % 16)
            return torch.nn.functional.linear(x, weight)
        return shipped(layer, x, weight, bias)

    ampere_thin_gemm.shipped = shipped
    return ampere_thin_gemm
