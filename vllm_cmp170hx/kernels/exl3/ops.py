# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bindings for the shipped BF16 EXL3 extension."""

import importlib
from functools import cache

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm_cmp170hx.kernels.exl3 import dense_prefill_tc
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_loaded = False
# The base fake implementations, registered once per process: a retry after a
# resolve that raised must not register them again (torch refuses a second).
_fakes_registered = False
# VLLM_EXL3_DENSE_SPLITK with its gate open: _linear calls linear_out_splitk.
# Fixed by ensure_extension, before any launch or graph capture.
_dense_splitk = False

# One fixed FP32 scratch (32 MiB) per stream role, whatever the batch size.
# Each matrix processes rows in chunks whose native workspace fits it.
LINEAR_SCRATCH_ELEMENTS = 8 << 20

# moe_out_v2 (csrc/exl3/moe_v2.cu) covers exactly one configuration: calls of
# 1..MOE_V2_MAX_ROWS rows, hidden 2560, intermediate 640, top-10, 256 of 512
# experts local, and every routed matrix stored as below.
MOE_V2_MAX_ROWS = 64
MOE_V2_HIDDEN = 2560
MOE_V2_INTERMEDIATE = 640
MOE_V2_TOP_K = 10
MOE_V2_LOCAL_EXPERTS = 256
MOE_V2_GLOBAL_EXPERTS = 512
# [padded_in, padded_out, bits_x2, codebook] of gate, up and down.
MOE_V2_EXPERT_META = [
    [MOE_V2_HIDDEN, MOE_V2_INTERMEDIATE, 10, 2],
    [MOE_V2_HIDDEN, MOE_V2_INTERMEDIATE, 10, 2],
    [MOE_V2_INTERMEDIATE, MOE_V2_HIDDEN, 10, 2],
]
# Prefill keeps the Qwen family and separately covers the exact GLM PP2 family.
# GLM's 3.5-bit first routed layer and all decode calls remain on the base path.
MOE_PREFILL_MIN_ROWS = MOE_V2_MAX_ROWS + 1
MOE_PREFILL_MAX_ROWS = 3460
# 2312 matches the KDA-wide and sparse-MLA PP prefill bounds, so a
# max-num-batched-tokens of 2307 (DFlash k3: 2304 prefill rows + 3 draft slots)
# yields 2304-token chunks -- two 1152-token KDA alignment units at block size
# 4608 -- that stay on this path. moe_out_prefill itself has no upper row bound;
# the workspace is sized from min(max_num_batched_tokens, this bound).
MOE_PREFILL_GLM_MAX_ROWS = 2312
MOE_PREFILL_GLM_SHAPE = (4096, 2048, 8, 288, 288)
MOE_PREFILL_GLM_EXPERT_BITS = (5, 7)

# moe_out_decode_tc (csrc/exl3/moe_prefill.cu, the small-row schedule) covers
# decode-step calls of 1..MOE_DECODE_TC_MAX_ROWS rows of the exact GLM family
# above: the same shape, uniform expert format and SwiGLU limit gate.
MOE_DECODE_TC_MAX_ROWS = 8


def _linear_fake(
    x, trellis, suh, svh, bits_x2, codebook, logical_in, logical_out, out, workspace
):
    torch._check(x.dtype == torch.bfloat16)
    torch._check(out.dtype == torch.bfloat16)
    torch._check(out.shape[0] == x.shape[0])
    torch._check(out.shape[1] == logical_out)
    torch._check(x.shape[1] == logical_in)
    torch._check(workspace.dtype == torch.float32)


def _moe_fake(
    x,
    topk_ids,
    topk_weights,
    packed_ptrs,
    packed_meta,
    expert_map,
    intermediate_size,
    bits_x2,
    out,
    workspace,
    swiglu_limit=0.0,
):
    torch._check(x.dtype == torch.bfloat16)
    torch._check(out.dtype == torch.bfloat16)
    torch._check(out.shape == x.shape)
    torch._check(topk_ids.shape == topk_weights.shape)
    torch._check(workspace.dtype == torch.float32)


def _ple_fake(
    packed_rows,
    global_ids,
    head_bias,
    head_offsets,
    head_vocab_sizes,
    owner_lo,
    owner_hi,
    bits,
    out,
):
    torch._check(out.dtype == torch.bfloat16)
    torch._check(out.shape[0] == packed_rows.shape[0])
    torch._check(out.shape[1] == head_bias.shape[1])
    torch._check(global_ids.numel() == packed_rows.shape[0])


_ABI_VERSION = 2
# Operators this tree calls; a library without one of them is refused.
_OPERATORS = (
    "linear_out",
    "linear_out_splitk",
    "moe_out",
    "moe_out_v2",
    "moe_out_prefill",
    "moe_out_decode_tc",
    "ple_decode_out",
    "ple_read_rows",
    "linear_workspace_size",
    "linear_splitk_splits",
    "moe_workspace_size",
    "moe_workspace_size_v2",
    "moe_workspace_size_prefill",
    "moe_workspace_size_decode_tc",
)
# Classes this tree constructs; a library without one of them is refused.
_CLASSES = ("PleHostReader",)


def _runtime_build_info() -> dict:
    cuda = torch.version.cuda
    major, minor = (int(v) for v in cuda.split(".")[:2]) if cuda else (0, 0)
    return {
        "abi_version": _ABI_VERSION,
        "torch_version": torch.__version__.split("+")[0],
        "cuda_version": major * 1000 + minor * 10,
        "cxx11_abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
    }


def _validate_extension(module) -> None:
    """module.cpp registers through TORCH_LIBRARY, which is bound to the
    PyTorch build: refuse a library built against another one."""
    built = module.build_info()
    for key, expected in _runtime_build_info().items():
        actual = built.get(key)
        # CUDA minor toolkits within one major share the runtime ABI.
        if key == "cuda_version" and isinstance(actual, int):
            actual, expected = actual // 1000, expected // 1000
        if actual != expected:
            raise ValueError(
                f"{key} mismatch: extension {built.get(key)!r}, runtime {expected!r}"
            )
    for name in _OPERATORS:
        torch._C._dispatch_find_schema_or_throw(f"_exl3_C::{name}", "")
    for name in _CLASSES:
        getattr(torch.classes._exl3_C, name)


def dense_splitk_closed_reason(capability: tuple[int, int] | None) -> str | None:
    """Why linear_out_splitk must not serve this process, or None when it may.

    The operator itself keeps linear_out for every launch outside the
    configuration its kernels cover (formats, block shapes, grid sizes).
    """
    if capability != (8, 0):
        return f"device capability {capability}, not (8, 0)"
    return None


def _resolve_dense_splitk() -> bool:
    """VLLM_EXL3_DENSE_SPLITK for this process, with its log line."""
    if not envs.VLLM_EXL3_DENSE_SPLITK:
        return False
    capability = (
        tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
    )
    reason = dense_splitk_closed_reason(capability)
    if reason is not None:
        logger.info_once(
            "VLLM_EXL3_DENSE_SPLITK is set but the gate is closed: %s; every "
            "dense EXL3 call keeps linear_out.",
            reason,
            scope="process",
        )
        return False
    logger.info_once(
        "EXL3 dense split-K active (VLLM_EXL3_DENSE_SPLITK): dense launches of "
        "at most 1024 tiny or small blocks (bits_x2 10/12/14, codebook 2) split "
        "K in two on warp-specialized blocks; other launches keep linear_out.",
        scope="process",
    )
    return True


def ensure_extension() -> None:
    global _loaded, _dense_splitk, _fakes_registered
    if _loaded:
        return
    try:
        module = importlib.import_module("vllm_cmp170hx._exl3_C")
        _validate_extension(module)
    except (ImportError, OSError, AttributeError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "EXL3 requires the compiled vllm_cmp170hx._exl3_C extension, built for the "
            "running PyTorch: build it with python csrc/exl3/build_standalone.py "
            f"--out vllm (in the vLLM source tree). Detail: {exc}"
        ) from exc
    if not _fakes_registered:
        for name, fake in (
            ("linear_out", _linear_fake),
            ("linear_out_splitk", _linear_fake),
            ("moe_out", _moe_fake),
            ("moe_out_v2", _moe_fake),
            ("moe_out_prefill", _moe_fake),
            ("moe_out_decode_tc", _moe_fake),
            ("ple_decode_out", _ple_fake),
        ):
            torch.library.register_fake(f"_exl3_C::{name}")(fake)
        _fakes_registered = True
    from vllm_cmp170hx.kernels.exl3 import dense_mma

    dense_mma.resolve()
    _dense_splitk = _resolve_dense_splitk()
    dense_prefill_tc.resolve()
    _loaded = True


@cache
def linear_workspace_size(max_rows: int, padded_in: int, padded_out: int) -> int:
    ensure_extension()
    return torch.ops._exl3_C.linear_workspace_size(max_rows, padded_in, padded_out)


@cache
def linear_row_chunk(padded_in: int, padded_out: int) -> int:
    """Return the most rows whose linear workspace fits the fixed scratch."""
    rows, limit = 0, LINEAR_SCRATCH_ELEMENTS
    while rows < limit:
        middle = (rows + limit + 1) // 2
        if linear_workspace_size(middle, padded_in, padded_out) <= (
            LINEAR_SCRATCH_ELEMENTS
        ):
            rows = middle
        else:
            limit = middle - 1
    if rows == 0:
        raise ValueError("One EXL3 matrix row exceeds the fixed linear scratch")
    return rows


@cache
def moe_workspace_size(
    max_rows: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    local_experts: int,
) -> int:
    ensure_extension()
    return torch.ops._exl3_C.moe_workspace_size(
        max_rows, hidden_size, intermediate_size, topk, local_experts
    )


@cache
def moe_workspace_size_v2(
    max_rows: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    local_experts: int,
) -> int:
    ensure_extension()
    return torch.ops._exl3_C.moe_workspace_size_v2(
        max_rows, hidden_size, intermediate_size, topk, local_experts
    )


def moe_v2_closed_reason(
    capability: tuple[int, int] | None,
    dtype: torch.dtype,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    local_experts: int,
    global_experts: int,
    metadata: list[list[list[int]]],
) -> str | None:
    """Why moe_out_v2 must not serve this layer, or None when it may.

    Checked once per layer from host-side values (``metadata`` is the
    per-expert [padded_in, padded_out, bits_x2, codebook] list the layer's
    ``exl3_meta`` buffer is built from), never from device tensors.
    """
    return _moe_family_closed_reason(
        capability,
        dtype,
        hidden_size,
        intermediate_size,
        top_k,
        local_experts,
        global_experts,
        metadata,
        (
            MOE_V2_HIDDEN,
            MOE_V2_INTERMEDIATE,
            MOE_V2_TOP_K,
            MOE_V2_LOCAL_EXPERTS,
            MOE_V2_GLOBAL_EXPERTS,
        ),
        MOE_V2_EXPERT_META,
    )


def moe_prefill_closed_reason(
    capability: tuple[int, int] | None,
    dtype: torch.dtype,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    local_experts: int,
    global_experts: int,
    metadata: list[list[list[int]]],
    bits_x2: int,
) -> str | None:
    """Select only the exact Qwen or GLM prefill family from host metadata.

    ``bits_x2`` is the layer's uniform expert format (GLM: 5 or 7); the
    device family is selected by (shape, bits_x2).
    """
    if (hidden_size, intermediate_size) != MOE_PREFILL_GLM_SHAPE[:2]:
        return moe_v2_closed_reason(
            capability,
            dtype,
            hidden_size,
            intermediate_size,
            top_k,
            local_experts,
            global_experts,
            metadata,
        )
    if bits_x2 not in MOE_PREFILL_GLM_EXPERT_BITS:
        return f"GLM prefill runs the 5-bit or 3.5-bit family, not bits_x2 {bits_x2}"
    expected_meta = [
        [4096, 2048, bits_x2, 2],
        [4096, 2048, bits_x2, 2],
        [2048, 4096, bits_x2, 2],
    ]
    return _moe_family_closed_reason(
        capability,
        dtype,
        hidden_size,
        intermediate_size,
        top_k,
        local_experts,
        global_experts,
        metadata,
        MOE_PREFILL_GLM_SHAPE,
        expected_meta,
    )


def moe_decode_tc_closed_reason(
    capability: tuple[int, int] | None,
    dtype: torch.dtype,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    local_experts: int,
    global_experts: int,
    metadata: list[list[list[int]]],
    bits_x2: int,
) -> str | None:
    """Why moe_out_decode_tc must not serve this layer, or None when it may.

    The GLM decode family is the GLM prefill family on the small-row
    schedule: the same (hidden, intermediate, top_k, experts) shape, the same
    uniform 5-bit or 3.5-bit expert format with codebook 2, and the same
    capability and dtype checks. The caller keeps the family to the exact GLM
    layout and its SwiGLU limit 10.0, to 16-byte-aligned trellis bases, and
    to calls of 1..MOE_DECODE_TC_MAX_ROWS rows in decode steps.
    """
    if (hidden_size, intermediate_size) != MOE_PREFILL_GLM_SHAPE[:2]:
        return (
            f"(hidden, intermediate) = {(hidden_size, intermediate_size)}, not "
            f"{MOE_PREFILL_GLM_SHAPE[:2]} (the small-row decode schedule "
            "serves the GLM family; other decode calls keep their kernels)"
        )
    if bits_x2 not in MOE_PREFILL_GLM_EXPERT_BITS:
        return f"GLM decode runs the 5-bit or 3.5-bit family, not bits_x2 {bits_x2}"
    expected_meta = [
        [4096, 2048, bits_x2, 2],
        [4096, 2048, bits_x2, 2],
        [2048, 4096, bits_x2, 2],
    ]
    return _moe_family_closed_reason(
        capability,
        dtype,
        hidden_size,
        intermediate_size,
        top_k,
        local_experts,
        global_experts,
        metadata,
        MOE_PREFILL_GLM_SHAPE,
        expected_meta,
    )


def _moe_family_closed_reason(
    capability,
    dtype,
    hidden_size,
    intermediate_size,
    top_k,
    local_experts,
    global_experts,
    metadata,
    expected_shape,
    expected_meta,
) -> str | None:
    if capability != (8, 0):
        return f"device capability {capability}, not (8, 0)"
    if dtype != torch.bfloat16:
        return f"activation dtype {dtype}, not bfloat16"
    shape = (hidden_size, intermediate_size, top_k, local_experts, global_experts)
    if shape != expected_shape:
        return (
            f"(hidden, intermediate, top_k, local experts, global experts) = "
            f"{shape}, not {expected_shape}"
        )
    if len(metadata) != local_experts:
        return f"{len(metadata)} expert entries for {local_experts} local experts"
    for local, meta in enumerate(metadata):
        if [list(m) for m in meta] != expected_meta:
            return (
                f"local expert {local} stores [padded_in, padded_out, bits_x2, "
                f"codebook] {[list(m) for m in meta]}, not {expected_meta}"
            )
    return None


def linear_out(
    x,
    trellis,
    suh,
    svh,
    bits_x2: int,
    codebook: int,
    logical_in: int,
    logical_out: int,
    out,
    workspace,
) -> None:
    ensure_extension()
    torch.ops._exl3_C.linear_out(
        x,
        trellis,
        suh,
        svh,
        bits_x2,
        codebook,
        logical_in,
        logical_out,
        out,
        workspace,
    )


def linear_out_splitk(
    x,
    trellis,
    suh,
    svh,
    bits_x2: int,
    codebook: int,
    logical_in: int,
    logical_out: int,
    out,
    workspace,
) -> None:
    """linear_out's contract (arguments, checks, workspace); see
    csrc/exl3/dense_splitk.h for the launches it changes."""
    ensure_extension()
    torch.ops._exl3_C.linear_out_splitk(
        x,
        trellis,
        suh,
        svh,
        bits_x2,
        codebook,
        logical_in,
        logical_out,
        out,
        workspace,
    )


def linear_splitk_splits(
    rows: int, padded_out: int, sms: int, bits_x2: int, codebook: int
) -> int:
    """K split factor linear_out_splitk uses for this launch: 2 on the
    split-K path, 1 where it runs linear_out."""
    ensure_extension()
    return torch.ops._exl3_C.linear_splitk_splits(
        rows, padded_out, sms, bits_x2, codebook
    )


def moe_out(
    x,
    topk_ids,
    topk_weights,
    packed_ptrs,
    packed_meta,
    expert_map,
    intermediate_size: int,
    out,
    workspace,
    swiglu_limit: float = 0.0,
) -> None:
    ensure_extension()
    torch.ops._exl3_C.moe_out(
        x,
        topk_ids,
        topk_weights,
        packed_ptrs,
        packed_meta,
        expert_map,
        intermediate_size,
        out,
        workspace,
        swiglu_limit,
    )


def moe_out_v2(
    x,
    topk_ids,
    topk_weights,
    packed_ptrs,
    packed_meta,
    expert_map,
    intermediate_size: int,
    out,
    workspace,
) -> None:
    ensure_extension()
    torch.ops._exl3_C.moe_out_v2(
        x,
        topk_ids,
        topk_weights,
        packed_ptrs,
        packed_meta,
        expert_map,
        intermediate_size,
        out,
        workspace,
    )


_moe_v2_warm_devices: set[torch.device] = set()


def warmup_moe_v2(
    packed_ptrs: torch.Tensor,
    packed_meta: torch.Tensor,
    expert_map: torch.Tensor,
    global_expert_ids: list[int],
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
) -> None:
    """Run moe_out_v2 per device before any CUDA graph capture.

    Each launch configuration's first call queries the SM count and the
    kernel's occupancy and keeps them (moe_v2.cu launch_lane). The operator
    picks short or long K slices from rows x top_k (160 routed assignments),
    so one call of one row and one of MOE_V2_MAX_ROWS rows, all with local
    routes, launch every kernel it can pick, and no capture is a first call.
    The temporary buffers are freed again.
    """
    device = packed_ptrs.device
    if device in _moe_v2_warm_devices:
        return
    rows = MOE_V2_MAX_ROWS
    ids = torch.tensor(
        [global_expert_ids[:top_k]] * rows, dtype=torch.int32, device=device
    )
    weights = torch.full((rows, top_k), 1.0 / top_k, dtype=torch.float32, device=device)
    x = torch.ones((rows, hidden_size), dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)
    workspace = torch.empty(
        moe_workspace_size_v2(
            rows, hidden_size, intermediate_size, top_k, packed_ptrs.shape[0]
        ),
        dtype=torch.float32,
        device=device,
    )
    for count in (1, rows):
        moe_out_v2(
            x[:count],
            ids[:count],
            weights[:count],
            packed_ptrs,
            packed_meta,
            expert_map,
            intermediate_size,
            out[:count],
            workspace,
        )
    torch.cuda.synchronize(device)
    _moe_v2_warm_devices.add(device)


@cache
def moe_workspace_size_prefill(
    max_rows: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    local_experts: int,
) -> int:
    ensure_extension()
    return torch.ops._exl3_C.moe_workspace_size_prefill(
        max_rows, hidden_size, intermediate_size, topk, local_experts
    )


def moe_out_prefill(
    x,
    topk_ids,
    topk_weights,
    packed_ptrs,
    packed_meta,
    expert_map,
    intermediate_size: int,
    bits_x2: int,
    out,
    workspace,
) -> None:
    """Guarded Qwen (unclipped) or GLM (SwiGLU limit 10) prefill family.

    ``bits_x2`` is the layer's uniform expert format (GLM: 5 or 7); the
    family selection is by (shape, bits_x2).
    """
    ensure_extension()
    torch.ops._exl3_C.moe_out_prefill(
        x,
        topk_ids,
        topk_weights,
        packed_ptrs,
        packed_meta,
        expert_map,
        intermediate_size,
        bits_x2,
        out,
        workspace,
    )


_moe_prefill_warm_devices: set[tuple[torch.device, int, int, int]] = set()


def warmup_moe_prefill(
    packed_ptrs: torch.Tensor,
    packed_meta: torch.Tensor,
    expert_map: torch.Tensor,
    global_expert_ids: list[int],
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    bits_x2: int,
) -> None:
    """Run moe_out_prefill once per device, geometry and format before CUDA
    graph capture.

    The operator sets its GEMM kernels' shared-memory limit on its first call
    and launches the same kernels at every row count (the row tiles of an
    expert are chosen on the device), so one call of MOE_PREFILL_MIN_ROWS
    rows, all with local routes, loads every kernel it can launch. The
    temporary buffers are freed again.
    """
    device = packed_ptrs.device
    key = (device, hidden_size, intermediate_size, bits_x2)
    if key in _moe_prefill_warm_devices:
        return
    rows = MOE_PREFILL_MIN_ROWS
    ids = torch.tensor(
        [global_expert_ids[:top_k]] * rows, dtype=torch.int32, device=device
    )
    weights = torch.full((rows, top_k), 1.0 / top_k, dtype=torch.float32, device=device)
    x = torch.ones((rows, hidden_size), dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)
    workspace = torch.empty(
        moe_workspace_size_prefill(
            rows, hidden_size, intermediate_size, top_k, packed_ptrs.shape[0]
        ),
        dtype=torch.float32,
        device=device,
    )
    moe_out_prefill(
        x,
        ids,
        weights,
        packed_ptrs,
        packed_meta,
        expert_map,
        intermediate_size,
        bits_x2,
        out,
        workspace,
    )
    torch.cuda.synchronize(device)
    _moe_prefill_warm_devices.add(key)


@cache
def moe_workspace_size_decode_tc(
    max_rows: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    local_experts: int,
) -> int:
    ensure_extension()
    return torch.ops._exl3_C.moe_workspace_size_decode_tc(
        max_rows, hidden_size, intermediate_size, topk, local_experts
    )


def moe_out_decode_tc(
    x,
    topk_ids,
    topk_weights,
    packed_ptrs,
    packed_meta,
    expert_map,
    intermediate_size: int,
    bits_x2: int,
    out,
    workspace,
) -> None:
    """The GLM 5-bit or 3.5-bit family on the small-row schedule (1..8 rows).

    The same tensor-core arithmetic as moe_out_prefill with 8-row route
    tiles and one 8-row MMA group per GEMM block; the caller keeps the
    family checks (shape, format, SwiGLU limit, aligned trellis, decode
    step). Results differ from moe_out in the last bits.
    """
    ensure_extension()
    torch.ops._exl3_C.moe_out_decode_tc(
        x,
        topk_ids,
        topk_weights,
        packed_ptrs,
        packed_meta,
        expert_map,
        intermediate_size,
        bits_x2,
        out,
        workspace,
    )


_moe_decode_tc_warm_devices: set[tuple[torch.device, int, int, int]] = set()


def warmup_moe_decode_tc(
    packed_ptrs: torch.Tensor,
    packed_meta: torch.Tensor,
    expert_map: torch.Tensor,
    global_expert_ids: list[int],
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    bits_x2: int,
) -> None:
    """Run moe_out_decode_tc once per device, geometry and format before CUDA
    graph capture.

    The operator sets its GEMM kernels' shared-memory limit on its first call
    and launches the same kernels at every row count (route tiles of at most
    8 rows are chosen on the device), so one call of MOE_DECODE_TC_MAX_ROWS
    rows, all with local routes, loads every kernel it can launch. The
    temporary buffers are freed again.
    """
    device = packed_ptrs.device
    key = (device, hidden_size, intermediate_size, bits_x2)
    if key in _moe_decode_tc_warm_devices:
        return
    rows = MOE_DECODE_TC_MAX_ROWS
    ids = torch.tensor(
        [global_expert_ids[:top_k]] * rows, dtype=torch.int32, device=device
    )
    weights = torch.full((rows, top_k), 1.0 / top_k, dtype=torch.float32, device=device)
    x = torch.ones((rows, hidden_size), dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)
    workspace = torch.empty(
        moe_workspace_size_decode_tc(
            rows, hidden_size, intermediate_size, top_k, packed_ptrs.shape[0]
        ),
        dtype=torch.float32,
        device=device,
    )
    moe_out_decode_tc(
        x,
        ids,
        weights,
        packed_ptrs,
        packed_meta,
        expert_map,
        intermediate_size,
        bits_x2,
        out,
        workspace,
    )
    torch.cuda.synchronize(device)
    _moe_decode_tc_warm_devices.add(key)


def ple_decode_out(
    packed_rows,
    global_ids,
    head_bias,
    head_offsets,
    head_vocab_sizes,
    owner_lo: int,
    owner_hi: int,
    bits: int,
    out,
) -> None:
    ensure_extension()
    torch.ops._exl3_C.ple_decode_out(
        packed_rows,
        global_ids,
        head_bias,
        head_offsets,
        head_vocab_sizes,
        owner_lo,
        owner_hi,
        bits,
        out,
    )


def ple_read_rows(
    fd: int,
    ids: torch.Tensor,
    dest: torch.Tensor,
    count: int,
    base: int,
    row_bytes: int,
    owner_lo: int,
    owner_hi: int,
    head_offsets: torch.Tensor,
    head_vocab_sizes: torch.Tensor,
    retries: int,
    max_threads: int,
) -> torch.Tensor:
    """Read PLE rows from storage in one host call that releases the GIL
    (csrc/exl3/ple_read.cpp); returns the [index, errno, offset] failures."""
    ensure_extension()
    return torch.ops._exl3_C.ple_read_rows(
        fd,
        ids,
        dest,
        count,
        base,
        row_bytes,
        owner_lo,
        owner_hi,
        head_offsets,
        head_vocab_sizes,
        retries,
        max_threads,
    )


def ple_host_reader(
    fd: int,
    ids: torch.Tensor,
    rows: torch.Tensor,
    base: int,
    row_bytes: int,
    owner_lo: int,
    owner_hi: int,
    head_offsets: torch.Tensor,
    head_vocab_sizes: torch.Tensor,
    retries: int,
    max_threads: int,
    slots: int,
):
    """A PLE row reader of ids into rows whose reads run as host functions
    on a CUDA stream (csrc/exl3/ple_read.cpp, PleHostReader)."""
    ensure_extension()
    return torch.classes._exl3_C.PleHostReader(
        fd,
        ids,
        rows,
        base,
        row_bytes,
        owner_lo,
        owner_hi,
        head_offsets,
        head_vocab_sizes,
        retries,
        max_threads,
        slots,
    )


def _linear_scratch(device: torch.device) -> torch.Tensor:
    from vllm.utils.torch_utils import aux_stream, current_stream
    from vllm.v1.worker.workspace import current_workspace_manager

    def allocate():
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("EXL3 scratch must be allocated before graph capture")
        # The compute and graph-capture streams never run model work at the
        # same time and share one scratch. The shared-expert aux stream
        # overlaps the compute stream, so it gets its own. Both are allocated
        # on first use, before memory profiling ends, and never resized.
        count = 1 if aux_stream() is None else 2
        return [
            torch.empty(LINEAR_SCRATCH_ELEMENTS, dtype=torch.float32, device=device)
            for _ in range(count)
        ]

    scratch = current_workspace_manager().get_persistent_resource(
        "exl3_linear_scratch", allocate
    )
    aux = aux_stream()
    return scratch[1] if aux is not None and current_stream() == aux else scratch[0]


def _linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    bits_x2: int,
    codebook: int,
    logical_out: int,
    row_chunk: int,
) -> torch.Tensor:
    if dense_prefill_tc.active and dense_prefill_tc.once_takes(
        x, trellis, bits_x2, codebook, logical_out
    ):
        from vllm_cmp170hx.kernels.exl3 import dense_prefill_once

        once = dense_prefill_once.linear(
            x, trellis, suh, svh, bits_x2, codebook, logical_out
        )
        if once is not None:
            return once
    workspace = _linear_scratch(x.device)
    out = torch.empty((x.shape[0], logical_out), dtype=x.dtype, device=x.device)
    operator = linear_out_splitk if _dense_splitk else linear_out
    if dense_prefill_tc.active:
        operator = dense_prefill_tc.route(operator, x.shape[0], _dense_splitk)
    for start in range(0, x.shape[0], row_chunk):
        operator(
            x[start : start + row_chunk],
            trellis,
            suh,
            svh,
            bits_x2,
            codebook,
            x.shape[1],
            logical_out,
            out[start : start + row_chunk],
            workspace,
        )
    return out


def _linear_meta(x, trellis, suh, svh, bits_x2, codebook, logical_out, row_chunk):
    torch._check(x.dtype == torch.bfloat16)
    torch._check(x.dim() == 2)
    return x.new_empty((x.shape[0], logical_out))


direct_register_custom_op(
    op_name="exl3_linear",
    op_func=_linear,
    mutates_args=[],
    fake_impl=_linear_meta,
)
