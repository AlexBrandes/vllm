# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bindings for the shipped BF16 EXL3 extension."""

import importlib
from functools import lru_cache

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_loaded = False

# One fixed FP32 scratch (32 MiB) per stream role, whatever the batch size.
# Each matrix processes rows in chunks whose native workspace fits it.
LINEAR_SCRATCH_ELEMENTS = 8 << 20


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
    out,
    workspace,
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


_ABI_VERSION = 1
# Operators this tree calls; a library without one of them is refused.
_OPERATORS = (
    "linear_out",
    "moe_out",
    "ple_decode_out",
    "linear_workspace_size",
    "moe_workspace_size",
)


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


def ensure_extension() -> None:
    global _loaded
    if _loaded:
        return
    try:
        module = importlib.import_module("vllm._exl3_C")
        _validate_extension(module)
    except (ImportError, OSError, AttributeError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "EXL3 requires the compiled vllm._exl3_C extension, built for the "
            "running PyTorch: build it with python csrc/exl3/build_standalone.py "
            f"--out vllm (in the vLLM source tree). Detail: {exc}"
        ) from exc
    for name, fake in (
        ("linear_out", _linear_fake),
        ("moe_out", _moe_fake),
        ("ple_decode_out", _ple_fake),
    ):
        torch.library.register_fake(f"_exl3_C::{name}")(fake)
    _loaded = True


@lru_cache(maxsize=None)
def linear_workspace_size(max_rows: int, padded_in: int, padded_out: int) -> int:
    ensure_extension()
    return torch.ops._exl3_C.linear_workspace_size(max_rows, padded_in, padded_out)


@lru_cache(maxsize=None)
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


@lru_cache(maxsize=None)
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
    )


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
    workspace = _linear_scratch(x.device)
    out = torch.empty((x.shape[0], logical_out), dtype=x.dtype, device=x.device)
    for start in range(0, x.shape[0], row_chunk):
        linear_out(
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
