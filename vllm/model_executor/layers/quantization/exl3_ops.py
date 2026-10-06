# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bindings for the shipped BF16 EXL3 extension."""

import importlib
from functools import lru_cache

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_loaded = False


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


def ensure_extension() -> None:
    global _loaded
    if _loaded:
        return
    try:
        importlib.import_module("vllm._exl3_C")
    except ImportError as exc:
        raise RuntimeError(
            "EXL3 requires the compiled vllm._exl3_C extension; "
            "rebuild vLLM with CUDA EXL3 support."
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


def validate_moe_tables(
    packed_ptrs: torch.Tensor,
    packed_meta: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    ensure_extension()
    torch.ops._exl3_C.validate_moe_tables(
        packed_ptrs, packed_meta, hidden_size, intermediate_size
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


def _linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    bits_x2: int,
    codebook: int,
    logical_out: int,
    workspace_elements: int,
    max_rows: int,
) -> torch.Tensor:
    from vllm.v1.worker.workspace import current_workspace_manager

    manager = current_workspace_manager()
    stream = torch.cuda.current_stream(x.device)

    def allocate():
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("EXL3 scratch must be warmed before graph capture")
        return torch.empty(workspace_elements, dtype=torch.float32, device=x.device)

    storage = manager.get_persistent_resource(
        ("exl3_linear_workspace", stream.cuda_stream), lambda: [allocate()]
    )
    if storage[0].numel() < workspace_elements:
        if manager.is_locked():
            raise RuntimeError("EXL3 scratch capacity was not warmed before capture")
        storage[0] = allocate()
    workspace = storage[0]
    out = torch.empty((x.shape[0], logical_out), dtype=x.dtype, device=x.device)
    for start in range(0, x.shape[0], max_rows):
        linear_out(
            x[start : start + max_rows],
            trellis,
            suh,
            svh,
            bits_x2,
            codebook,
            x.shape[1],
            logical_out,
            out[start : start + max_rows],
            workspace,
        )
    return out


def _linear_meta(
    x, trellis, suh, svh, bits_x2, codebook, logical_out, workspace_elements, max_rows
):
    torch._check(x.dtype == torch.bfloat16)
    torch._check(x.dim() == 2)
    return x.new_empty((x.shape[0], logical_out))


direct_register_custom_op(
    op_name="exl3_linear",
    op_func=_linear,
    mutates_args=[],
    fake_impl=_linear_meta,
)
