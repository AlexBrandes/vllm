# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hidden-stream short conv fused with the AttnRes mix + RMSNorm of the next
sublayer, for large batches of an already reduced input (CUDA)."""

import os

import torch

from vllm.platforms import current_platform

from . import use_sm12x_path
from .attn_res import attn_res
from .sconv import short_conv
from .seq_parallel import SeqParallel

try:
    from vllm.models.inkling2.nvidia.ops.sm80.inkling_sm80_cuda import (
        HAS_CUDA_EXT as _SM80_HAS_CUDA,
    )
    from vllm.models.inkling2.nvidia.ops.sm80.inkling_sm80_cuda import (
        conv_mix_cuda as _sm80_conv_mix_cuda,
    )
except Exception:
    _SM80_HAS_CUDA = False
    _sm80_conv_mix_cuda = None


def conv_mix(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    stream_offset: int,
    stream_width: int,
    conv_out: torch.Tensor,
    residual: torch.Tensor | None,
    addend: torch.Tensor | None,
    stack: torch.Tensor,
    last: torch.Tensor | None,
    queries: torch.Tensor,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    eps: float = 1e-6,
    sp: SeqParallel | None = None,
    halo: int = 0,
    prefix_align: int = 0,
) -> torch.Tensor:
    """`short_conv` the reduced bf16 `x` into `conv_out` (with the
    optional `(residual + addend) + y` update), then `attn_res` over
    `[*stack, (last +) conv_out]` with each query. Returns the `(Q, T, D)`
    bf16 mix outputs, or without `norm_weight` the fp32 mixes before the
    RMSNorm (not with `sp`).

    With `sp`, the tokens are this rank's shard of the batch, `x` starts
    with the `halo` rows before it (or is `(world, halo + T, D)`, the
    partial rows every rank pushed, summed in rank order), and the mix
    outputs of all shards land in the all-gather buffer of every rank
    (returned once all have landed).
    Every rank writes the conv-state cache rows that later steps read (see
    `SeqParallel`) given the batch's `prefix_align`.
    """
    num_tokens, d = conv_out.shape
    assert num_tokens > 0 and conv_out.is_contiguous()
    assert x.shape[-2:] == (halo + num_tokens, d)
    assert x.stride(-1) == 1 and x.stride(-2) == d
    assert x.dtype == torch.bfloat16
    for t in (residual, addend, last):
        assert t is None or (t.is_contiguous() and t.shape == conv_out.shape)
    assert residual is None or addend is not None
    assert stack.stride(1) == d and stack.stride(2) == 1
    assert queries.dtype == torch.float32 and queries.is_contiguous()
    if norm_weight is not None:
        assert norm_weight.shape == queries.shape and norm_weight.is_contiguous()
    assert conv_weight.shape == (d, 4) and conv_weight.is_contiguous()
    assert d == cache.shape[1] * stream_width and cache.stride(3) == 1
    assert positions.dtype == slot_mapping.dtype == torch.int64
    args = (
        x,
        cache,
        conv_weight,
        positions,
        seq_idx,
        slot_mapping,
        block_table,
        query_start,
        stream_offset,
        stream_width,
        5**-0.5,
        residual,
        addend,
        conv_out,
        stack,
        last,
        queries,
        norm_weight,
        eps,
        norm_eps,
    )
    if sp is None:
        assert halo == 0 and positions.shape[0] == num_tokens and x.dim() == 2
        if use_sm12x_path():
            # sm_80: fused CUDA conv_mix (100% bit-exact):
            # - Decode (T <= 256): 288t for n_stack<=3, 576t zero-spill for n_stack=4..7 (1.33x faster)
            # - Prefill (T > 256, both Q == 1 AND Q == 3): 1-barrier intra-fast-path 288t CUDA conv_mix (1.33-1.73x faster)
            if (
                _SM80_HAS_CUDA
                and current_platform.is_device_capability_family(80)
                and os.environ.get("INKLING_SM80_NO_CUDA_CONVMIX") != "1"
                and d == 2304
                and stream_width % 8 == 0
                and stream_offset % 8 == 0
                and stack.shape[0] <= 7
                and queries.shape[0] in (1, 3)
                and conv_weight.dtype == torch.bfloat16
                and cache.dtype == torch.bfloat16
                and stack.dtype == torch.bfloat16
                and conv_out.dtype == torch.bfloat16
                and (norm_weight is None or norm_weight.dtype == torch.bfloat16)
                and (residual is None or residual.dtype == torch.bfloat16)
                and (addend is None or addend.dtype == torch.bfloat16)
                and (last is None or last.dtype == torch.bfloat16)
                and block_table.dtype == torch.int32
                and seq_idx.dtype == torch.int32
                and query_start.dtype == torch.int32
            ):
                return _sm80_conv_mix_cuda(
                    x,
                    conv_weight,
                    cache,
                    positions,
                    block_table,
                    seq_idx,
                    slot_mapping,
                    query_start,
                    stream_offset,
                    stream_width,
                    conv_out,
                    residual,
                    addend,
                    stack,
                    last,
                    queries,
                    norm_weight,
                    norm_eps,
                    eps,
                )
            # Native conv/mix kernels require Hopper or SM10x. Keep the
            # loop executor's cache writes and residual rounding on SM12x.
            short_conv(
                x,
                conv_weight,
                cache,
                positions,
                block_table,
                seq_idx,
                slot_mapping,
                query_start,
                stream_offset,
                stream_width,
                addend,
                residual,
                conv_out,
            )
            return attn_res(
                stack,
                conv_out if last is None else last,
                None if last is None else conv_out,
                queries,
                norm_weight,
                norm_eps,
                True,
                eps,
            )
        out = conv_out.new_empty(
            (queries.shape[0], num_tokens, d),
            dtype=torch.float32 if norm_weight is None else conv_out.dtype,
        )
        torch.ops._C.inkling2_conv_mix(*args, out)
        return out
    assert norm_weight is not None
    batch_tokens = positions.shape[0]
    token_base, end, _ = sp.shard(batch_tokens)
    assert end - token_base == num_tokens
    assert x.dim() == 2 or x.shape[0] == sp.world_size
    torch.ops._C.inkling2_seq_parallel_conv_mix(
        *args,
        token_base,
        halo,
        sp.push_bases.data_ptr(),
        sp.gather_offset,
        sp.world_size,
        sp.max_tokens,
        sp.rank,
        sp.gather_slot,
        sp.gather_epochs,
        sp.staging_offset,
        sp.tail_rows,
        prefix_align,
    )
    return sp.gathered(queries.shape[0], batch_tokens)
