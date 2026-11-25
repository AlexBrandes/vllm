# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""moe_align_block_size for GPUs the image's compiled op does not cover (170HX, sm_80), 21 Nov 2026.

vllm/vllm-openai:inkling-2-mini builds _moe_C's moe_align_block_size kernels for sm_100 only: on sm_80 the op launches
nothing and returns uninitialised buffers (cudaErrorNoKernelImageForDevice left pending, wrong routing). Same contract
as vllm.model_executor.layers.fused_moe.moe_align_block_size with expert_map=None and ignore_invalid_experts=True (the
Inkling2 Hopper expert path's only use): ids outside [0, num_experts) are skipped; sorted_ids holds each expert's
routed row indices, its block padded with `numel`; expert_ids one expert per block (unused blocks 0, never read:
Humming stops at num_tokens_padded); num_tokens_post_pad [1]. Plain torch ops with fixed shapes and no host sync, so it
can be captured in a CUDA graph. Rows of one expert keep their original order (stable sort).
"""

import torch


def moe_align_block_size_torch(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
):
    numel = topk_ids.numel()
    max_pad = numel + num_experts * (block_size - 1)
    if numel < num_experts:
        max_pad = min(numel * block_size, max_pad)
    dev = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)
    valid = (flat >= 0) & (flat < num_experts)
    key = torch.where(valid, flat, torch.full_like(flat, num_experts))
    counts = torch.zeros(num_experts + 1, dtype=torch.int64, device=dev)
    counts.scatter_add_(0, key, torch.ones_like(key))
    padded = (counts[:num_experts] + (block_size - 1)) // block_size * block_size
    pad_end = torch.cumsum(padded, 0)
    pad_start = torch.cat(
        [pad_end - padded, torch.full((1,), max_pad, dtype=torch.int64, device=dev)]
    )
    order = torch.sort(key, stable=True).indices
    sorted_keys = key[order]
    cnt_start = torch.cumsum(counts, 0) - counts
    rank = torch.arange(numel, dtype=torch.int64, device=dev) - cnt_start[sorted_keys]
    dest = (
        pad_start[sorted_keys] + rank
    )  # skipped rows land past max_pad and are cut off
    buf = torch.full((max_pad + numel,), numel, dtype=torch.int32, device=dev)
    buf.scatter_(0, dest, order.to(torch.int32))
    sorted_ids = buf[:max_pad]
    max_blocks = -(-max_pad // block_size)
    starts = torch.arange(max_blocks, dtype=torch.int64, device=dev) * block_size
    total = pad_end[-1:]
    expert_ids = torch.searchsorted(pad_end, starts, right=True)
    expert_ids = torch.where(
        starts < total, expert_ids, torch.zeros_like(expert_ids)
    ).to(torch.int32)
    return sorted_ids, expert_ids, total.to(torch.int32)


# One-kernel version for decode-sized batches (21 Nov 2026): the torch version above is ~30 small kernels (~80 us a
# call on a 170HX), and it runs in each of the 48 MoE passes of a step (12 looped layers x 3 + 12), ~4 ms of a ~19 ms
# step at 8 tokens. One program sorts (expert * BLOCK + row) keys in registers (stable by row), so this covers up to
# _TRITON_MAX routed rows; bigger (prefill) batches keep the torch version.
import triton
import triton.language as tl

_TRITON_MAX = 4096


@triton.jit
def _moe_align_small_kernel(
    ids_ptr,
    sorted_ptr,
    expert_ids_ptr,
    total_ptr,
    numel,
    max_pad,
    max_blocks,
    E: tl.constexpr,
    EP: tl.constexpr,
    BS: tl.constexpr,
    BLOCK: tl.constexpr,
    CH: tl.constexpr,
    NBC: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    ids = tl.load(ids_ptr + i, mask=i < numel, other=E)
    key = tl.where((ids >= 0) & (ids < E), ids, E).to(tl.int32)
    bins = tl.arange(0, EP)
    counts = tl.histogram(key, EP)
    padded = tl.where(bins < E, (counts + BS - 1) // BS * BS, 0)
    pad_end = tl.cumsum(padded, 0)
    pad_start = pad_end - padded
    cnt_start = tl.cumsum(counts, 0) - counts
    total = tl.sum(padded, 0)
    for c in range(0, max_pad, CH):
        o = c + tl.arange(0, CH)
        tl.store(sorted_ptr + o, tl.zeros((CH,), tl.int32) + numel, mask=o < max_pad)
    tl.debug_barrier()
    s = tl.sort(key * BLOCK + i)
    e_s = s // BLOCK
    dest = tl.gather(pad_start, e_s, 0) + (i - tl.gather(cnt_start, e_s, 0))
    tl.store(sorted_ptr + dest, s % BLOCK, mask=e_s < E)
    for c in range(0, max_blocks, NBC):
        b = c + tl.arange(0, NBC)
        start = b * BS
        before = (pad_end[None, :] <= start[:, None]) & (bins[None, :] < E)
        ex = tl.where(start < total, tl.sum(before.to(tl.int32), 1), 0)
        tl.store(expert_ids_ptr + b, ex, mask=b < max_blocks)
    tl.store(total_ptr, total)


def moe_align_block_size_fast(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
):
    numel = topk_ids.numel()
    if numel > _TRITON_MAX:
        if num_experts <= 256 and topk_ids.dtype == torch.int32:
            try:
                from vllm.models.inkling2.nvidia.ops.sm80.inkling_sm80_cuda import (
                    HAS_CUDA_EXT,
                    moe_align_large_cuda,
                )

                if HAS_CUDA_EXT:
                    return moe_align_large_cuda(topk_ids, block_size, num_experts)
            except Exception:
                pass
        return moe_align_block_size_torch(topk_ids, block_size, num_experts)
    max_pad = numel + num_experts * (block_size - 1)
    if numel < num_experts:
        max_pad = min(numel * block_size, max_pad)
    max_blocks = -(-max_pad // block_size)
    dev = topk_ids.device
    sorted_ids = torch.empty((max_pad,), dtype=torch.int32, device=dev)
    expert_ids = torch.empty((max_blocks,), dtype=torch.int32, device=dev)
    total = torch.empty((1,), dtype=torch.int32, device=dev)
    block = max(triton.next_power_of_2(numel), 16)
    _moe_align_small_kernel[(1,)](
        topk_ids.reshape(-1),
        sorted_ids,
        expert_ids,
        total,
        numel,
        max_pad,
        max_blocks,
        E=num_experts,
        EP=triton.next_power_of_2(num_experts + 1),
        BS=block_size,
        BLOCK=block,
        CH=1024,
        NBC=32,
        num_warps=4 if block <= 1024 else 8,
    )
    return sorted_ids, expert_ids, total
