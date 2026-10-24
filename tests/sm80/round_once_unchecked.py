# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Timing-only control: identical fused transport without numerical validation.

Never imported by serving. The factory first sends the same immutable input
through the checked path; its deferred error must be consumed at the caller's
existing synchronization before accepting any measurements. Only the checked
transport can gate a numerical or performance decision.
"""
import torch
import triton

from vllm.sm80.distributed.round_once_all_reduce import nccl_round_once_all_reduce
from vllm.sm80.distributed.round_once_kernels import _sum_round


def make_unchecked_control(t, pynccl):
    nccl_round_once_all_reduce(t, pynccl)
    rank, world = pynccl.rank, pynccl.world_size
    numel = t.numel()
    chunk = triton.cdiv(numel, world)
    span = (45 if t.dtype == torch.bfloat16 else 42) - (world - 1).bit_length()

    def run():
        flat = t.reshape(-1)
        if chunk * world != numel:
            padded = torch.zeros(chunk * world, dtype=t.dtype, device=t.device)
            padded[:numel] = flat
            flat = padded
        recv = torch.empty((world, chunk), dtype=t.dtype, device=t.device)
        sends = list(flat.view(world, chunk).unbind())
        recvs = list(recv.unbind())
        recvs[rank].copy_(sends[rank])
        stream = torch.cuda.current_stream()
        pynccl.group_start()
        for peer in range(world):
            if peer != rank:
                pynccl.send(sends[peer], peer, stream)
                pynccl.recv(recvs[peer], peer, stream)
        pynccl.group_end()
        mine = torch.empty(chunk, dtype=t.dtype, device=t.device)
        _sum_round[(triton.cdiv(chunk, 256),)](recv, mine, chunk, world, 2.0**span, False, 256)
        out = torch.empty(chunk * world, dtype=t.dtype, device=t.device)
        pynccl.all_gather(out, mine, stream)
        return out[:numel].view(t.shape)
    return run
