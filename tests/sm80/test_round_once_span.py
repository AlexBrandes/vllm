# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The exactness precondition must reject collectively, not strand peers."""
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.sm80.test_round_once_all_reduce import exact_sum, reference_round
from vllm.sm80.distributed.round_once_all_reduce import round_once_all_reduce_with


def _worker(rank, init_file, errors):
    world = 4
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=world, timeout=timedelta(seconds=60))

    def exchange(sends, recvs):
        ops = []
        for peer in range(world):
            if peer != rank:
                ops += [dist.P2POp(dist.isend, sends[peer], peer),
                        dist.P2POp(dist.irecv, recvs[peer], peer)]
        for work in dist.batch_isend_irecv(ops):
            work.wait()

    def gather(out, chunk):
        dist.all_gather(list(out.view(world, -1).unbind()), chunk.contiguous())

    cases = [
        ("boundary", torch.bfloat16, [1, 2.0**-43, -1, 0], False),
        ("outside", torch.bfloat16, [1, 2.0**-44, -1, 0], True),
        ("lost", torch.bfloat16, [1, 2.0**-60, -1, 0], True),
        ("zeros", torch.bfloat16, [0, 0, 0, 0], False),
        ("negative_zero", torch.bfloat16, [-0.0] * world, False),
        ("fp16_range", torch.float16, [65504, 2.0**-24, -65504, 0], False),
    ]
    for dtype in (torch.bfloat16, torch.float16):
        for value in (float("inf"), -float("inf"), float("nan")):
            cases.append((str(value), dtype, [0, value, 0, 0], True))
    try:
        for name, dtype, values, reject in cases:
            # Only rank 2 owns the offending element after reduce-scatter.
            # Other columns have unrelated scales and cannot share a bound.
            parts = torch.zeros((world, 13), dtype=dtype)
            parts[:, 8] = torch.tensor(values, dtype=dtype)
            parts[0, 1] = 2.0**-24
            parts[3, 11] = 1024
            original = parts[rank].clone()
            try:
                got = round_once_all_reduce_with(parts[rank], rank, world, exchange, gather)
            except ValueError as error:
                if not reject or "finite exponent span" not in str(error):
                    errors.put((rank, name, str(error)))
            else:
                if reject:
                    errors.put((rank, name, "unsupported input accepted"))
                else:
                    want = reference_round(exact_sum(parts), dtype)
                    if not torch.equal(got.view(torch.int16), want.view(torch.int16)):
                        errors.put((rank, name, "valid input changed bits"))
            if not torch.equal(original.view(torch.int16), parts[rank].view(torch.int16)):
                errors.put((rank, name, "input modified"))
            # Every rank must have completed the same collectives on refusal.
            alive = torch.ones(1)
            dist.all_reduce(alive)
            if alive.item() != world:
                errors.put((rank, name, "communicator unusable after refusal"))
    finally:
        dist.destroy_process_group()


def test_span_refusal_is_collective_safe(tmp_path):
    ctx = mp.get_context("spawn")
    errors = ctx.Queue()
    mp.start_processes(_worker, args=(str(tmp_path / "gloo"), errors),
                       nprocs=4, join=True, start_method="spawn")
    found = []
    while not errors.empty():
        found.append(errors.get())
    assert not found, found


def test_single_rank_nonzero_columns_need_no_common_scale():
    t = torch.tensor([1, 2.0**-60, -0.0], dtype=torch.bfloat16)
    def unused(*args):
        pytest.fail("single rank must not call a transport")
    got = round_once_all_reduce_with(t, 0, 1, unused, unused)
    assert torch.equal(got.view(torch.int16), t.view(torch.int16))


def test_guard_uses_only_the_existing_data_collective():
    calls = []
    def exchange(sends, recvs):
        recvs[1].copy_(sends[1])
    def gather(out, chunk):
        calls.append(chunk.numel())
        out.view(2, -1).copy_(chunk)
    round_once_all_reduce_with(torch.ones(4, dtype=torch.bfloat16), 0, 2, exchange, gather)
    assert calls == [2], "the guard must travel in the existing data all-gather"
