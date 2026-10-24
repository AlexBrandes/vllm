# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The round-once all-reduce over gloo (CPU): every element is the exact sum of
the ranks' partials correctly rounded to the message dtype, on every rank, for
TP 2, 4 and 8 and odd sizes (vllm/sm80/distributed/round_once_all_reduce.py).

The reference rounding here is independent of the module's: the two
neighbours of the exact FP64 sum in the target format and round to nearest,
ties to even, decided on exact FP64 differences.

    pytest -q tests/sm80/test_round_once_all_reduce.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.sm80.distributed import round_once_all_reduce as R

BF16, FP16 = torch.bfloat16, torch.float16
_INT = {BF16: torch.int16, FP16: torch.int16}


def _step(v: torch.Tensor, up: bool) -> torch.Tensor:
    """The next representable value of v's format above (``up``) or below."""
    i = v.view(_INT[v.dtype]).long()
    width = 8 * v.element_size()
    neg = i < 0
    mag = torch.where(neg, i + (1 << (width - 1)), i)  # magnitude bits
    signed = torch.where(neg, -mag, mag)  # ordered integer: -0 -> 0
    signed = signed + (1 if up else -1)
    sign = signed < 0
    mag = signed.abs()
    i = torch.where(sign, mag - (1 << (width - 1)), mag)
    return i.to(_INT[v.dtype]).view(v.dtype)


def reference_round(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """FP64 ``x`` rounded to nearest, ties to even, in ``dtype``."""
    b = x.to(dtype)
    lo = torch.where(b.double() > x, _step(b, up=False), b)
    hi = _step(lo, up=True)
    d_lo = x - lo.double()
    d_hi = hi.double() - x
    even_lo = (lo.view(_INT[dtype]) & 1) == 0
    pick_lo = (d_lo < d_hi) | ((d_lo == d_hi) & even_lo)
    out = torch.where(pick_lo, lo, hi)
    # a nonzero sum that rounds to zero keeps its sign (IEEE), an exact zero
    # is the FP64 sum's own zero
    signed_zero = torch.where(x < 0, torch.zeros_like(out).neg(), torch.zeros_like(out))
    zero = torch.where(x == 0, x.to(dtype), signed_zero)
    return torch.where(out == 0, zero, out)


def exact_sum(parts: torch.Tensor) -> torch.Tensor:
    """The FP64 sum of the ranks' partials (exact for these inputs), added
    from the first partial on so that -0 + -0 stays -0 as IEEE says (a
    reduction from a +0 identity would give +0)."""
    acc = parts[0].double()
    for p in parts[1:]:
        acc = acc + p.double()
    return acc


def _partials(case: str, world: int, shape, dtype, seed: int) -> torch.Tensor:
    """[world, *shape] partials of every rank (each rank builds them all)."""
    orig = tuple(shape)
    g = torch.Generator().manual_seed(seed)
    rows = shape[0] if len(shape) > 1 else 1
    if len(shape) == 1:
        shape = (1, shape[0])
    p = torch.randn((world, *shape), generator=g, dtype=torch.float64)
    if case == "row_scales":  # A2-like: per-rank, per-row scales 2^[-6, 3]
        s = torch.randint(-6, 4, (world, rows), generator=g).double()
        p = p * torch.exp2(s.reshape((world, rows) + (1,) * (len(shape) - 1)))
    elif case == "wide":  # magnitudes [1, 2) * 2^[-span, span] per element
        span = 6 if dtype == FP16 else 15
        e = torch.randint(-span, span + 1, p.shape, generator=g).double()
        sign = torch.where(torch.rand(p.shape, generator=g) < 0.5, -1.0, 1.0)
        p = sign * (1 + torch.rand(p.shape, generator=g, dtype=torch.float64))
        p = p * torch.exp2(e)
    elif case == "cancel":  # rank pairs nearly cancel, leaving a small remainder
        n = p[1::2].shape[0]
        eps = torch.randn(p[1::2].shape, generator=g, dtype=torch.float64)
        p[1::2] = -p[0::2][:n] * (1 + 2.0**-6 * eps)
    elif case == "tiny" and dtype == BF16:  # BF16 subnormals and their sums
        p = p * 2.0**-130
    elif case == "zeros":
        p[torch.rand(p.shape, generator=g) < 0.5] = 0
    return p.to(dtype).reshape((world, *orig))


CASES = ["row_scales", "wide", "cancel", "tiny", "zeros"]
SHAPES = [(1,), (7,), (3, 333), (1023, 13), (5, 512)]


def _worker(rank, world, init_file, jobs, failures):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    try:
        for case, shape, dtype, seed in jobs:
            parts = _partials(case, world, shape, dtype, seed)

            def exchange(sends, recvs):
                ops = []
                for j in range(world):
                    if j != rank:
                        ops.append(dist.P2POp(dist.isend, sends[j].contiguous(), j))
                        ops.append(dist.P2POp(dist.irecv, recvs[j], j))
                for w in dist.batch_isend_irecv(ops):
                    w.wait()

            def all_gather(out, chunk):
                pieces = list(out.view(world, -1).unbind(0))
                dist.all_gather(pieces, chunk.contiguous())
                out.copy_(torch.cat(pieces))

            mine = parts[rank].clone()
            got = R.round_once_all_reduce_with(mine, rank, world, exchange, all_gather)
            exact = exact_sum(parts)
            want = reference_round(exact, dtype)
            same = got.view(_INT[dtype]) == want.view(_INT[dtype])
            if got.shape != parts.shape[1:] or not bool(same.all()):
                failures.put((rank, case, shape, str(dtype), int((~same).sum())))
            if not torch.equal(mine, parts[rank]):
                failures.put((rank, case, shape, str(dtype), "input modified"))
            # every rank holds the same bits
            bits = got.view(_INT[dtype]).int().contiguous()
            all_bits = [torch.empty_like(bits) for _ in range(world)]
            dist.all_gather(all_bits, bits)
            if any(not torch.equal(b, all_bits[0]) for b in all_bits):
                failures.put((rank, case, shape, str(dtype), "ranks differ"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("dtype", [BF16, FP16])
def test_round_once_is_the_correctly_rounded_exact_sum(tmp_path, world, dtype):
    jobs = [
        (case, shape, dtype, 1000 * i + j)
        for i, case in enumerate(CASES)
        for j, shape in enumerate(SHAPES)
    ]
    ctx = mp.get_context("spawn")
    failures = ctx.Queue()
    mp.start_processes(
        _worker,
        args=(world, str(tmp_path / "init"), jobs, failures),
        nprocs=world,
        join=True,
        start_method="spawn",
    )
    found = []
    while not failures.empty():
        found.append(failures.get())
    assert not found, found


@pytest.mark.parametrize("dtype", [BF16, FP16])
def test_round_from_fp64_matches_the_reference(dtype) -> None:
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1 << 16, generator=g, dtype=torch.float64)
    x = x * torch.exp2(torch.randint(-20, 21, x.shape, generator=g).double())
    if dtype == FP16:
        x = x.clamp(-60000, 60000)
    # exact ties and values on the format's grid
    on_grid = x.to(dtype).double()
    tie = (on_grid + _step(x.to(dtype), up=True).double()) / 2
    for v in (x, on_grid, tie, -tie):
        got = R.round_from_fp64(v, dtype)
        want = reference_round(v, dtype)
        assert torch.equal(got.view(_INT[dtype]), want.view(_INT[dtype]))


def test_ring_rounding_differs_where_round_once_does_not() -> None:
    """The reason for the module: a TP4 BF16 ring (each hop stored in BF16)
    misses the correctly rounded sum on many elements; round-once never does."""
    world = 4
    parts = _partials("row_scales", world, (64, 512), BF16, 3)
    acc = parts[0]
    for j in range(1, world):
        acc = (acc.float() + parts[j].float()).to(BF16)
    want = reference_round(exact_sum(parts), BF16)
    assert (acc.view(torch.int16) != want.view(torch.int16)).float().mean() > 0.1


def test_single_rank_and_unsupported_dtype() -> None:
    t = torch.randn(5, 3).to(BF16)
    got = R.round_once_all_reduce_with(t, 0, 1, None, None)
    assert torch.equal(got, t) and got.data_ptr() != t.data_ptr()
    for dtype in (torch.float32, torch.int32):
        with pytest.raises(TypeError):
            R.round_once_all_reduce_with(torch.zeros(4, dtype=dtype), 0, 1, None, None)


def test_grouped_takes_needs_a_cuda_tensor() -> None:
    comm = type("C", (), {"pynccl_comm": None})()
    assert R.grouped_round_once_takes(torch.zeros(8, dtype=BF16), comm) is False
