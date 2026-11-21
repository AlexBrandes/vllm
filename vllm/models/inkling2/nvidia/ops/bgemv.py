# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperInkling batched low-M GEMV: `out[s] = x[s] @ w[s].T` for a few
streams `s`, at most MAX_M rows each (decode), with an fp32 or bf16 result.

`ops.gemv`'s kernel with the stream as a second grid axis. It replaces the
loop controllers' `torch.bmm`s (rank-64 down / up projections): at decode
sizes cuBLAS runs them on its wmma kernels, latency-bound at 20-30 us for
well under a megabyte of weights.
"""

import torch

from vllm.triton_utils import tl, triton

MAX_M = 16


@triton.jit
def _bgemv_kernel(
    X, W, Y, M, N, K,
    SXS, SXM, SWS, SWN, SYS, SYM,
    BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    s = tl.program_id(1).to(tl.int64)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + s * SWS + cols[None, :].to(tl.int64) * SWN + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + s * SXS + rows[:, None] * SXM + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    tl.store(
        Y + s * SYS + rows[:, None] * SYM + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _down_split_kernel(
    X, W, P, M, N, K, KS,
    SXS, SXM, SWS, SWN,
    BN: tl.constexpr, BK: tl.constexpr, MP: tl.constexpr,
):  # fmt: skip
    """fp32 partials P[split, s, MP, N] of x[s] @ w[s].T over K slice `split`."""
    pid = tl.program_id(0)
    s = tl.program_id(1)
    split = tl.program_id(2)
    num_s = tl.num_programs(1)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    k_lo = split * KS
    for k0 in range(k_lo, k_lo + KS, BK):
        ks = k0 + tl.arange(0, BK)
        k_ok = ks < tl.minimum(k_lo + KS, K)
        w = tl.load(
            W + s.to(tl.int64) * SWS + cols[None, :].to(tl.int64) * SWN + ks[:, None],
            (cols[None, :] < N) & k_ok[:, None],
            other=0,
        )
        x = tl.load(
            X + s.to(tl.int64) * SXS + rows[:, None] * SXM + ks[None, :],
            (rows[:, None] < M) & k_ok[None, :],
            other=0,
        )
        acc += tl.dot(x, w)
    base = (split * num_s + s).to(tl.int64) * MP * N
    tl.store(P + base + rows[:, None] * N + cols[None, :], acc, cols[None, :] < N)


@triton.jit
def _up_from_partials_kernel(
    P, W, Y, M, N, R, SPLITS,
    SWS, SWN, SYS, SYM,
    BN: tl.constexpr, BR: tl.constexpr, MP: tl.constexpr,
):  # fmt: skip
    """y[s] = bf16(sum of the down partials)[s] @ w[s].T, w (S, N, R)."""
    pid = tl.program_id(0)
    s = tl.program_id(1)
    num_s = tl.num_programs(1)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    rs = tl.arange(0, BR)
    hidden = tl.zeros((MP, BR), tl.float32)
    for split in range(SPLITS):
        base = (split * num_s + s).to(tl.int64) * MP * R
        hidden += tl.load(
            P + base + rows[:, None] * R + rs[None, :], rs[None, :] < R, other=0.0
        )
    w = tl.load(
        W + s.to(tl.int64) * SWS + cols[None, :].to(tl.int64) * SWN + rs[:, None],
        (cols[None, :] < N) & (rs[:, None] < R),
        other=0,
    )
    # The reference rounds the rank-r hidden state to bf16 (its bmm output).
    acc = tl.dot(hidden.to(tl.bfloat16), w)
    tl.store(
        Y + s.to(tl.int64) * SYS + rows[:, None] * SYM + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


DOWN_SPLITS = 8


def controller_gemv(
    x: torch.Tensor, down: torch.Tensor, up: torch.Tensor
) -> torch.Tensor:
    """The loop controllers' `bmm(bmm(x, down.T), up.T)` in fp32 for decode
    rows: x (S, M, K) bf16, down (S, R, K), up (S, N, R) -> (S, M, N) fp32.

    The wide-K, rank-R down projection runs in DOWN_SPLITS K slices (more
    programs in flight than rank / tile alone gives); the up projection sums
    the slices, in a fixed order, as it loads them."""
    s, m, k = x.shape
    r, n = down.shape[1], up.shape[1]
    assert m <= MAX_M and down.shape == (s, r, k) and up.shape == (s, n, r)
    assert x.stride(2) == 1 and down.stride(2) == 1 and up.stride(2) == 1
    assert r <= 128
    out = torch.empty((s, m, n), dtype=torch.float32, device=x.device)
    if m == 0:
        return out
    partials = torch.empty(
        (DOWN_SPLITS, s, 16, r), dtype=torch.float32, device=x.device
    )
    ks = triton.cdiv(triton.cdiv(k, DOWN_SPLITS), 64) * 64
    _down_split_kernel[(triton.cdiv(r, 16), s, DOWN_SPLITS)](
        x, down, partials, m, r, k, ks,
        x.stride(0), x.stride(1), down.stride(0), down.stride(1),
        BN=16, BK=64, MP=16, num_warps=4,
    )  # fmt: skip
    _up_from_partials_kernel[(triton.cdiv(n, 64), s)](
        partials, up, out, m, n, r, DOWN_SPLITS,
        up.stride(0), up.stride(1), out.stride(0), out.stride(1),
        BN=64, BR=triton.next_power_of_2(r), MP=16, num_warps=4,
    )  # fmt: skip
    return out


def bgemv(
    x: torch.Tensor, w: torch.Tensor, out_dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """bf16 `x (S, M, K)`, `w (S, N, K)` -> `(S, M, N)` of `out_dtype`."""
    s, m, k = x.shape
    n = w.shape[1]
    assert m <= MAX_M and w.shape == (s, n, k)
    assert x.stride(2) == 1 and w.stride(2) == 1
    out = torch.empty((s, m, n), dtype=out_dtype, device=x.device)
    if m == 0:
        return out
    # Narrow K (the up projection, K = rank) wants wide column tiles; wide K
    # (the down projection, N = rank) narrow ones, to spread over the SMs.
    bk = min(256, triton.next_power_of_2(k))
    bn = 64 if k <= 128 else 16
    _bgemv_kernel[(triton.cdiv(n, bn), s)](
        x, w, out, m, n, k,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BN=bn, BK=bk, MP=16, num_warps=4,
    )  # fmt: skip
    return out
