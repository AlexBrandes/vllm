# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperInkling W8A16 linears: bf16 weights quantized at load time to int8
(symmetric, one bf16 scale per 128 inputs of each output channel), run by
a Triton kernel derived from `ops.gemv`'s.

The weight keeps the natural `[N, K]` layout, so slicing rows stays free
and full layers can project q, k and v (different inputs) in one launch:
`W8Linear.multi` stacks the inputs, reads the weight once and returns each
input's column range. Decode is bandwidth bound, so the extra rows cost
little. Prefill (many rows) runs the same kernel with larger row tiles.

Group-128 int8 round-to-nearest needs no calibration and works on the
released checkpoint as is.
"""

import os

import torch

from vllm.triton_utils import tl, triton

GROUP = 128


def quantize_int8(weight: torch.Tensor):
    """bf16 `[N, K]` -> int8 codes `[N, K]` and bf16 scales `[N, K / GROUP]`."""
    g = weight.float().unflatten(1, (-1, GROUP))
    scale = (g.abs().amax(-1) / 127.0).clamp_min(1e-10).to(torch.bfloat16)
    q = (g / scale.float()[..., None]).round().clamp(-127, 127)
    return q.flatten(1).to(torch.int8), scale


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (q.float().unflatten(1, (-1, GROUP)) * scale.float()[..., None]).flatten(1)


@triton.jit
def _w8_kernel(
    X, W, S, Y, M, N, K, SX, SW, SS, SY,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):  # fmt: skip
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    col_ok = cols < N
    acc = tl.zeros((BM, BN), tl.float32)
    for g in range(0, tl.cdiv(K, BK)):
        ks = g * BK + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            col_ok[None, :],
            other=0,
        )
        x = tl.load(
            X + rows[:, None].to(tl.int64) * SX + ks[None, :],
            rows[:, None] < M,
            other=0,
        )
        s = tl.load(S + cols.to(tl.int64) * SS + g, col_ok, other=0).to(tl.float32)
        acc += tl.dot(x, w.to(tl.bfloat16)) * s[None, :]
    tl.store(
        Y + rows[:, None].to(tl.int64) * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & col_ok[None, :],
    )


@triton.jit
def _w8_splitk_kernel(
    X, W, S, P, M, N, K, SX, SW, SS, SPK, SPM, GROUPS_PER_SPLIT,
    BM: tl.constexpr, BN: tl.constexpr, G: tl.constexpr,
):  # fmt: skip
    # Decode / verify (M <= 64): every row in one tile; program (n, k) covers one
    # column tile and one K range and writes its fp32 partial to P[k] (no atomics:
    # the reduce kernel sums the splits in a fixed order, so results do not depend
    # on scheduling). The split count depends only on N and K, so a row's
    # arithmetic is the same at any M (batch invariant).
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    col_ok = cols < N
    row_ok = rows[:, None] < M
    ks = tl.arange(0, G)
    acc = tl.zeros((BM, BN), tl.float32)
    g0 = pid_k * GROUPS_PER_SPLIT
    g1 = tl.minimum(g0 + GROUPS_PER_SPLIT, tl.cdiv(K, G))
    for g in range(g0, g1):
        w = tl.load(
            W + cols[:, None].to(tl.int64) * SW + (g * G + ks)[None, :],
            col_ok[:, None],
            other=0,
        )
        s = tl.load(S + cols.to(tl.int64) * SS + g, col_ok, other=0).to(tl.float32)
        x = tl.load(
            X + rows[:, None].to(tl.int64) * SX + g * G + ks[None, :], row_ok, other=0
        )
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16))) * s[None, :]
    tl.store(
        P + pid_k.to(tl.int64) * SPK + rows[:, None].to(tl.int64) * SPM + cols[None, :],
        acc,
        row_ok & col_ok[None, :],
    )


@triton.jit
def _w8_gemv_kernel(
    X, W, S, Y, M, N, K, SX, SW, SS, SY,
    BN: tl.constexpr, G: tl.constexpr, MP: tl.constexpr,
):  # fmt: skip
    # M <= MP rows, no tensor cores: each program owns BN output channels and
    # walks all of K in 128-wide groups (one scale each), int8 -> fp32 FMA with
    # fp32 accumulation. Avoids the 16-row MMA padding and the shared-memory
    # operand round trip that make the dot kernel latency bound at M = 1.
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    col_ok = cols < N
    ks = tl.arange(0, G)
    ms = tl.arange(0, MP)
    acc = tl.zeros((BN, MP), tl.float32)
    for g in range(0, tl.cdiv(K, G)):
        w = tl.load(
            W + cols[:, None].to(tl.int64) * SW + (g * G + ks)[None, :],
            col_ok[:, None],
            other=0,
        ).to(tl.float32)
        s = tl.load(S + cols.to(tl.int64) * SS + g, col_ok, other=0).to(tl.float32)
        x = tl.load(
            X + ms[:, None].to(tl.int64) * SX + (g * G + ks)[None, :],
            ms[:, None] < M,
            other=0,
        ).to(tl.float32)
        part = tl.sum(w[:, None, :] * x[None, :, :], axis=2)
        acc += part * s[:, None]
    tl.store(
        Y + ms[None, :].to(tl.int64) * SY + cols[:, None],
        acc.to(Y.dtype.element_ty),
        col_ok[:, None] & (ms[None, :] < M),
    )


# FMA GEMV path for M <= W8_GEMV_MAX_M (VLLM_INKLING2_W8_GEMV=1).
W8_GEMV = os.environ.get("VLLM_INKLING2_W8_GEMV", "0") == "1"
W8_GEMV_MAX_M = int(os.environ.get("VLLM_INKLING2_W8_GEMV_MAX_M", "4"))
W8_GEMV_BN = int(os.environ.get("VLLM_INKLING2_W8_GEMV_BN", "8"))
W8_GEMV_WARPS = int(os.environ.get("VLLM_INKLING2_W8_GEMV_WARPS", "4"))
W8_GEMV_STAGES = int(os.environ.get("VLLM_INKLING2_W8_GEMV_STAGES", "3"))


def w8_matmul_gemv(
    x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, k = x.shape
    n = q.shape[0]
    mp = max(1, triton.next_power_of_2(m))
    _w8_gemv_kernel[(triton.cdiv(n, W8_GEMV_BN),)](
        x, q, scale, out, m, n, k,
        x.stride(0), q.stride(0), scale.stride(0), out.stride(0),
        BN=W8_GEMV_BN, G=GROUP, MP=mp,
        num_warps=W8_GEMV_WARPS, num_stages=W8_GEMV_STAGES,
    )  # fmt: skip
    return out


@triton.jit
def _w8_reduce_kernel(P, Y, M, N, SPK, SPM, SY, SPLITS, BLOCK: tl.constexpr):
    row = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = cols < N
    acc = tl.zeros((BLOCK,), tl.float32)
    for s in range(0, SPLITS):
        acc += tl.load(P + s * SPK + row * SPM + cols, ok, other=0.0)
    tl.store(Y + row.to(tl.int64) * SY + cols, acc.to(Y.dtype.element_ty), ok)


# Split-K decode path (VLLM_INKLING2_W8_SPLITK=1). Tunables for the CMP 170HX
# sweep; the defaults are set from that sweep.
W8_SPLITK = os.environ.get("VLLM_INKLING2_W8_SPLITK", "0") == "1"
W8_SK_BN = int(os.environ.get("VLLM_INKLING2_W8_SK_BN", "64"))
W8_SK_WARPS = int(os.environ.get("VLLM_INKLING2_W8_SK_WARPS", "4"))
W8_SK_STAGES = int(os.environ.get("VLLM_INKLING2_W8_SK_STAGES", "3"))
W8_SK_PROGRAMS_PER_SM = int(os.environ.get("VLLM_INKLING2_W8_SK_PPSM", "4"))
# Rows at which W8Linear(keep_bf16=True) switches to the bf16 weight (0 = off).
W8_PREFILL_BF16_MIN_M = int(os.environ.get("VLLM_INKLING2_W8_PREFILL_BF16_MIN_M", "0"))
_NUM_SMS: int | None = None


def _num_sms(device: torch.device) -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS


def w8_splitk_plan(n: int, k: int, device: torch.device) -> tuple[int, int, int]:
    """(column tiles, splits, groups per split): a function of N and K only."""
    groups = k // GROUP
    tiles = triton.cdiv(n, W8_SK_BN)
    programs = W8_SK_PROGRAMS_PER_SM * _num_sms(device)
    splits = max(1, min(groups, triton.cdiv(programs, tiles)))
    per_split = triton.cdiv(groups, splits)
    return tiles, triton.cdiv(groups, per_split), per_split


def w8_matmul_splitk(
    x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, k = x.shape
    n = q.shape[0]
    tiles, splits, per_split = w8_splitk_plan(n, k, x.device)
    bm = max(16, triton.next_power_of_2(m))
    part = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    _w8_splitk_kernel[(tiles, splits)](
        x, q, scale, part, m, n, k,
        x.stride(0), q.stride(0), scale.stride(0), part.stride(0), part.stride(1),
        per_split, BM=bm, BN=W8_SK_BN, G=GROUP,
        num_warps=W8_SK_WARPS, num_stages=W8_SK_STAGES,
    )  # fmt: skip
    block = 1024
    _w8_reduce_kernel[(triton.cdiv(n, block), m)](
        part, out, m, n, part.stride(0), part.stride(1), out.stride(0), splits,
        BLOCK=block, num_warps=4,
    )  # fmt: skip
    return out


def w8_matmul(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """`x [M, K] bf16 @ dequant(q, scale).T` -> `[M, N]` bf16."""
    m, k = x.shape
    n = q.shape[0]
    assert k % GROUP == 0 and q.shape[1] == k and x.stride(1) == 1
    if out is None:
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0:
        return out
    if W8_GEMV and m <= W8_GEMV_MAX_M:
        return w8_matmul_gemv(x, q, scale, out)
    if W8_SPLITK and m <= 64:
        return w8_matmul_splitk(x, q, scale, out)
    # Tiles tuned on an RTX 3090 Ti (L2-cold weights; tests/test_linear_w8.py
    # covers every branch). Decode: one row tile, column tiles narrow enough
    # to spread the weight over every SM.
    # Verify steps (2-64 rows: requests x (1 + drafts)) have their own tiles:
    # the prefill tile at 48 rows ran `wo` at 84 GB/s instead of 334.
    wide = n >= 4096
    if m == 1:
        bm, bn, warps, stages = (16, 32, 4, 3) if wide else (16, 16, 2, 4)
    elif m <= 16:
        bm, bn, warps, stages = (16, 128, 8, 3) if wide else (16, 32, 4, 4)
    elif m <= 32:
        bm, bn, warps, stages = (32, 128, 8, 3) if wide else (32, 32, 8, 3)
    elif m <= 64:
        bm, bn, warps, stages = (64, 128, 8, 3) if wide else (64, 32, 4, 4)
    else:
        bm, bn, warps, stages = 64, 256, 8, 3
    grid = (triton.cdiv(n, bn), triton.cdiv(m, bm))
    _w8_kernel[grid](
        x, q, scale, out, m, n, k,
        x.stride(0), q.stride(0), scale.stride(0), out.stride(0),
        BM=bm, BN=bn, BK=GROUP, num_warps=warps, num_stages=stages,
    )  # fmt: skip
    return out


class W8Linear:
    """`x @ weight.T` for a bf16 `weight [N, K]`, as int8 W8A16."""

    def __init__(self, weight: torch.Tensor, keep_bf16: bool = False) -> None:
        self.size_n, self.size_k = weight.shape
        self.q, self.scale = quantize_int8(weight)
        # Prefill (M >= W8_PREFILL_BF16_MIN_M) on the original bf16 weight via
        # cuBLAS, which beats the dequantizing Triton kernel at large M.
        self.bf16 = weight.contiguous() if keep_bf16 else None

    def __call__(
        self, x: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.bf16 is not None and x.shape[0] >= W8_PREFILL_BF16_MIN_M:
            if out is None:
                return torch.nn.functional.linear(x, self.bf16)
            return torch.matmul(x, self.bf16.t(), out=out)
        return w8_matmul(x.contiguous(), self.q, self.scale, out)

    def multi(
        self, inputs: list[torch.Tensor], col_ranges: list[tuple[int, int]]
    ) -> list[torch.Tensor]:
        """`inputs[i] @ weight[col_ranges[i]].T` for each i, reading the weight
        once when the stacked rows fit one decode tile, else one launch each."""
        rows = [x.shape[0] for x in inputs]
        if self.bf16 is not None and min(rows) >= W8_PREFILL_BF16_MIN_M:
            return [
                torch.nn.functional.linear(x, self.bf16[lo:hi])
                for x, (lo, hi) in zip(inputs, col_ranges)
            ]
        if sum(rows) <= 32:
            y = w8_matmul(torch.cat(inputs), self.q, self.scale)
            out, start = [], 0
            for r, (lo, hi) in zip(rows, col_ranges):
                out.append(y[start : start + r, lo:hi])
                start += r
            return out
        return [
            w8_matmul(x.contiguous(), self.q[lo:hi], self.scale[lo:hi])
            for x, (lo, hi) in zip(inputs, col_ranges)
        ]
