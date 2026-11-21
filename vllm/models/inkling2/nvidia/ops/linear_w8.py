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

    def __init__(self, weight: torch.Tensor) -> None:
        self.size_n, self.size_k = weight.shape
        self.q, self.scale = quantize_int8(weight)

    def __call__(
        self, x: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        return w8_matmul(x.contiguous(), self.q, self.scale, out)

    def multi(
        self, inputs: list[torch.Tensor], col_ranges: list[tuple[int, int]]
    ) -> list[torch.Tensor]:
        """`inputs[i] @ weight[col_ranges[i]].T` for each i, reading the weight
        once when the stacked rows fit one decode tile, else one launch each."""
        rows = [x.shape[0] for x in inputs]
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
