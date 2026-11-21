# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperInkling W4A16 linears: bf16 weights quantized at load time to int4
(asymmetric round-to-nearest, one bf16 scale and one zero point per 128
inputs of each output channel), run by a Triton kernel derived from
`linear_w8`'s.

Codes are packed two per byte along K (`[N, K / 2]` uint8, even k in the
low nibble), so rows stay contiguous and slicing rows is free, as for
W8Linear. The kernel does not interleave the nibbles: it multiplies the low
nibbles by the even input columns and the high nibbles by the odd ones.

For weights whose error only costs speed, not output (the DSpark drafter:
the target verifies every draft), round-to-nearest needs no calibration.
"""

import torch

from vllm.triton_utils import tl, triton

GROUP = 128
# Decode (m <= 64) split-K launch shape, tuned on an RTX 3090 Ti.
W4_DECODE_BN = 128
W4_DECODE_WARPS = 4
W4_DECODE_STAGES = 2
W4_PROGRAMS_PER_SM = 2


def quantize_int4(weight: torch.Tensor):
    """bf16 `[N, K]` -> packed codes `[N, K / 2]` uint8, bf16 scales and
    uint8 zero points `[N, K / GROUP]`."""
    g = weight.float().unflatten(1, (-1, GROUP))
    lo, hi = g.amin(-1), g.amax(-1)
    scale = ((hi - lo) / 15.0).clamp_min(1e-10).to(torch.bfloat16)
    s = scale.float()
    zero = (-lo / s).round().clamp(0, 15)
    q = ((g / s[..., None]).round() + zero[..., None]).clamp(0, 15).to(torch.uint8)
    q = q.flatten(1)
    packed = q[:, 0::2] | (q[:, 1::2] << 4)
    return packed.contiguous(), scale, zero.to(torch.uint8)


def dequantize_int4(
    packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor
) -> torch.Tensor:
    q = torch.stack([packed & 15, packed >> 4], dim=-1).flatten(1).float()
    g = q.unflatten(1, (-1, GROUP)) - zero.float()[..., None]
    return (g * scale.float()[..., None]).flatten(1)


@triton.jit
def _w4_kernel(
    X, W, S, Z, Y, M, N, K, SX, SW, SS, SZ, SY,
    BM: tl.constexpr, BN: tl.constexpr, G: tl.constexpr,
):  # fmt: skip
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    col_ok = cols < N
    row_ok = rows[:, None] < M
    half = tl.arange(0, G // 2)
    acc = tl.zeros((BM, BN), tl.float32)
    for g in range(0, tl.cdiv(K, G)):
        b = tl.load(
            W + cols[None, :].to(tl.int64) * SW + (g * (G // 2) + half)[:, None],
            col_ok[None, :],
            other=0,
        )
        z = tl.load(Z + cols.to(tl.int64) * SZ + g, col_ok, other=0).to(tl.float32)
        s = tl.load(S + cols.to(tl.int64) * SS + g, col_ok, other=0).to(tl.float32)
        lo = ((b & 15).to(tl.float32) - z[None, :]).to(tl.bfloat16)
        hi = ((b >> 4).to(tl.float32) - z[None, :]).to(tl.bfloat16)
        xk = rows[:, None].to(tl.int64) * SX + g * G + 2 * half[None, :]
        x_even = tl.load(X + xk, row_ok, other=0)
        x_odd = tl.load(X + xk + 1, row_ok, other=0)
        part = tl.dot(x_even, lo)
        part = tl.dot(x_odd, hi, part)
        acc += part * s[None, :]
    tl.store(
        Y + rows[:, None].to(tl.int64) * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        row_ok & col_ok[None, :],
    )


@triton.jit
def _w4_splitk_kernel(
    X, W, S, Z, Y, M, N, K, SX, SW, SS, SZ, SY, GROUPS_PER_SPLIT,
    BM: tl.constexpr, BN: tl.constexpr, G: tl.constexpr,
):  # fmt: skip
    # Decode: all rows in one tile, each program one column tile and one K
    # range, fp32 partials added into Y. Weights load as [BN, G / 2] (K
    # contiguous) and the activations as [BM, G], split into even/odd columns.
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    col_ok = cols < N
    row_ok = rows[:, None] < M
    half = tl.arange(0, G // 2)
    ks = tl.arange(0, G)
    acc = tl.zeros((BM, BN), tl.float32)
    g0 = pid_k * GROUPS_PER_SPLIT
    g1 = tl.minimum(g0 + GROUPS_PER_SPLIT, tl.cdiv(K, G))
    for g in range(g0, g1):
        b = tl.load(
            W + cols[:, None].to(tl.int64) * SW + (g * (G // 2) + half)[None, :],
            col_ok[:, None],
            other=0,
        )
        z = tl.load(Z + cols.to(tl.int64) * SZ + g, col_ok, other=0).to(tl.float32)
        s = tl.load(S + cols.to(tl.int64) * SS + g, col_ok, other=0).to(tl.float32)
        lo = ((b & 15).to(tl.float32) - z[:, None]).to(tl.bfloat16)
        hi = ((b >> 4).to(tl.float32) - z[:, None]).to(tl.bfloat16)
        x = tl.load(
            X + rows[:, None].to(tl.int64) * SX + g * G + ks[None, :], row_ok, other=0
        )
        x_even, x_odd = tl.split(tl.reshape(x, (BM, G // 2, 2)))
        part = tl.dot(x_even, tl.trans(lo))
        part = tl.dot(x_odd, tl.trans(hi), part)
        acc += part * s[None, :]
    tl.atomic_add(
        Y + rows[:, None].to(tl.int64) * SY + cols[None, :],
        acc,
        row_ok & col_ok[None, :],
        sem="relaxed",
    )


_NUM_SMS: int | None = None


def _num_sms(device: torch.device) -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS


def w4_matmul(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """`x [M, K] bf16 @ dequant(packed, scale, zero).T` -> `[M, N]` bf16."""
    m, k = x.shape
    n = packed.shape[0]
    assert k % GROUP == 0 and packed.shape[1] * 2 == k and x.stride(1) == 1
    if out is None:
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0:
        return out
    if m <= 64:
        bm = max(16, triton.next_power_of_2(m))
        bn = W4_DECODE_BN
        groups = k // GROUP
        tiles = triton.cdiv(n, bn)
        programs = W4_PROGRAMS_PER_SM * _num_sms(x.device)
        splits = min(groups, triton.cdiv(programs, tiles))
        per_split = triton.cdiv(groups, splits)
        splits = triton.cdiv(groups, per_split)
        acc = torch.zeros((m, n), dtype=torch.float32, device=x.device)
        _w4_splitk_kernel[(tiles, splits)](
            x, packed, scale, zero, acc, m, n, k,
            x.stride(0), packed.stride(0), scale.stride(0), zero.stride(0),
            acc.stride(0), per_split, BM=bm, BN=bn, G=GROUP,
            num_warps=W4_DECODE_WARPS, num_stages=W4_DECODE_STAGES,
        )  # fmt: skip
        out.copy_(acc)
        return out
    bm, bn, warps, stages = 64, 128, 8, 3
    grid = (triton.cdiv(n, bn), triton.cdiv(m, bm))
    _w4_kernel[grid](
        x, packed, scale, zero, out, m, n, k,
        x.stride(0), packed.stride(0), scale.stride(0), zero.stride(0), out.stride(0),
        BM=bm, BN=bn, G=GROUP, num_warps=warps, num_stages=stages,
    )  # fmt: skip
    return out


class W4Linear:
    """`x @ weight.T` for a bf16 `weight [N, K]`, as int4 W4A16."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.size_n, self.size_k = weight.shape
        self.packed, self.scale, self.zero = quantize_int4(weight)

    def __call__(
        self, x: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        return w4_matmul(x.contiguous(), self.packed, self.scale, self.zero, out)
