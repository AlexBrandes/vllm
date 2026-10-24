# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused FP64 rank sums, with an in-payload unsupported-input marker.

NaN cannot be a valid result of the supported finite exact sum (overflow to
infinity is allowed). A NaN marker therefore carries a span/nonfinite violation
through the existing result all-gather, without another collective. The scan
latches a device error word; it never traps or synchronizes with the host.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sum_round(Parts, Out, N: tl.constexpr, WORLD: tl.constexpr,
               SPAN: tl.constexpr, CHECK: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < N
    acc = tl.load(Parts + offsets, active, other=0).to(tl.float64)
    if CHECK:
        largest = tl.abs(acc)
        smallest = tl.where(largest == 0, float("inf"), largest)
        invalid = (largest != largest) | (largest == float("inf"))
    for rank in tl.static_range(1, WORLD):
        part = tl.load(Parts + rank * N + offsets, active, other=0).to(tl.float64)
        acc = acc + part
        if CHECK:
            magnitude = tl.abs(part)
            invalid = invalid | (magnitude != magnitude) | (magnitude == float("inf"))
            largest = tl.maximum(largest, magnitude)
            smallest = tl.minimum(smallest, tl.where(magnitude == 0, float("inf"), magnitude))
    if CHECK:
        invalid = invalid | (largest > smallest * SPAN)
    # Round-to-odd FP32 followed by round-to-nearest-even in the output dtype.
    fp32 = acc.to(tl.float32)
    back = fp32.to(tl.float64)
    inexact = back != acc
    away = inexact & (tl.abs(back) > tl.abs(acc))
    bits = fp32.to(tl.int32, bitcast=True)
    bits = tl.where(away, bits - 1, bits)
    bits = tl.where(inexact, bits | 1, bits)
    rounded = bits.to(tl.float32, bitcast=True)
    if CHECK:
        rounded = tl.where(invalid, float("nan"), rounded)
    tl.store(Out + offsets, rounded, active)


@triton.jit
def _record_error(Values, Error, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < N
    values = tl.load(Values + offsets, active, other=0).to(tl.float32)
    invalid = tl.sum((active & (values != values)).to(tl.int32), 0) != 0
    if invalid:
        # Sticky across calls/streams/steps. Never clear a failure while an
        # older output copy is pending, and never trigger a device assertion.
        tl.atomic_or(Error, 1, sem="relaxed")


def reduce_scattered(parts: torch.Tensor) -> torch.Tensor:
    """The serving path always checks; CHECK=False is only a timing control."""
    world, n = parts.shape
    out = torch.empty(n, dtype=parts.dtype, device=parts.device)
    span = (45 if parts.dtype == torch.bfloat16 else 42) - (world - 1).bit_length()
    _sum_round[(triton.cdiv(n, 256),)](parts, out, n, world, 2.0**span, True, 256)
    return out


def record_error(values: torch.Tensor, error: torch.Tensor) -> None:
    _record_error[(triton.cdiv(values.numel(), 1024),)](values, error, values.numel(), 1024)
