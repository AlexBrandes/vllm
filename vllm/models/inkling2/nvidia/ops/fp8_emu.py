# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Software E4M3 (FP8, OCP `float8_e4m3fn`) for SM8x, which has neither the
`cvt.*.e4m3` instructions nor Triton's fp8e4nv type.

Inkling2's FP8 KV cache is unit-scale E4M3 (K is L2-clipped and V RMS-
normalized before the store, so no scales are kept). On SM8x the cache is a
uint8 view of the same bytes; these helpers encode on the store and decode on
the load, bit-exact with the hardware conversions (round to nearest even,
saturating at +-448, as `cvt.rn.satfinite`).
"""

from vllm.triton_utils import tl, triton

_E4M3_MAX = tl.constexpr(448.0)
_MIN_NORMAL = tl.constexpr(0.015625)  # 2^-6


@triton.jit
def e4m3_decode(b):
    """uint8 E4M3 codes -> fp32 values."""
    x = b.to(tl.int32)
    sign = (x & 0x80) << 8
    # A normal E4M3 code shifted into FP16 position: exponent bias 7 -> 15.
    half_bits = (((x & 0x7F) << 7) + 0x2000) | sign
    normal = half_bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    sub = (x & 7).to(tl.float32) * 0.001953125  # m * 2^-9
    sub = tl.where(sign != 0, -sub, sub)
    return tl.where((x & 0x78) == 0, sub, normal)


@triton.jit
def e4m3_decode_fp16_scaled(b):
    """uint8 E4M3 codes -> fp16 holding value * 2^-8, exactly, subnormals
    included: the code's (exponent, mantissa) bits moved into the FP16
    fields, as Marlin's FP8 dequant does. Callers fold the 2^8 back in."""
    x = b.to(tl.int32)
    bits = ((x & 0x7F) << 7) | ((x & 0x80) << 8)
    return bits.to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def e4m3_decode_bf16(b):
    """uint8 E4M3 codes -> bf16, integer ops only (every E4M3 value is exact
    in bf16): the attention's inner-loop decode."""
    x = b.to(tl.int32)
    mag = x & 0x7F
    # Normal: exponent bias 7 -> 127 (+120 << 7), mantissa 3 -> 7 bits (<< 4).
    normal = (mag << 4) + 0x3C00
    # Subnormal m * 2^-9 (m = 1..7): normalize on the leading bit p.
    p = (mag >= 2).to(tl.int32) + (mag >= 4).to(tl.int32)
    sub = ((118 + p) << 7) | ((mag - (1 << p)) << (7 - p))
    sub = tl.where(mag == 0, 0, sub)
    bits = tl.where(mag < 8, sub, normal) | ((x & 0x80) << 8)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def e4m3_encode(y):
    """fp32 values -> uint8 E4M3 codes (RNE, saturating)."""
    sign = (y.to(tl.uint32, bitcast=True) >> 24) & 0x80
    a = tl.minimum(tl.abs(y), _E4M3_MAX)
    ab = a.to(tl.uint32, bitcast=True)
    # Normal range: round the (exponent | mantissa) field to 3 mantissa bits
    # (round half to even), then rebias the exponent 127 -> 7.
    r = (ab + 0x7FFFF + ((ab >> 20) & 1)) >> 20
    code_n = r.to(tl.int32) - (120 << 3)
    # Subnormal range: a multiple of 2^-9, rounded half to even.
    t = a * 512.0
    fl = tl.floor(t)
    frac = t - fl
    odd = (fl.to(tl.int32) & 1) == 1
    code_s = fl.to(tl.int32) + ((frac > 0.5) | ((frac == 0.5) & odd)).to(tl.int32)
    code = tl.where(a < _MIN_NORMAL, code_s, code_n)
    code = tl.minimum(code, 0x7E)
    return (code | sign.to(tl.int32)).to(tl.uint8)
