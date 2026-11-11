# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A8 prefill through a dense INT8 GEMM (Qwen3.8-27B).

On the CMP 170HX the W4A8 Marlin kernel runs large prefill chunks at about
145 TOPS, while vLLM's CUTLASS INT8 scaled_mm runs the same shapes at about
230 TOPS. For a large-M call this module rebuilds the layer's weights as int8
with one scale per output channel, straight from Marlin's int8-activation
layout (no extra resident memory), and runs ``ops.cutlass_scaled_mm``.

Weights: Marlin int8-activation layout (gptq_marlin_repack is_a_8bit=True):
(K, N) codes cut into 32x16 tiles, tile rows of N*32 codes, each 1024-code
chunk permuted by ``get_weight_perm(4, True)``, 8 codes per int32 word.
Scales: int16 codes s (s / s.max() * 4096, marlin_act_int8_process_scales)
viewed as the model dtype, each row permuted in 32-column chunks
(``scale_perm_single``). Per channel n the int8 weight is
round(q * s[g, n] / T[n]) with T[n] = 8 * max_g s[g, n] / 127, so |w8| <= 127.
The extra rounding is at most T/2 per weight, against the int4 step s[g, n].
Output = (a8 @ w8.T) * a_scales[m] * T[n] (a_scales already carry Marlin's
global scale factor, exactly as the Marlin call uses them).

Switch: VLLM_CMP_W4A8_PREFILL_MIN_M=<rows> (0 / unset = off).

GEMM: VLLM_CMP_I8MM=1 runs the GEMM through cmp_i8mm (VLLM_CMP_I8MM_LIB, built
from vllm-cmp170hx csrc/exl3/lab_once/cmp_i8mm.cu): vLLM's scale epilogue,
bit-identical output, with a 128x256x64 / 3-stage / swizzle-8 mainloop that
spends fewer bytes per MAC and so runs at a higher clock under the power cap (+9-13% synthetic).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from vllm.triton_utils import tl, triton

MIN_M = int(os.environ.get("VLLM_CMP_W4A8_PREFILL_MIN_M", "0") or 0)
I8MM_LIB = os.environ.get("VLLM_CMP_I8MM_LIB", "/opt/cmp170hx/q27/cmp_i8mm.so")
_I8MM = None


def _i8mm():
    """torch.ops.cmp_i8mm.scaled_mm when VLLM_CMP_I8MM=1 (loaded once), else None."""
    global _I8MM
    if _I8MM is None:
        _I8MM = False
        if os.environ.get("VLLM_CMP_I8MM", "0") == "1":
            torch.ops.load_library(I8MM_LIB)
            _I8MM = torch.ops.cmp_i8mm.scaled_mm
    return _I8MM or None


_INV_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _inverse_perms(device):
    hit = _INV_CACHE.get(device)
    if hit is not None:
        return hit
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        get_scale_perms,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        get_weight_perm,
    )

    perm = get_weight_perm(4, True).numpy()
    inv_w = np.empty_like(perm)
    inv_w[perm] = np.arange(perm.size)
    ps = np.array(get_scale_perms()[1])
    inv_s = np.empty_like(ps)
    inv_s[ps] = np.arange(ps.size)
    out = (
        torch.tensor(inv_w, dtype=torch.int32, device=device),
        torch.tensor(inv_s, dtype=torch.int32, device=device),
    )
    _INV_CACHE[device] = out
    return out


@triton.jit
def _scale_max_kernel(s_ptr, inv_s_ptr, t_ptr, N, G, BN: tl.constexpr):
    pid = tl.program_id(0)
    n = pid * BN + tl.arange(0, BN)
    n_ok = n < N
    s_col = (n // 32) * 32 + tl.load(inv_s_ptr + (n % 32), mask=n_ok, other=0)
    smax = tl.zeros([BN], dtype=tl.float32)
    for g in range(0, G):
        s = tl.load(s_ptr + g * N + s_col, mask=n_ok, other=0).to(tl.float32)
        smax = tl.maximum(smax, tl.abs(s))
    tl.store(t_ptr + n, tl.maximum(smax * 8.0 / 127.0, 1e-30), mask=n_ok)


@triton.jit
def _marlin_a8_to_int8_kernel(
    q_ptr,
    s_ptr,
    inv_w_ptr,
    inv_s_ptr,
    w8_ptr,
    t_ptr,
    N,
    K,
    GROUP: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    # program (n block of BN channels, k block of BK rows inside one group)
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n = pid_n * BN + tl.arange(0, BN)
    n_ok = n < N
    g = (pid_k * BK) // GROUP
    s_col = (n // 32) * 32 + tl.load(inv_s_ptr + (n % 32), mask=n_ok, other=0)
    s = tl.load(s_ptr + g * N + s_col, mask=n_ok, other=0).to(tl.float32)
    t = tl.load(t_ptr + n, mask=n_ok, other=1.0)
    k = pid_k * BK + tl.arange(0, BK)
    r = k // 32
    kr = k % 32
    c = (n // 16)[None, :] * 512 + kr[:, None] * 16 + (n % 16)[None, :]  # [BK, BN]
    p = (c // 1024) * 1024 + tl.load(inv_w_ptr + (c % 1024))
    word = tl.load(
        q_ptr + r[:, None].to(tl.int64) * (N * 4) + p // 8, mask=n_ok[None, :], other=0
    )
    v = (word >> ((p % 8) * 4)) & 0xF
    w = (v.to(tl.float32) - 8.0) * (s / t)[None, :]
    w = tl.extra.cuda.libdevice.rint(w)
    w = tl.minimum(tl.maximum(w, -127.0), 127.0)
    tl.store(
        w8_ptr + n[None, :].to(tl.int64) * K + k[:, None],
        w.to(tl.int8),
        mask=n_ok[None, :],
    )


@triton.jit
def _marlin_a8_to_int8_chunk_kernel(
    q_ptr, s_ptr, inv_w_ptr, inv_s_ptr, w8_ptr, t_ptr, N, K, GROUP: tl.constexpr
):
    # program (32-row tile row r, 1024-code chunk ch = 32 output channels):
    # read the chunk's 128 words once (coalesced), unpermute in registers,
    # write 32 contiguous int8 per channel
    r = tl.program_id(0)
    ch = tl.program_id(1)
    words = tl.load(q_ptr + r.to(tl.int64) * (N * 4) + ch * 128 + tl.arange(0, 128))
    nl = tl.arange(0, 32)[:, None]
    kr = tl.arange(0, 32)[None, :]
    c = (nl // 16) * 512 + kr * 16 + (nl % 16)  # [32 n, 32 k]
    src = tl.reshape(tl.load(inv_w_ptr + c), [1024])
    wv = tl.gather(words, src // 8, 0)
    v = tl.reshape((wv >> ((src % 8) * 4)) & 0xF, [32, 32])
    n1 = ch * 32 + tl.arange(0, 32)
    g = (r * 32) // GROUP
    s_col = ch * 32 + tl.load(inv_s_ptr + tl.arange(0, 32))
    s = tl.load(s_ptr + g * N + s_col).to(tl.float32)
    t = tl.load(t_ptr + n1)
    w = (v.to(tl.float32) - 8.0) * (s / t)[:, None]
    w = tl.extra.cuda.libdevice.rint(w)
    w = tl.minimum(tl.maximum(w, -127.0), 127.0)
    tl.store(w8_ptr + n1[:, None].to(tl.int64) * K + r * 32 + kr, w.to(tl.int8))


# Layout fact used by the v3 kernels (from get_weight_perm(4, True) and the
# a8 interleave [0, 4, 1, 5, 2, 6, 3, 7]): inside a 1024-code chunk (32
# channels x 32 k of one tile row) channel nl owns exactly four words,
# word(nl, a) = 16 * (nl % 8) + 4 * a + nl // 8 (a = 0..3), and nibble i
# = 2j + h of word a holds k = 16h + 4a + j. So one channel's 32 int8 for a
# tile row come from 4 words of the same 512-byte chunk, no table needed.
@triton.jit
def _marlin_a8_to_int8_v3_kernel(
    q_ptr,
    s_ptr,
    inv_s_ptr,
    w8_ptr,
    t_ptr,
    N,
    K,
    GROUP: tl.constexpr,
    CH: tl.constexpr,
    R: tl.constexpr,
    UNPACK: tl.constexpr,
    STORE32: tl.constexpr = False,
    MATH: tl.constexpr = True,
):
    # program: CH chunks (CH*32 channels) x R tile rows (32 k each)
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)
    C: tl.constexpr = CH * 32
    nl = tl.arange(0, C)
    nn = nl % 32
    n = pid_c * C + nl
    wbase = (n // 32) * 128 + 16 * (nn % 8) + nn // 8  # word of a = 0
    s_col = (n // 32) * 32 + tl.load(inv_s_ptr + nn)
    t = tl.load(t_ptr + n)
    kk = tl.arange(0, 32)
    for rr in range(R):
        r = pid_r * R + rr
        g = (r * 32) // GROUP
        mult = tl.load(s_ptr + g * N + s_col).to(tl.float32) / t
        row = q_ptr + r.to(tl.int64) * (N * 4)
        if UNPACK:
            a = tl.arange(0, 4)
            words = tl.load(row + wbase[:, None] + 4 * a[None, :])  # [C, 4]
            sh = tl.arange(0, 8) * 4
            v = (words[:, :, None] >> sh[None, None, :]) & 0xF  # [C, a, i=2j+h]
            v = tl.reshape(v, [C, 4, 4, 2])  # [C, a, j, h]
            v = tl.permute(v, (0, 3, 1, 2))  # [C, h, a, j]
            v = tl.reshape(v, [C, 32])  # k = 16h + 4a + j
        else:
            ka = (kk % 16) // 4
            shift = 4 * (2 * (kk % 4) + kk // 16)
            word = tl.load(row + wbase[:, None] + 4 * ka[None, :])
            v = (word >> shift[None, :]) & 0xF
        if MATH:
            w = (v.to(tl.float32) - 8.0) * mult[:, None]
            w = tl.extra.cuda.libdevice.rint(w)
            w = tl.minimum(tl.maximum(w, -127.0), 127.0).to(tl.int32)
        else:  # timing diagnostic only (not the right values)
            w = v - 8
        if STORE32:
            # four int8 per int32 (little endian: byte b of word kw is k = 4 kw + b)
            b = tl.reshape(w & 0xFF, [C, 8, 4]) << (8 * tl.arange(0, 4))[None, None, :]
            word32 = tl.sum(b, axis=2)
            kw = tl.arange(0, 8)
            tl.store(
                w8_ptr.to(tl.pointer_type(tl.int32))
                + n[:, None].to(tl.int64) * (K // 4)
                + r * 8
                + kw[None, :],
                word32,
            )
        else:
            tl.store(
                w8_ptr + n[:, None].to(tl.int64) * K + r * 32 + kk[None, :],
                w.to(tl.int8),
            )


@triton.jit
def _marlin_a8_to_int8_v4_kernel(
    q_ptr,
    s_ptr,
    inv_s_ptr,
    w8_ptr,
    t_ptr,
    N,
    K,
    GROUP: tl.constexpr,
    C: tl.constexpr,
    KB: tl.constexpr,
    CLAMP: tl.constexpr,
):
    # program: C channels (multiple of 32) x KB k (KB divides GROUP: one scale
    # group); same layout fact as v3, all loads of the tile issued at once.
    # No clamp: |v - 8| <= 8 and s / t <= 127 / 8, so |w| <= 127 already
    # (the clamp cost ~20 ms per 8192-token chunk). Bit-exact with CONV=1.
    pid_c = tl.program_id(0)
    pid_k = tl.program_id(1)
    nl = tl.arange(0, C)
    nn = nl % 32
    n = pid_c * C + nl
    wbase = (n // 32) * 128 + 16 * (nn % 8) + nn // 8
    s_col = (n // 32) * 32 + tl.load(inv_s_ptr + nn)
    g = (pid_k * KB) // GROUP
    mult = tl.load(s_ptr + g * N + s_col).to(tl.float32) / tl.load(t_ptr + n)
    kk = tl.arange(0, KB)
    k = pid_k * KB + kk
    r = k // 32
    kl = kk % 32
    woff = (
        r.to(tl.int64)[None, :] * (N * 4)
        + wbase[:, None]
        + (4 * ((kl % 16) // 4))[None, :]
    )
    word = tl.load(q_ptr + woff)
    v = (word >> (4 * (2 * (kl % 4) + kl // 16))[None, :]) & 0xF
    w = tl.extra.cuda.libdevice.rint((v.to(tl.float32) - 8.0) * mult[:, None])
    if CLAMP:
        w = tl.minimum(tl.maximum(w, -127.0), 127.0)
    tl.store(w8_ptr + n[:, None].to(tl.int64) * K + k[None, :], w.to(tl.int8))


CONV = int(os.environ.get("VLLM_CMP_W4A8_PREFILL_CONV", "5"))
V3_CH = int(os.environ.get("VLLM_CMP_W4A8_PREFILL_V3_CH", "2"))
V3_R = int(os.environ.get("VLLM_CMP_W4A8_PREFILL_V3_R", "2"))
V3_WARPS = int(os.environ.get("VLLM_CMP_W4A8_PREFILL_V3_WARPS", "4"))
V3_STORE32 = int(os.environ.get("VLLM_CMP_W4A8_PREFILL_V3_STORE32", "0"))
_V3_MATH = 1  # tests only
V4_C = int(os.environ.get("VLLM_CMP_W4A8_PREFILL_V4_C", "32"))
V4_KB = int(os.environ.get("VLLM_CMP_W4A8_PREFILL_V4_KB", "128"))
V4_WARPS = int(os.environ.get("VLLM_CMP_W4A8_PREFILL_V4_WARPS", "4"))
V4_CLAMP = int(os.environ.get("VLLM_CMP_W4A8_PREFILL_V4_CLAMP", "0"))
BK = int(os.environ.get("VLLM_CMP_W4A8_PREFILL_BK", "64"))
BN = int(os.environ.get("VLLM_CMP_W4A8_PREFILL_BN", "128"))
WARPS = int(os.environ.get("VLLM_CMP_W4A8_PREFILL_WARPS", "8"))
WARPS2 = int(os.environ.get("VLLM_CMP_W4A8_PREFILL_WARPS2", "4"))


def marlin_a8_weights_to_int8(
    b_q_weight: torch.Tensor, b_scales: torch.Tensor, size_k: int, size_n: int
):
    """Marlin int8-activation W4 weights -> (w8 [N, K] int8, t [N] float32),
    t in the units of Marlin's int16 scale codes."""
    G = b_scales.shape[0]
    group = size_k // G
    assert group * G == size_k and size_n % 32 == 0 and size_k % 32 == 0
    assert CONV in (2, 3, 4, 5) or (group % BK == 0 and size_n % BN == 0)
    assert group % 32 == 0
    inv_w, inv_s = _inverse_perms(b_q_weight.device)
    s16 = b_scales.view(torch.int16) if b_scales.dtype != torch.int16 else b_scales
    w8 = torch.empty((size_n, size_k), dtype=torch.int8, device=b_q_weight.device)
    t = torch.empty((size_n,), dtype=torch.float32, device=b_q_weight.device)
    _scale_max_kernel[(triton.cdiv(size_n, 128),)](s16, inv_s, t, size_n, G, BN=128)
    if CONV == 5:
        c = V4_C if size_n % V4_C == 0 else 32
        kb = min(V4_KB, group)
        assert c % 32 == 0 and group % kb == 0
        _marlin_a8_to_int8_v4_kernel[(size_n // c, size_k // kb)](
            b_q_weight,
            s16,
            inv_s,
            w8,
            t,
            size_n,
            size_k,
            GROUP=group,
            C=c,
            KB=kb,
            CLAMP=bool(V4_CLAMP),
            num_warps=V4_WARPS,
        )
    elif CONV in (3, 4):
        ch = V3_CH if size_n % (32 * V3_CH) == 0 else 1
        rr = V3_R if (size_k // 32) % V3_R == 0 else 1
        _marlin_a8_to_int8_v3_kernel[(size_n // (32 * ch), size_k // (32 * rr))](
            b_q_weight,
            s16,
            inv_s,
            w8,
            t,
            size_n,
            size_k,
            GROUP=group,
            CH=ch,
            R=rr,
            UNPACK=(CONV == 4),
            STORE32=bool(V3_STORE32),
            MATH=bool(_V3_MATH),
            num_warps=V3_WARPS,
        )
    elif CONV == 2:
        _marlin_a8_to_int8_chunk_kernel[(size_k // 32, size_n // 32)](
            b_q_weight,
            s16,
            inv_w,
            inv_s,
            w8,
            t,
            size_n,
            size_k,
            GROUP=group,
            num_warps=WARPS2,
        )
    else:
        _marlin_a8_to_int8_kernel[(size_n // BN, size_k // BK)](
            b_q_weight,
            s16,
            inv_w,
            inv_s,
            w8,
            t,
            size_n,
            size_k,
            GROUP=group,
            BK=BK,
            BN=BN,
            num_warps=WARPS,
        )
    return w8, t


def prefill_gemm(
    a8: torch.Tensor,
    a_scales: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    w8, t = marlin_a8_weights_to_int8(b_q_weight, b_scales, size_k, size_n)
    sa = a_scales.reshape(-1, 1).to(torch.float32)
    mm = _i8mm()
    if mm is not None and bias is None and out_dtype == torch.bfloat16:
        a = a8[:size_m]
        return mm(
            a if a.is_contiguous() else a.contiguous(),
            w8,
            sa[:size_m].contiguous(),
            t.reshape(-1).contiguous(),
        )
    return ops.cutlass_scaled_mm(
        a8[:size_m], w8.t(), sa[:size_m], t.reshape(1, -1), out_dtype, bias
    )


def use_for(a: torch.Tensor, size_m: int, b_zeros, b_q_type_id: int) -> bool:
    if MIN_M <= 0 or a.dtype != torch.int8 or size_m < MIN_M:
        return False
    if b_zeros is not None and b_zeros.numel() > 0:
        return False
    from vllm.scalar_type import scalar_types

    return b_q_type_id == scalar_types.uint4b8.id
