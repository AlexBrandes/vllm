# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-M BF16 tensor-core projections and fused sm_80 controller/vocab kernels."""

import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton

try:
    from vllm.models.inkling2.nvidia.ops.sm80.inkling_sm80_cuda import (
        HAS_CUDA_EXT as _SM80_HAS_CUDA,
    )
    from vllm.models.inkling2.nvidia.ops.sm80.inkling_sm80_cuda import (
        fused_pair_norm_ctrl_down_cuda as _sm80_fused_pair_norm_ctrl_down,
    )
    from vllm.models.inkling2.nvidia.ops.sm80.inkling_sm80_cuda import (
        pair_rms_norm_rank2_cuda as _sm80_pair_rms_norm_rank2,
    )
    from vllm.models.inkling2.nvidia.ops.sm80.inkling_sm80_cuda import (
        rms_norm_f32_to_bf16_cuda as _sm80_rms_norm_f32_to_bf16,
    )
except Exception:
    _SM80_HAS_CUDA = False
    _sm80_fused_pair_norm_ctrl_down = None
    _sm80_pair_rms_norm_rank2 = None
    _sm80_rms_norm_f32_to_bf16 = None


MAX_M = 16
SM80_MAX_M = 64
_BN, _BK = 32, 256


@triton.jit
def _gemv_kernel(
    X,
    W,
    B,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    SY,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(B + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def _sm80_tiles(n: int, k: int, m: int = 16) -> tuple[int, int, int, int]:
    if n >= 65536:
        if m > 16:
            return (128, 128, 4, 3) if m <= 32 else (128, 128, 4, 4)
        return (64, 64, 2, 3) if k <= 512 else (64, 256, 4, 4)
    if n <= 1024:
        return 16, 256, 2, 3
    if n >= 16384:
        if m > 16:
            return (128, 128, 4, 3) if m <= 32 else (128, 64, 4, 4)
        return 64, 128, 8, 2
    if n >= 8192:
        return (128, 128, 4, 4) if m > 16 else (128, 128, 4, 3)
    if k >= 8192 and m > 16:
        return (64, 256, 4, 4) if m <= 32 else (32, 256, 4, 4)
    if k >= 3072:
        return 32, 256, 4, 4
    if n == 2560 and m <= 32:
        return 64, 256, 4, 4
    return 32, 128, 4, 4


@triton.jit
def _seg3_gemv_kernel(
    X0,
    X1,
    X2,
    W,
    Y,
    M,
    K,
    SX0,
    SX1,
    SX2,
    SW,
    SY,
    N0,
    N1,
    N2,
    NB0,
    NB1,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < NB0:
        X = X0
        SX = SX0
        lo = 0
        hi = N0
        local = pid
    elif pid < NB0 + NB1:
        X = X1
        SX = SX1
        lo = N0
        hi = N0 + N1
        local = pid - NB0
    else:
        X = X2
        SX = SX2
        lo = N0 + N1
        hi = N0 + N1 + N2
        local = pid - NB0 - NB1
    cols = lo + local * BN + tl.arange(0, BN)
    cmask = cols < hi
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            cmask[None, :] & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    tl.store(
        Y + rows[:, None] * SY + cols[None, :],
        acc.to(Y.dtype.element_ty),
        (rows[:, None] < M) & cmask[None, :],
    )


def _sm80_seg3_tiles(m: int) -> tuple[int, int, int, int]:
    # M <= 16: BN=64 (29.4us); 16 < M <= 64 (8u M=40): BN=128 (34.6us vs 39.3us).
    if m > 16:
        return 128, 128, 4, 3
    return 64, 128, 4, 3


def sm80_seg3_gemv(
    xs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
    widths: tuple[int, int, int],
) -> torch.Tensor | None:
    if (
        not current_platform.is_device_capability_family(80)
        or weight.dtype != torch.bfloat16
    ):
        return None
    m, k = xs[0].shape[0], weight.shape[1]
    for x in xs:
        if (
            x.ndim != 2
            or x.dtype != torch.bfloat16
            or x.shape != (m, k)
            or x.stride(-1) != 1
            or x.device != weight.device
        ):
            return None
    if (
        not 0 < m <= SM80_MAX_M
        or weight.stride(-1) != 1
        or weight.shape[0] < sum(widths)
    ):
        return None
    n = sum(widths)
    out = torch.empty((m, n), device=weight.device, dtype=torch.bfloat16)
    bn, bk, warps, stages = _sm80_seg3_tiles(m)
    mp = 16 if m <= 16 else (32 if m <= 32 else 64)
    nb = [triton.cdiv(w, bn) for w in widths]
    _seg3_gemv_kernel[(sum(nb),)](
        xs[0],
        xs[1],
        xs[2],
        weight,
        out,
        m,
        k,
        xs[0].stride(0),
        xs[1].stride(0),
        xs[2].stride(0),
        weight.stride(0),
        out.stride(0),
        *widths,
        nb[0],
        nb[1],
        BN=bn,
        BK=bk,
        MP=mp,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _seg3_dyn_gemv_kernel(
    X0,
    X1,
    X2,
    W,
    WK_dyn,
    WV_dyn,
    Y,
    YK_dyn,
    YV_dyn,
    M,
    K,
    SX0,
    SX1,
    SX2,
    SW,
    SWK,
    SWV,
    SY,
    N0,
    N1,
    N2,
    N_DYN,
    NB0,
    NB1,
    NB2,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < NB0:
        X = X0
        SX = SX0
        WP = W
        SW_U = SW
        YP = Y
        SY_U = SY
        lo = 0
        hi = N0
        local = pid
    elif pid < NB0 + NB1:
        X = X1
        SX = SX1
        WP = W
        SW_U = SW
        YP = Y
        SY_U = SY
        lo = N0
        hi = N0 + N1
        local = pid - NB0
    elif pid < NB0 + NB1 + NB2:
        X = X2
        SX = SX2
        WP = W
        SW_U = SW
        YP = Y
        SY_U = SY
        lo = N0 + N1
        hi = N0 + N1 + N2
        local = pid - NB0 - NB1
    elif pid == NB0 + NB1 + NB2:
        X = X1
        SX = SX1
        WP = WK_dyn
        SW_U = SWK
        YP = YK_dyn
        SY_U = N_DYN
        lo = 0
        hi = N_DYN
        local = 0
    else:
        X = X2
        SX = SX2
        WP = WV_dyn
        SW_U = SWV
        YP = YV_dyn
        SY_U = N_DYN
        lo = 0
        hi = N_DYN
        local = 0
    cols = lo + local * BN + tl.arange(0, BN)
    cmask = cols < hi
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            WP + cols[None, :].to(tl.int64) * SW_U + ks[:, None],
            cmask[None, :] & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    tl.store(
        YP + rows[:, None] * SY_U + cols[None, :],
        acc.to(YP.dtype.element_ty),
        (rows[:, None] < M) & cmask[None, :],
    )


def sm80_seg3_dyn_gemv(
    xs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    weight: torch.Tensor,
    widths: tuple[int, int, int],
    wk_dyn: torch.Tensor,
    wv_dyn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        not current_platform.is_device_capability_family(80)
        or weight.dtype != torch.bfloat16
        or wk_dyn.dtype != torch.bfloat16
        or wv_dyn.dtype != torch.bfloat16
    ):
        return None
    m, k = xs[0].shape[0], weight.shape[1]
    n_dyn = wk_dyn.shape[0]
    bn, bk, warps, stages = _sm80_seg3_tiles(m)
    if (
        not 0 < m <= SM80_MAX_M
        or weight.stride(-1) != 1
        or wk_dyn.stride(-1) != 1
        or wv_dyn.stride(-1) != 1
        or weight.shape[0] < sum(widths)
        or wk_dyn.shape != (n_dyn, k)
        or wv_dyn.shape != (n_dyn, k)
        or n_dyn > bn
    ):
        return None
    for x in xs:
        if (
            x.ndim != 2
            or x.dtype != torch.bfloat16
            or x.shape != (m, k)
            or x.stride(-1) != 1
            or x.device != weight.device
        ):
            return None
    n = sum(widths)
    out = torch.empty((m, n), device=weight.device, dtype=torch.bfloat16)
    yk = torch.empty((m, n_dyn), device=weight.device, dtype=torch.bfloat16)
    yv = torch.empty((m, n_dyn), device=weight.device, dtype=torch.bfloat16)
    mp = 16 if m <= 16 else (32 if m <= 32 else 64)
    nb0, nb1, nb2 = (triton.cdiv(w, bn) for w in widths)
    _seg3_dyn_gemv_kernel[(nb0 + nb1 + nb2 + 2,)](
        xs[0],
        xs[1],
        xs[2],
        weight,
        wk_dyn,
        wv_dyn,
        out,
        yk,
        yv,
        m,
        k,
        xs[0].stride(0),
        xs[1].stride(0),
        xs[2].stride(0),
        weight.stride(0),
        wk_dyn.stride(0),
        wv_dyn.stride(0),
        out.stride(0),
        widths[0],
        widths[1],
        widths[2],
        n_dyn,
        nb0,
        nb1,
        nb2,
        BN=bn,
        BK=bk,
        MP=mp,
        num_warps=warps,
        num_stages=stages,
    )
    return out, yk, yv


@triton.jit
def _ctrl_gf_blend_kernel(
    P,
    W,
    Dest,
    Msg,
    OutX,
    M,
    D: tl.constexpr,
    SWS,
    SWN,
    stride_dest_s,
    SPLITS: tl.constexpr,
    R: tl.constexpr,
    BN: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    s = tl.program_id(1)
    S = tl.num_programs(1)
    rows = tl.arange(0, MP)
    rk = tl.arange(0, R)
    h = tl.zeros((MP, R), tl.float32)
    for sp in tl.static_range(SPLITS):
        h += tl.load(P + ((sp * S + s) * MP + rows[:, None]) * R + rk[None, :])
    h = h.to(tl.bfloat16)

    cols = pid * BN + tl.arange(0, BN)
    cmask = cols < D
    rmask = rows < M
    mask = rmask[:, None] & cmask[None, :]

    w_base = W + s.to(tl.int64) * SWS
    wg = tl.load(
        w_base + cols[None, :].to(tl.int64) * SWN + rk[:, None], cmask[None, :], other=0
    )
    wf = tl.load(
        w_base + (D + cols[None, :]).to(tl.int64) * SWN + rk[:, None],
        cmask[None, :],
        other=0,
    )
    gain = tldevice.tanh(tl.dot(h, wg))
    forget = tldevice.tanh(tl.dot(h, wf))

    dest = tl.load(
        Dest + s.to(tl.int64) * stride_dest_s + rows[:, None] * D + cols[None, :],
        mask,
        other=0.0,
    )
    msg = tl.load(
        Msg + (s.to(tl.int64) * M + rows[:, None]) * D + cols[None, :], mask, other=0.0
    )
    x = (1.0 - forget) * dest + gain * msg
    tl.store(OutX + (s.to(tl.int64) * M + rows[:, None]) * D + cols[None, :], x, mask)


@triton.jit
def _ctrl_gf_blend_prefill_kernel(
    H,
    W,
    Dest,
    Msg,
    OutX,
    M,
    D: tl.constexpr,
    R: tl.constexpr,
    SHS,
    SHM,
    SWS,
    SWN,
    stride_dest_s,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """Fused gain/forget projection + tanh + blend for large M (prefill & 8u S=1).
    Eliminates materializing the fp32 (S, M, 2*D) gain_forget tensor (~453 MB at 8k)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    s = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, R)
    rmask = rows < M
    cmask = cols < D
    mask = rmask[:, None] & cmask[None, :]

    h = tl.load(
        H + s.to(tl.int64) * SHS + rows[:, None].to(tl.int64) * SHM + rk[None, :],
        rmask[:, None],
        other=0,
    )
    w_base = W + s.to(tl.int64) * SWS
    wg = tl.load(
        w_base + cols[None, :].to(tl.int64) * SWN + rk[:, None], cmask[None, :], other=0
    )
    wf = tl.load(
        w_base + (D + cols[None, :]).to(tl.int64) * SWN + rk[:, None],
        cmask[None, :],
        other=0,
    )
    gain = tldevice.tanh(tl.dot(h, wg))
    forget = tldevice.tanh(tl.dot(h, wf))

    dest = tl.load(
        Dest
        + s.to(tl.int64) * stride_dest_s
        + rows[:, None].to(tl.int64) * D
        + cols[None, :],
        mask,
        other=0.0,
    )
    msg = tl.load(
        Msg + (s.to(tl.int64) * M + rows[:, None]).to(tl.int64) * D + cols[None, :],
        mask,
        other=0.0,
    )
    x = (1.0 - forget) * dest + gain * msg
    tl.store(
        OutX + (s.to(tl.int64) * M + rows[:, None]).to(tl.int64) * D + cols[None, :],
        x,
        mask,
    )


@triton.jit
def _ctrl_gf_blend_direct_h_kernel(
    H,
    W,
    Dest,
    Msg,
    OutX,
    M,
    D: tl.constexpr,
    SWS,
    SWN,
    stride_dest_s,
    R: tl.constexpr,
    BN: tl.constexpr,
    MP: tl.constexpr,
):
    """One-token decode: direct-H gain/forget+blend (reads bf16 H directly, no SPLITS reduction)."""
    pid = tl.program_id(0)
    s = tl.program_id(1)
    rows = tl.arange(0, MP)
    rk = tl.arange(0, R)
    rmask = rows < M
    h = tl.load(
        H + (s.to(tl.int64) * M + rows[:, None]) * R + rk[None, :],
        rmask[:, None],
        other=0,
    )
    cols = pid * BN + tl.arange(0, BN)
    cmask = cols < D
    mask = rmask[:, None] & cmask[None, :]

    w_base = W + s.to(tl.int64) * SWS
    wg = tl.load(
        w_base + cols[None, :].to(tl.int64) * SWN + rk[:, None], cmask[None, :], other=0
    )
    wf = tl.load(
        w_base + (D + cols[None, :]).to(tl.int64) * SWN + rk[:, None],
        cmask[None, :],
        other=0,
    )
    gain = tldevice.tanh(tl.dot(h, wg))
    forget = tldevice.tanh(tl.dot(h, wf))

    dest = tl.load(
        Dest + s.to(tl.int64) * stride_dest_s + rows[:, None] * D + cols[None, :],
        mask,
        other=0.0,
    )
    msg = tl.load(
        Msg + (s.to(tl.int64) * M + rows[:, None]) * D + cols[None, :], mask, other=0.0
    )
    x = (1.0 - forget) * dest + gain * msg
    tl.store(OutX + (s.to(tl.int64) * M + rows[:, None]) * D + cols[None, :], x, mask)


@triton.jit
def _ctrl_gf_blend_prefill_rank2_kernel(
    H,
    W,
    Dest,
    Bank,
    CRaw,
    OutX,
    M,
    D: tl.constexpr,
    R: tl.constexpr,
    SHS,
    SHM,
    SWS,
    SWN,
    stride_dest_s,
    stride_bank_0,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """Rank-2 prefill gain/forget+blend: reconstructs msg = c0*bank[0] + c1*bank[1] on the fly."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    s = tl.program_id(2)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, R)
    rmask = rows < M
    cmask = cols < D
    mask = rmask[:, None] & cmask[None, :]

    h = tl.load(
        H + s.to(tl.int64) * SHS + rows[:, None].to(tl.int64) * SHM + rk[None, :],
        rmask[:, None],
        other=0,
    )
    w_base = W + s.to(tl.int64) * SWS
    wg = tl.load(
        w_base + cols[None, :].to(tl.int64) * SWN + rk[:, None], cmask[None, :], other=0
    )
    wf = tl.load(
        w_base + (D + cols[None, :]).to(tl.int64) * SWN + rk[:, None],
        cmask[None, :],
        other=0,
    )
    gain = tldevice.tanh(tl.dot(h, wg))
    forget = tldevice.tanh(tl.dot(h, wf))

    dest = tl.load(
        Dest
        + s.to(tl.int64) * stride_dest_s
        + rows[:, None].to(tl.int64) * D
        + cols[None, :],
        mask,
        other=0.0,
    )
    c_off = (s.to(tl.int64) * M + rows).to(tl.int64) * 2
    c0 = tl.load(CRaw + c_off, rmask, other=0.0)
    c1 = tl.load(CRaw + c_off + 1, rmask, other=0.0)
    b_off = rows[:, None].to(tl.int64) * D + cols[None, :]
    b0 = tl.load(Bank + b_off, mask, other=0).to(tl.float32)
    b1 = tl.load(Bank + stride_bank_0 + b_off, mask, other=0).to(tl.float32)
    msg = c0[:, None] * b0 + c1[:, None] * b1
    x = (1.0 - forget) * dest + gain * msg
    tl.store(
        OutX + (s.to(tl.int64) * M + rows[:, None]).to(tl.int64) * D + cols[None, :],
        x,
        mask,
    )


def sm80_blend_rms_norm_rank2(
    dest: torch.Tensor,
    bank: torch.Tensor,
    c_raw: torch.Tensor,
    c_norm: torch.Tensor,
    down: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor | None:
    """sm_80: rank-2 prefill controller blend + RMSNorm (no materialized [S, T, D] fp32 messages)."""
    if (
        not _SM80_HAS_CUDA
        or not current_platform.is_device_capability_family(80)
        or os.environ.get("INKLING_SM80_NO_HYBRID_BLEND") == "1"
        or dest.dtype != torch.float32
        or bank.dtype != torch.bfloat16
        or c_raw.dtype != torch.float32
        or c_norm.dtype != torch.float32
        or down.dtype != torch.bfloat16
        or gain_forget.dtype != torch.bfloat16
        or norm_weight.dtype != torch.bfloat16
        or dest.ndim != 3
        or bank.ndim != 3
        or bank.shape[0] != 2
    ):
        return None
    s, m, d = dest.shape
    r = down.shape[1]
    if (
        d != 2304
        or m <= 0
        or bank.shape[1] != m
        or bank.shape[2] != d
        or c_raw.shape != (s, m, 2)
        or c_norm.shape != (s, m, 2)
        or not c_raw.is_contiguous()
        or not c_norm.is_contiguous()
        or r not in (16, 32, 64, 128)
        or down.shape != (s, r, 2 * d)
        or gain_forget.shape != (s, 2 * d, r)
        or norm_weight.shape != (s, d)
        or dest.stride(1) != d
        or dest.stride(2) != 1
        or bank.stride(1) != d
        or bank.stride(2) != 1
        or down.stride(-1) != 1
        or gain_forget.stride(-1) != 1
        or norm_weight.stride(1) != 1
    ):
        return None
    pair = _sm80_pair_rms_norm_rank2(dest, bank, c_norm, 1e-6)
    hidden = torch.bmm(pair, down.transpose(1, 2))
    del pair
    x_tmp = torch.empty((s, m, d), device=dest.device, dtype=torch.float32)
    if m >= 128:
        bm, bn, nw, ns = (64, 64, 4, 3) if r >= 128 else (32, 64, 4, 4)
    else:
        bm, bn, nw, ns = 16, 64, 4, 3
    _ctrl_gf_blend_prefill_rank2_kernel[(triton.cdiv(m, bm), triton.cdiv(d, bn), s)](
        hidden,
        gain_forget,
        dest,
        bank,
        c_raw,
        x_tmp,
        m,
        d,
        r,
        hidden.stride(0),
        hidden.stride(1),
        gain_forget.stride(0),
        gain_forget.stride(1),
        dest.stride(0),
        bank.stride(0),
        BM=bm,
        BN=bn,
        num_warps=nw,
        num_stages=ns,
        enable_fp_fusion=False,
    )
    return _sm80_rms_norm_f32_to_bf16(x_tmp, norm_weight, norm_eps)


def sm80_blend_rms_norm(
    dest: torch.Tensor,
    message: torch.Tensor,
    down: torch.Tensor,
    gain_forget: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor | None:
    """sm_80 hybrid fused loop message controller blend + RMSNorm for ALL M (decode & prefill)."""
    if (
        not _SM80_HAS_CUDA
        or not current_platform.is_device_capability_family(80)
        or os.environ.get("INKLING_SM80_NO_HYBRID_BLEND") == "1"
        or dest.dtype != torch.float32
        or message.dtype != torch.float32
        or down.dtype != torch.bfloat16
        or gain_forget.dtype != torch.bfloat16
        or norm_weight.dtype != torch.bfloat16
        or dest.ndim != 3
        or message.ndim != 3
        or dest.shape != message.shape
    ):
        return None
    s, m, d = message.shape
    r = down.shape[1]
    if (
        d != 2304
        or m <= 0
        or r not in (16, 32, 64, 128)
        or down.shape != (s, r, 2 * d)
        or gain_forget.shape != (s, 2 * d, r)
        or norm_weight.shape != (s, d)
        or dest.stride(1) != d
        or dest.stride(2) != 1
        or not message.is_contiguous()
        or down.stride(-1) != 1
        or gain_forget.stride(-1) != 1
        or norm_weight.stride(1) != 1
    ):
        return None
    x_tmp = torch.empty((s, m, d), device=dest.device, dtype=torch.float32)

    # 1u decode 3-kernel fast path (S == 1, M <= 8): fuses pair_rms_norm + ctrl_down directly into bf16 H (1.08x)
    if s == 1 and m <= 8 and os.environ.get("INKLING_SM80_NO_FUSED_PNCD") != "1":
        hidden = _sm80_fused_pair_norm_ctrl_down(dest, message, down, 1e-6)
        _ctrl_gf_blend_direct_h_kernel[(triton.cdiv(d, 32), s)](
            hidden,
            gain_forget,
            dest,
            message,
            x_tmp,
            m,
            d,
            gain_forget.stride(0),
            gain_forget.stride(1),
            dest.stride(0),
            R=r,
            BN=32,
            MP=16,
            num_warps=4,
            num_stages=3,
            enable_fp_fusion=False,
        )
        return _sm80_rms_norm_f32_to_bf16(x_tmp, norm_weight, norm_eps)

    from .loop import pair_rms_norm

    pair = pair_rms_norm(dest, message)

    if m <= (SM80_MAX_M if s > 1 else 20):
        mp = 16 if m <= 16 else (32 if m <= 32 else 64)
        if s > 1:
            splits = 6
            if m <= 16:
                bnd, bkd, w_d, st_d, bngf, w_gf = 32, 128, 4, 3, 32, 4
            else:
                bnd, bkd, w_d, st_d, bngf, w_gf = 64, 128, 4, 4, 32, 4
        else:
            splits = 8
            bnd, bkd, w_d, st_d, bngf, w_gf = 32, 128, 4, 3, 32, 4
        part = torch.empty((splits, s, mp, r), device=dest.device, dtype=torch.float32)
        k_per = triton.cdiv(triton.cdiv(2 * d, splits), bkd) * bkd
        _ctrl_down_kernel[(triton.cdiv(r, bnd), splits, s)](
            pair,
            down,
            part,
            m,
            r,
            2 * d,
            pair.stride(0),
            pair.stride(1),
            down.stride(0),
            down.stride(1),
            K_PER=k_per,
            BN=bnd,
            BK=bkd,
            MP=mp,
            num_warps=w_d,
            num_stages=st_d,
        )
        _ctrl_gf_blend_kernel[(triton.cdiv(d, bngf), s)](
            part,
            gain_forget,
            dest,
            message,
            x_tmp,
            m,
            d,
            gain_forget.stride(0),
            gain_forget.stride(1),
            dest.stride(0),
            SPLITS=splits,
            R=r,
            BN=bngf,
            MP=mp,
            num_warps=w_gf,
            num_stages=3,
            enable_fp_fusion=False,
        )
    else:
        # Large M (prefill & 8u S=1): cuBLAS bmm for down_proj, fused Triton gain/forget+blend
        # (32, 32, w=2, s=3) beats (64, 128, w=4, s=3) by 1.5-1.6x in prefill (-14ms/chunk)
        hidden = torch.bmm(pair, down.transpose(1, 2))
        if m >= 128:
            bm, bn, nw = 32, 32, 2
        else:
            bm, bn, nw = 16, 64, 4
        _ctrl_gf_blend_prefill_kernel[(triton.cdiv(m, bm), triton.cdiv(d, bn), s)](
            hidden,
            gain_forget,
            dest,
            message,
            x_tmp,
            m,
            d,
            r,
            hidden.stride(0),
            hidden.stride(1),
            gain_forget.stride(0),
            gain_forget.stride(1),
            dest.stride(0),
            BM=bm,
            BN=bn,
            num_warps=nw,
            num_stages=3,
            enable_fp_fusion=False,
        )
    return _sm80_rms_norm_f32_to_bf16(x_tmp, norm_weight, norm_eps)


@triton.jit
def _lm_head_softcap_kernel(
    X,
    W,
    B,
    Y,
    M,
    N,
    K,
    SX,
    SW,
    SY,
    softcap,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    cmask = cols < N
    rmask = rows < M
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            cmask[None, :] & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            rmask[:, None] & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    if HAS_BIAS:
        acc += tl.load(B + cols, cmask, other=0).to(tl.float32)[None, :]
    acc = tldevice.tanh(acc / softcap) * softcap
    tl.store(
        Y + rows[:, None] * SY + cols[None, :], acc, rmask[:, None] & cmask[None, :]
    )


def sm80_lm_head_softcap(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    softcap: float,
) -> torch.Tensor | None:
    """sm_80: fused fp32 lm_head GEMV + softcap(tanh(x/c)*c) for M <= 64 (bit-exact)."""
    if (
        not current_platform.is_device_capability_family(80)
        or not x.is_cuda
        or weight.device != x.device
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.ndim < 2
        or weight.ndim != 2
        or x.shape[-1] != weight.shape[1]
        or x.stride(-1) != 1
        or weight.stride(-1) != 1
    ):
        return None
    if bias is not None and (bias.device != x.device or bias.stride(0) != 1):
        return None
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    m, k = x2.shape
    if not 0 < m <= SM80_MAX_M:
        return None
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    if m <= 16:
        mp, bn, bk, warps, stages = 16, 64, 256, 8, 4
    elif m <= 32:
        mp, bn, bk, warps, stages = 32, 128, 128, 4, 3
    else:
        mp, bn, bk, warps, stages = 64, 128, 128, 4, 4
    _lm_head_softcap_kernel[(triton.cdiv(n, bn),)](
        x2,
        weight,
        bias if bias is not None else x2,
        out,
        m,
        n,
        k,
        x2.stride(0),
        weight.stride(0),
        out.stride(0),
        float(softcap),
        HAS_BIAS=bias is not None,
        BN=bn,
        BK=bk,
        MP=mp,
        num_warps=warps,
        num_stages=stages,
    )
    return out.view(*lead, n)


def sm80_draft_base_logits(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor | None:
    """sm_80: bf16 draft base_logits GEMV up to M <= 64 (beats cuBLAS by -95us at 8u M=32!)."""
    if (
        not current_platform.is_device_capability_family(80)
        or not x.is_cuda
        or weight.device != x.device
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.ndim < 2
        or weight.ndim != 2
        or x.shape[-1] != weight.shape[1]
        or x.stride(-1) != 1
        or weight.stride(-1) != 1
    ):
        return None
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    m, k = x2.shape
    if not 0 < m <= SM80_MAX_M:
        return None
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    mp = 16 if m <= 16 else (32 if m <= 32 else 64)
    bn, bk, warps, stages = _sm80_tiles(n, k, m)
    _gemv_kernel[(triton.cdiv(n, bn),)](
        x2,
        weight,
        x2,
        out,
        m,
        n,
        k,
        x2.stride(0),
        weight.stride(0),
        out.stride(0),
        HAS_BIAS=False,
        BN=bn,
        BK=bk,
        MP=mp,
        num_warps=warps,
        num_stages=stages,
    )
    return out.view(*lead, n)


@triton.jit
def _gemv_splitk_kernel(
    X,
    W,
    P,
    M,
    N,
    K,
    SX,
    SW,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
    K_PER: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    cols = pid_n * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    k_start = pid_k * K_PER
    k_end = tl.minimum(k_start + K_PER, K)
    for k0 in range(k_start, k_end, BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + cols[None, :].to(tl.int64) * SW + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + rows[:, None] * SX + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    tl.store(
        P + (pid_k * M + rows[:, None]).to(tl.int64) * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def _splitk_reduce_kernel(P, Y, M, N, SY, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(1).to(tl.int64)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    acc = tl.zeros((BLOCK,), tl.float32)
    for sp in tl.static_range(SPLIT_K):
        acc += tl.load(P + ((sp * M + row) * N + cols).to(tl.int64), mask, other=0.0)
    tl.store(Y + row * SY + cols, acc.to(Y.dtype.element_ty), mask)


def sm80_draft_dn_gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor | None:
    """sm_80: split-K=2 Triton GEMM for draft MLP down_proj (N=2304, K=9216) at 16 < M <= 64."""
    if (
        not current_platform.is_device_capability_family(80)
        or not x.is_cuda
        or weight.device != x.device
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.ndim != 2
        or weight.ndim != 2
        or x.stride(-1) != 1
        or weight.stride(-1) != 1
    ):
        return None
    m, k = x.shape
    n = weight.shape[0]
    if not (16 < m <= SM80_MAX_M and k == weight.shape[1] and k >= 8192 and n <= 4096):
        return None
    sk, bn, bk = 2, 64, 128
    mp = 32 if m <= 32 else 64
    k_per = triton.cdiv(triton.cdiv(k, sk), bk) * bk
    part = torch.empty((sk, m, n), device=x.device, dtype=torch.float32)
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    _gemv_splitk_kernel[(triton.cdiv(n, bn), sk)](
        x,
        weight,
        part,
        m,
        n,
        k,
        x.stride(0),
        weight.stride(0),
        BN=bn,
        BK=bk,
        MP=mp,
        K_PER=k_per,
        num_warps=4,
        num_stages=3,
    )
    _splitk_reduce_kernel[(triton.cdiv(n, 512), m)](
        part, out, m, n, out.stride(0), SPLIT_K=sk, BLOCK=512, num_warps=4
    )
    return out


@triton.jit
def _ctrl_down_kernel(
    X,
    W,
    P,
    M,
    N,
    K,
    SXS,
    SXM,
    SWS,
    SWN,
    K_PER: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    s = tl.program_id(2)
    S = tl.num_programs(2)
    cols = pid * BN + tl.arange(0, BN)
    rows = tl.arange(0, MP)
    acc = tl.zeros((MP, BN), tl.float32)
    for k0 in range(split * K_PER, tl.minimum(split * K_PER + K_PER, K), BK):
        ks = k0 + tl.arange(0, BK)
        w = tl.load(
            W + s.to(tl.int64) * SWS + cols[None, :].to(tl.int64) * SWN + ks[:, None],
            (cols[None, :] < N) & (ks[:, None] < K),
            other=0,
        )
        x = tl.load(
            X + s.to(tl.int64) * SXS + rows[:, None] * SXM + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        acc += tl.dot(x, w)
    tl.store(
        P + ((split * S + s) * MP + rows[:, None]) * N + cols[None, :],
        acc,
        cols[None, :] < N,
    )


@triton.jit
def _ctrl_gf_kernel(
    P,
    W,
    Y,
    M,
    N,
    SWS,
    SWN,
    SYS,
    SYM,
    SPLITS: tl.constexpr,
    R: tl.constexpr,
    BN: tl.constexpr,
    MP: tl.constexpr,
):
    pid = tl.program_id(0)
    s = tl.program_id(1)
    S = tl.num_programs(1)
    rows = tl.arange(0, MP)
    rk = tl.arange(0, R)
    h = tl.zeros((MP, R), tl.float32)
    for sp in tl.static_range(SPLITS):
        h += tl.load(P + ((sp * S + s) * MP + rows[:, None]) * R + rk[None, :])
    h = h.to(tl.bfloat16)
    cols = pid * BN + tl.arange(0, BN)
    w = tl.load(
        W + s.to(tl.int64) * SWS + cols[None, :].to(tl.int64) * SWN + rk[:, None],
        cols[None, :] < N,
        other=0,
    )
    acc = tl.dot(h, w)
    tl.store(
        Y + s.to(tl.int64) * SYS + rows[:, None] * SYM + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def sm80_controller_gain_forget(
    pair: torch.Tensor, down: torch.Tensor, gain_forget: torch.Tensor
) -> torch.Tensor | None:
    if (
        current_platform.has_device_capability(90)
        or pair.dtype != torch.bfloat16
        or down.dtype != torch.bfloat16
        or gain_forget.dtype != torch.bfloat16
        or pair.ndim != 3
        or not 0 < pair.shape[1] <= MAX_M
        or pair.stride(-1) != 1
        or down.stride(-1) != 1
        or gain_forget.stride(-1) != 1
        or down.shape[1] not in (16, 32, 64, 128)
        or gain_forget.shape[2] != down.shape[1]
        or pair.shape[2] != down.shape[2]
        or not pair.shape[0] == down.shape[0] == gain_forget.shape[0]
    ):
        return None
    splits, bn_gf, warps = (6, 128, 4) if pair.shape[0] > 1 else (8, 128, 4)
    s, m, k = pair.shape
    r = down.shape[1]
    n = gain_forget.shape[1]
    part = torch.empty((splits, s, 16, r), device=pair.device, dtype=torch.float32)
    k_per = triton.cdiv(triton.cdiv(k, splits), 64) * 64
    _ctrl_down_kernel[(triton.cdiv(r, 16), splits, s)](
        pair,
        down,
        part,
        m,
        r,
        k,
        pair.stride(0),
        pair.stride(1),
        down.stride(0),
        down.stride(1),
        K_PER=k_per,
        BN=16,
        BK=64,
        MP=16,
        num_warps=warps,
        num_stages=3,
    )
    out = torch.empty((s, m, n), device=pair.device, dtype=torch.float32)
    _ctrl_gf_kernel[(triton.cdiv(n, bn_gf), s)](
        part,
        gain_forget,
        out,
        m,
        n,
        gain_forget.stride(0),
        gain_forget.stride(1),
        out.stride(0),
        out.stride(1),
        SPLITS=splits,
        R=r,
        BN=bn_gf,
        MP=16,
        num_warps=4,
    )
    return out


def sm12x_bf16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    m, k = x2.shape
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
    bn, bk, warps, stages = _BN, _BK, 4, 3
    mp = 16 if m <= 16 else (32 if m <= 32 else 64)
    if not current_platform.is_device_capability_family(120):
        bn, bk, warps, stages = _sm80_tiles(n, k, m)
    if m:
        _gemv_kernel[(triton.cdiv(n, bn),)](
            x2,
            weight,
            bias,
            out,
            m,
            n,
            k,
            x2.stride(0),
            weight.stride(0),
            out.stride(0),
            HAS_BIAS=bias is not None,
            BN=bn,
            BK=bk,
            MP=mp,
            num_warps=warps,
            num_stages=stages,
        )
    return out.view(*lead, n)


def use_sm12x_bf16_gemv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> bool:
    if not (
        x.is_cuda
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim >= 2
        and weight.ndim == 2
        and x.shape[-1] == weight.shape[1]
        and x.stride(-1) == weight.stride(-1) == 1
        and (
            bias is None
            or (
                bias.dtype == torch.bfloat16
                and bias.device == x.device
                and bias.stride(0) == 1
            )
        )
    ):
        return False
    m = x.numel() // x.shape[-1]
    if current_platform.is_device_capability_family(120):
        return 0 < m <= MAX_M
    if current_platform.is_device_capability_family(80):
        if 0 < m <= MAX_M:
            return True
        if 16 < m <= SM80_MAX_M and bias is None:
            n, k = weight.shape
            # sm_80: Triton _gemv_kernel also beats cuBLAS + splitKreduce at 4u/8u (M=17..64) for:
            # - qkvr_local (N~8576, K<=2560)
            # - o_proj (N=2304, K=3072): 14.3/15.4us vs cuBLAS 17.1-22.7us (-0.14..0.35ms/step)
            # - draft_dn (N=2304, K=8192): 36.8/40.4us vs cuBLAS 48.6/51.5us (-0.14ms/step)
            # - draft_up (N=16384, K=2304): 72.4/69.8us vs cuBLAS 75.1/75.4us (-67us/step)
            # - draft_qkv (N=2560, K=2304, M<=32): 22.9us vs cuBLAS 24.2us
            if 8000 <= n <= 10000 and k <= 2560:
                return True
            if n == 2304 and (k == 3072 or k == 8192):
                return True
            if n == 16384 and k == 2304:
                return True
            if n == 2560 and k == 2304 and m <= 32:
                return True
        return False
    return False
