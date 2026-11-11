# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 per-token activation producers for W4A8 Marlin consumers (Qwen3.8-27B).

Every W4A8 Marlin linear quantizes its BF16 input itself (per_token_quant_int8:
one more read of the activation and a kernel per linear). With
VLLM_CMP_INT8_ACT_FUSION=1 the Marlin kernel advertises kInt8DynamicTokenSym
(QuantizedActivation contract, fusion/quant_activation.py) and the producers
here emit the int8 activation and its per-token scale directly:

  * gemma (add +) RMSNorm -> int8  (in_proj_qkvz / qkv_proj, gate_up_proj)
  * SiLU-and-mul -> int8           (down_proj, via maybe_fused_act_quant)

Numerics follow the unfused path: the producer rounds its output to the
activation dtype first (as the unfused norm / silu_and_mul write it), then
quantizes exactly like per_token_quant_int8 (absmax / 127, round half away).
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
)
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)
ENABLED = os.environ.get("VLLM_CMP_INT8_ACT_FUSION", "0") == "1"
# GDN gated norm -> out_proj int8 (separate switch until confirmed)
GDN_NORM = os.environ.get("VLLM_CMP_INT8_ACT_FUSION_GDN", "0") == "1"

KEY = kInt8DynamicTokenSym


@triton.jit
def _quant_row_store(y, q_ptr, s_ptr, row, cols, mask, g_ptr, HAS_G: tl.constexpr):
    # y: fp32 values already rounded to the activation dtype. With HAS_G the
    # stored scale is (absmax / 127) * g, g the consumer's W4A8 Marlin global
    # scale: the same fp32 multiply apply_gptq_marlin_linear did as a separate
    # kernel, so the Marlin input is bit-identical.
    absmax = tl.maximum(tl.max(tl.abs(y)), 1e-10)
    q = tl.extra.cuda.libdevice.round(y * (127 / absmax)).to(tl.int8)
    tl.store(q_ptr + cols, q, mask=mask)
    sc = absmax / 127
    if HAS_G:
        sc = sc * tl.load(g_ptr).to(tl.float32)
    tl.store(s_ptr + row, sc)


def consumer_gscale(linear) -> torch.Tensor | None:
    """The W4A8 Marlin global activation scale of a consumer linear (or None).
    With the fusion on, every int8 producer folds it into its scales and the
    Marlin call skips its own multiply (marlin.py)."""
    g = getattr(linear, "input_global_scale", None) if linear is not None else None
    return g if isinstance(g, torch.Tensor) else None


@triton.jit
def _quant_int8_g_kernel(
    x_ptr, q_ptr, s_ptr, g_ptr, stride_x, N, BLOCK: tl.constexpr, HAS_G: tl.constexpr
):
    # per_token_quant_int8 (int8_utils) with the global-scale multiply folded in
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    _quant_row_store(x, q_ptr + row * N, s_ptr, row, cols, mask, g_ptr, HAS_G)


def per_token_quant_int8_g(x: torch.Tensor, gscale: torch.Tensor | None):
    """X [M, N] (row stride arbitrary, unit column stride) -> (int8 [M, N],
    fp32 scales [M, 1] already multiplied by gscale)."""
    assert x.dim() == 2 and x.stride(-1) == 1
    m, n = x.shape
    q = torch.empty((m, n), dtype=torch.int8, device=x.device)
    s = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    if m:
        block = triton.next_power_of_2(n)
        _quant_int8_g_kernel[(m,)](
            x,
            q,
            s,
            gscale if gscale is not None else s,
            x.stride(0),
            n,
            BLOCK=block,
            HAS_G=gscale is not None,
            num_warps=min(max(block // 256, 1), 8),
            num_stages=1,
        )
    return q, s


@triton.jit
def _gemma_add_rms_norm_int8_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    q_ptr,
    s_ptr,
    y_ptr,
    g_ptr,
    eps,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    WRITE_Y: tl.constexpr,
    HAS_G: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0)
    h = x.to(tl.float32)
    if HAS_RESIDUAL:
        r = tl.load(res_ptr + row * H + cols, mask=mask, other=0.0)
        h += r.to(tl.float32)
        # the norm uses the fp32 sum, like flashinfer's fused_add_rmsnorm
        tl.store(res_ptr + row * H + cols, h.to(res_ptr.dtype.element_ty), mask=mask)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    var = tl.sum(h * h) / H
    rstd = tl.rsqrt(var + eps)
    y = (h * rstd * (1.0 + w.to(tl.float32))).to(x_ptr.dtype.element_ty)
    if WRITE_Y:
        tl.store(y_ptr + row * H + cols, y, mask=mask)
    _quant_row_store(
        y.to(tl.float32), q_ptr + row * H, s_ptr, row, cols, mask, g_ptr, HAS_G
    )


def gemma_add_rms_norm_int8(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    write_y: bool = False,
    gscale: torch.Tensor | None = None,
):
    """Gemma RMSNorm of x (+ residual, updated in place) -> (int8 [M, H],
    scales fp32 [M, 1], normed activation or None). x is not modified."""
    assert x.dim() == 2 and x.is_contiguous()
    assert residual is None or (residual.shape == x.shape and residual.is_contiguous())
    m, h = x.shape
    q = torch.empty((m, h), dtype=torch.int8, device=x.device)
    s = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    y = torch.empty_like(x) if write_y else None
    if m == 0:
        return q, s, y
    _gemma_add_rms_norm_int8_kernel[(m,)](
        x,
        residual if residual is not None else x,
        weight,
        q,
        s,
        y if y is not None else x,
        gscale if gscale is not None else s,
        eps,
        H=h,
        BLOCK=triton.next_power_of_2(h),
        HAS_RESIDUAL=residual is not None,
        WRITE_Y=write_y,
        HAS_G=gscale is not None,
        num_warps=8 if m <= 512 else 4,
    )
    return q, s, y


@triton.jit
def _gdn_gated_rms_norm_int8_kernel(
    x_ptr,
    z_ptr,
    w_ptr,
    q_ptr,
    s_ptr,
    g_ptr,
    stride_x,
    stride_z,
    eps,
    NH: tl.constexpr,
    NHP: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    SIGMOID: tl.constexpr,
    HAS_G: tl.constexpr,
):
    # one token per program, all heads: per-head gated RMS norm (as
    # fused_gdn_gated_rms_norm), rounded to the dtype, then int8 per token
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, NHP)
    c = tl.arange(0, BLOCK_N)
    mask = (h[:, None] < NH) & (c[None, :] < N)
    off = h[:, None] * N + c[None, :]
    xr = tl.load(x_ptr + row * stride_x + off, mask=mask, other=0.0)
    x = xr.to(tl.float32)
    z = tl.load(z_ptr + row * stride_z + off, mask=mask, other=0.0).to(tl.float32)
    if SIGMOID:
        g = tl.sigmoid(z)
    else:
        g = z * tl.sigmoid(z)
    if not NORM_BEFORE_GATE:
        x *= g
    xbar = tl.where(mask, x, 0.0)
    rstd = tl.rsqrt(tl.sum(xbar * xbar, axis=1) / N + eps)
    w = tl.load(w_ptr + c, mask=c < N, other=0.0).to(tl.float32)
    y = x * rstd[:, None] * w[None, :]
    if NORM_BEFORE_GATE:
        y *= g
    y = tl.where(mask, y.to(xr.dtype).to(tl.float32), 0.0)
    absmax = tl.maximum(tl.max(tl.max(tl.abs(y), axis=1), axis=0), 1e-10)
    q = tl.extra.cuda.libdevice.round(y * (127 / absmax)).to(tl.int8)
    tl.store(q_ptr + row * (NH * N) + off, q, mask=mask)
    sc = absmax / 127
    if HAS_G:
        sc = sc * tl.load(g_ptr).to(tl.float32)
    tl.store(s_ptr + row, sc)


def gdn_gated_rms_norm_int8(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_v_heads: int,
    head_v_dim: int,
    norm_before_gate: bool,
    activation: str,
    gscale: torch.Tensor | None = None,
):
    """fused_gdn_gated_rms_norm with the output quantized int8 per token:
    (int8 [T, NH*N], scales fp32 [T, 1])."""
    assert x.dim() == 2 and x.stride(-1) == 1 and z.stride(-1) == 1
    assert x.shape[1] == num_v_heads * head_v_dim and activation in (
        "swish",
        "silu",
        "sigmoid",
    )
    t = x.shape[0]
    q = torch.empty((t, num_v_heads * head_v_dim), dtype=torch.int8, device=x.device)
    s = torch.empty((t, 1), dtype=torch.float32, device=x.device)
    if t:
        _gdn_gated_rms_norm_int8_kernel[(t,)](
            x,
            z,
            weight,
            q,
            s,
            gscale if gscale is not None else s,
            x.stride(0),
            z.stride(0),
            eps,
            NH=num_v_heads,
            NHP=triton.next_power_of_2(num_v_heads),
            N=head_v_dim,
            BLOCK_N=triton.next_power_of_2(head_v_dim),
            NORM_BEFORE_GATE=norm_before_gate,
            SIGMOID=activation == "sigmoid",
            HAS_G=gscale is not None,
            num_warps=8,
        )
    return q, s


@triton.jit
def _silu_mul_int8_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    g_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_G: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    g = tl.load(x_ptr + row * (2 * D) + cols, mask=mask, other=0.0)
    u = tl.load(x_ptr + row * (2 * D) + D + cols, mask=mask, other=0.0)
    gf = g.to(tl.float32)
    # silu_and_mul: silu rounded to the dtype, then the product rounded
    act = (gf / (1.0 + tl.extra.cuda.libdevice.exp(-gf))).to(g.dtype)
    y = (act.to(tl.float32) * u.to(tl.float32)).to(g.dtype).to(tl.float32)
    _quant_row_store(y, q_ptr + row * D, s_ptr, row, cols, mask, g_ptr, HAS_G)


def silu_and_mul_int8(x: torch.Tensor, gscale: torch.Tensor | None = None):
    """X [..., 2D] -> (int8 [M, D], scales fp32 [M, 1])."""
    d = x.shape[-1] // 2
    x2 = x.reshape(-1, 2 * d)
    assert x2.is_contiguous()
    m = x2.shape[0]
    q = torch.empty((m, d), dtype=torch.int8, device=x.device)
    s = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    if m:
        _silu_mul_int8_kernel[(m,)](
            x2,
            q,
            s,
            gscale if gscale is not None else s,
            D=d,
            BLOCK=triton.next_power_of_2(d),
            HAS_G=gscale is not None,
            num_warps=16 if d > 8192 else 8,
        )
    return q, s


def _silu_and_mul_int8_producer(x: torch.Tensor, linear) -> QuantizedActivation:
    d = x.shape[-1] // 2
    out_shape = x.shape[:-1] + (d,)
    q, s = silu_and_mul_int8(x, consumer_gscale(linear))
    return QuantizedActivation(
        data=q.view(out_shape),
        scale=s,
        orig_dtype=x.dtype,
        orig_shape=out_shape,
        quant_key=KEY,
    )


def register() -> None:
    """Add the SiLU-and-mul int8 producer to maybe_fused_act_quant's table."""
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.fusion import fused_act_quant as _faq

    _faq._FUSED_ACT_QUANT[(SiluAndMul, KEY)] = _silu_and_mul_int8_producer


if ENABLED:
    register()
    logger.info("cmp int8 activation fusion enabled (VLLM_CMP_INT8_ACT_FUSION=1)")
