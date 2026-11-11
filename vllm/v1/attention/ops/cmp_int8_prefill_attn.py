# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill attention with INT8 Q.K^T (SageAttention-style) (Qwen3.8-27B).

On the CMP 170HX (GA100, power-capped) FA2's prefill kernel runs at ~120
TFLOPS at head dim 256, near the capped BF16 tensor peak; INT8 MMA has twice
the BF16 rate. This kernel computes S = Q.K^T on INT8 tensor cores and keeps
P.V in BF16 with FP32 accumulation:

  * K is centred per (kv head, channel) by its mean over the keys (q.(k - m)
    differs from q.k by a per-row constant, which softmax ignores), then
    quantized per (token, kv head): k8 = rint(k' / sk), sk = max|k'| / 127.
  * Q is quantized per (token, head) the same way, sm_scale * log2(e) folded
    into sq.
  * S_ij = (q8_i . k8_j) * sq_i * sk_j, online softmax in base 2.

Contiguous K/V of one request ([Nk, Hkv, D]), causal aligned bottom-right
(query row i sees keys <= i + Nk - Nq), GQA by head index. Lab-only until a
paired confirm.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

LOG2E = 1.4426950408889634
# VLLM_CMP_INT8_ATTN_GQA_PACK=1: the G query heads of a kv head share one program
GQA_PACK = __import__("os").environ.get("VLLM_CMP_INT8_ATTN_GQA_PACK", "0") == "1"


@triton.jit
def _quant_rows_kernel(
    x_ptr,
    mean_ptr,
    y_ptr,
    s_ptr,
    mult,
    H,
    TOTAL,
    D: tl.constexpr,
    HAS_MEAN: tl.constexpr,
    ROWS: tl.constexpr,
):
    # one program: ROWS (token, head) rows of D values -> int8 + scale
    pid = tl.program_id(0)
    r = pid * ROWS + tl.arange(0, ROWS)
    ok = r < TOTAL
    d = tl.arange(0, D)
    x = tl.load(
        x_ptr + r[:, None].to(tl.int64) * D + d[None, :], mask=ok[:, None], other=0.0
    ).to(tl.float32)
    if HAS_MEAN:
        x = x - tl.load(mean_ptr + (r % H)[:, None] * D + d[None, :])
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-20)
    sc = amax / 127.0
    y = tl.extra.cuda.libdevice.rint(x / sc[:, None])
    tl.store(
        y_ptr + r[:, None].to(tl.int64) * D + d[None, :],
        y.to(tl.int8),
        mask=ok[:, None],
    )
    tl.store(s_ptr + r, sc * mult, mask=ok)


def quant_rows(
    x: torch.Tensor, mult: float, mean: torch.Tensor | None = None, rows: int = 16
):
    """X [N, H, D] (contiguous) -> (int8 [N, H, D], scales fp32 [N, H])."""
    n, h, d = x.shape
    total = n * h
    assert x.is_contiguous()
    y = torch.empty((n, h, d), dtype=torch.int8, device=x.device)
    s = torch.empty((n, h), dtype=torch.float32, device=x.device)
    _quant_rows_kernel[(triton.cdiv(total, rows),)](
        x,
        mean if mean is not None else x,
        y,
        s,
        mult,
        h,
        total,
        D=d,
        HAS_MEAN=mean is not None,
        ROWS=rows,
        num_warps=4,
    )
    return y, s


@triton.jit
def _attn_fwd_kernel(
    q_ptr,
    qs_ptr,
    k_ptr,
    ks_ptr,
    v_ptr,
    o_ptr,
    qf_ptr,
    kf_ptr,
    Nq,
    Nk,
    H: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    INT8: tl.constexpr,
    STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    hk = h // (H // HKV)
    off = Nk - Nq
    m_rows = pid_m * BM + tl.arange(0, BM)
    m_ok = m_rows < Nq
    d = tl.arange(0, D)
    if INT8:
        q = tl.load(
            q_ptr + (m_rows[:, None].to(tl.int64) * H + h) * D + d[None, :],
            mask=m_ok[:, None],
            other=0,
        )
        sq = tl.load(qs_ptr + m_rows * H + h, mask=m_ok, other=0.0)
    else:
        q = tl.load(
            qf_ptr + (m_rows[:, None].to(tl.int64) * H + h) * D + d[None, :],
            mask=m_ok[:, None],
            other=0.0,
        )
        sq = tl.load(
            qs_ptr + m_rows * H + h, mask=m_ok, other=0.0
        )  # sm_scale*log2e per row
    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    # keys visible to this block: [0, off + last row]
    hi = tl.minimum(off + (pid_m + 1) * BM, Nk)
    full = tl.maximum(
        tl.minimum(off + pid_m * BM + 1, Nk) // BN * BN, 0
    )  # blocks without masking
    for start in tl.range(0, hi, BN, num_stages=STAGES):
        n_cols = start + tl.arange(0, BN)
        n_ok = n_cols < Nk
        if INT8:
            k = tl.load(
                k_ptr + (n_cols[:, None].to(tl.int64) * HKV + hk) * D + d[None, :],
                mask=n_ok[:, None],
                other=0,
            )
            sk = tl.load(ks_ptr + n_cols * HKV + hk, mask=n_ok, other=0.0)
            s = (
                tl.dot(q, tl.trans(k), out_dtype=tl.int32).to(tl.float32)
                * sq[:, None]
                * sk[None, :]
            )
        else:
            k = tl.load(
                kf_ptr + (n_cols[:, None].to(tl.int64) * HKV + hk) * D + d[None, :],
                mask=n_ok[:, None],
                other=0.0,
            )
            s = tl.dot(q, tl.trans(k)) * sq[:, None]
        if start + BN > full:
            vis = (n_cols[None, :] <= (m_rows[:, None] + off)) & n_ok[None, :]
            s = tl.where(vis, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_new = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(
            v_ptr + (n_cols[:, None].to(tl.int64) * HKV + hk) * D + d[None, :],
            mask=n_ok[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    out = acc / tl.maximum(l_i, 1e-30)[:, None]
    tl.store(
        o_ptr + (m_rows[:, None].to(tl.int64) * H + h) * D + d[None, :],
        out.to(o_ptr.dtype.element_ty),
        mask=m_ok[:, None],
    )


@triton.jit
def _attn_fwd_gqa_kernel(
    q_ptr,
    qs_ptr,
    k_ptr,
    ks_ptr,
    v_ptr,
    o_ptr,
    Nq,
    Nk,
    H: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    STAGES: tl.constexpr,
):
    # GQA-packed INT8 variant: one program owns BM rows of the flattened
    # (token, q head of this kv head) sequence, so the G = H / HKV query heads
    # that share a kv head read each K/V block once instead of G times.
    G: tl.constexpr = H // HKV
    pid_m = tl.program_id(0)
    hk = tl.program_id(1)
    off = Nk - Nq
    r = pid_m * BM + tl.arange(0, BM)
    t = r // G
    h = hk * G + r % G
    r_ok = t < Nq
    d = tl.arange(0, D)
    qrow = t.to(tl.int64) * H + h
    q = tl.load(q_ptr + qrow[:, None] * D + d[None, :], mask=r_ok[:, None], other=0)
    sq = tl.load(qs_ptr + qrow, mask=r_ok, other=0.0)
    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    t_lo = (pid_m * BM) // G
    t_hi = tl.minimum((pid_m * BM + BM - 1) // G, Nq - 1)
    hi = tl.minimum(off + t_hi + 1, Nk)
    full = tl.maximum(tl.minimum(off + t_lo + 1, Nk) // BN * BN, 0)
    for start in tl.range(0, hi, BN, num_stages=STAGES):
        n_cols = start + tl.arange(0, BN)
        n_ok = n_cols < Nk
        k = tl.load(
            k_ptr + (n_cols[:, None].to(tl.int64) * HKV + hk) * D + d[None, :],
            mask=n_ok[:, None],
            other=0,
        )
        sk = tl.load(ks_ptr + n_cols * HKV + hk, mask=n_ok, other=0.0)
        s = (
            tl.dot(q, tl.trans(k), out_dtype=tl.int32).to(tl.float32)
            * sq[:, None]
            * sk[None, :]
        )
        if start + BN > full:
            vis = (n_cols[None, :] <= (t[:, None] + off)) & n_ok[None, :]
            s = tl.where(vis, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_new = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(
            v_ptr + (n_cols[:, None].to(tl.int64) * HKV + hk) * D + d[None, :],
            mask=n_ok[:, None],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    out = acc / tl.maximum(l_i, 1e-30)[:, None]
    tl.store(
        o_ptr + qrow[:, None] * D + d[None, :],
        out.to(o_ptr.dtype.element_ty),
        mask=r_ok[:, None],
    )


def prefill_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    sm_scale: float,
    int8: bool = True,
    bm: int = 64,
    bn: int = 64,
    warps: int = 4,
    stages: int = 2,
    gqa_pack: bool | None = None,
) -> torch.Tensor:
    """Q [Nq, H, D], k/v [Nk, Hkv, D] (contiguous, bf16/fp16), causal bottom-right."""
    nq, h, d = q.shape
    nk, hkv, _ = k.shape
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    mult = sm_scale * LOG2E
    if int8:
        mean = k.float().mean(0).contiguous()  # [Hkv, D]
        q8, sq = quant_rows(q, mult)
        k8, sk = quant_rows(k, 1.0, mean)
        qf = kf = q
    else:
        q8, k8, sk = q, k, q
        sq = torch.full((nq, h), mult, dtype=torch.float32, device=q.device)
        qf, kf = q, k
    dst = out if out.is_contiguous() else torch.empty_like(q)
    if gqa_pack is None:
        gqa_pack = GQA_PACK
    if int8 and gqa_pack and h % hkv == 0 and h != hkv:
        grid = (triton.cdiv(nq * (h // hkv), bm), hkv)
        _attn_fwd_gqa_kernel[grid](
            q8,
            sq,
            k8,
            sk,
            v,
            dst,
            nq,
            nk,
            H=h,
            HKV=hkv,
            D=d,
            BM=bm,
            BN=bn,
            STAGES=stages,
            num_warps=warps,
        )
        if dst is not out:
            out.copy_(dst)
        return out
    grid = (triton.cdiv(nq, bm), h)
    _attn_fwd_kernel[grid](
        q8,
        sq,
        k8,
        sk,
        v,
        dst,
        qf,
        kf,
        nq,
        nk,
        H=h,
        HKV=hkv,
        D=d,
        BM=bm,
        BN=bn,
        INT8=int8,
        STAGES=stages,
        num_warps=warps,
    )
    if dst is not out:
        out.copy_(dst)
    return out
