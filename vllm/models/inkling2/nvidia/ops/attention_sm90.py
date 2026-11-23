# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 relative-bias paged attention for Hopper and SM12x (Triton).

Stands in for tml_fa4 (prefill) and the tcgen05 decode/verify kernel of
`ops/attention.py`, reading the same sheared relative logits of
`attention_prologue`: column `c` of token `t` holds key
`c + cdiv(past + tile_end, 128) * 128 - (E + 256)`, where `tile_end` ends
the token's 128-row query tile (see `qr_clip_rel_tokens`).

A program owns `BLOCK_M` rows (tokens x GQA group) of one request and KV
head, and one KV split: split `s` of `S` takes key blocks `s, s + S, ...`
of the rows' visible range, so a split count that depends only on the batch
holds for every replay of a CUDA graph.
"""

import os

import torch

from vllm.triton_utils import tl, triton

from .fp8_emu import (
    e4m3_decode_bf16,
    e4m3_decode_fp16_scaled,
    e4m3_decode_fp16_scaled_x4,
)

# VLLM_INKLING2_FP8_PRMT=1: decode SM8x E4M3 KV four codes per word (prmt/lop3),
# bit-identical to the scalar decode.
_FP8_PRMT = tl.constexpr(os.environ.get("VLLM_INKLING2_FP8_PRMT", "0") == "1")
# Screening knobs for the SM8x E4M3 prefill path: load K codes already transposed
# ([D, N], as the bf16 path does) instead of [N, D] + .T; query tile rows and warps
# for byte-KV launches with more than 128 rows (0 = the default tiling).
_FP8_KT = tl.constexpr(os.environ.get("VLLM_INKLING2_FP8_KT", "0") == "1")
_FP8_BM = int(os.environ.get("VLLM_INKLING2_FP8_BM", "0"))
_FP8_WARPS = int(os.environ.get("VLLM_INKLING2_FP8_WARPS", "0"))
_FP8_BN = int(os.environ.get("VLLM_INKLING2_FP8_BN", "0"))
# Prefill tiling knobs for any KV dtype (launches with more than 128 rows; 0 = default):
# VLLM_INKLING2_PF_BM / _BN / _WARPS / _STAGES.
_PF_BM = int(os.environ.get("VLLM_INKLING2_PF_BM", "0"))
_PF_BN = int(os.environ.get("VLLM_INKLING2_PF_BN", "0"))
_PF_WARPS = int(os.environ.get("VLLM_INKLING2_PF_WARPS", "0"))
_PF_STAGES = int(os.environ.get("VLLM_INKLING2_PF_STAGES", "0"))
# VLLM_INKLING2_FP8_STAGE=1: for full-layer prefill tiles over a byte (E4M3) cache,
# decode each request's visible pages once into a bf16 scratch (exact: every E4M3
# value is a bf16) and run the bf16 kernel on it, instead of every query tile
# re-decoding every key block.
_FP8_STAGE = os.environ.get("VLLM_INKLING2_FP8_STAGE", "0") == "1"
_STAGE_SCRATCH: dict = {}


def ensure_fp8_stage_scratch(max_model_len: int, num_kv_heads: int, row: int) -> None:
    """Allocate the bf16 staging scratch once per device (VLLM_INKLING2_FP8_STAGE=1):
    `round_up(max_model_len, 256) + 256` tokens x `num_kv_heads` x `row` (K|V)
    elements, so one request at max_model_len always fits; larger batches fall back
    to the in-kernel decode."""
    if not _FP8_STAGE or not torch.accelerator.is_available():
        return
    device = torch.accelerator.current_accelerator()
    assert device is not None
    device = torch.device(device.type, torch.accelerator.current_device_index())
    tokens = (max_model_len + 255) // 256 * 256 + 256
    need = tokens * num_kv_heads * row
    buf = _STAGE_SCRATCH.get(device)
    if buf is None or buf.numel() < need:
        _STAGE_SCRATCH[device] = torch.empty(need, dtype=torch.bfloat16, device=device)


@triton.jit
def _stage_fp8_kv_kernel(
    KV,
    BlockTable,
    SeqLens,
    Out,
    stride_kv_blk,
    stride_kv_slot,
    stride_kv_h,
    stride_bt,
    max_pages,
    PAGE: tl.constexpr,
    HEADS: tl.constexpr,
    C: tl.constexpr,
):
    """Decode page `p` of request `r` (one [PAGE, KV heads, K|V] block of bytes)
    into scratch page `r * max_pages + p` (bf16, same layout)."""
    p = tl.program_id(0)
    r = tl.program_id(1)
    seqlen = tl.load(SeqLens + r)
    if p * PAGE >= seqlen:
        return
    blk = tl.load(BlockTable + r.to(tl.int64) * stride_bt + p).to(tl.int64)
    dst = (r.to(tl.int64) * max_pages + p) * (PAGE * HEADS * C)
    offs_h = tl.arange(0, HEADS)
    offs_c = tl.arange(0, C)
    src = KV + blk * stride_kv_blk + offs_h[:, None] * stride_kv_h + offs_c[None, :]
    out = Out + dst + offs_h[:, None] * C + offs_c[None, :]
    for slot in range(PAGE):
        codes = tl.load(src + slot * stride_kv_slot)
        tl.store(out + slot * (HEADS * C), e4m3_decode_bf16(codes))


def _stage_fp8_kv(
    kv_bytes: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """bf16 copy of each request's first `ceil(seqlen / PAGE)` pages and its block
    table (request r owns scratch pages `r * max_pages ...`); `kv_bytes` is the
    layer's [blocks, PAGE, KV heads, 2 * D] byte cache. None if the preallocated
    scratch is too small (the caller then decodes in the kernel)."""
    num_reqs = cache_seqlens.shape[0]
    page, heads, c = kv_bytes.shape[1:4]
    assert kv_bytes.stride(3) == 1
    max_pages = triton.cdiv(max_seq_len, page)
    need = num_reqs * max_pages * page * heads * c
    buf = _STAGE_SCRATCH.get(kv_bytes.device)
    if buf is None or buf.numel() < need:
        return None
    _stage_fp8_kv_kernel[(max_pages, num_reqs)](
        kv_bytes,
        block_table,
        cache_seqlens,
        buf,
        kv_bytes.stride(0),
        kv_bytes.stride(1),
        kv_bytes.stride(2),
        block_table.stride(0),
        max_pages,
        PAGE=page,
        HEADS=heads,
        C=c,
        num_warps=4,
    )
    scratch = buf[:need].view(num_reqs * max_pages, page, heads, c)
    table = torch.arange(
        num_reqs * max_pages, device=kv_bytes.device, dtype=torch.int32
    )
    return scratch, table.view(num_reqs, max_pages)


# SM8x E4M3 KV loads come back scaled by 2^-8 (`e4m3_decode_fp16_scaled`).
_U8_KV_SCALE = tl.constexpr(256.0)

# The query tile of tml_fa4 that shears the relative logits.
_SHEAR_TILE = tl.constexpr(128)

# Output of `_rel_attention_kernel`: fp32 output and natural-log LSE (the
# tml_fa4 replacement), the gated bf16 output, or fp32 split partials with
# log2 LSEs (merged by `_rel_decode_combine_kernel`).
MODE_OUT_LSE = 0
MODE_GATED = 1
MODE_PARTIAL = 2


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp2(-x * 1.4426950408889634))


@triton.jit
def _load_fp8_kv(ptr, offsets, mask, dtype: tl.constexpr, D: tl.constexpr):
    if ptr.dtype.element_ty == tl.uint8:
        # HyperInkling SM8x: the E4M3 cache as raw bytes, decoded in software.
        codes = tl.load(
            ptr + offsets[:, None] + tl.arange(0, D)[None, :],
            mask=mask[:, None],
            other=0,
        )
        # fp16 holding value * 2^-8 (see `_U8_KV_SCALE` in the kernel).
        if _FP8_PRMT:
            return e4m3_decode_fp16_scaled_x4(codes)
        return e4m3_decode_fp16_scaled(codes)
    # Load each E4M3 pair once, retaining 16-bit packing for BF16 MMA order.
    half = tl.arange(0, D // 2)
    pair = tl.load(
        ptr.to(tl.pointer_type(tl.uint16))
        + tl.multiple_of(offsets // 2, D // 2)[:, None]
        + half[None, :],
        mask=mask[:, None],
        other=0,
    )
    lo, hi = tl.inline_asm_elementwise(
        "{ .reg .b32 pair; cvt.rn.f16x2.e4m3x2 pair, $2; mov.b32 {$0, $1}, pair; }",
        "=h,=h,h",
        [pair],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    return tl.interleave(lo, hi).to(dtype)


@triton.jit
def _rel_attention_kernel(
    Q,
    K,
    V,
    Rel,
    BlockTable,
    SeqLens,
    CuSeqlensQ,
    Out,
    Lse,
    Gate,
    Sink,
    softmax_scale,
    num_tokens,
    num_splits,
    stride_q_t,
    stride_q_h,
    stride_kv_blk,
    stride_kv_slot,
    stride_kv_h,
    stride_rel_t,
    stride_bt,
    stride_out_t,
    stride_out_h,
    stride_lse_h,
    stride_lse_t,
    stride_gate,
    stride_sink,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    DRAFT: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    TOKENS: tl.constexpr = BLOCK_M // G
    E_PAD: tl.constexpr = E + 256
    LOG2E: tl.constexpr = 1.4426950408889634
    m_tile = tl.program_id(0)
    kv_head = tl.program_id(1) // num_splits
    split = tl.program_id(1) % num_splits
    req = tl.program_id(2)
    q_start = tl.load(CuSeqlensQ + req)
    q_len = tl.load(CuSeqlensQ + req + 1) - q_start
    first_tok = m_tile * TOKENS
    if first_tok >= q_len:
        return
    seqlen_k = tl.load(SeqLens + req)
    past = seqlen_k - q_len

    rows = tl.arange(0, BLOCK_M)
    tok = first_tok + rows // G
    row_valid = (rows < TOKENS * G) & (tok < q_len)
    head = kv_head * G + rows % G
    token = (q_start + tok).to(tl.int64)
    if DRAFT:
        pos = tl.zeros([BLOCK_M], tl.int32) + seqlen_k - 1
        pos_lo = seqlen_k - 1
        pos_hi = seqlen_k - 1
    else:
        pos = past + tok
        pos_lo = past + first_tok
        pos_hi = past + tl.minimum(first_tok + TOKENS, q_len) - 1
    # Visible key blocks of the tile, split `split` taking every
    # `num_splits`-th of them.
    n_lo = 0
    if WINDOW > 0:
        n_lo = tl.maximum(pos_lo - (WINDOW - 1), 0) // BLOCK_N
    n_hi = tl.minimum(pos_hi + 1, seqlen_k)
    n_hi = (n_hi + BLOCK_N - 1) // BLOCK_N
    num_iters = tl.maximum(n_hi - n_lo - split + num_splits - 1, 0) // num_splits

    offs_d = tl.arange(0, D)
    q = tl.load(
        Q + token[:, None] * stride_q_t + head[:, None] * stride_q_h + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )
    if not DRAFT:
        tile_end = tl.minimum((tok // _SHEAR_TILE + 1) * _SHEAR_TILE, q_len)
        rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
        rel_row = Rel + token * stride_rel_t + head * E_PAD - rel_base

    qk_scale = softmax_scale * LOG2E
    U8_KV: tl.constexpr = K.dtype.element_ty == tl.uint8
    if U8_KV:
        # fp16 MMAs against the scaled fp16 keys / values.
        q = q.to(tl.float16)
        qk_scale = qk_scale * _U8_KV_SCALE
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    block_table = BlockTable + req.to(tl.int64) * stride_bt
    kv_off = kv_head * stride_kv_h
    for i in range(num_iters):
        n = n_lo + split + i * num_splits
        keys = n * BLOCK_N + tl.arange(0, BLOCK_N)
        key_valid = keys < seqlen_k
        page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        if U8_KV and _FP8_KT:
            k_codes = tl.load(
                K + kv_ptrs[None, :] + offs_d[:, None],
                mask=key_valid[None, :],
                other=0,
            )
            if _FP8_PRMT:
                k = e4m3_decode_fp16_scaled_x4(k_codes)
            else:
                k = e4m3_decode_fp16_scaled(k_codes)
        elif K.dtype.element_ty.primitive_bitwidth == 8:
            k = _load_fp8_kv(K, kv_ptrs, key_valid, q.dtype, D).T
        else:
            k = tl.load(
                K + kv_ptrs[None, :] + offs_d[:, None],
                mask=key_valid[None, :],
                other=0.0,
            ).to(q.dtype)
        s = tl.dot(q, k) * qk_scale
        dist = pos[:, None] - keys[None, :]
        visible = (dist >= 0) & key_valid[None, :]
        if WINDOW > 0:
            visible &= dist < WINDOW
        if not DRAFT:
            in_extent = visible & (dist < E) & row_valid[:, None]
            bias = tl.load(rel_row[:, None] + keys[None, :], mask=in_extent, other=0.0)
            s += bias.to(tl.float32) * LOG2E
        s = tl.where(visible, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if V.dtype.element_ty.primitive_bitwidth == 8:
            v = _load_fp8_kv(V, kv_ptrs, key_valid, q.dtype, D)
        else:
            v = tl.load(
                V + kv_ptrs[:, None] + offs_d[None, :],
                mask=key_valid[:, None],
                other=0.0,
            ).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    empty = l_i == 0.0
    if U8_KV:
        acc = acc * _U8_KV_SCALE
    o = acc / tl.where(empty, 1.0, l_i)[:, None]
    lse2 = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))
    if MODE == 0:
        tl.store(
            Out
            + token[:, None] * stride_out_t
            + head[:, None] * stride_out_h
            + offs_d[None, :],
            o,
            mask=row_valid[:, None],
        )
        tl.store(
            Lse + head * stride_lse_h + token * stride_lse_t,
            lse2 * 0.6931471805599453,
            mask=row_valid,
        )
    elif MODE == 1:
        cols = head[:, None] * D + offs_d[None, :]
        if not DRAFT:
            gate = tl.load(
                Gate + token[:, None] * stride_gate + cols,
                mask=row_valid[:, None],
                other=0.0,
            )
            sink = tl.load(Sink + token * stride_sink + head, mask=row_valid, other=0.0)
            o = o * _sigmoid(gate.to(tl.float32))
            o = o * _sigmoid(lse2 * 0.6931471805599453 - sink.to(tl.float32))[:, None]
        tl.store(
            Out + token[:, None] * stride_out_t + cols,
            o.to(Out.dtype.element_ty),
            mask=row_valid[:, None],
        )
    else:
        part = (split * num_tokens + token) * H + head
        tl.store(Lse + part, lse2, mask=row_valid)
        tl.store(Out + part[:, None] * D + offs_d[None, :], o, mask=row_valid[:, None])


def _block_m(max_query_len: int, group: int) -> int:
    rows = max_query_len * group
    return 16 if rows <= 16 else 32 if rows <= 32 else 64 if rows <= 64 else 128


def rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    rel_logits: torch.Tensor | None,
    gate: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    softmax_scale: float,
    window: int,
    num_splits: int,
    mode: int,
    max_seq_len: int = 0,
) -> None:
    """Launch `_rel_attention_kernel` over the requests of `cu_seqlens_q`.

    `rel_logits` (sheared, `(T + 128, H, extent + 256)`) is None for the
    DSpark draft: plain block attention with all queries of a request at
    its last position. `window` (keys up to a query's position, inclusive) is
    0 for full layers. `mode` selects the output: `MODE_OUT_LSE` writes the
    fp32 `out` `(T, H, D)` and natural-log `lse` `(H, T)`; `MODE_GATED`
    the bf16 `(T, H * D)` `out` (gated by `gate` and `sinks` unless
    drafting; a single split); `MODE_PARTIAL` the fp32 `out`
    `(S, T, H, D)` and log2 `lse` `(S, T, H)` partials.
    """
    num_tokens, num_heads, head_dim = q.shape
    page, num_kv_heads = key_cache.shape[1:3]
    if (
        _FP8_STAGE
        and key_cache.dtype == torch.uint8
        and window == 0
        and rel_logits is not None
        and max_query_len * (num_heads // num_kv_heads) > 128
        and max_seq_len
    ):
        # key_cache / value_cache: K / V halves of one [blocks, PAGE, KV, 2D] cache.
        kv_bytes = torch.as_strided(
            key_cache,
            (key_cache.shape[0], page, num_kv_heads, 2 * head_dim),
            key_cache.stride(),
            key_cache.storage_offset(),
        )
        staged = _stage_fp8_kv(kv_bytes, block_table, cache_seqlens, max_seq_len)
        if staged is not None:
            scratch, block_table = staged
            key_cache, value_cache = scratch.split(head_dim, dim=-1)
    group = num_heads // num_kv_heads
    assert key_cache.stride() == value_cache.stride() and key_cache.stride(3) == 1
    assert q.stride(2) == 1
    assert mode != MODE_GATED or num_splits == 1
    draft = rel_logits is None
    extent = 0
    if rel_logits is not None:
        extent = rel_logits.shape[2] - 256
        assert rel_logits.stride(1) == rel_logits.shape[2]
        assert rel_logits.stride(2) == 1
    if mode == MODE_GATED and not draft:
        assert gate is not None and sinks is not None
        assert gate.stride(1) == sinks.stride(1) == 1
    # Unused pointers keep the launches typed alike.
    rel = q if rel_logits is None else rel_logits
    gate = q if gate is None else gate
    sinks = q if sinks is None else sinks
    if lse is None:
        lse = out
    block_m = _block_m(max_query_len, group)
    u8_wide = (
        key_cache.dtype == torch.uint8 and _FP8_BM > 0 and max_query_len * group > 128
    )
    if u8_wide:
        block_m = _FP8_BM
    pf_tune = max_query_len * group > 128 and not u8_wide
    if pf_tune and _PF_BM:
        block_m = _PF_BM
    tokens_per_tile = block_m // group
    num_reqs = cache_seqlens.shape[0]
    grid = (
        triton.cdiv(max_query_len, tokens_per_tile),
        num_kv_heads * num_splits,
        num_reqs,
    )
    if mode == MODE_OUT_LSE:
        out_strides = (out.stride(0), out.stride(1), lse.stride(0), lse.stride(1))
    elif mode == MODE_GATED:
        assert out.is_contiguous()
        out_strides = (out.stride(0), 0, 0, 0)
    else:
        assert out.is_contiguous() and lse.is_contiguous()
        out_strides = (0, 0, 0, 0)
    _rel_attention_kernel[grid](
        q,
        key_cache,
        value_cache,
        rel,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        out,
        lse,
        gate,
        sinks,
        softmax_scale,
        num_tokens,
        num_splits,
        q.stride(0),
        q.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        rel.stride(0),
        block_table.stride(0),
        *out_strides,
        gate.stride(0),
        sinks.stride(0),
        H=num_heads,
        G=group,
        D=head_dim,
        E=extent,
        PAGE=page,
        WINDOW=window,
        DRAFT=draft,
        MODE=mode,
        BLOCK_M=block_m,
        BLOCK_N=(
            _FP8_BN
            if u8_wide and _FP8_BN
            else _PF_BN
            if pf_tune and _PF_BN
            else 64
            if block_m >= 128
            else 128
            if block_m <= 32
            else 64
        ),
        num_warps=(
            _FP8_WARPS
            if u8_wide and _FP8_WARPS
            else _PF_WARPS
            if pf_tune and _PF_WARPS
            else 8
            if block_m >= 128
            else 4
        ),
        num_stages=_PF_STAGES if pf_tune and _PF_STAGES else 2,
    )
