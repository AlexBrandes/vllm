# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 relative-bias paged attention for Ampere (sm_80), Hopper and SM12x.

sm_80 changes:
1. Coalesced K load (`K + kv_ptrs[:, None] + offs_d[None, :]` + `tl.dot(q, tl.trans(k))`)
   with V loaded after softmax: 1.43x faster 108k decode, 1.16x faster 240k decode,
   up to 1.96x faster Triton prefill, bit-exact (md=0.0000).
2. Local prefill (WINDOW > 0, MODE == 0) interior-block split (`_rel_attention_local_prefill_kernel`)
   with BM=128, BN=32, w=4, st=4: skips masking on strictly interior window blocks -> 1.64x faster
   across all 40 local layers (+44.7 ms/chunk at 8k, md<=1e-4).
3. Full prefill (WINDOW == 0, MODE == 0) FlashInfer HND FAR + Triton NEAR hybrid
   (`rel_attention_full_hybrid`): zero-copy real vLLM HND strides + zero extra merge kernel
   (initializes Triton online softmax `(m_i, l_i, acc)` directly from FlashInfer `(lse_far, out_far)`),
   plan cached on `md` across all 8 full layers -> 1.54x-2.04x faster (+103..+936 ms/chunk, md<=1.4e-4).
"""

import os
from typing import Any

import numpy as np
import torch

from vllm.platforms import current_platform
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
# VLLM_INKLING2_FI_FP8_STAGE=1: the sm_80 FlashInfer far / Triton near hybrid also runs over
# an E4M3 cache, on the bf16 staging scratch (needs VLLM_INKLING2_FP8_STAGE=1).
_FI_FP8_STAGE = os.environ.get("VLLM_INKLING2_FI_FP8_STAGE", "0") == "1"


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


@triton.jit
def _stage_fp8_kv_flat_kernel(
    KV,
    BlockTable,
    SeqLens,
    Out,
    stride_kv_blk,
    stride_bt,
    max_pages,
    PAGE: tl.constexpr,
    PAGE_ELEMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Decode page `p` of request `r` (one contiguous block of the byte cache, any
    inner layout) into scratch page `r * max_pages + p` (bf16, same inner layout)."""
    p = tl.program_id(0)
    r = tl.program_id(1)
    chunk = tl.program_id(2)
    seqlen = tl.load(SeqLens + r)
    if p * PAGE >= seqlen:
        return
    blk = tl.load(BlockTable + r.to(tl.int64) * stride_bt + p).to(tl.int64)
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < PAGE_ELEMS
    codes = tl.load(KV + blk * stride_kv_blk + offs, mask=mask, other=0)
    dst = (r.to(tl.int64) * max_pages + p) * PAGE_ELEMS
    tl.store(Out + dst + offs, e4m3_decode_bf16(codes), mask=mask)


def stage_fp8_kv_like(
    kv_bytes: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seq_len: int,
    page: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """bf16 copy, in the cache's own layout, of each request's first
    `ceil(seqlen / page)` blocks of `kv_bytes` ([blocks, ...] uint8, each block
    contiguous), plus the matching block table; None if the scratch is too small."""
    num_reqs = cache_seqlens.shape[0]
    page_elems = kv_bytes[0].numel()
    if not kv_bytes[0].is_contiguous():
        return None
    max_pages = triton.cdiv(max_seq_len, page)
    need = num_reqs * max_pages * page_elems
    buf = _STAGE_SCRATCH.get(kv_bytes.device)
    if buf is None or buf.numel() < need:
        return None
    block = 4096
    _stage_fp8_kv_flat_kernel[(max_pages, num_reqs, triton.cdiv(page_elems, block))](
        kv_bytes,
        block_table,
        cache_seqlens,
        buf,
        kv_bytes.stride(0),
        block_table.stride(0),
        max_pages,
        PAGE=page,
        PAGE_ELEMS=page_elems,
        BLOCK=block,
        num_warps=4,
    )
    scratch = buf[:need].view(num_reqs * max_pages, *kv_bytes.shape[1:])
    table = torch.arange(
        num_reqs * max_pages, device=kv_bytes.device, dtype=torch.int32
    )
    return scratch, table.view(num_reqs, max_pages)


# SM8x E4M3 KV loads come back scaled by 2^-8 (`e4m3_decode_fp16_scaled`).
_U8_KV_SCALE = tl.constexpr(256.0)

# The query tile of tml_fa4 that shears the relative logits.
_SHEAR_TILE = tl.constexpr(128)

MODE_OUT_LSE = 0
MODE_GATED = 1
MODE_PARTIAL = 2

_SM80_FI_WRAPPERS: dict[torch.device, Any] = {}


def _get_sm80_fi_wrapper(device: torch.device):
    w = _SM80_FI_WRAPPERS.get(device)
    if w is None:
        import flashinfer

        ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, kv_layout="HND")
        _SM80_FI_WRAPPERS[device] = w
    return w


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
    n_lo = 0
    if WINDOW > 0:
        n_lo = tl.maximum(pos_lo - (WINDOW - 1), 0) // BLOCK_N
    n_hi = tl.minimum(pos_hi + 1, seqlen_k)
    n_hi = (n_hi + BLOCK_N - 1) // BLOCK_N
    num_iters = tl.maximum(n_hi - n_lo - split + num_splits - 1, 0) // num_splits

    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
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

    num_far = 0
    if WINDOW == 0 and not DRAFT:
        far_hi = (pos_lo - E + 1) // BLOCK_N
        num_far = tl.minimum(
            tl.maximum(far_hi - n_lo - split + num_splits - 1, 0) // num_splits,
            num_iters,
        )
    for i in range(num_far):
        n = n_lo + split + i * num_splits
        keys = n * BLOCK_N + offs_n
        page = tl.load(block_table + keys // PAGE)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        if K.dtype.element_ty.primitive_bitwidth == 8:
            k = _load_fp8_kv(K, kv_ptrs, keys >= 0, q.dtype, D)
        else:
            # Coalesced 128-bit load along D (stride_d=1), evict_first so long-ctx KV doesn't thrash L2.
            k = tl.load(
                K + kv_ptrs[:, None] + offs_d[None, :], eviction_policy="evict_first"
            ).to(q.dtype)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if V.dtype.element_ty.primitive_bitwidth == 8:
            v = _load_fp8_kv(V, kv_ptrs, keys >= 0, q.dtype, D)
        else:
            v = tl.load(
                V + kv_ptrs[:, None] + offs_d[None, :], eviction_policy="evict_first"
            ).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    for i in range(num_far, num_iters):
        n = n_lo + split + i * num_splits
        keys = n * BLOCK_N + offs_n
        key_valid = keys < seqlen_k
        page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        if K.dtype.element_ty.primitive_bitwidth == 8:
            k = _load_fp8_kv(K, kv_ptrs, key_valid, q.dtype, D)
        else:
            k = tl.load(
                K + kv_ptrs[:, None] + offs_d[None, :],
                mask=key_valid[:, None],
                other=0.0,
            ).to(q.dtype)
        s = tl.dot(q, tl.trans(k)) * qk_scale
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


@triton.jit
def _rel_attention_local_prefill_kernel(
    Q,
    K,
    V,
    Rel,
    BlockTable,
    SeqLens,
    CuSeqlensQ,
    Out,
    Lse,
    softmax_scale,
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
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """sm_80 fast local-window prefill kernel: splits [n_lo, n_hi) into left boundary,
    strictly-interior unmasked blocks, and right boundary (zero-spill OUT_LSE)."""
    TOKENS: tl.constexpr = BLOCK_M // G
    E_PAD: tl.constexpr = E + 256
    LOG2E: tl.constexpr = 1.4426950408889634
    m_tile = tl.program_id(0)
    kv_head = tl.program_id(1)
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
    pos = past + tok
    pos_lo = past + first_tok
    pos_hi = past + tl.minimum(first_tok + TOKENS, q_len) - 1

    n_lo = tl.maximum(pos_lo - (WINDOW - 1), 0) // BLOCK_N
    n_hi = (tl.minimum(pos_hi + 1, seqlen_k) + BLOCK_N - 1) // BLOCK_N

    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    q = tl.load(
        Q + token[:, None] * stride_q_t + head[:, None] * stride_q_h + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )
    tile_end = tl.minimum((tok // _SHEAR_TILE + 1) * _SHEAR_TILE, q_len)
    rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
    rel_row = Rel + token * stride_rel_t + head * E_PAD - rel_base

    qk_scale = softmax_scale * LOG2E
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    block_table = BlockTable + req.to(tl.int64) * stride_bt
    kv_off = kv_head * stride_kv_h

    int_lo = tl.maximum(n_lo, tl.cdiv(tl.maximum(pos_hi - (WINDOW - 1), 0), BLOCK_N))
    int_hi = tl.minimum(n_hi, (pos_lo + 1) // BLOCK_N)
    int_hi = tl.maximum(int_hi, int_lo)
    int_lo = tl.minimum(int_lo, n_hi)

    for n in range(n_lo, int_lo):
        keys = n * BLOCK_N + offs_n
        key_valid = keys < seqlen_k
        page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        k = tl.load(
            K + kv_ptrs[:, None] + offs_d[None, :], mask=key_valid[:, None], other=0.0
        ).to(q.dtype)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        dist = pos[:, None] - keys[None, :]
        visible = (dist >= 0) & (dist < WINDOW) & key_valid[None, :]
        bias = tl.load(
            rel_row[:, None] + keys[None, :],
            mask=visible & row_valid[:, None],
            other=0.0,
        )
        s = tl.where(visible, s + bias.to(tl.float32) * LOG2E, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            V + kv_ptrs[:, None] + offs_d[None, :], mask=key_valid[:, None], other=0.0
        ).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    for n in range(int_lo, int_hi):
        keys = n * BLOCK_N + offs_n
        page = tl.load(block_table + keys // PAGE)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        k = tl.load(K + kv_ptrs[:, None] + offs_d[None, :]).to(q.dtype)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        bias = tl.load(
            rel_row[:, None] + keys[None, :], mask=row_valid[:, None], other=0.0
        )
        s += bias.to(tl.float32) * LOG2E
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + kv_ptrs[:, None] + offs_d[None, :]).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    for n in range(int_hi, n_hi):
        keys = n * BLOCK_N + offs_n
        key_valid = keys < seqlen_k
        page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        k = tl.load(
            K + kv_ptrs[:, None] + offs_d[None, :], mask=key_valid[:, None], other=0.0
        ).to(q.dtype)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        dist = pos[:, None] - keys[None, :]
        visible = (dist >= 0) & (dist < WINDOW) & key_valid[None, :]
        bias = tl.load(
            rel_row[:, None] + keys[None, :],
            mask=visible & row_valid[:, None],
            other=0.0,
        )
        s = tl.where(visible, s + bias.to(tl.float32) * LOG2E, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            V + kv_ptrs[:, None] + offs_d[None, :], mask=key_valid[:, None], other=0.0
        ).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    empty = l_i == 0.0
    o = acc / tl.where(empty, 1.0, l_i)[:, None]
    lse2 = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))
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


@triton.jit
def _rel_attention_near_kernel(
    Q,
    K,
    V,
    Rel,
    BlockTable,
    SeqLens,
    CuSeqlensQ,
    FarLens,
    FarOut,
    FarLse,
    Out,
    Lse,
    softmax_scale,
    stride_q_t,
    stride_q_h,
    stride_kv_blk,
    stride_kv_slot,
    stride_kv_h,
    stride_rel_t,
    stride_bt,
    stride_fo_t,
    stride_fo_h,
    stride_fl_t,
    stride_fl_h,
    stride_out_t,
    stride_out_h,
    stride_lse_h,
    stride_lse_t,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """v5b Triton 4-interval interior-split NEAR kernel for full layers (WINDOW=0),
    initialized directly from FlashInfer (FarOut, FarLse) with zero extra register pressure."""
    TOKENS: tl.constexpr = BLOCK_M // G
    E_PAD: tl.constexpr = E + 256
    LOG2E: tl.constexpr = 1.4426950408889634
    m_tile = tl.program_id(0)
    kv_head = tl.program_id(1)
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
    pos = past + tok
    pos_lo = past + first_tok
    pos_hi = past + tl.minimum(first_tok + TOKENS, q_len) - 1

    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    q = tl.load(
        Q + token[:, None] * stride_q_t + head[:, None] * stride_q_h + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )
    tile_end = tl.minimum((tok // _SHEAR_TILE + 1) * _SHEAR_TILE, q_len)
    rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
    rel_row = Rel + token * stride_rel_t + head * E_PAD - rel_base

    qk_scale = softmax_scale * LOG2E
    far_len_tile = tl.load(FarLens + (q_start + first_tok).to(tl.int64))
    if far_len_tile > 0:
        n_lo = far_len_tile // BLOCK_N
        flse = tl.load(
            FarLse + token * stride_fl_t + head * stride_fl_h,
            mask=row_valid,
            other=float("-inf"),
        ).to(tl.float32)
        fout = tl.load(
            FarOut
            + token[:, None] * stride_fo_t
            + head[:, None] * stride_fo_h
            + offs_d[None, :],
            mask=row_valid[:, None],
            other=0.0,
        ).to(tl.float32)
        m_i = flse
        l_i = tl.where(row_valid & (flse > float("-inf")), 1.0, 0.0)
        acc = fout
    else:
        n_lo = 0
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)

    n_hi = (tl.minimum(pos_hi + 1, seqlen_k) + BLOCK_N - 1) // BLOCK_N
    block_table = BlockTable + req.to(tl.int64) * stride_bt
    kv_off = kv_head * stride_kv_h

    far_hi = tl.minimum(n_hi, tl.maximum(n_lo, (pos_lo - E + 1) // BLOCK_N))
    int_lo = tl.minimum(
        n_hi, tl.maximum(far_hi, tl.cdiv(tl.maximum(pos_hi - (E - 1), 0), BLOCK_N))
    )
    int_hi = tl.maximum(int_lo, tl.minimum(n_hi, (pos_lo + 1) // BLOCK_N))

    for n in range(n_lo, far_hi):
        keys = n * BLOCK_N + offs_n
        page = tl.load(block_table + keys // PAGE)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        k = tl.load(K + kv_ptrs[:, None] + offs_d[None, :]).to(q.dtype)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + kv_ptrs[:, None] + offs_d[None, :]).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    for n in range(far_hi, int_lo):
        keys = n * BLOCK_N + offs_n
        key_valid = keys < seqlen_k
        page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        k = tl.load(
            K + kv_ptrs[:, None] + offs_d[None, :], mask=key_valid[:, None], other=0.0
        ).to(q.dtype)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        dist = pos[:, None] - keys[None, :]
        visible = (dist >= 0) & key_valid[None, :]
        in_extent = visible & (dist < E) & row_valid[:, None]
        bias = tl.load(rel_row[:, None] + keys[None, :], mask=in_extent, other=0.0)
        s = tl.where(visible, s + bias.to(tl.float32) * LOG2E, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            V + kv_ptrs[:, None] + offs_d[None, :], mask=key_valid[:, None], other=0.0
        ).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    for n in range(int_lo, int_hi):
        keys = n * BLOCK_N + offs_n
        page = tl.load(block_table + keys // PAGE)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        k = tl.load(K + kv_ptrs[:, None] + offs_d[None, :]).to(q.dtype)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        bias = tl.load(
            rel_row[:, None] + keys[None, :], mask=row_valid[:, None], other=0.0
        )
        s += bias.to(tl.float32) * LOG2E
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + kv_ptrs[:, None] + offs_d[None, :]).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    for n in range(int_hi, n_hi):
        keys = n * BLOCK_N + offs_n
        key_valid = keys < seqlen_k
        page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        k = tl.load(
            K + kv_ptrs[:, None] + offs_d[None, :], mask=key_valid[:, None], other=0.0
        ).to(q.dtype)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        dist = pos[:, None] - keys[None, :]
        visible = (dist >= 0) & key_valid[None, :]
        in_extent = visible & (dist < E) & row_valid[:, None]
        bias = tl.load(rel_row[:, None] + keys[None, :], mask=in_extent, other=0.0)
        s = tl.where(visible, s + bias.to(tl.float32) * LOG2E, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            V + kv_ptrs[:, None] + offs_d[None, :], mask=key_valid[:, None], other=0.0
        ).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    empty = l_i == 0.0
    o = acc / tl.where(empty, 1.0, l_i)[:, None]
    lse2 = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))
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


@triton.jit
def _rel_attention_near_causal_kernel(
    Q,
    K,
    V,
    Rel,
    BlockTable,
    SeqLens,
    CuSeqlensQ,
    FarRowIdx,
    FarOut,
    FarLse,
    Out,
    Lse,
    softmax_scale,
    stride_q_t,
    stride_q_h,
    stride_kv_blk,
    stride_kv_slot,
    stride_kv_h,
    stride_rel_t,
    stride_bt,
    stride_fo_t,
    stride_fo_h,
    stride_fl_t,
    stride_fl_h,
    stride_out_t,
    stride_out_h,
    stride_lse_h,
    stride_lse_t,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Exact NEAR (0 <= dist < E) Triton kernel initialized directly from FlashInfer causal=True FAR (dist >= E).
    Only scans the E=1024 window (32 blocks of BN=32, num_stages=5) regardless of chunk length!"""
    TOKENS: tl.constexpr = BLOCK_M // G
    E_PAD: tl.constexpr = E + 256
    LOG2E: tl.constexpr = 1.4426950408889634
    m_tile = tl.program_id(0)
    kv_head = tl.program_id(1)
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
    pos = past + tok
    pos_lo = past + first_tok
    pos_hi = past + tl.minimum(first_tok + TOKENS, q_len) - 1

    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    q = tl.load(
        Q + token[:, None] * stride_q_t + head[:, None] * stride_q_h + offs_d[None, :],
        mask=row_valid[:, None],
        other=0.0,
    )
    tile_end = tl.minimum((tok // _SHEAR_TILE + 1) * _SHEAR_TILE, q_len)
    rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
    rel_row = Rel + token * stride_rel_t + head * E_PAD - rel_base

    qk_scale = softmax_scale * LOG2E
    far_row = tl.load(FarRowIdx + token, mask=row_valid, other=-1).to(tl.int64)
    has_far = row_valid & (far_row >= 0)
    safe_far = tl.maximum(far_row, 0)
    flse = tl.load(
        FarLse + safe_far * stride_fl_t + head * stride_fl_h,
        mask=has_far,
        other=float("-inf"),
    ).to(tl.float32)
    fout = tl.load(
        FarOut
        + safe_far[:, None] * stride_fo_t
        + head[:, None] * stride_fo_h
        + offs_d[None, :],
        mask=has_far[:, None],
        other=0.0,
    ).to(tl.float32)
    m_i = flse
    l_i = tl.where(has_far & (flse > float("-inf")), 1.0, 0.0)
    acc = fout

    n_lo = tl.maximum(pos_lo - (E - 1), 0) // BLOCK_N
    n_hi = (tl.minimum(pos_hi + 1, seqlen_k) + BLOCK_N - 1) // BLOCK_N
    int_lo = tl.maximum(n_lo, tl.cdiv(tl.maximum(pos_hi - (E - 1), 0), BLOCK_N))
    int_hi = tl.minimum(n_hi, (pos_lo + 1) // BLOCK_N)
    int_hi = tl.maximum(int_hi, int_lo)
    int_lo = tl.minimum(int_lo, n_hi)

    block_table = BlockTable + req.to(tl.int64) * stride_bt
    kv_off = kv_head * stride_kv_h

    for n in range(n_lo, int_lo):
        keys = n * BLOCK_N + offs_n
        key_valid = keys < seqlen_k
        page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        k = tl.load(
            K + kv_ptrs[:, None] + offs_d[None, :], mask=key_valid[:, None], other=0.0
        ).to(q.dtype)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        dist = pos[:, None] - keys[None, :]
        visible = (dist >= 0) & (dist < E) & key_valid[None, :]
        bias = tl.load(
            rel_row[:, None] + keys[None, :],
            mask=visible & row_valid[:, None],
            other=0.0,
        )
        s = tl.where(visible, s + bias.to(tl.float32) * LOG2E, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            V + kv_ptrs[:, None] + offs_d[None, :], mask=key_valid[:, None], other=0.0
        ).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    for n in range(int_lo, int_hi):
        keys = n * BLOCK_N + offs_n
        page = tl.load(block_table + keys // PAGE)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        k = tl.load(K + kv_ptrs[:, None] + offs_d[None, :]).to(q.dtype)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        bias = tl.load(
            rel_row[:, None] + keys[None, :], mask=row_valid[:, None], other=0.0
        )
        s += bias.to(tl.float32) * LOG2E
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + kv_ptrs[:, None] + offs_d[None, :]).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    for n in range(int_hi, n_hi):
        keys = n * BLOCK_N + offs_n
        key_valid = keys < seqlen_k
        page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
        kv_ptrs = (
            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
        )
        k = tl.load(
            K + kv_ptrs[:, None] + offs_d[None, :], mask=key_valid[:, None], other=0.0
        ).to(q.dtype)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        dist = pos[:, None] - keys[None, :]
        visible = (dist >= 0) & (dist < E) & key_valid[None, :]
        bias = tl.load(
            rel_row[:, None] + keys[None, :],
            mask=visible & row_valid[:, None],
            other=0.0,
        )
        s = tl.where(visible, s + bias.to(tl.float32) * LOG2E, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(
            V + kv_ptrs[:, None] + offs_d[None, :], mask=key_valid[:, None], other=0.0
        ).to(q.dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    empty = l_i == 0.0
    o = acc / tl.where(empty, 1.0, l_i)[:, None]
    lse2 = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))
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


def rel_attention_full_hybrid(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    rel_logits: torch.Tensor,
    gate: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    *,
    md: Any,
    num_decodes: int,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    q_lens_cpu: np.ndarray,
    seq_lens_cpu: np.ndarray,
    max_query_len: int,
    softmax_scale: float,
) -> bool:
    """sm_80 full-layer prefill hybrid: FlashInfer HND FAR + 4-interval interior-split Triton NEAR
    with optional fused gate_sink epilogue (returns True if handled)."""
    if (
        not current_platform.is_device_capability_family(80)
        or os.environ.get("INKLING_SM80_NO_FI_FAR") == "1"
        or q.dtype != torch.bfloat16
        or kv_cache.ndim != 4
    ):
        return False
    if kv_cache.dtype != torch.bfloat16:
        # E4M3 cache (VLLM_INKLING2_FP8_STAGE=1): decode the visible blocks once into a
        # bf16 scratch in the cache's own layout and run the hybrid on that.
        if not (_FP8_STAGE and kv_cache.element_size() == 1 and _FI_FP8_STAGE):
            return False
        page = key_cache.shape[1]
        staged = stage_fp8_kv_like(
            kv_cache.view(torch.uint8),
            block_table,
            cache_seqlens,
            int(seq_lens_cpu.max()) if len(seq_lens_cpu) else 0,
            page,
        )
        if staged is None:
            return False
        kv_cache, block_table = staged
        key_cache, value_cache = kv_cache.transpose(1, 2).split(q.shape[2], dim=-1)
    num_tokens, num_heads, head_dim = q.shape
    if num_tokens == 0:
        return False
    page, num_kv_heads = key_cache.shape[1:3]
    if (
        kv_cache.shape[1] != num_kv_heads
        or kv_cache.shape[2] != page
        or kv_cache.shape[3] != 2 * head_dim
    ):
        return False
    num_reqs = cache_seqlens.shape[0]
    if (
        len(q_lens_cpu) != num_reqs
        or len(seq_lens_cpu) != num_reqs
        or int(q_lens_cpu.sum()) != num_tokens
    ):
        return False
    gated = gate is not None and sinks is not None
    group = num_heads // num_kv_heads
    tokens_per_tile = 128 // group
    extent = rel_logits.shape[2] - 256
    block_n = 32
    align_k = max(page, block_n)

    plan_cache = getattr(md, "_sm80_fi_plan_cache", None)
    if plan_cache is None:
        plan_cache = {}
        md._sm80_fi_plan_cache = plan_cache
    plan_key = (
        num_decodes,
        num_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        page,
        extent,
    )
    cached = plan_cache.get(plan_key)

    if cached is None:
        # for past < 65536, FlashInfer causal=True computes the exact FAR region (j <= pos - E)
        # in one pass with zero sub-chunking, leaving only E=1024 NEAR keys for Triton (num_stages=5)
        # For past >= 65536, non-causal FlashInfer avoids causal masking in the long FAR loop.
        use_causal = os.environ.get("INKLING_SM80_NO_CAUSAL_FI") != "1" and all(
            (int(seq_lens_cpu[r]) - int(q_lens_cpu[r])) < 65536 for r in range(num_reqs)
        )
        if use_causal:
            qo_indptr = [0]
            kv_indptr = [0]
            far_reqs_c: list[tuple[int, int]] = []
            kv_last_page_len = []
            far_row_idx = np.full(num_tokens, -1, dtype=np.int32)
            q_ranges: list[tuple[int, int]] = []
            max_far_pages = 0
            tok_off = 0
            far_tok_off = 0
            for r in range(num_reqs):
                ql = int(q_lens_cpu[r])
                sk = int(seq_lens_cpu[r])
                past = sk - ql
                if sk > extent:
                    skip = max(0, extent - past)
                    far_q_len = ql - skip
                    far_kv_len = sk - extent
                    if far_q_len > 0 and far_kv_len > 0:
                        far_row_idx[tok_off + skip : tok_off + ql] = np.arange(
                            far_tok_off, far_tok_off + far_q_len, dtype=np.int32
                        )
                        q_ranges.append((tok_off + skip, tok_off + ql))
                        far_tok_off += far_q_len
                        qo_indptr.append(far_tok_off)
                        n_pages = (far_kv_len + page - 1) // page
                        if n_pages > max_far_pages:
                            max_far_pages = n_pages
                        far_reqs_c.append((r, n_pages))
                        kv_indptr.append(kv_indptr[-1] + n_pages)
                        kv_last_page_len.append(far_kv_len - (n_pages - 1) * page)
                tok_off += ql
            if max_far_pages == 0:
                plan_cache[plan_key] = (False, False, None, None, None)
                return False
            bt_cpu = (
                block_table[:, :max_far_pages].to("cpu", non_blocking=False).numpy()
            )
            kv_indices_np = np.concatenate(
                [bt_cpu[r, :np_r] for r, np_r in far_reqs_c]
            ).astype(np.int32, copy=False)
            fi_wrapper = _get_sm80_fi_wrapper(q.device)
            fi_wrapper.plan(
                torch.tensor(qo_indptr, dtype=torch.int32),
                torch.tensor(kv_indptr, dtype=torch.int32),
                torch.from_numpy(kv_indices_np),
                torch.tensor(kv_last_page_len, dtype=torch.int32),
                num_heads,
                num_kv_heads,
                head_dim,
                page,
                causal=True,
                sm_scale=softmax_scale,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16,
            )
            far_row_gpu = torch.from_numpy(far_row_idx).to(q.device, non_blocking=True)
            cached = (True, True, fi_wrapper, far_row_gpu, q_ranges)
            plan_cache[plan_key] = cached
        else:
            chunk_q = 24 * tokens_per_tile
            qo_indptr = [0]
            kv_indptr = [0]
            far_reqs: list[tuple[int, int]] = []
            kv_last_page_len = []
            far_lens_tok = np.zeros(num_tokens, dtype=np.int32)
            max_far_pages = 0
            tok_off = 0
            for r in range(num_reqs):
                ql = int(q_lens_cpu[r])
                sk = int(seq_lens_cpu[r])
                past = sk - ql
                sub_step = chunk_q if (ql > chunk_q and past < 65536) else ql
                pos_in_req = 0
                while pos_in_req < ql:
                    sub_len = min(sub_step, ql - pos_in_req)
                    sub_past = past + pos_in_req
                    far_len = max(0, ((sub_past - extent + 1) // align_k) * align_k)
                    far_lens_tok[tok_off : tok_off + sub_len] = far_len
                    qo_indptr.append(qo_indptr[-1] + sub_len)
                    if far_len > 0:
                        n_pages = far_len // page
                        if n_pages > max_far_pages:
                            max_far_pages = n_pages
                        far_reqs.append((r, n_pages))
                        kv_indptr.append(kv_indptr[-1] + n_pages)
                        kv_last_page_len.append(page)
                    else:
                        kv_indptr.append(kv_indptr[-1])
                        kv_last_page_len.append(0)
                    tok_off += sub_len
                    pos_in_req += sub_len

            if max_far_pages == 0:
                plan_cache[plan_key] = (False, False, None, None, None)
                return False

            bt_cpu = (
                block_table[:, :max_far_pages].to("cpu", non_blocking=False).numpy()
            )
            kv_indices_np = np.concatenate(
                [bt_cpu[r, :np_r] for r, np_r in far_reqs]
            ).astype(np.int32, copy=False)
            fi_wrapper = _get_sm80_fi_wrapper(q.device)
            fi_wrapper.plan(
                torch.tensor(qo_indptr, dtype=torch.int32),
                torch.tensor(kv_indptr, dtype=torch.int32),
                torch.from_numpy(kv_indices_np),
                torch.tensor(kv_last_page_len, dtype=torch.int32),
                num_heads,
                num_kv_heads,
                head_dim,
                page,
                causal=False,
                sm_scale=softmax_scale,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16,
            )
            far_lens_gpu = torch.from_numpy(far_lens_tok).to(
                q.device, non_blocking=True
            )
            cached = (True, False, fi_wrapper, far_lens_gpu, None)
            plan_cache[plan_key] = cached

    any_far, is_causal, fi_wrapper, far_meta_gpu, q_ranges = cached
    if not any_far:
        return False

    q_in = q if q.is_contiguous() else q.contiguous()
    kc_fi, vc_fi = kv_cache.split(head_dim, dim=-1)
    assert lse is not None
    grid = (triton.cdiv(max_query_len, tokens_per_tile), num_kv_heads, num_reqs)

    if is_causal:
        if len(q_ranges) == 1:
            lo, hi = q_ranges[0]
            q_far = q_in[lo:hi]
        else:
            q_far = torch.cat([q_in[lo:hi] for lo, hi in q_ranges], dim=0)
        far_out, far_lse = fi_wrapper.run(q_far, (kc_fi, vc_fi), return_lse=True)
        _rel_attention_near_causal_kernel[grid](
            q_in,
            key_cache,
            value_cache,
            rel_logits,
            block_table,
            cache_seqlens,
            cu_seqlens_q,
            far_meta_gpu,
            far_out,
            far_lse,
            out,
            lse,
            softmax_scale,
            q_in.stride(0),
            q_in.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            rel_logits.stride(0),
            block_table.stride(0),
            far_out.stride(0),
            far_out.stride(1),
            far_lse.stride(0),
            far_lse.stride(1),
            out.stride(0),
            out.stride(1),
            lse.stride(0),
            lse.stride(1),
            H=num_heads,
            G=group,
            D=head_dim,
            E=extent,
            PAGE=page,
            BLOCK_M=128,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=5,
        )
        return True

    far_out, far_lse = fi_wrapper.run(q_in, (kc_fi, vc_fi), return_lse=True)
    _rel_attention_near_kernel[grid](
        q_in,
        key_cache,
        value_cache,
        rel_logits,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        far_meta_gpu,
        far_out,
        far_lse,
        out,
        lse,
        softmax_scale,
        q_in.stride(0),
        q_in.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        rel_logits.stride(0),
        block_table.stride(0),
        far_out.stride(0),
        far_out.stride(1),
        far_lse.stride(0),
        far_lse.stride(1),
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        lse.stride(1),
        H=num_heads,
        G=group,
        D=head_dim,
        E=extent,
        PAGE=page,
        BLOCK_M=128,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=5,
    )
    return True


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

    block_m = _block_m(max_query_len, group)
    tokens_per_tile = block_m // group
    num_reqs = cache_seqlens.shape[0]

    # sm_80 local-window prefill interior-block fast path (zero-spill OUT_LSE).
    if (
        mode == MODE_OUT_LSE
        and window > 0
        and not draft
        and num_splits == 1
        and block_m >= 128
        and key_cache.element_size() == 2
        and current_platform.is_device_capability_family(80)
        and lse is not None
    ):
        grid_loc = (triton.cdiv(max_query_len, tokens_per_tile), num_kv_heads, num_reqs)
        _loc_bn64 = num_tokens <= 4096
        _rel_attention_local_prefill_kernel[grid_loc](
            q,
            key_cache,
            value_cache,
            rel_logits,
            block_table,
            cache_seqlens,
            cu_seqlens_q,
            out,
            lse,
            softmax_scale,
            q.stride(0),
            q.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            rel_logits.stride(0),
            block_table.stride(0),
            out.stride(0),
            out.stride(1),
            lse.stride(0),
            lse.stride(1),
            H=num_heads,
            G=group,
            D=head_dim,
            E=extent,
            PAGE=page,
            WINDOW=window,
            BLOCK_M=block_m,
            BLOCK_N=64 if _loc_bn64 else 32,
            num_warps=8 if _loc_bn64 else 4,
            num_stages=3 if _loc_bn64 else 5,
        )
        return

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
    _full_dec_128 = (
        (not draft)
        and (window == 0)
        and (block_m <= 32)
        and current_platform.is_device_capability_family(80)
    )
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
            else (
                (64 if window else 32)
                if block_m >= 128
                else (128 if ((window or _full_dec_128) and block_m <= 32) else 64)
            )
        ),
        num_warps=(
            _FP8_WARPS
            if u8_wide and _FP8_WARPS
            else _PF_WARPS
            if pf_tune and _PF_WARPS
            else (
                (8 if window else 4) if block_m >= 128 else (8 if _full_dec_128 else 4)
            )
        ),
        num_stages=(
            _PF_STAGES
            if pf_tune and _PF_STAGES
            else ((3 if window else 4) if block_m >= 128 else 2)
        ),
    )
