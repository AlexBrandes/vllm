# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# From HyperQwen (https://github.com/syv-ai/HyperQwen, patches/spec-decode-attn.patch),
# Apache-2.0; ported to this engine with a GA100 split policy (VLLM_SPEC_ATTN_POLICY=ga100).
"""Split-KV paged attention for speculative-decode batches (a few query tokens per
request, GQA), Triton.

Neither vLLM's FlashAttention-2 path nor its Triton unified attention split the KV
sequence across SMs when a request has more than one query token: with MTP k=4
(5 queries) on a 24-head model that leaves 24 (FA) or ~8 (Triton) thread blocks on an
82-SM RTX 3090 and the attention layer takes ~57 us for a 1.5k-token context.
This kernel gives every (request, kv-head, query tile) NUM_SEGMENTS blocks, each
computing an online-softmax partial over its slice of the KV cache for all its query
rows at once (G = query heads per kv head), followed by a tiny combine kernel.

Long blocks: the query rows of one request (q_len x G of them) are split into tiles of
BLOCK_M rows, so the kernel is not capped at BLOCK_M // G query tokens. That cap is what
made lookup-augmented drafting unaffordable past a block of 7 -- a 16-token verify fell
back to FA2 and doubled the step at 25k context. Each tile re-reads the KV segment, so
BLOCK_M is chosen to keep the tile count at 1 where the register budget allows.

Layout matches vLLM's FLASH_ATTN backend: q [T, Hq, D], key/value cache
[num_blocks, block_size, Hkv, D] (block_size any multiple of 16), block_table
[num_reqs, max_blocks], seqused_k [num_reqs] = kv length including the new tokens,
cu_seqlens_q [num_reqs + 1]. Query token i of a request sits at kv position
seqused_k - q_len + i and attends causally.

int8 KV (QUANT=1): the same kernel reads TRITON_ATTN's `int8_per_token_head` cache. That
backend hands out K/V views of shape [num_blocks, block_size, Hkv, D + 4] -- the head dim
is padded so one float32 scale per (token, head) sits inline after the data -- plus f32
scale views [num_blocks, block_size, Hkv]. Only the element type and the scale multiply
change here: the padded head dim is already carried by stride_kh, so the addressing is
untouched. Both scales are per (token, head), i.e. constant along D, so folding them in
after the dot is exact rather than an approximation:
  s   = (q . k_int8) * k_scale        instead of  q . (k_int8 * k_scale)
  acc = (p * v_scale) . v_int8        instead of  p . (v_int8 * v_scale)
int8 -> bf16 is itself exact (bf16 has 8 mantissa bits, int8 needs 7), so the only new
rounding is the one bf16 multiply the stock Triton kernel also does.

Halving the bytes matters because this kernel is bandwidth-bound at long context: at a
128k context and 8 query tokens it reads 524 MB of bf16 KV in 1,198 us, which is 437 GB/s
of the 3090's ~936.

Restrictions: q_len <= QMAX_TOKENS per request, D a power of two <= 256, no sliding
window / softcap / alibi.
"""

import os

import torch
import triton
import triton.language as tl

NUM_SEGMENTS = 16
# CMP 170HX / GA100 launch policy (VLLM_SPEC_ATTN_POLICY=ga100; off by default).
# The stock grid gives one request 16 segments x Hkv CTAs (64 four-warp CTAs for
# Qwen3.8-27B's 4 KV heads on 74 SMs) with no load pipelining, which reads long
# contexts at ~600-730 GB/s of the card's ~1.6 TB/s. With the policy a
# one-request call uses 32 segments, and every call pipelines its KV loads
# (num_stages 2, 32-token tiles at BLOCK_M <= 32; GA100 has 164 KB of shared
# memory against sm86's 99 KB). The segment count depends only on the request
# count, so each captured CUDA graph keeps a fixed grid.
_POLICY_GA100 = os.environ.get("VLLM_SPEC_ATTN_POLICY", "") == "ga100"
NSEG_ONE_REQUEST = 32
BLOCK_M = 64  # query rows (q_len * G) per program at the default register budget
BLOCK_M_BIG = 128  # ... and with 8 warps, which keeps a 16-token block in one tile
QMAX_TOKENS = 64  # query tokens per request the caller may ask for


@triton.jit
def _spec_attn_partial(
    q_ptr,
    k_ptr,
    v_ptr,
    bt_ptr,
    seqused_ptr,
    cu_q_ptr,
    part_o_ptr,
    part_m_ptr,
    part_l_ptr,
    ks_ptr,
    vs_ptr,
    scale,
    stride_qt,
    stride_qh,
    stride_kb,
    stride_ks,
    stride_kh,
    stride_vb,
    stride_vs,
    stride_vh,
    stride_bt,
    stride_ksb,
    stride_kss,
    stride_ksh,
    stride_vsb,
    stride_vss,
    stride_vsh,
    G: tl.constexpr,
    Hq: tl.constexpr,
    QMAX: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE: tl.constexpr,
    NSEG: tl.constexpr,
    QT: tl.constexpr,
    NTILE: tl.constexpr,
    QUANT: tl.constexpr,
):
    pid = tl.program_id(0)
    req = pid // NTILE
    qtile = pid % NTILE
    kvh = tl.program_id(1)
    seg = tl.program_id(2)

    q_start = tl.load(cu_q_ptr + req)
    q_len = tl.load(cu_q_ptr + req + 1) - q_start
    kv_len = tl.load(seqused_ptr + req)

    # rows: r = i * G + g  -> query token qtile * QT + i (0..q_len-1), head kvh*G + g
    r = tl.arange(0, BLOCK_M)
    ri = qtile * QT + r // G
    rg = r % G
    row_ok = (r < QT * G) & (ri < q_len)
    q_pos = kv_len - q_len + ri  # kv position of each query row
    d = tl.arange(0, D)
    q_ptrs = (
        q_ptr
        + (q_start + ri)[:, None] * stride_qt
        + (kvh * G + rg)[:, None] * stride_qh
        + d[None, :]
    )
    q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)

    # this segment's key range
    tiles_total = (kv_len + TILE - 1) // TILE
    tiles_per_seg = (tiles_total + NSEG - 1) // NSEG
    t0 = seg * tiles_per_seg
    t1 = tl.minimum(t0 + tiles_per_seg, tiles_total)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    qs = (q * scale).to(tl.bfloat16)

    for t in range(t0, t1):
        pos = t * TILE + tl.arange(0, TILE)
        k_ok = pos < kv_len
        blk = tl.load(
            bt_ptr + req * stride_bt + pos // BLOCK_SIZE, mask=k_ok, other=0
        ).to(tl.int64)
        slot = pos % BLOCK_SIZE
        k_ptrs = (
            k_ptr
            + blk[:, None] * stride_kb
            + slot[:, None] * stride_ks
            + kvh * stride_kh
            + d[None, :]
        )
        v_ptrs = (
            v_ptr
            + blk[:, None] * stride_vb
            + slot[:, None] * stride_vs
            + kvh * stride_vh
            + d[None, :]
        )
        k = tl.load(k_ptrs, mask=k_ok[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=k_ok[:, None], other=0.0)
        if QUANT:
            # int8 data, one f32 scale per (token, head); both constant along D.
            k = k.to(tl.bfloat16)
            v = v.to(tl.bfloat16)
            k_sc = tl.load(
                ks_ptr + blk * stride_ksb + slot * stride_kss + kvh * stride_ksh,
                mask=k_ok,
                other=0.0,
            )
            v_sc = tl.load(
                vs_ptr + blk * stride_vsb + slot * stride_vss + kvh * stride_vsh,
                mask=k_ok,
                other=0.0,
            )
            s = tl.dot(qs, tl.trans(k)).to(tl.float32) * k_sc[None, :]
        else:
            s = tl.dot(qs, tl.trans(k)).to(tl.float32)  # [BLOCK_M, TILE]
        allowed = k_ok[None, :] & (pos[None, :] <= q_pos[:, None]) & row_ok[:, None]
        s = tl.where(allowed, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(tl.where(m_i == float("-inf"), float("-inf"), m_i - m_safe))
        l_i = l_i * alpha + tl.sum(p, 1)
        if QUANT:
            acc = acc * alpha[:, None] + tl.dot(
                (p * v_sc[None, :]).to(tl.bfloat16), v
            ).to(tl.float32)
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        m_i = m_new

    # store partials at flat index ((req*Hq + head)*QMAX + i)*NSEG + seg
    hrow = kvh * G + rg
    pidx = ((req * Hq + hrow) * QMAX + ri) * NSEG + seg
    tl.store(part_o_ptr + pidx[:, None] * D + d[None, :], acc, mask=row_ok[:, None])
    tl.store(part_m_ptr + pidx, m_i, mask=row_ok)
    tl.store(part_l_ptr + pidx, l_i, mask=row_ok)


@triton.jit
def _spec_attn_combine(
    part_o_ptr,
    part_m_ptr,
    part_l_ptr,
    out_ptr,
    cu_q_ptr,
    stride_ot,
    stride_oh,
    Hq: tl.constexpr,
    QMAX: tl.constexpr,
    D: tl.constexpr,
    NSEG: tl.constexpr,
):
    req = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + req)
    q_len = tl.load(cu_q_ptr + req + 1) - q_start
    if i < q_len:
        base = ((req * Hq + h) * QMAX + i) * NSEG
        segs = tl.arange(0, NSEG)
        m = tl.load(part_m_ptr + base + segs)
        l = tl.load(part_l_ptr + base + segs)
        m_max = tl.max(m, 0)
        m_max = tl.where(m_max == float("-inf"), 0.0, m_max)
        w = tl.exp(m - m_max)  # segments with -inf give 0
        l_tot = tl.sum(l * w, 0)
        d = tl.arange(0, D)
        o = tl.load(part_o_ptr + (base + segs)[:, None] * D + d[None, :])  # [NSEG, D]
        o = tl.sum(o * w[:, None], 0) / tl.maximum(l_tot, 1e-30)
        tl.store(
            out_ptr + (q_start + i) * stride_ot + h * stride_oh + d,
            o.to(out_ptr.dtype.element_ty),
        )


class SpecDecodeAttention:
    """Holds the partial buffers; call .run(...) per layer."""

    def __init__(
        self, max_num_reqs, num_heads, head_dim, device, qmax, num_segments=NUM_SEGMENTS
    ):
        # The partial buffers are allocated once, for the longest query block the caller
        # will ever pass (qmax): a CUDA graph captures their addresses, so growing them
        # later would leave the captured decode graph pointing at freed memory.
        if _POLICY_GA100:
            # Buffers hold max(NSEG_ONE_REQUEST, max_num_reqs * num_segments)
            # partial rows per (head, query): the stock size whenever
            # max_num_reqs >= 2.
            num_segments = max(
                num_segments, -(-NSEG_ONE_REQUEST // max(1, max_num_reqs))
            )
        self.nseg = num_segments
        self.nseg_multi = (
            min(num_segments, NUM_SEGMENTS) if _POLICY_GA100 else num_segments
        )
        self.qmax = qmax
        self.max_num_reqs = max_num_reqs
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        n = max_num_reqs * num_heads * qmax * self.nseg
        self.part_o = torch.empty(n, head_dim, dtype=torch.float32, device=device)
        self.part_m = torch.empty(n, dtype=torch.float32, device=device)
        self.part_l = torch.empty(n, dtype=torch.float32, device=device)

    def _plan(self, q_len, G, D):
        """(BLOCK_M, tokens per tile, tile count, warps). One tile is preferred: every
        extra tile re-reads this request's KV segment."""
        override = os.environ.get("VLLM_SPEC_ATTN_BLOCK_M")
        rows = q_len * G
        if override:
            block_m = int(override)
        elif rows <= 32:
            block_m = 32
        elif rows <= 64:
            block_m = 64
        else:
            block_m = BLOCK_M_BIG
        block_m = max(block_m, G if G > 1 else 1)
        block_m = 1 << (block_m - 1).bit_length()
        qt = max(1, block_m // G)
        return block_m, qt, triton.cdiv(q_len, qt), 8 if block_m >= 128 else 4

    def run(
        self,
        q,
        key_cache,
        value_cache,
        out,
        cu_seqlens_q,
        seqused_k,
        block_table,
        scale,
        num_reqs,
        max_query_len,
        k_scale_cache=None,
        v_scale_cache=None,
    ):
        """key/value_cache: [num_blocks, block_size, Hkv, D] bf16, or [.., D + pad] int8
        with k/v_scale_cache [num_blocks, block_size, Hkv] float32 (per-token-head int8)."""
        Hq, D = q.shape[1], q.shape[2]
        Hkv = key_cache.shape[2]
        G = Hq // Hkv
        quant = k_scale_cache is not None
        assert max_query_len <= self.qmax, (
            "too many query tokens per request for this kernel"
        )
        assert num_reqs <= self.max_num_reqs
        # shared memory on sm86 is 99 KB: q tile + one K and one V tile + scores must fit
        block_m, qt, ntile, warps = self._plan(max_query_len, G, D)
        tile = 64 if (block_m <= 32 or D <= 128) else 32
        nseg, stages = self.nseg, 1
        if _POLICY_GA100:
            nseg = NSEG_ONE_REQUEST if num_reqs == 1 else self.nseg_multi
            stages = 2
            if block_m <= 32:
                tile, warps = 32, 4
        grid = (num_reqs * ntile, Hkv, nseg)
        _spec_attn_partial[grid](
            q,
            key_cache,
            value_cache,
            block_table,
            seqused_k,
            cu_seqlens_q,
            self.part_o,
            self.part_m,
            self.part_l,
            k_scale_cache,
            v_scale_cache,
            scale,
            q.stride(0),
            q.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            block_table.stride(0),
            *(k_scale_cache.stride() if quant else (0, 0, 0)),
            *(v_scale_cache.stride() if quant else (0, 0, 0)),
            G=G,
            Hq=Hq,
            QMAX=self.qmax,
            D=D,
            BLOCK_SIZE=key_cache.shape[1],
            BLOCK_M=block_m,
            TILE=tile,
            NSEG=nseg,
            QT=qt,
            NTILE=ntile,
            QUANT=quant,
            num_warps=warps,
            num_stages=stages,
        )
        _spec_attn_combine[(num_reqs, Hq, max_query_len)](
            self.part_o,
            self.part_m,
            self.part_l,
            out,
            cu_seqlens_q,
            out.stride(0),
            out.stride(1),
            Hq=Hq,
            QMAX=self.qmax,
            D=D,
            NSEG=nseg,
            num_warps=4,
        )
        return out
