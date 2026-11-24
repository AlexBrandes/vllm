# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 sparse attention with sinks on sm_80 (Triton).

One program per (query, block of 16 heads) walks the query's window slots in
the layer's sliding BF16 cache, then its selected compressed positions in the
kv source's BF16 cache (paged through that request's block table), with the
online softmax of the reference ``sparse_attn`` kernel: fp32 scores times the
softmax scale, a finite running-max floor (-1e30) so a query with no valid
row ends at zero output, probabilities rounded to BF16 before they weight the
rows (the running sum keeps them in fp32), and the per-head sink added to the
denominator at the end. Every paged offset is int64. No split-K, no atomics.

Either cache may hold exact packed records instead (``sm80_kv_pack``: MXFP8
window rows, NVFP4 compressed rows): they are decoded to the same BF16 tile
the BF16 rows load as, so the result is bit for bit the same.
"""

import os

import torch

from vllm.triton_utils import tl, triton
from vllm_cmp170hx.models.deepseek_v41.ops import sm80_kv_pack as kvpack
from vllm_cmp170hx.models.deepseek_v41.ops.sm80_kv_pack import load_records_tl

_BLOCK_H = 16
_BLOCK_N = 32
_SMS = {}


def _sm_count(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SMS:
        _SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SMS[idx]


@triton.jit
def _attend(q, kv, ok, m, l, acc, scale):
    s = tl.dot(q, tl.trans(kv)) * scale
    s = tl.where(ok[None, :], s, float("-inf"))
    m_new = tl.maximum(m, tl.max(s, axis=1))
    alpha = tl.exp(m - m_new)
    p = tl.exp(s - m_new[:, None])
    l = l * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv)
    return m_new, l, acc


@triton.jit(do_not_specialize=["n_window", "n_comp"])
def _sparse_attn_kernel(
    q_ptr,
    swa_ptr,
    swa_idx_ptr,
    comp_ptr,
    comp_bt_ptr,
    comp_idx_ptr,
    req_ptr,
    sink_ptr,
    out_ptr,
    n_heads,
    n_window,
    n_comp,
    scale,
    stride_qt,
    stride_qh,
    stride_sb,
    stride_se,
    stride_wi,
    stride_cb,
    stride_ce,
    stride_bt,
    stride_ci,
    stride_ot,
    stride_oh,
    SWA_E: tl.constexpr,
    COMP_E: tl.constexpr,
    HAS_COMP: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SWA_PACKED: tl.constexpr = False,
    COMP_PACKED: tl.constexpr = False,
):
    t = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = heads < n_heads
    d = tl.arange(0, D)
    q = tl.load(
        q_ptr + t * stride_qt + heads[:, None] * stride_qh + d[None, :],
        mask=hmask[:, None],
        other=0.0,
    )
    m = tl.full([BLOCK_H], -1.0e30, dtype=tl.float32)
    l = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)
    n = tl.arange(0, BLOCK_N)
    for n0 in range(0, n_window, BLOCK_N):
        cols = n0 + n
        slot = tl.load(swa_idx_ptr + t * stride_wi + cols, mask=cols < n_window, other=-1)
        ok = slot >= 0
        s64 = tl.maximum(slot, 0).to(tl.int64)
        off = (s64 // SWA_E) * stride_sb + (s64 % SWA_E) * stride_se
        if SWA_PACKED:
            kv = load_records_tl(swa_ptr, off, ok, D, False, 32, False, D)
        else:
            kv = tl.load(swa_ptr + off[:, None] + d[None, :], mask=ok[:, None], other=0.0)
        m, l, acc = _attend(q, kv, ok, m, l, acc, scale)
    if HAS_COMP:
        req = tl.load(req_ptr + t).to(tl.int64)
        for n0 in range(0, n_comp, BLOCK_N):
            cols = n0 + n
            j = tl.load(comp_idx_ptr + t * stride_ci + cols, mask=cols < n_comp, other=-1)
            ok = j >= 0
            j64 = tl.maximum(j, 0).to(tl.int64)
            blk = tl.load(comp_bt_ptr + req * stride_bt + j64 // COMP_E, mask=ok, other=0)
            off = blk.to(tl.int64) * stride_cb + (j64 % COMP_E) * stride_ce
            if COMP_PACKED:
                kv = load_records_tl(comp_ptr, off, ok, D, True, 16, True, D // 2)
            else:
                kv = tl.load(
                    comp_ptr + off[:, None] + d[None, :], mask=ok[:, None], other=0.0
                )
            m, l, acc = _attend(q, kv, ok, m, l, acc, scale)
    sink = tl.load(sink_ptr + heads, mask=hmask, other=float("-inf"))
    l = l + tl.exp(sink - m)
    o = acc / l[:, None]
    tl.store(
        out_ptr + t * stride_ot + heads[:, None] * stride_oh + d[None, :],
        o.to(out_ptr.dtype.element_ty),
        mask=hmask[:, None],
    )


# ---------------------------------------------------------------------------
# All-heads CUDA kernel (VLLM_DEEPSEEK_V41_ATTN_MLA=1, csrc/dsv41_mla): one CTA per (row, segment of 32-entry
# tiles) over all 64 heads, Q staged in shared memory once and each tile's records decoded once (instead of once
# per 16-head program); the same reference arithmetic (fp32 scores x scale, finite floor, BF16 probabilities for
# the value product, fp32 sum, sink in the denominator) in a different accumulation order: a numerics change.
# Segments per row: about 64 CTAs per call (_mla_segments).
# Results differ by <= 1 BF16 ulp between segment counts (not batch invariant).
# ---------------------------------------------------------------------------
_MLA_LIB = None


def _mla() -> bool:
    import os
    return os.environ.get("VLLM_DEEPSEEK_V41_ATTN_MLA") in ("1", "2", "3")


def _mla_invariant() -> bool:
    import os
    return os.environ.get("VLLM_DEEPSEEK_V41_ATTN_MLA") == "2"


def _mla_fixed() -> bool:
    import os
    return os.environ.get("VLLM_DEEPSEEK_V41_ATTN_MLA") == "3"


# VLLM_DEEPSEEK_V41_ATTN_MLA=3: the 64-head kernel with a FIXED segment size for every call (VLLM_DEEPSEEK_V41_MLA_TPS
# tiles of 32 entries, default 4) and always split + combine, so a row's computation never depends on the call
# (batch invariant); rows go in sub-calls of _MLA_FIXED_ROWS to bound the FP32 partial scratch.
_MLA_FIXED_ROWS = 32


def _mla_fixed_tps() -> int:
    import os
    return int(os.environ.get("VLLM_DEEPSEEK_V41_MLA_TPS", "4"))


# Batch-invariant form (VLLM_DEEPSEEK_V41_ATTN_MLA=2, op sparse_mla_inv): fixed segments of _MLA_SEG_TILES 32-entry
# tiles for every call, merged in segment order by one function, so a row's result does not depend on the call's
# row count; split (one CTA per segment + combine) while the call has at most _MLA_SPLIT_MAX_CTAS segment CTAs,
# else fused (one CTA per row and head group). Both give the same bits.
_MLA_SEG_TILES = int(__import__("os").environ.get("VLLM_DEEPSEEK_V41_MLA_INV_TPS", "4"))
_MLA_SPLIT_MAX_CTAS = int(__import__("os").environ.get("VLLM_DEEPSEEK_V41_MLA_SPLIT_MAX", "200"))


def _mla_inv_heads() -> int:
    import os
    v = os.environ.get("VLLM_DEEPSEEK_V41_MLA_INV_HC", "64")
    return int(v) if v in ("16", "32", "64") else 64


def _mla_heads_per_cta() -> int:
    import os
    return 32 if os.environ.get("VLLM_DEEPSEEK_V41_MLA_HC") == "32" else 64


def _mla_segments(rows: int, tiles: int, groups: int = 1) -> int:
    if groups > 1:  # 32 heads per CTA: two CTAs per SM, so about twice the CTAs
        return max(1, min(tiles, round(128 / max(rows * groups, 1))))
    # About 64 CTAs: fewer under-fills the card, more costs combine traffic (card 4: 6 rows best at 10 segments,
    # 12 at 5, 24 at 2-3; 18 rows at 2 segments was slower than the Triton path).
    return max(1, min(tiles, round(64 / max(rows, 1))))


def _mla_call(q, swa_cache, swa_indices, comp_cache, comp_block_table, comp_local, req, attn_sink, scale,
              swa_packed, comp_packed, out):
    global _MLA_LIB
    import os
    if _MLA_LIB is None:
        torch.ops.load_library(os.environ.get("VLLM_DEEPSEEK_V41_MLA_LIB", "/opt/dsv41-mla/sparse_mla.so"))
        _MLA_LIB = True
    n = q.shape[0]
    i32 = torch.int32
    if comp_local is None:  # no compressed keys: an empty selection, the window's tensors as placeholders
        comp_local = swa_indices[:, :0]
        comp_block_table = swa_indices
        comp_packed = swa_packed
    comp_local = comp_local if comp_local.dtype == i32 else comp_local.to(i32)
    comp_block_table = comp_block_table if comp_block_table.dtype == i32 else comp_block_table.to(i32)
    swa_idx = swa_indices if swa_indices.dtype == i32 else swa_indices.to(i32)
    sink = attn_sink if attn_sink.dtype == torch.float32 else attn_sink.float()
    tiles = triton.cdiv(swa_idx.shape[1], 32) + triton.cdiv(comp_local.shape[1], 32)
    if _mla_fixed():
        tps = _mla_fixed_tps()
        n_seg = triton.cdiv(tiles, tps)
        r = min(n, _MLA_FIXED_ROWS)
        pm = torch.empty((r, n_seg, 64), dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
        pacc = torch.empty((r, n_seg, 64, 512), dtype=torch.float32, device=q.device) if n_seg > 1 else pm
        for s0 in range(0, n, _MLA_FIXED_ROWS):
            s1 = min(n, s0 + _MLA_FIXED_ROWS)
            torch.ops.dsv41_mla.sparse_mla(q[s0:s1], swa_cache, swa_idx[s0:s1], comp_cache, comp_block_table,
                                           comp_local[s0:s1], req[s0:s1].to(i32), sink, float(scale),
                                           bool(swa_packed), bool(comp_packed), tps, out[s0:s1], pm, pl, pacc)
        return out
    if _mla_invariant():
        n_segs = triton.cdiv(tiles, _MLA_SEG_TILES)
        fused = n * n_segs > _MLA_SPLIT_MAX_CTAS
        # Every head grouping computes each head identically (bitwise, card 4), so split calls take the fastest:
        # 1 row 16 heads 35 us / 32: 42 / 64: 58; 6 rows 32: 46 / 64: 59; 12-24 rows 64; 32 rows 32: 169 / 64: 182.
        ih = 64 if fused else (16 if n <= 2 else 32 if n <= 8 else 64 if n <= 24 else 32)
        if os.environ.get("VLLM_DEEPSEEK_V41_MLA_INV_HC"):
            ih = _mla_inv_heads()
        if fused:
            pm = pl = pacc = torch.empty(1, dtype=torch.float32, device=q.device)
        else:
            pm = torch.empty((n, n_segs, 64), dtype=torch.float32, device=q.device)
            pl = torch.empty_like(pm)
            pacc = torch.empty((n, n_segs, 64, 512), dtype=torch.float32, device=q.device)
        torch.ops.dsv41_mla.sparse_mla_inv(q, swa_cache, swa_idx, comp_cache, comp_block_table, comp_local,
                                           req.to(i32), sink, float(scale), bool(swa_packed), bool(comp_packed),
                                           _MLA_SEG_TILES, fused, out, pm, pl, pacc, ih)
        return out
    hc = _mla_heads_per_cta()
    n_seg = _mla_segments(n, tiles, 64 // hc)
    tps = triton.cdiv(tiles, n_seg)
    n_seg = triton.cdiv(tiles, tps)
    pm = torch.empty((n, n_seg, 64), dtype=torch.float32, device=q.device)
    pl = torch.empty_like(pm)
    pacc = torch.empty((n, n_seg, 64, 512), dtype=torch.float32, device=q.device) if n_seg > 1 else pm
    torch.ops.dsv41_mla.sparse_mla(q, swa_cache, swa_idx, comp_cache, comp_block_table, comp_local,
                                   req.to(i32), sink, float(scale), bool(swa_packed), bool(comp_packed), tps,
                                   out, pm, pl, pacc, hc)
    return out


def sparse_attention_triton(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    comp_cache: torch.Tensor | None,
    comp_block_table: torch.Tensor | None,
    comp_local: torch.Tensor | None,
    req: torch.Tensor,
    attn_sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """See ``sm80_attn_core.sparse_attention_torch`` (same contract); either
    cache may be packed (uint8 records, ``sm80_kv_pack``)."""
    n, h, d = q.shape
    out = torch.empty((n, h, d), dtype=torch.bfloat16, device=q.device)
    if n == 0:
        return out
    assert q.stride(2) == 1 and swa_cache.stride(2) == 1
    assert swa_indices.stride(1) == 1
    swa_packed = kvpack.is_packed(swa_cache)
    if swa_packed:
        kvpack.check_cache(swa_cache, kvpack.MXFP8, d)
    has_comp = comp_local is not None and comp_local.shape[1] > 0
    comp_packed = has_comp and kvpack.is_packed(comp_cache)
    if has_comp:
        assert comp_cache is not None and comp_block_table is not None
        assert comp_cache.stride(2) == 1 and comp_local.stride(1) == 1
        if comp_packed:
            kvpack.check_cache(comp_cache, kvpack.NVFP4, d)
        comp_args = (
            comp_cache,
            comp_block_table,
            comp_local,
            comp_cache.stride(0),
            comp_cache.stride(1),
            comp_block_table.stride(0),
            comp_local.stride(0),
            comp_local.shape[1],
            comp_cache.shape[1],
        )
    else:
        comp_args = (swa_cache, swa_indices, swa_indices, 0, 0, 0, 0, 0, 1)
    comp, comp_bt, comp_idx, s_cb, s_ce, s_bt, s_ci, n_comp, comp_e = comp_args
    if _mla() and h == 64 and d == 512:
        return _mla_call(q, swa_cache, swa_indices, comp_cache if has_comp else swa_cache,
                         comp_block_table if has_comp else None, comp_local if has_comp else None,
                         req, attn_sink, scale, swa_packed, comp_packed, out)
    grid = (n, triton.cdiv(h, _BLOCK_H))
    # One program per SM at 8 warps (228 registers); 4 warps (255 registers) fit two per SM, faster once the
    # grid exceeds one wave (card 4: packed 24 rows 204 -> 152 us, 2,048 rows 11.7 -> 8.4 ms) and slower
    # below it (1 row 115 -> 143 us). Same bits either way (tested 1-2,048 rows, packed and BF16), so the
    # choice by grid size keeps batch invariance.
    warps = 4 if grid[0] * grid[1] > _sm_count(q.device) else 8
    _sparse_attn_kernel[grid](
        q,
        swa_cache,
        swa_indices,
        comp,
        comp_bt,
        comp_idx,
        req.to(torch.int32).contiguous(),
        attn_sink,
        out,
        h,
        swa_indices.shape[1],
        n_comp,
        scale,
        q.stride(0),
        q.stride(1),
        swa_cache.stride(0),
        swa_cache.stride(1),
        swa_indices.stride(0),
        s_cb,
        s_ce,
        s_bt,
        s_ci,
        out.stride(0),
        out.stride(1),
        SWA_E=swa_cache.shape[1],
        COMP_E=comp_e,
        HAS_COMP=has_comp,
        D=d,
        BLOCK_H=_BLOCK_H,
        BLOCK_N=_BLOCK_N,
        SWA_PACKED=swa_packed,
        COMP_PACKED=comp_packed,
        num_warps=warps,
    )
    return out
