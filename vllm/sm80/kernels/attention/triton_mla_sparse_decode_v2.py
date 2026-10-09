# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA decode for sm_80 at up to 64 query rows (``VLLM_AMPERE_SPARSE_MLA_DECODE_V2``).

Reached only through the gate in
``vllm/sm80/kernels/attention/triton_mla_sparse.py::triton_mla_sparse_fwd``
(bf16, the 512-wide NoPE layout, 16 or 64 query heads, 2048 or 2176 index slots,
<= 64 rows, sm_80); same signature and outputs as that function.

What differs from it: each CTA reads its whole index row once and walks only the
tiles between the row's first and last valid entry; a row whose valid entries
one split owns is finished by that split; with four or more splits a first pass
finishes dense rows and finds each sparse row's exact logit max, and a second
pass spreads the sparse row's valid tiles evenly over the splits, rounding P
against that max; partials are combined by a parallel merge kernel. At 16-head
blocks of 64-key tiles every pass runs the Gluon walk of
``sparse_mla_decode_gluon.py`` (double-buffered cp.async gather); the other
schedules (64 heads at 8 or 16 rows) run the Triton kernel below. The
arithmetic order differs from the shipped kernel (FP64 error no larger).

CUDA graphs: no host sync and no allocation besides the returned tensors on a
repeat call; the split-k workspace of each (device, rows, heads, splits) is
created on the first eager call at that shape (vLLM's warmup at every capture
size, before capture).

Layout conventions (shared with the other sparse MLA backends):

* ``q``       -- ``[num_tokens, num_heads_q, dim_qk]`` bf16/fp16, where
  ``dim_qk = kv_lora_rank + qk_rope_head_dim`` (512 for NoPE GLM-5.3-Flash,
  576 for DeepSeek).
* ``kv``      -- ``[num_rows, 1, dim_qk]``, a flat row view of the paged
  latent cache (``flat_kv_row_view``). The first ``d_v`` dims double as V.
* ``indices`` -- ``[num_tokens, 1, topk]`` int32 global cache rows; ``-1``
  (or any out-of-range row) is masked out of the softmax.
"""

import functools

import torch

from vllm.sm80.kernels.attention.sparse_mla_decode_gluon import gluon_walk_kernel
from vllm.triton_utils import LOG2E, LOGE2, tl, triton


def _cdiv(a: int, b: int) -> int:
    """Ceiling division. ``triton.cdiv`` is a ``ConstexprFunction``: it costs
    ~0.9 us of host time per call, which matters on the decode path."""
    return -(a // -b)


@triton.jit
def _mla_sparse_kernel(
    q_buffer,
    k_buffer,
    v_buffer,
    indices_ptr,
    out_ptr,
    softmax_lse_ptr,
    part_acc_ptr,  # fp32 [T, H, NUM_SPLITS, BLOCK_DV] workspace (NUM_SPLITS > 1)
    part_sum_ptr,  # fp32 [T, H, NUM_SPLITS]
    part_max_ptr,  # fp32 [T, H, NUM_SPLITS]
    row_max_ptr,  # fp32 [T, H, NUM_SPLITS]: pass-1 split maxima of a sparse row
    split_bounds_ptr,  # int32 [T, NUM_SPLITS, 4]: pass-1 walks (start, end, use_max, -)
    max_logits_ptr,
    seq_kv,
    h_q,
    dim_qk,
    dim_v,
    stride_q_token,
    stride_q_head,
    stride_k_token,
    stride_k_head,
    stride_v_token,
    stride_v_head,
    stride_out_token,
    stride_out_head,
    stride_lse,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    split_len,
    kv_group_num: tl.constexpr,
    index_topk: tl.constexpr,
    NUM_SPLITS: tl.constexpr,  # >1: write unnormalised partials for _mla_merge_kernel
    BLOCK_H: tl.constexpr,  # block size for num heads
    BLOCK_N: tl.constexpr,  # block size for indices
    BLOCK_DV: tl.constexpr,  # block size for dim_v
    BLOCK_DMODEL: tl.constexpr,  # block size for dim_nope
    BLOCK_DPE: tl.constexpr,  # block size for positional embedding
    V_IS_K: tl.constexpr,  # v_buffer rows == k_buffer nope rows (d_v == BLOCK_DMODEL)
    LOGE2: tl.constexpr,
    PASS: tl.constexpr,  # 0 single pass; 1 / 2 the two passes (see below)
    ROW_W: tl.constexpr,  # power of two >= index_topk: one load scans the row
):
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_split = tl.program_id(2)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < h_q)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)

    off_q = cur_q * stride_q_token + cur_head[:, None] * stride_q_head + offs_d[None, :]
    mask_dmodel = offs_d < BLOCK_DMODEL
    q = tl.load(
        q_buffer + off_q, mask=(mask_h[:, None]) & (mask_dmodel[None, :]), other=0.0
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        off_qpe = (
            cur_q * stride_q_token
            + cur_head[:, None] * stride_q_head
            + offs_dpe[None, :]
        )
        # dim_qk == BLOCK_DMODEL + BLOCK_DPE (checked by the launcher)
        mask_dpe = offs_dpe < dim_qk
        qpe = tl.load(
            q_buffer + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
        )

    # Use a finite sentinel rather than -inf: if every key in a chunk is
    # masked (e.g. leading -1 padding in `indices`) an -inf running max gives
    # re_scale = exp2(-inf - -inf) = NaN, permanently poisoning acc / e_sum.
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - 1.0e30
    if BLOCK_DPE == 0:
        qpe = q
        offs_dpe = offs_d
    index_row_ptr = indices_ptr + cur_q * stride_indices_token + cur_kv_head_id * stride_indices_head

    if PASS == 0:
        # Single pass. One vector load of the whole index row gives its first
        # and last valid column on the device (any layout). Each split walks
        # only the tiles between them that fall in its columns, on its own tile
        # grid: a skipped leading tile would only have added to e_sum what the
        # first valid tile's re_scale = exp2(-1e30 - m) = 0 wipes, a skipped
        # trailing tile adds p = 0 with re_scale = 1; a -1 gap inside the range
        # is walked and masked. A row whose valid entries all fall in one split
        # (a short row) is finished by that split, which stores the output
        # directly; the other splits skip it and the merge leaves it alone
        # (flag in split_bounds[t, 0, 3]). Merging it with empty splits would
        # give the same bits (weight 1 for the owner, 0 for the rest).
        offs_row = tl.arange(0, ROW_W)
        idx_row = tl.load(index_row_ptr + offs_row, mask=offs_row < index_topk, other=-1)
        valid_row = (idx_row >= 0) & (idx_row < seq_kv)
        first_valid = tl.min(tl.where(valid_row, offs_row, ROW_W), 0)
        last_valid = tl.max(tl.where(valid_row, offs_row, -1), 0)
        split_start = cur_split * split_len
        split_end = tl.minimum(split_start + split_len, index_topk)
        lo = tl.maximum(first_valid, split_start)
        hi = tl.minimum(last_valid, split_end - 1)
        loop_start = split_start + ((lo - split_start) // BLOCK_N) * BLOCK_N
        loop_end = hi + 1
        if NUM_SPLITS > 1:
            owned = (first_valid // split_len) == (last_valid // split_len)
            direct = owned & (first_valid // split_len == cur_split)
            if (cur_split == 0) & (cur_head_id == 0):
                tl.store(split_bounds_ptr + cur_q * NUM_SPLITS * 4 + 3, owned.to(tl.int32))
            run = (owned == 0) | direct
        else:
            direct = False
            run = True
        if run:
            _mla_attend(
                q, qpe, k_buffer, v_buffer, indices_ptr, out_ptr, softmax_lse_ptr,
                part_acc_ptr, part_sum_ptr, part_max_ptr, max_logits_ptr,
                loop_start, loop_end, e_max, direct,
                cur_q, cur_head, cur_head_id, cur_split, cur_kv_head_id, mask_h,
                offs_d, offs_dv, offs_dpe, mask_dmodel, seq_kv, h_q, dim_qk, dim_v,
                stride_k_token, stride_k_head, stride_v_token, stride_v_head,
                stride_out_token, stride_out_head, stride_lse,
                stride_indices_token, stride_indices_head, sm_scale,
                index_topk, NUM_SPLITS, BLOCK_H, BLOCK_N, BLOCK_DV, BLOCK_DPE, V_IS_K, LOGE2,
            )
    elif PASS == 1:
        # Two passes (NoPE, V = K, NUM_SPLITS > 1). Pass 1 scans the index row,
        # stores every split's walk for pass 2, and then
        # * dense row: does the whole single-pass job (column split, merge,
        #   output) right here, so pass 2 costs it only an early exit;
        # * sparse row: computes this split's max logit over its share of the
        #   row's valid tiles (same gather, dot, scale and mask as the walk).
        loop_start, loop_end, use_max = _mla_split_bounds(
            index_row_ptr, seq_kv, index_topk, cur_split, split_len, NUM_SPLITS, BLOCK_N, ROW_W,
        )
        if cur_head_id == 0:
            offs_b = tl.arange(0, 4)
            bounds = tl.where(
                offs_b == 0, loop_start, tl.where(offs_b == 1, loop_end, use_max.to(tl.int32))
            )
            tl.store(split_bounds_ptr + (cur_q * NUM_SPLITS + cur_split) * 4 + offs_b, bounds)
        if use_max:
            for start_indice in range(loop_start, loop_end, BLOCK_N):
                offs_indice = start_indice + tl.arange(0, BLOCK_N)
                indices = tl.load(
                    index_row_ptr + offs_indice, mask=offs_indice < index_topk, other=-1
                )
                mask_kv = (indices >= 0) & (indices < seq_kv)
                kn = tl.load(
                    k_buffer + indices[:, None].to(tl.int64) * stride_k_token + offs_d[None, :],
                    mask=mask_kv[:, None],
                    other=0.0,
                )
                qk = tl.dot(q, tl.trans(kn))
                qk *= sm_scale
                qk = tl.where((mask_h[:, None]) & (mask_kv[None, :]), qk, -1.0e30)
                e_max = tl.maximum(e_max, tl.max(qk, 1))
            tl.store(
                row_max_ptr + (cur_q * h_q + cur_head) * NUM_SPLITS + cur_split, e_max, mask=mask_h
            )
        else:
            _mla_attend(
                q, qpe, k_buffer, v_buffer, indices_ptr, out_ptr, softmax_lse_ptr,
                part_acc_ptr, part_sum_ptr, part_max_ptr, max_logits_ptr,
                loop_start, loop_end, e_max, False,
                cur_q, cur_head, cur_head_id, cur_split, cur_kv_head_id, mask_h,
                offs_d, offs_dv, offs_dpe, mask_dmodel, seq_kv, h_q, dim_qk, dim_v,
                stride_k_token, stride_k_head, stride_v_token, stride_v_head,
                stride_out_token, stride_out_head, stride_lse,
                stride_indices_token, stride_indices_head, sm_scale,
                index_topk, NUM_SPLITS, BLOCK_H, BLOCK_N, BLOCK_DV, BLOCK_DPE, V_IS_K, LOGE2,
            )
    else:
        # Pass 2: sparse rows only (a dense row was finished by pass 1). The
        # walk starts from the exact row max: n_e_max stays m_row, re_scale is
        # exactly 1, and P is rounded to bf16 against the row max like the
        # reference, however finely the valid tiles are spread over splits;
        # the merge then weighs every partial by exp2(0) = 1.
        split_ptr = split_bounds_ptr + (cur_q * NUM_SPLITS + cur_split) * 4
        use_max = tl.load(split_ptr + 2) != 0
        if use_max:
            loop_start = tl.load(split_ptr)
            loop_end = tl.load(split_ptr + 1)
            m_ptr = row_max_ptr + (cur_q * h_q + cur_head) * NUM_SPLITS
            for i in tl.static_range(NUM_SPLITS):
                e_max = tl.maximum(e_max, tl.load(m_ptr + i, mask=mask_h, other=-1.0e30))
            _mla_attend(
                q, qpe, k_buffer, v_buffer, indices_ptr, out_ptr, softmax_lse_ptr,
                part_acc_ptr, part_sum_ptr, part_max_ptr, max_logits_ptr,
                loop_start, loop_end, e_max, False,
                cur_q, cur_head, cur_head_id, cur_split, cur_kv_head_id, mask_h,
                offs_d, offs_dv, offs_dpe, mask_dmodel, seq_kv, h_q, dim_qk, dim_v,
                stride_k_token, stride_k_head, stride_v_token, stride_v_head,
                stride_out_token, stride_out_head, stride_lse,
                stride_indices_token, stride_indices_head, sm_scale,
                index_topk, NUM_SPLITS, BLOCK_H, BLOCK_N, BLOCK_DV, BLOCK_DPE, V_IS_K, LOGE2,
            )


@triton.jit
def _mla_attend(
    q, qpe, k_buffer, v_buffer, indices_ptr, out_ptr, softmax_lse_ptr,
    part_acc_ptr, part_sum_ptr, part_max_ptr, max_logits_ptr,
    loop_start, loop_end, e_max, direct,
    cur_q, cur_head, cur_head_id, cur_split, cur_kv_head_id, mask_h,
    offs_d, offs_dv, offs_dpe, mask_dmodel,
    seq_kv, h_q, dim_qk, dim_v,
    stride_k_token, stride_k_head, stride_v_token, stride_v_head,
    stride_out_token, stride_out_head, stride_lse,
    stride_indices_token, stride_indices_head, sm_scale,
    index_topk: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    V_IS_K: tl.constexpr,
    LOGE2: tl.constexpr,
):
    """Online-softmax walk of tiles [loop_start, loop_end) from the running max
    e_max, then this split's partial (or, unsplit, the output). With NUM_SPLITS
    > 1 the last CTA of the (row, head block) to arrive merges every split."""
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)
    for start_indice in range(loop_start, loop_end, BLOCK_N):
        offs_indice = start_indice + tl.arange(0, BLOCK_N)
        mask_indice = offs_indice < index_topk
        indices = tl.load(
            indices_ptr
            + (
                cur_q * stride_indices_token
                + cur_kv_head_id * stride_indices_head
                + offs_indice
            ),
            mask=mask_indice,
            other=-1,
        )

        mask_kv = (indices >= 0) & (indices < seq_kv)
        mask_kv_d = mask_dmodel
        if V_IS_K:
            # V is the leading dim_v == BLOCK_DMODEL dims of the same rows: gather
            # each row once in its natural [BLOCK_N, D] layout and reuse it for
            # both dots instead of reading the row twice from L2.
            offs_kn = (
                indices[:, None].to(tl.int64) * stride_k_token
                + cur_kv_head_id * stride_k_head
                + offs_d[None, :]
            )
            kn = tl.load(
                k_buffer + offs_kn, mask=(mask_kv[:, None]) & (mask_kv_d[None, :]), other=0.0
            ).to(q.dtype)
            qk = tl.dot(q, tl.trans(kn))
        else:
            offs_k = (
                indices[None, :].to(tl.int64) * stride_k_token
                + cur_kv_head_id * stride_k_head
                + offs_d[:, None]
            )
            # q_nope @ k_nope
            k = tl.load(
                k_buffer + offs_k, mask=(mask_kv[None, :]) & (mask_kv_d[:, None]), other=0.0
            )
            qk = tl.dot(q, k.to(q.dtype))

        if BLOCK_DPE > 0:
            # q_rope @ k_rope
            offs_kpe = (
                indices[None, :].to(tl.int64) * stride_k_token
                + cur_kv_head_id * stride_k_head
                + offs_dpe[:, None]
            )
            mask_k_dpe = offs_dpe < dim_qk
            kpe = tl.load(
                k_buffer + offs_kpe,
                mask=(mask_kv[None, :]) & (mask_k_dpe[:, None]),
                other=0.0,
            )
            qk += tl.dot(qpe, kpe.to(q.dtype))

        # apply scaling (sm_scale already carries log2(e))
        qk *= sm_scale
        qk = tl.where((mask_h[:, None]) & (mask_kv[None, :]), qk, -1.0e30)

        if not V_IS_K:
            # load v
            mask_v_d = offs_dv < dim_v
            offs_v = (
                indices[:, None].to(tl.int64) * stride_v_token
                + cur_kv_head_id * stride_v_head
                + offs_dv[None, :]
            )
            v = tl.load(
                v_buffer + offs_v, mask=(mask_kv[:, None]) & (mask_v_d[None, :]), other=0.0
            )

        # online softmax
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        acc *= re_scale[:, None]

        # score @ v
        if V_IS_K:
            acc += tl.dot(p.to(kn.dtype), kn)
        else:
            acc += tl.dot(p.to(v.dtype), v)

        # update global sum and max
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    if (NUM_SPLITS > 1) & (direct == 0):
        # Publish this split's unnormalised partial (acc, e_sum, base-2 max);
        # _mla_merge_kernel combines the NUM_SPLITS partials. Every split of a
        # merged row writes its partial on every call (an empty one: 0, 0, -1e30).
        row = cur_q * h_q + cur_head
        offs_p = (row[:, None] * NUM_SPLITS + cur_split) * BLOCK_DV + offs_dv[None, :]
        tl.store(part_acc_ptr + offs_p, acc, mask=mask_h[:, None])
        offs_ps = row * NUM_SPLITS + cur_split
        tl.store(part_sum_ptr + offs_ps, e_sum, mask=mask_h)
        tl.store(part_max_ptr + offs_ps, e_max, mask=mask_h)
    else:
        _mla_store_output(
            out_ptr, softmax_lse_ptr, max_logits_ptr, acc, e_sum, e_max,
            cur_q, cur_head, mask_h, offs_dv, dim_v,
            stride_out_token, stride_out_head, stride_lse, LOGE2,
        )


@triton.jit
def _mla_merge_kernel(
    out_ptr,
    softmax_lse_ptr,
    max_logits_ptr,
    part_acc_ptr,  # fp32 [T, H, NUM_SPLITS, BLOCK_DV]
    part_sum_ptr,  # fp32 [T, H, NUM_SPLITS]
    part_max_ptr,  # fp32 [T, H, NUM_SPLITS]
    split_bounds_ptr,  # int32 [T, NUM_SPLITS, 4]; [t, 0, 3] = row finished directly
    h_q,
    dim_v,
    stride_out_token,
    stride_out_head,
    stride_lse,
    NUM_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DV: tl.constexpr,  # the partials' row length
    BLOCK_DC: tl.constexpr,  # output columns per CTA
    LOGE2: tl.constexpr,
    CHECK_DIRECT: tl.constexpr,  # single-pass producer: skip rows it finished
):
    """Combine the split partials of (row, head block, column chunk): many CTAs
    share the merge instead of one CTA per row reading NUM_SPLITS x 32 KB. Same
    arithmetic and summation order as a single merging CTA; every chunk
    recomputes the (tiny) statistics, chunk 0 stores lse / max_logits."""
    cur_q = tl.program_id(0)
    cur_head = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    chunk = tl.program_id(2)
    if CHECK_DIRECT:
        skip = tl.load(split_bounds_ptr + cur_q * NUM_SPLITS * 4 + 3)
    else:
        skip = 0
    if skip == 0:
        mask_h = cur_head < h_q
        row = cur_q * h_q + cur_head
        offs_s = tl.arange(0, NUM_SPLITS)
        offs_ms = row[:, None] * NUM_SPLITS + offs_s[None, :]
        m_all = tl.load(part_max_ptr + offs_ms, mask=mask_h[:, None], other=-1.0e30)
        s_all = tl.load(part_sum_ptr + offs_ms, mask=mask_h[:, None], other=0.0)
        e_max = tl.max(m_all, 1)
        e_sum = tl.sum(tl.exp2(m_all - e_max[:, None]) * s_all, 1)
        offs_c = chunk * BLOCK_DC + tl.arange(0, BLOCK_DC)
        acc = tl.zeros([BLOCK_H, BLOCK_DC], dtype=tl.float32)
        for i in tl.static_range(NUM_SPLITS):
            m_i = tl.load(part_max_ptr + row * NUM_SPLITS + i, mask=mask_h, other=-1.0e30)
            offs_a = (row[:, None] * NUM_SPLITS + i) * BLOCK_DV + offs_c[None, :]
            acc_i = tl.load(part_acc_ptr + offs_a, mask=mask_h[:, None], other=0.0)
            acc += acc_i * tl.exp2(m_i - e_max)[:, None]

        e_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
        acc /= e_sum[:, None]
        offs_o = cur_q * stride_out_token + cur_head[:, None] * stride_out_head + offs_c[None, :]
        tl.store(
            out_ptr + offs_o,
            acc.to(out_ptr.dtype.element_ty),
            mask=(mask_h[:, None]) & (offs_c[None, :] < dim_v),
        )
        if chunk == 0:
            max_logits = e_max * LOGE2
            lse = max_logits + tl.log2(e_sum) * LOGE2
            offs_lse = cur_q * stride_lse + cur_head
            tl.store(softmax_lse_ptr + offs_lse, lse, mask=mask_h)
            tl.store(max_logits_ptr + offs_lse, max_logits, mask=mask_h)


@triton.jit
def _mla_split_bounds(
    index_row_ptr, seq_kv, index_topk, cur_split, split_len,
    NUM_SPLITS: tl.constexpr, BLOCK_N: tl.constexpr, ROW_W: tl.constexpr,
):
    """(loop_start, loop_end, use_max) of one split's tile walk, from one load of
    the whole index row on the device (any layout).

    Sparse row (its first..last valid range spans at most half the row's tiles):
    the range's BLOCK_N tiles (row grid) are dealt out evenly over the splits,
    and the caller rounds P against the exact row max (use_max). Dense row: the
    column split, trimmed to the row's first..last valid index (a skipped
    all-masked leading tile only fed e_sum what the first valid tile's
    re_scale = 0 wipes; a trailing one adds p = 0), bitwise the full walk.
    A -1 gap inside the range is walked and masked. No valid index: no tile.
    """
    offs_row = tl.arange(0, ROW_W)
    idx_row = tl.load(index_row_ptr + offs_row, mask=offs_row < index_topk, other=-1)
    valid_row = (idx_row >= 0) & (idx_row < seq_kv)
    first_valid = tl.min(tl.where(valid_row, offs_row, ROW_W), 0)
    last_valid = tl.max(tl.where(valid_row, offs_row, -1), 0)
    first_tile = first_valid // BLOCK_N
    num_tiles = tl.maximum(last_valid // BLOCK_N + 1 - first_tile, 0)
    use_max = num_tiles * 2 <= (index_topk + BLOCK_N - 1) // BLOCK_N

    tiles_per_split = (num_tiles + NUM_SPLITS - 1) // NUM_SPLITS
    bal_start = (first_tile + cur_split * tiles_per_split) * BLOCK_N
    bal_end = tl.minimum(bal_start + tiles_per_split * BLOCK_N, last_valid + 1)

    split_start = cur_split * split_len
    split_end = tl.minimum(split_start + split_len, index_topk)
    lo = tl.maximum(first_valid, split_start)
    hi = tl.minimum(last_valid, split_end - 1)
    col_start = split_start + ((lo - split_start) // BLOCK_N) * BLOCK_N
    col_end = hi + 1

    loop_start = tl.where(use_max, bal_start, col_start)
    loop_end = tl.where(use_max, bal_end, col_end)
    return loop_start, loop_end, use_max


@triton.jit
def _mla_store_output(
    out_ptr, softmax_lse_ptr, max_logits_ptr, acc, e_sum, e_max,
    cur_q, cur_head, mask_h, offs_dv, dim_v,
    stride_out_token, stride_out_head, stride_lse, LOGE2: tl.constexpr,
):
    # rescaling. A row with no valid key skipped every tile: e_sum = 0, acc = 0
    # and e_max = -1e30, so dividing by 1 gives out 0 and lse = max_logits, a
    # finite sentinel.
    e_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    acc /= e_sum[:, None]

    max_logits = e_max * LOGE2
    # calculate lse (natural log, matching FlashMLA / FA conventions)
    lse = max_logits + tl.log2(e_sum) * LOGE2

    # write output
    offs_o = (
        cur_q * stride_out_token
        + cur_head[:, None] * stride_out_head
        + offs_dv[None, :]
    )
    mask_out_d = offs_dv < dim_v
    tl.store(
        out_ptr + offs_o,
        acc.to(out_ptr.dtype.element_ty),
        mask=(mask_h[:, None]) & (mask_out_d[None, :]),
    )

    offs_lse = cur_q * stride_lse + cur_head
    tl.store(softmax_lse_ptr + offs_lse, lse, mask=mask_h)
    tl.store(max_logits_ptr + offs_lse, max_logits, mask=mask_h)


@functools.lru_cache
def _smem_budget(device_index: int) -> int:
    """Shared memory an sm_80 CTA may opt into, read from the device.

    BLOCK_N=64 at two stages asks for 128 KB at dim_qk=512, which GA100 grants
    and the 100 KB parts (sm_86 / sm_89) do not, so the tile is chosen against
    the real limit rather than against a constant.
    """
    try:
        return torch.cuda.get_device_properties(
            device_index
        ).shared_memory_per_block_optin
    except Exception:
        return 101376


@functools.lru_cache
def _num_sms(device_index: int) -> int:
    try:
        return torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:
        return 108


# Decode schedules for a card holding more than 16 query heads (all 64 when
# one card runs whole layers under pipeline parallel), keyed by the largest
# query-row count each covers: (BLOCK_H, BLOCK_N, num_splits, num_warps,
# num_stages). Measured at h_q = 64, top-k 2048, over 640 schedules per row
# count against the 16-head rule below applied to 64 heads (graph replay,
# rotated caches, median of 20):
#
#     rows  16-head rule at 64 heads   us     measured best       us     gain
#     4     H32 N64 S8 W4 P2         65.6    H16 N64 S4 W4 P2   48.5    1.35x
#     8     H32 N64 S4 W4 P2         95.2    H16 N32 S4 W4 P2   71.9    1.32x
#     16    H32 N64 S2 W4 P2        142.9    H32 N32 S4 W4 P2   97.4    1.47x
#     32    H32 N64 S1 W4 P2        159.4    (same)                       -
#
# (context 8K at 4 rows, 64K otherwise.) The rule's CTA ramp stops at two
# CTAs per SM, which leaves a 64-head grid with too few splits; one wave of
# 16-head blocks with four splits is what these shapes want. Past 16 rows the
# rule already picks the measured best.
_WIDE_HEAD_DECODE = ((4, (16, 64, 4, 4, 2)),
                     (8, (16, 32, 4, 4, 2)),
                     (16, (32, 32, 4, 4, 2)))


def _pick_config(
    num_tokens: int,
    index_topk: int,
    num_heads_q: int = 64,
    dim_qk: int = 512,
    device_index: int = 0,
) -> tuple[int, int, int, int, int]:
    """(BLOCK_H, BLOCK_N, num_splits, num_warps, num_stages).

    Measured on sm_80 (GA100, 70 SMs) at the deployed shape -- 16 heads per
    rank, top-k 2048, dim_qk 512 -- over 550 schedules at 4 and 16 query rows
    and context 8K / 64K / 200K, each against this kernel's previous schedule
    in the same run (graph replay, rotated caches, median of 20):

        rows  previous            us     new                 us     gain
        4     H16 N32 S8 W4 P3   32.1    H16 N64 S8 W4 P2   30.3    1.06x
        16    H32 N32 S4 W4 P3   59.8    H16 N64 S4 W4 P2   50.2    1.19x

    The previous schedule ranked 6th of 550 at 4 rows and 19th at 16. Three
    things changed:

    * **64 keys per tile, not 32.** Worth most of both numbers. This kernel is
      issue-bound rather than gather-bound -- a 12x change in the KV working
      set moves it under 1 % -- so what pays is halving the trip count over the
      top-k list, not touching the bytes. It costs shared memory, which is why
      the tile is sized against the device below.
    * **Two pipeline stages, not three.** The gather is an indirect load that
      Triton cannot pipeline, so a third stage double-buffers the wrong thing.
    * **The head tile is sized to the heads that exist.** The old rule took 32
      above 8 query rows, reasoning about a 64-head rank; a rank owning 16
      heads then builds a 32-row tile with half its rows masked and throws away
      half of every qk and pv dot. Worth 1.04x on its own at 16 rows (59.8 ->
      57.4 us), and nothing at 4, where the old rule already chose 16.
      Repartitioning heads across CTAs cannot change a result bit -- each
      head's softmax reduction is independent.

    The split ramp is also computed from the real head-block count rather than
    from a hard-coded 64 heads, and fills one wave of SMs rather than a fixed
    160 CTAs. At the two shapes above both forms agree (8 splits at 4 rows, 4
    at 16); they differ at 8 query rows, where the old estimate double-counts
    the CTAs and stops at 4. Splits stay <= 8 because the last-CTA merge reads
    splits x BLOCK_H x 2 KB, and past the wave the merge tail costs more than
    the parallelism it buys (16 rows: 4 splits 50.2 us, 8 splits 60.8, 16
    splits 81.4).
    """
    smem = _smem_budget(device_index)
    if num_heads_q > 16:
        for max_rows, cfg in _WIDE_HEAD_DECODE:
            if num_tokens <= max_rows:
                bh, bn, splits, warps, stages = cfg
                if stages * bn * dim_qk * 2 <= smem:
                    return bh, bn, min(splits, _cdiv(index_topk, bn)), warps, stages
                break

    # Never build a head tile wider than the heads the rank holds; 32 stays the
    # ceiling, so a rank with more than 16 heads keeps today's tile.
    BLOCK_H = min(32, max(16, 1 << (num_heads_q - 1).bit_length()))

    # stages * BLOCK_N * dim_qk * 2 B of staging must fit what the device will
    # grant one CTA. Give up the wide tile before the second stage: the tile is
    # worth more than the pipelining on a kernel whose loads are indirect.
    BLOCK_N, num_stages = 64, 2
    while BLOCK_N > 16 and num_stages * BLOCK_N * dim_qk * 2 > smem:
        BLOCK_N //= 2
    while num_stages > 1 and num_stages * BLOCK_N * dim_qk * 2 > smem:
        num_stages -= 1

    head_blocks = _cdiv(num_heads_q, min(BLOCK_H, num_heads_q))
    ctas = num_tokens * head_blocks
    num_splits = 1
    while num_splits < 8 and ctas * num_splits * 2 <= _num_sms(device_index):
        num_splits *= 2
    num_splits = min(num_splits, _cdiv(index_topk, BLOCK_N))
    return BLOCK_H, BLOCK_N, num_splits, 4, num_stages


_WORKSPACE: dict[tuple, tuple[torch.Tensor, ...]] = {}


def _workspace(device, num_tokens, num_heads, num_splits, block_dv):
    """Split-k scratch, cached per shape (no per-call memset: every split
    rewrites its partial and pass 1 its walk bounds on every call)."""
    key = (device, num_tokens, num_heads, num_splits, block_dv)
    ws = _WORKSPACE.get(key)
    if ws is None:
        acc = torch.empty(
            (num_tokens, num_heads, num_splits, block_dv), dtype=torch.float32, device=device
        )
        # part_sum, part_max, and the pass-1 row max (TWO_PASS)
        stats = torch.empty((3, num_tokens, num_heads, num_splits), dtype=torch.float32, device=device)
        # the pass-1 split walks (two passes)
        bounds = torch.zeros((num_tokens, num_splits, 4), dtype=torch.int32, device=device)
        ws = (acc, stats[0], stats[1], stats[2], bounds)
        _WORKSPACE[key] = ws
    return ws


def triton_mla_sparse_fwd(
    q: torch.Tensor,  # [num_tokens, num_heads_q, dim_qk]
    kv: torch.Tensor,  # [num_rows, num_heads_kv, dim_qk]
    indices: torch.Tensor,  # [num_tokens, num_heads_kv, topk] int32
    sm_scale: float,
    d_v: int = 512,
    block_dpe: int | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse MQA attention over ``indices``-selected cache rows.

    Returns ``(out, max_logits, lse)``:

    * ``out``        -- ``[num_tokens, num_heads_q, d_v]`` in ``q.dtype``
    * ``max_logits`` -- ``[num_tokens, num_heads_q]`` fp32
    * ``lse``        -- ``[num_tokens, num_heads_q]`` fp32, natural-log
      logsumexp of the (scaled) logits over the valid indices.

    Args:
        block_dpe: width of the RoPE tail of ``dim_qk`` (``qk_rope_head_dim``).
            Defaults to ``dim_qk - d_v``: 0 for the NoPE 512 layout, 64 for
            the DeepSeek 576 layout.
    """
    num_tokens, num_heads_q, dim_qk = q.shape
    _, num_heads_kv, _ = kv.shape
    assert dim_qk == kv.shape[2], "q and kv have different head dimensions"
    assert indices.dtype == torch.int32
    assert indices.shape[0] == num_tokens
    _, _, index_topk = indices.shape

    if block_dpe is None:
        block_dpe = dim_qk - d_v

    BLOCK_H, BLOCK_N, num_splits, num_warps, num_stages = _pick_config(
        num_tokens, index_topk, num_heads_q, dim_qk, q.device.index or 0
    )
    BLOCK_DPE = block_dpe
    BLOCK_DMODEL = dim_qk - BLOCK_DPE
    BLOCK_DV = d_v
    assert BLOCK_DV & (BLOCK_DV - 1) == 0, "d_v must be a power of two"
    assert BLOCK_DMODEL & (BLOCK_DMODEL - 1) == 0, "dim_nope must be a power of two"
    assert BLOCK_DPE == 0 or BLOCK_DPE & (BLOCK_DPE - 1) == 0, (
        "block_dpe must be 0 or a power of two"
    )
    assert d_v <= BLOCK_DMODEL, "V is read from the leading dims of the latent"
    assert num_heads_kv == 1, "only support kv head = 1 for now"
    assert index_topk > 0

    sm_scale = sm_scale * LOG2E

    kv_group_num = num_heads_q // num_heads_kv
    split_len = _cdiv(_cdiv(index_topk, num_splits), BLOCK_N) * BLOCK_N
    num_splits = _cdiv(index_topk, split_len)
    # merge kernel iterates tl.arange(0, NUM_SPLITS) -> power of two
    num_splits = 1 << (num_splits - 1).bit_length()
    grid = (
        num_tokens,
        _cdiv(num_heads_q, min(BLOCK_H, kv_group_num)),
        num_splits,
    )

    if out is None:
        out = torch.empty(
            (num_tokens, num_heads_q, d_v), dtype=q.dtype, device=q.device
        )
    else:
        assert out.shape == (num_tokens, num_heads_q, d_v)
        assert out.dtype == q.dtype
    # One allocation for both [num_tokens, num_heads_q] fp32 statistics: a
    # torch.empty costs ~2.6 us of host time, which is a fifth of a decode
    # launch. The two returned tensors are ordinary contiguous views.
    stats = torch.empty(
        (2, num_tokens, num_heads_q), dtype=torch.float32, device=q.device
    )
    max_logits, softmax_lse = stats[0], stats[1]

    # kv[..., :d_v] would have kv's pointer and strides anyway (d_v <=
    # BLOCK_DMODEL, and the kernel masks the V columns), so skip the slice.
    k = v = kv

    # Two passes when a row is split across CTAs (NoPE, V = K): pass 1 gives
    # each sparse row its exact logit max (and finishes dense rows outright),
    # so pass 2 can spread a sparse row's valid tiles over every split and
    # still round P against the row max. Only from 4 splits: at 2 (T 24 / 32
    # on 16 heads) the extra pass costs as much as the balance saves (measured).
    two_pass = num_splits >= 4 and BLOCK_DPE == 0 and d_v == BLOCK_DMODEL
    if num_splits == 1:
        # unused; any tensor works as a placeholder pointer
        part_acc = part_sum = part_max = row_max = split_bounds = max_logits
    else:
        part_acc, part_sum, part_max, row_max, split_bounds = _workspace(
            q.device, num_tokens, num_heads_q, num_splits, BLOCK_DV
        )

    sq, skv, so, si = q.stride(), kv.stride(), out.stride(), indices.stride()
    # At the production layout (16-head blocks, NoPE, V = K, 512 dims, 64-key
    # tiles, bf16) every pass runs in Gluon with a double-buffered gather.
    use_gluon = (
        BLOCK_H == 16 and BLOCK_N == 64 and BLOCK_DPE == 0
        and d_v == BLOCK_DMODEL == 512 and q.dtype == kv.dtype == torch.bfloat16
    )
    if use_gluon:
        for pass_id in ((1, 2) if two_pass else (0,)):
            gluon_walk_kernel[grid](
                q, kv, indices, out, softmax_lse, max_logits,
                part_acc, part_sum, part_max, row_max, split_bounds,
                kv.shape[0], num_heads_q, sq[0], sq[1], skv[0], so[0], so[1], si[0],
                sm_scale, split_len,
                index_topk=index_topk, NUM_SPLITS=num_splits,
                ROW_W=1 << (index_topk - 1).bit_length(),
                BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, D=BLOCK_DMODEL, LOGE2=LOGE2, PASS=pass_id,
                num_warps=num_warps,
            )
    else:
        for pass_id in ((1, 2) if two_pass else (0,)):
            _launch_pass(
                grid, pass_id, q, k, v, indices, out, softmax_lse, part_acc, part_sum,
                part_max, row_max, split_bounds, max_logits, kv.shape[0], num_heads_q,
                dim_qk, d_v, sq, skv, so, si, sm_scale, split_len, kv_group_num, index_topk,
                num_splits, BLOCK_H, BLOCK_N, BLOCK_DV, BLOCK_DMODEL, BLOCK_DPE, num_warps,
                num_stages,
            )
    if num_splits > 1:
        # One CTA per (row, head block, 64 output columns): the partials are
        # read by many SMs at once rather than by one CTA per row.
        merge_bh = min(BLOCK_H, 16)
        _mla_merge_kernel[(num_tokens, _cdiv(num_heads_q, merge_bh), _cdiv(BLOCK_DV, 64))](
            out, softmax_lse, max_logits, part_acc, part_sum, part_max, split_bounds,
            num_heads_q, d_v, so[0], so[1], num_heads_q,
            NUM_SPLITS=num_splits, BLOCK_H=merge_bh, BLOCK_DV=BLOCK_DV, BLOCK_DC=64,
            LOGE2=LOGE2, CHECK_DIRECT=not two_pass, num_warps=4,
        )

    return out, max_logits, softmax_lse


def _launch_pass(
    grid, pass_id, q, k, v, indices, out, softmax_lse, part_acc, part_sum, part_max,
    row_max, split_bounds, max_logits, seq_kv, num_heads_q, dim_qk, d_v,
    sq, skv, so, si, sm_scale, split_len, kv_group_num, index_topk, num_splits,
    BLOCK_H, BLOCK_N, BLOCK_DV, BLOCK_DMODEL, BLOCK_DPE, num_warps, num_stages,
):
    _mla_sparse_kernel[grid](
        q_buffer=q,
        k_buffer=k,
        v_buffer=v,
        indices_ptr=indices,
        out_ptr=out,
        softmax_lse_ptr=softmax_lse,
        part_acc_ptr=part_acc,
        part_sum_ptr=part_sum,
        part_max_ptr=part_max,
        row_max_ptr=row_max,
        split_bounds_ptr=split_bounds,
        max_logits_ptr=max_logits,
        seq_kv=seq_kv,
        h_q=num_heads_q,
        dim_qk=dim_qk,
        dim_v=d_v,
        stride_q_token=sq[0],
        stride_q_head=sq[1],
        stride_k_token=skv[0],
        stride_k_head=skv[1],
        stride_v_token=skv[0],
        stride_v_head=skv[1],
        stride_out_token=so[0],
        stride_out_head=so[1],
        stride_lse=num_heads_q,  # a contiguous [num_tokens, num_heads_q] row
        stride_indices_token=si[0],
        stride_indices_head=si[1],
        sm_scale=sm_scale,
        split_len=split_len,
        kv_group_num=kv_group_num,
        index_topk=index_topk,
        NUM_SPLITS=num_splits,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        BLOCK_DV=BLOCK_DV,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        V_IS_K=(d_v == BLOCK_DMODEL),
        LOGE2=LOGE2,
        PASS=pass_id,
        ROW_W=1 << (index_topk - 1).bit_length(),
        num_warps=num_warps,
        num_stages=num_stages,
    )
