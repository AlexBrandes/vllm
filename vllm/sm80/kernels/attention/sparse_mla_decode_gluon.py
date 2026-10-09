# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gluon tile walk of the sm_80 sparse MLA decode v2 for the NoPE (D 512, V = K) layout at
16 heads per CTA (Triton 3.7 ``triton.experimental.gluon``; launched by
``triton_mla_sparse_decode_v2.py``).

Same contract as ``_mla_sparse_kernel`` (triton_mla_sparse_decode_v2.py), pass for pass;
results are bitwise those of the Triton walk:

* PASS 0 (single pass): one CTA per (row, head block, split) scans the index row on the
  device and walks the split's columns of the row's valid range. It stores the output (an
  unsplit launch, or a row whose valid entries one split owns; flag in
  ``split_bounds[t, 0, 3]``) or the split's partial for ``_mla_merge_kernel``.
* PASS 1 (two passes): scans the row, stores every split's walk ``(start, end, use_max,
  use_max)`` in ``split_bounds``; a dense row does the single-pass job on its column split
  (partial), a sparse row walks its balanced share of the valid tiles for the max logit
  (``row_max``).
* PASS 2: sparse rows only. The walk starts from the exact row max (the max of the pass-1
  split maxima), so P is rounded against it like the reference; it stores the partial.

Why Gluon: Triton keeps the gathered tile in one shared buffer and waits for its gather
before computing on it, so every tile pays gather + compute back to back. Here the gathered
rows live in two shared buffers: the masked cp.async gather of tile i + 1 is issued before
tile i is computed, and the index tile it needs was loaded one iteration earlier.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp


@gluon.jit
def _gl_store_block(acc, offs_c, e_sum, e_max, out_ptr, lse_ptr, mlog_ptr, part_acc_ptr,
                    part_sum_ptr, part_max_ptr, cur_q, cur_head, cur_split, mask_h, h_q,
                    stride_out_token, stride_out_head, direct,
                    NUM_SPLITS: gl.constexpr, BLOCK_DV: gl.constexpr, LOGE2: gl.constexpr,
                    FIRST: gl.constexpr):
    """Store one 128-column block of the split's partial, or (unsplit / owned row) of the
    output; the FIRST block also stores the row statistics."""
    if (NUM_SPLITS > 1) & (direct == 0):
        row = cur_q * h_q + cur_head
        offs_p = (row[:, None] * NUM_SPLITS + cur_split) * BLOCK_DV + offs_c[None, :]
        gl.store(part_acc_ptr + offs_p, acc, mask=mask_h[:, None])
        if FIRST:
            offs_ps = row * NUM_SPLITS + cur_split
            gl.store(part_sum_ptr + offs_ps, e_sum, mask=mask_h)
            gl.store(part_max_ptr + offs_ps, e_max, mask=mask_h)
    else:
        s = gl.where(e_sum > 0.0, e_sum, 1.0)
        o = acc / s[:, None]
        offs_o = cur_q * stride_out_token + cur_head[:, None] * stride_out_head + offs_c[None, :]
        gl.store(out_ptr + offs_o, o.to(out_ptr.dtype.element_ty), mask=mask_h[:, None])
        if FIRST:
            ml = e_max * LOGE2
            lse = ml + gl.log2(s) * LOGE2
            gl.store(lse_ptr + cur_q * h_q + cur_head, lse, mask=mask_h)
            gl.store(mlog_ptr + cur_q * h_q + cur_head, ml, mask=mask_h)


@gluon.jit
def _gl_walk(q_ptr, k_ptr, idx_row_ptr, cur_q, cur_head_id, mask_h, e_max,
             loop_start, loop_end, seq_kv, h_q, stride_q_token, stride_q_head, stride_k_token,
             sm_scale, index_topk: gl.constexpr, BLOCK_H: gl.constexpr, BLOCK_N: gl.constexpr,
             D: gl.constexpr, QK_ONLY: gl.constexpr):
    """Online-softmax walk of tiles [loop_start, loop_end) from the running max e_max with a
    double-buffered gather. QK_ONLY: only the running max of the logits (pass 1)."""
    NW: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, NW], instr_shape=[16, 8])
    a_op: gl.constexpr = gl.DotOperandLayout(parent=mma, operand_index=0, k_width=2)
    b_op: gl.constexpr = gl.DotOperandLayout(parent=mma, operand_index=1, k_width=2)
    gat: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [NW, 1], [1, 0])
    smem_l: gl.constexpr = gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0])
    col_l: gl.constexpr = gl.SliceLayout(0, mma)
    rowh_l: gl.constexpr = gl.SliceLayout(1, mma)
    DQ: gl.constexpr = 64  # QK k-chunk
    DC: gl.constexpr = 128  # PV column block

    n_tiles = gl.maximum((loop_end - loop_start + BLOCK_N - 1) // BLOCK_N, 0)

    # q -> shared (A operand chunks are read per k-chunk)
    qb: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [NW, 1], [1, 0])
    q_rows = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, qb))
    q_cols = gl.arange(0, D, layout=gl.SliceLayout(0, qb))
    q = gl.load(q_ptr + cur_q * stride_q_token + q_rows[:, None] * stride_q_head + q_cols[None, :],
                mask=(q_rows < h_q)[:, None], other=0.0)
    q_s = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_H, D], smem_l, q)
    k_s = gl.allocate_shared_memory(gl.bfloat16, [2, BLOCK_N, D], smem_l)

    g_rows = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, gat))
    g_cols = gl.arange(0, D, layout=gl.SliceLayout(0, gat))
    c_rows = gl.arange(0, BLOCK_N, layout=col_l)

    # prologue: tile 0 gather, tile 1 indices
    offs0 = loop_start + g_rows
    idx0 = gl.load(idx_row_ptr + offs0, mask=(offs0 < index_topk) & (n_tiles > 0), other=-1)
    m0 = (idx0 >= 0) & (idx0 < seq_kv)
    cp.async_copy_global_to_shared(
        k_s.index(0), k_ptr + idx0[:, None].to(gl.int64) * stride_k_token + g_cols[None, :],
        mask=m0[:, None])
    cp.commit_group()
    offs1 = loop_start + BLOCK_N + g_rows
    idx_next = gl.load(idx_row_ptr + offs1, mask=(offs1 < index_topk) & (n_tiles > 1), other=-1)

    e_sum = gl.zeros([BLOCK_H], gl.float32, layout=rowh_l)
    acc0 = gl.zeros([BLOCK_H, DC], gl.float32, layout=mma)
    acc1 = gl.zeros([BLOCK_H, DC], gl.float32, layout=mma)
    acc2 = gl.zeros([BLOCK_H, DC], gl.float32, layout=mma)
    acc3 = gl.zeros([BLOCK_H, DC], gl.float32, layout=mma)
    for i in range(n_tiles):
        start = loop_start + i * BLOCK_N
        # issue gather(i + 1) into the other buffer (masked off past the walk)
        mn = (idx_next >= 0) & (idx_next < seq_kv)
        cp.async_copy_global_to_shared(
            k_s.index((i + 1) % 2),
            k_ptr + idx_next[:, None].to(gl.int64) * stride_k_token + g_cols[None, :],
            mask=mn[:, None])
        cp.commit_group()
        # indices of tile i + 2 (consumed next iteration)
        offs2 = start + 2 * BLOCK_N + g_rows
        idx_next = gl.load(idx_row_ptr + offs2, mask=(offs2 < index_topk) & (i + 2 < n_tiles),
                           other=-1)
        # this tile's validity in the qk column layout
        offs_c = start + c_rows
        idx_c = gl.load(idx_row_ptr + offs_c, mask=offs_c < index_topk, other=-1)
        mask_kv = (idx_c >= 0) & (idx_c < seq_kv)

        cp.wait_group(1)
        gl.barrier()
        kb = k_s.index(i % 2)
        qk = gl.zeros([BLOCK_H, BLOCK_N], gl.float32, layout=mma)
        for kk in gl.static_range(D // DQ):
            a = q_s.slice(kk * DQ, DQ, dim=1).load(a_op)
            b = kb.slice(kk * DQ, DQ, dim=1).permute((1, 0)).load(b_op)
            qk = mma_v2(a, b, qk)
        qk = qk * sm_scale
        qk = gl.where(mask_h[:, None] & mask_kv[None, :], qk, -1.0e30)
        if QK_ONLY:
            e_max = gl.maximum(e_max, gl.max(qk, 1))
        else:
            n_e_max = gl.maximum(gl.max(qk, 1), e_max)
            re_scale = gl.exp2(e_max - n_e_max)
            p = gl.exp2(qk - n_e_max[:, None])
            pa = gl.convert_layout(p.to(gl.bfloat16), a_op)
            acc0 = mma_v2(pa, kb.slice(0 * DC, DC, dim=1).load(b_op), acc0 * re_scale[:, None])
            acc1 = mma_v2(pa, kb.slice(1 * DC, DC, dim=1).load(b_op), acc1 * re_scale[:, None])
            acc2 = mma_v2(pa, kb.slice(2 * DC, DC, dim=1).load(b_op), acc2 * re_scale[:, None])
            acc3 = mma_v2(pa, kb.slice(3 * DC, DC, dim=1).load(b_op), acc3 * re_scale[:, None])
            e_sum = e_sum * re_scale + gl.sum(p, 1)
            e_max = n_e_max
        gl.barrier()
    cp.wait_group(0)
    return e_max, e_sum, acc0, acc1, acc2, acc3


@gluon.jit
def _gl_publish(e_max, e_sum, acc0, acc1, acc2, acc3, out_ptr, lse_ptr, mlog_ptr,
                part_acc_ptr, part_sum_ptr, part_max_ptr, cur_q, cur_head, cur_split, mask_h,
                h_q, stride_out_token, stride_out_head, direct,
                NUM_SPLITS: gl.constexpr, D: gl.constexpr, LOGE2: gl.constexpr):
    NW: gl.constexpr = gl.num_warps()
    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, NW], instr_shape=[16, 8])
    DC: gl.constexpr = 128
    offs_cc = gl.arange(0, DC, layout=gl.SliceLayout(0, mma))
    _gl_store_block(acc0, 0 * DC + offs_cc, e_sum, e_max, out_ptr, lse_ptr, mlog_ptr, part_acc_ptr,
                    part_sum_ptr, part_max_ptr, cur_q, cur_head, cur_split, mask_h, h_q,
                    stride_out_token, stride_out_head, direct, NUM_SPLITS, D, LOGE2, True)
    _gl_store_block(acc1, 1 * DC + offs_cc, e_sum, e_max, out_ptr, lse_ptr, mlog_ptr, part_acc_ptr,
                    part_sum_ptr, part_max_ptr, cur_q, cur_head, cur_split, mask_h, h_q,
                    stride_out_token, stride_out_head, direct, NUM_SPLITS, D, LOGE2, False)
    _gl_store_block(acc2, 2 * DC + offs_cc, e_sum, e_max, out_ptr, lse_ptr, mlog_ptr, part_acc_ptr,
                    part_sum_ptr, part_max_ptr, cur_q, cur_head, cur_split, mask_h, h_q,
                    stride_out_token, stride_out_head, direct, NUM_SPLITS, D, LOGE2, False)
    _gl_store_block(acc3, 3 * DC + offs_cc, e_sum, e_max, out_ptr, lse_ptr, mlog_ptr, part_acc_ptr,
                    part_sum_ptr, part_max_ptr, cur_q, cur_head, cur_split, mask_h, h_q,
                    stride_out_token, stride_out_head, direct, NUM_SPLITS, D, LOGE2, False)


@gluon.jit
def gluon_walk_kernel(
    q_ptr, k_ptr, idx_ptr, out_ptr, lse_ptr, mlog_ptr,
    part_acc_ptr, part_sum_ptr, part_max_ptr, row_max_ptr, split_bounds_ptr,
    seq_kv, h_q, stride_q_token, stride_q_head, stride_k_token,
    stride_out_token, stride_out_head, stride_idx_token,
    sm_scale, split_len,
    index_topk: gl.constexpr, NUM_SPLITS: gl.constexpr, ROW_W: gl.constexpr,
    BLOCK_H: gl.constexpr, BLOCK_N: gl.constexpr, D: gl.constexpr, LOGE2: gl.constexpr,
    PASS: gl.constexpr,
):
    gl.static_assert(BLOCK_H == 16)
    NW: gl.constexpr = gl.num_warps()
    cur_q = gl.program_id(0)
    cur_head_id = gl.program_id(1)
    cur_split = gl.program_id(2)

    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, NW], instr_shape=[16, 8])
    row1d: gl.constexpr = gl.BlockedLayout([4], [32], [NW], [0])
    rowh_l: gl.constexpr = gl.SliceLayout(1, mma)

    cur_head = cur_head_id * BLOCK_H + gl.arange(0, BLOCK_H, layout=rowh_l)
    mask_h = cur_head < h_q
    idx_row_ptr = idx_ptr + cur_q * stride_idx_token
    # Finite sentinel, not -inf: an all-masked tile must not give exp2(-inf - -inf) = NaN.
    e_max0 = gl.full([BLOCK_H], -1.0e30, gl.float32, layout=rowh_l)

    if PASS == 2:
        split_ptr = split_bounds_ptr + (cur_q * NUM_SPLITS + cur_split) * 4
        use_max = gl.load(split_ptr + 2) != 0
        if use_max:
            loop_start = gl.load(split_ptr)
            loop_end = gl.load(split_ptr + 1)
            e_max = e_max0
            m_ptr = row_max_ptr + (cur_q * h_q + cur_head) * NUM_SPLITS
            for i in gl.static_range(NUM_SPLITS):
                e_max = gl.maximum(e_max, gl.load(m_ptr + i, mask=mask_h, other=-1.0e30))
            e_max, e_sum, acc0, acc1, acc2, acc3 = _gl_walk(
                q_ptr, k_ptr, idx_row_ptr, cur_q, cur_head_id, mask_h, e_max,
                loop_start, loop_end, seq_kv, h_q, stride_q_token, stride_q_head, stride_k_token,
                sm_scale, index_topk, BLOCK_H, BLOCK_N, D, False)
            _gl_publish(e_max, e_sum, acc0, acc1, acc2, acc3, out_ptr, lse_ptr, mlog_ptr,
                        part_acc_ptr, part_sum_ptr, part_max_ptr, cur_q, cur_head, cur_split,
                        mask_h, h_q, stride_out_token, stride_out_head, False, NUM_SPLITS, D, LOGE2)
    else:
        # one load of the whole index row: first / last valid column (any layout)
        offs_row = gl.arange(0, ROW_W, layout=row1d)
        idx_row = gl.load(idx_row_ptr + offs_row, mask=offs_row < index_topk, other=-1)
        valid_row = (idx_row >= 0) & (idx_row < seq_kv)
        first_valid = gl.min(gl.where(valid_row, offs_row, ROW_W), 0)
        last_valid = gl.max(gl.where(valid_row, offs_row, -1), 0)
        # the split's columns of the valid range, on the split's tile grid (dense rows):
        # a skipped all-masked leading tile only fed e_sum what the first valid tile's
        # re_scale = 0 wipes, a trailing one adds p = 0; a -1 gap inside is walked
        split_start = cur_split * split_len
        split_end = gl.minimum(split_start + split_len, index_topk)
        lo = gl.maximum(first_valid, split_start)
        hi = gl.minimum(last_valid, split_end - 1)
        col_start = split_start + ((lo - split_start) // BLOCK_N) * BLOCK_N
        col_end = hi + 1
        if PASS == 0:
            if NUM_SPLITS > 1:
                owned = (first_valid // split_len) == (last_valid // split_len)
                direct = owned & (first_valid // split_len == cur_split)
                if (cur_split == 0) & (cur_head_id == 0):
                    gl.store(split_bounds_ptr + cur_q * NUM_SPLITS * 4 + 3, owned.to(gl.int32))
                run = (owned == 0) | direct
            else:
                direct = False
                run = True
            if run:
                e_max, e_sum, acc0, acc1, acc2, acc3 = _gl_walk(
                    q_ptr, k_ptr, idx_row_ptr, cur_q, cur_head_id, mask_h, e_max0,
                    col_start, col_end, seq_kv, h_q, stride_q_token, stride_q_head,
                    stride_k_token, sm_scale, index_topk, BLOCK_H, BLOCK_N, D, False)
                _gl_publish(e_max, e_sum, acc0, acc1, acc2, acc3, out_ptr, lse_ptr, mlog_ptr,
                            part_acc_ptr, part_sum_ptr, part_max_ptr, cur_q, cur_head, cur_split,
                            mask_h, h_q, stride_out_token, stride_out_head, direct, NUM_SPLITS,
                            D, LOGE2)
        else:
            # PASS 1. Sparse row (valid range spans at most half the row's tiles): its
            # tiles are dealt out evenly over the splits; this split's max logit goes
            # to row_max. Dense row: the single-pass job on the column split.
            first_tile = first_valid // BLOCK_N
            num_tiles = gl.maximum(last_valid // BLOCK_N + 1 - first_tile, 0)
            use_max = num_tiles * 2 <= (index_topk + BLOCK_N - 1) // BLOCK_N
            tiles_per_split = (num_tiles + NUM_SPLITS - 1) // NUM_SPLITS
            bal_start = (first_tile + cur_split * tiles_per_split) * BLOCK_N
            bal_end = gl.minimum(bal_start + tiles_per_split * BLOCK_N, last_valid + 1)
            loop_start = gl.where(use_max, bal_start, col_start)
            loop_end = gl.where(use_max, bal_end, col_end)
            if cur_head_id == 0:
                bl: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
                offs_b = gl.arange(0, 4, layout=bl)
                bounds = gl.where(offs_b == 0, loop_start,
                                  gl.where(offs_b == 1, loop_end, use_max.to(gl.int32)))
                gl.store(split_bounds_ptr + (cur_q * NUM_SPLITS + cur_split) * 4 + offs_b, bounds)
            if use_max:
                e_max, e_sum, acc0, acc1, acc2, acc3 = _gl_walk(
                    q_ptr, k_ptr, idx_row_ptr, cur_q, cur_head_id, mask_h, e_max0,
                    loop_start, loop_end, seq_kv, h_q, stride_q_token, stride_q_head,
                    stride_k_token, sm_scale, index_topk, BLOCK_H, BLOCK_N, D, True)
                gl.store(row_max_ptr + (cur_q * h_q + cur_head) * NUM_SPLITS + cur_split, e_max,
                         mask=mask_h)
            else:
                e_max, e_sum, acc0, acc1, acc2, acc3 = _gl_walk(
                    q_ptr, k_ptr, idx_row_ptr, cur_q, cur_head_id, mask_h, e_max0,
                    loop_start, loop_end, seq_kv, h_q, stride_q_token, stride_q_head,
                    stride_k_token, sm_scale, index_topk, BLOCK_H, BLOCK_N, D, False)
                _gl_publish(e_max, e_sum, acc0, acc1, acc2, acc3, out_ptr, lse_ptr, mlog_ptr,
                            part_acc_ptr, part_sum_ptr, part_max_ptr, cur_q, cur_head, cur_split,
                            mask_h, h_q, stride_out_token, stride_out_head, False, NUM_SPLITS,
                            D, LOGE2)
