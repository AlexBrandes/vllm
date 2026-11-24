# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch, bit-exact replacements for ``sm80_attn_core.rope`` and ``sm80_attn_core.rms_norm``
(``VLLM_DEEPSEEK_V41_EXACT_GLUE=1``).

The torch reference forms launch ~10 kernels per RoPE call (FP32 angles, ``polar``, ``conj``, complex views, the
complex64 multiply, the BF16 cast, a clone and a slice copy) and ~6 per RMSNorm call; on the served PP3 decode
path that is ~49 small kernels per layer. These kernels compute the same bytes in one launch each:

* RoPE: ``sm80_attn_pipeline._rope_row``'s arithmetic (FP32 angle ``pos * freq``, libdevice ``cos`` / ``sin``,
  the complex64 multiply as CUDA compiles c10::complex: ``fma(re, cos, -(im * sin))`` /
  ``fma(re, sin, im * cos)``), the inverse rotation by the conjugate (``sin`` negated, exact), rounded once to the
  input dtype; the non-rotated head of the row is copied unchanged.
* RMSNorm: ``sm80_draft_prologue.rms_norm_row`` (squares rounded one by one, summed in the order of torch's CUDA
  ``mean(-1)`` for this ``[rows, width]`` shape, times ``1 / width``, ``+ eps``, ``rsqrt``, the products with the
  row and the weight), rounded to BF16; widths that are powers of two from 128 to 2048 (others keep torch).

Compiled without FMA contraction (as nvcc builds torch's elementwise kernels), so only the explicit ``fma`` of
the complex multiply fuses. No host sync; launches capture in CUDA graphs.
"""
import os

import torch

from vllm.triton_utils import tl, triton
from vllm_cmp170hx.models.deepseek_v41.ops.sm80_draft_prologue import _fold, mean_block_width, rms_norm_row
from triton.language.extra import libdevice


def enabled() -> bool:
    return os.environ.get("VLLM_DEEPSEEK_V41_EXACT_GLUE") == "1"


@triton.jit
def _rope_kernel(x_ptr, pos_ptr, freqs_ptr, out_ptr, M, sx_t, sx_m, so_t, so_m,
                 D: tl.constexpr, RD: tl.constexpr, INVERSE: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    t = pid // M
    m = pid % M
    d = tl.arange(0, D)
    x = tl.load(x_ptr + t * sx_t + m * sx_m + d)
    r = tl.arange(0, RD)
    a = tl.gather(x, D - RD + r, 0).to(tl.float32)
    b = tl.gather(x, D - RD + (r ^ 1), 0).to(tl.float32)
    angle = tl.load(pos_ptr + t).to(tl.float32) * tl.load(freqs_ptr + r // 2)
    sine = libdevice.sin(angle)
    if INVERSE:
        sine = -sine
    cosine = libdevice.cos(angle)
    rotated = tl.where(r % 2 == 0,
                       libdevice.fma(a, cosine, -libdevice.mul_rn(b, sine)),
                       libdevice.fma(b, sine, libdevice.mul_rn(a, cosine)))
    rotated = rotated.to(x.dtype)
    out = tl.where(d >= D - RD, tl.gather(rotated, tl.maximum(d - (D - RD), 0), 0), x)
    tl.store(out_ptr + t * so_t + m * so_m + d, out)


def rope(x: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor, inverse: bool = False):
    """``sm80_attn_core.rope`` in one launch, or None when this shape is not served."""
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16) or x.dim() < 2:
        return None
    d = x.shape[-1]
    rd = 2 * freqs.numel()
    if d & (d - 1) or rd & (rd - 1) or rd > d or x.stride(-1) != 1 or positions.shape[0] != x.shape[0]:
        return None
    t = x.shape[0]
    try:
        x3 = x.view(t, -1, d)
    except RuntimeError:
        return None
    m = x3.shape[1]
    out = torch.empty_like(x)
    o3 = out.view(t, -1, d)
    if t * m:
        _rope_kernel[(t * m,)](x3, positions, freqs.to(torch.float32), o3, m, x3.stride(0), x3.stride(1),
                               o3.stride(0), o3.stride(1), D=d, RD=rd, INVERSE=bool(inverse),
                               num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _rope_tail_kernel(x_ptr, pos_ptr, freqs_ptr, M, sx_t, sx_m,
                      D: tl.constexpr, RD: tl.constexpr, BM: tl.constexpr, INVERSE: tl.constexpr):
    """In-place ``_rope_kernel`` on the rotated tail only: program (t, BM heads); pair j of each head is
    (even, odd) = (x[D - RD + 2j], x[D - RD + 2j + 1]). The per-element operations are ``_rope_kernel``'s
    (angle = fp32(position) * freq, libdevice sin / cos, even' = fma(e, cos, -(o * sin)), odd' =
    fma(e, sin, o * cos), one rounding to x's dtype), so the bytes are the same; the untouched head
    part is not copied."""
    t = tl.program_id(0).to(tl.int64)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    j = tl.arange(0, RD // 2)
    mask = (m < M)[:, None]
    ptr = x_ptr + t * sx_t + m.to(tl.int64)[:, None] * sx_m + (D - RD) + 2 * j[None, :]
    xe = tl.load(ptr, mask=mask)
    e = xe.to(tl.float32)
    o = tl.load(ptr + 1, mask=mask).to(tl.float32)
    angle = tl.load(pos_ptr + t).to(tl.float32) * tl.load(freqs_ptr + j)
    sine = libdevice.sin(angle)
    if INVERSE:
        sine = -sine
    cosine = libdevice.cos(angle)
    ne = libdevice.fma(e, cosine[None, :], -libdevice.mul_rn(o, sine[None, :]))
    no = libdevice.fma(e, sine[None, :], libdevice.mul_rn(o, cosine[None, :]))
    tl.store(ptr, ne.to(xe.dtype), mask=mask)
    tl.store(ptr + 1, no.to(xe.dtype), mask=mask)


def rope_(x: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor, inverse: bool = False):
    """:func:`rope` in place on ``x``'s rotated tail (bitwise the same result; the caller must not need
    the unrotated ``x``), or None when this shape is not served."""
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16) or x.dim() < 2:
        return None
    d = x.shape[-1]
    rd = 2 * freqs.numel()
    if d & (d - 1) or rd & (rd - 1) or rd > d or x.stride(-1) != 1 or positions.shape[0] != x.shape[0]:
        return None
    t = x.shape[0]
    try:
        x3 = x.view(t, -1, d)
    except RuntimeError:
        return None
    m = x3.shape[1]
    if t * m:
        bm = 16
        _rope_tail_kernel[(t, triton.cdiv(m, bm))](
            x3, positions, freqs.to(torch.float32), m, x3.stride(0), x3.stride(1),
            D=d, RD=rd, BM=bm, INVERSE=bool(inverse), num_warps=4, enable_fp_fusion=False)
    return x


@triton.jit
def _rms_kernel(x_ptr, w_ptr, out_ptr, eps, sx, so, D: tl.constexpr, BW: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, D)
    x = tl.load(x_ptr + t * sx + d).to(tl.float32)
    w = tl.load(w_ptr + d).to(tl.float32)
    tl.store(out_ptr + t * so + d, rms_norm_row(x, w, eps, D, BW))


def _mean_block_width_any(rows: int, width: int) -> int:
    """``sm80_draft_prologue.mean_block_width`` for any width that is a multiple of 4 and at least 128 (torch's
    ``ReduceConfig.set_block_dimension`` on ``dim0 = width / 4`` vectors, ``dim1 = rows``)."""
    dim0 = width // 4
    last = lambda n: 1 << (n.bit_length() - 1)  # noqa: E731
    dim0_pow2 = last(dim0) if dim0 < 512 else 512
    dim1_pow2 = last(rows) if rows < 512 else 512
    block_width = min(dim0_pow2, 32)
    block_height = min(dim1_pow2, 512 // block_width)
    return min(dim0_pow2, 512 // block_height)


@triton.jit
def _rms_pad_kernel(x_ptr, w_ptr, out_ptr, eps, sx, so, D: tl.constexpr, K: tl.constexpr, BW: tl.constexpr):
    """``_rms_kernel`` for a width ``D`` that is not a power of two (or needs more than 4 vectors per lane):
    torch's lane ``x`` adds its vectors ``x, x + BW, ...`` (of 4) into 4 accumulators in that order; here chunk
    ``k`` of ``4 BW`` values is vector ``k BW + x`` for lane ``x``, read in order, values past ``D`` read as 0
    (a lane without that vector adds +0.0 to a non-negative sum: the same value). Then the 4 accumulators and the
    lane folds of ``torch_cuda_mean_sum``, ``* (1 / D)``, ``rsqrt``, the products; BF16."""
    t = tl.program_id(0).to(tl.int64)
    M: tl.constexpr = 4 * BW
    acc = tl.zeros([M], dtype=tl.float32)
    for k in tl.static_range(K):
        offs = k * M + tl.arange(0, M)
        xv = tl.load(x_ptr + t * sx + offs, mask=offs < D, other=0.0).to(tl.float32)
        acc = acc + libdevice.mul_rn(xv, xv)
    even, odd = tl.split(tl.reshape(acc, (BW, 2, 2)))
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    lane = ((a0 + a1) + a2) + a3
    if BW >= 512:
        lane = _fold(lane, 256)
    if BW >= 256:
        lane = _fold(lane, 128)
    if BW >= 128:
        lane = _fold(lane, 64)
    if BW >= 64:
        lane = _fold(lane, 32)
    lane = _fold(lane, 16)
    lane = _fold(lane, 8)
    lane = _fold(lane, 4)
    lane = _fold(lane, 2)
    total = _fold(lane, 1)
    mean = total * (1.0 / D)
    inv = libdevice.rsqrt(mean + eps)
    for k in tl.static_range(K):
        offs = k * M + tl.arange(0, M)
        m = offs < D
        xv = tl.load(x_ptr + t * sx + offs, mask=m, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
        y = (wv * (xv * inv)).to(tl.bfloat16)
        tl.store(out_ptr + t * so + offs, tl.reshape(y, (M,)), mask=m)


_RMS_ANY = os.environ.get("VLLM_DEEPSEEK_V41_RMS_ANY", "0") == "1"


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """``sm80_attn_core.rms_norm`` in one launch for BF16 rows of a power-of-two width (128..2048), else None
    (``VLLM_DEEPSEEK_V41_RMS_ANY=1``: any width that is a multiple of 4 from 128 to 8192)."""
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.dim() != 2 or x.stride(-1) != 1:
        return None
    w = x.shape[1]
    if _RMS_ANY and weight.numel() == w and weight.is_contiguous() and w % 4 == 0 and 128 <= w <= 8192 and (
            w & (w - 1) or w > 2048):
        t = x.shape[0]
        out = torch.empty((t, w), dtype=torch.bfloat16, device=x.device)
        if t:
            bw = _mean_block_width_any(t, w)
            k = triton.cdiv(w, 4 * bw)
            _rms_pad_kernel[(t,)](x, weight, out, float(eps), x.stride(0), out.stride(0), D=w, K=k, BW=bw,
                                  num_warps=4, enable_fp_fusion=False)
        return out
    if w < 128 or w > 2048 or w & (w - 1) or weight.numel() != w or not weight.is_contiguous():
        return None
    t = x.shape[0]
    out = torch.empty((t, w), dtype=torch.bfloat16, device=x.device)
    if t:
        bw = mean_block_width(t, w)
        if bw < 32 or bw > 128 or w // (4 * bw) > 4:
            return None
        _rms_kernel[(t,)](x, weight, out, float(eps), x.stride(0), out.stride(0), D=w, BW=bw,
                          num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _compress2_kernel(kv_ptr, ring_ptr, pos_ptr, req_ptr, qsl_ptr, comp_ptr, rslot_ptr, w_ptr, closes_ptr,
                      out_ptr, eps, s_kv, s_rb, s_re, s_out, CAP, D: tl.constexpr, BW: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, D)
    pos = tl.load(pos_ptr + t).to(tl.int64)
    req = tl.load(req_ptr + t).to(tl.int64)
    first = t == tl.load(qsl_ptr + req).to(tl.int64)
    cslot = tl.load(comp_ptr + t)
    tl.store(closes_ptr + t, ((cslot >= 0) & ((pos + 1) % 2 == 0)).to(tl.int8))
    rs = tl.maximum(tl.load(rslot_ptr + t).to(tl.int64), 0)
    re = ((pos - 1) % CAP + CAP) % CAP  # torch's floor modulo
    ring_row = ring_ptr + (rs // CAP) * s_rb + re * s_re
    prev_row = kv_ptr + tl.maximum(t - 1, 0) * s_kv
    cur_row = kv_ptr + t * s_kv
    pk = tl.where(first, tl.load(ring_row + d), tl.load(prev_row + d))
    ps = tl.where(first, tl.load(ring_row + D + d), tl.load(prev_row + D + d))
    ck = tl.load(cur_row + d)
    cs = tl.load(cur_row + D + d)
    # softmax over the pair: exact max, exp(x - max), sum of two (order-free), exp(x - max) / sum
    m = tl.maximum(ps, cs)
    e0 = libdevice.exp(ps - m)
    e1 = libdevice.exp(cs - m)
    s = e0 + e1
    p0 = libdevice.div_rn(e0, s)
    p1 = libdevice.div_rn(e1, s)
    pooled = libdevice.mul_rn(pk, p0) + libdevice.mul_rn(ck, p1)
    x = pooled.to(tl.bfloat16).to(tl.float32)
    w = tl.load(w_ptr + d).to(tl.float32)
    tl.store(out_ptr + t * s_out + d, rms_norm_row(x, w, eps, D, BW))


def compress2(kv_score, norm_weight, ring, positions, token_to_req, query_start_loc, comp_slots, ring_slots,
              head_dim, eps):
    """``sm80_attn_core.compress_rows``' ratio-2 pooling and norm in one launch: (closes, latent), or None when
    not served (the ring-tail store stays with the caller)."""
    d = head_dim
    t = positions.shape[0]
    if (not kv_score.is_cuda or kv_score.dtype != torch.float32 or ring is None or ring.dtype != torch.float32
            or kv_score.stride(-1) != 1 or ring.stride(-1) != 1 or d & (d - 1) or d < 128 or d > 2048
            or norm_weight.numel() != d or t == 0):
        return None
    bw = mean_block_width(t, d)
    if bw < 32 or bw > 128 or d // (4 * bw) > 4:
        return None
    closes = torch.empty(t, dtype=torch.int8, device=kv_score.device)
    latent = torch.empty((t, d), dtype=torch.bfloat16, device=kv_score.device)
    _compress2_kernel[(t,)](kv_score, ring, positions, token_to_req, query_start_loc, comp_slots, ring_slots,
                            norm_weight, closes, latent, float(eps), kv_score.stride(0), ring.stride(0),
                            ring.stride(1), latent.stride(0), ring.shape[1], D=d, BW=bw, num_warps=4,
                            enable_fp_fusion=False)
    return closes.bool(), latent


@triton.jit
def _block_max_kernel(s_ptr, nvis_ptr, out_ptr, n_cols, nb, s_s, s_o, BS: tl.constexpr, NB: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1) * NB + tl.arange(0, NB)
    j = tl.arange(0, BS)
    cols = b[:, None] * BS + j[None, :]
    v = tl.load(s_ptr + row * s_s + cols, mask=cols < n_cols, other=float("-inf"))
    m = tl.max(v, axis=1)
    nv = tl.load(nvis_ptr + row).to(tl.int64)
    last = tl.where(nv > 0, (nv - 1) // BS, -1)
    m = tl.where(b == last, float("inf"), m)
    tl.store(out_ptr + row * s_o + b, m, mask=b < nb)


def select_candidate_blocks(scores, n_vis, topk_blocks, block_size):
    """``sm80_attn_core.select_candidate_blocks`` (block maxima, the newest visible block pinned, canonical top
    blocks ascending, -1 padded) as a block-max launch + the pipeline's stable radix selection; None if not
    served."""
    from vllm_cmp170hx.models.deepseek_v41.ops import sm80_index_topk as itk
    n, n_cols = scores.shape
    if (not scores.is_cuda or scores.dtype != torch.float32 or scores.stride(1) != 1 or n == 0
            or block_size & (block_size - 1) or topk_blocks <= 0):
        return None
    nb = triton.cdiv(n_cols, block_size)
    bm = torch.empty((n, nb), dtype=torch.float32, device=scores.device)
    NB = 128
    _block_max_kernel[(n, triton.cdiv(nb, NB))](scores, n_vis, bm, n_cols, nb, scores.stride(0), bm.stride(0),
                                                BS=block_size, NB=NB, num_warps=4)
    bnvis = torch.div(n_vis.to(torch.int64) + (block_size - 1), block_size, rounding_mode="floor").to(torch.int32)
    out = torch.empty((n, topk_blocks), dtype=torch.int32, device=scores.device)
    itk.select(bm, bnvis, out, itk.SelectionWorkspace(n, nb, scores.device))
    return out


@triton.jit
def _mask_candidates_kernel(s_ptr, cand_ptr, out_ptr, n_cols, nb, k, s_s, s_c, s_o, BS: tl.constexpr,
                            BLOCK: tl.constexpr, LOGK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = cols < n_cols
    v = tl.load(s_ptr + row * s_s + cols, mask=ok, other=0.0)
    blk = cols // BS
    # candidates: ascending valid ids then -1 pads; find blk by binary search (pads read as +inf)
    lo = tl.zeros([BLOCK], dtype=tl.int32)
    hi = tl.zeros([BLOCK], dtype=tl.int32) + k
    for _ in range(LOGK + 1):
        mid = (lo + hi) // 2
        c = tl.load(cand_ptr + row * s_c + mid, mask=mid < k, other=-1)
        c = tl.where(c < 0, 2147483647, c)
        go_right = c < blk
        lo = tl.where((lo < hi) & go_right, mid + 1, lo)
        hi = tl.where((lo < hi) & (~go_right), mid, hi)
    c = tl.load(cand_ptr + row * s_c + lo, mask=lo < k, other=-1)
    found = (lo < k) & (c == blk) & (blk < nb)
    keep = found | (v == float("-inf"))
    tl.store(out_ptr + row * s_o + cols, tl.where(keep, v, -3.4028234663852886e38), mask=ok)


def mask_to_candidates(scores, candidates, block_size):
    """``sm80_attn_core.mask_to_candidates`` in one launch (candidates ascending, -1 padded); None if not
    served."""
    n, n_cols = scores.shape
    if (not scores.is_cuda or scores.dtype != torch.float32 or scores.stride(1) != 1 or n == 0
            or candidates.dtype != torch.int32 or candidates.stride(1) != 1):
        return None
    k = candidates.shape[1]
    nb = triton.cdiv(n_cols, block_size)
    out = torch.empty_like(scores)
    BLOCK = 1024
    _mask_candidates_kernel[(n, triton.cdiv(n_cols, BLOCK))](scores, candidates, out, n_cols, nb, k,
                                                             scores.stride(0), candidates.stride(0),
                                                             out.stride(0), BS=block_size, BLOCK=BLOCK,
                                                             LOGK=max(k, 1).bit_length(), num_warps=4)
    return out



# ---------------------------------------------------------------------------
# Boot-time compile of the row-class-specialised RMSNorm kernels
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field  # noqa: E402

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel  # noqa: E402


def _rms_class(rows: int, width: int):
    """The compile class of an exact RMSNorm call of ``rows`` rows (None when torch keeps the call)."""
    if 128 <= width <= 2048 and not width & (width - 1):
        bw = mean_block_width(rows, width)
        if 32 <= bw <= 128 and width // (4 * bw) <= 4:
            return ("pow2", bw)
        return None
    if _RMS_ANY and width % 4 == 0 and 128 <= width <= 8192:
        bw = _mean_block_width_any(rows, width)
        return ("any", bw, triton.cdiv(width, 4 * bw))
    return None


class ExactGlueRmsWarmup(VllmJitKernel["ExactGlueRmsWarmup.Key"]):
    """Boot-time compile of :func:`rms_norm` for every lane-width class a norm weight's width reaches (the kernels
    specialise on ``BW``, which torch's reduction derives from the call's row count): without it, the first call of
    an unseen class (e.g. a mixed decode + prefill batch at concurrency) compiles inside a request. Registered by
    each attention layer for its norm weights (q / kv / compressor / indexer key), the real tensors, so the
    specialisation key (dtype, alignment) is the served one."""

    @dataclass(frozen=True)
    class Key:
        width: int
        rows: int
        cls: tuple
        weight: torch.Tensor = field(compare=False, hash=False, repr=False)

    def get_warmup_keys(self, weight: torch.Tensor, eps: float = 1e-6) -> list["ExactGlueRmsWarmup.Key"]:
        if not enabled() or not weight.is_cuda or weight.dim() != 1:
            return []
        width = weight.numel()
        keys, seen = [], set()
        for rows in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
            cls = _rms_class(rows, width)
            if cls is not None and cls not in seen:
                seen.add(cls)
                keys.append(self.Key(width, rows, cls, weight))
        return keys

    def compile(self, compile_key: "ExactGlueRmsWarmup.Key") -> None:
        w = compile_key.weight
        x = torch.zeros((compile_key.rows, compile_key.width), dtype=torch.bfloat16, device=w.device)
        rms_norm(x, w, 1e-6)


EXACT_GLUE_RMS_WARMUP = ExactGlueRmsWarmup()
