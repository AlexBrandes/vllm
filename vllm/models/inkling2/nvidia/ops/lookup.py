# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drafting from the request's own context (HYPERINKLING_LOOKUP=1).

The DSpark drafter sees a 1024-token window; a document the answer quotes can be far
beyond it. One Triton program per request finds the most recent occurrence of the longest
suffix (NMIN..NMAX tokens) of the token history and proposes the tokens that followed it.
The proposal is fused with the drafter's: a match of `nstrong` tokens or more replaces the
drafts on its own, a shorter one (down to `nmin`) only where the drafter independently
proposed the same first `agree_min` tokens.

Lossless: every position taken from the lookup gets a point mass on the proposed token in
the draft-logits cache, a legal q for the rejection sampler (acceptance becomes p(x)).

After HyperQwen's dflash2-lookup-drafting (syv-ai/HyperQwen, Apache-2.0).
"""

import torch

from vllm.triton_utils import tl, triton

# "Longest match, then most recent" as one max over len * SCORE_STRIDE + end index.
SCORE_STRIDE = tl.constexpr(1 << 32)


@triton.jit
def _suffix_lookup_kernel(
    tokens_ptr,
    tokens_stride,
    total_len_ptr,
    idx_mapping_ptr,
    idx_mapping_stride,
    out_tokens_ptr,
    out_len_ptr,
    out_valid_ptr,
    search_max,
    K: tl.constexpr,
    NMAX: tl.constexpr,
    NMIN: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    req = tl.program_id(0)
    tl.store(out_len_ptr + req, 0)
    tl.store(out_valid_ptr + req, 0)
    req_state = tl.load(idx_mapping_ptr + req * idx_mapping_stride)
    if req_state < 0:
        return
    total_len = tl.load(total_len_ptr + req_state)
    if total_len < NMIN + 2:
        return
    base = tokens_ptr + req_state.to(tl.int64) * tokens_stride
    end_of_suffix = total_len - 1
    # Candidate ends before the suffix's own end; a candidate may overlap the suffix, so
    # a repeating pattern (a list marker, an indent) is proposed from its own period.
    hi = end_of_suffix
    lo = tl.maximum(NMIN - 1, total_len - search_max)
    best = tl.full([], -1, tl.int64)
    for start in range(lo, hi, BLOCK):
        e = start + tl.arange(0, BLOCK)
        alive = (e < hi) & (e >= lo)
        # The NMIN-token prefix first: most blocks die here.
        for j in range(NMIN):
            s_j = tl.load(base + end_of_suffix - j)
            t_j = tl.load(base + e - j, mask=alive & ((e - j) >= 0), other=-1)
            alive = alive & (t_j == s_j)
        if tl.max(alive.to(tl.int32)) > 0:
            match = tl.where(alive, NMIN, 0)
            longer = alive
            for j in range(NMIN, NMAX):
                s_j = tl.load(
                    base + end_of_suffix - j, mask=(end_of_suffix - j) >= 0, other=-2
                )
                t_j = tl.load(base + e - j, mask=longer & ((e - j) >= 0), other=-1)
                longer = longer & (t_j == s_j)
                match = tl.where(longer, match + 1, match)
            score = tl.where(
                alive, match.to(tl.int64) * SCORE_STRIDE + e.to(tl.int64), -1
            )
            best = tl.maximum(best, tl.max(score, axis=0))
    if best < 0:
        return
    end = (best % SCORE_STRIDE).to(tl.int32)
    # A match close to the suffix has fewer than K tokens after it.
    valid = tl.minimum(K, end_of_suffix - end)
    idx = tl.arange(0, BLOCK_K)
    toks = tl.load(base + end + 1 + idx, mask=idx < valid, other=0)
    tl.store(out_tokens_ptr + req * K + idx, toks, mask=idx < K)
    tl.store(out_len_ptr + req, (best // SCORE_STRIDE).to(tl.int32))
    tl.store(out_valid_ptr + req, valid)


@triton.jit
def _fuse_kernel(
    draft_ptr,
    draft_stride,
    look_ptr,
    len_ptr,
    valid_ptr,
    use_ptr,
    idx_mapping_ptr,
    idx_mapping_stride,
    cache_ptr,
    cache_stride_row,
    cache_stride_step,
    nmin,
    nstrong,
    agree_min,
    vocab,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_CACHE: tl.constexpr,
):
    """Fuse, then rewrite the draft-logits rows of the taken positions to a point mass.
    Grid (requests, K): program (r, 0) decides, every program rewrites its own step."""
    req = tl.program_id(0)
    step = tl.program_id(1)
    req_state = tl.load(idx_mapping_ptr + req * idx_mapping_stride)
    if req_state < 0:
        return
    idx = tl.arange(0, BLOCK_K)
    kmask = idx < K
    match_len = tl.load(len_ptr + req)
    valid = tl.load(valid_ptr + req)
    drafted = tl.load(draft_ptr + req * draft_stride + idx, mask=kmask, other=0)
    looked = tl.load(look_ptr + req * K + idx, mask=kmask, other=0).to(tl.int64)
    disagree = (drafted != looked) & (idx < valid) & kmask
    # Positions the match did not supply are not agreement.
    agree = tl.minimum(tl.min(tl.where(disagree, idx, K)), valid)
    take = (match_len >= nstrong) | ((match_len >= nmin) & (agree >= agree_min))
    # Every program of the request computes the same decision; each writes its own step.
    if take & (step < valid):
        token = tl.load(look_ptr + req * K + step).to(tl.int64)
        tl.store(draft_ptr + req * draft_stride + step, token)
        if HAS_CACHE:
            row = (
                cache_ptr
                + req_state.to(tl.int64) * cache_stride_row
                + step * cache_stride_step
            )
            for v0 in range(0, vocab, BLOCK_V):
                offs = v0 + tl.arange(0, BLOCK_V)
                tl.store(row + offs, float("-inf"), mask=offs < vocab)
            tl.store(row + token, 0.0)
    if step == 0:
        tl.store(use_ptr + req, (take & (valid > 0)).to(tl.int32))


class ContextLookup:
    """Persistent buffers for one speculator (CUDA-graph safe: no host reads)."""

    def __init__(
        self,
        max_num_reqs: int,
        k: int,
        device: torch.device,
        nmin: int = 6,
        nstrong: int = 8,
        agree_min: int = 2,
        nmax: int = 32,
        search_max: int = 1 << 30,
    ):
        self.k, self.nmin, self.nstrong, self.agree_min = k, nmin, nstrong, agree_min
        self.nmax, self.search_max = nmax, search_max
        self.tokens = torch.zeros(max_num_reqs, k, dtype=torch.int32, device=device)
        self.match_len = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.valid = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # Per request: 1 when this step's drafts came from the lookup (for stats).
        self.used = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    def apply(
        self,
        token_history: torch.Tensor,
        total_len: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_stride: int,
        num_reqs: int,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> None:
        """`draft_tokens[:num_reqs]` <- the fused proposal; `draft_logits` (the
        [rows, K, vocab] cache indexed by request state) <- point masses where taken."""
        _suffix_lookup_kernel[(num_reqs,)](
            token_history,
            token_history.stride(0),
            total_len,
            idx_mapping,
            idx_mapping_stride,
            self.tokens,
            self.match_len,
            self.valid,
            self.search_max,
            K=self.k,
            NMAX=self.nmax,
            NMIN=min(self.nmin, 4),
            BLOCK=1024,
            BLOCK_K=triton.next_power_of_2(self.k),
            num_warps=4,
        )
        cache = draft_logits if draft_logits is not None else draft_tokens
        _fuse_kernel[(num_reqs, self.k)](
            draft_tokens,
            draft_tokens.stride(0),
            self.tokens,
            self.match_len,
            self.valid,
            self.used,
            idx_mapping,
            idx_mapping_stride,
            cache,
            cache.stride(0),
            cache.stride(1) if cache.dim() > 1 else 0,
            self.nmin,
            self.nstrong,
            self.agree_min,
            cache.shape[-1] if draft_logits is not None else 0,
            K=self.k,
            BLOCK_K=triton.next_power_of_2(self.k),
            BLOCK_V=4096,
            HAS_CACHE=draft_logits is not None,
            num_warps=4,
        )
