# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lookup-augmented DFlash2 drafting for the vllm-cmp170hx engine (Qwen3.8-27B).

A minimal port of HyperQwen's ``dflash2-lookup-drafting.patch``
(LABD): when the tokens a request just produced already occurred earlier in
its own context, propose the continuation that followed them instead of the
drafter's guess. One Triton program per request finds the most recent
occurrence of the longest suffix (NMIN..NMAX tokens) of the token history and,
when the match is at least NSTRONG tokens long, overwrites the drafted block
with the tokens that followed it (as many as the history holds, up to the
block width). Shorter matches (NMIN..NSTRONG-1) leave the drafter's block
alone, unless AGREE > 0: then a shorter match takes over when the drafter's
first AGREE tokens equal the lookup continuation's.

Not ported: the long verify block filled past the drafter's width (needs the
scheduler's adaptive verify length) and the point-mass rewrite of the draft
distribution. The engine drafts greedily (DFlash ``draft_sample_method``
default), so every drafted token is already a point mass for the rejection
sampler and replacing it keeps decoding lossless; ``enabled_for`` refuses
probabilistic drafting.

Switch: VLLM_CMP_DFLASH_LOOKUP=1 (also keeps the token history dense on the
GPU). Tunables: VLLM_CMP_DFLASH_LOOKUP_NMIN / _NMAX / _NSTRONG /
_AGREE / _SEARCH (tokens of history searched back from the end; 0 = all).
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

SCORE_STRIDE = tl.constexpr(1 << 32)


def enabled() -> bool:
    return os.environ.get("VLLM_CMP_DFLASH_LOOKUP", "0") == "1"


@triton.jit
def _suffix_lookup_fuse_kernel(
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    idx_mapping_ptr,
    draft_tokens_ptr,
    draft_stride,
    hits_ptr,
    search_max,
    k: tl.constexpr,
    NMAX: tl.constexpr,
    NMIN: tl.constexpr,
    NSTRONG: tl.constexpr,
    AGREE: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_KT: tl.constexpr,
):
    req = tl.program_id(0)
    req_state = tl.load(idx_mapping_ptr + req)
    if req_state < 0:
        return
    total_len = tl.load(total_len_ptr + req_state)
    if total_len < NMIN + 2:
        return
    base = all_token_ids_ptr + req_state.to(tl.int64) * all_token_ids_stride
    end_of_suffix = total_len - 1
    hi = end_of_suffix
    lo = tl.maximum(NMIN - 1, total_len - search_max)
    best = tl.full([], -1, tl.int64)
    for start in range(lo, hi, BLOCK):
        e = start + tl.arange(0, BLOCK)
        alive = (e < hi) & (e >= lo)
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
    match_len = (best // SCORE_STRIDE).to(tl.int32)
    if AGREE == 0:
        if match_len < NSTRONG:
            return
    end = (best % SCORE_STRIDE).to(tl.int32)
    valid = tl.minimum(k, end_of_suffix - end)
    idx = tl.arange(0, BLOCK_KT)
    take = idx < valid
    toks = tl.load(base + end + 1 + idx, mask=take, other=0).to(tl.int64)
    if AGREE > 0:
        # weak match: take over only if the drafter already agrees on the
        # first AGREE tokens of the continuation
        head = idx < AGREE
        cur = tl.load(
            draft_tokens_ptr + req * draft_stride + idx, mask=head, other=0
        ).to(tl.int64)
        bad = tl.sum((head & ((cur != toks) | (idx >= valid))).to(tl.int32), axis=0)
        if (match_len < NSTRONG) & (bad > 0):
            return
    tl.store(draft_tokens_ptr + req * draft_stride + idx, toks, mask=take)
    tl.atomic_add(hits_ptr, 1)


class DFlashLookup:
    def __init__(self, device: torch.device):
        self.nmin = int(os.environ.get("VLLM_CMP_DFLASH_LOOKUP_NMIN", 6))
        self.nmax = int(os.environ.get("VLLM_CMP_DFLASH_LOOKUP_NMAX", 12))
        self.nstrong = int(os.environ.get("VLLM_CMP_DFLASH_LOOKUP_NSTRONG", 6))
        self.agree = int(os.environ.get("VLLM_CMP_DFLASH_LOOKUP_AGREE", 0))
        self.search = int(os.environ.get("VLLM_CMP_DFLASH_LOOKUP_SEARCH", 0)) or (
            1 << 30
        )
        self.hits = torch.zeros(1, dtype=torch.int64, device=device)

    def fuse(
        self,
        all_token_ids: torch.Tensor,
        total_len: torch.Tensor,
        idx_mapping: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> None:
        num_reqs, k = draft_tokens.shape[0], draft_tokens.shape[1]
        if num_reqs == 0 or k == 0:
            return
        _suffix_lookup_fuse_kernel[(num_reqs,)](
            all_token_ids,
            all_token_ids.stride(0),
            total_len,
            idx_mapping,
            draft_tokens,
            draft_tokens.stride(0),
            self.hits,
            self.search,
            k=k,
            NMAX=self.nmax,
            NMIN=self.nmin,
            NSTRONG=self.nstrong,
            AGREE=self.agree,
            BLOCK=256,
            BLOCK_KT=triton.next_power_of_2(max(k, 2)),
            num_warps=4,
        )


def enabled_for(speculative_config) -> bool:
    if not enabled() or speculative_config is None:
        return False
    if getattr(speculative_config, "method", None) not in ("dflash", "dflash2"):
        return False
    method = getattr(speculative_config, "draft_sample_method", "greedy")
    if method not in (None, "greedy"):
        raise ValueError(
            "VLLM_CMP_DFLASH_LOOKUP needs greedy drafts "
            f"(draft_sample_method={method!r}): the port does not "
            "rewrite the draft distribution"
        )
    return True
