# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Wrappers for the sm_80 CUDA kernels in inkling_sm80_cuda.cu.

Ported from github.com/snailbrainx/inkling-2-mini-sm80 (Apache-2.0, commit 8b998e0a2f; site/inkling_sm80_cuda.py and
csrc/inkling_sm80_cuda.cu). There the extension is prebuilt by csrc/build.sh; here it is JIT-compiled on first use
with the same nvcc flags and cached under the torch extensions directory. VLLM_INKLING2_SM80_CUDA=0 disables it.
"""

import os
from pathlib import Path

import torch

_ext = None
HAS_CUDA_EXT = False
if os.environ.get("VLLM_INKLING2_SM80_CUDA", "1") != "0":
    try:
        if torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 0):
            from torch.utils.cpp_extension import load

            _ext = load(
                name="_inkling_sm80_cuda",
                sources=[str(Path(__file__).with_name("inkling_sm80_cuda.cu"))],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "-std=c++20",
                    "-arch=sm_80",
                ],
                extra_cflags=["-O3", "-std=c++20"],
                verbose=False,
            )
            HAS_CUDA_EXT = True
    except Exception as _e:  # keep the Triton paths when the build fails
        _ext = None
        HAS_CUDA_EXT = False
        print(f"[inkling2 sm80] CUDA extension unavailable: {_e!r}", flush=True)

_ROUTE_SCRATCH: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
_SELECT_SCRATCH: dict[
    torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _get_route_scratch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    buf = _ROUTE_SCRATCH.get(device)
    if buf is None:
        buf = (
            torch.empty((8, 128), dtype=torch.float32, device=device),
            torch.zeros((1,), dtype=torch.int32, device=device),
        )
        _ROUTE_SCRATCH[device] = buf
    return buf


def _get_select_scratch(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    buf = _SELECT_SCRATCH.get(device)
    if buf is None:
        buf = (
            torch.zeros((128,), dtype=torch.int32, device=device),
            torch.empty((64 * 8,), dtype=torch.int32, device=device),
            torch.zeros((1,), dtype=torch.int32, device=device),
        )
        _SELECT_SCRATCH[device] = buf
    return buf


def conv_mix_cuda(
    x,
    conv_w,
    cache,
    positions,
    block_table,
    seq_idx,
    slot_mapping,
    query_start,
    stream_off,
    stream_w,
    conv_out,
    residual,
    addend,
    stack,
    last,
    queries,
    norm_w,
    norm_eps,
    eps,
):
    return _ext.conv_mix(
        x,
        conv_w,
        cache,
        positions,
        block_table,
        seq_idx,
        slot_mapping,
        query_start,
        stream_off,
        stream_w,
        conv_out,
        residual,
        addend,
        stack,
        last,
        queries,
        norm_w,
        norm_eps,
        eps,
    )


def batched_bank_messages_cuda(
    bank: torch.Tensor, queries: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return _ext.batched_bank_messages(bank, queries, eps)


def bank_stats_cuda(bank: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _ext.bank_stats(bank, eps)


def bank_rank2_finalize_cuda(
    dots: torch.Tensor,
    stats: torch.Tensor,
    num_layers: int,
    num_streams: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _ext.bank_rank2_finalize(dots, stats, int(num_layers), int(num_streams), eps)


def pair_rms_norm_rank2_cuda(
    dest: torch.Tensor, bank: torch.Tensor, c_norm: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return _ext.pair_rms_norm_rank2(dest, bank, c_norm, eps)


def fused_pair_norm_ctrl_down_cuda(
    dest: torch.Tensor, msg: torch.Tensor, down: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return _ext.fused_pair_norm_ctrl_down(dest, msg, down, eps)


def rms_norm_f32_to_bf16_cuda(
    x: torch.Tensor, norm_w: torch.Tensor, eps: float
) -> torch.Tensor:
    return _ext.rms_norm_f32_to_bf16(x, norm_w, eps)


def route_align_1k_cuda(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    is_pad: torch.Tensor | None,
    route_scale: float,
):
    g_logits, g_counter = _get_route_scratch(x.device)
    return _ext.route_align_1k(x, w, bias, is_pad, g_logits, g_counter, route_scale)


def select_align_cuda(
    partials: torch.Tensor,
    bias: torch.Tensor,
    is_pad: torch.Tensor | None,
    route_scale: float,
    block_size: int = 8,
):
    g_counts, g_row_rank, g_done = _get_select_scratch(partials.device)
    return _ext.select_align(
        partials,
        bias,
        is_pad,
        g_counts,
        g_row_rank,
        g_done,
        route_scale,
        int(block_size),
    )


def moe_align_large_cuda(ids: torch.Tensor, block_size: int, num_experts: int):
    return _ext.moe_align_large(ids.contiguous(), block_size, num_experts)


def markov_rnn_step_cuda(state, prev_tokens, w1_embed, rnn_w, context):
    return _ext.markov_rnn_step(state, prev_tokens, w1_embed, rnn_w, context)


def draft_ctx_kv_cuda(
    kv_proj, k_norm_w, cos_sin, positions, slot_mapping, key_cache, val_cache, eps
):
    _ext.draft_ctx_kv(
        kv_proj, k_norm_w, cos_sin, positions, slot_mapping, key_cache, val_cache, eps
    )


def draft_qk_norm_rope_cache_cuda(
    qkv,
    q_norm_w,
    k_norm_w,
    cos_sin,
    positions,
    slot_mapping,
    key_cache,
    val_cache,
    nhq,
    nhk,
    eps,
):
    _ext.draft_qk_norm_rope_cache(
        qkv,
        q_norm_w,
        k_norm_w,
        cos_sin,
        positions,
        slot_mapping,
        key_cache,
        val_cache,
        nhq,
        nhk,
        eps,
    )


def draft_embed_block_cuda(input_ids, embed_w, norm_w, mask_e, block_size, eps):
    return _ext.draft_embed_block(input_ids, embed_w, norm_w, mask_e, block_size, eps)


def fused_markov_fold_gumbel_b1_cuda(
    latent,
    fold_w,
    fold_b,
    base_logits,
    logits_cache,
    local_argmax,
    local_max,
    idx_map,
    seeds,
    pos,
    temps,
    lc_col,
    lc_row_map,
    vocab_size,
    softcap,
):
    _ext.fused_markov_fold_gumbel_b1(
        latent,
        fold_w,
        fold_b,
        base_logits,
        logits_cache,
        local_argmax,
        local_max,
        idx_map,
        seeds,
        pos,
        temps,
        lc_col,
        lc_row_map,
        int(vocab_size),
        float(softcap),
    )
