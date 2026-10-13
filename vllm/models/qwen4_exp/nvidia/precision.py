# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp precision switches of the CMP 170HX (sm_80) fork.

``VLLM_QWEN4_ROUTER_FP32`` makes the MoE router GEMM return FP32 logits;
``VLLM_QWEN4_MTP_PER_STREAM_NORM`` normalizes the MTP hidden input per
hyper-connection stream instead of over the flattened stack. Both are read
once, when the owning module is constructed.
"""

import torch
from torch import nn

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def apply_router_precision(gate: nn.Module) -> None:
    """Select the router logits dtype for a Qwen4Exp sparse MoE block."""
    if envs.VLLM_QWEN4_ROUTER_FP32:
        gate.set_out_dtype(torch.float32)
    logger.info_once(
        "Qwen4Exp router logits: %s (VLLM_QWEN4_ROUTER_FP32=%d)",
        "FP32" if envs.VLLM_QWEN4_ROUTER_FP32 else "model dtype",
        int(envs.VLLM_QWEN4_ROUTER_FP32),
    )


class MTPHiddenNorm:
    """Pre-FC norm of the MTP hidden input, shape ``[T, hc_count, hidden]``.

    A plain callable, not a module: it borrows ``norm`` (the checkpoint's
    ``pre_fc_norm_hidden``) so the parameter layout is unchanged.
    """

    def __init__(self, norm: nn.Module, hc_count: int, hidden_size: int) -> None:
        self.norm = norm
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.per_stream = envs.VLLM_QWEN4_MTP_PER_STREAM_NORM
        logger.info_once(
            "Qwen4Exp MTP hidden norm: %s (VLLM_QWEN4_MTP_PER_STREAM_NORM=%d)",
            "per-stream" if self.per_stream else "flat",
            int(self.per_stream),
        )

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.per_stream:
            return self.norm(hidden_states.flatten(-2)).view_as(hidden_states)
        # Keep the checkpoint's distinct affine weights for every stream.
        # Like GemmaRMSNorm, apply (1 + weight) in FP32 before one BF16 cast.
        x = hidden_states.float()
        variance = x.square().mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.norm.variance_epsilon)
        weight = self.norm.weight.float().view(self.hc_count, self.hidden_size)
        return (x * (1.0 + weight)).to(hidden_states.dtype)
