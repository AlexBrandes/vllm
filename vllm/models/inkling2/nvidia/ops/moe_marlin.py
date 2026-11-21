# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed experts on Ampere (SM8x): Marlin MoE GEMMs (bf16
activations, weights dequantized in the kernel) in place of Hopper's Humming
ones. SM8x has neither FP4 nor FP8 tensor cores, so the activations stay in
bf16, as on Hopper.

Weights are the checkpoint's NVFP4 / MXFP8 experts, or HyperInkling's INT4
(AWQ layout: asymmetric, group 128, `uint4` with zero points).

The interface and row order are `moe_sm90.HummingExperts`': rows stay in
the flat (token * topk + slot) order, the gate/up GEMM gathers each row's
token, `moe_sm90.swiglu` applies the softcapped SiTU and the routing
weight, and the down GEMM writes row by row for `ops.moe.combine`.
"""

import math
from types import SimpleNamespace
from typing import Literal

import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_workspace,
    marlin_moe_intermediate_size,
    marlin_moe_permute_scales,
    moe_awq_to_marlin_zero_points,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_nvfp4_moe_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    prepare_mxfp8_moe_layer_for_marlin,
)
from vllm.scalar_type import scalar_types

from .moe_sm90 import swiglu

ExpertFormat = Literal["nvfp4", "mxfp8", "int4"]
INT4_GROUP = 128


def _repack_int4(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, size_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AWQ-layout `[E, K, N / 8]` int32 weights, `[E, K / g, N]` scales and
    `[E, K / g, N / 8]` zero points into Marlin's layout (as
    `fused_moe.oracle.int_wna16._process_awq_weights_marlin`)."""
    size_n = qweight.shape[2] * 8
    marlin_qweight = ops.awq_marlin_moe_repack(
        qweight, size_k=size_k, size_n=size_n, num_bits=4
    )
    marlin_scales = marlin_moe_permute_scales(
        s=scales, size_k=size_k, size_n=size_n, group_size=INT4_GROUP
    )
    marlin_zeros = moe_awq_to_marlin_zero_points(
        qzeros, size_k=qzeros.shape[1], size_n=size_n, num_bits=4
    )
    return marlin_qweight, marlin_scales, marlin_zeros


class MarlinExperts:
    """Marlin tensors of the two expert projections.

    Args:
        weights: Per projection (`w13`, `w2`) a triple. NVFP4 / MXFP8: the
            row-major checkpoint weights `[E, N, K / pack]`, block scales
            `[E, N, K / group]` and, for NVFP4, the per-expert global scales
            `[E]` (else None). INT4: the AWQ-layout `qweight`, `scales` and
            `qzeros` (see `_repack_int4`).
        shapes: `(N, K)` of each projection; `w13`'s N interleaves the
            gate and up rows, which Marlin keeps in order.
        fmt: The experts' weight format.

    """

    def __init__(
        self,
        weights: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]],
        shapes: tuple[tuple[int, int], tuple[int, int]],
        fmt: ExpertFormat,
    ) -> None:
        (n13, hidden), (_, inter) = shapes
        assert n13 == 2 * inter
        w13, s13, x13 = weights["w13"]
        w2, s2, x2 = weights["w2"]
        self.num_experts = w13.shape[0]
        self.hidden_size, self.intermediate_size = hidden, inter
        layer = SimpleNamespace(
            num_experts=self.num_experts,
            hidden_size=hidden,
            intermediate_size_per_partition=inter,
            params_dtype=torch.bfloat16,
        )
        # Per projection: global scales (NVFP4) and zero points (INT4).
        g13 = g2 = z13 = z2 = None
        if fmt == "nvfp4":
            assert x13 is not None and x2 is not None
            self.quant_type = scalar_types.float4_e2m1f
            w13, s13, g13, w2, s2, g2 = prepare_nvfp4_moe_layer_for_marlin(
                layer, w13, s13, x13, w2, s2, x2, is_act_and_mul=True
            )
        elif fmt == "mxfp8":
            self.quant_type = scalar_types.float8_e4m3fn
            w13, w2, s13, s2 = prepare_mxfp8_moe_layer_for_marlin(
                layer, w13, w2, s13, s2
            )
        else:
            assert x13 is not None and x2 is not None
            self.quant_type = scalar_types.uint4
            w13, s13, z13 = _repack_int4(w13, s13, x13, hidden)
            w2, s2, z2 = _repack_int4(w2, s2, x2, inter)
        # The repacks pad the intermediate size only when Marlin's tiles
        # need it; 1152 needs none, and padding would split the (gate, up)
        # interleave.
        assert marlin_moe_intermediate_size(w13, w2) == inter
        self.w13, self.s13, self.g13, self.z13 = w13, s13, g13, z13
        self.w2, self.s2, self.g2, self.z2 = w2, s2, g2, z2
        self.workspace = get_marlin_workspace(w13.device)

    def _block_m(self, valid_rows: int) -> int:
        # fused_marlin_moe's choice: the smallest block that leaves fewer
        # than 0.9 blocks of rows per expert.
        for block_m in (8, 16, 32, 48, 64):
            if valid_rows / self.num_experts / block_m < 0.9:
                break
        return block_m

    def _gemm(
        self,
        inputs: torch.Tensor,
        out: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        global_scale: torch.Tensor | None,
        zeros: torch.Tensor | None,
        alignment: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        topk_weights: torch.Tensor,
        block_m: int,
        top_k: int,
        size_m: int,
        size_n: int,
        size_k: int,
    ) -> torch.Tensor:
        sorted_ids, expert_ids, num_tokens_padded = alignment
        return ops.moe_wna16_marlin_gemm(
            inputs,
            out,
            weight,
            None,
            scale,
            None,  # a_scales: the a0d wheel's op still takes it (upstream dropped it)
            global_scale,
            zeros,
            self.workspace,
            sorted_ids,
            expert_ids,
            num_tokens_padded,
            topk_weights,
            moe_block_size=block_m,
            top_k=top_k,
            mul_topk_weights=False,
            b_q_type=self.quant_type,
            size_m=size_m,
            size_n=size_n,
            size_k=size_k,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )

    def __call__(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        src2dst: torch.Tensor,
        num_global_experts: int,
        gate_cap: float,
        up_cap: float,
    ) -> torch.Tensor:
        """bf16 `[T * topk, H]` down-projection rows of the local slots of
        `ids` (`select_experts`); the other rows are left unset."""
        tokens, topk = ids.shape
        rows = tokens * topk
        valid_rows = math.ceil(rows * self.num_experts / num_global_experts)
        block_m = self._block_m(valid_rows)
        alignment = moe_align_block_size(
            ids, block_m, self.num_experts, ignore_invalid_experts=True
        )[:3]
        # The routing weights are applied by `swiglu`, not by the GEMMs.
        topk_weights = weights.view(tokens, topk)
        n13 = 2 * self.intermediate_size
        gate_up = torch.empty((rows, n13), dtype=torch.bfloat16, device=x.device)
        self._gemm(
            x, gate_up, self.w13, self.s13, self.g13, self.z13, alignment,
            topk_weights, block_m, topk, tokens, n13, self.hidden_size,
        )  # fmt: skip
        activated = swiglu(gate_up, weights, src2dst, gate_cap, up_cap)
        del gate_up
        down = torch.empty(
            (rows, self.hidden_size), dtype=torch.bfloat16, device=x.device
        )
        return self._gemm(
            activated, down, self.w2, self.s2, self.g2, self.z2, alignment,
            topk_weights, block_m, 1, rows, self.hidden_size, self.intermediate_size,
        )  # fmt: skip
