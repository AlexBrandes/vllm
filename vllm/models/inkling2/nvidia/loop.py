# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Looped Inkling2 checkpoints (`loop_config`).

The core layers `[core_start, core_end)` run `num_passes` times: a prelude,
then each pass from the AttnRes state after the prelude, then the coda from
the last pass's state. Each pass of a core layer has its own KV and
conv-state caches. In passes after the first, every core sublayer input
(attention q/k/v and MLP) is blended with a message: a normalized-value
AttnRes read of the previous pass's blocks, merged by a gated low-rank
controller. Each sublayer's short conv runs fused with the next AttnRes mix
(and small batches at TP > 1 with its all-reduce); without sequence parallel.
"""

import os
from collections.abc import Sequence

import torch
from torch import nn

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.platforms import current_platform

from ..configs import Inkling2ModelConfig
from ..quant_config import Inkling2QuantConfig
from .model import (
    Inkling2DecoderLayer,
    Inkling2ForConditionalGeneration,
    Inkling2Model,
)
from .ops import use_sm12x_path
from .ops.attn_res import attn_res
from .ops.bgemv import MAX_M as BGEMV_MAX_M
from .ops.bgemv import controller_gemv
from .ops.gemv import (
    sm80_blend_rms_norm,
    sm80_blend_rms_norm_rank2,
    sm80_controller_gain_forget,
)
from .ops.linear import linear_k_groups
from .ops.loop import blend_rms_norm, pair_rms_norm
from .sconv import conv_metadata

try:
    from vllm.models.inkling2.nvidia.ops.sm80.inkling_sm80_cuda import (
        HAS_CUDA_EXT as _SM80_HAS_CUDA,
    )
    from vllm.models.inkling2.nvidia.ops.sm80.inkling_sm80_cuda import (
        bank_rank2_finalize_cuda as _sm80_bank_rank2_finalize,
    )
    from vllm.models.inkling2.nvidia.ops.sm80.inkling_sm80_cuda import (
        bank_stats_cuda as _sm80_bank_stats,
    )
    from vllm.models.inkling2.nvidia.ops.sm80.inkling_sm80_cuda import (
        batched_bank_messages_cuda as _sm80_batched_bank_messages,
    )
except Exception:
    _SM80_HAS_CUDA = False
    _sm80_bank_rank2_finalize = None
    _sm80_bank_stats = None
    _sm80_batched_bank_messages = None

LOOP_STREAMS = ("attn", "k", "v", "mlp")


_FUSE_PAIRNORM = os.environ.get("VLLM_INKLING2_FUSE_PAIRNORM", "0") == "1"


class Inkling2MessageController(nn.Module):
    """Blends a loop message into a core sublayer input: from the rank-`r`
    projection `h` of the RMS-normalized `[dest, message]`,
    `(1 - tanh(forget(h))) * dest + tanh(gain(h)) * message` in fp32.

    The weights are views of the layer's stream-stacked weights
    (`Inkling2LoopDecoderLayer`), which run every stream at once; `forward`
    is the per-stream reference.
    """

    def __init__(
        self, down: torch.Tensor, gain: torch.Tensor, forget: torch.Tensor
    ) -> None:
        super().__init__()
        self.down_weight = nn.Parameter(down, requires_grad=False)
        self.gain_weight = nn.Parameter(gain, requires_grad=False)
        self.forget_weight = nn.Parameter(forget, requires_grad=False)

    def forward(self, dest: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """fp32 `(T, D)` `dest` and `message` -> fp32 `(T, D)`."""
        pair = torch.stack((dest, message), dim=1)
        pair = pair * torch.rsqrt(pair.square().mean(-1, keepdim=True) + 1e-6)
        normalized = pair.to(torch.bfloat16).flatten(1)
        hidden = linear_k_groups(normalized, self.down_weight, torch.bfloat16)
        gain = torch.mm(hidden, self.gain_weight.t(), out_dtype=torch.float32)
        forget = torch.mm(hidden, self.forget_weight.t(), out_dtype=torch.float32)
        return (1.0 - torch.tanh(forget)) * dest + torch.tanh(gain) * message


class Inkling2LoopDecoderLayer(Inkling2DecoderLayer):
    """A loop core layer: per-pass caches and a message controller per
    stream (`LOOP_STREAMS`), whose weights are stacked over the streams."""

    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__(
            config, layer_id, quant_config, prefix, num_passes=config.num_loop_passes
        )
        assert config.loop_config is not None
        rank, width = config.loop_config["gate_rank"], config.hidden_size
        streams = len(LOOP_STREAMS)
        # down (S, r, 2D) and [gain; forget] (S, 2D, r), bf16.
        self.register_buffer(
            "controller_down",
            torch.empty(streams, rank, 2 * width, dtype=torch.bfloat16),
            persistent=False,
        )
        self.register_buffer(
            "controller_gain_forget",
            torch.empty(streams, 2 * width, rank, dtype=torch.bfloat16),
            persistent=False,
        )
        self.controllers = nn.ModuleDict(
            {
                stream: Inkling2MessageController(
                    self.controller_down[i],
                    self.controller_gain_forget[i, :width],
                    self.controller_gain_forget[i, width:],
                )
                for i, stream in enumerate(LOOP_STREAMS)
            }
        )

    def process_weights_after_loading(self) -> None:
        # The controllers' weights must still alias the stacked ones.
        for i, controller in enumerate(self.controllers.values()):
            down, gain_forget = self.controller_down, self.controller_gain_forget
            assert controller.down_weight.data_ptr() == down[i].data_ptr()
            assert controller.gain_weight.data_ptr() == gain_forget[i].data_ptr()

    def blend(
        self,
        streams: slice,
        dest: torch.Tensor,
        message: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor:
        """The `streams` controllers on fp32 `(S, T, D)` `dest` (may
        broadcast over streams) and `message`, each output RMS-normalized
        with its row of `norm_weight` `(S, D)`: bf16 `(S, T, D)`."""
        # 170HX (sm_80): hybrid fused gain/forget+blend+CUDA rms_norm (halves HBM writes, M <= 64).
        fused = sm80_blend_rms_norm(
            dest,
            message,
            self.controller_down[streams],
            self.controller_gain_forget[streams],
            norm_weight,
            norm_eps,
        )
        if fused is not None:
            return fused
        pair = pair_rms_norm(dest, message)
        # 170HX (sm_80), 21 Nov 2026: both products in two Triton kernels for decode-sized batches (ops/gemv.py).
        gain_forget = sm80_controller_gain_forget(
            pair, self.controller_down[streams], self.controller_gain_forget[streams]
        )
        if gain_forget is None and pair.shape[1] <= BGEMV_MAX_M and use_sm12x_path():
            # HyperInkling: decode-size controllers on a batched Triton GEMV.
            gain_forget = controller_gemv(
                pair,
                self.controller_down[streams],
                self.controller_gain_forget[streams],
            )
        if gain_forget is None:
            hidden = torch.bmm(pair, self.controller_down[streams].transpose(1, 2))
            gain_forget = torch.bmm(
                hidden,
                self.controller_gain_forget[streams].transpose(1, 2),
                out_dtype=torch.float32,
            )
        return blend_rms_norm(dest, message, gain_forget, norm_weight, norm_eps)

    def blend_rank2(
        self,
        streams: slice,
        dest: torch.Tensor,
        bank: torch.Tensor,
        c_raw: torch.Tensor,
        c_norm: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_eps: float,
    ) -> torch.Tensor:
        """Rank-2 prefill blend directly from (bank, c_raw, c_norm) without fp32 message tensor."""
        out = sm80_blend_rms_norm_rank2(
            dest,
            bank,
            c_raw,
            c_norm,
            self.controller_down[streams],
            self.controller_gain_forget[streams],
            norm_weight,
            norm_eps,
        )
        assert out is not None
        return out


class Inkling2LoopModel(Inkling2Model):
    seq_parallel_paths = False

    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        # Serving options set through --hf-overrides skip the constructor.
        config.check_loop_options()
        super().__init__(config, quant_config, prefix)
        self.loop_core = config.loop_core_layers
        self.hidden_size = config.hidden_size
        # Message pseudo-queries (raw), two per core layer: attn/mlp and k/v.
        shape = (2 * len(self.loop_core), config.hidden_size)
        self.middle_cycle_cross_pass_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(
            torch.zeros(shape, dtype=torch.float32), requires_grad=False
        )
        # Per core layer, the `LOOP_STREAMS` queries scaled by D**-0.5.
        self.register_buffer(
            "message_queries",
            torch.zeros(
                len(self.loop_core),
                len(LOOP_STREAMS),
                config.hidden_size,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self._q_split_bf16_t: torch.Tensor | None = None

    def process_weights_after_loading(self) -> None:
        queries = self.middle_cycle_cross_pass_queries.view(-1, 2, self.hidden_size)
        kv_queries = self.middle_cycle_cross_pass_kv_queries.view(
            -1, 2, self.hidden_size
        )
        # (attn, k, v, mlp) = (queries[0], kv[0], kv[1], queries[1]).
        stacked = torch.stack(
            (queries[:, 0], kv_queries[:, 0], kv_queries[:, 1], queries[:, 1]), 1
        )
        self.message_queries.copy_(stacked * self.hidden_size**-0.5)
        self._q_split_bf16_t = None

    def _get_q_split_bf16_t(self) -> torch.Tensor:
        qt = self._q_split_bf16_t
        if qt is None or qt.device != self.message_queries.device:
            q_flat = self.message_queries.reshape(-1, self.hidden_size)
            q_hi = q_flat.to(torch.bfloat16)
            q_lo = (q_flat - q_hi.to(torch.float32)).to(torch.bfloat16)
            qt = torch.cat((q_hi, q_lo), dim=0).t().contiguous()
            self._q_split_bf16_t = qt
        return qt

    def _make_layer(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> Inkling2DecoderLayer:
        if layer_id in config.loop_core_layers:
            return Inkling2LoopDecoderLayer(config, layer_id, quant_config, prefix)
        return super()._make_layer(config, layer_id, quant_config, prefix)

    def _max_loop_blocks(self) -> int:
        """Completed AttnRes blocks a loop forward holds at most: each pass
        restarts from the prelude's, and the coda continues the last pass."""
        num_blocks = max_blocks = 0
        core = self.loop_core
        for i in [*range(core.start), *core, *range(core.stop, len(self.layers))]:
            num_blocks += i % self.block_size == 0
            max_blocks = max(max_blocks, num_blocks)
        return max_blocks

    def _run_layer(
        self,
        i: int,
        pass_idx: int,
        stack: torch.Tensor,
        num_blocks: int,
        inputs: torch.Tensor | None,
        bank: torch.Tensor | None,
        next_query: torch.Tensor,
        next_norm: nn.Module | None,
        positions: torch.Tensor,
        tau: torch.Tensor | None,
        plan: list[tuple[bool, bool]],
        precomputed_messages: torch.Tensor | None = None,
        precomputed_rank2: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[int, torch.Tensor]:
        """Run decoder layer `i` in loop pass `pass_idx` on its `inputs`
        (the attention inputs; for a visit with messages from `bank`, the
        fp32 mixes they blend into). Returns the number of completed AttnRes
        blocks after it and the mixes of its output with `next_query` through
        `next_norm` (fp32 without it), the next visit's inputs."""
        layer = self.layers[i]
        partial = stack[num_blocks]
        owner = layer.conv_states[pass_idx]
        workspace = self.allreduce_workspace
        (attn_fused, attn_two_shot), (moe_fused, moe_two_shot) = plan
        has_msg = bank is not None
        messages = None
        attn_inputs: Sequence[torch.Tensor]
        if bank is not None:
            assert isinstance(layer, Inkling2LoopDecoderLayer) and inputs is not None
            norm = layer.attn_norm
            dests, norm_weight = inputs, norm.weight
            if layer.is_local:
                dests = dests.expand(3, -1, -1)
                norm_weight = norm_weight.expand(3, -1)
            if precomputed_rank2 is not None:
                c_raw, c_norm = precomputed_rank2
                attn_inputs = layer.blend_rank2(
                    slice(0, 3),
                    dests,
                    bank,
                    c_raw[:3],
                    c_norm[:3],
                    norm_weight,
                    norm.eps,
                )
            else:
                # All four streams' messages: a mix of the bank's two blocks.
                if precomputed_messages is not None:
                    messages = precomputed_messages
                else:
                    messages = attn_res(
                        bank[:1],
                        bank[1],
                        None,
                        self.message_queries[i - self.loop_core.start],
                        None,
                        0.0,
                        True,
                    )
                attn_inputs = layer.blend(
                    slice(0, 3), dests, messages[:3], norm_weight, norm.eps
                )
        elif inputs is None:
            assert num_blocks == 0
            attn_inputs = [
                layer.attn_norm(partial, j) for j in range(layer.num_attn_inputs)
            ]
        else:
            attn_inputs = inputs
        attn_partial = layer.attention(positions, attn_inputs, tau, pass_idx=pass_idx)

        # The MLP input mixes the blocks with (partial +) attn_out.
        first = i % self.block_size == 0
        attn_out = torch.empty_like(partial)
        mix_args = (
            positions,
            attn_out,
            stack[: num_blocks + first],
            None if first else partial,
            self.attn_res.mlp_queries[i],
            None if has_msg else layer.mlp_norm,
        )
        if attn_fused:
            assert workspace is not None
            mlp_in = layer.attn_sconv.all_reduce_mix(
                workspace,
                *mix_args,
                x=attn_partial,
                two_shot=attn_two_shot,
                owner=owner,
            )
        else:
            attn_reduced = tensor_model_parallel_all_reduce(attn_partial)
            mlp_in = layer.attn_sconv.conv_mix(attn_reduced, *mix_args, owner=owner)
        if has_msg:
            assert isinstance(layer, Inkling2LoopDecoderLayer) and bank is not None
            norm = layer.mlp_norm
            if precomputed_rank2 is not None:
                c_raw, c_norm = precomputed_rank2
                mlp_in = layer.blend_rank2(
                    slice(3, 4),
                    mlp_in,
                    bank,
                    c_raw[3:],
                    c_norm[3:],
                    norm.weight,
                    norm.eps,
                )
            else:
                assert messages is not None
                mlp_in = layer.blend(
                    slice(3, 4), mlp_in, messages[3:], norm.weight, norm.eps
                )

        # The MLP short conv also applies the residual update.
        residual = None if first else partial
        if residual is None:
            num_blocks += 1
        mix_args = (
            positions,
            stack[num_blocks],
            stack[:num_blocks],
            None,
            next_query,
            next_norm,
        )
        if moe_fused:
            assert workspace is not None
            outputs = layer.mlp_sconv.all_reduce_mix(
                workspace,
                *mix_args,
                combine=layer.mlp(mlp_in[0], reduce=False, pass_idx=pass_idx),
                addend=attn_out,
                residual=residual,
                two_shot=moe_two_shot,
                owner=owner,
            )
        else:
            # The MoE output is all-reduced (in bf16) inside layer.mlp.
            outputs = layer.mlp_sconv.conv_mix(
                layer.mlp(mlp_in[0], pass_idx=pass_idx),
                *mix_args,
                addend=attn_out,
                residual=residual,
                owner=owner,
            )
        return num_blocks, outputs

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Prelude, `num_passes` core passes, then the coda. Each sublayer's
        short conv runs fused with the next AttnRes mix (and small batches
        with its all-reduce)."""
        num_tokens, hidden = inputs_embeds.shape
        core = self.loop_core
        num_passes = self.config.num_loop_passes
        stack = inputs_embeds.new_empty(
            (self._max_loop_blocks() + 1, num_tokens, hidden)
        )
        stack[0].copy_(inputs_embeds)
        tau = self._log_scaling_tau(positions)
        workspace = self.allreduce_workspace
        plan = [(False, False)] * 2
        if (
            workspace is not None
            and conv_metadata(self.layers[0].conv_state) is not None
        ):
            plan = workspace.plan(num_tokens)
        # (layer, pass) visits; the prelude and coda run as pass 0.
        visits = [(i, 0) for i in range(core.start)]
        visits += [(i, p) for p in range(num_passes) for i in core]
        visits += [(i, 0) for i in range(core.stop, len(self.layers))]
        aux_hidden_states = []
        next_norm: nn.Module | None
        num_blocks = 0
        inputs = bank = pass_messages = pass_c_raw = pass_c_norm = prelude_blocks = (
            prelude_partial
        ) = prelude_dests = None
        for visit, (i, pass_idx) in enumerate(visits):
            looped = pass_idx > 0
            if i == core.start and not looped:
                prelude_blocks = num_blocks
                prelude_partial = stack[num_blocks].clone()
            elif i == core.start:
                # Every pass restarts from the prelude's state, with the
                # previous pass's blocks (after the prelude's) and partial as
                # the messages' bank.
                assert prelude_blocks is not None and prelude_partial is not None
                bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                pass_messages = None
                pass_c_raw = None
                pass_c_norm = None
                if (
                    _SM80_HAS_CUDA
                    and current_platform.is_device_capability_family(80)
                    and bank.shape[0] == 2
                    and hidden == 2304
                    and bank.dtype == torch.bfloat16
                    and bank.stride(1) == 2304
                    and bank.stride(2) == 1
                    and self.message_queries.dtype == torch.float32
                    and self.message_queries.is_contiguous()
                ):
                    if (
                        num_tokens <= 256
                        and os.environ.get("INKLING_SM80_NO_BATCHED_BANK") != "1"
                    ):
                        pass_messages = _sm80_batched_bank_messages(
                            bank, self.message_queries, 1e-6
                        )
                    elif (
                        num_tokens > 256
                        and bank.is_contiguous()
                        and os.environ.get("INKLING_SM80_NO_RANK2_BANK") != "1"
                        and os.environ.get("INKLING_SM80_NO_HYBRID_BLEND") != "1"
                    ):
                        stats = _sm80_bank_stats(bank, 1e-6)
                        dots = torch.mm(
                            bank.view(2 * num_tokens, hidden),
                            self._get_q_split_bf16_t(),
                            out_dtype=torch.float32,
                        )
                        pass_c_raw, pass_c_norm = _sm80_bank_rank2_finalize(
                            dots, stats, len(core), len(LOOP_STREAMS), 1e-6
                        )
                num_blocks = prelude_blocks
                stack[num_blocks].copy_(prelude_partial)
                if prelude_dests is None:
                    query = self.attn_res.attn_queries[i - 1]
                    prelude_dests = attn_res(
                        stack[:num_blocks],
                        prelude_partial,
                        None,
                        query,
                        None,
                        0.0,
                        True,
                    )
                inputs = prelude_dests
            if visit in self.aux_hidden_state_layers:
                aux_hidden_states.append(stack[num_blocks].clone())
            if visit + 1 == len(visits):
                next_query, next_norm = self.attn_res.output_query, self.norm
            else:
                next_i, next_pass = visits[visit + 1]
                next_query = self.attn_res.attn_queries[next_i - 1]
                # Messages blend into fp32 mixes; a pass restart discards it.
                next_norm = None if next_pass else self.layers[next_i].attn_norm
            num_blocks, inputs = self._run_layer(
                i,
                pass_idx,
                stack,
                num_blocks,
                inputs,
                bank if looped else None,
                next_query,
                next_norm,
                positions,
                tau,
                plan,
                pass_messages[i - core.start]
                if (looped and pass_messages is not None)
                else None,
                (pass_c_raw[i - core.start], pass_c_norm[i - core.start])
                if (looped and pass_c_raw is not None and pass_c_norm is not None)
                else None,
            )
        assert inputs is not None
        out = inputs[0]
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out


class Inkling2LoopForConditionalGeneration(Inkling2ForConditionalGeneration):
    """The `inkling2_loop` checkpoints' architecture name; the model is
    `Inkling2ForConditionalGeneration` with an `Inkling2LoopModel`."""
