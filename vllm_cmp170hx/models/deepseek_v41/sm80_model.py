# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 model classes of the sm_80 fork.

The fork plugin (``vllm_cmp170hx.plugin``) registers these in vLLM's model
registry in place of the upstream ``DeepseekV41ForCausalLM`` (the vision
wrapper around the text model) and the DSpark drafter
(``DSparkV41DraftModel``). Each subclasses the upstream class, keeps its
name, and is built by the upstream constructor; the modules it builds are
then given the fork classes below, so every module and parameter keeps its
upstream name (and the EXL3 checkpoint maps onto them through
``sm80_exl3.DeepseekV41Exl3Layout``).

What changes on sm_80:

- mHC runs in the reference's FP32 formulas (``ops/sm80_mhc.py``) instead of
  the TileLang/DeepGEMM kernels: the decoder layers' forward and ``write_kv``
  (the decoder replay's KV write), the backbone's final post and the DSpark
  drafter's forward call them. With
  ``VLLM_DEEPSEEK_V41_FUSED_MHC`` the decoder layers' delayed pre and fused post +
  pre run through the sm_80 mHC kernel A instead, with the finish in torch or,
  with ``VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH``, as a second kernel
  (``ops/sm80_mhc_fused.py``, same contracts and rounding points; :func:`mhc_ops`);
- each MoE keeps the upstream modules and EXL3 experts, with an FP32 gate and
  the reference routing (``sm80_moe.adopt_moe``);
- the drafter holds no embedding or LM head of its own (the DSpark loader
  aliases the target's), and keeps its confidence head across the EXL3
  loader's one-tensor-at-a-time ``load_weights`` calls; its sm_80 stages
  write their context KV from their own ``wkv``, so it drops upstream's
  stacked context-KV projection and that projection's checkpoint copies;
- the logits, the DSpark Markov bias and the drafts' logits are FP32, as the
  reference computes them: the plugin defaults ``head_dtype`` to float32
  (``vllm_cmp170hx.plugin.HF_CONFIG_DEFAULTS``), the EXL3 head stores its FP32
  result in that dtype and both models' logits processors serve it
  (``Exl3HeadLogitsProcessor``);
- with ``VLLM_SM80_SHARD_ARGMAX`` the target's ``compute_logits`` returns,
  inside a greedy-only sampling step of the fork runner, rows that hold only
  the vocabulary-parallel argmax instead of the all-gathered logits
  (``vllm_cmp170hx/worker/target_argmax.py``: ``ShardArgmaxLogits``).

The attention layers are the platform's (``_select_dsv4_attn_cls``); the
layer topology this supports is TP with whole-expert EP and no sequence
parallelism, mHC overlap or fused all-reduce (none of which sm_80 enables).
"""

import os
from itertools import islice

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.models.deepseek_v41.nvidia import dspark as upstream_dspark
from vllm.models.deepseek_v41.nvidia import model as upstream_model
from vllm.models.deepseek_v41.nvidia import vl_model as upstream_vl
from vllm.sequence import IntermediateTensors
from vllm_cmp170hx.models.deepseek_v41.ops import sm80_mhc
from vllm_cmp170hx.models.deepseek_v41.ops import sm80_mhc_side
from vllm_cmp170hx import capture
from vllm_cmp170hx.models.deepseek_v41.ops.sm80_mhc import mhc_post
from vllm_cmp170hx.models.deepseek_v41 import sm80_pp_shadow
from vllm_cmp170hx.models.deepseek_v41 import sm80_split_reduce
from vllm_cmp170hx.models.deepseek_v41.sm80_attention import DeepseekV41Sm80Attention
from vllm_cmp170hx.models.deepseek_v41.sm80_moe import adopt_moe, adopt_row_mask
from vllm_cmp170hx.kernels.exl3.quant import Exl3HeadLogitsProcessor
from vllm_cmp170hx.worker import target_argmax

logger = init_logger(__name__)


def mhc_ops(config):
    """The module whose delayed mHC ops (``mhc_pre_delayed``,
    ``mhc_fused_post_pre_delayed``, ``mhc_post``) the decoder layers call:
    ``ops/sm80_mhc_fused.py`` with ``VLLM_DEEPSEEK_V41_FUSED_MHC``, else the
    reference's torch formulas ``ops/sm80_mhc.py``. The layers look each op
    up on the module at every call. With ``VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2``
    the finish v2's gate is decided here, once: open, its kernels are registered
    for the pre-capture warmup; closed, the switch is cleared and the line
    says why."""
    if not envs.VLLM_DEEPSEEK_V41_FUSED_MHC:
        return sm80_mhc
    from vllm_cmp170hx.models.deepseek_v41.ops import sm80_mhc_fused

    reason = sm80_mhc_fused.supports(config.hc_mult, config.hidden_size)
    if reason is not None:
        raise ValueError(f"VLLM_DEEPSEEK_V41_FUSED_MHC: {reason}")
    logger.info_once(
        "DeepSeek-V4.1 sm_80: the decoder layers' delayed mHC runs through the "
        "sm_80 mHC kernel A (VLLM_DEEPSEEK_V41_FUSED_MHC=1), finish %s",
        "kernel (VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH=1)" if sm80_mhc_fused.FINISH_KERNEL
        else "in torch",
        scope="global",
    )
    if sm80_mhc_fused.FINISH_V2:
        why = sm80_mhc_fused.finish_v2_unavailable(
            config.hc_mult, config.hidden_size, config.hc_sinkhorn_iters
        )
        if why is None:
            from vllm_cmp170hx.models.deepseek_v41.ops.sm80_mhc_finish_v2 import (
                FINISH_V2_WARMUP,
            )

            FINISH_V2_WARMUP.register_warmup()
            logger.info_once(
                "DeepSeek-V4.1 sm_80: delayed mHC finish v2 live "
                "(VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2=1): kernel A with its "
                "staging reuse, then _delayed_finish_v2_kernel, for 1..2048 rows",
                scope="global",
            )
        else:
            sm80_mhc_fused.FINISH_V2 = False
            logger.info_once(
                "DeepSeek-V4.1 sm_80: VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2 is set "
                "but its gate is closed (%s); the finish v2 does not run",
                why,
                scope="global",
            )
    return sm80_mhc_fused


class DeepseekV4DecoderLayer(upstream_model.DeepseekV4DecoderLayer):
    """Decoder layer whose mHC steps run the module :func:`mhc_ops` picks
    (``self._mhc_ops``).

    ``forward`` is the upstream one for the sm_80 topology: the first layer
    takes the repeated embedding (its coefficients from the whole stream),
    an Engram layer injects into the previous layer's post, every other
    boundary is the previous sublayer's post fused with the next delayed pre.
    With ``_split_reduce`` (VLLM_SM80_PREFILL_REDUCE_OVERLAP) an eager
    step of enough rows runs everything after the attention in two row slices
    (``sm80_split_reduce.py``). With ``_mhc_side``
    (VLLM_DEEPSEEK_V41_MHC_SIDE_STREAM) the forward runs inside
    ``sm80_mhc_side.LAYER``: the mHC finish's coefficient CTAs go beside the
    sublayers' all-reduces (``ops/sm80_mhc_side.py``).
    """

    _split_reduce: bool = False
    # sm80_mhc_side.LAYER, or None
    _mhc_side = None

    @classmethod
    def adopt(cls, layer: upstream_model.DeepseekV4DecoderLayer, config) -> None:
        assert type(layer) is upstream_model.DeepseekV4DecoderLayer, type(layer)
        if layer.use_sequence_parallel or layer.fuse_mhc_all_reduce:
            raise ValueError(
                "DeepSeek-V4.1 on sm_80 runs without sequence parallelism or "
                "the fused mHC all-reduce"
            )
        if layer.mhc_stream is not None:
            raise ValueError("DeepSeek-V4.1 on sm_80 has no mHC overlap stream")
        layer.__class__ = cls
        adopt_moe(layer.ffn, config)
        layer._mhc_ops = mhc_ops(config)
        if capture.active:
            layer._mhc_ops = _CapturedMhc(layer._mhc_ops, layer)
        if sm80_mhc_side.ACTIVE:
            layer._mhc_side = sm80_mhc_side.layer_scope(layer._mhc_ops)

    def _mixes(self, attn: bool) -> tuple:
        if attn:
            fn, scale, base = self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        else:
            fn, scale, base = self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        return (
            fn,
            scale,
            base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
        )

    def forward(self, *args, **kwargs):
        """:meth:`_forward_layer`, inside ``_mhc_side`` when it is set."""
        if self._mhc_side is None:
            return self._forward_layer(*args, **kwargs)
        with self._mhc_side:
            return self._forward_layer(*args, **kwargs)

    def _forward_layer(
        self,
        x,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
        *,
        capture_previous_aux: bool = False,
        mega_gate_metadata=None,
    ):
        previous_aux = None
        ops = self._mhc_ops
        attn_norm = {
            "norm_weight": self.attn_norm.weight,
            "norm_eps": self.attn_norm.variance_epsilon,
        }
        if residual is None:
            # Model entry: the embedding repeated hc_mult times; a None
            # pre-mix is the identity one (stream zero). A pipeline stage
            # entry receives the hc stream, into which an Engram layer
            # injects before this block's coefficients are computed.
            if x.dim() == 2:
                residual = x.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
            else:
                residual = x
                if self.engram is not None and engram_hashes is not None:
                    residual = self.engram(
                        residual,
                        engram_hashes[:, self.engram.layer_hash_index],
                        engram_mask,
                    )
            post_mix, res_mix, x, attn_pre = ops.mhc_pre_delayed(
                residual, *self._mixes(True), pre_mix=pre_mix, **attn_norm
            )
        elif self.engram is not None and engram_hashes is not None:
            # Engram injects into the previous sublayer's post, before this
            # block's coefficients are computed from the stream.
            previous_post = ops.mhc_post(x, residual, post_mix, res_mix)
            if capture_previous_aux:
                previous_aux = previous_post.mean(dim=1)
            residual = self.engram(
                previous_post,
                engram_hashes[:, self.engram.layer_hash_index],
                engram_mask,
            )
            post_mix, res_mix, x, attn_pre = ops.mhc_pre_delayed(
                residual, *self._mixes(True), pre_mix=pre_mix, **attn_norm
            )
        else:
            residual, post_mix, res_mix, x, attn_pre, aux = (
                ops.mhc_fused_post_pre_delayed(
                    x,
                    residual,
                    post_mix,
                    res_mix,
                    *self._mixes(True),
                    pre_mix=pre_mix,
                    capture_aux=capture_previous_aux,
                    **attn_norm,
                )
            )
            if capture_previous_aux:
                previous_aux = aux

        cut = (
            sm80_split_reduce.split_point(x, mega_gate_metadata)
            if self._split_reduce
            else None
        )
        if cut is not None:
            x, residual, post_mix, res_mix, ffn_pre = sm80_split_reduce.post_attention(
                self, positions, x, cut, residual, post_mix, res_mix, attn_pre, input_ids
            )
            return x, residual, post_mix, res_mix, ffn_pre, previous_aux

        x = self.attn(positions, x, None)

        # The FFN collapses with the pre-mix this layer's attention produced.
        residual, post_mix, res_mix, x, ffn_pre, _ = ops.mhc_fused_post_pre_delayed(
            x,
            residual,
            post_mix,
            res_mix,
            *self._mixes(False),
            pre_mix=attn_pre,
            norm_weight=self.ffn_norm.weight,
            norm_eps=self.ffn_norm.variance_epsilon,
        )
        x = self.ffn(x, input_ids, mega_gate_metadata)
        return x, residual, post_mix, res_mix, ffn_pre, previous_aux

    def write_kv(
        self,
        x,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        """Upstream ``write_kv`` with the sm_80 mHC."""
        assert self.engram is None
        *_, x, _, _ = self._mhc_ops.mhc_fused_post_pre_delayed(
            x,
            residual,
            post_mix,
            res_mix,
            *self._mixes(True),
            pre_mix=pre_mix,
            norm_weight=self.attn_norm.weight,
            norm_eps=self.attn_norm.variance_epsilon,
        )
        self.attn.forward_kv(positions, x)


class _CapturedMhc:
    """The layer's mHC ops with ``mhc_fused_post_pre_delayed`` recorded by
    the kernel call capture (vllm_cmp170hx/capture.py, family ``mhc``, point
    ``mhc_delayed``): the call's arguments by parameter name and its outputs
    (out_residual, out_post, out_comb, out_layer_input, out_pre). The prefix
    is the layer's, then ``hc_attn`` (the call before attention) or
    ``hc_ffn`` (before the FFN)."""

    def __init__(self, ops, layer):
        self._ops = ops
        self._layer = layer
        self._base = layer.attn.prefix.removesuffix(".attn")

    def __getattr__(self, name):
        return getattr(self._ops, name)

    def mhc_fused_post_pre_delayed(
        self, x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
        pre_mix=None, norm_weight=None, norm_eps=1e-6, capture_aux=False,
    ):
        which = "hc_attn" if fn is self._layer.hc_attn_fn else "hc_ffn"
        record = capture.begin(
            "mhc", "mhc_delayed", f"{self._base}.{which}", x.shape[0]
        )
        if record is not None:
            record.put(
                x=x, residual=residual, post_layer_mix=post_layer_mix,
                comb_res_mix=comb_res_mix, fn=fn, hc_scale=hc_scale,
                hc_base=hc_base, rms_eps=float(rms_eps),
                hc_pre_eps=float(hc_pre_eps),
                hc_sinkhorn_eps=float(hc_sinkhorn_eps),
                hc_post_mult_value=float(hc_post_mult_value),
                sinkhorn_repeat=int(sinkhorn_repeat), pre_mix=pre_mix,
                norm_weight=norm_weight, norm_eps=float(norm_eps),
            )
        out = self._ops.mhc_fused_post_pre_delayed(
            x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
            sinkhorn_repeat, pre_mix=pre_mix, norm_weight=norm_weight,
            norm_eps=norm_eps, capture_aux=capture_aux,
        )
        if record is not None:
            streams, post, comb, layer_input, pre, _ = out
            record.put(
                out_residual=streams, out_post=post, out_comb=comb,
                out_layer_input=layer_input, out_pre=pre,
            ).done()
        return out


def _adopt_layers(layers, config) -> None:
    for layer in layers:
        if not isinstance(layer, PPMissingLayer):
            DeepseekV4DecoderLayer.adopt(layer, config)


class DeepseekV4Model(upstream_model.DeepseekV4Model):
    """Backbone whose final post runs ``ops/sm80_mhc.py``; the last collapse
    stays ``hc_collapse_triton`` (the reference ``hc_pre`` order: FP32
    products summed per stream, one BF16 rounding)."""

    @classmethod
    def adopt(cls, model: upstream_model.DeepseekV4Model) -> None:
        assert type(model) is upstream_model.DeepseekV4Model, type(model)
        if model.use_sequence_parallel or model.fuse_mhc_all_reduce:
            raise ValueError(
                "DeepSeek-V4.1 on sm_80 runs without sequence parallelism or "
                "the fused mHC all-reduce"
            )
        model.__class__ = cls
        _adopt_layers(islice(model.layers, model.start_layer, model.end_layer), model.config)
        adopt_row_mask(islice(model.layers, model.start_layer, model.end_layer))
        if envs.VLLM_DEEPSEEK_V41_ENGRAM_EARLY_GATHER:
            from .sm80_engram_prefetch import adopt_early_engram

            adopt_early_engram(model)
        if envs.VLLM_DEEPSEEK_V41_ENGRAM_WKV_REPLICATED_PREFILL:
            from .sm80_engram_wkv import adopt_replicated_wkv

            adopt_replicated_wkv(model)
        if envs.VLLM_SM80_PREFILL_REDUCE_OVERLAP:
            sm80_split_reduce.adopt_split_reduce(
                islice(model.layers, model.start_layer, model.end_layer)
            )
        if envs.VLLM_SM80_ALLREDUCE:
            _init_ce_ring(model)
        replay = model.decoder_replay_layers
        if replay is not None:
            # Upstream bound the replay's KV write to the layer before its
            # adoption: that write_kv runs the TileLang mHC.
            replay.write_batch_kv = model.layers[model.decoder_replay_start].write_kv
        # Pipeline stages cut inside KV-sharing groups (sm80_pp_shadow.py).
        from vllm.config import get_current_vllm_config

        sm80_pp_shadow.attach_shadows(model, get_current_vllm_config())

    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        tensors = super().make_empty_intermediate_tensors(batch_size, dtype, device)
        for key, width in sm80_pp_shadow.receive_shapes(self).items():
            tensors[key] = torch.zeros((batch_size, width), dtype=dtype, device=device)
        return tensors

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds=None,
        lookback_token_ids=None,
        decode_step=None,
    ):
        sm80_pp_shadow.run_shadows(self, positions, intermediate_tensors)
        out = super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            lookback_token_ids=lookback_token_ids,
            decode_step=decode_step,
        )
        if isinstance(out, IntermediateTensors):
            out = sm80_pp_shadow.add_shipments(
                self, out, intermediate_tensors, positions.shape[0]
            )
        return out

    def _collapse(
        self,
        hidden_states,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        aux_hidden_by_layer: dict[int, torch.Tensor],
        full_num_tokens: int,
    ) -> torch.Tensor:
        assert isinstance(hidden_states, torch.Tensor)
        hidden_states = mhc_post(hidden_states, residual, post_mix, res_mix)
        if self.end_layer in self.aux_hidden_state_layers:
            aux_hidden_by_layer[self.end_layer] = hidden_states.mean(dim=1)
        return hidden_states


def _init_ce_ring(model) -> None:
    """VLLM_SM80_ALLREDUCE: build the TP group's copy-engine ring
    (collective), the transport of the layer-owned sums the host-staged and
    custom all-reduce do not take, sized for the largest FP32 ``wo_b`` sum of
    a step: ``max_num_batched_tokens`` rows of the hidden size in FP32.

    The ring copies into the other cards' memory over PCIe peer-to-peer, so it
    is built only under VLLM_CUSTOM_ALLREDUCE_PCIE_P2P (the hardware gate of
    every peer-to-peer path; the recipe's P2P check clears it where peer access
    does not work) and only where every rank can map the others' device IPC
    buffers. Without it those sums take the TP group's other transports
    (host-staged where it takes the message, else NCCL)."""
    from vllm.distributed.parallel_state import get_tp_group
    from vllm_cmp170hx.distributed.ce_ring_all_reduce import PeerBuffersUnavailable, init_ce_ring
    from vllm_cmp170hx.distributed.pcie_p2p import pcie_p2p_custom_allreduce_allowed

    group = get_tp_group()
    if group.world_size < 2:
        return
    off = "DeepSeek-V4.1 sm_80: copy-engine ring off (%s); prefill-sized TP all-reduces take the other transports"
    if not pcie_p2p_custom_allreduce_allowed():
        logger.info_once(off, "no PCIe peer access: VLLM_CUSTOM_ALLREDUCE_PCIE_P2P is off", scope="global")
        return
    rows = model.vllm_config.scheduler_config.max_num_batched_tokens
    try:
        ring = init_ce_ring(group, rows * model.config.hidden_size * 4)
    except PeerBuffersUnavailable as exc:
        logger.warning_once(off, str(exc), scope="global")
        return
    logger.info_once(
        "DeepSeek-V4.1 sm_80: prefill-sized TP all-reduces run on the copy-engine "
        "ring (VLLM_SM80_ALLREDUCE=1): messages up to %d bytes, "
        "%.1f MiB of device memory per rank",
        ring.capacity_bytes,
        ring.device_bytes / 2**20,
        scope="global",
    )


class DeepseekV41ForCausalLM(
    target_argmax.ShardArgmaxLogits, upstream_vl.DeepseekV41ForCausalLM
):
    """The DeepSeek-V4.1 vision wrapper with the sm_80 text model.

    The vision tower and aligner stay upstream's (BF16, replicated: EXL3 runs
    them data-parallel, ``--mm-encoder-tp-mode data``); their encoder
    attention is the platform's ViT backend (FLASH_ATTN on sm_80)."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        DeepseekV4Model.adopt(self.language_model.model)
        # The wrappers bound the upstream method before the adoption; the
        # pipeline receive buffers include the shadow source inputs.
        self.language_model.make_empty_intermediate_tensors = (
            self.language_model.model.make_empty_intermediate_tensors
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        # Upstream builds the head unquantized (the base checkpoint's is
        # BF16); the EXL3 checkpoint's head is packed, so it is built with
        # the model's quantization config.
        config = self.language_model.config
        from vllm.distributed import get_pp_group

        if not get_pp_group().is_first_rank:
            # Images are encoded on the first pipeline stage only; later
            # stages keep no vision tower or aligner (about 1 GB each).
            self.vision = PPMissingLayer()
            self.aligner = PPMissingLayer()

        # Only the last pipeline stage holds the head (PPMissingLayer elsewhere).
        if get_pp_group().is_last_rank:
            self.language_model.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{maybe_prefix(prefix, 'language_model')}.lm_head",
            )
        Exl3HeadLogitsProcessor.adopt(self.language_model.logits_processor)

    def load_weights(self, weights) -> set[str]:
        """Upstream's; with ``VLLM_DEEPSEEK_V41_EMBED_INT8=1`` the target's
        input embedding (first pipeline stage) is then held as INT8 rows
        (:class:`Int8RowEmbedding`, 0.66 GB instead of 1.32 GB). Target
        numerics change (row-wise INT8 rounding of the embedding): judge it
        with fidelity-v1 before serving it."""
        loaded = super().load_weights(weights)
        backbone = self.language_model.model
        if (
            os.environ.get("VLLM_DEEPSEEK_V41_EMBED_INT8") == "1"
            and any(n.endswith("model.embed_tokens.weight") for n in loaded)
            and not isinstance(backbone.embed_tokens, (PPMissingLayer, Int8RowEmbedding))
        ):
            backbone.embed_tokens = Int8RowEmbedding.from_embedding(backbone.embed_tokens)
        return loaded


def sm80_writes_context_kv(model: upstream_dspark.DSparkDeepseekV4Model) -> bool:
    """Whether the DSpark drafter ``model``'s stages are sm_80 layers, which
    write their context window rows themselves, each from its own ``wkv``
    (:meth:`DSparkDeepseekV4Model.precompute_and_store_context_kv`). Every
    stage takes its class from the same selector, so the first one decides."""
    layers = list(model.layers)
    return bool(layers) and isinstance(layers[0].attn, DeepseekV41Sm80Attention)


class DSparkDeepseekV4Model(upstream_dspark.DSparkDeepseekV4Model):
    """DSpark stages whose mHC runs ``ops/sm80_mhc.py`` and whose sm_80 draft
    layers write their context window rows themselves."""

    @classmethod
    def adopt(
        cls, model: upstream_dspark.DSparkDeepseekV4Model, keep_embed: bool = False
    ) -> None:
        assert type(model) is upstream_dspark.DSparkDeepseekV4Model, type(model)
        if model.use_sequence_parallel or model.use_native_mega_moe:
            raise ValueError(
                "DeepSeek-V4.1 DSpark on sm_80 runs without sequence "
                "parallelism or MegaMoE"
            )
        model.__class__ = cls
        _adopt_layers(model.layers, model.config)
        if not keep_embed:
            # Aliased from the target by the DSpark loader (load_dspark_model).
            model.embed_tokens = PPMissingLayer()
        if sm80_writes_context_kv(model) and hasattr(model, "context_wkv_proj"):
            # The sm_80 stages write their context KV from their own wkv
            # (precompute_and_store_context_kv), so the stacked context
            # projection never runs: drop it, and its checkpoint copies at load
            # (DSparkDeepseekV4ForCausalLM._remap_dspark_name).
            del model.context_wkv_proj

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        residual = post_mix = res_mix = pre_mix = None
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix, pre_mix, _ = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
            )
        hidden_states = mhc_post(hidden_states, residual, post_mix, res_mix)
        # The pre-norm head hidden: the last FFN pre-mix collapse
        # (compute_logits applies the norm).
        assert pre_mix is not None
        return hc_collapse_triton(hidden_states, pre_mix)

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        main_x: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mappings: list[torch.Tensor | None] | None = None,
    ) -> None:
        """sm_80 draft layers store their context KV through
        :meth:`DeepseekV41Sm80Attention.sm80_store_context_kv` (the target's
        rounding); other layers take upstream's insertion."""
        if not sm80_writes_context_kv(self):
            return super().precompute_and_store_context_kv(
                main_x, context_positions, context_slot_mappings
            )
        for i, layer in enumerate(self.layers):
            slot_mapping = (
                None if context_slot_mappings is None else context_slot_mappings[i]
            )
            layer.attn.sm80_store_context_kv(main_x, context_positions, slot_mapping)
        return None


class DSparkDeepseekV4ForCausalLM(upstream_dspark.DSparkDeepseekV4ForCausalLM):
    """The DeepSeek-V4.1 DSpark drafter with the sm_80 stages."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        from vllm.distributed import get_pp_group

        # Under pipeline parallelism the drafter runs on the last stage, which
        # holds the target's head but not its input embedding: the drafter
        # keeps its own embedding there and loads it from the checkpoint's
        # ``embed`` (load_dspark_model then leaves it alone).
        self.has_own_embed_tokens = get_pp_group().world_size > 1
        DSparkDeepseekV4Model.adopt(self.model, keep_embed=self.has_own_embed_tokens)
        # Aliased from the target by the DSpark loader (load_dspark_model).
        self.lm_head = PPMissingLayer()
        # Base logits through the target's EXL3 head; the Markov bias through
        # the upstream unquantized-head path, in the head dtype too.
        Exl3HeadLogitsProcessor.adopt(self.logits_processor)

    def compute_logits_local(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's vocabulary shard of ``compute_logits`` (no all-gather,
        padding columns kept), for the per-rank greedy argmax
        (``vllm_cmp170hx/spec_decode/shard_argmax.py``)."""
        return self.logits_processor(
            self.lm_head, self.model.norm(hidden_states), skip_gather=True
        )

    def _remap_dspark_name(self, name: str) -> str | None:
        """Upstream's mapping; the stacked context-WKV copies of the stage
        ``wkv`` weights load nowhere once the drafter has dropped that
        projection (:meth:`DSparkDeepseekV4Model.adopt`)."""
        if name.startswith("context_wkv_proj.") and not hasattr(
            self.model, "context_wkv_proj"
        ):
            return None
        if name.startswith("embed.") and self.has_own_embed_tokens:
            return "model.embed_tokens." + name.removeprefix("embed.")
        return super()._remap_dspark_name(name)

    def load_weights(self, weights) -> set[str]:
        """Upstream ``load_weights``, without dropping the confidence head
        when one call does not carry it: the EXL3 loader passes one tensor
        per call and refuses a checkpoint whose head parameters stay unloaded.
        """
        confidence_head = self.model.confidence_head
        loaded = super().load_weights(weights)
        self.model.confidence_head = confidence_head
        if (
            "model.embed_tokens.weight" in loaded
            and self.has_own_embed_tokens
            and os.environ.get("VLLM_DEEPSEEK_V41_DSPARK_EMBED_INT8") == "1"
        ):
            self.model.embed_tokens = Int8RowEmbedding.from_embedding(
                self.model.embed_tokens
            )
        return loaded


class Int8RowEmbedding(torch.nn.Module):
    """The DSpark drafter's own input-embedding copy on the last pipeline
    stage, stored as INT8 rows with one FP32 scale per row (half the bytes
    of the BF16 table: 0.66 GB less on the most loaded stage). A lookup
    gathers the rows and scales them back to BF16. Only the drafter's
    embedding is stored this way (draft numerics, not the target's), with
    ``VLLM_DEEPSEEK_V41_DSPARK_EMBED_INT8=1``."""

    def __init__(self, q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
        super().__init__()
        self.register_buffer("q", q, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.out_dtype = dtype

    @classmethod
    def from_embedding(cls, embedding: torch.nn.Module) -> "Int8RowEmbedding":
        weight = embedding.weight.data
        if getattr(embedding, "tp_size", 1) != 1:
            raise ValueError("The INT8 drafter embedding runs at TP=1 only")
        amax = weight.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        scale = amax / 127.0
        q = torch.round(weight.float() / scale).clamp_(-127, 127).to(torch.int8)
        logger.info_once(
            "DeepSeek-V4.1 DSpark: the drafter's input embedding is INT8 rows "
            "(%.2f GB instead of %.2f GB)",
            q.numel() / 1e9, weight.numel() * weight.element_size() / 1e9,
            scope="global",
        )
        return cls(q, scale.squeeze(1).contiguous(), weight.dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows = self.q.index_select(0, input_ids.reshape(-1).long())
        out = rows.float() * self.scale.index_select(0, input_ids.reshape(-1).long())[:, None]
        return out.to(self.out_dtype).reshape(*input_ids.shape, -1)
