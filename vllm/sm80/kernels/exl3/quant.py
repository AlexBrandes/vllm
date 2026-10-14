# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 packed linear and whole-expert inference with BF16 activations."""

import re
from functools import cache
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sm80.kernels.exl3.utils import (
    Exl3Manifest,
    PackedMatrix,
    aligned_shard,
)

logger = init_logger(__name__)

# Qwen4Exp QSA projects Q/gate, K, V and the replicated indexer Q/K in one
# layer. It first builds a three-output QKV layer, then replaces it with the
# four-output fused one, and the target, MTP and LoRA packed mappings disagree
# on the indexer, so each layer's output count selects its members.
_QSA_QKV_MEMBERS = ("q_proj", "k_proj", "v_proj", "indexer.index_qk_proj")


def _verify_launch_options(vllm_config) -> None:
    """Refuse a model built without the EXL3 launch options.

    EXL3 weights load only through the "exl3" load format (Exl3ModelLoader),
    and EXL3 vision towers run data-parallel. Without the options vLLM would
    pick the default loader or shard the vision tower by weights, so fail
    instead. initialize_model sets the current config while a model is
    built; a layer built outside a model (unit tests) has none to check.
    """
    if vllm_config is None:
        return
    load_format = vllm_config.load_config.load_format
    if load_format != "exl3":
        raise ValueError(
            f"EXL3 checkpoints need --load-format exl3, not {load_format!r}"
        )
    model_config = vllm_config.model_config
    multimodal = model_config.multimodal_config
    # The MTP draft config carries no launch options of its own.
    if (
        multimodal is not None
        and model_config.runner_type != "draft"
        and multimodal.mm_encoder_tp_mode != "data"
    ):
        raise ValueError(
            "EXL3 vision towers need --mm-encoder-tp-mode data, not "
            f"{multimodal.mm_encoder_tp_mode!r}"
        )


@cache
def _require_sm80() -> None:
    """The extension is built for SM 8.0 only; refuse other GPUs up front."""
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    if capability is not None and capability.to_int() != 80:
        raise ValueError(
            "EXL3 kernels are built for SM 8.0 only; this GPU is SM "
            f"{capability.major}.{capability.minor}"
        )


class Exl3Config(QuantizationConfig):
    """Registered as "exl3" by vllm/sm80/plugin.py.

    Upstream model code consults the attributes and methods below through
    getattr, so they apply only to this config.
    """

    min_transform_size = 128
    # Qwen3-Next shared experts: TP shards keep whole transform blocks.
    shared_expert_shard_group_size = min_transform_size

    def __init__(self):
        super().__init__()
        self.ple_placement = envs.VLLM_EXL3_PLE_PLACEMENT
        self.manifest: Exl3Manifest | None = None
        self.mtp_start_layer_idx = 0
        self.mapped_prefixes: dict[str, str] = {}
        self.is_qwen = False

    @classmethod
    def get_name(cls) -> Literal["exl3"]:
        return "exl3"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> None:
        """Refuse EXL3 off CUDA while ModelConfig verifies quantization.

        ModelConfig asks every registered method before it checks the
        platform; EXL3 never replaces the checkpoint's method.
        """
        if hf_quant_cfg.get("quant_method") == "exl3" or user_quant == "exl3":
            from vllm.platforms import current_platform

            if not current_platform.is_cuda():
                raise ValueError("EXL3 execution requires the NVIDIA CUDA platform")
        return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls()

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        if not Path(model_name).is_dir():
            raise ValueError("Download the EXL3 safetensors checkpoint before loading")
        is_qwen = getattr(hf_config, "model_type", "") in (
            "qwen4_exp",
            "qwen4_exp_text",
            "qwen4_exp_mtp",
        )
        self.manifest = Exl3Manifest(model_name, require_qwen_aux=is_qwen)
        self.is_qwen = is_qwen
        text = getattr(hf_config, "text_config", hf_config)
        self.mtp_start_layer_idx = getattr(text, "num_hidden_layers", 0)

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        if self.manifest is None:
            raise ValueError("EXL3 name mapping requires a checkpoint manifest")
        mapped_prefixes = {}
        for source in self.manifest.matrices:
            mapped = hf_to_vllm_mapper.map_name(f"{source}.weight")
            if mapped is None:
                continue
            mapped = mapped.removesuffix(".weight")
            if mapped in mapped_prefixes and mapped_prefixes[mapped] != source:
                raise ValueError(f"Conflicting EXL3 module mapping: {mapped}")
            mapped_prefixes[mapped] = source
        self.mapped_prefixes = mapped_prefixes

    def get_ple_embedding_method(self):
        """Qwen4Exp PLE method (vllm/models/qwen4_exp/common/ngram_embedding.py)."""
        from vllm.models.qwen4_exp.nvidia.exl3_ple import Exl3PLEEmbeddingMethod

        return Exl3PLEEmbeddingMethod(self)

    @property
    def ple_embedding_cls(self):
        """Qwen4Exp PLE table (vllm/models/qwen4_exp/nvidia/ngram_embedding.py)."""
        from vllm.models.qwen4_exp.nvidia.exl3_ple import Exl3PLEEmbedding

        return Exl3PLEEmbedding

    def set_mtp_start_layer_idx(self, index: int) -> None:
        """MTP draft layers are numbered after the target's (nvidia/mtp.py)."""
        self.mtp_start_layer_idx = index

    @staticmethod
    def mtp_shares_target_embeddings(config) -> bool:
        """Whether the Qwen4Exp MTP draft reuses the target embedding and head
        (nvidia/mtp.py builds PPMissingLayer placeholders for them)."""
        return not getattr(config, "mtp_use_dedicated_embeddings", False)

    def release_manifest(self) -> None:
        """Drop the checkpoint inventory once this config's model is loaded."""
        self.manifest = None
        self.mapped_prefixes = {}

    def source_prefix(self, prefix: str) -> str:
        """Resolve model construction prefixes to checkpoint module names."""
        assert self.manifest is not None, (
            "EXL3 manifest must precede model construction"
        )
        if prefix in self.mapped_prefixes:
            return self.mapped_prefixes[prefix]
        if not self.is_qwen:
            return prefix
        if prefix.startswith("model.language_model.model."):
            prefix = prefix.replace(
                "model.language_model.model.", "model.language_model.", 1
            )
        elif prefix.startswith("language_model.model."):
            prefix = prefix.replace("language_model.model.", "model.language_model.", 1)
        elif prefix.startswith(("model.layers.", "model.embed_tokens.")):
            prefix = "model.language_model." + prefix.removeprefix("model.")
        for head_prefix in ("model.language_model.lm_head", "language_model.lm_head"):
            if prefix == head_prefix:
                prefix = "lm_head"
        if prefix.startswith("visual."):
            prefix = "model." + prefix
        if prefix.startswith("model.mtp."):
            prefix = prefix.removeprefix("model.")
        if prefix.startswith("mtp.layers."):
            match = re.match(r"mtp.layers.(\d+)\.(.*)", prefix)
            assert match is not None
            index = int(match[1])
            if index >= self.mtp_start_layer_idx:
                prefix = f"mtp.layers.{index - self.mtp_start_layer_idx}.{match[2]}"
        return prefix

    def constituents(self, prefix: str, outputs: int = 0) -> list[PackedMatrix]:
        assert self.manifest is not None
        source = self.source_prefix(prefix)
        if source in self.manifest.matrices:
            return [self.manifest.matrices[source]]
        parent, _, leaf = source.rpartition(".")
        members = self.packed_modules_mapping.get(leaf, [leaf])
        if leaf == "qkv" and ".visual." in source:
            members = ["q_proj", "k_proj", "v_proj"]
        elif leaf == "qkv_proj" and self.is_qwen and outputs:
            members = list(_QSA_QKV_MEMBERS[:outputs])
        names = [
            self.source_prefix(f"{parent}.{member}" if parent else member)
            for member in members
        ]
        found = [
            self.manifest.matrices[n] for n in names if n in self.manifest.matrices
        ]
        if found and len(found) != len(names):
            raise ValueError(f"Mixed plain/packed EXL3 fused projection: {prefix}")
        return found

    def get_quant_method(self, layer: nn.Module, prefix: str):
        from vllm.config import get_current_vllm_config_or_none

        _verify_launch_options(get_current_vllm_config_or_none())
        _require_sm80()
        if isinstance(layer, RoutedExperts):
            return Exl3MoEMethod(self, prefix, layer.moe_config)
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            matrices = self.constituents(
                prefix, len(getattr(layer, "output_sizes", ()))
            )
            if matrices:
                return Exl3LinearMethod(matrices)
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
        return None


class Exl3PackedWeight(nn.Module):
    def __init__(
        self,
        matrix: PackedMatrix,
        logical_in: int,
        logical_out: int,
        input_slice: slice,
        output_slices: list[slice],
    ):
        super().__init__()
        for part, limit in [(input_slice, matrix.padded_in)] + [
            (part, matrix.padded_out) for part in output_slices
        ]:
            if (
                part.step is not None
                or not 0 <= part.start < part.stop <= limit
                or part.start % 128
                or part.stop % 128
            ):
                raise ValueError("EXL3 shards must select complete H128 blocks")
        self.matrix = matrix
        self.logical_in = logical_in
        self.logical_out = logical_out
        self.input_slice = input_slice
        self.output_slices = output_slices
        self.padded_in = input_slice.stop - input_slice.start
        self.padded_out = sum(s.stop - s.start for s in output_slices)
        if (
            not 0 < logical_in <= self.padded_in
            or not 0 < logical_out <= self.padded_out
        ):
            raise ValueError("EXL3 logical dimensions exceed assigned packed blocks")
        self.trellis = nn.Parameter(
            torch.empty(
                self.padded_in // 16,
                self.padded_out // 16,
                8 * matrix.bits_x2,
                dtype=torch.int16,
            ),
            requires_grad=False,
        )
        self.suh = nn.Parameter(
            torch.empty(self.padded_in, dtype=torch.float16), requires_grad=False
        )
        self.svh = nn.Parameter(
            torch.empty(self.padded_out, dtype=torch.float16), requires_grad=False
        )
        self.loaded = False

    def load(self, files, ledger):
        manifest = ledger.manifest
        prefix = self.matrix.prefix
        ks = slice(self.input_slice.start // 16, self.input_slice.stop // 16)
        header = manifest.tensors[f"{prefix}.trellis"]
        packed = files[header.path].get_slice(header.name)
        offset = 0
        for output_slice in self.output_slices:
            ns = slice(output_slice.start // 16, output_slice.stop // 16)
            width = ns.stop - ns.start
            self.trellis.data[:, offset : offset + width].copy_(packed[ks, ns, :])
            offset += width
        for suffix, param, slices in (
            ("suh", self.suh, [self.input_slice]),
            ("svh", self.svh, self.output_slices),
        ):
            header = manifest.tensors[f"{prefix}.{suffix}"]
            data = files[header.path].get_slice(header.name)
            offset = 0
            for part in slices:
                width = part.stop - part.start
                param.data[offset : offset + width].copy_(data[part])
                offset += width
        for key in self.matrix.keys:
            ledger.mark(key, "consumed", prefix)
        self.loaded = True


class Exl3LinearMethod(LinearMethodBase):
    def __init__(self, matrices: list[PackedMatrix]):
        self.matrices = matrices

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16:
            raise ValueError("EXL3 requires BF16 activations")
        rank, size = layer.tp_rank, layer.tp_size
        if isinstance(layer, ReplicatedLinear):
            rank, size = 0, 1
        row_parallel = isinstance(layer, RowParallelLinear)
        parts = list(output_partition_sizes)
        gdn = self.matrices[0].prefix.endswith(".in_proj_qkv")
        if gdn:
            if len(parts) != 4 or len(self.matrices) != 2:
                raise ValueError("EXL3 GDN requires separate Q/K/V/Z logical slices")
            groups = [parts[:3], parts[3:]]
        else:
            groups = [[n] for n in parts]
        if len(groups) != len(self.matrices):
            raise ValueError("EXL3 packed constituents do not match linear outputs")
        # The fused QSA layer's K/V follow the KV heads like QKVParallelLinear;
        # its indexer Q/K is replicated on every rank.
        qsa_indexer = len(self.matrices) == 4 and self.matrices[3].prefix.endswith(
            ".indexer.index_qk_proj"
        )
        kv_heads = isinstance(layer, QKVParallelLinear) or qsa_indexer
        weights = []
        for index, (matrix, group) in enumerate(zip(self.matrices, groups)):
            output_rank, output_size_tp = rank, size
            if row_parallel or (qsa_indexer and index == 3):
                output_rank, output_size_tp = 0, 1
            elif kv_heads and index in (1, 2):
                output_rank //= layer.num_kv_head_replicas
                output_size_tp //= layer.num_kv_head_replicas
            logical_output = sum(group) * output_size_tp
            matrix.validate_logical(input_size, logical_output)
            input_slice = aligned_shard(
                input_size,
                matrix.padded_in,
                rank if row_parallel else 0,
                size if row_parallel else 1,
            )
            output_slices = []
            start = 0
            for local_width in group:
                full_width = local_width * output_size_tp
                padded_width = matrix.padded_out if len(group) == 1 else full_width
                part = aligned_shard(
                    full_width, padded_width, output_rank, output_size_tp
                )
                output_slices.append(slice(start + part.start, start + part.stop))
                start += full_width
            weights.append(
                Exl3PackedWeight(
                    matrix,
                    input_size_per_partition,
                    sum(group),
                    input_slice,
                    output_slices,
                )
            )
        layer.exl3_weights = nn.ModuleList(weights)

    def process_weights_after_loading(self, layer):
        from vllm.sm80.kernels.exl3 import ops as exl3_ops

        self.row_chunks = []
        for weight in layer.exl3_weights:
            if not weight.loaded:
                raise ValueError(f"Unloaded EXL3 matrix: {weight.matrix.prefix}")
            if not all(p.is_cuda for p in weight.parameters()):
                raise ValueError("EXL3 packed weights and scales must remain on CUDA")
            self.row_chunks.append(
                exl3_ops.linear_row_chunk(weight.padded_in, weight.padded_out)
            )

    def apply(self, layer, x, bias=None):
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1]).contiguous()
        outputs = [
            torch.ops.vllm.exl3_linear(
                x,
                weight.trellis,
                weight.suh,
                weight.svh,
                weight.matrix.bits_x2,
                weight.matrix.codebook,
                weight.logical_out,
                row_chunk,
            )
            for weight, row_chunk in zip(layer.exl3_weights, self.row_chunks)
        ]
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if bias is not None:
            output.add_(bias)
        return output.reshape(*shape, output.shape[-1])


class Exl3MoEMethod(FusedMoEMethodBase):
    def __init__(self, config, prefix, moe):
        super().__init__(moe)
        self.config = config
        self.prefix = config.source_prefix(f"{prefix}.0.gate_proj").removesuffix(
            ".0.gate_proj"
        )
        if moe.moe_parallel_config.tp_size != 1:
            raise ValueError("EXL3 routed experts require whole-expert EP, not TP")
        if moe.has_bias or moe.activation != MoEActivation.SILU:
            raise ValueError("EXL3 experts require bias-free SiLU gate/up/down")
        if moe.is_lora_enabled:
            raise ValueError("EXL3 experts do not support LoRA")
        if any(
            value is not None
            for value in (
                moe.swiglu_limit,
                moe.swiglu_alpha,
                moe.swiglu_beta,
            )
        ):
            raise ValueError("EXL3 experts require the standard SiLU activation")

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16 or layer.apply_router_weight_on_input:
            raise ValueError("EXL3 MoE requires BF16 and routing weights after down")
        assert self.config.manifest is not None
        self.intermediate_size = intermediate_size_per_partition
        self.local_experts = num_experts
        self.hidden_size = hidden_size
        weights = [[] for _ in range(num_experts)]
        self.global_expert_ids = [-1] * num_experts
        self.nonlocal_keys = []
        for expert in range(layer.global_num_experts):
            local = layer._map_global_expert_id_to_local_expert_id(expert)
            for role in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"{self.prefix}.{expert}.{role}"
                matrix = self.config.manifest.matrices.get(prefix)
                if matrix is None:
                    raise ValueError(f"Missing EXL3 expert matrix: {prefix}")
                k, n = hidden_size, intermediate_size_per_partition
                if role == "down_proj":
                    k, n = n, k
                matrix.validate_logical(k, n)
                if local < 0:
                    self.nonlocal_keys.extend(matrix.keys)
                else:
                    self.global_expert_ids[local] = expert
                    weights[local].append(
                        Exl3PackedWeight(
                            matrix,
                            k,
                            n,
                            slice(0, matrix.padded_in),
                            [slice(0, matrix.padded_out)],
                        )
                    )
        if any(len(expert) != 3 for expert in weights):
            raise ValueError("EXL3 expert assignment is incomplete or duplicated")
        layer.exl3_experts = nn.ModuleList(nn.ModuleList(w) for w in weights)

    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig.make()

    def process_weights_after_loading(self, layer):
        from vllm.model_executor.layers.fused_moe.all2all_utils import (
            maybe_make_prepare_finalize,
        )
        from vllm.sm80.kernels.exl3 import ops as exl3_ops

        pointers, metadata = [], []
        for expert in layer.exl3_experts:
            ptrs, meta = [], []
            for weight in expert:
                if not weight.loaded or not all(p.is_cuda for p in weight.parameters()):
                    raise ValueError("EXL3 expert weights must be loaded on CUDA")
                ptrs.append(
                    [
                        weight.trellis.data_ptr(),
                        weight.suh.data_ptr(),
                        weight.svh.data_ptr(),
                    ]
                )
                meta.append(
                    [
                        weight.padded_in,
                        weight.padded_out,
                        weight.matrix.bits_x2,
                        weight.matrix.codebook,
                    ]
                )
            pointers.append(ptrs)
            metadata.append(meta)
        device = layer.exl3_experts[0][0].trellis.device
        layer.register_buffer(
            "exl3_ptrs", torch.tensor(pointers, dtype=torch.int64, device=device)
        )
        layer.register_buffer(
            "exl3_meta", torch.tensor(metadata, dtype=torch.int64, device=device)
        )
        if layer.expert_map is None:
            layer.register_buffer(
                "exl3_expert_map",
                torch.arange(
                    layer.global_num_experts, dtype=torch.int32, device=device
                ),
            )
        else:
            layer.register_buffer("exl3_expert_map", layer.expert_map)
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        prepare = maybe_make_prepare_finalize(
            moe=self.moe,
            quant_config=self.moe_quant_config,
            routing_tables=layer._expert_routing_tables(),
            allow_new_interface=True,
        )
        max_rows = dispatched_rows(prepare, self.moe)
        workspace = exl3_ops.moe_workspace_size(
            max_rows,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
            self.local_experts,
        )
        moe_v2_rows = 0
        if envs.VLLM_EXL3_MOE_V2:
            moe_v2_rows, workspace = self._moe_v2(layer, metadata, max_rows, workspace)
        prefill_decode_len = 0
        if envs.VLLM_EXL3_MOE_PREFILL_TC:
            prefill_decode_len, workspace = self._moe_prefill(
                layer, metadata, max_rows, workspace
            )
        experts = Exl3Experts(
            self.moe,
            self.moe_quant_config,
            self.intermediate_size,
            layer.exl3_expert_map,
            workspace,
            max_rows,
            moe_v2_rows,
            prefill_decode_len,
        )
        self.moe_kernel = mk.FusedMoEKernel(prepare, experts)

    def _moe_v2(self, layer, metadata, max_rows: int, workspace: int):
        """Rows up to which this layer's calls take moe_out_v2, and a workspace
        size that fits both operators (0 rows: the gate is closed)."""
        from vllm.sm80.kernels.exl3 import ops as exl3_ops

        device = layer.exl3_ptrs.device
        reason = exl3_ops.moe_v2_closed_reason(
            tuple(torch.cuda.get_device_capability(device)),
            self.moe.in_dtype,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
            self.local_experts,
            layer.global_num_experts,
            metadata,
        )
        if reason is not None:
            logger.info_once(
                "VLLM_EXL3_MOE_V2 is set but the gate is closed: %s; these MoE "
                "layers keep moe_out.",
                reason,
            )
            return 0, workspace
        rows = min(max_rows, exl3_ops.MOE_V2_MAX_ROWS)
        workspace = max(
            workspace,
            exl3_ops.moe_workspace_size_v2(
                rows,
                self.hidden_size,
                self.intermediate_size,
                layer.top_k,
                self.local_experts,
            ),
        )
        exl3_ops.warmup_moe_v2(
            layer.exl3_ptrs,
            layer.exl3_meta,
            layer.exl3_expert_map,
            self.global_expert_ids,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
        )
        logger.info_once(
            "EXL3 MoE v2 kernel active (VLLM_EXL3_MOE_V2): tensor-core routed "
            "experts for MoE calls of 1..%d rows; larger calls keep moe_out.",
            rows,
        )
        return rows, workspace

    def _moe_prefill(self, layer, metadata, max_rows: int, workspace: int):
        """The most tokens one request has in a decode step, above which a
        step is a prefill step whose calls of more than 64 rows take
        moe_out_prefill, and a workspace size that fits every operator (0:
        the gate is closed)."""
        from vllm.config import get_current_vllm_config
        from vllm.sm80.kernels.exl3 import ops as exl3_ops
        from vllm.sm80.kernels.exl3.step_phase import max_decode_query_len

        device = layer.exl3_ptrs.device
        # moe_out_prefill covers the configuration moe_out_v2 covers.
        reason = exl3_ops.moe_v2_closed_reason(
            tuple(torch.cuda.get_device_capability(device)),
            self.moe.in_dtype,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
            self.local_experts,
            layer.global_num_experts,
            metadata,
        )
        if reason is None and max_rows < exl3_ops.MOE_PREFILL_MIN_ROWS:
            reason = f"MoE calls have at most {max_rows} rows"
        if reason is not None:
            logger.info_once(
                "VLLM_EXL3_MOE_PREFILL_TC is set but the gate is closed: %s; "
                "these MoE layers keep moe_out in prefill steps.",
                reason,
            )
            return 0, workspace
        workspace = max(
            workspace,
            exl3_ops.moe_workspace_size_prefill(
                max_rows,
                self.hidden_size,
                self.intermediate_size,
                layer.top_k,
                self.local_experts,
            ),
        )
        exl3_ops.warmup_moe_prefill(
            layer.exl3_ptrs,
            layer.exl3_meta,
            layer.exl3_expert_map,
            self.global_expert_ids,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
        )
        decode_len = max_decode_query_len(get_current_vllm_config())
        logger.info_once(
            "EXL3 MoE prefill kernel active (VLLM_EXL3_MOE_PREFILL_TC): "
            "tensor-core routed experts for MoE calls of %d..%d rows in prefill "
            "steps (a request scheduled more than %d tokens); decode steps and "
            "calls of at most %d rows keep their kernels.",
            exl3_ops.MOE_PREFILL_MIN_ROWS,
            max_rows,
            decode_len,
            exl3_ops.MOE_PREFILL_MIN_ROWS - 1,
        )
        return decode_len, workspace

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.exl3_ptrs,
            w2=layer.exl3_meta,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=False,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.exl3_expert_map,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


class Exl3Experts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe,
        quant,
        intermediate_size,
        expert_map,
        workspace,
        max_rows,
        moe_v2_rows=0,
        prefill_decode_len=0,
    ):
        super().__init__(moe, quant)
        self.intermediate_size = intermediate_size
        self.expert_map = expert_map
        self.workspace_elements = workspace
        self.max_rows = max_rows
        # Calls of 1..moe_v2_rows rows take moe_out_v2 (VLLM_EXL3_MOE_V2).
        self.moe_v2_rows = moe_v2_rows
        # Nonzero (VLLM_EXL3_MOE_PREFILL_TC with its gate open): calls of more
        # than 64 rows in a step that schedules some request more than this
        # many tokens take moe_out_prefill.
        self.prefill_decode_len = prefill_decode_len

    @staticmethod
    def activation_format():
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device():
        from vllm.platforms import current_platform

        return current_platform.is_cuda()

    @staticmethod
    def _supports_no_act_and_mul():
        return False

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation):
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config):
        return moe_parallel_config.tp_size == 1

    def moe_problem_size(self, a1, w1, w2, topk_ids):
        return (
            w1.shape[0],
            a1.shape[0],
            2 * self.intermediate_size,
            a1.shape[-1],
            topk_ids.shape[1],
        )

    def workspace_shapes(
        self,
        M,
        N,
        K,
        topk,
        global_num_experts,
        local_num_experts,
        expert_tokens_meta,
        activation,
    ):
        # workspace2 uses BF16 storage viewed as FP32 by the native operator.
        return (0,), (2 * self.workspace_elements,), (M, K)

    def finalize_weight_and_reduce_impl(self):
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceNoOP,
        )

        return TopKWeightAndReduceNoOP()

    def apply(
        self,
        output,
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        global_num_experts,
        expert_map,
        a1q_scale,
        a2_scale,
        workspace13,
        workspace2,
        expert_tokens_meta,
        apply_router_weight_on_input,
    ):
        from vllm.sm80.kernels.exl3.ops import moe_out, moe_out_v2

        rows = hidden_states.shape[0]
        if rows > self.max_rows:
            raise ValueError("EXL3 MoE exceeds configured dispatched row capacity")
        if (
            apply_router_weight_on_input
            or a1q_scale is not None
            or a2_scale is not None
        ):
            raise ValueError("EXL3 MoE requires unquantized, unweighted BF16 inputs")
        operator = moe_out_v2 if 0 < rows <= self.moe_v2_rows else moe_out
        if self.prefill_decode_len:
            operator = self._prefill_operator(rows, operator)
        operator(
            hidden_states,
            topk_ids,
            topk_weights,
            w1,
            w2,
            self.expert_map if expert_map is None else expert_map,
            self.intermediate_size,
            output,
            workspace2.view(torch.float32),
        )

    def _prefill_operator(self, rows: int, operator):
        """moe_out_prefill for a call of more than 64 rows in a prefill step
        (the phase the scheduler gave the step, never the row count);
        otherwise ``operator``."""
        from vllm.sm80.kernels.exl3.ops import MOE_PREFILL_MIN_ROWS, moe_out_prefill
        from vllm.sm80.kernels.exl3.step_phase import prefill_step

        if rows >= MOE_PREFILL_MIN_ROWS and prefill_step(self.prefill_decode_len):
            return moe_out_prefill
        return operator


def dispatched_rows(prepare, moe) -> int:
    """Most rows the experts of one rank receive in one MoE call."""
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
        MoEPrepareAndFinalizeNaiveDPEPModular,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )

    parallel = moe.moe_parallel_config
    rows = moe.max_num_tokens
    if isinstance(prepare, MoEPrepareAndFinalizeNoDPEPModular):
        # No dispatch (TP + EP without DP): each rank's experts see only the
        # rows of its own forward.
        return rows
    if not isinstance(prepare, MoEPrepareAndFinalizeNaiveDPEPModular):
        raise ValueError("EXL3 EP requires allgather_reducescatter dispatch")
    if parallel.is_sequence_parallel:
        # The EP group gathers every rank's padded sequence-parallel share.
        return parallel.ep_size * -(-rows // parallel.sp_size)
    # The DP (or PCP) group gathers every replica's rows.
    return rows * parallel.dp_size * parallel.pcp_size
