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
    ColumnParallelLinear,
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm_cmp170hx.kernels.exl3.utils import (
    Exl3Manifest,
    PackedMatrix,
    aligned_shard,
)

logger = init_logger(__name__)


# Lab (default off): VLLM_EXL3_ROUTE_HIST_DIR=<dir> counts routed assignments per (MoE layer, global expert) on
# eager calls (CUDA-graph captures are skipped) and writes <dir>/route-hist-<pid>.pt every 64 calls: {layer_name:
# int64[global_experts]}. For calibration-coverage studies (tokens per expert of a calibration set); no effect on
# outputs.
import os as _os

_ROUTE_HIST_DIR = _os.environ.get("VLLM_EXL3_ROUTE_HIST_DIR", "").strip()
_route_hist: dict = {}
_route_calls = [0]


def _record_routes(layer, topk_ids) -> None:
    if torch.cuda.is_current_stream_capturing():
        return
    n = int(getattr(layer, "global_num_experts", 0) or 0)
    if n <= 0:
        return
    name = str(getattr(layer, "layer_name", None) or getattr(layer, "prefix", None) or id(layer))
    ids = topk_ids.reshape(-1).to(torch.int64)
    ids = ids[(ids >= 0) & (ids < n)]
    h = _route_hist.get(name)
    if h is None:
        h = _route_hist[name] = torch.zeros(n, dtype=torch.int64, device=topk_ids.device)
    h += torch.bincount(ids, minlength=n)
    _route_calls[0] += 1
    if _route_calls[0] % 64 == 0:
        _os.makedirs(_ROUTE_HIST_DIR, exist_ok=True)
        tmp = _os.path.join(_ROUTE_HIST_DIR, f".route-hist-{_os.getpid()}.pt")
        torch.save({k: v.cpu() for k, v in _route_hist.items()}, tmp)
        _os.replace(tmp, _os.path.join(_ROUTE_HIST_DIR, f"route-hist-{_os.getpid()}.pt"))

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
    """Registered as "exl3" by vllm_cmp170hx/plugin.py.

    Upstream model code consults the attributes and methods below through
    getattr, so they apply only to this config.
    """

    min_transform_size = 128
    # Qwen3-Next shared experts: TP shards keep whole transform blocks.
    shared_expert_shard_group_size = min_transform_size
    # GLM-5.3 checkpoints quantize attention and vision projections that the
    # FP8 checkpoint deliberately leaves in BF16.
    quantize_glm5_attention = True
    split_glm5_kda_input = True

    def __init__(self):
        super().__init__()
        self.ple_placement = envs.VLLM_EXL3_PLE_PLACEMENT
        self.manifest: Exl3Manifest | None = None
        self.mtp_start_layer_idx = 0
        self.mapped_prefixes: dict[str, str] = {}
        self.is_qwen = False
        self.is_glm = False
        # DeepSeek-V4.1: checkpoint names follow DeepSeek's own modules
        # (models/deepseek_v41/sm80_exl3.py, DeepseekV41Exl3Layout).
        self.is_dsv41 = False
        self.layout = None

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
        model_type = getattr(hf_config, "model_type", "")
        is_qwen = model_type in (
            "qwen4_exp",
            "qwen4_exp_text",
            "qwen4_exp_mtp",
        )
        text = getattr(hf_config, "text_config", hf_config)
        if is_qwen:
            # The Qwen4Exp EXL3 path (its PLE tables) is not carried by this
            # plugin; the plugin serves GLM-5.3 EXL3 checkpoints.
            raise NotImplementedError(
                "Qwen4Exp EXL3 checkpoints are not supported by this plugin"
            )
        self.manifest = Exl3Manifest(model_name, require_qwen_aux=is_qwen)
        self.is_qwen = is_qwen
        self.is_glm = (
            model_type in ("glm5_next", "glm5_next_mtp")
            or getattr(text, "model_type", "") == "glm5_next_text"
        )
        self.mtp_start_layer_idx = getattr(text, "num_hidden_layers", 0)
        self.is_dsv41 = (
            model_type == "deepseek_v41"
            or getattr(text, "model_type", "") in ("deepseek_v41", "deepseek_v41_text")
            or "DSparkV41DraftModel" in (getattr(hf_config, "architectures", None) or ())
        )
        if self.is_dsv41:
            from vllm_cmp170hx.models.deepseek_v41.sm80_exl3 import (
                DeepseekV41Exl3Layout,
            )

            self.layout = DeepseekV41Exl3Layout(hf_config)

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

    def is_draft_source(self, name: str) -> bool:
        """Whether a checkpoint tensor belongs to the native draft model."""
        if self.is_qwen or self.is_dsv41:
            return name.startswith("mtp.")
        if self.is_glm:
            match = re.match(r"model\.language_model\.layers\.(\d+)\.", name)
            return match is not None and int(match[1]) >= self.mtp_start_layer_idx
        return False

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
        if self.layout is not None:
            return self.layout.source_prefix(self, prefix)
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
        members = None
        if self.layout is not None:
            members = self.layout.members(self, leaf, source, outputs)
        if members is None:
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
            groups = self.grouped_constituents(prefix)
            if groups and isinstance(layer, LinearBase):
                return Exl3GroupedLinearMethod(groups)
            if (
                self.is_dsv41
                and isinstance(layer, (ReplicatedLinear, ColumnParallelLinear))
                and _tp_world_size() == 1
                and _dense_w8_covers(self.source_prefix(prefix))
            ):
                return W8SidecarLinearMethod(self.source_prefix(prefix))
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
        return None

    def grouped_constituents(self, prefix: str) -> list[PackedMatrix]:
        """Per-group matrices of a block-diagonal projection stored as one
        EXL3 matrix per group (DeepSeek-V4.1 ``wo_a.slice.<g>``), in group
        order; empty for every other projection."""
        if self.layout is None:
            return []
        assert self.manifest is not None
        names = self.layout.grouped_members(self, self.source_prefix(prefix))
        missing = [n for n in names if n not in self.manifest.matrices]
        if missing:
            raise ValueError(f"Missing EXL3 group matrices: {missing[:4]}")
        return [self.manifest.matrices[n] for n in names]


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
        from vllm_cmp170hx.kernels.exl3 import dense_w8

        # Lab: an INT8 sidecar built from the original checkpoint replaces the
        # trellis for whole (unsharded) matrices it covers (VLLM_EXL3_DENSE_W8).
        self.w8_mode = (
            dense_w8.covers(matrix.prefix)
            and len(output_slices) == 1
            and input_slice.start == 0
            and input_slice.stop == matrix.padded_in
            and output_slices[0].start == 0
            and output_slices[0].stop == matrix.padded_out
            and logical_in == matrix.padded_in
        )
        if self.w8_mode:
            n = logical_out
            self.w8 = nn.Parameter(
                torch.empty(n, self.padded_in, dtype=torch.int8), requires_grad=False
            )
            self.sw = nn.Parameter(
                torch.empty(n, dtype=torch.float32), requires_grad=False
            )
            self.trellis = nn.Parameter(
                torch.empty(0, dtype=torch.int16), requires_grad=False
            )
            self.svh = nn.Parameter(
                torch.empty(0, dtype=torch.float16), requires_grad=False
            )
            self.suh = nn.Parameter(
                torch.empty(
                    0 if _dense_w8_head(matrix.prefix) else self.padded_in,
                    dtype=torch.float16,
                ),
                requires_grad=False,
            )
            self.loaded = False
            return
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
        if self.w8_mode:
            from vllm_cmp170hx.kernels.exl3 import dense_w8

            w8 = dense_w8.tensor(prefix, "w8")
            if tuple(w8.shape) != tuple(self.w8.shape):
                raise ValueError(
                    f"EXL3 dense W8 sidecar shape {tuple(w8.shape)} != "
                    f"{tuple(self.w8.shape)} for {prefix}"
                )
            self.w8.data.copy_(w8)
            self.sw.data.copy_(dense_w8.tensor(prefix, "sw"))
            if self.suh.numel():
                self.suh.data.copy_(dense_w8.tensor(prefix, "suh"))
            for key in self.matrix.keys:
                ledger.mark(key, "consumed", prefix)
            self.loaded = True
            return
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
        # Output dtype: BF16, or FP32 for a layer whose output the model keeps
        # in FP32 (set_output_dtype; an LM head takes ModelConfig.head_dtype).
        # The kernels here return BF16; an FP32 output is that result widened
        # (not the FP32 accumulator as vllm-sm80's EXL3 kernels store it).
        self.out_dtype = torch.bfloat16
        # Keys of the matrices other ranks hold (grouped projections).
        self.nonlocal_keys: list[str] = []

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
        if isinstance(layer, ParallelLMHead):
            from vllm.config import get_current_vllm_config_or_none

            vllm_config = get_current_vllm_config_or_none()
            model_config = getattr(vllm_config, "model_config", None)
            head_dtype = getattr(model_config, "head_dtype", None)
            if head_dtype is not None:
                self.out_dtype = _linear_out_dtype(head_dtype)
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
        from vllm_cmp170hx.kernels.exl3 import ops as exl3_ops

        self.row_chunks = []
        self._fuse_w8(layer)
        for weight in layer.exl3_weights:
            if not weight.loaded:
                raise ValueError(f"Unloaded EXL3 matrix: {weight.matrix.prefix}")
            if not all(p.is_cuda for p in weight.parameters()):
                raise ValueError("EXL3 packed weights and scales must remain on CUDA")
            if weight.w8_mode:
                from vllm_cmp170hx.kernels.exl3 import dense_w8

                dense_w8.load_libs()
                if _dense_w8_head(weight.matrix.prefix):
                    q, s = dense_w8.marlin_pack_lm_head(weight.w8.data, weight.sw.data)
                    weight.marlin_q = nn.Parameter(q, requires_grad=False)
                    weight.marlin_s = nn.Parameter(s, requires_grad=False)
                    weight.w8 = nn.Parameter(
                        torch.empty(0, dtype=torch.int8, device=q.device),
                        requires_grad=False,
                    )
                self.row_chunks.append(0)
                continue
            self.row_chunks.append(
                exl3_ops.linear_row_chunk(weight.padded_in, weight.padded_out)
            )

    @staticmethod
    def _fuse_w8(layer):
        """Lab: merge a layer's sidecar matrices that share one input rotation (same suh, same K) into
        one matrix, so a merged layer (gate/up, q_a/kv_a) runs one H128 + quant and one GEMM. The
        output is the concatenation apply() would build. Only sidecars built with --share-suh have
        equal sign vectors; with per-matrix EXL3 signs nothing is merged (VLLM_EXL3_DENSE_W8_FUSE=0
        disables it)."""
        import os

        weights = list(layer.exl3_weights)
        if (
            os.environ.get("VLLM_EXL3_DENSE_W8_FUSE", "1") != "1"
            or len(weights) < 2
            or not all(w.loaded and w.w8_mode and not _dense_w8_head(w.matrix.prefix) for w in weights)
            or len({w.padded_in for w in weights}) != 1
            or not all(torch.equal(w.suh.data, weights[0].suh.data) for w in weights[1:])
        ):
            return
        first = weights[0]
        first.w8 = nn.Parameter(torch.cat([w.w8.data for w in weights], 0), requires_grad=False)
        first.sw = nn.Parameter(torch.cat([w.sw.data for w in weights], 0), requires_grad=False)
        first.logical_out = sum(w.logical_out for w in weights)
        first.padded_out = first.logical_out
        layer.exl3_weights = nn.ModuleList([first])
        logger.info_once(
            "EXL3 dense sidecar: fusing shared-input matrices (e.g. %s)",
            " + ".join(w.matrix.prefix for w in weights),
            scope="process",
        )

    def apply(self, layer, x, bias=None):
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1]).contiguous()
        outputs = [
            (
                torch.ops.vllm.exl3_w8_lm_head(
                    x, weight.marlin_q, weight.marlin_s, weight.logical_out,
                    weight.padded_in,
                )
                if weight.w8_mode and _dense_w8_head(weight.matrix.prefix)
                else torch.ops.vllm.exl3_w8_linear(x, weight.w8, weight.sw, weight.suh)
                if weight.w8_mode
                else torch.ops.vllm.exl3_linear(
                    x,
                    weight.trellis,
                    weight.suh,
                    weight.svh,
                    weight.matrix.bits_x2,
                    weight.matrix.codebook,
                    weight.logical_out,
                    row_chunk,
                )
            )
            for weight, row_chunk in zip(layer.exl3_weights, self.row_chunks)
        ]
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        if bias is not None:
            output.add_(bias)
        return output.reshape(*shape, output.shape[-1])

    @staticmethod
    def _run(weight, row_chunk: int, x: torch.Tensor) -> torch.Tensor:
        """One packed matrix: the sidecar's W8A16 head or W8A8 GEMM, else the
        EXL3 trellis kernel (the per-matrix dispatch of :meth:`apply`)."""
        if weight.w8_mode and _dense_w8_head(weight.matrix.prefix):
            return torch.ops.vllm.exl3_w8_lm_head(
                x, weight.marlin_q, weight.marlin_s, weight.logical_out,
                weight.padded_in,
            )
        if weight.w8_mode:
            return torch.ops.vllm.exl3_w8_linear(x, weight.w8, weight.sw, weight.suh)
        return torch.ops.vllm.exl3_linear(
            x,
            weight.trellis,
            weight.suh,
            weight.svh,
            weight.matrix.bits_x2,
            weight.matrix.codebook,
            weight.logical_out,
            row_chunk,
        )


def _tp_world_size() -> int:
    from vllm.distributed import get_tensor_model_parallel_world_size

    return get_tensor_model_parallel_world_size()


def _dense_w8_covers(prefix: str) -> bool:
    from vllm_cmp170hx.kernels.exl3 import dense_w8

    return dense_w8.covers(prefix)


class W8SidecarLinearMethod(LinearMethodBase):
    """An unsharded linear the EXL3 checkpoint keeps unquantized (DeepSeek-V4.1
    ``layers.{1,14}.engram.wkv``: the converter builds the Engram projection
    without a quantization map) but the dense W8 sidecar covers: it runs
    W8A8 from the sidecar like the EXL3-covered matrices
    (``torch.ops.vllm.exl3_w8_linear``). The checkpoint's BF16 ``.weight``
    is left unused (EXL3 loader), so the layer holds only the INT8 copy."""

    def __init__(self, source: str):
        self.source = source

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
        from vllm_cmp170hx.kernels.exl3 import dense_w8

        n, k = sum(output_partition_sizes), input_size_per_partition
        shape = tuple(dense_w8._open().get_slice(f"{self.source}.w8").get_shape())
        if shape != (n, k):
            raise ValueError(
                f"EXL3 dense W8 sidecar shape {shape} != {(n, k)} for {self.source}"
            )
        layer.w8 = nn.Parameter(torch.empty(n, k, dtype=torch.int8), requires_grad=False)
        layer.sw = nn.Parameter(torch.empty(n, dtype=torch.float32), requires_grad=False)
        layer.suh = nn.Parameter(torch.empty(k, dtype=torch.float16), requires_grad=False)
        layer.w8_sidecar_source = self.source

    def load_sidecar(self, layer) -> None:
        from vllm_cmp170hx.kernels.exl3 import dense_w8

        layer.w8.data.copy_(dense_w8.tensor(self.source, "w8"))
        layer.sw.data.copy_(dense_w8.tensor(self.source, "sw"))
        layer.suh.data.copy_(dense_w8.tensor(self.source, "suh"))

    def process_weights_after_loading(self, layer):
        from vllm_cmp170hx.kernels.exl3 import dense_w8

        dense_w8.load_libs()

    def apply(self, layer, x, bias=None):
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1]).contiguous()
        out = torch.ops.vllm.exl3_w8_linear(x, layer.w8, layer.sw, layer.suh)
        if bias is not None:
            out = out + bias
        return out.reshape(*shape, out.shape[-1])


def _dense_w8_head(prefix: str) -> bool:
    from vllm_cmp170hx.kernels.exl3 import dense_w8

    return dense_w8.is_head(prefix)


def _linear_out_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"EXL3 linear outputs are BF16 or FP32, not {dtype}")
    return dtype


class OutDtypeLinearMethod(UnquantizedLinearMethod):
    """An unquantized linear that returns its FP32-accumulated product in
    ``out_dtype``: FP32 keeps the sum of the exact BF16 products unrounded,
    as an FP32 linear of the same BF16 input and weight computes it."""

    def __init__(self, out_dtype: torch.dtype):
        super().__init__()
        self.out_dtype = out_dtype

    def apply(self, layer, x, bias=None):
        shape = x.shape
        output = torch.mm(
            x.reshape(-1, shape[-1]), layer.weight.t(), out_dtype=self.out_dtype
        )
        if bias is not None:
            output = output + bias.to(self.out_dtype)
        return output.reshape(*shape[:-1], output.shape[-1])


def set_output_dtype(layer: nn.Module, dtype: torch.dtype) -> None:
    """Have the EXL3 or unquantized linear ``layer`` return ``dtype`` (BF16
    or FP32); an unquantized layer takes OutDtypeLinearMethod."""
    method = getattr(layer, "quant_method", None)
    if isinstance(method, Exl3LinearMethod):
        method.out_dtype = _linear_out_dtype(dtype)
    elif type(method) in (UnquantizedLinearMethod, OutDtypeLinearMethod):
        layer.quant_method = OutDtypeLinearMethod(_linear_out_dtype(dtype))
    else:
        raise TypeError(
            f"{type(layer).__name__} has no EXL3 or unquantized linear method "
            f"({type(method).__name__}) to set an output dtype on"
        )


def set_moe_prefill_v2(layer: nn.Module) -> str | None:
    """vllm-sm80's moe_out_prefill_v2 is not carried: this plugin's routed
    prefill is the Q4N-X INT8 operator (VLLM_EXL3_Q4N_PREFILL). Returns why."""
    return (
        "moe_out_prefill_v2 is not in this plugin; routed prefill takes the "
        "Q4N-X operators (VLLM_EXL3_Q4N_PREFILL)"
    )


class Exl3HeadLogitsProcessor(LogitsProcessor):
    """LogitsProcessor that also serves a head_dtype different from the
    model dtype with an EXL3 LM head: the head stores its logits in the head
    dtype (Exl3LinearMethod.create_weights), so it runs as it is. Every
    other head takes the upstream path."""

    @classmethod
    def adopt(cls, processor: LogitsProcessor) -> None:
        assert type(processor) is LogitsProcessor, type(processor)
        processor.__class__ = cls

    def _apply_head(self, lm_head, hidden_states, embedding_bias):
        method = lm_head.quant_method
        if not isinstance(method, Exl3LinearMethod):
            return super()._apply_head(lm_head, hidden_states, embedding_bias)
        head_dtype = self.head_dtype or hidden_states.dtype
        if method.out_dtype != head_dtype:
            raise ValueError(
                f"The EXL3 LM head stores {method.out_dtype}, the head dtype is "
                f"{head_dtype}"
            )
        return method.apply(lm_head, hidden_states, bias=embedding_bias)


class Exl3GroupedLinearMethod(Exl3LinearMethod):
    """A block-diagonal projection stored as one EXL3 matrix per group:
    output group g is input group g times W_g^T. TP ranks own whole groups in
    order. The layer has no dense weight; callers project through
    :meth:`apply_grouped` (DeepSeek-V4.1 attention ``wo_a``)."""

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
        groups, size = len(self.matrices), layer.tp_size
        if groups % size or output_size % groups:
            raise ValueError(f"EXL3 cannot split {groups} groups over {size} ranks")
        local, group_out = groups // size, output_size // groups
        mine = range(layer.tp_rank * local, (layer.tp_rank + 1) * local)
        self.nonlocal_keys = [
            key
            for group, matrix in enumerate(self.matrices)
            if group not in mine
            for key in matrix.keys
        ]
        if sum(output_partition_sizes) != local * group_out:
            raise ValueError("EXL3 group widths do not match the layer partition")
        weights = []
        for matrix in self.matrices[layer.tp_rank * local : (layer.tp_rank + 1) * local]:
            matrix.validate_logical(input_size, group_out)
            weights.append(
                Exl3PackedWeight(
                    matrix,
                    input_size,
                    group_out,
                    slice(0, matrix.padded_in),
                    [slice(0, matrix.padded_out)],
                )
            )
        layer.exl3_weights = nn.ModuleList(weights)

    @staticmethod
    def _fuse_w8(layer):
        # Groups have distinct inputs: never merge them.
        return

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        from vllm_cmp170hx.kernels.exl3 import dense_w8

        weights = list(layer.exl3_weights)
        # One activation quantisation for every group (dense_w8._w8_grouped)
        # when all groups are sidecar matrices with the same input signs.
        self.w8_grouped = (
            dense_w8.GROUPED
            and len(weights) > 1
            and all(w.w8_mode for w in weights)
            and all(torch.equal(w.suh.data, weights[0].suh.data) for w in weights[1:])
        )

    def apply(self, layer, x, bias=None):
        raise ValueError("Grouped EXL3 projections run through apply_grouped")

    def apply_grouped(self, layer, x: torch.Tensor) -> torch.Tensor:
        """``x`` [tokens, local groups, input] -> [tokens, local groups *
        output], group by group in this rank's group order."""
        if x.dim() != 3 or x.shape[1] != len(layer.exl3_weights):
            raise ValueError("Grouped EXL3 input must be [tokens, groups, input]")
        if getattr(self, "w8_grouped", False):
            weights = layer.exl3_weights
            output = torch.ops.vllm.exl3_w8_grouped(
                x.contiguous(),
                [w.w8 for w in weights],
                [w.sw for w in weights],
                weights[0].suh,
            )
            if output.dtype != self.out_dtype:
                output = output.to(self.out_dtype)
            return output
        outputs = [
            self._run(weight, row_chunk, x[:, group].contiguous())
            for group, (weight, row_chunk) in enumerate(
                zip(layer.exl3_weights, self.row_chunks)
            )
        ]
        output = torch.cat(outputs, dim=-1)
        if output.dtype != self.out_dtype:
            output = output.to(self.out_dtype)
        return output


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
        if moe.swiglu_alpha is not None or moe.swiglu_beta is not None:
            raise ValueError("EXL3 experts do not support custom SwiGLU alpha or beta")

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
                if self.config.layout is not None:
                    prefix = self.config.layout.expert_matrix(
                        self.config, self.prefix, expert, role
                    )
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
        from vllm_cmp170hx.kernels.exl3 import ops as exl3_ops

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
        prefill_max_rows = None
        prefill_bits_x2 = 0
        if envs.VLLM_EXL3_MOE_PREFILL_TC:
            prefill_decode_len, workspace, prefill_bits_x2 = self._moe_prefill(
                layer, metadata, max_rows, workspace
            )
            family = exl3_ops.moe_tc_family(self.hidden_size, self.intermediate_size)
            if family is not None and prefill_decode_len:
                prefill_max_rows = family.prefill_max_rows
        decode_tc_rows = 0
        decode_tc_bits_x2 = 0
        decode_query_len = 0
        if envs.VLLM_EXL3_MOE_DECODE_TC:
            (
                decode_tc_rows,
                decode_tc_bits_x2,
                decode_query_len,
                workspace,
            ) = self._moe_decode_tc(layer, metadata, max_rows, workspace)
        # Q4N / Q4N-X experts (codebook 3/4) have no base-kernel path: the base MoE
        # GEMM (csrc/exl3/gemm.cu) traps on any format outside mul1/codebook 1-2.
        # Such a layer must run every call on the tensor-core kernels, so both
        # gates must be open and the layer format uniform.
        q4n = any(int(proj[3]) >= 3 for expert in metadata for proj in expert)
        if q4n:
            key = _layer_bits_x2(metadata)
            missing = [
                name
                for name, ok in (
                    ("uniform Q4N expert format", key is not None),
                    ("VLLM_EXL3_MOE_PREFILL_TC gate", bool(prefill_decode_len)),
                    ("VLLM_EXL3_MOE_DECODE_TC gate", bool(decode_tc_rows)),
                )
                if not ok
            ]
            if missing:
                raise ValueError(
                    "EXL3 Q4N routed experts run only on the tensor-core kernels "
                    "(the base MoE GEMM has no codebook 3/4 path); missing: "
                    + ", ".join(missing)
                )
        experts = Exl3Experts(
            self.moe,
            self.moe_quant_config,
            self.intermediate_size,
            layer.exl3_expert_map,
            workspace,
            max_rows,
            0.0 if self.moe.swiglu_limit is None else float(self.moe.swiglu_limit),
            moe_v2_rows,
            prefill_decode_len,
            prefill_requires_explicit=(
                bool(prefill_decode_len)
                and exl3_ops.moe_tc_family(self.hidden_size, self.intermediate_size)
                is not None
            ),
            prefill_bits_x2=prefill_bits_x2,
            prefill_max_rows=prefill_max_rows,
            decode_tc_rows=decode_tc_rows,
            decode_tc_bits_x2=decode_tc_bits_x2,
            decode_query_len=decode_query_len,
            q4n_tc_only=q4n,
        )
        self.moe_kernel = mk.FusedMoEKernel(prepare, experts)

    def _moe_v2(self, layer, metadata, max_rows: int, workspace: int):
        """Rows up to which this layer's calls take moe_out_v2, and a workspace
        size that fits both operators (0 rows: the gate is closed)."""
        from vllm_cmp170hx.kernels.exl3 import ops as exl3_ops

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
        """The decode query bound, workspace and family bits for a covered
        prefill family.

        The MOE_TC_FAMILIES shapes (GLM, DeepSeek-V4.1) additionally require
        explicit pure-prefill metadata at dispatch, a uniform admitted expert
        format, 74 SMs, the family's SwiGLU limit and its model layout. A
        zero bound means the gate is closed.
        """
        from vllm.config import get_current_vllm_config
        from vllm_cmp170hx.kernels.exl3 import ops as exl3_ops
        from vllm_cmp170hx.kernels.exl3.step_phase import max_decode_query_len

        device = layer.exl3_ptrs.device
        family = exl3_ops.moe_tc_family(self.hidden_size, self.intermediate_size)
        bits_x2 = _layer_bits_x2(metadata)
        reason = exl3_ops.moe_prefill_closed_reason(
            tuple(torch.cuda.get_device_capability(device)),
            self.moe.in_dtype,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
            self.local_experts,
            layer.global_num_experts,
            metadata,
            bits_x2 if bits_x2 is not None else 0,
        )
        if reason is None and family is not None:
            reason = _family_device_closed_reason(family, "prefill", device, self.moe)
        if reason is None and max_rows < exl3_ops.MOE_PREFILL_MIN_ROWS:
            reason = f"MoE calls have at most {max_rows} rows"
        if reason is None:
            # Host copy of the layer's pointer table: one device sync, at load.
            reason = _unaligned_trellis_reason(layer.exl3_ptrs.tolist())
        vllm_config = get_current_vllm_config()
        if reason is None:
            reason = _prefill_layout_closed_reason(vllm_config, family=family)
        if reason is not None:
            logger.info_once(
                "VLLM_EXL3_MOE_PREFILL_TC is set but the gate is closed: %s; "
                "these MoE layers keep moe_out in prefill steps.",
                reason,
            )
            return 0, workspace, 0
        rows = min(
            max_rows,
            family.prefill_max_rows if family is not None else exl3_ops.MOE_PREFILL_MAX_ROWS,
        )
        workspace = max(
            workspace,
            exl3_ops.moe_workspace_size_prefill(
                rows,
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
            bits_x2,
        )
        decode_len = max_decode_query_len(vllm_config)
        logger.info_once(
            "EXL3 MoE prefill kernel active (VLLM_EXL3_MOE_PREFILL_TC): "
            "tensor-core routed experts for MoE calls of %d..%d rows; "
            "explicit-prefill metadata=%s. Decode, capture and uncovered "
            "calls keep their kernels.",
            exl3_ops.MOE_PREFILL_MIN_ROWS,
            rows,
            family is not None,
        )
        compilation = getattr(vllm_config, "compilation_config", None)
        graph_max = getattr(compilation, "max_cudagraph_capture_size", None) or 0
        if graph_max >= exl3_ops.MOE_PREFILL_MIN_ROWS:
            logger.info_once(
                "VLLM_EXL3_MOE_PREFILL_TC: CUDA graphs are captured up to %d "
                "tokens; prefill steps padded to a captured size replay moe_out.",
                graph_max,
            )
        return decode_len, workspace, bits_x2

    def _moe_decode_tc(self, layer, metadata, max_rows: int, workspace: int):
        """The row bound, expert format, decode phase bound and workspace for
        a small-row tensor-core decode family (GLM, DeepSeek-V4.1; 0 rows: the
        gate closed).

        The same family checks as the prefill gate: the family's model layout
        (GLM: the exact PP2 DFlash-k3/k5/k7 layout), 74 SMs, the family's
        SwiGLU limit, a uniform admitted expert format and 16-byte-aligned
        trellis bases. The row range is 1..MOE_DECODE_TC_MAX_ROWS rows in
        decode steps.
        """
        from vllm.config import get_current_vllm_config
        from vllm_cmp170hx.kernels.exl3 import ops as exl3_ops
        from vllm_cmp170hx.kernels.exl3.step_phase import max_decode_query_len

        device = layer.exl3_ptrs.device
        family = exl3_ops.moe_tc_family(self.hidden_size, self.intermediate_size)
        bits_x2 = _layer_bits_x2(metadata)
        reason = exl3_ops.moe_decode_tc_closed_reason(
            tuple(torch.cuda.get_device_capability(device)),
            self.moe.in_dtype,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
            self.local_experts,
            layer.global_num_experts,
            metadata,
            bits_x2 if bits_x2 is not None else 0,
        )
        if reason is None:
            reason = _family_device_closed_reason(family, "decode", device, self.moe)
        if reason is None and max_rows < 1:
            reason = f"MoE calls have at most {max_rows} rows"
        if reason is None:
            # Host copy of the layer's pointer table: one device sync, at load.
            reason = _unaligned_trellis_reason(layer.exl3_ptrs.tolist())
        vllm_config = get_current_vllm_config()
        if reason is None:
            reason = _prefill_layout_closed_reason(vllm_config, family=family)
        if reason is not None:
            logger.info_once(
                "VLLM_EXL3_MOE_DECODE_TC is set but the gate is closed: %s; "
                "these MoE layers keep moe_out in decode steps.",
                reason,
            )
            return 0, 0, 0, workspace
        rows = min(max_rows, exl3_ops.MOE_DECODE_TC_MAX_ROWS)
        workspace = max(
            workspace,
            exl3_ops.moe_workspace_size_decode_tc(
                rows,
                self.hidden_size,
                self.intermediate_size,
                layer.top_k,
                self.local_experts,
            ),
        )
        exl3_ops.warmup_moe_decode_tc(
            layer.exl3_ptrs,
            layer.exl3_meta,
            layer.exl3_expert_map,
            self.global_expert_ids,
            self.hidden_size,
            self.intermediate_size,
            layer.top_k,
            bits_x2,
        )
        logger.info_once(
            "EXL3 MoE decode kernel active (VLLM_EXL3_MOE_DECODE_TC): "
            "small-row routed tensor-core experts for decode-step MoE calls "
            "of 1..%d rows; prefill steps, larger calls and uncovered "
            "configurations keep their kernels.",
            rows,
        )
        compilation = getattr(vllm_config, "compilation_config", None)
        graph_max = getattr(compilation, "max_cudagraph_capture_size", None) or 0
        if graph_max >= 1:
            logger.info_once(
                "VLLM_EXL3_MOE_DECODE_TC: CUDA graphs are captured up to %d "
                "tokens; a captured decode graph replays the kernels it was "
                "captured with, and a prefill step padded to a captured size "
                "of at most %d rows replays this schedule (same results).",
                graph_max,
                rows,
            )
        return rows, bits_x2, max_decode_query_len(vllm_config), workspace

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        assert self.moe_kernel is not None
        if _ROUTE_HIST_DIR:
            _record_routes(layer, topk_ids)
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
        swiglu_limit,
        moe_v2_rows=0,
        prefill_decode_len=0,
        prefill_requires_explicit=False,
        prefill_bits_x2=0,
        decode_tc_rows=0,
        decode_tc_bits_x2=0,
        decode_query_len=0,
        q4n_tc_only=False,
        prefill_max_rows=None,
    ):
        super().__init__(moe, quant)
        self.intermediate_size = intermediate_size
        self.expert_map = expert_map
        self.workspace_elements = workspace
        self.max_rows = max_rows
        self.swiglu_limit = swiglu_limit
        # Calls of 1..moe_v2_rows rows take moe_out_v2 (VLLM_EXL3_MOE_V2).
        self.moe_v2_rows = moe_v2_rows
        # A nonzero bound enables the prefill phase check; parallel drafting
        # additionally requires explicit homogeneous prefill token counts.
        self.prefill_decode_len = prefill_decode_len
        self.prefill_requires_explicit = prefill_requires_explicit
        # The layer's uniform expert format for the prefill family (0: closed).
        self.prefill_bits_x2 = prefill_bits_x2
        # Decode-step calls of 1..decode_tc_rows rows take moe_out_decode_tc
        # (VLLM_EXL3_MOE_DECODE_TC); 0: the gate is closed.
        self.decode_tc_rows = decode_tc_rows
        self.decode_tc_bits_x2 = decode_tc_bits_x2
        # 1 + num_speculative_tokens: the bound of the decode step check.
        self.decode_query_len = decode_query_len
        # Q4N layers: every call on the tensor-core kernels, whatever the step
        # phase or capture state (the base kernels trap on codebook 3/4).
        self.q4n_tc_only = q4n_tc_only
        from vllm_cmp170hx.kernels.exl3.ops import (
            MOE_PREFILL_GLM_MAX_ROWS,
            MOE_PREFILL_MAX_ROWS,
        )

        # The family's prefill chunk bound (MOE_TC_FAMILIES), else the former
        # rule (GLM's bound for the explicit-metadata family, Qwen's otherwise).
        self.prefill_max_rows = (
            prefill_max_rows
            if prefill_max_rows is not None
            else MOE_PREFILL_GLM_MAX_ROWS
            if prefill_requires_explicit
            else MOE_PREFILL_MAX_ROWS
        )

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
        from vllm_cmp170hx.kernels.exl3.ops import (
            moe_out,
            moe_out_decode_tc,
            moe_out_prefill,
            moe_out_v2,
        )

        rows = hidden_states.shape[0]
        if rows > self.max_rows:
            raise ValueError("EXL3 MoE exceeds configured dispatched row capacity")
        if (
            apply_router_weight_on_input
            or a1q_scale is not None
            or a2_scale is not None
        ):
            raise ValueError("EXL3 MoE requires unquantized, unweighted BF16 inputs")
        if self.q4n_tc_only:
            self._apply_q4n(
                rows, hidden_states, topk_ids, topk_weights, w1, w2,
                self.expert_map if expert_map is None else expert_map,
                output, workspace2.view(torch.float32),
            )
            return
        operator = moe_out_v2 if 0 < rows <= self.moe_v2_rows else moe_out
        if self.prefill_decode_len:
            operator = self._prefill_operator(rows, operator)
        if self.decode_tc_rows and operator is not moe_out_prefill:
            operator = self._decode_tc_operator(rows, operator)
        if operator is moe_out:
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
                self.swiglu_limit,
            )
        elif operator is moe_out_prefill:
            operator(
                hidden_states,
                topk_ids,
                topk_weights,
                w1,
                w2,
                self.expert_map if expert_map is None else expert_map,
                self.intermediate_size,
                self.prefill_bits_x2,
                output,
                workspace2.view(torch.float32),
            )
        elif operator is moe_out_decode_tc:
            operator(
                hidden_states,
                topk_ids,
                topk_weights,
                w1,
                w2,
                self.expert_map if expert_map is None else expert_map,
                self.intermediate_size,
                self.decode_tc_bits_x2,
                output,
                workspace2.view(torch.float32),
            )
        else:
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

    def _apply_q4n(self, rows, x, ids, weights, ptrs, meta, expert_map, out, ws):
        """Q4N routing: 1..decode_tc_rows rows on moe_out_decode_tc; up to
        MOE_PREFILL_MIN_ROWS - 1 rows on moe_out_decode_tc in decode_tc_rows
        slices (each token's routed output depends only on its own routes, and
        the slice count is fixed by the row count, so capture is safe); larger
        calls on moe_out_prefill, in chunks of at most its workspace bound."""
        from vllm_cmp170hx.kernels.exl3.ops import (
            MOE_PREFILL_MIN_ROWS,
            moe_out_decode_tc,
            moe_out_prefill,
        )

        if rows == 0:
            return
        if rows >= MOE_PREFILL_MIN_ROWS:
            # Calls above the workspace bound run as consecutive chunks of at
            # most prefill_max_rows rows (a token's routed output depends only
            # on its own routes, so chunking leaves every row's result as it
            # is); the last chunk is widened backwards to the kernel's row
            # floor when it would fall below it.
            step = self.prefill_max_rows
            lo = 0
            while lo < rows:
                hi = min(rows, lo + step)
                if hi - lo < MOE_PREFILL_MIN_ROWS:
                    lo = max(0, hi - MOE_PREFILL_MIN_ROWS)
                moe_out_prefill(
                    x[lo:hi], ids[lo:hi], weights[lo:hi], ptrs, meta, expert_map,
                    self.intermediate_size, self.prefill_bits_x2, out[lo:hi], ws,
                )
                lo = hi
            return
        step = self.decode_tc_rows
        for lo in range(0, rows, step):
            hi = min(rows, lo + step)
            moe_out_decode_tc(
                x[lo:hi], ids[lo:hi], weights[lo:hi], ptrs, meta, expert_map,
                self.intermediate_size, self.decode_tc_bits_x2, out[lo:hi], ws,
            )

    def _prefill_operator(self, rows: int, operator):
        """Use tensor cores only within the selected family's prefill range."""
        from vllm_cmp170hx.kernels.exl3.ops import (
            MOE_PREFILL_MIN_ROWS,
            moe_out_prefill,
        )
        from vllm_cmp170hx.kernels.exl3.step_phase import prefill_step

        if self.prefill_requires_explicit and torch.cuda.is_current_stream_capturing():
            return operator
        if MOE_PREFILL_MIN_ROWS <= rows <= self.prefill_max_rows and prefill_step(
            self.prefill_decode_len,
            require_explicit=self.prefill_requires_explicit,
        ):
            logger.info_once("EXL3 routed tensor-core prefill dispatch active")
            return moe_out_prefill
        return operator

    def _decode_tc_operator(self, rows: int, operator):
        """Use the small-row tensor-core schedule only for decode steps of
        the covered family's row range (1..decode_tc_rows rows).

        The step is a decode step when the prefill rule rejects it; with the
        explicit hybrid token counts of the GLM layout only a pure prefill
        step is a prefill step, so a speculative verification step (one
        sampled token plus the draft per request) is a decode step here. A
        captured CUDA graph replays the kernels it was captured with.
        """
        from vllm_cmp170hx.kernels.exl3.ops import moe_out_decode_tc
        from vllm_cmp170hx.kernels.exl3.step_phase import prefill_step

        if not 1 <= rows <= self.decode_tc_rows:
            return operator
        if prefill_step(self.decode_query_len, require_explicit=True):
            return operator
        logger.info_once("EXL3 routed tensor-core decode dispatch active")
        return moe_out_decode_tc


def _family_device_closed_reason(family, phase: str, device, moe) -> str | None:
    """A MOE_TC_FAMILIES family's device and model checks: the 74-SM CMP 170HX
    the schedules were sized on, and the SwiGLU limit the family hard-codes."""
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if sms != 74:
        return f"{family.name} {phase} device has {sms} SMs, not 74"
    if moe.swiglu_limit != family.swiglu_limit:
        return (
            f"{family.name} tensor-core {phase} requires SwiGLU clipping at "
            f"{family.swiglu_limit}"
        )
    return None


# DeepSeek-V4.1 text configs (target and DSpark drafter) the family serves.
_DSV41_MODEL_TYPES = ("deepseek_v41_text", "deepseek_v41")


def _dsv41_layout_closed_reason(vllm_config) -> str | None:
    """DeepSeek-V4.1: a BF16 model of this hidden size. Its routed experts are
    Q4N-X only, which run every call on the tensor-core kernels by row count
    (Exl3Experts._apply_q4n), so no parallel or drafting layout is required."""
    model = getattr(vllm_config, "model_config", None)
    text = getattr(model, "hf_text_config", None)
    if (
        getattr(text, "model_type", None) not in _DSV41_MODEL_TYPES
        or getattr(text, "hidden_size", None) != 5120
        or getattr(model, "dtype", None) != torch.bfloat16
    ):
        return "DeepSeek-V4.1 tensor-core MoE requires a BF16 DeepSeek-V4.1 model"
    return None


def _prefill_layout_closed_reason(
    vllm_config, *, glm: bool = False, family=None
) -> str | None:
    """Keep Qwen's existing layout, bound GLM PP2 drafts to 4/6/8 rows, and
    keep DeepSeek-V4.1's family to its model (``family``: the layer's
    MOE_TC_FAMILIES entry, None for Qwen)."""
    if family is not None:
        if family.name == "DeepSeek-V4.1":
            return _dsv41_layout_closed_reason(vllm_config)
        glm = True
    parallel = vllm_config.parallel_config
    layout = (parallel.tensor_parallel_size, parallel.pipeline_parallel_size)
    spec = vllm_config.speculative_config
    if glm:
        model = getattr(vllm_config, "model_config", None)
        text = getattr(model, "hf_text_config", None)
        if (
            layout != (1, 2)
            or getattr(text, "model_type", None) != "glm5_next_text"
            or getattr(text, "hidden_size", None) != 4096
            or getattr(model, "dtype", None) != torch.bfloat16
        ):
            return "GLM prefill requires BF16 GLM5Next at TP1/PP2"
        if (
            spec is None
            or spec.method != "dflash"
            or vllm_config.num_speculative_tokens not in (3, 5, 7)
        ):
            return "GLM prefill requires DFlash-k3, k5 or k7"
        return None
    if layout != (2, 1):
        return f"(tensor, pipeline) parallel size {layout}, not (2, 1)"
    if spec is not None and (
        getattr(spec, "parallel_drafting", False)
        or getattr(spec, "method", None) in ("dflash", "dspark")
    ):
        return f"speculative method {spec.method} drafts with parallel slots"
    return None


def _layer_bits_x2(metadata) -> int | None:
    """The layer's expert format when every expert and projection shares one
    bits_x2, else None (per-expert mixed formats keep the base kernels).

    ``metadata`` is either the host list or the packed tensor: per expert one
    ``[hidden, intermediate, bits_x2, codebook]`` row per projection, so the
    format is entry 2 of each row.
    """
    formats = {(int(projection[2]), int(projection[3])) for expert in metadata for projection in expert}
    if len(formats) != 1:
        return None
    bits, codebook = formats.pop()
    # Q4N layers (codebook 3) are dispatched by key bits_x2 + 64 (ops.split_moe_key).
    return bits + 128 if codebook == 4 else bits + 64 if codebook == 3 else bits


def _unaligned_trellis_reason(pointers) -> str | None:
    """Why moe_out_prefill must not read these experts, or None: it loads
    every expert trellis in 16-byte units and stops the device on a trellis
    base that is not 16-byte aligned. ``pointers`` is the layer's pointer
    table as a host list: per local expert and projection (gate, up, down)
    the [trellis, suh, svh] addresses."""
    roles = ("gate_proj", "up_proj", "down_proj")
    for local, expert in enumerate(pointers):
        for role, (trellis, _suh, _svh) in zip(roles, expert):
            if trellis % 16:
                return f"local expert {local} {role} trellis is not 16-byte aligned"
    return None


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
