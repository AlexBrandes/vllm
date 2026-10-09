# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA layer of the sm_80 fork and its attention metadata.

``vllm.sm80.plugin.register_glm_layers`` registers
:class:`Glm5NextLinearAttention` as the out-of-tree replacement of the
upstream layer of the same name (``vllm/models/glm5next/common/kda.py``), so
every KDA layer the GLM decoder builds is this subclass. It adds:

- the decode attributes the fused sm_80 KDA decode step and the KDA chunked
  prefill read in the layer's forward (``VLLM_AMPERE_DECODE_KERNELS``,
  ``VLLM_GLM5_DECODE_KDA_WIDE``, ``VLLM_AMPERE_KDA_PREFILL_WIDE`` /
  ``_NARROW``), and the deferred ``g_b_proj`` fill;
- RecoverSSM (``vllm/sm80/kernels/kda/recover.py``): the per-layer record
  buffer, no draft-position state pages in the KV spec, and the builder's
  commit metadata;
- its own GDN attention backend, whose builder adds the step's bound on the
  tokens of a speculative request (``spec_max_query_len``, the decode
  kernel's token tile) and the RecoverSSM wrap to the GDN metadata. The
  metadata class itself is defined in ``vllm/sm80/kernels/kda/recover.py``,
  which must import without the GLM model package.
"""

from dataclasses import replace

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.models.glm5next.common import kda as glm_kda
from vllm.sm80.kernels.kda.recover import Glm5NextGDNAttentionMetadata
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec

logger = init_logger(__name__)


class Glm5NextGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):
    def __init__(
        self,
        kv_cache_spec: MambaSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # The GDN builder enables metadata reuse across groups only for its
        # exact class (subclasses may keep per-group metadata). This builder
        # adds only batch-level fields, so it takes the same choice as the GDN
        # builder; RecoverSSM below turns reuse off again.
        self.supports_update_block_table = (
            vllm_config.use_v2_model_runner and device.type == "cuda"
        )
        if self.supports_update_block_table:
            # Opts into MRV2's CUDA-only aligned-index precompute.
            self.mamba_aligned_state_indices: torch.Tensor | None = None
        # RecoverSSM (GLM-5 KDA layers, vllm/sm80/kernels/kda/recover.py):
        # commit metadata on every spec-decode step; None otherwise.
        self._kda_recover = None
        if self.use_spec_decode:
            from vllm.sm80.kernels.kda.recover import (
                BuilderRecover,
                kda_recover_enabled,
            )

            if kda_recover_enabled(vllm_config):
                self._kda_recover = BuilderRecover(self)
                # Recovery commit indices and contexts belong to this group,
                # unlike the batch-level metadata shared by ordinary GDN.
                self.supports_update_block_table = False

    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )
        m = common_attn_metadata
        # The GDN builder returns a new GDNAttentionMetadata per build. Give it
        # the GLM class in place (a dataclass subclass adding one defaulted
        # field, no slots), so no copy is made per build.
        assert type(metadata) is GDNAttentionMetadata, type(metadata)
        metadata.__class__ = Glm5NextGDNAttentionMetadata
        metadata.spec_max_query_len = (
            min(int(m.max_query_len), self.num_spec + 1)
            if metadata.num_spec_decodes > 0 and getattr(m, "max_query_len", None)
            else None
        )
        if self._kda_recover is not None:
            return self._kda_recover.wrap(metadata, m, metadata.spec_sequence_masks_cpu)
        return metadata


class Glm5NextGDNAttentionBackend(GDNAttentionBackend):
    @staticmethod
    def get_builder_cls() -> type[Glm5NextGDNAttentionMetadataBuilder]:
        return Glm5NextGDNAttentionMetadataBuilder


class Glm5NextLinearAttention(glm_kda.Glm5NextLinearAttention):
    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)
        # The fused sm_80 KDA decode step
        # (vllm/models/glm5next/nvidia/ops/kda_decode.py)
        # whenever the decode kernels are on: forward() then defers f_b_proj /
        # g_b_proj to _forward so the kernel can consume f_a / g_a directly;
        # every other path computes them there.
        from vllm import envs as _envs

        self._kda_v2 = bool(_envs.VLLM_AMPERE_DECODE_KERNELS)
        # 64 heads per card (pipeline parallel): the measured decode shapes on
        # v2 (wide schedule beyond one short sequence), token tile sized per
        # step (VLLM_GLM5_DECODE_KDA_WIDE); 0 keeps the earlier split.
        self._kda_wide = bool(
            self._kda_v2
            and self.local_num_heads == 64
            and _envs.VLLM_GLM5_DECODE_KDA_WIDE
        )
        # Token tile from the step's bound on tokens per request; the
        # full speculative window only for 64 heads with the wide path off.
        self._kda_step_tile = self._kda_v2 and (
            self.local_num_heads != 64 or self._kda_wide
        )
        if self._kda_v2:
            # Imported here rather than on first use so that the pre-capture
            # warmup in kernel_warmup.py, which does not import model
            # packages, finds it among the loaded modules.
            from vllm.models.glm5next.nvidia.ops import kda_decode  # noqa: F401

            logger.info_once(
                "sm_80 KDA decode v2 active: f_b_proj + g_b_proj + conv + "
                "gated delta rule + gated RMSNorm in one kernel, token tile "
                "sized per step."
            )
        if self._kda_v2 and self.local_num_heads == 64:
            if self._kda_wide:
                logger.info_once(
                    "sm_80 KDA decode v2 wide: 64 heads, the measured shapes "
                    "of up to 8 sequences of up to 8 tokens "
                    "(VLLM_GLM5_DECODE_KDA_WIDE=1)."
                )
            else:
                logger.info_once(
                    "sm_80 KDA decode v2 wide off (VLLM_GLM5_DECODE_KDA_WIDE=0): "
                    "64 heads take v2 for one sequence of up to 5 tokens only."
                )

        # RecoverSSM (vllm/sm80/kernels/kda/recover.py): one state per
        # request; the verify stores per-token records instead of states.
        self._kda_recover = False
        if self._kda_v2:
            from vllm.sm80.kernels.kda.recover import kda_recover_enabled

            if kda_recover_enabled(vllm_config):
                self._init_kda_recover(config, vllm_config, prefix)

        # VLLM_AMPERE_KDA_PREFILL_WIDE (64 heads) / VLLM_AMPERE_KDA_PREFILL_NARROW (16
        # heads): the layer part of the gate, resolved once.
        self._pp_kda_prefill = False
        if _envs.VLLM_AMPERE_KDA_PREFILL_WIDE or _envs.VLLM_AMPERE_KDA_PREFILL_NARROW:
            from vllm.models.glm5next.nvidia.ops.kda_prefill import (
                enabled_heads,
                use_for_layer,
            )

            self._pp_kda_prefill = use_for_layer(
                self.kda_prefill_backend,
                self.local_num_heads,
                self.head_dim,
                vllm_config.model_config.dtype,
                self.kda_safe_gate,
                self.kda_lower_bound,
                allowed_heads=enabled_heads(),
            )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Glm5NextGDNAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> MambaSpec:
        spec = super().get_kv_cache_spec(vllm_config)
        if self._kda_recover:
            # Recovery records replace draft-position state pages, not the
            # independent prefill checkpoint reservation.
            spec = replace(spec, num_speculative_blocks=0)
        return spec

    def _init_kda_recover(self, config, vllm_config: VllmConfig, prefix: str) -> None:
        import re

        from vllm.sm80.kernels.kda.recover import (
            RecordPool,
            layer_records,
            records_rows,
        )

        kda_layers = [
            i for i, t in enumerate(config.layer_types) if t == "linear_attention"
        ]
        m = re.search(r"layers\.(\d+)\.", prefix + ".")
        if m is None or int(m.group(1)) not in kda_layers:
            raise ValueError(f"RecoverSSM: no KDA layer index in {prefix!r}")
        self._kda_recover = True
        self._kda_recover_index = kda_layers.index(int(m.group(1)))
        device = torch.empty(0).device
        if device.type == "cpu" and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        self._kda_recover_pool = RecordPool.get(
            device, len(kda_layers), records_rows(vllm_config)
        )
        self._kda_recover_records = layer_records(
            self._kda_recover_pool, self._kda_recover_index
        )

    def _fill_deferred_g2(self, g_a: torch.Tensor, g2: torch.Tensor) -> None:
        """g_b_proj(g_a) into the caller's g2 buffer (decode-kernel
        paths that do not fuse the norm): the same GEMM the unfused forward
        runs, copied bitwise into the buffer o_norm reads."""
        g2.copy_(
            self.g_b_proj(g_a)[0].reshape(-1, self.local_num_heads, self.head_dim)
        )
