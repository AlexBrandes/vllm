# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state of the fork's GLM-5.3-Flash classes.

The fork GLM classes (``sm80_model.py``) return this class from
``get_model_state_cls`` wherever vLLM's own resolution would pick
``MambaHybridModelState`` (hybrid or attention-free configs), and that
resolution's choice otherwise (``resolve_glm_model_state_cls``). It adds the
RecoverSSM state whenever the fork's KDA RecoverSSM is enabled
(``vllm/sm80/kernels/kda/recover.py``), next to upstream's
``cache_config.use_kda_recoverssm`` switch. The cache config itself is left
alone: upstream reads ``use_kda_recoverssm`` in other places too (CUDA graph
capture sizes, config checks), which must not change.
"""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


class Glm5NextModelState(MambaHybridModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        if self.recoverssm is None:
            from vllm.sm80.kernels.kda.recover import kda_recover_enabled

            if kda_recover_enabled(vllm_config):
                self.recoverssm = RecoverSSMState()


class _WithoutModelStateHook:
    """A model as vLLM's model-state resolution sees it, minus the
    ``get_model_state_cls`` hook, so the resolution takes its own path."""

    def __init__(self, model: nn.Module) -> None:
        self.modules = model.modules


def resolve_glm_model_state_cls(vllm_config: VllmConfig, model: nn.Module) -> type:
    """vLLM's model-state choice for ``model`` without the hook, with
    :class:`Glm5NextModelState` in place of ``MambaHybridModelState``."""
    from vllm.v1.worker.gpu.model_states import resolve_model_state_cls

    cls = resolve_model_state_cls(vllm_config, _WithoutModelStateHook(model))
    return Glm5NextModelState if cls is MambaHybridModelState else cls
