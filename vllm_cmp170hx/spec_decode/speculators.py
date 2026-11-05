# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The speculator factory: vLLM's ``init_speculator``
(``vllm/v1/worker/gpu/spec_decode/__init__.py``) asks :func:`create_speculator`
first and builds its own speculator when it returns None.

- ``dspark``: :func:`vllm_cmp170hx.spec_decode.dspark_width.create_dspark_speculator`
  (draft widths, shard argmax, the DSpark switches).
- the speculator vLLM resolves to its DFlash2 speculator
  (``speculative_config.speculator_config.speculator_class``):
  :class:`Sm80DFlash2Speculator`, its subclass (load-following draft width,
  split draft tail). vLLM's own resolution decides which DFlash-family
  speculator a draft checkpoint gets (XPress, LiLiCorr, DFlash2, ...).

Everything else is vLLM's own speculator.
"""

import torch

from vllm.config import VllmConfig

# The upstream speculator Sm80DFlash2Speculator extends.
UPSTREAM_DFLASH2 = "vllm.v1.worker.gpu.spec_decode.dflash2.speculator.DFlash2Speculator"


def create_speculator(vllm_config: VllmConfig, device: torch.device):
    """The plugin's speculator for this config, or None (vLLM builds its own)."""
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "dspark":
        from vllm_cmp170hx.spec_decode.dspark_width import create_dspark_speculator

        return create_dspark_speculator(vllm_config, device)
    speculator_config = speculative_config.speculator_config
    if getattr(speculator_config, "speculator_class", None) != UPSTREAM_DFLASH2:
        return None
    from vllm_cmp170hx.spec_decode.dflash2_speculator import Sm80DFlash2Speculator

    return Sm80DFlash2Speculator(vllm_config, device)
