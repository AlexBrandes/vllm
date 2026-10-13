# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the Qwen QSA canonical top-k (``VLLM_QWEN4_QSA_TOPK_CANONICAL``).

The QSA indexer (``vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py``) calls
:func:`canonical_qsa_topk_` right after its prefill and decode top-k. The
switch is read and the banner logged once, when the indexer imports this
module. The kernels (``topk_tiefix.py``) are imported only once the tie fix
runs, so with the switch unset nothing new is loaded.
"""

import functools

import torch

import vllm.envs as envs
from vllm.logger import init_logger

# Resolve before warmup/capture; never read the environment in the hot path.
_QSA_TOPK_CANONICAL = bool(envs.VLLM_QWEN4_QSA_TOPK_CANONICAL)
if _QSA_TOPK_CANONICAL:
    init_logger(__name__).info_once(
        "Qwen QSA canonical top-k active: blocks tied with the k-th score go "
        "to the lowest block index (VLLM_QWEN4_QSA_TOPK_CANONICAL=1; set 0 "
        "to disable)"
    )


@functools.cache
def _canonical_topk_fix():
    """The shared indexer tie fix (``topk_tiefix_``), imported on first use so
    that with VLLM_QWEN4_QSA_TOPK_CANONICAL unset nothing new is loaded. The
    first call is the first QSA top-k, which runs eagerly before any CUDA
    graph capture: compile the kernel variants there."""
    from vllm.sm80.kernels.indexer.topk_tiefix import topk_tiefix_, warm_tiefix

    warm_tiefix()
    return topk_tiefix_


def canonical_qsa_topk_(
    logits: torch.Tensor,
    block_indices: torch.Tensor,
    row_ends: torch.Tensor,
    sort: bool,
) -> bool:
    """With VLLM_QWEN4_QSA_TOPK_CANONICAL, make the selection the QSA top-k
    just wrote into ``block_indices`` canonical in place: the top-k places
    blocks tied with the k-th score by atomic arrival order, and the
    contested slots go to the lowest block indices. With ``sort`` each row is
    also sorted in the same launch (same bytes as the indexer's separate
    sort). Returns whether it ran, i.e. whether the caller's sort is done."""
    if not _QSA_TOPK_CANONICAL:
        return False
    _canonical_topk_fix()(logits, block_indices, row_ends=row_ends, sort=sort)
    return True
