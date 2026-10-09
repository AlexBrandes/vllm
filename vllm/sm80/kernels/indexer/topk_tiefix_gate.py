# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the indexer top-k tie fix (``VLLM_GLM5_TOPK_TIEFIX``, with
``VLLM_GLM5_TOPK_SORTED`` and ``VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS``).

The GLM kpool sparse indexer (``vllm/models/glm5next/nvidia/sparse_indexer.py``)
asks these after its prefill top-k and after its decode top-k
(:func:`tiefix_decode_topk_`, behind upstream's ``SparseIndexerTopk``). The
kernels (``topk_tiefix.py``) are imported only once the tie fix runs, so with
the flags unset nothing new is loaded.
"""

import functools

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk

logger = init_logger(__name__)


@functools.cache
def use_sorted_topk() -> bool:
    """Whether the indexer sorts its selected indices (VLLM_GLM5_TOPK_SORTED).
    The sort runs inside the tie-fix launch, so it needs VLLM_GLM5_TOPK_TIEFIX
    (``use_tiefix_topk`` rejects it alone). Read once per process; tests call
    cache_clear."""
    import vllm.envs as envs

    on = bool(envs.VLLM_GLM5_TOPK_SORTED)
    if on:
        logger.info_once(
            "GLM-5 sorted top-k active: the sparse indexer's selected indices "
            "are put in ascending order before attention "
            "(VLLM_GLM5_TOPK_SORTED=1; set 0 to disable)"
        )
    return on


@functools.cache
def use_tiefix_topk() -> bool:
    """Whether the indexer repairs boundary ties of the fast top-k kernels
    (VLLM_GLM5_TOPK_TIEFIX). Read once per process; tests call cache_clear.

    Raises on VLLM_GLM5_TOPK_SORTED=1 without the tie fix: the sort exists
    only inside the tie-fix launch. Every indexer forward asks this, so the
    profiling run at startup fails on that combination."""
    import vllm.envs as envs

    on = bool(envs.VLLM_GLM5_TOPK_TIEFIX)
    if not on and envs.VLLM_GLM5_TOPK_SORTED:
        raise ValueError(
            "VLLM_GLM5_TOPK_SORTED=1 needs VLLM_GLM5_TOPK_TIEFIX=1: the indexer "
            "sorts its selected indices inside the tie-fix launch"
        )
    if on:
        logger.info_once(
            "GLM-5 tie-consistent top-k active: the sparse indexer's selected "
            "set breaks ties at the k-th score by lowest index "
            "(VLLM_GLM5_TOPK_TIEFIX=1; set 0 to disable)"
        )
        # First call is the first indexer forward (the profiling run, eager):
        # compile the kernel variants there, before any graph capture.
        from vllm.sm80.kernels.indexer.topk_tiefix import warm_tiefix

        warm_tiefix()
    return on


@functools.cache
def tiefix_split_rows() -> int:
    """Row bound of the split tie fix + sort (VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS;
    0 = off). Only meaningful with the tie fix and the sort both on. Read
    once per process; tests call cache_clear."""
    import vllm.envs as envs

    rows = max(0, int(envs.VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS))
    if rows and not (use_tiefix_topk() and use_sorted_topk()):
        logger.info_once(
            "GLM-5 split tie-fix scan NOT active: VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS "
            "needs VLLM_GLM5_TOPK_TIEFIX=1 and VLLM_GLM5_TOPK_SORTED=1"
        )
        return 0
    if rows:
        logger.info_once(
            "GLM-5 split tie-fix scan active: batches of at most %d rows over "
            "wide logits run the tie fix + sort over (row, chunk) programs "
            "(VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS=%d; set 0 to disable)",
            rows,
            rows,
        )
    return rows


def tiefix_topk_(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    row_ends: torch.Tensor,
    row_starts: torch.Tensor | None = None,
    relative: bool = False,
    sort: bool = False,
) -> torch.Tensor:
    """Resolve boundary ties of an exact top-k to the lowest indices, in
    place; with ``sort`` also sort each row (VLLM_GLM5_TOPK_SORTED) in the
    same launch (two launches in all on the split path, ``use_split``). See
    ``topk_tiefix.py``; imported lazily so that with the flag unset nothing
    new is loaded."""
    from vllm.sm80.kernels.indexer.topk_tiefix import (
        topk_tiefix_,
        use_split,
    )

    split = sort and use_split(
        topk_indices.shape[0], logits.shape[1], tiefix_split_rows()
    )
    return topk_tiefix_(
        logits,
        topk_indices,
        row_ends=row_ends,
        row_starts=row_starts,
        relative=relative,
        sort=sort,
        split=split,
    )


def tiefix_decode_topk_(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    next_n: int,
) -> None:
    """The decode tie fix: with VLLM_GLM5_TOPK_TIEFIX, make the selected set
    that ``SparseIndexerTopk`` (``get_indexer_topk``) just wrote into
    ``topk_indices`` canonical on boundary ties, over the same per-row ends
    as its kernels (``SparseIndexerTopk._row_ends`` of the same arguments)."""
    if use_tiefix_topk():
        tiefix_topk_(
            logits,
            topk_indices,
            row_ends=SparseIndexerTopk._row_ends(seq_lens, next_n, logits.shape[0]),
            sort=use_sorted_topk(),
        )
