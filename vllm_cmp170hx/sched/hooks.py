# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calls ``vllm/v1/core/sched/scheduler.py`` makes into the fork.

``Scheduler.schedule()`` has no extension point inside its two request loops,
where fair chunked prefill charges its prefill budget, PP decode spreading
skips decodes and a cached prompt tail keeps its prefill shape. The upstream
file makes one call per site into the per-step object :func:`step` returns.
For a scheduler without the fork mixin (``vllm_cmp170hx.sched.scheduler``) that
object is :data:`UPSTREAM_STEP`, whose every method leaves the schedule as
upstream computes it.

This module is imported by the upstream scheduler module, so it imports
nothing from it.
"""

from typing import Any

_NO_FAILED_INPUTS: frozenset[str] = frozenset()


class UpstreamStep:
    """Per-step hooks of a scheduler without the fork mixin: no-ops."""

    __slots__ = ()

    def skip(self, request: Any) -> bool:
        return False

    def cap(self, request: Any, num_new_tokens: int) -> int:
        return num_new_tokens

    def admit(self, request: Any, num_computed_tokens: int, num_new_tokens: int) -> int:
        return num_new_tokens

    def ceil(self, token_budget: int, input_budget: int) -> int | None:
        return None

    def refund(self, request: Any, restored: int) -> None:
        pass

    def charge(self, request: Any, num_new_tokens: int) -> None:
        pass

    def prompt_tail(self, request: Any, num_computed_tokens: int) -> bool:
        return False

    def finish(self, total_num_scheduled_tokens: int, num_scheduled_tokens) -> None:
        pass


UPSTREAM_STEP = UpstreamStep()


def sm80_step(
    scheduler: Any,
    token_budget: int,
    draft_slots: int,
    long_prefill_token_threshold: int,
):
    """This step's hook object (called once per ``schedule()``, before the
    running loop)."""
    new_step = getattr(scheduler, "_sm80_new_step", None)
    if new_step is None:
        return UPSTREAM_STEP
    return new_step(token_budget, draft_slots, long_prefill_token_threshold)


def mamba_block_size(scheduler: Any) -> int:
    """Block grid of Mamba align-mode chunk ends: the Mamba group's own block
    size (``Sm80SchedulerMixin.mamba_state_block_size``), else upstream's
    ``cache_config.block_size``."""
    return (
        getattr(scheduler, "mamba_state_block_size", None)
        or scheduler.cache_config.block_size
    )


# Sub-block chunk ends are rounded up to this many tokens (the KDA/FLA chunk).
SUBBLOCK_ALIGN = 64


def mamba_subblock_end(
    scheduler: Any, start: int, end: int, block_size: int, max_prefill_tokens: int
) -> int:
    """End of a prefill chunk inside a Mamba block wider than one chunk.

    Upstream advances such a block in chunks of the full budget and a remainder
    (a 4608-token block under a 2048-token budget: 2048, 2048, 512). Under
    pipeline parallelism at most two micro-batches are in flight, so the first
    stage finishes the short chunk and waits for the next stage's full one;
    on GLM-5.3-Flash PP2 that idles each card ~725 ms per block, ~20% of cold
    prefill. With the fork scheduler the block is cut into equal parts of
    ceil(block / ceil(block / budget)) tokens, rounded up to SUBBLOCK_ALIGN
    (4608 / 2048 -> three chunks of 1536), measured from the block start, so
    every chunk still ends on or before the block boundary. A chunk with no
    sub-block end inside it (budget-limited) keeps upstream's end. Without the
    fork mixin the end is upstream's."""
    if (
        getattr(scheduler, "mamba_state_block_size", None) is None
        or block_size <= max_prefill_tokens
    ):
        return end
    parts = -(-block_size // max_prefill_tokens)
    width = -(-block_size // parts)
    width = min(max_prefill_tokens, -(-width // SUBBLOCK_ALIGN) * SUBBLOCK_ALIGN)
    block_start = start - start % block_size
    sub_end = block_start + (end - block_start) // width * width
    return sub_end if sub_end > start else end


def failed_inputs(scheduler: Any, model_runner_output: Any):
    """Request IDs whose step output was computed from inputs a worker could
    not produce (``ModelRunnerOutput.failed_req_ids``); see
    ``vllm_cmp170hx.failed_inputs``. Empty when none failed."""
    failed_req_ids = model_runner_output.failed_req_ids
    if not failed_req_ids:
        return _NO_FAILED_INPUTS
    from vllm_cmp170hx.failed_inputs import handle_failed_inputs

    return handle_failed_inputs(scheduler, failed_req_ids)
