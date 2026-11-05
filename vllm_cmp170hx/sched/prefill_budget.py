# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A per-step prefill budget beside running decodes (VLLM_MIXED_PREFILL_CHUNK).

On a hybrid Mamba model with a large Mamba block (Qwen4Exp at block size
1,680), the aligned split floors a prefill chunk to whole blocks, so a new long
prompt can get no tokens at all for many steps behind running decodes, or a
burst of prompts can take whole-block steps that stall every decoder.

While any running request is decoding, the budget gives each step
``chunk x (prompts awaiting prefill)`` prefill tokens, at most one Mamba block
(``max(block_size, chunk)``), spent oldest-first in the order the scheduler
visits requests. A cap below the block takes the aligned split's sub-block
path, so a lone prompt advances every step and decoders see short steps.
With no request decoding, prefill uses the scheduler's full budget. With the
variable unset (0), every method is a no-op.

The scheduler (vllm/v1/core/sched/scheduler.py on the fork's vLLM branch)
calls :meth:`start_step` once per ``schedule()``, :meth:`cap` where it sizes a
request's tokens, :meth:`cap_split` in ``_mamba_block_aligned_split`` and
:meth:`charge` once the request is scheduled.
"""

from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.logger import init_logger
from vllm_cmp170hx import plugin

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

logger = init_logger(__name__)


class PrefillBudget:
    def __init__(self, scheduler: "Scheduler") -> None:
        # The scheduler can be built in a process where the general plugin
        # did not run (unit tests); the variable must still resolve.
        plugin.register_fork_env_vars()
        self.scheduler = scheduler
        self.chunk: int = envs.VLLM_MIXED_PREFILL_CHUNK
        # Prefill tokens left in this step; None while no budget applies.
        self.left: int | None = None
        # The budget when the request being sized was capped; the aligned
        # split caps its chunk at it.
        self.step_cap = 0
        self._capped: set[str] = set()
        if self.chunk < 0:
            raise ValueError(f"VLLM_MIXED_PREFILL_CHUNK must be >= 0, got {self.chunk}")
        if self.chunk:
            logger.info(
                "Prefill budget beside running decodes: %d tokens per awaiting "
                "prompt, at most one block",
                self.chunk,
            )

    @property
    def active(self) -> bool:
        return self.left is not None

    def start_step(self) -> None:
        """Set this step's budget: on only while a running request decodes.

        Under pipeline parallelism a decoder skips steps, but the steps in
        between still delay it, so the budget keys on decoders existing, not
        on decode rows in this step.
        """
        self.left = None
        self.step_cap = 0
        self._capped.clear()
        if not self.chunk:
            return
        running = self.scheduler.running
        if all(request.is_prefill_chunk for request in running):
            return
        _, num_waiting = self.scheduler.get_request_counts()
        awaiting = num_waiting + sum(1 for r in running if r.is_prefill_chunk)
        block_size = self.scheduler.cache_config.block_size
        self.left = min(self.chunk * max(1, awaiting), max(block_size, self.chunk))

    def cap(self, request: "Request", num_new_tokens: int, prefill: bool) -> int:
        """``num_new_tokens`` capped at the budget left, for a prefill chunk."""
        if self.left is None or not prefill:
            return num_new_tokens
        self.step_cap = self.left
        self._capped.add(request.request_id)
        return min(num_new_tokens, self.left)

    def cap_split(self, max_prefill_tokens: int) -> int:
        """The aligned split's chunk limit, capped at the request's budget."""
        if self.left is None:
            return max_prefill_tokens
        return min(max_prefill_tokens, self.step_cap)

    def charge(self, request: "Request", num_new_tokens: int) -> None:
        """Spend the budget on a capped request once it is scheduled."""
        if self.left is None or request.request_id not in self._capped:
            return
        self._capped.discard(request.request_id)
        self.left = max(0, self.left - num_new_tokens)
