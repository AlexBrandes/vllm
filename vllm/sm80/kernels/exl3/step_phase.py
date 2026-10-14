# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The scheduler phase of the running forward step, for kernels that serve
only prefill steps.

A step is a prefill step when the scheduler gave some request more than
1 + num_speculative_tokens tokens (a prompt or a chunk of one); otherwise it
is a decode step (one token per request, or the verified draft of a
speculative step). This is the rule the Qwen4Exp QSA layer uses to pick its
prefill configuration (``max_query_len > 1 + num_speculative_tokens``). It
is read from the attention metadata of the current forward context, so it
reflects what the scheduler sent, never the row count a layer sees: a decode
step of many requests stays a decode step.

Drafters: an MTP or EAGLE drafter without parallel drafting runs with the
target step's query lengths, so its calls get the step's phase. A drafter
with parallel drafting slots (or DFlash) schedules more than 1 +
num_speculative_tokens tokens per request in decode steps; this rule would
call those steps prefill, so it does not cover such drafters.

Without a forward context, without attention metadata (profiling runs and
piecewise CUDA-graph capture), or with per-microbatch metadata (a list,
under dual-batch overlap; the QSA layer reads the first microbatch), the
step counts as a decode step, so those paths keep the kernels that serve
decode. A captured CUDA graph replays the kernels it was captured with, so a
prefill step padded to a captured graph size runs the decode kernels.
"""

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

logger = init_logger(__name__)


def max_decode_query_len(vllm_config) -> int:
    """Most tokens one request is scheduled in a decode step: the sampled
    token plus the speculative draft being verified."""
    return 1 + vllm_config.num_speculative_tokens


def prefill_step(max_decode_query_len: int) -> bool:
    """True when the running forward step is a prefill step (some request
    has more than ``max_decode_query_len`` tokens scheduled)."""
    if not is_forward_context_available():
        return False
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return False
    for layer_metadata in metadata.values():
        query_len = getattr(layer_metadata, "max_query_len", None)
        if isinstance(query_len, int):
            return query_len > max_decode_query_len
    if metadata:
        logger.warning_once(
            "No attention metadata of this step carries an integer "
            "max_query_len: the step counts as a decode step and the EXL3 "
            "prefill kernels are not used."
        )
    return False
