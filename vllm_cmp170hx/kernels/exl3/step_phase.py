# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The scheduler phase of the running forward step, for phase-specific kernels.

A step is a prefill step when the scheduler gave some request more than
1 + num_speculative_tokens tokens (a prompt or a chunk of one); otherwise it
is a decode step (one token per request, or the verified draft of a
speculative step). This is the rule the Qwen4Exp QSA layer uses to pick its
prefill configuration (``max_query_len > 1 + num_speculative_tokens``). It
is read from the attention metadata of the current forward context, so it
reflects what the scheduler sent, never the row count a layer sees: a decode
step of many requests stays a decode step.

Drafters: an MTP or EAGLE drafter without parallel drafting runs with the
target step's query lengths. Parallel drafting cannot use that heuristic:
``require_explicit`` instead requires homogeneous prefill token counts from
hybrid attention metadata. Missing counts, mixed batches and speculative
verification all keep the decode path, regardless of query length.

Without a forward context, without attention metadata (profiling runs and
piecewise CUDA-graph capture), or with per-microbatch metadata (a list,
under dual-batch overlap; the QSA layer reads the first microbatch), the
step counts as a decode step, so those paths keep the kernels that serve
decode. A captured CUDA graph replays the kernels it was captured with, so a
prefill step padded to a captured graph size runs the decode kernels.

That conservative fallback is not proof of homogeneous decode. ``decode_step``
requires explicit hybrid metadata with no prefill and positive decode/speculative
tokens; unknown and mixed steps keep their existing kernels.
"""

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

logger = init_logger(__name__)


def max_decode_query_len(vllm_config) -> int:
    """Most tokens one request is scheduled in a decode step: the sampled
    token plus the speculative draft being verified."""
    return 1 + vllm_config.num_speculative_tokens


def prefill_step(max_decode_query_len: int, *, require_explicit: bool = False) -> bool:
    """Classify a scheduled step, failing closed for parallel drafting.

    Explicit mode requires hybrid prefill/decode/speculative token counts;
    the query-length heuristic remains for the covered nonparallel drafters.
    """
    if not is_forward_context_available():
        return False
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return False
    if require_explicit:
        observed = False
        for layer_metadata in metadata.values():
            # MLA metadata alone cannot distinguish parallel draft slots.
            if not hasattr(layer_metadata, "num_spec_decode_tokens"):
                continue
            counts = (
                getattr(layer_metadata, "num_prefill_tokens", None),
                getattr(layer_metadata, "num_decode_tokens", None),
                layer_metadata.num_spec_decode_tokens,
            )
            if not all(isinstance(value, int) for value in counts):
                return False
            prefill, decode, speculative = counts
            if prefill <= 0 or decode != 0 or speculative != 0:
                return False
            observed = True
        return observed
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


def decode_step() -> bool:
    """Require homogeneous decode metadata, including during full graph capture."""
    if not is_forward_context_available():
        return False
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return False
    observed = False
    for layer_metadata in metadata.values():
        if not hasattr(layer_metadata, "num_spec_decode_tokens"):
            continue
        counts = (
            getattr(layer_metadata, "num_prefill_tokens", None),
            getattr(layer_metadata, "num_decode_tokens", None),
            layer_metadata.num_spec_decode_tokens,
        )
        if not all(type(value) is int and value >= 0 for value in counts):
            return False
        prefill, decode, speculative = counts
        if prefill != 0 or decode + speculative == 0:
            return False
        observed = True
    return observed
