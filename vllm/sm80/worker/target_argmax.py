# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The idea (hand the sampler logits that hold only the vocabulary-parallel
# argmax when every request is greedy) follows GNmac-lab's greedy-argmax
# patch: Copyright (c) 2026 GNmac-lab, MIT.
"""Greedy target sampling with a per-rank argmax, without the all-gather of
the target logits.

``GPUModelRunner.sample`` computes the target logits once
(``self.model.compute_logits``: the vocabulary-parallel head, then an
all-gather over the TP group) before the grammar bitmask, the sampler and the
rejection sampler read them. When every request of the step is greedy and
nothing reads the logits beyond their argmax, that gather only feeds an
argmax.

With ``VLLM_SM80_TARGET_SHARD_ARGMAX`` the fork's V2 model runner
(``vllm/sm80/worker/graph_mem_worker.py``) runs ``sample`` inside
:func:`step`, which decides from the step's batch, grammar output and sampler
states whether the step is greedy-only (:func:`greedy_only`). While such a
step runs (:func:`active`), the sm_80 DeepSeek-V4.1 target's
``compute_logits`` (``vllm/models/deepseek_v41/nvidia/sm80_model.py``)
returns :func:`argmax_rows` in place of the gathered logits; every other
call (other steps, prompt logprobs, the dummy sampler run) gathers.

:func:`argmax_rows` keeps each rank's own columns
(``compute_logits_local``), takes their argmax per 4096-column chunk and
reduces the chunks over the TP group with the DSpark drafter's slot
all-reduce (``vllm/sm80/spec_decode/shard_argmax.py``: ``chunk_slots``,
``tensor_model_parallel_all_reduce``, ``pick``). The tokens are torch.argmax's
over the gathered row, ties and NaN included: the first index of the largest
value, the first NaN when any is NaN, -0.0 equal to +0.0. The samplers then
get rows that hold 0.0 at that token and -inf everywhere else, in the shape
and dtype ``compute_logits`` returns: the sampler's token, the rejection
sampler's target argmax and the bonus token are that token.

A step is greedy-only when there is no grammar bitmask, the sampler and
rejection sampler are the stock classes (no watermarking or model-specific
sampler), the sampler neither counts NaN in the logits nor returns a sampling
mask, no draft logits narrower than the target vocabulary cut the rejection
sampler's rows, and for every request of the batch: temperature 0, no logits
processing (penalties, logit bias, allowed tokens, bad words, min tokens,
thinking budget, custom processors), no logprobs and no top-k/top-p in the
sampling states (SamplingParams resets them for a greedy request; top-p can
drop the first of tied maxima). On such a step the stock samplers read the
logits only through their argmax: ``apply_sampling_params`` returns them as
they are and ``Sampler.sample`` applies no top-k/top-p. So the sampled
tokens are the gathered path's whenever its rows hold no NaN; on a row with
NaN the token is torch.argmax's first NaN. Every TP rank holds the same
sampling states and batch, so all ranks take the same path and enter the
same collective.

No host synchronization and no data-dependent shape: the slots, the token
ids and the rows have shapes fixed by the step's row count.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

import vllm.envs as envs
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.sm80.spec_decode import shard_argmax
from vllm.triton_utils import triton
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

logger = init_logger(__name__)

SWITCH = "VLLM_SM80_TARGET_SHARD_ARGMAX"

# True while the runner samples a greedy-only step (:func:`step`).
_active = False


def _head(model):
    """The module holding the target's ``lm_head`` and ``logits_processor``."""
    return getattr(model, "language_model", model)


def refusal(runner) -> str | None:
    """Why the per-rank argmax cannot serve this runner's target."""
    if not getattr(runner, "target_argmax_steps", False):
        return (
            f"the model runner {type(runner).__name__} does not sample in "
            "greedy steps (the fork V2 runner of vllm/sm80/worker/"
            f"graph_mem_worker.py, chosen at boot with {SWITCH}=1)"
        )
    if not getattr(runner, "is_last_pp_rank", True):
        return None  # samples nothing
    model = getattr(runner, "model", None)
    if not getattr(model, "target_argmax_logits", False):
        return (
            f"{type(model).__name__} does not return the argmax rows in a "
            "greedy step (compute_logits)"
        )
    if not hasattr(model, "compute_logits_local"):
        return (
            f"{type(model).__name__} has no vocabulary-shard logits "
            "(compute_logits_local)"
        )
    lm = _head(model)
    head = getattr(lm, "lm_head", None)
    processor = getattr(lm, "logits_processor", None)
    if not hasattr(head, "tp_rank") or processor is None:
        return f"{type(model).__name__} has no vocabulary-parallel LM head"
    if getattr(processor, "logits_as_input", False):
        return "the logits processor takes logits as input"
    if processor.org_vocab_size > shard_argmax.MAX_VOCAB:
        return f"a vocabulary above {shard_argmax.MAX_VOCAB} tokens"
    if getattr(runner, "lora_config", None):
        return "LoRA adapters replace the logits processor"
    if getattr(runner, "batch_sharder", None) is not None:
        return "batch-sharded sampling replaces the all-gather already"
    return None


def set_enabled(runner, on: bool) -> None:
    """Turn the per-rank argmax on or off for ``runner``; on is refused
    (ValueError) for a runner it cannot serve (:func:`refusal`)."""
    if on:
        reason = refusal(runner)
        if reason is not None:
            raise ValueError(f"{SWITCH}: {reason}")
        logger.info_once(
            "Greedy target sampling takes a per-rank argmax (%s=1): no "
            "all-gather of the target logits when every request is greedy.",
            SWITCH,
        )
    runner._target_shard_argmax = on


def enabled(runner) -> bool:
    on = getattr(runner, "_target_shard_argmax", None)
    if on is None:
        set_enabled(runner, bool(getattr(envs, SWITCH)))
        on = runner._target_shard_argmax
    return on


def greedy_only(runner, input_batch, grammar_output) -> bool:
    """True when the step's samplers read the target logits only through
    their argmax (see the module docstring). Host state only."""
    if grammar_output is not None:
        return False
    sampler = runner.sampler
    if type(sampler) is not Sampler:
        return False
    rejection = runner.rejection_sampler
    if rejection is not None and type(rejection) is not RejectionSampler:
        return False
    if sampler.compute_nans or sampler.return_sampling_mask:
        return False
    draft_logits = getattr(getattr(runner, "speculator", None), "draft_logits", None)
    if draft_logits is not None:
        vocab = _head(runner.model).logits_processor.org_vocab_size
        if draft_logits.shape[-1] < vocab:
            return False
    idx = input_batch.idx_mapping_np
    states = sampler.sampling_states
    return (
        bool(np.all(states.temperature.np[idx] == 0.0))
        and not np.any(sampler.needs_logits_processing[idx])
        and sampler.get_logprobs_dims(idx) is None
        # No top-k/top-p (get_top_k_top_p): top-p may drop the first of tied
        # maxima. SamplingParams resets both for a greedy request.
        and bool(np.all(states.top_k.np[idx] == states.vocab_size))
        and bool(np.all(states.top_p.np[idx] == 1.0))
    )


@contextmanager
def step(runner, input_batch, grammar_output) -> Iterator[None]:
    """Run one ``GPUModelRunner.sample``: while it runs, :func:`active` says
    whether the target's ``compute_logits`` returns the argmax rows."""
    global _active
    _active = enabled(runner) and greedy_only(runner, input_batch, grammar_output)
    try:
        yield
    finally:
        _active = False


def active() -> bool:
    return _active


def argmax_rows(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """Rows holding 0.0 at torch.argmax of ``model.compute_logits(
    hidden_states)`` and -inf elsewhere, in its shape and dtype, without
    gathering the logits."""
    lm = _head(model)
    local = model.compute_logits_local(hidden_states)
    rows, width = local.shape
    rank, tp_size = lm.lm_head.tp_rank, lm.lm_head.tp_size
    vocab = lm.logits_processor.org_vocab_size
    out = local.new_full((rows, vocab), float("-inf"))
    if rows == 0:
        # Every rank has the same rows: none enters the all-reduce.
        return out
    # The gathered row is the ranks' columns in rank order, cut at the
    # vocabulary size.
    col_start = rank * width
    n_cols = min(max(vocab - col_start, 0), width)
    cols = local[:, :n_cols]
    if cols.dtype != torch.float32:
        cols = cols.float()  # exact; keeps order, ties and NaN
    chunks = triton.cdiv(width, shard_argmax.BLOCK)
    slots = shard_argmax.chunk_slots(cols, col_start, rank, tp_size, chunks)
    if tp_size > 1:
        slots = tensor_model_parallel_all_reduce(slots)
    return out.scatter_(1, shard_argmax.pick(slots, vocab)[:, None], 0.0)


def warm(runner, hidden_states: torch.Tensor | None = None) -> None:
    """With the switch on, refuse a runner it cannot serve and compile the
    slot kernels on ``hidden_states`` (the dummy sampler run at boot), or on
    one zero row of the model's hidden size (the A/B switch's arm), outside
    graph capture. Every rank of the last pipeline stage calls it with the
    same rows."""
    if not (enabled(runner) and getattr(runner, "is_last_pp_rank", True)):
        return
    if hidden_states is None:
        config = runner.model_config
        hidden_states = torch.zeros(
            1, config.get_hidden_size(), dtype=config.dtype, device=runner.device
        )
    with torch.inference_mode():
        argmax_rows(runner.model, hidden_states)
