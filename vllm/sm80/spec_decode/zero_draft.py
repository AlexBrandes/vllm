# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Zero-draft steps for the block drafters (DFlash, DSpark).

Dynamic speculative decoding (``num_speculative_tokens_per_batch_size``) may
schedule zero drafts for a step, so the next step verifies nothing and runs
as plain decode. A block drafter has no width 0: ``set_width(0)`` keeps the
configured width, so ``propose`` would still run the whole draft block and its
sampling, only for the runner to overwrite every draft with -1.

On such a step only the drafter's context K/V is written: the K/V of this
step's target tokens, which the drafter attends to once drafting resumes. The
block forward and the sampling are skipped. Under data parallelism every rank
must join the same collectives, so the full step runs there (dynamic
speculative decoding is disabled with data parallelism anyway).
"""


def draft_context_only(speculator, num_speculative_tokens, num_target_tokens) -> bool:
    """Store the context K/V and return True when this step drafts nothing.

    Called by ``DFlashSpeculator.propose`` after the drafter inputs (context
    positions and slots) are prepared; True means ``propose`` returns without
    drafting.
    """
    if num_speculative_tokens != 0 or speculator.dp_size > 1:
        return False
    speculator._precompute_context_kv(0, num_target_tokens)
    return True
