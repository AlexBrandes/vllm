# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The scheduler's hooks for the fork plugin's per-step prefill budget
(vllm_cmp170hx.sched.prefill_budget, VLLM_MIXED_PREFILL_CHUNK)."""

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

CHUNK = 64
BLOCK = 16


def _step(scheduler: Scheduler) -> SchedulerOutput:
    """Schedule one step and sample one token for each request whose prompt
    is complete."""
    output = scheduler.schedule()
    req_ids = list(output.num_scheduled_tokens)
    sampled = [
        [1000]
        if scheduler.requests[req_id].num_computed_tokens
        >= scheduler.requests[req_id].num_prompt_tokens
        else []
        for req_id in req_ids
    ]
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return output


def _scheduler(monkeypatch, chunk: int | None) -> Scheduler:
    if chunk is None:
        monkeypatch.delenv("VLLM_MIXED_PREFILL_CHUNK", raising=False)
    else:
        monkeypatch.setenv("VLLM_MIXED_PREFILL_CHUNK", str(chunk))
    return create_scheduler(max_num_batched_tokens=1024, block_size=BLOCK)


def _requests(*prompt_lens: int):
    return [
        create_requests(1, num_tokens=n, max_tokens=64, req_ids=[f"r{i}"])[0]
        for i, n in enumerate(prompt_lens)
    ]


def test_budget_caps_prefill_beside_a_decoder_oldest_first(monkeypatch):
    scheduler = _scheduler(monkeypatch, CHUNK)
    decoder, first, second = _requests(32, 400, 400)

    scheduler.add_request(decoder)
    # No request decodes yet: the prompt gets the full budget.
    assert _step(scheduler).num_scheduled_tokens == {"r0": 32}

    scheduler.add_request(first)
    # One prompt awaits prefill beside the decoder: CHUNK tokens.
    assert _step(scheduler).num_scheduled_tokens == {"r0": 1, "r1": CHUNK}

    scheduler.add_request(second)
    # Two prompts await, but one step's budget is at most one block
    # (max(block, CHUNK) = CHUNK), spent by the older prompt first.
    assert _step(scheduler).num_scheduled_tokens == {"r0": 1, "r1": CHUNK}


def test_budget_scales_with_awaiting_prompts_up_to_the_block(monkeypatch):
    chunk = BLOCK // 2
    scheduler = _scheduler(monkeypatch, chunk)
    decoder, first, second = _requests(32, 400, 400)
    scheduler.add_request(decoder)
    _step(scheduler)
    scheduler.add_request(first)
    scheduler.add_request(second)
    # chunk x 2 awaiting = BLOCK, the cap; oldest-first: r1 takes it all.
    assert _step(scheduler).num_scheduled_tokens == {"r0": 1, "r1": BLOCK}


def test_no_budget_without_a_decoder(monkeypatch):
    scheduler = _scheduler(monkeypatch, CHUNK)
    for request in _requests(400, 400):
        scheduler.add_request(request)
    assert _step(scheduler).num_scheduled_tokens == {"r0": 400, "r1": 400}


@pytest.mark.parametrize("chunk", [None, 0])
def test_unset_is_upstream_behaviour(monkeypatch, chunk):
    scheduler = _scheduler(monkeypatch, chunk)
    decoder, first = _requests(32, 400)
    scheduler.add_request(decoder)
    _step(scheduler)
    scheduler.add_request(first)
    assert _step(scheduler).num_scheduled_tokens == {"r0": 1, "r1": 400}
