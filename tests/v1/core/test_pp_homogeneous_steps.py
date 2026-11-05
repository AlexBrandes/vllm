# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Homogeneous PP steps (VLLM_PP_HOMOGENEOUS_STEPS).

With the fork scheduler (vllm_cmp170hx.sched), the flag on and pipeline
parallelism, no scheduling step carries both prefill and decode tokens: while
both kinds of work are ready the steps alternate, and with one kind ready the
step is unrestricted.

The scheduler is built with pipeline_parallel_size=1 (ParallelConfig checks
the world size against the visible GPUs) and switched to PP afterwards; only
the scheduling arithmetic reads it.
"""

import pytest

from vllm.v1.outputs import ModelRunnerOutput

from .sm80_utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

MAX_LEN = 2048


def _pp_scheduler(homogeneous: bool, **kwargs):
    scheduler = create_scheduler(
        max_num_batched_tokens=256, max_num_seqs=8, max_model_len=MAX_LEN, **kwargs
    )
    scheduler.parallel_config.pipeline_parallel_size = 2
    scheduler.use_pp = True
    scheduler.pp_homogeneous_steps = homogeneous
    return scheduler


def _step(scheduler):
    """Schedule one step, feed it back, and return (decode ids, prefill ids)."""
    before = {
        req_id: (request.num_computed_tokens, request.num_tokens)
        for req_id, request in scheduler.requests.items()
    }
    output = scheduler.schedule()
    decodes, prefills = set(), set()
    for req_id in output.num_scheduled_tokens:
        num_computed, total = before[req_id]
        (prefills if num_computed < total - 1 else decodes).add(req_id)
    req_ids = list(output.num_scheduled_tokens.keys())
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=[
                [] if scheduler.requests[r].is_prefill_chunk else [100]
                for r in req_ids
            ],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return decodes, prefills


def _decoding_with_long_prompt(scheduler, num_streams=3, prompt_len=1800):
    streams = create_requests(
        num_requests=num_streams,
        num_tokens=8,
        max_tokens=512,
        ignore_eos=True,
        req_ids=[f"dec{i}" for i in range(num_streams)],
    )
    for request in streams:
        scheduler.add_request(request)
    _step(scheduler)
    (long_req,) = create_requests(
        num_requests=1, num_tokens=prompt_len, max_tokens=4, req_ids=["long"]
    )
    scheduler.add_request(long_req)
    return streams, long_req


def test_flag_off_mixes_prefill_and_decode():
    scheduler = _pp_scheduler(homogeneous=False)
    _decoding_with_long_prompt(scheduler)
    decodes, prefills = _step(scheduler)
    assert decodes == {"dec0", "dec1", "dec2"} and prefills == {"long"}


def test_steps_never_mix_and_alternate_while_both_are_ready():
    scheduler = _pp_scheduler(homogeneous=True)
    streams, long_req = _decoding_with_long_prompt(scheduler)
    kinds = []
    while long_req.is_prefill_chunk or long_req.num_computed_tokens == 0:
        decodes, prefills = _step(scheduler)
        assert not (decodes and prefills), (decodes, prefills)
        kinds.append("d" if decodes else "p")
    # Decodes go first (the previous step was not a decode), then strict
    # alternation until the prompt is done.
    assert kinds[0] == "d"
    assert all(a != b for a, b in zip(kinds, kinds[1:]))
    # A prefill-only step gives the whole budget to the prompt.
    assert kinds.count("p") == -(-1800 // 256)
    assert all(not r.is_prefill_chunk for r in streams)


def test_one_kind_ready_is_unrestricted():
    scheduler = _pp_scheduler(homogeneous=True)
    streams = create_requests(
        num_requests=3, num_tokens=8, max_tokens=512, ignore_eos=True
    )
    for request in streams:
        scheduler.add_request(request)
    _step(scheduler)
    for _ in range(4):
        decodes, prefills = _step(scheduler)
        assert len(decodes) == 3 and not prefills


def test_composes_with_the_decode_aware_prefill_cap():
    scheduler = _pp_scheduler(homogeneous=True, prefill_chunk_with_decodes=64)
    _, long_req = _decoding_with_long_prompt(scheduler)
    _step(scheduler)  # decode-only
    before = long_req.num_computed_tokens
    decodes, prefills = _step(scheduler)
    assert prefills == {"long"} and not decodes
    assert long_req.num_computed_tokens - before == 64


def test_a_blocked_prefill_cannot_hold_decodes_off():
    # One running slot is free in the count but the waiting request can never
    # be admitted (max_num_seqs reached): decodes must keep running.
    scheduler = _pp_scheduler(homogeneous=True)
    scheduler.max_num_running_reqs = scheduler.max_num_active_reqs = 3
    _decoding_with_long_prompt(scheduler)
    progressed = 0
    for _ in range(6):
        decodes, prefills = _step(scheduler)
        assert not prefills
        progressed += bool(decodes)
    assert progressed == 6


def test_inert_without_pp():
    scheduler = create_scheduler(max_model_len=MAX_LEN)
    assert not scheduler.pp_homogeneous_steps
