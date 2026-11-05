# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The per-step prefill budget beside running decodes
(vllm_cmp170hx/sched/prefill_budget.py). The scheduler hooks that call it are
tested on the vLLM branch (tests/v1/core/test_prefill_budget.py)."""

from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm_cmp170hx import plugin
from vllm_cmp170hx.envs import FORK_ENV_VARS
from vllm_cmp170hx.sched.prefill_budget import PrefillBudget


def _request(request_id: str, prefill: bool) -> SimpleNamespace:
    return SimpleNamespace(request_id=request_id, is_prefill_chunk=prefill)


def _scheduler(running, num_waiting=0, block_size=1680) -> SimpleNamespace:
    return SimpleNamespace(
        running=running,
        get_request_counts=lambda: (len(running), num_waiting),
        cache_config=SimpleNamespace(block_size=block_size),
    )


def _budget(monkeypatch, chunk, scheduler) -> PrefillBudget:
    monkeypatch.setenv("VLLM_MIXED_PREFILL_CHUNK", str(chunk))
    return PrefillBudget(scheduler)


def test_env_var_is_registered_with_vllm():
    plugin.register_fork_env_vars()
    for name, getter in FORK_ENV_VARS.items():
        assert envs.environment_variables[name] is getter


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_MIXED_PREFILL_CHUNK", raising=False)
    budget = PrefillBudget(_scheduler([_request("d", False)], num_waiting=1))
    budget.start_step()
    assert not budget.active
    assert budget.cap(_request("p", True), 5000, prefill=True) == 5000
    assert budget.cap_split(2048) == 2048


def test_no_budget_without_a_decoder(monkeypatch):
    budget = _budget(monkeypatch, 560, _scheduler([_request("p", True)], 1))
    budget.start_step()
    assert not budget.active


def test_budget_is_chunk_per_awaiting_prompt_up_to_one_block(monkeypatch):
    running = [_request("d", False), _request("p", True)]
    budget = _budget(monkeypatch, 560, _scheduler(running, num_waiting=1))
    budget.start_step()
    assert budget.left == 1120  # 2 awaiting x 560
    budget = _budget(monkeypatch, 560, _scheduler(running, num_waiting=5))
    budget.start_step()
    assert budget.left == 1680  # one block
    budget = _budget(monkeypatch, 4096, _scheduler(running, num_waiting=5))
    budget.start_step()
    assert budget.left == 4096  # a chunk wider than the block


def test_spent_oldest_first_and_only_by_capped_prefills(monkeypatch):
    running = [_request("d", False)]
    budget = _budget(monkeypatch, 560, _scheduler(running, num_waiting=2))
    budget.start_step()
    old, new, decode = _request("a", True), _request("b", True), running[0]

    assert budget.cap(decode, 6, prefill=False) == 6
    budget.charge(decode, 6)
    assert budget.left == 1120

    assert budget.cap(old, 5000, prefill=True) == 1120
    assert budget.cap_split(2048) == 1120
    budget.charge(old, 1000)
    assert budget.left == 120
    assert budget.cap(new, 5000, prefill=True) == 120
    budget.charge(new, 120)
    assert budget.cap(_request("c", True), 5000, prefill=True) == 0

    budget.start_step()  # the next step starts over
    assert budget.left == 1120


def test_negative_chunk_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="must be >= 0"):
        _budget(monkeypatch, -1, _scheduler([]))
