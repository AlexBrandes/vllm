# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The scheduler phase helper of the EXL3 prefill kernels (CPU only)."""

from types import SimpleNamespace

import pytest

import vllm.forward_context as forward_context
from vllm.sm80.kernels.exl3.step_phase import max_decode_query_len, prefill_step

# One sampled token plus three draft tokens under verification.
DECODE_QUERY_LEN = 4


def _context(monkeypatch, attn_metadata):
    monkeypatch.setattr(
        forward_context,
        "_forward_context",
        SimpleNamespace(attn_metadata=attn_metadata),
    )


def _attention(max_query_len):
    return SimpleNamespace(max_query_len=max_query_len, num_actual_tokens=0)


def test_max_decode_query_len_counts_the_verified_draft():
    assert max_decode_query_len(SimpleNamespace(num_speculative_tokens=3)) == 4
    assert max_decode_query_len(SimpleNamespace(num_speculative_tokens=0)) == 1


@pytest.mark.parametrize(
    "max_query_len, expected",
    [(1, False), (4, False), (5, True), (309, True), (3460, True)],
)
def test_phase_follows_the_longest_scheduled_request(
    monkeypatch, max_query_len, expected
):
    # A linear-attention layer's metadata has no max_query_len; the first
    # layer that has one decides.
    linear = SimpleNamespace(num_prefills=0, num_decodes=8)
    _context(
        monkeypatch,
        {
            "layers.0.linear_attn": linear,
            "layers.3.self_attn": _attention(max_query_len),
        },
    )
    assert prefill_step(DECODE_QUERY_LEN) is expected


def test_many_decode_requests_stay_a_decode_step(monkeypatch):
    # 64 requests verifying 3 drafts each: 256 rows, still decode.
    _context(monkeypatch, {"layers.3.self_attn": _attention(DECODE_QUERY_LEN)})
    assert prefill_step(DECODE_QUERY_LEN) is False


@pytest.mark.parametrize(
    "attn_metadata",
    [None, {}, [{"layers.3.self_attn": _attention(3460)}], {"a": _attention(None)}],
    ids=["none", "empty", "microbatches", "no_int"],
)
def test_missing_or_unusable_metadata_counts_as_decode(monkeypatch, attn_metadata):
    _context(monkeypatch, attn_metadata)
    assert prefill_step(DECODE_QUERY_LEN) is False


def test_metadata_without_query_len_counts_as_decode_and_says_so(monkeypatch):
    from vllm.sm80.kernels.exl3 import step_phase

    warned = []
    monkeypatch.setattr(step_phase.logger, "warning_once", warned.append)
    _context(monkeypatch, {"layers.0.linear_attn": SimpleNamespace(num_prefills=1)})
    assert prefill_step(DECODE_QUERY_LEN) is False
    assert len(warned) == 1 and "max_query_len" in warned[0]
    # No metadata at all (capture, profiling): decode without a warning.
    _context(monkeypatch, {})
    assert prefill_step(DECODE_QUERY_LEN) is False
    assert len(warned) == 1


def test_no_forward_context_counts_as_decode(monkeypatch):
    monkeypatch.setattr(forward_context, "_forward_context", None)
    assert prefill_step(DECODE_QUERY_LEN) is False
