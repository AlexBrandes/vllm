# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperInkling: the verify length of each step, chosen online.

Ported from HyperInkling (github.com/Pento95/HyperInkling,
patches/vllm-hyperinkling-sm86.patch, Apache-2.0); the step-cost prior slope
is configurable for the CMP 170HX.

A block drafter (DSpark) drafts a fixed number of tokens in one pass; how
many of them the target verifies is free (dynamic speculative decoding
truncates the drafts). Verifying more costs more, and on a MoE much more:
each verified row routes to its own experts. So the length that pays depends
on the text (predictable text accepts long windows) and on the batch.

As basalt's draft policy (jesdga95/basalt, include/basalt/spec/draft_policy.hpp,
MIT), choose the length with the most expected committed tokens per
millisecond:

    E(k) = 1 + p_0 + p_0 p_1 + ... + p_0 ... p_{k-1}

with p_i the conditional acceptance of draft position i (decayed counts over
recent requests), against the measured wall time of pure decode steps of
that batch size and length. Lengths not measured lately take their cost from
the measured ones through a prior shape. One length for the whole batch, so
decode batches stay uniform and keep their CUDA graphs.

Enabled by HYPERINKLING_SPEC_ADAPTIVE="1,2,3,4,5,7" (the candidate lengths)
on top of a `num_speculative_tokens_per_batch_size` schedule.
"""

import os
import time

_PRIOR_P = 0.6
_PRIOR_WEIGHT = 4.0
_DECAY = 0.98
_COST_EMA = 0.1
_EXPLORE_EVERY = 16
_MIN_COST_SAMPLES = 3


def adaptive_candidates() -> list[int] | None:
    raw = os.environ.get("HYPERINKLING_SPEC_ADAPTIVE", "")
    if not raw:
        return None
    return sorted({int(k) for k in raw.split(",") if k.strip()})


# Step time relative to k = 1. HyperInkling measured 0.073 per extra verified
# token on an RTX 3090 Ti; on the CMP 170HX a k = 4 verify costs about 1.5x a
# plain step (MoE rows route to their own experts). Measured costs replace the
# prior after _MIN_COST_SAMPLES steps of a (batch, k).
_PRIOR_SLOPE = float(os.environ.get("HYPERINKLING_SPEC_ADAPTIVE_PRIOR_SLOPE", "0.073"))


def _prior_shape(k: int) -> float:
    return 1.0 + _PRIOR_SLOPE * (k - 1)


class AdaptiveVerifyLength:
    def __init__(self, candidates: list[int], max_k: int) -> None:
        assert candidates and max(candidates) <= max_k
        self.candidates = candidates
        self.ok = [_PRIOR_P * _PRIOR_WEIGHT] * max_k
        self.seen = [_PRIOR_WEIGHT] * max_k
        self.cost: dict[tuple[int, int], tuple[float, int]] = {}
        self.num_choices = 0
        self.last_time: float | None = None
        self.last_choice: dict[int, int] = {}

    def p(self, i: int) -> float:
        return self.ok[i] / self.seen[i]

    def expected_tokens(self, k: int) -> float:
        total, survive = 1.0, 1.0
        for i in range(k):
            survive *= self.p(i)
            total += survive
        return total

    def cost_ms(self, batch: int, k: int) -> float:
        own = self.cost.get((batch, k))
        if own is not None and own[1] >= _MIN_COST_SAMPLES:
            return own[0]
        # Scale from the measured lengths of this batch size, else 1.0 ms
        # units: only ratios between lengths of one batch size matter.
        ref = [
            (v[0] / _prior_shape(kk))
            for (b, kk), v in self.cost.items()
            if b == batch and v[1] >= _MIN_COST_SAMPLES
        ]
        unit = sum(ref) / len(ref) if ref else 1.0
        return unit * _prior_shape(k)

    def choose(self, batch: int) -> int:
        self.num_choices += 1
        rates = [
            (self.expected_tokens(k) / self.cost_ms(batch, k), k)
            for k in self.candidates
        ]
        best = max(rates)[1]
        if self.num_choices % _EXPLORE_EVERY == 0:
            # Keep the neighbours' costs and the next position's acceptance
            # measured: alternate one step longer and one step shorter.
            i = self.candidates.index(best)
            step = 1 if (self.num_choices // _EXPLORE_EVERY) % 2 else -1
            best = self.candidates[min(max(i + step, 0), len(self.candidates) - 1)]
        self.last_choice[batch] = best
        return best

    def observe_request(self, num_verified: int, num_accepted: int) -> None:
        """One request's verdict: drafts 0..num_accepted-1 accepted and, if
        any was left, draft num_accepted rejected."""
        reached = min(num_accepted + 1, num_verified)
        for i in range(reached):
            self.ok[i] = self.ok[i] * _DECAY + (1.0 if i < num_accepted else 0.0)
            self.seen[i] = self.seen[i] * _DECAY + 1.0
        # Positions this verdict did not reach drift toward the last one it
        # did, so a long-ago run of predictable text does not keep them high.
        target = self.p(reached - 1) if reached else _PRIOR_P
        for i in range(reached, len(self.ok)):
            self.ok[i] = self.ok[i] * _DECAY + target * (1 - _DECAY) * _PRIOR_WEIGHT
            self.seen[i] = self.seen[i] * _DECAY + (1 - _DECAY) * _PRIOR_WEIGHT

    def observe_step(self, key: tuple[int, int] | None) -> None:
        """Called once per engine step output: `key` is (batch, verified
        length) of a pure, uniform decode step, else None."""
        now = time.monotonic()
        last, self.last_time = self.last_time, now
        if key is None or last is None:
            if key is None:
                self.last_time = None
            return
        ms = (now - last) * 1e3
        old = self.cost.get(key)
        if old is None:
            self.cost[key] = (ms, 1)
        else:
            self.cost[key] = (old[0] + _COST_EMA * (ms - old[0]), old[1] + 1)
