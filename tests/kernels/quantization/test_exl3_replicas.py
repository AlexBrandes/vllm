# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for replicated hot routed experts (vllm/sm80/kernels/exl3/replicas.py):
plan validation, the owner rule's exactly-once and determinism properties, exact
sums against ordinary placement, the real loader's replica bytes and their
accounting, the per-call map kernel (Triton interpreter, a subprocess with
``TRITON_INTERPRET=1``) against the reference, and the experts' dispatch."""

import hashlib
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open

from tests.kernels.quantization.test_exl3_expert_buffers import (
    EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    _check_bytes,
    _config,
    _file_array,
    synthetic,  # noqa: F401
)
from tests.kernels.quantization.test_exl3_placement import PREFIX, _Layer
from vllm import envs
from vllm.sm80.kernels.exl3 import quant
from vllm.sm80.kernels.exl3 import replicas as rp
from vllm.sm80.kernels.exl3.utils import Exl3LoadLedger

ROOT = Path(__file__).resolve().parents[3]


def _enable(monkeypatch, tmp_path, plan):
    path = tmp_path / "replicas.json"
    path.write_text(json.dumps(plan))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setenv("VLLM_EXL3_EXPERT_REPLICAS", str(path))
    monkeypatch.setenv("VLLM_EXL3_EXPERT_REPLICAS_SHA256", digest)
    monkeypatch.delenv("VLLM_EXL3_EXPERT_PLACEMENT", raising=False)
    monkeypatch.delenv("VLLM_EXL3_EXPERT_PLACEMENT_SHA256", raising=False)
    rp.load_plan.cache_clear()
    return path, digest


def _replicas(experts=384, ep=4, hot=16, rank=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(experts, generator=g)[:hot].tolist()
    return rp.layer_replicas(tuple(order), rp.home_ranks(ep, experts), rank, ep)


def _routes(rows, topk, experts, seed, skew=0.0):
    """Top-k ids like the router's: distinct per row, some experts favoured."""
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(rows, experts, generator=g)
    scores[:, : experts // 16] += skew
    return scores.topk(topk, dim=-1)[1].to(torch.int32)


def _busiest(ids, maps):
    return max(len({int(e) for e in ids.flatten() if m[e] >= 0}) for m in maps)


# ------------------------------------------------------------------ env and plan


def test_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_EXPERT_REPLICAS", raising=False)
    monkeypatch.delenv("VLLM_EXL3_EXPERT_REPLICAS_SHA256", raising=False)
    assert envs.VLLM_EXL3_EXPERT_REPLICAS == ""
    assert envs.VLLM_EXL3_EXPERT_REPLICAS_SHA256 == ""
    assert rp.plan_layer(object(), object(), "unused") is None


def test_digest_without_file_is_refused(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_EXPERT_REPLICAS", raising=False)
    monkeypatch.setenv("VLLM_EXL3_EXPERT_REPLICAS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="requires a plan file"):
        rp.plan_layer(object(), object(), "unused")


@pytest.mark.parametrize("bad", [
    {}, {"version": 2, "ep_size": 4, "layers": {"x": [1]}},
    {"version": 1, "ep_size": 3, "layers": {"x": [1]}},
    {"version": 1, "ep_size": True, "layers": {"x": [1]}},
    {"version": 1, "ep_size": 4, "layers": {}},
    {"version": 1, "ep_size": 4, "layers": {"x": []}},
    {"version": 1, "ep_size": 4, "layers": {"x": [1, 1]}},
    {"version": 1, "ep_size": 4, "layers": {"x": [-1]}},
    {"version": 1, "ep_size": 4, "layers": {"x": [True]}},
    {"version": 1, "ep_size": 4, "layers": {"x": list(range(rp.MAX_HOT + 1))}},
    {"version": 1, "ep_size": 4, "layers": {"": [1]}},
])
def test_invalid_plan(bad):
    with pytest.raises(ValueError):
        rp.validate_plan(bad)


def test_digest_mismatch_and_duplicate_keys(monkeypatch, tmp_path):
    path, digest = _enable(monkeypatch, tmp_path, {"version": 1, "ep_size": 4, "layers": {"x": [1]}})
    with pytest.raises(ValueError, match="SHA256"):
        rp.load_plan(str(path), "f" * 64)
    path.write_text('{"version": 1, "ep_size": 4, "layers": {"x": [1]}, "layers": {"y": [2]}}')
    dup = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        rp.load_plan(str(path), dup)


def test_placement_and_replicas_are_refused_together(monkeypatch, tmp_path, synthetic):
    _enable(monkeypatch, tmp_path, {"version": 1, "ep_size": 4, "layers": {PREFIX: [5]}})
    monkeypatch.setenv("VLLM_EXL3_EXPERT_PLACEMENT", str(tmp_path / "placement.json"))
    with pytest.raises(ValueError, match="cannot be combined"):
        rp.plan_layer(_Layer(4, 0), _config(synthetic), PREFIX)


def test_unknown_layer_or_expert_is_refused(monkeypatch, tmp_path, synthetic):
    _enable(monkeypatch, tmp_path, {"version": 1, "ep_size": 4, "layers": {"misspelled": [1]}})
    with pytest.raises(ValueError, match="no checkpoint experts"):
        rp.plan_layer(_Layer(4, 0), _config(synthetic), PREFIX)
    _enable(monkeypatch, tmp_path, {"version": 1, "ep_size": 4, "layers": {PREFIX: [EXPERTS]}})
    with pytest.raises(ValueError, match="no checkpoint experts"):
        rp.plan_layer(_Layer(4, 0), _config(synthetic), PREFIX)


@pytest.mark.parametrize("option,value", [
    ("ep_size", 2), ("tp_size", 2), ("dp_size", 2),
    ("enable_eplb", True), ("use_all2all_kernels", True),
])
def test_unsupported_dispatch_is_refused(monkeypatch, tmp_path, synthetic, option, value):
    _enable(monkeypatch, tmp_path, {"version": 1, "ep_size": 4, "layers": {PREFIX: [5]}})
    layer = _Layer(4, 0)
    setattr(layer.moe_config.moe_parallel_config, option, value)
    with pytest.raises(ValueError, match="require whole-expert EP"):
        rp.plan_layer(layer, _config(synthetic), PREFIX)


def test_unlisted_layer_is_unchanged(monkeypatch, tmp_path, synthetic):
    _enable(monkeypatch, tmp_path, {"version": 1, "ep_size": 4, "layers": {PREFIX: [5]}})
    assert rp.plan_layer(_Layer(4, 0), _config(synthetic), "another.valid.layer") is None


def test_non_linear_placement_is_refused(monkeypatch, tmp_path, synthetic):
    _enable(monkeypatch, tmp_path, {"version": 1, "ep_size": 4, "layers": {PREFIX: [5]}})
    layer = _Layer(4, 0)
    layer.expert_map_manager.expert_map = layer.expert_map.flip(0)
    with pytest.raises(ValueError, match="linear EP placement"):
        rp.plan_layer(layer, _config(synthetic), PREFIX)


# --------------------------------------------------------------- the owner rule


@pytest.mark.parametrize("ep,experts,topk,hot", [(4, 384, 6, 16), (4, 128, 3, 8), (2, 64, 4, 4)])
@pytest.mark.parametrize("rows", [1, 6, 16, 32, 64])
def test_every_assignment_has_exactly_one_rank(ep, experts, topk, hot, rows):
    for seed in range(20):
        ids = _routes(rows, topk, experts, seed, skew=0.3 * (seed % 3))
        per_rank = [_replicas(experts, ep, hot, r, seed) for r in range(ep)]
        maps = [rp.expert_map_reference(ids, rep) for rep in per_rank]
        owned = sum((m[ids.long()] >= 0).int() for m in maps)
        assert torch.equal(owned, torch.ones_like(ids, dtype=torch.int32))
        # Every rank computes the same owners from the same ids.
        owners = [rp.assign_owners(ids, rep) for rep in per_rank]
        assert all(o == owners[0] for o in owners)
        # A slot a rank uses holds that expert (home slot or its copy).
        for rep, m in zip(per_rank, maps):
            at_slot = {s: e for e, s in enumerate(rep.local) if s >= 0}
            for e in ids.flatten().tolist():
                if m[e] >= 0:
                    assert at_slot[int(m[e])] == e


def test_never_busier_than_ordinary_placement():
    for seed in range(200):
        ids = _routes(6, 6, 384, seed, skew=0.5)
        per_rank = [_replicas(384, 4, 32, r, seed % 7) for r in range(4)]
        maps = [rp.expert_map_reference(ids, rep) for rep in per_rank]
        static = [torch.tensor(rp.layer_replicas((), rep.home, rep.rank, 4).local) for rep in per_rank]
        assert _busiest(ids, maps) <= _busiest(ids, static)


def test_balances_an_overloaded_rank():
    home = rp.home_ranks(4, 384)
    # Rank 0 has five used experts, the others one each; experts 1-3 are hot.
    ids = torch.tensor([[0, 1, 2, 3, 4, 100], [200, 300, 0, 1, 2, 3]], dtype=torch.int32)
    per_rank = [rp.layer_replicas((1, 2, 3), home, r, 4) for r in range(4)]
    maps = [rp.expert_map_reference(ids, rep) for rep in per_rank]
    static = [torch.tensor(rp.layer_replicas((), home, r, 4).local) for r in range(4)]
    assert _busiest(ids, static) == 5 and _busiest(ids, maps) == 2
    assert rp.assign_owners(ids, per_rank[0]) == [1, 2, 3]


def test_balanced_call_keeps_ordinary_placement():
    home = rp.home_ranks(4, 384)
    ids = torch.tensor([[0, 96, 192, 288, 1, 97]], dtype=torch.int32)  # 2/2/1/1
    for r in range(4):
        rep = rp.layer_replicas((0, 96, 192, 288), home, r, 4)
        static = torch.tensor(rp.layer_replicas((), home, r, 4).local, dtype=torch.int32)
        static_used = static[ids.long()]
        assert torch.equal(rp.expert_map_reference(ids, rep)[ids.long()], static_used)


def test_unused_replicated_expert_is_ignored():
    rep = _replicas(384, 4, 16, rank=1, seed=3)
    ids = torch.tensor([[e for e in range(384) if e not in rep.hot][:6]], dtype=torch.int32)
    owners = rp.assign_owners(ids, rep)
    assert owners == [rep.home[e] for e in rep.hot]


@pytest.mark.parametrize("rows", [1, 6, 32])
def test_rank_sums_equal_ordinary_placement_for_exact_experts(rows):
    """Each expert computes w * x * (e + 1) on integer data, so every sum is
    exact: the all-reduced partials equal ordinary placement's bit for bit."""
    experts, ep, topk, hidden = 384, 4, 6, 8
    for seed in range(10):
        ids = _routes(rows, topk, experts, seed, skew=0.5).long()
        g = torch.Generator().manual_seed(seed)
        x = torch.randint(-8, 8, (rows, hidden), generator=g).double()
        w = torch.randint(1, 4, (rows, topk), generator=g).double()

        def partial(mapping):
            out = torch.zeros(rows, hidden, dtype=torch.float64)
            for t in range(rows):
                for k in range(topk):
                    e = int(ids[t, k])
                    if mapping[e] >= 0:
                        out[t] += w[t, k] * x[t] * (e + 1)
            return out

        per_rank = [_replicas(experts, ep, 16, r, seed) for r in range(ep)]
        replicated = sum(partial(rp.expert_map_reference(ids, rep)) for rep in per_rank)
        ordinary = sum(
            partial(torch.tensor(rp.layer_replicas((), rep.home, rep.rank, ep).local))
            for rep in per_rank
        )
        reference = partial(torch.zeros(experts, dtype=torch.int32))
        assert torch.equal(replicated, reference) and torch.equal(ordinary, reference)


# ----------------------------------------------------- loader and memory ledger


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("contiguous", [False, True])
def test_real_loader_holds_each_copy_and_counts_its_bytes(
    monkeypatch, tmp_path, synthetic, size, contiguous
):
    config = _config(synthetic)
    hot = [5, 0, 7]
    _enable(monkeypatch, tmp_path, {"version": 1, "ep_size": size, "layers": {PREFIX: hot}})
    monkeypatch.setenv("VLLM_EXL3_EXPERTS_CONTIGUOUS", str(int(contiguous)))
    home = rp.home_ranks(size, EXPERTS)
    for rank in range(size):
        layer = _Layer(size, rank)
        method = object.__new__(quant.Exl3MoEMethod)
        method.config, method.prefix = config, PREFIX
        method.create_weights(layer, EXPERTS // size, HIDDEN, INTERMEDIATE, torch.bfloat16)
        copies = [e for e in hot if home[e] != rank]
        homes = [e for e in range(EXPERTS) if home[e] == rank]
        assert method.home_experts == EXPERTS // size
        assert method.local_experts == EXPERTS // size + len(copies)
        assert method.global_expert_ids == homes + copies
        for local, logical in enumerate(method.global_expert_ids):
            for weight in layer.exl3_experts[local]:
                assert f".{logical}." in weight.matrix.prefix
        ledger = Exl3LoadLedger(config.manifest)
        with ExitStack() as stack:
            files = {p: stack.enter_context(safe_open(str(p), framework="pt", device="cpu"))
                     for p in {h.path for h in config.manifest.tensors.values()}}
            for key in method.nonlocal_keys:
                ledger.mark(key, "unused", "another rank")
            for expert in layer.exl3_experts:
                for weight in expert:
                    weight.load(files, ledger)
        ledger.finish()
        _check_bytes(layer, config.manifest)
        expected = sum(
            np.array(_file_array(config.manifest, f"{name}.{kind}")).nbytes
            for e in copies
            for role in ("gate_proj", "up_proj", "down_proj")
            for name in [config.layout.expert_matrix(config, PREFIX, e, role)]
            for kind in ("trellis", "suh", "svh")
        )
        assert method.replica_bytes == expected
        assert (method.replica_bytes > 0) == bool(copies)


def test_replicas_off_leaves_the_layer_unchanged(monkeypatch, synthetic):
    monkeypatch.delenv("VLLM_EXL3_EXPERT_REPLICAS", raising=False)
    monkeypatch.delenv("VLLM_EXL3_EXPERT_REPLICAS_SHA256", raising=False)
    config = _config(synthetic)
    layer = _Layer(4, 1)
    method = object.__new__(quant.Exl3MoEMethod)
    method.config, method.prefix = config, PREFIX
    method.create_weights(layer, EXPERTS // 4, HIDDEN, INTERMEDIATE, torch.bfloat16)
    assert method.replicas is None and method.replica_bytes == 0
    assert method.local_experts == method.home_experts == EXPERTS // 4
    assert method.global_expert_ids == [2, 3]


# ------------------------------------------------------------- experts dispatch


def _experts(replica_map, max_rows=128):
    experts = object.__new__(quant.Exl3Experts)
    experts.intermediate_size = INTERMEDIATE
    experts.expert_map = torch.zeros(EXPERTS, dtype=torch.int32)
    experts.workspace_elements = 8
    experts.max_rows = max_rows
    experts.moe_out_rows = max_rows
    experts.moe_v2_rows = 64
    experts.moe_v2_bits_x2 = 6
    experts.prefill_decode_len = 0
    experts.prefill_v2_bits_x2 = 0
    experts.prefill_v2_decode_len = 0
    experts.swiglu_limit = 0.0
    experts.replica_map = replica_map
    return experts


@pytest.mark.parametrize("rows,dynamic", [(1, True), (6, True), (64, True), (65, False)])
def test_calls_up_to_max_rows_take_the_per_call_map(monkeypatch, rows, dynamic):
    from vllm.sm80.kernels.exl3 import ops

    seen = []

    def record(x, ids, weights, w1, w2, expert_map, *args, **kwargs):
        seen.append(expert_map)

    monkeypatch.setattr(ops, "moe_out_v2", record)
    monkeypatch.setattr(ops, "moe_out", record)
    per_call = torch.full((EXPERTS,), 7, dtype=torch.int32)
    calls = []

    def replica_map(ids):
        calls.append(ids)
        return per_call

    experts = _experts(replica_map)
    static = torch.arange(EXPERTS, dtype=torch.int32)
    ids = torch.zeros(rows, 6, dtype=torch.int32)
    x = torch.zeros(rows, HIDDEN, dtype=torch.bfloat16)
    experts.apply(
        torch.empty_like(x), x, None, None, torch.ones(rows, 6), ids, None, EXPERTS,
        static, None, None, None, torch.empty(16, dtype=torch.bfloat16), None, False,
    )
    assert len(seen) == 1
    assert (seen[0] is per_call) == dynamic and (seen[0] is static) == (not dynamic)
    assert (calls == [ids]) == dynamic


# ------------------------------------------------- the map kernel (interpreter)


def kernel_cases() -> list:
    """Run in the Triton interpreter: kernel map vs reference map per case."""
    assert os.environ.get("TRITON_INTERPRET") == "1"
    out = []
    cases = [(4, 384, 6, 16), (4, 384, 6, 1), (4, 128, 3, 8), (2, 64, 4, 4), (4, 384, 8, 64)]
    for ep, experts, topk, hot in cases:
        for rows in (1, 6, 33, 64):
            for seed in range(3):
                ids = _routes(rows, topk, experts, seed, skew=0.4 * seed)
                for dtype in (torch.int32, torch.int64):
                    for rank in range(ep):
                        rep = _replicas(experts, ep, hot, rank, seed)
                        kernel = rp.ReplicaMap(rep, "cpu", torch.int32)
                        got = kernel(ids.to(dtype).contiguous()).clone()
                        want = rp.expert_map_reference(ids, rep)
                        out.append([ep, experts, topk, hot, rows, seed, str(dtype), rank,
                                    bool(torch.equal(got, want))])
    # Repeated calls through the same buffers: no state carries over.
    rep = _replicas(384, 4, 16, 2, 9)
    kernel = rp.ReplicaMap(rep, "cpu", torch.int64)
    for seed in range(5):
        ids = _routes(6, 6, 384, 100 + seed, skew=1.0)
        got = kernel(ids).clone()
        out.append(["repeat", seed, bool(torch.equal(got, rp.expert_map_reference(ids, rep).long()))])
    return out


@pytest.fixture(scope="module")
def kernels():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    code = ("import json; from tests.kernels.quantization.test_exl3_replicas import "
            "kernel_cases as f; print('RESULT ' + json.dumps(f()))")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT "):])


def test_kernel_matches_the_reference(kernels):
    bad = [case for case in kernels if not case[-1]]
    assert len(kernels) > 100 and not bad, bad[:8]


def test_too_many_assignments_are_refused():
    rep = _replicas(384, 4, 16, 0, 0)
    kernel = rp.ReplicaMap(rep, "cpu", torch.int32)
    with pytest.raises(ValueError, match="too large"):
        kernel(torch.zeros(rp.BLOCK_ASSIGNMENTS + 1, dtype=torch.int32))
