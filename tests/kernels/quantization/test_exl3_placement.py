# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for fixed whole-expert ownership and actual packed loading."""

import hashlib
import json
from contextlib import ExitStack
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from tests.kernels.quantization.test_exl3_expert_buffers import (
    EXPERTS,
    HIDDEN,
    INTERMEDIATE,
    _check_bytes,
    _check_layout,
    _config,
    synthetic,  # noqa: F401
)
from vllm import envs
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_map,
)
from vllm.sm80.kernels.exl3 import placement, quant
from vllm.sm80.kernels.exl3.utils import Exl3LoadLedger

PREFIX = "model.layers.0.mlp.experts"


def _plan(size=4):
    return {
        "version": 1,
        "ep_size": size,
        "layers": {PREFIX: [list(reversed(range(r, EXPERTS, size))) for r in range(size)]},
    }


def _enable(monkeypatch, tmp_path, plan):
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(plan))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setenv("VLLM_EXL3_EXPERT_PLACEMENT", str(path))
    monkeypatch.setenv("VLLM_EXL3_EXPERT_PLACEMENT_SHA256", digest)
    return path, digest


class _Layer(torch.nn.Module):
    def __init__(self, size, rank):
        super().__init__()
        local, mapping, mask = determine_expert_map(size, rank, EXPERTS)
        self.global_num_experts = EXPERTS
        self.local_num_experts = local
        self.apply_router_weight_on_input = False
        self.register_buffer("expert_map", mapping)
        self.expert_map_manager = SimpleNamespace(
            expert_map=mapping, expert_mask=mask, routing_tables=None,
            num_fused_shared_experts=0,
        )
        self.moe_config = SimpleNamespace(moe_parallel_config=SimpleNamespace(
            ep_size=size, ep_rank=rank, tp_size=1, dp_size=1,
            enable_eplb=False, use_all2all_kernels=False,
        ))

    def _map_global_expert_id_to_local_expert_id(self, expert):
        return int(self.expert_map_manager.expert_map[expert])


def test_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_EXPERT_PLACEMENT", raising=False)
    monkeypatch.delenv("VLLM_EXL3_EXPERT_PLACEMENT_SHA256", raising=False)
    assert envs.VLLM_EXL3_EXPERT_PLACEMENT == ""
    assert envs.VLLM_EXL3_EXPERT_PLACEMENT_SHA256 == ""
    placement.apply_placement(object(), object(), "unused")


def test_digest_without_file_is_refused(monkeypatch):
    monkeypatch.delenv("VLLM_EXL3_EXPERT_PLACEMENT", raising=False)
    monkeypatch.setenv("VLLM_EXL3_EXPERT_PLACEMENT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="requires a placement file"):
        placement.apply_placement(object(), object(), "unused")


@pytest.mark.parametrize("bad", [
    {}, {"version": 2, "ep_size": 4, "layers": {}},
    {"version": 1, "ep_size": True, "layers": {}},
    {"version": 1, "ep_size": 4, "layers": {}},
    {"version": 1, "ep_size": 2, "layers": {"x": [[0, 1], [1, 2]]}},
    {"version": 1, "ep_size": 2, "layers": {"x": [[0, 1], [2, 4]]}},
    {"version": 1, "ep_size": 2, "layers": {"x": [[0, 1, 2], [3]]}},
    {"version": 1, "ep_size": 2, "layers": {"x": [[0, True], [2, 3]]}},
    {"version": 1, "ep_size": 2, "layers": {"x": [[0, 1]]}},
    {"version": 1, "ep_size": 2, "layers": {"x": [[0, 1], [2, 3]], "": [[0], [1]]}},
])
def test_invalid_plan(bad):
    with pytest.raises(ValueError):
        placement.validate_plan(bad)


def test_uneven_capacity():
    ranks = placement.validate_plan({
        "version": 1, "ep_size": 2, "layers": {"x": [[4, 2, 0], [3, 1]]},
    })["x"]
    assert placement.local_map(ranks, 0) == [2, -1, 1, -1, 0]
    assert placement.local_map(ranks, 1) == [-1, 1, -1, 0, -1]


def test_digest_and_duplicate_keys(monkeypatch, tmp_path):
    path, digest = _enable(monkeypatch, tmp_path, _plan())
    with pytest.raises(ValueError, match="SHA256"):
        placement.load_plan(str(path), "0" * 64)
    size, ranks = placement.load_plan(str(path), digest)
    assert size == 4 and ranks[PREFIX][0] == (4, 0)
    path.write_text('{"version":1,"version":1,"ep_size":4,"layers":{}}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Duplicate JSON"):
        placement.load_plan(str(path), digest)


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("contiguous", [False, True])
def test_real_loader_preserves_every_expert_byte(
    monkeypatch, tmp_path, synthetic, size, contiguous
):
    config = _config(synthetic)
    plan = _plan(size)
    _enable(monkeypatch, tmp_path, plan)
    monkeypatch.setenv("VLLM_EXL3_EXPERTS_CONTIGUOUS", str(int(contiguous)))
    seen = []
    for rank in range(size):
        layer = _Layer(size, rank)
        original_map = layer.expert_map
        method = object.__new__(quant.Exl3MoEMethod)
        method.config, method.prefix = config, PREFIX
        method.create_weights(layer, EXPERTS // size, HIDDEN, INTERMEDIATE, torch.bfloat16)
        assert layer.expert_map is original_map
        assert layer.expert_map_manager.expert_map is original_map
        assert method.global_expert_ids == plan["layers"][PREFIX][rank]
        for local, logical in enumerate(method.global_expert_ids):
            assert layer._map_global_expert_id_to_local_expert_id(logical) == local
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
        _check_layout(layer, contiguous)
        seen.extend(method.global_expert_ids)
    assert sorted(seen) == list(range(EXPERTS))


@pytest.mark.parametrize("option,value", [
    ("ep_size", 2), ("tp_size", 2), ("dp_size", 2),
    ("enable_eplb", True), ("use_all2all_kernels", True),
])
def test_unsupported_dispatch_is_refused(monkeypatch, tmp_path, synthetic, option, value):
    _enable(monkeypatch, tmp_path, _plan())
    layer = _Layer(4, 0)
    setattr(layer.moe_config.moe_parallel_config, option, value)
    with pytest.raises(ValueError, match="requires whole-expert"):
        placement.apply_placement(layer, _config(synthetic), PREFIX)


def test_unknown_checkpoint_prefix_is_refused(monkeypatch, tmp_path, synthetic):
    plan = _plan()
    plan["layers"]["misspelled"] = plan["layers"].pop(PREFIX)
    _enable(monkeypatch, tmp_path, plan)
    with pytest.raises(ValueError, match="matching checkpoint experts"):
        placement.apply_placement(_Layer(4, 0), _config(synthetic), PREFIX)


def test_unlisted_layer_keeps_existing_map(monkeypatch, tmp_path, synthetic):
    _enable(monkeypatch, tmp_path, _plan())
    layer = _Layer(4, 0)
    before = layer.expert_map.clone()
    placement.apply_placement(layer, _config(synthetic), "another.valid.layer")
    assert torch.equal(layer.expert_map, before)


def test_map_does_not_mutate_logical_routes():
    ranks = placement.validate_plan(_plan())[PREFIX]
    ids = torch.tensor([[7, 0, 3], [3, 4, 1], [2, 5, 6]])
    weights = torch.tensor([[0.2, 0.3, 0.5]]).expand(3, -1).clone()
    original_ids, original_weights = ids.clone(), weights.clone()
    coverage = torch.zeros_like(ids)
    for rank in range(4):
        mapping = torch.tensor(placement.local_map(ranks, rank))
        owned = mapping[ids] >= 0
        coverage += owned
        logical = torch.tensor(ranks[rank])[mapping[ids[owned]]]
        assert torch.equal(logical, ids[owned])
    assert torch.equal(coverage, torch.ones_like(ids))
    assert torch.equal(ids, original_ids) and torch.equal(weights, original_weights)
