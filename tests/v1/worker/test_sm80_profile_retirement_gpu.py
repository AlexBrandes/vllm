# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-device allocator/replay proof; never part of the CPU check batch."""

import contextlib
import gc
from types import SimpleNamespace

import pytest
import torch

from tests.utils import create_new_process_for_each_test
from tests.v1.worker.test_gpu_model_runner_v2_cudagraph_profiling import (
    _make_profiling_runner,
)
from vllm.config import CUDAGraphMode
from vllm.sm80 import capture
from vllm.v1.worker.gpu import cudagraph_utils as cgu


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(not cgu.current_platform.is_cuda(), reason="requires CUDA")
def test_profile_registry_releases_pool_then_recaptures(monkeypatch):
    runner = _make_profiling_runner(CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.compilation_config.static_forward_context = {}
    runner.model_state = SimpleNamespace(supports_mm_inputs=False)
    runner.cache_config = SimpleNamespace(num_gpu_blocks=1)
    runner.kv_caches = []
    runner.attn_groups = []
    runner.kv_cache_config = SimpleNamespace()
    runner.lora_config = None
    runner.maybe_remove_all_loras = lambda _: None
    monkeypatch.setattr(capture, "_pending", [])
    monkeypatch.setattr(cgu, "set_current_vllm_config", lambda _: contextlib.nullcontext())
    elements = (16 << 20) // 4
    source = torch.ones(elements, device="cuda")
    torch.accelerator.synchronize()
    gc.collect()
    torch.accelerator.empty_cache()
    before = torch.accelerator.memory_reserved()
    memory = {}

    def initialize(r):
        r.speculator = SimpleNamespace(query_cudagraph_manager=None, _cg_managers={})
        for width in (3, 5):
            manager = cgu.CudaGraphManager.__new__(cgu.CudaGraphManager)
            manager.pool = cgu.current_platform.get_global_graph_pool()
            manager.graphs = capture._BindingGraphs()
            r.speculator._cg_managers[width] = manager
        r.speculator.query_cudagraph_manager = r.speculator._cg_managers[5]

    def capture_one(manager, key):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=manager.pool):
            record = capture.Record("attention", "lifecycle", "layer", 1, {}, in_graph=True)
            record.put(snapshot=source * 2)
            record.done()
        manager.graphs[key] = graph

    def capture_model(*, profile_only=False):
        assert profile_only
        runner.cudagraph_manager.graphs = capture._BindingGraphs()
        capture_one(runner.cudagraph_manager, "target")
        for width, manager in runner.speculator._cg_managers.items():
            capture_one(manager, width)
        torch.accelerator.synchronize()
        assert not capture._pending
        memory["captured"] = torch.accelerator.memory_reserved()
        return memory["captured"] - before

    monkeypatch.setattr(cgu, "_init_minimal_kv_cache_for_profiling", initialize)
    runner.capture_model = capture_model
    cgu.profile_cudagraph_memory(runner)
    after = torch.accelerator.memory_reserved()
    assert memory["captured"] > before
    assert runner.speculator.query_cudagraph_manager is None
    assert runner.speculator._cg_managers == {}
    assert after == before

    # New graph handles bind the restored persistent pool. Their real replay
    # must refresh graph-owned snapshots from new input rather than stale state.
    initialize(runner)
    for width, manager in runner.speculator._cg_managers.items():
        capture_one(manager, width)
    source.fill_(7)
    for width, manager in runner.speculator._cg_managers.items():
        graph = manager.graphs[width]
        graph.replay()
        torch.accelerator.synchronize()
        observed = graph._sm80_capture_records[0].materialize()["snapshot"]
        torch.testing.assert_close(observed, torch.full_like(observed, 14))
    assert not capture._pending
