# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid snapshots must follow consumed rows, not paged-table capacity."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.models.deepseek_v41.nvidia import hybrid_capture as hybrid
from vllm.models.deepseek_v41.nvidia.ops import sm80_attn_core as core
from vllm.sm80.capture import Record


class NoCacheSnapshot(TorchDispatchMode):
    """The capture-time path may copy metadata, never a reader's KV payload."""

    def __init__(self, *caches):
        super().__init__()
        self.pointers = {x.untyped_storage().data_ptr() for x in caches if x is not None}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.index.Tensor, torch.ops.aten.clone.default):
            assert args[0].untyped_storage().data_ptr() not in self.pointers
        return func(*args, **(kwargs or {}))


def index_args(device="cpu"):
    table = torch.full((2, 65536), 999999, dtype=torch.int32, device=device)
    table[:, :2] = torch.tensor([[2, 4], [2, 5]], device=device)
    return dict(
        q=torch.arange(32, device=device).reshape(4, 2, 4).to(torch.bfloat16) / 8,
        w=torch.ones(4, 2, device=device),
        n_vis=torch.tensor([5, 3, 7, 0], device=device),
        req=torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device),
        idxk=torch.arange(96, device=device).reshape(6, 4, 4).to(torch.bfloat16) / 32,
        block_table=table, n_cols=8, heads_per_rank=2,
    )


def attention_args(kind="compressed", device="cpu"):
    table = torch.full((2, 65536), 999999, dtype=torch.int32, device=device)
    table[:, :2] = torch.tensor([[2, 4], [2, 5]], device=device)
    cache = torch.arange(96, device=device).reshape(6, 4, 4).to(torch.bfloat16) / 32
    local = torch.tensor([[0, 5, -1, 0], [1, 6, 4, -1]], dtype=torch.int32, device=device)
    if kind in ("absent", "none_selection"):
        local = None
    elif kind == "empty_selection":
        local = local[:, :0]
    elif kind == "invalid_selection":
        local.fill_(-1)
    return dict(
        q=torch.arange(16, device=device).reshape(2, 2, 4).to(torch.bfloat16) / 8,
        swa_cache=cache,
        swa_indices=torch.tensor([[8, 9, -1, 8], [8, 21, 9, -1]],
                                 dtype=torch.int32, device=device),
        comp_cache=None if kind == "absent" else cache.clone() + 1,
        comp_block_table=None if kind == "absent" else table,
        comp_local=local,
        req=torch.tensor([0, 1], dtype=torch.int32, device=device),
        sink=torch.zeros(2, device=device), scale=0.5,
    )


def reader_record(op, args):
    record = Record("hybrid", "hybrid_linear_consumer", "layers.8.attn", args["q"].shape[0], {})
    caches = [args[name] for name in ("idxk", "swa_cache", "comp_cache") if name in args]
    with NoCacheSnapshot(*caches):
        hybrid._defer_reader(record, op, args)
    return record


def test_hybrid_index_snapshot_uses_visible_unique_blocks_and_owns_metadata():
    args = index_args()
    expected = core.index_scores_torch(**args)
    record = reader_record("index_scores", args)
    assert "idxk" not in record.tensors and "block_table" not in record.tensors
    # These caller-owned buffers can be reused after the point was recorded.
    args["block_table"].fill_(999999)
    args["req"].zero_()
    args["n_vis"].zero_()
    args["q"].zero_()
    saved = record.materialize()
    assert saved["idxk"].shape == (4, 4, 4)  # null + physical blocks 2,4,5
    assert saved["block_table"][0, 0] == saved["block_table"][1, 0]
    assert bool((saved["block_table"][:, 2:] == 0).all())
    assert torch.equal(core.index_scores_torch(**saved), expected)


@pytest.mark.parametrize("visible", [0, 1, 4, 5, 8])
def test_hybrid_index_snapshot_visibility_boundaries(visible):
    args = index_args()
    args["n_vis"].fill_(visible)
    expected = core.index_scores_torch(**args)
    saved = reader_record("index_scores", args).materialize()
    want_blocks = 1 if visible == 0 else (2 if visible <= 4 else 4)
    assert saved["idxk"].shape[0] == want_blocks
    assert torch.equal(core.index_scores_torch(**saved), expected)


@pytest.mark.parametrize("kind", ["compressed", "absent", "none_selection", "empty_selection", "invalid_selection"])
def test_hybrid_sparse_snapshot_preserves_consumed_rows_and_negative_slots(kind):
    args = attention_args(kind)
    expected = core.sparse_attention(**args)
    record = reader_record("sparse_attention", args)
    assert not {"swa_cache", "comp_cache", "swa_indices", "comp_block_table"} & record.tensors.keys()
    args["swa_indices"].fill_(-1)
    args["req"].zero_()
    if args["comp_block_table"] is not None:
        args["comp_block_table"].fill_(999999)
    if args["comp_local"] is not None:
        args["comp_local"].fill_(-1)
    saved = record.materialize()
    assert saved["swa_cache"].shape == (3, 4, 4)  # null + physical blocks 2,5
    assert saved["swa_indices"][0, 0] == saved["swa_indices"][0, 3] == saved["swa_indices"][1, 0]
    assert saved["swa_indices"][0, 2] == saved["swa_indices"][1, 3] == -1
    if kind == "absent":
        assert saved["comp_cache"] is None and saved["comp_block_table"] is None
    else:
        assert saved["comp_cache"].shape[0] == (4 if kind == "compressed" else 1)
        assert bool((saved["comp_block_table"][:, 2:] == 0).all())
    assert torch.equal(core.sparse_attention(**saved), expected)


def intercept_records(monkeypatch):
    records = []
    monkeypatch.setattr(hybrid, "_outputs", {})

    def begin(family, point, prefix, rows, point_phase=None, **meta):
        return Record(family, point, prefix, rows, meta, point_phase=point_phase)

    monkeypatch.setattr(hybrid.capture, "begin", begin)
    monkeypatch.setattr(Record, "done", lambda record: records.append(record))
    return records


@pytest.mark.parametrize("op", ["store_rows", "compress_rows"])
def test_hybrid_mutating_snapshot_still_keeps_pre_post_and_guard_blocks(monkeypatch, op):
    records = intercept_records(monkeypatch)
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    produced = hybrid._record("linear", {"x": x}, lambda: x + 1,
                              linear={"prefix": "layers.8.attn.wkv"})
    cache = torch.arange(192, dtype=torch.float32).reshape(6, 4, 8)
    before = cache.clone()
    slots = torch.tensor([8, 9, -1])
    name = "ring" if op == "compress_rows" else "cache"
    slot_name = "ring_slots" if op == "compress_rows" else "slots"
    input_name = "kv_score" if op == "compress_rows" else "rows"

    def run():
        valid = slots >= 0
        cache[slots[valid] // 4, slots[valid] % 4] = produced[valid]
        return produced.clone() if op == "compress_rows" else None

    hybrid._record(op, {name: cache, slot_name: slots, input_name: produced}, run)
    after = cache.clone()
    cache.add_(10000)
    record = records[-1]
    saved = record.materialize()
    ids = saved["snapshot_blocks"]
    assert set(ids.tolist()) == {0, 1, 2, 3}
    assert torch.equal(saved[name], before[ids])
    assert torch.equal(saved["out.ring" if op == "compress_rows" else "out"], after[ids])
    assert record.meta["parents"][input_name]["call_id"] == records[0].meta["call_id"]


@pytest.mark.parametrize("op", ["index_scores", "sparse_attention"])
def test_hybrid_reader_record_preserves_real_parent_and_successor_links(monkeypatch, op):
    records = intercept_records(monkeypatch)
    args = index_args() if op == "index_scores" else attention_args()
    q = args["q"]
    args["q"] = hybrid._record("linear", {"x": q}, lambda: q.clone(),
                               linear={"prefix": "layers.8.attn.wq_b"})
    run = core.index_scores if op == "index_scores" else core.sparse_attention
    result = hybrid._record(op, args, lambda: run(**args))
    link = hybrid._link(result)
    assert link["call_id"] == records[-1].meta["call_id"]
    assert link["output"] == "out"
    assert records[-1].meta["parents"]["q"]["call_id"] == records[0].meta["call_id"]
    saved = records[-1].materialize()
    observed = saved.pop("out")
    assert torch.equal(run(**saved), observed)
    assert torch.equal(observed, result)


def test_hybrid_unselected_call_does_not_snapshot_or_defer(monkeypatch):
    monkeypatch.setattr(hybrid, "_outputs", {})
    args = index_args()
    expected = core.index_scores_torch(**args)
    with NoCacheSnapshot(args["idxk"]):
        # No producer means no capture. The dummy run avoids confusing the
        # actual reader's legitimate cache reads with diagnostic snapshots.
        result = hybrid._record("index_scores", args, lambda: expected)
    assert result is expected
