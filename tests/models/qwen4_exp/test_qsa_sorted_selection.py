# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen QSA selected-index order: the sorted selection, the
VLLM_QWEN_QSA_TOPK_SORTED switch and the native top-k contract."""

import importlib

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as qsa_indexer_ops

from .test_qsa_reference import _expand_qsa_indices_reference, requires_qsa_kernels


requires_cuda_device = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a visible CUDA device"
)


@requires_qsa_kernels
@pytest.mark.parametrize("block_topk", [3, 512])
def test_qsa_sorted_selection_preserves_packed_expansion(block_topk: int) -> None:
    """Sorted padded/strided selections keep the causal tail and count ABI."""
    storage = torch.full(
        (5, 2 * block_topk + 3), -99, device="cuda", dtype=torch.int32
    )
    blocks = storage[1:4, 1 : 2 * block_topk + 1 : 2]
    blocks.fill_(-1)
    blocks[1, 0], blocks[1, -1] = 1, 0
    blocks[2] = torch.arange(block_topk - 1, -1, -1, device="cuda") + 3
    expected_blocks = torch.full_like(blocks, -1)
    expected_blocks[1, :2] = torch.arange(2, device="cuda")
    expected_blocks[2] = torch.arange(block_topk, device="cuda") + 3
    expected_storage = storage.clone()
    expected_storage[1:4, 1 : 2 * block_topk + 1 : 2] = expected_blocks

    qsa_indexer_ops._sort_qsa_selected_indices_(blocks)

    torch.testing.assert_close(storage, expected_storage)
    positions = torch.tensor([2, 10, 4 * (block_topk + 5) + 2], device="cuda")
    visible_blocks = ((positions + 1) // 4).to(torch.int32)
    token_topk = 4 * block_topk
    packed_width = token_topk + 4
    packed_storage = torch.full(
        (5, 2 * packed_width + 3), -99, device="cuda", dtype=torch.int32
    )
    packed = packed_storage[1:4, 1 : 2 * packed_width + 1 : 2]
    expected_packed_storage = packed_storage.clone()
    expected = _expand_qsa_indices_reference(
        expected_blocks, positions, positions + 1, 4, token_topk
    )
    counts = torch.tensor([3, 11, token_topk + 3], device="cuda")
    expected_packed_storage[1:4, 1 : 2 * packed_width + 1 : 2] = torch.cat(
        (expected, counts[:, None]), dim=1
    )

    qsa_indexer_ops.expand_qsa_block_indices(
        blocks, positions, visible_blocks, 4, token_topk, packed
    )

    torch.testing.assert_close(packed_storage, expected_packed_storage)


@pytest.mark.parametrize("value,expected", [(None, True), ("1", True), ("0", False)])
def test_qsa_sort_switch_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool
) -> None:
    if value is None:
        monkeypatch.delenv("VLLM_QWEN_QSA_TOPK_SORTED", raising=False)
    else:
        monkeypatch.setenv("VLLM_QWEN_QSA_TOPK_SORTED", value)
    try:
        assert importlib.reload(qsa_indexer_ops)._QSA_TOPK_SORTED is expected
    finally:
        monkeypatch.undo()
        importlib.reload(qsa_indexer_ops)


def _native_topk_case(width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows with 0, 1, K-1, K, K+1 and all visible blocks, plus a tied row."""
    visible = sorted(v for v in {0, 1, 511, 512, 513, width} if v <= width)
    generator = torch.Generator(device="cuda").manual_seed(width)
    logits = torch.randn(len(visible) + 1, width, device="cuda", generator=generator)
    visible.append(width)
    logits[-1] = 0.0
    lengths = torch.tensor(visible, dtype=torch.int32, device="cuda")
    # Blocks past the visible count must never be read.
    columns = torch.arange(width, device="cuda")
    logits.masked_fill_(columns >= lengths[:, None], 1e30)
    return logits, lengths


def _topk_workspace() -> torch.Tensor:
    return torch.empty(
        qsa_indexer_ops._TOPK_WORKSPACE_BYTES, dtype=torch.uint8, device="cuda"
    )


@requires_qsa_kernels
@requires_cuda_device
@pytest.mark.parametrize("width", [600, 4096, 65536])
def test_qsa_native_topk_fills_exactly_min_visible_k_leading_slots(
    monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    """Sorting keeps membership only if the native top-k writes exactly
    min(visible, K) distinct valid ids into the leading slots, -1 after."""
    monkeypatch.setattr(qsa_indexer_ops, "_QSA_TOPK_SORTED", False)
    logits, lengths = _native_topk_case(width)
    block_topk = 512
    out = torch.full(
        (logits.shape[0], block_topk), -7, dtype=torch.int32, device="cuda"
    )
    qsa_indexer_ops._topk(logits, lengths, 4 * block_topk, 4, out, _topk_workspace())
    for row, visible in enumerate(lengths.tolist()):
        count = min(visible, block_topk)
        assert (out[row, :count] >= 0).all() and (out[row, count:] == -1).all()
        chosen = out[row, :count].long()
        assert chosen.unique().numel() == count
        assert (chosen < visible).all()
        if row < logits.shape[0] - 1 and count:
            expected = torch.topk(logits[row, :visible], count).indices
            assert set(chosen.tolist()) == set(expected.tolist())


@requires_qsa_kernels
@requires_cuda_device
def test_qsa_sort_kill_switch_keeps_native_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The native top-k order is not repeatable between calls (the reason for
    # the sort), so compare membership and whether the sort ran.
    logits, lengths = _native_topk_case(4096)
    shape = (logits.shape[0], 512)
    native = torch.full(shape, -7, dtype=torch.int32, device="cuda")
    workspace = _topk_workspace()
    torch.ops._C.persistent_topk(
        logits, lengths, native, workspace, 512, logits.shape[1]
    )
    sort = qsa_indexer_ops._sort_qsa_selected_indices_
    calls = []

    def counted_sort(block_indices: torch.Tensor) -> None:
        calls.append(block_indices.data_ptr())
        sort(block_indices)

    monkeypatch.setattr(qsa_indexer_ops, "_sort_qsa_selected_indices_", counted_sort)
    outputs = {}
    for flag in (False, True):
        monkeypatch.setattr(qsa_indexer_ops, "_QSA_TOPK_SORTED", flag)
        calls.clear()
        outputs[flag] = torch.full(shape, -7, dtype=torch.int32, device="cuda")
        qsa_indexer_ops._topk(logits, lengths, 2048, 4, outputs[flag], workspace)
        assert calls == ([outputs[flag].data_ptr()] if flag else [])

    def ordered(blocks: torch.Tensor) -> torch.Tensor:
        # Ascending with -1 last: the order of the selected ids as uint32.
        key = torch.where(blocks < 0, 2**32 - 1, blocks.long())
        return blocks.gather(1, key.argsort(dim=1))

    assert torch.equal(outputs[True], ordered(outputs[True]))
    # Membership matches a separate native call except on the tied last row,
    # where any 512 of the equal scores are a valid selection.
    for output in outputs.values():
        assert torch.equal(ordered(output)[:-1], ordered(native)[:-1])
