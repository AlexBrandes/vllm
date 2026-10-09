# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the sorted sparse-indexer top-k (VLLM_GLM5_TOPK_SORTED).

The stock prefill and decode top-k kernels return the correct index set in an
arrival-dependent order, and the sparse MLA kernel accumulates in index order.
With VLLM_GLM5_TOPK_TIEFIX the tie-fix launch also sorts each row (ascending,
-1 fill last; the bytes are checked in test_topk_tiefix.py). The sort has no
other implementation, so SORTED without TIEFIX is rejected. CPU only; no CUDA,
no engine.
"""

import ast
import pathlib

import pytest
import torch

import vllm.envs as envs
from vllm.sm80.kernels.indexer.topk_tiefix_gate import (
    use_sorted_topk,
    use_tiefix_topk,
)


def _clear():
    for f in (use_sorted_topk, use_tiefix_topk):
        f.cache_clear()


def test_flag_defaults_off():
    assert envs.VLLM_GLM5_TOPK_SORTED is False
    _clear()
    try:
        assert use_sorted_topk() is False
    finally:
        _clear()


def test_flag_is_read_when_set(monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_TOPK_SORTED", "1")
    _clear()
    try:
        assert use_sorted_topk() is True
    finally:
        _clear()


def test_flag_is_part_of_the_compile_key():
    # It changes what the indexer computes, so it must not be ignored.
    assert "VLLM_GLM5_TOPK_SORTED" in envs.compile_factors()


def test_sorted_without_tiefix_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_GLM5_TOPK_SORTED", "1")
    monkeypatch.delenv("VLLM_GLM5_TOPK_TIEFIX", raising=False)
    _clear()
    try:
        with pytest.raises(ValueError, match="VLLM_GLM5_TOPK_TIEFIX=1"):
            use_tiefix_topk()
    finally:
        _clear()


def _function_source(module, name):
    """Source of a top-level function or class, read from its file: importing
    the GLM-5 model module pulls in the vendored FLA ops, which query the
    device at import (and would initialise CUDA on a machine with a GPU)."""
    import vllm

    parts = module.split(".")[1:]
    path = pathlib.Path(vllm.__file__).parent.joinpath(*parts).with_suffix(".py")
    text = path.read_text()
    node = next(n for n in ast.parse(text).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name == name)
    return ast.get_source_segment(text, node)


def test_both_paths_sort_inside_the_tie_fix():
    # Prefill runs the tie fix right after the top-k selection and before pool
    # expansion; decode runs it right after the shared decode top-k and before
    # the DCP merge (tiefix_decode_topk_). Both pass the sort flag into the
    # same launch.
    src = _function_source("vllm.models.glm5next.nvidia.sparse_indexer",
                           "sparse_attn_indexer_kpool")
    assert src.count("if use_tiefix_topk():") == 1
    assert src.count("sort=use_sorted_topk()") == 1
    pre = src.index("torch.ops._C.top_k_per_row_prefill")
    fix = src.index("sort=use_sorted_topk()")
    assert pre < fix < src.index("expand_pools_and_append_tail")
    assert src.count("tiefix_decode_topk_(") == 1
    dec_topk = src.index("get_indexer_topk(topk_backend)(")
    dec_fix = src.index("tiefix_decode_topk_(")
    assert dec_topk < dec_fix < src.index("_merge_dcp_topk_global(", dec_topk)
    dec = _function_source("vllm.sm80.kernels.indexer.topk_tiefix_gate",
                           "tiefix_decode_topk_")
    assert dec.count("if use_tiefix_topk():") == 1
    assert dec.count("sort=use_sorted_topk()") == 1


def test_no_cuda_was_initialised():
    assert not torch.cuda.is_initialized()
