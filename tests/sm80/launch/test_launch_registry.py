# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The registry comes from the target tree, never from the launcher's own."""

from __future__ import annotations

import os

import pytest

from errors import Refusal
from helpers import FORK_VARS, World, make_tree
from registry import read_registry


def test_each_tree_reads_its_own_tables(tmp_path):
    a = make_tree(str(tmp_path / "a"), fork=dict(FORK_VARS, VLLM_NEW_SWITCH="0"), retired={})
    b = make_tree(str(tmp_path / "b"), fork=FORK_VARS,
                  retired={"VLLM_NEW_SWITCH": "renamed to VLLM_THIN_GEMM"})
    ra, rb = read_registry(a, "sm80"), read_registry(b, "sm80")
    assert ra.sm80_envs_sha256 != rb.sm80_envs_sha256
    assert ra.check("VLLM_NEW_SWITCH") is None
    assert "renamed to VLLM_THIN_GEMM" in rb.check("VLLM_NEW_SWITCH")
    assert ra.defaults["VLLM_THIN_GEMM"] == "0"


def test_same_launcher_accepts_on_one_tree_refuses_on_other(tmp_path):
    w1 = World(tmp_path / "1", fork=dict(FORK_VARS, VLLM_NEW_SWITCH="0"), retired={})
    w2 = World(tmp_path / "2", fork=FORK_VARS, retired={"VLLM_NEW_SWITCH": "renamed to VLLM_THIN_GEMM"})
    assert w1.resolve("--set", "env.VLLM_NEW_SWITCH=1")["child_env"]["VLLM_NEW_SWITCH"]["value"] == "1"
    assert "renamed" in w2.refused("--set", "env.VLLM_NEW_SWITCH=1")


def test_tree_without_envs_refused(tmp_path):
    os.makedirs(tmp_path / "t" / "vllm")
    with pytest.raises(Refusal, match="no vllm/envs.py"):
        read_registry(str(tmp_path / "t"), "sm80")
