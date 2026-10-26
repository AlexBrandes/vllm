# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The registry and the server's option names come from the target tree,
never from the launcher's own."""

from __future__ import annotations

import os

import pytest

from errors import Refusal
from helpers import (FORK_VARS, NEW_PARSER, OLD_PARSER, PARSER, ROOT, SERVE, World, make_tree,
                     serve_py, write)
from registry import read_arg_options, read_registry


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


def test_server_options_read_from_this_tree():
    """This engine tree's own serve command: frontend and engine options."""
    opts = read_arg_options(ROOT).options
    assert {"port", "tool-call-parser", "headless", "api-server-count", "max-model-len",
            "tensor-parallel-size"} <= opts


FRONTEND = PARSER.replace("def make_arg_parser(parser):\n",
                          'def make_arg_parser(parser):\n    parser.add_argument("--port", type=int)\n'
                          '    parser.add_argument("--tool-call-parser", "--tcp", type=str)\n')


@pytest.mark.parametrize("parser_file", [OLD_PARSER, NEW_PARSER])
def test_parser_module_found_from_the_serve_command(tmp_path, parser_file):
    """Both layouts: the parser module is the one the serve command imports
    make_arg_parser from; the engine options come from the module of the
    class it installs."""
    tree = make_tree(str(tmp_path / "t"), extra_files={SERVE: serve_py(parser_file), parser_file: FRONTEND})
    if parser_file != OLD_PARSER:
        os.remove(os.path.join(tree, OLD_PARSER))
    opts = read_arg_options(tree)
    assert {"port", "tool-call-parser", "tcp", "max-model-len", "max-num-seqs", "help"} <= opts.options
    assert opts.aliases == {"tcp": "tool-call-parser", "device-memory-utilization": "gpu-memory-utilization"}


@pytest.mark.parametrize("changes, want", [
    ({SERVE: None}, f"has no {SERVE}; the server's argument files cannot be found"),
    ({SERVE: "def subparser_init(p):\n    return p\n"}, "no 'from ... import make_arg_parser'"),
    ({SERVE: "from .common import make_arg_parser\n"}, "make_arg_parser is imported relatively"),
    ({OLD_PARSER: None}, "make_arg_parser comes from vllm.entrypoints.launchers.cli_args, "
                         "which this engine tree lacks"),
    ({OLD_PARSER: "from vllm.x import make_arg_parser\n"}, f"{OLD_PARSER} defines no function make_arg_parser"),
    ({OLD_PARSER: PARSER.replace("from vllm.engine.arg_utils import AsyncEngineArgs\n", "")},
     "installs AsyncEngineArgs, which the module neither defines nor imports by name"),
    ({"vllm/engine/arg_utils.py": None}, "AsyncEngineArgs comes from vllm.engine.arg_utils, which this engine "
                                         "tree lacks (no vllm/engine/arg_utils.py)"),
    ({"vllm/engine/arg_utils.py": "from vllm.engine.base import AsyncEngineArgs\n"},
     "vllm/engine/arg_utils.py defines no class AsyncEngineArgs"),
    ({OLD_PARSER: PARSER.replace("    return parser", "    add_more(parser)\n    return parser")},
     "make_arg_parser passes its parser to add_more"),
])
def test_unfollowable_argument_source_refused(tmp_path, changes, want):
    """A file the serve command's imports lead to that is missing, or a link
    that cannot be followed, is a refusal naming it: never a shorter list."""
    tree = make_tree(str(tmp_path / "t"))
    for rel, text in changes.items():
        if text is None:
            os.remove(os.path.join(tree, rel))
        else:
            write(os.path.join(tree, rel), text)
    with pytest.raises(Refusal) as exc:
        read_arg_options(tree)
    assert want in str(exc.value)


def test_launch_refused_without_the_parser_import(tmp_path):
    w = World(tmp_path, extra_files={SERVE: "def subparser_init(p):\n    return p\n"})
    assert f"{SERVE}: no 'from ... import make_arg_parser'" in w.refused()
