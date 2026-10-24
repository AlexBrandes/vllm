# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment variables retired in this release (vllm/sm80/envs.py).

A retired name is never read for a value; setting it only logs one warning
per process when vllm.envs is imported. vLLM's generic unknown-variable check
may also name it. Remove this file with the table in the next release.
"""

import os
import subprocess
import sys

import pytest

import vllm.envs as envs
from vllm.sm80 import envs as fork_envs

RENAMED = {
    "VLLM_GLM5_DECODE_MOE_MAX_TOKENS": "VLLM_MOE_MASK_PADDING_MAX_TOKENS",
    "VLLM_GLM5_FLA_PIN_AUTOTUNE": "VLLM_FLA_PIN_AUTOTUNE",
    "VLLM_GLM5_HOST_ALLREDUCE": "VLLM_HOST_ALLREDUCE",
    "VLLM_GLM5_HOST_ALLREDUCE_MAX_SIZE": "VLLM_HOST_ALLREDUCE_MAX_SIZE",
    "VLLM_GLM5_MEM_ATTRIBUTION": "VLLM_MEM_ATTRIBUTION",
    "VLLM_GLM5_MOE_MASK_PADDING": "VLLM_MOE_MASK_PADDING",
    "VLLM_GLM5_SHARED_EXPERT_REORDER": "VLLM_SHARED_EXPERT_REORDER",
    "VLLM_GLM5_STATE_INDEX_CHECK": "VLLM_STATE_INDEX_CHECK",
    "VLLM_GLM5_THIN_GEMM": "VLLM_THIN_GEMM",
    "VLLM_GLM5_THIN_GEMM_MAX_TOKENS": "VLLM_THIN_GEMM_MAX_TOKENS",
    "VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE": "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P",
    "VLLM_BUILD_AMPERE_MARLIN": "VLLM_AMPERE_MARLIN_BUILD",
    "VLLM_DSA_INDEXER_REF": "VLLM_INDEXER_DSA_REF",
    "VLLM_GLM5_DECODE_IDX_GLUE": "VLLM_AMPERE_DECODE_IDX_GLUE",
    "VLLM_GLM5_DECODE_KERNELS": "VLLM_AMPERE_DECODE_KERNELS",
    "VLLM_GLM5_DECODE_MOE_ROUTE_V2": "VLLM_MOE_DECODE_ROUTE_V2",
    "VLLM_GLM5_DETERMINISTIC_MOE_ALIGN": "VLLM_MOE_DETERMINISTIC_ALIGN",
    "VLLM_GLM5_DFLASH_BOUNDARY_CACHE": "VLLM_DFLASH_BOUNDARY_CACHE",
    "VLLM_GLM5_INDEXER_DECODE_RAW_K": "VLLM_INDEXER_DECODE_RAW_K",
    "VLLM_GLM5_MARLIN_DECODE_CUDA": "VLLM_AMPERE_MARLIN_DECODE_CUDA",
    "VLLM_GLM5_PP_KDA_PREFILL": "VLLM_AMPERE_KDA_PREFILL_WIDE",
    "VLLM_GLM5_PP_MARLIN_PREFILL": "VLLM_AMPERE_MARLIN_PREFILL_WIDE",
    "VLLM_GLM5_PP_MARLIN_PREFILL_COMPILED": "VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED",
    "VLLM_GLM5_PP_SPARSE_MLA_PREFILL": "VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE",
    "VLLM_GLM5_TOOL_CHOICE_NONE_MASK": "VLLM_TOOL_CHOICE_NONE_MASK",
    "VLLM_GLM5_TP4_KDA_PREFILL": "VLLM_AMPERE_KDA_PREFILL_NARROW",
    "VLLM_GLM5_TP4_MARLIN_PREFILL": "VLLM_AMPERE_MARLIN_PREFILL_NARROW",
    "VLLM_DEEPSEEK_V41_DSPARK_SHARD_ARGMAX": "VLLM_DSPARK_SHARD_ARGMAX",
    "VLLM_DEEPSEEK_V41_TARGET_SHARD_ARGMAX": "VLLM_SM80_TARGET_SHARD_ARGMAX",
}

REPLACED_OR_REMOVED = {
    "VLLM_GLM5_DFLASH_ADAPTIVE_K",
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_DEPTHS",
    "VLLM_GLM5_DFLASH_ADAPTIVE_DRAFT_WIDTH",
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_ACCEPT",
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_COSTS",
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_COSTS_MULTI",
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_HYST",
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_PRIOR",
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_LOG",
    "VLLM_GLM5_LOCAL_LOGITS",
    "VLLM_GLM5_HOST_ALLREDUCE_BUILD_DIR",
    "VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR",
    "VLLM_GLM5_AUX_HIDDEN_TENSOR",
    "VLLM_GLM5_DECODE_MHC",
    "VLLM_GLM5_DECODE_MHC_MAX_TOKENS",
    "VLLM_GLM5_DECODE_MHC_V3",
    "VLLM_GLM5_DECODE_MHC_V2_FN_BF16",
    "VLLM_GLM5_THIN_GEMM_V74",
    "VLLM_GLM5_PREFILL_PACK_BF16X2",
    "VLLM_GLM5_MARLIN_DECODE_VARIANT",
    "VLLM_GLM5_CUSTOM_ALLREDUCE_MAX_SIZE",
    "THIN_GEMM_MAX_M",
    "VLLM_KV_SWA_INFLIGHT_SCRATCH",
    "VLLM_GLM5_INDEXER_DECODE_ROWS",
    "VLLM_GLM5_INDEXER_GATHER_CLAMP",
    "VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED",
    "VLLM_GLM5_PREFILL_OVERLAP",
    "VLLM_GLM5_PREFILL_OVERLAP_SPLITS",
    "VLLM_GLM5_PREFILL_OVERLAP_MIN_TOKENS",
    "VLLM_GLM5_PREFILL_OVERLAP_DEBUG",
    "VLLM_GLM5_PREFILL_OVERLAP_UNDER_CA",
    "VLLM_GLM5_PREFILL_OVERLAP_BACKEND",
    "VLLM_GLM5_PREFILL_OBSERVE",
    "VLLM_PP_DRAFT_TAIL_VERIFY",
    "VLLM_GLM5_HOST_ALLREDUCE_TEST_FAIL_RANK",
    "VLLM_HOST_ALLREDUCE_TEST_FAIL_RANK",
    "VLLM_EXL3",
    "VLLM_ENGRAM_NATIVE_READ",
}


def test_table_lists_every_retired_name():
    assert set(fork_envs.RETIRED_ENV_VARS) == set(RENAMED) | REPLACED_OR_REMOVED
    assert all(fork_envs.RETIRED_ENV_VARS.values())


@pytest.mark.parametrize("name", sorted(set(RENAMED) | REPLACED_OR_REMOVED))
def test_retired_name_is_not_declared(name):
    assert name not in envs.environment_variables


@pytest.mark.parametrize("old,new", sorted(RENAMED.items()))
def test_renamed_name_points_at_its_replacement(old, new):
    assert new in envs.environment_variables
    assert fork_envs.RETIRED_ENV_VARS[old].startswith(f"renamed to {new}")


@pytest.mark.parametrize("old,new", sorted(RENAMED.items()))
def test_retired_value_is_never_used(monkeypatch, old, new):
    monkeypatch.delenv(old, raising=False)
    monkeypatch.delenv(new, raising=False)
    default = envs.environment_variables[new]()
    # Not a valid value for any of the new names: an alias would raise.
    monkeypatch.setenv(old, "not-a-valid-value")
    assert envs.environment_variables[new]() == default


def test_warns_once_per_process(monkeypatch):
    retired = []
    monkeypatch.setattr(fork_envs, "_warned_retired_env_vars", set())
    monkeypatch.setattr(
        fork_envs.logger,
        "warning",
        lambda msg, *args: retired.append(msg % args),
    )
    for name in fork_envs.RETIRED_ENV_VARS:
        monkeypatch.setenv(name, "1")

    fork_envs.warn_retired_env_vars()
    fork_envs.warn_retired_env_vars()

    for name, replacement in fork_envs.RETIRED_ENV_VARS.items():
        assert retired.count(f"{name} is retired and ignored: {replacement}.") == 1
    assert len(retired) == len(fork_envs.RETIRED_ENV_VARS)


def test_process_startup_warns_once_and_ignores_the_value():
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("VLLM_") and k not in fork_envs.RETIRED_ENV_VARS
    }
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["VLLM_GLM5_MOE_MASK_PADDING"] = "1"
    env["VLLM_GLM5_HOST_ALLREDUCE_BUILD_DIR"] = "/nonexistent"
    code = (
        "import vllm.envs as e\n"
        "from vllm.platforms.interface import Platform\n"
        "Platform.validate_environ(False)\n"
        "assert e.VLLM_MOE_MASK_PADDING is False\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    for name in ("VLLM_GLM5_MOE_MASK_PADDING", "VLLM_GLM5_HOST_ALLREDUCE_BUILD_DIR"):
        line = f"{name} is retired and ignored: {fork_envs.RETIRED_ENV_VARS[name]}."
        assert out.count(line) == 1, out
        # The generic unknown-variable warning may follow; it is not a second
        # retired-name warning and the value is still ignored.
