# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only eligibility boundaries for the exact GLM routed prefill family."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vllm_cmp170hx.kernels.exl3 import ops
from vllm_cmp170hx.kernels.exl3.quant import (
    _layer_bits_x2,
    _prefill_layout_closed_reason,
)


def _family(bits_x2=5):
    return dict(
        capability=(8, 0),
        dtype=torch.bfloat16,
        hidden_size=4096,
        intermediate_size=2048,
        top_k=8,
        local_experts=288,
        global_experts=288,
        metadata=[
            [
                [4096, 2048, bits_x2, 2],
                [4096, 2048, bits_x2, 2],
                [2048, 4096, bits_x2, 2],
            ]
            for _ in range(288)
        ],
        bits_x2=bits_x2,
    )


def test_glm_prefill_does_not_open_the_decode_operator():
    family = _family()
    assert ops.moe_prefill_closed_reason(**family) is None
    v2 = {k: v for k, v in family.items() if k != "bits_x2"}
    assert ops.moe_v2_closed_reason(**v2) is not None


@pytest.mark.parametrize("local", [0, 256, 287])
def test_nonuniform_expert_formats_keep_the_base_path(local):
    family = _family()
    family["metadata"][local][2][2] = 7 if family["bits_x2"] == 5 else 5
    assert ops.moe_prefill_closed_reason(**family) is not None


def test_layer_bits_reads_the_uniform_format_and_fails_closed_on_mixed():
    assert _layer_bits_x2(_family()["metadata"]) == 5
    assert _layer_bits_x2(_family(bits_x2=7)["metadata"]) == 7
    mixed = _family(bits_x2=7)["metadata"]
    mixed[3][1][2] = 5
    assert _layer_bits_x2(mixed) is None


def test_uniform_35_bit_layer_is_covered():
    assert ops.moe_prefill_closed_reason(**_family(bits_x2=7)) is None


def test_noninteger_or_mixed_layer_bits_fail_closed():
    family = _family()
    family["bits_x2"] = 6
    assert ops.moe_prefill_closed_reason(**family) is not None
    family = _family(bits_x2=7)
    family["metadata"][0][0][2] = 5
    assert ops.moe_prefill_closed_reason(**family) is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("capability", (8, 6)),
        ("dtype", torch.float16),
        ("intermediate_size", 1024),
        ("top_k", 10),
        ("local_experts", 256),
        ("global_experts", 512),
    ],
)
def test_uncovered_routed_geometry_fails_closed(field, value):
    family = _family()
    family[field] = value
    assert ops.moe_prefill_closed_reason(**family) is not None


def test_missing_expert_metadata_fails_closed():
    family = _family()
    family["metadata"].pop()
    assert ops.moe_prefill_closed_reason(**family) is not None


def _config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=2
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(
                model_type="glm5_next_text", hidden_size=4096
            ),
        ),
        speculative_config=SimpleNamespace(method="dflash", parallel_drafting=True),
        num_speculative_tokens=3,
    )


def test_glm_layout_gate_is_separate_from_qwen():
    config = _config()
    assert _prefill_layout_closed_reason(config, glm=True) is None
    assert _prefill_layout_closed_reason(config) is not None
    for field, value in [("tensor_parallel_size", 2), ("pipeline_parallel_size", 1)]:
        invalid = deepcopy(config)
        setattr(invalid.parallel_config, field, value)
        assert _prefill_layout_closed_reason(invalid, glm=True) is not None


@pytest.mark.parametrize("depth", [3, 5, 7])
def test_glm_dflash_depth_stays_within_decode_row_capacity(depth):
    from vllm_cmp170hx.kernels.exl3.step_phase import max_decode_query_len

    config = _config()
    config.num_speculative_tokens = depth
    assert _prefill_layout_closed_reason(config, glm=True) is None
    assert max_decode_query_len(config) <= ops.MOE_DECODE_TC_MAX_ROWS


@pytest.mark.parametrize("change", ["model", "dtype", "drafter", "depth"])
def test_glm_configuration_rejects_unchecked_contracts(change):
    config = _config()
    if change == "model":
        config.model_config.hf_text_config.model_type = "other"
    elif change == "dtype":
        config.model_config.dtype = torch.float16
    elif change == "drafter":
        config.speculative_config.method = "mtp"
    else:
        config.num_speculative_tokens = 9
    assert _prefill_layout_closed_reason(config, glm=True) is not None
