# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and load the sm_80 DeepSeek-V4.1 EXL3 target per pipeline stage at
TP=1 (the CMP PP3 layout: VLLM_PP_LAYER_PARTITION, both Engram layers on the
first stage) and the DSpark drafter on the last stage, on CPU. Cuts inside a
KV-sharing group build shadow source attention modules (sm80_pp_shadow.py).

Across the stages every target tensor of the tiny checkpoint is consumed by
exactly one stage (embedding and vision on the first, final norm and head on
the last, each decoder layer on the stage the partition gives it), and the
last stage's drafter owns every ``mtp.*`` tensor plus its own copy of the
input embedding (the target's lives on the first stage).
"""

import re
from collections import defaultdict

import pytest

from tests.models.deepseek_v41.sm80 import construct
from tests.models.deepseek_v41.sm80 import exl3_fixture as fixture
from tests.models.deepseek_v41.sm80.test_exl3_construction import _load
from tests.models.test_native_checkpoint_config_meta import (
    meta_platform,  # noqa: F401
)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny_pp")
    fixture.write_checkpoint(directory, fixture.tiny_config())
    return directory


def _layer(name: str) -> int | None:
    match = re.match(r"layers\.(\d+)\.", name)
    return None if match is None else int(match[1])


# The tiny model has 8 decoder layers and Engram layers 1 and 4: 5,1,2 keeps
# both Engram layers on stage 0 as 16,9,15 does for the real 40 layers;
# 3,3,2 puts one on each of the first two stages.
@pytest.mark.parametrize("partition", ["5,1,2", "3,3,2"])
def test_pp3_stages_own_disjoint_tensors(
    monkeypatch, meta_platform, checkpoint, partition  # noqa: F811
):
    from vllm_cmp170hx import plugin

    plugin._register_exl3()
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    sizes = [int(x) for x in partition.split(",")]
    bounds, start = [], 0
    for size in sizes:
        bounds.append((start, start + size))
        start += size
    stage_of = {}
    for stage, (lo, hi) in enumerate(bounds):
        for layer in range(lo, hi):
            stage_of[layer] = stage
    from vllm.transformers_utils.config import get_config, get_hf_text_config
    from vllm_cmp170hx.models.deepseek_v41.sm80_pp_shadow import shadow_sources

    text = get_hf_text_config(get_config(str(checkpoint), trust_remote_code=False))
    shadowed = defaultdict(set)
    for stage, (lo, hi) in enumerate(bounds):
        for source in shadow_sources(text, lo, hi):
            shadowed[source].add(stage)
    consumed = defaultdict(list)
    why = defaultdict(list)
    tensors = None
    draft_result = None
    for stage in range(len(sizes)):
        construct.set_groups(monkeypatch, 1, 0, pp_size=len(sizes), pp_rank=stage)
        construct.stub_engram_tables(monkeypatch, 1)
        target_config, draft_config = construct.vllm_configs(
            str(checkpoint), 1, pp_size=len(sizes)
        )
        target = construct.build_model(target_config)
        tensors = set(target_config.quant_config.manifest.tensors)
        result = _load(monkeypatch, target, target_config)
        assert set(result) == tensors
        for name, (use, by) in result.items():
            if use != "unused":
                consumed[name].append(stage)
                why[name].append(by)
        if stage == len(sizes) - 1:
            draft = construct.build_model(draft_config)
            assert draft.has_own_embed_tokens
            draft_result = _load(monkeypatch, draft, draft_config)
            assert set(draft_result) == tensors
    assert tensors is not None and draft_result is not None
    for name in tensors:
        if name.startswith("mtp."):
            assert name not in consumed, name
            assert draft_result[name][0] != "unused", name
            continue
        stages = consumed.get(name, [])
        layer = _layer(name)
        if (
            layer is not None
            and ".attn." in name
            and layer in shadowed
            and not re.search(r"\.attn\.wo_[ab]\.", name)
        ):
            # A shadowed source's attention (but its output projections)
            # loads on its own stage and on every stage that shadows it.
            assert stages == sorted({stage_of[layer], *shadowed[layer]}), (name, stages)
        elif layer is not None:
            assert stages == [stage_of[layer]], (name, stages)
        elif name.startswith(("embed.", "vision.", "aligner.")):
            assert stages == [0], (name, stages)
        elif name.startswith("image_"):
            # Three hidden-size sentinel vectors, kept on every stage.
            assert stages, name
        elif name.startswith(("head.", "norm.", "hc_head")):
            assert stages == [len(sizes) - 1], (name, stages, why[name])
        else:
            assert len(stages) == 1, (name, stages)
    # The last stage's drafter loads its own embedding copy.
    embed = [n for n in tensors if n.startswith("embed.")]
    assert embed and all(draft_result[n][0] != "unused" for n in embed)


def test_cmp_pp3_plan_for_the_real_config(monkeypatch):
    """DeepSeek-V4.1-Flash-1017 on cards 1, 4, 2 (16,9,15): stage 1 shadows
    layer 14 (KV group 14-19), stage 2 shadows 20 (KV group 20-39, the
    candidate source) and 24 (the index source of 25-27); stage 0 ships
    layer 14's attention input, stage 1 ships 20's and 24's."""
    from types import SimpleNamespace

    from vllm_cmp170hx.models.deepseek_v41.sm80_pp_shadow import stage_plans

    config = SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
    )
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "16,9,15")
    plans = stage_plans(config, 3)
    assert [(p.start, p.end) for p in plans] == [(0, 16), (16, 25), (25, 40)]
    assert [p.shadows for p in plans] == [(), (14,), (20, 24)]
    assert [p.send for p in plans] == [(14,), (20, 24), ()]
    assert [p.capture for p in plans] == [(14,), (20, 24), ()]
    assert [p.receive for p in plans] == [(), (14,), (20, 24)]
    # A partition on the group boundaries needs no shadow at all.
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "14,6,20")
    assert all(not p.shadows for p in stage_plans(config, 3))
