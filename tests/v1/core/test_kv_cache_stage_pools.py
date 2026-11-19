# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-PP-stage KV cache block pools (--kv-cache-pool-per-pp-stage).

Planning: each set of workers that holds a group's layers gets its own pool,
sized by water-filling, so a stage is charged only for the KV cache of layers
it holds. Scheduling: admission, preemption, prefix caching and frees are per
pool; a request needs room in every pool.
"""

import pytest
import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.core import kv_cache_planning
from vllm.v1.core.kv_cache_planning import (
    _max_memory_usage_bytes_from_groups,
    _project_kv_cache_groups_to_worker,
    group_blocks_per_request,
    held_pool_groups,
    plan_stage_pool_num_blocks,
)
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_max_concurrency_for_kv_cache_config,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .utils import create_requests, new_kv_cache_spec, new_sliding_window_spec

PAGE = 32 * 1024  # DeepSeek-V4.1 sm80 BF16 pages after unification


@pytest.fixture(autouse=True)
def _v2_model_runner(monkeypatch):
    # Stage pools are implemented for Model Runner V2 only.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")


# --------------------------------------------------------------------------
# DeepSeek-V4.1-Flash-1017 on PP 15,10,15 (the KV audit's layout)
# --------------------------------------------------------------------------

COMPRESS_RATIOS = [0, 0] + [2] * 18 + [1] * 20
KV_SOURCES = (2, 8, 14, 20)


def _swa_spec() -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=32,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=128,
        alignment=512,
        model_version="deepseek_v4",
    )


def _compressed_spec(ratio: int) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        tokens_per_state=ratio,
        alignment=512,
        model_version="deepseek_v4",
    )


def _index_spec(ratio: int) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=32,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=ratio,
        alignment=512,
    )


def _ring_spec() -> CircularBufferSpec:
    return CircularBufferSpec(
        block_size=8, num_kv_heads=1, head_size=1024, head_size_v=0, dtype=torch.float32
    )


def _layer_specs(layer: int) -> dict:
    """The caches a stage holds for `layer` (also for a PP shadow of it)."""
    specs = {f"model.layers.{layer}.attn.swa_cache": _swa_spec()}
    if layer in KV_SOURCES:
        ratio = COMPRESS_RATIOS[layer]
        specs[f"model.layers.{layer}.attn"] = _compressed_spec(ratio)
        specs[f"model.layers.{layer}.attn.indexer.k_cache"] = _index_spec(ratio)
        if ratio == 2:
            specs[f"model.layers.{layer}.attn.compressor"] = _ring_spec()
    return specs


def _dsv41_pp3_specs() -> list[dict]:
    """Stages 0-14 | 15-24 + shadow 14 | 25-39 + shadows 20, 24."""
    stages = [range(0, 15), [14, *range(15, 25)], [20, 24, *range(25, 40)]]
    return [
        {name: spec for layer in layers for name, spec in _layer_specs(layer).items()}
        for layers in stages
    ]


def _dsv41_vllm_config(max_model_len: int, stage_pools: bool) -> VllmConfig:
    """PP 3, async scheduling on Model Runner V2, 2048-token steps: a sliding
    window layer reserves cdiv(127 + 4 * 2048, 32) + 1 = 261 blocks."""
    vllm_config = VllmConfig(
        model_config=ModelConfig(max_model_len=8192),
        scheduler_config=SchedulerConfig(
            max_model_len=8192,
            max_num_batched_tokens=2048,
            async_scheduling=True,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(
            kv_cache_pool_per_pp_stage=stage_pools,
            enable_extensible_kv_cache=False,
        ),
    )
    # Set after validation: no GPUs are visible to the test.
    vllm_config.parallel_config.pipeline_parallel_size = 3
    vllm_config.model_config.max_model_len = max_model_len
    vllm_config.scheduler_config.max_model_len = max_model_len
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    vllm_config.cache_config.prefix_cache_retention_interval = None
    assert vllm_config.max_in_flight_tokens == 4 * 2048
    return vllm_config


def _charged_blocks(vllm_config, worker_groups) -> int:
    builder = kv_cache_planning.DefaultKVCacheConfigBuilder()
    return (
        _max_memory_usage_bytes_from_groups(builder, vllm_config, worker_groups) // PAGE
    )


@pytest.mark.parametrize("stage_pools", [False, True])
def test_dsv41_pp3_charges_match_audit(stage_pools: bool):
    """Per-request charges at 262144 tokens (runtime/dsv41/kv-audit.md):
    one pool charges every rank 36,043 blocks (1.100 GiB); stage pools charge
    each rank only for the pools it holds: 19,278 / 18,232 / 14,677."""
    vllm_config = _dsv41_vllm_config(262144, stage_pools)
    specs = _dsv41_pp3_specs()
    groups = kv_cache_planning.get_global_kv_cache_groups(vllm_config, specs)
    assert len(groups) == 51
    assert {g.kv_cache_spec.page_size_bytes for g in groups} == {PAGE}

    projected = [_project_kv_cache_groups_to_worker(groups, s) for s in specs]
    charged = [_charged_blocks(vllm_config, p) for p in projected]
    if not stage_pools:
        assert {g.pool_id for g in groups} == {0}
        assert charged == [36043] * 3
        gib = charged[0] * PAGE / 2**30
        assert gib == pytest.approx(1.100, abs=5e-4)
        return

    # Five pools: {0}, {0,1}, {1}, {1,2}, {2}.
    demand = {}
    members = {}
    for pool_id, pool in kv_cache_planning.pool_groups(groups).items():
        demand[pool_id] = sum(group_blocks_per_request(vllm_config, g) for g in pool)
        members[pool_id] = tuple(
            w for w, p in enumerate(projected) if pool_id in held_pool_groups(p)
        )
    assert members == {0: (0,), 1: (0, 1), 2: (1,), 3: (1, 2), 4: (2,)}
    # {0}: 14 SWA x 261 + c2 main 2/8 + index 2/8 + rings 2/8
    # {0,1}: SWA 14 + c2 14 main/index/ring (shadow 14 on stage 1)
    # {1}: 8 SWA; {1,2}: SWA 20/24 + c1 20 main/index; {2}: 15 SWA
    assert demand == {0: 13896, 1: 5382, 2: 2088, 3: 10762, 4: 3915}
    assert charged == [19278, 18232, 14677]
    assert charged[0] * PAGE / 2**30 == pytest.approx(0.588, abs=5e-4)

    # Shadow sources stay in their source's group.
    shadow = next(g for g in groups if g.layer_names == ["model.layers.14.attn"])
    assert members[shadow.pool_id] == (0, 1)


def test_dsv41_pp3_fits_262144_only_with_stage_pools():
    """Rank 0 has 20,736 blocks of room (boot4), ranks 1/2 32,831."""
    available = [20736 * PAGE, 32831 * PAGE, 32831 * PAGE]
    specs = _dsv41_pp3_specs()

    vllm_config = _dsv41_vllm_config(262144, stage_pools=False)
    with pytest.raises(ValueError, match="larger than the available KV cache"):
        kv_cache_planning.get_kv_cache_configs(
            vllm_config,
            specs,
            available,
            kv_cache_planning.get_global_kv_cache_groups(vllm_config, specs),
        )

    vllm_config = _dsv41_vllm_config(262144, stage_pools=True)
    groups = kv_cache_planning.get_global_kv_cache_groups(vllm_config, specs)
    configs = kv_cache_planning.get_kv_cache_configs(
        vllm_config, specs, available, groups
    )
    pool_blocks = {0: configs[0].num_blocks, **configs[0].pool_num_blocks}
    assert all(c.pool_num_blocks == configs[0].pool_num_blocks for c in configs)
    assert all(c.num_blocks == configs[0].num_blocks for c in configs)

    demand = {
        pool_id: sum(group_blocks_per_request(vllm_config, g) for g in pool)
        for pool_id, pool in kv_cache_planning.pool_groups(groups).items()
    }
    # Every pool holds a full-length request plus its null block.
    assert all(pool_blocks[p] >= demand[p] + 1 for p in demand)
    for worker, (config, budget) in enumerate(zip(configs, available)):
        held = {t.pool_id for t in config.kv_cache_tensors}
        assert held == set(held_pool_groups(config.kv_cache_groups))
        size = config.kv_cache_tensors[0].size
        assert all(t.size == size for t in config.kv_cache_tensors)
        assert size == sum(pool_blocks[p] * PAGE for p in held) <= budget
        # Each pool's tensors sit in their own region, sized for that pool.
        for t in config.kv_cache_tensors:
            assert t.layer_stride == pool_blocks[t.pool_id] * PAGE
        if worker == 0:
            # The tightest rank uses its memory up to rounding.
            assert budget - size < len(held) * PAGE
    # Max-min water-filling: rank 0's two pools at the same scale; rank 1's
    # surplus goes to its own pools, which then fit more requests.
    scale = [(pool_blocks[p] - 1) / demand[p] for p in sorted(demand)]
    assert scale[0] == pytest.approx(scale[1], rel=1e-3)
    assert min(scale) == pytest.approx(scale[0], rel=1e-3)
    assert scale[2] > 1.9 and scale[3] > 1.9

    scheduler_config = generate_scheduler_kv_cache_config(configs)
    concurrency = get_max_concurrency_for_kv_cache_config(vllm_config, scheduler_config)
    assert concurrency == pytest.approx(min(scale), rel=1e-3)


def test_dsv41_pp3_max_model_len_estimate():
    """Rank 0 at 0.633 GiB: about 105k with one pool, about 287k with stage
    pools (audit section 4)."""
    builder = kv_cache_planning.DefaultKVCacheConfigBuilder()
    specs = _dsv41_pp3_specs()
    estimates = {}
    for stage_pools in (False, True):
        vllm_config = _dsv41_vllm_config(400000, stage_pools)
        groups = kv_cache_planning.get_global_kv_cache_groups(vllm_config, specs)
        rank0 = _project_kv_cache_groups_to_worker(groups, specs[0])
        nulls = len(held_pool_groups(rank0)) if stage_pools else 1
        estimates[stage_pools] = kv_cache_planning._estimate_max_model_len_from_groups(
            builder, vllm_config, rank0, (20736 - nulls) * PAGE
        )
    assert 105000 <= estimates[False] <= 105600
    assert 286500 <= estimates[True] <= 287100


# --------------------------------------------------------------------------
# Fallback and water-filling unit cases
# --------------------------------------------------------------------------


def _small_pp_config(stage_pools: bool, max_model_len: int = 64) -> VllmConfig:
    vllm_config = VllmConfig(
        model_config=ModelConfig(max_model_len=max_model_len),
        cache_config=CacheConfig(
            kv_cache_pool_per_pp_stage=stage_pools,
            enable_extensible_kv_cache=False,
        ),
    )
    vllm_config.parallel_config.pipeline_parallel_size = 2
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    vllm_config.cache_config.prefix_cache_retention_interval = None
    return vllm_config


def test_groups_spanning_all_stages_keep_one_pool():
    """A uniform model's group spans every stage: one pool, as without the
    flag, and the same configs."""
    specs = [
        {"layer.0": new_kv_cache_spec(), "layer.1": new_kv_cache_spec()},
        {"layer.2": new_kv_cache_spec(), "layer.3": new_kv_cache_spec()},
    ]
    page = new_kv_cache_spec().page_size_bytes
    results = []
    for stage_pools in (False, True):
        vllm_config = _small_pp_config(stage_pools)
        groups = kv_cache_planning.get_global_kv_cache_groups(vllm_config, specs)
        assert [g.pool_id for g in groups] == [0] * len(groups)
        results.append(
            kv_cache_planning.get_kv_cache_configs(
                vllm_config, specs, [page * 2 * 10, page * 2 * 12], groups
            )
        )
    assert results[0] == results[1]
    assert all(not c.pool_num_blocks for c in results[1])


def test_pp1_flag_is_a_no_op():
    vllm_config = _small_pp_config(True)
    vllm_config.parallel_config.pipeline_parallel_size = 1
    specs = [{"full": new_kv_cache_spec(), "swa": new_kv_cache_spec(block_size=32)}]
    groups = kv_cache_planning.get_global_kv_cache_groups(vllm_config, specs)
    assert {g.pool_id for g in groups} == {0}


def _stage_local_specs() -> list[dict]:
    """Stage 0 holds a sliding-window layer, stage 1 two full-attention
    layers: hybrid grouping makes one-layer groups, each on one stage."""
    return [
        {"layer.0": new_sliding_window_spec(sliding_window=16)},
        {"layer.1": new_kv_cache_spec(), "layer.2": new_kv_cache_spec()},
    ]


def test_stage_local_groups_get_stage_pools():
    """One pool per stage, each backed by its own stage only and sized from
    that stage's memory."""
    specs = _stage_local_specs()
    vllm_config = _small_pp_config(True)
    groups = kv_cache_planning.get_global_kv_cache_groups(vllm_config, specs)
    assert [g.pool_id for g in groups] == [0, 1, 1]
    page0 = page1 = new_kv_cache_spec().page_size_bytes
    configs = kv_cache_planning.get_kv_cache_configs(
        vllm_config, specs, [page0 * 11, page1 * 41], groups
    )
    # Stage 0 backs only pool 0, stage 1 only pool 1.
    assert {t.pool_id for t in configs[0].kv_cache_tensors} == {0}
    assert {t.pool_id for t in configs[1].kv_cache_tensors} == {1}
    assert configs[0].num_blocks == 11
    assert configs[0].pool_num_blocks == {1: 41}
    # Without the flag stage 0 is charged for stage 1's two layers as well:
    # 5 + 4 + 4 blocks per request do not fit its 10.
    vllm_config = _small_pp_config(False)
    groups = kv_cache_planning.get_global_kv_cache_groups(vllm_config, specs)
    with pytest.raises(ValueError, match="larger than the available KV cache"):
        kv_cache_planning.get_kv_cache_configs(
            vllm_config, specs, [page0 * 11, page1 * 41], groups
        )


def test_water_filling_respects_every_budget():
    """Three workers, pools {0}, {0,1}, {1}, {1,2}, {2}: each worker's pools
    fit its budget, and the tightest worker's pools share one scale."""
    vllm_config = _small_pp_config(True)
    groups = [
        KVCacheGroupSpec([f"l{p}"], new_kv_cache_spec(), pool_id=p) for p in range(5)
    ]
    pool_bytes = [{0: 100, 1: 100}, {1: 100, 2: 50, 3: 100}, {3: 100, 4: 200}]
    budgets = [10_000, 50_000, 30_000]
    blocks = plan_stage_pool_num_blocks(vllm_config, groups, pool_bytes, budgets)
    for worker_bytes, budget in zip(pool_bytes, budgets):
        assert sum(blocks[p] * b for p, b in worker_bytes.items()) <= budget
    # Worker 0 binds; its pools get equal shares of its budget (same demand).
    assert blocks[0] == blocks[1] == 50
    assert blocks[2] > blocks[0] and blocks[4] > blocks[0]
    override = _small_pp_config(True)
    override.cache_config.num_gpu_blocks_override = 7
    assert plan_stage_pool_num_blocks(override, groups, pool_bytes, budgets) == (
        dict.fromkeys(range(5), 7)
    )
    override.cache_config.num_gpu_blocks_override = 4  # 64 tokens need 4 + null
    with pytest.raises(ValueError, match="num_gpu_blocks_override"):
        plan_stage_pool_num_blocks(override, groups, pool_bytes, budgets)


def test_stage_pools_refuse_kv_connectors():
    from vllm.config import KVTransferConfig

    specs = _stage_local_specs()
    vllm_config = _small_pp_config(True)
    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="ExampleConnector", kv_role="kv_both"
    )
    with pytest.raises(ValueError, match="KV connectors"):
        kv_cache_planning.get_global_kv_cache_groups(vllm_config, specs)


# --------------------------------------------------------------------------
# Scheduler with two pools
# --------------------------------------------------------------------------

BLOCK = 16


def _pooled_scheduler(
    pool_blocks: tuple[int, int],
    enable_prefix_caching: bool = False,
    async_scheduling: bool = False,
    max_num_batched_tokens: int = 256,
    mixed_precision: bool = False,
    max_model_len: int = 256,
) -> Scheduler:
    """Two one-layer groups, "a" in pool 0 and "b" in pool 1."""
    model_config = ModelConfig(
        model="facebook/opt-125m",
        dtype="float16",
        seed=42,
        max_model_len=max_model_len,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            enable_chunked_prefill=True,
            async_scheduling=async_scheduling,
            is_encoder_decoder=False,
            watermark=0.0,
        ),
        cache_config=CacheConfig(
            block_size=BLOCK,
            gpu_memory_utilization=0.9,
            cache_dtype="auto",
            enable_prefix_caching=enable_prefix_caching,
        ),
    )
    spec_a = FullAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    spec_b = FullAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=2 if mixed_precision else 1,
        dtype=torch.float16 if mixed_precision else torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=pool_blocks[0],
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["a"], spec_a, pool_id=0),
            KVCacheGroupSpec(["b"], spec_b, pool_id=1),
        ],
        pool_num_blocks={1: pool_blocks[1]},
    )
    vllm_config.cache_config.num_gpu_blocks = pool_blocks[0]
    register_all_kvcache_specs(vllm_config)
    cls = AsyncScheduler if async_scheduling else Scheduler
    return cls(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def _free_blocks(scheduler) -> list[int]:
    return [
        pool.get_num_free_blocks()
        for pool in scheduler.kv_cache_manager.coordinator.block_pools
    ]


def _output(scheduler_output, token: int = 0) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[[token] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_pools_are_separate_block_pools():
    scheduler = _pooled_scheduler((100, 9))
    coordinator = scheduler.kv_cache_manager.coordinator
    assert len(coordinator.block_pools) == 2
    managers = coordinator.single_type_managers
    assert managers[0].block_pool is coordinator.block_pools[0]
    assert managers[1].block_pool is coordinator.block_pools[1]
    assert _free_blocks(scheduler) == [99, 8]


def test_admission_needs_room_in_every_pool():
    """Pool 1 holds 8 usable blocks: four 32-token requests take two each;
    the fifth waits although pool 0 is nearly empty."""
    scheduler = _pooled_scheduler((100, 9))
    requests = create_requests(5, num_tokens=32, block_size=BLOCK, max_tokens=1)
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert len(output.num_scheduled_tokens) == 4
    assert requests[4].status == RequestStatus.WAITING
    assert _free_blocks(scheduler) == [99 - 8, 0]
    assert scheduler.get_kv_cache_usage() == pytest.approx(1.0)
    # Each new request's block ids come from the group's own pool.
    for new_req in output.scheduled_new_reqs:
        ids_a, ids_b = new_req.block_ids
        assert len(ids_a) == len(ids_b) == 2
        assert max(ids_b) <= 8

    # Finishing frees every request's blocks back to its own pool.
    scheduler.update_from_output(output, _output(output))
    assert _free_blocks(scheduler) == [99, 8]
    output = scheduler.schedule()
    assert requests[4].request_id in output.num_scheduled_tokens


def test_admission_boundary_exact_fit():
    """A request that exactly fills pool 1 is admitted; one more token is not
    (a full-sequence gate per pool)."""
    scheduler = _pooled_scheduler((100, 5))  # 4 usable blocks in pool 1
    fits = create_requests(1, num_tokens=64, block_size=BLOCK, max_tokens=1)[0]
    scheduler.add_request(fits)
    assert fits.request_id in scheduler.schedule().num_scheduled_tokens

    scheduler = _pooled_scheduler((100, 5))
    too_long = create_requests(1, num_tokens=65, block_size=BLOCK, max_tokens=1)[0]
    scheduler.add_request(too_long)
    assert not scheduler.schedule().num_scheduled_tokens
    assert too_long.status == RequestStatus.WAITING
    assert _free_blocks(scheduler) == [99, 4]


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_preemption_when_one_pool_runs_out(async_scheduling: bool):
    """Two requests decode until pool 1 (6 usable blocks) is full: the later
    one is preempted while pool 0 still has plenty, and its blocks return to
    both pools."""
    scheduler = _pooled_scheduler((100, 7), async_scheduling=async_scheduling)
    requests = create_requests(
        2, num_tokens=47, block_size=BLOCK, max_tokens=64, ignore_eos=True
    )
    for request in requests:
        scheduler.add_request(request)
    preempted = False
    for _ in range(40):
        output = scheduler.schedule()
        if requests[1].status == RequestStatus.PREEMPTED:
            preempted = True
            break
        scheduler.update_from_output(output, _output(output))
    assert preempted
    assert requests[0].status == RequestStatus.RUNNING
    free = _free_blocks(scheduler)
    held = [len(ids) for ids in scheduler.kv_cache_manager.get_block_ids("0")]
    assert free[0] == 99 - held[0] and free[1] == 6 - held[1]
    assert free[0] > 80


def test_prefix_cache_hits_and_evicts_per_pool():
    # Both groups cache the prompt in their own pool: a hit.
    scheduler = _pooled_scheduler((100, 9), enable_prefix_caching=True)
    first, second = create_requests(
        2, num_tokens=64, block_size=BLOCK, max_tokens=1, same_prompt=True
    )
    scheduler.add_request(first)
    output = scheduler.schedule()
    scheduler.update_from_output(output, _output(output))
    assert first.is_finished()
    scheduler.add_request(second)
    output = scheduler.schedule()
    # 48 of 64 tokens hit; the last block is recomputed.
    assert output.num_scheduled_tokens[second.request_id] == 16

    # Pool 1 (4 usable blocks) evicts the cached prompt to admit two other
    # prompts; pool 0 keeps it, but a hit needs every group: no hit.
    scheduler = _pooled_scheduler((100, 5), enable_prefix_caching=True)
    cached, other1, other2 = create_requests(
        3, num_tokens=32, block_size=BLOCK, max_tokens=1
    )
    scheduler.add_request(cached)
    output = scheduler.schedule()
    scheduler.update_from_output(output, _output(output))
    scheduler.add_request(other1)
    scheduler.add_request(other2)
    output = scheduler.schedule()
    assert len(output.num_scheduled_tokens) == 2
    scheduler.update_from_output(output, _output(output))
    assert _free_blocks(scheduler) == [99, 4]

    again = create_requests(1, num_tokens=32, block_size=BLOCK, req_ids=["again"])[0]
    assert again.prompt_token_ids == cached.prompt_token_ids
    pool0 = scheduler.kv_cache_manager.coordinator.block_pools[0]
    hits0 = pool0.get_cached_block(again.block_hashes[0], [0])
    assert hits0 is not None  # pool 0 still caches the prompt
    scheduler.add_request(again)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[again.request_id] == 32


def test_zeroing_ids_are_split_by_pool():
    scheduler = _pooled_scheduler((100, 9), mixed_precision=True)
    assert scheduler.needs_kv_cache_zeroing
    request = create_requests(1, num_tokens=32, block_size=BLOCK, max_tokens=1)[0]
    scheduler.add_request(request)
    output = scheduler.schedule()
    ids_a, ids_b = output.scheduled_new_reqs[0].block_ids
    assert sorted(output.new_block_ids_to_zero) == sorted(ids_a)
    assert output.pool_block_ids_to_zero == {1: list(ids_b)}


def test_single_pool_scheduler_output_unchanged():
    from .utils import create_scheduler

    scheduler = create_scheduler()
    request = create_requests(1, num_tokens=32)[0]
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert output.pool_block_ids_to_zero is None
    assert len(scheduler.kv_cache_manager.coordinator.block_pools) == 1
    assert scheduler.kv_cache_manager.watermark_blocks_by_pool == [0]


def test_block_copies_apply_to_their_own_pool(monkeypatch):
    """A CoW copy names block ids of one pool; the worker applies it only to
    that pool's caches."""
    from vllm.utils import torch_utils
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
    from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
    pool0 = torch.arange(4 * 3, dtype=torch.float32).view(4, 3)
    pool1 = torch.arange(6 * 3, dtype=torch.float32).view(6, 3) + 100
    want0, want1 = pool0.clone(), pool1.clone()
    want0[2] = want0[1]
    want1[5] = want1[4]
    copies = [KVCacheBlockCopy(1, 2, pool_id=0), KVCacheBlockCopy(4, 5, pool_id=1)]
    copy_kv_cache_blocks_inplace([pool0], 4, copies, 0)
    copy_kv_cache_blocks_inplace([pool1], 6, copies, 1)
    assert torch.equal(pool0, want0)
    assert torch.equal(pool1, want1)


# --------------------------------------------------------------------------
# Extensible KV cache: pools sized from the memory measured after warmup
# --------------------------------------------------------------------------


def _extensible_dsv41_configs(available: list[int]):
    vllm_config = _dsv41_vllm_config(262144, stage_pools=True)
    vllm_config.cache_config.enable_extensible_kv_cache = True
    specs = _dsv41_pp3_specs()
    groups = kv_cache_planning.get_global_kv_cache_groups(vllm_config, specs)
    configs = kv_cache_planning.get_kv_cache_configs(
        vllm_config, specs, available, groups
    )
    return vllm_config, groups, configs


def test_extensible_stage_pools_sized_after_warmup():
    """Reservations are generous; after warmup each worker reports its budget
    in bytes and the pools are water-filled over those budgets, the same plan
    as without the extensible cache on that memory."""
    from vllm.v1.worker.worker_base import CompilationTimes

    vllm_config, groups, configs = _extensible_dsv41_configs([8 << 30] * 3)
    scheduler_config = generate_scheduler_kv_cache_config(configs)
    reserved = {0: scheduler_config.num_blocks, **scheduler_config.pool_num_blocks}
    # Every pool can take its tightest holder's whole 8 GiB budget.
    assert min(reserved.values()) >= (8 << 30) // PAGE // 3

    measured = [20736 * PAGE, 32831 * PAGE, 32831 * PAGE]
    times = [CompilationTimes(0.0, 0.0, kv_cache_bytes=b) for b in measured]
    num_blocks = kv_cache_planning.finalize_extensible_kv_cache(
        vllm_config, configs, scheduler_config, times
    )
    sized = {0: num_blocks, **scheduler_config.pool_num_blocks}
    builder = kv_cache_planning.DefaultKVCacheConfigBuilder()
    pool_bytes = kv_cache_planning._pool_bytes_per_block_by_worker(
        builder, vllm_config, [c.kv_cache_groups for c in configs]
    )
    assert sized == plan_stage_pool_num_blocks(
        vllm_config, groups, pool_bytes, measured
    )
    assert sized == {0: 14946, 1: 5789, 2: 4394, 3: 22647, 4: 10184}
    assert all(sized[p] <= reserved[p] for p in sized)
    for config in configs:
        assert {0: config.num_blocks, **config.pool_num_blocks} == sized
    assert vllm_config.cache_config.num_gpu_blocks == sized[0]


def test_extensible_stage_pools_refuse_a_budget_too_small():
    from vllm.v1.worker.worker_base import CompilationTimes

    vllm_config, _, configs = _extensible_dsv41_configs([8 << 30] * 3)
    scheduler_config = generate_scheduler_kv_cache_config(configs)
    # Rank 0 needs 19,278 blocks (+2 null) for one 262144-token request.
    measured = [19000 * PAGE, 32831 * PAGE, 32831 * PAGE]
    times = [CompilationTimes(0.0, 0.0, kv_cache_bytes=b) for b in measured]
    with pytest.raises(ValueError, match="stage pools"):
        kv_cache_planning.finalize_extensible_kv_cache(
            vllm_config, configs, scheduler_config, times
        )


class _FakeExtensibleTensor:
    """Records commits instead of mapping device memory."""

    granularity = 2 << 20

    def __init__(self, size, device, segment_capacities):
        self.capacities = list(segment_capacities)
        self.committed = [0] * len(self.capacities)

    def resize_segments_(self, sizes, zero_new=False):
        for old, new, capacity in zip(self.committed, sizes, self.capacities):
            assert old <= new <= capacity
        self.committed = list(sizes)

    def release_physical(self):
        self.committed = [0] * len(self.capacities)

    def segments_backed_by_one_chunk(self, sizes):
        return [True] * len(sizes)

    @property
    def physical_bytes(self):
        g = self.granularity
        return sum(-(-c // g) * g for c in self.committed)


def test_extensible_kv_cache_commits_each_pool(monkeypatch):
    """Rank 1 backs pools 1, 2 and 3: each pool's layer regions are segments
    of that pool's reservation, and commits are capped per pool."""
    from vllm.v1.kv_cache_layout import KVCacheLayout
    from vllm.v1.worker import extensible_kv_cache

    monkeypatch.setattr(extensible_kv_cache, "ExtensibleTensor", _FakeExtensibleTensor)
    _, _, configs = _extensible_dsv41_configs([1 << 30, 2 << 30, 2 << 30])
    config = configs[1]
    cache = extensible_kv_cache.ExtensibleKVCache(
        config, torch.device("cpu"), KVCacheLayout.LBNHC
    )
    reserved = {p: config.num_blocks_of_pool(p) for p in (1, 2, 3)}
    assert cache.capacity_by_pool == reserved
    assert set(cache.segment_pools) == {1, 2, 3}
    assert cache.bytes_per_block == len(cache.segment_pools) * PAGE

    cache.commit(10)
    assert cache.committed_by_pool == {1: 10, 2: 10, 3: 10}
    assert cache.num_committed_blocks == 10
    counts = {0: 5, 1: 100, 2: 7, 3: reserved[3] + 5}
    cache.commit(counts, shrink=True)  # pool 2 shrinks: remap
    assert cache.committed_by_pool == {1: 100, 2: 7, 3: reserved[3]}
    sizes = [
        cache.committed_by_pool[p] * s
        for p, s in zip(cache.segment_pools, cache.segment_strides)
    ]
    assert cache.buffer.committed == sizes
    tensors = cache.committed_kv_cache_tensors(
        config, KVCacheLayout.LBNHC, cache.committed_by_pool
    )
    for tensor in tensors:
        assert tensor.pool_id in (1, 2, 3)
        assert tensor.layer_stride == (
            cache.committed_by_pool[tensor.pool_id] * tensor.block_stride
        )
    # Warmup commits one count everywhere, within the budget.
    budget_blocks = cache.blocks_within(cache.physical_bytes_for(50))
    assert budget_blocks >= 50


def test_extensible_stage_pools_defer_the_fit_check_to_warmup():
    """Before warmup rank 2 (model + drafter) reports almost no memory; with
    the extensible cache the one-request check and the reservations wait for
    the measured budgets (boot 8)."""
    from vllm.v1.worker.worker_base import CompilationTimes

    planning = [int(0.92 * 2**30), int(0.5 * 2**30), int(0.01 * 2**30)]
    vllm_config, _, configs = _extensible_dsv41_configs(planning)
    scheduler_config = generate_scheduler_kv_cache_config(configs)
    reserved = {0: scheduler_config.num_blocks, **scheduler_config.pool_num_blocks}
    floor = kv_cache_planning.EXTENSIBLE_POOL_RESERVATION_BYTES
    assert min(reserved.values()) >= (floor - 3 * PAGE) // PAGE // 3

    # After warmup rank 2 has 0.6 GiB (these BF16 pages need 0.45 GiB there).
    measured = [int(0.81 * 2**30), int(1.0 * 2**30), int(0.6 * 2**30)]
    times = [CompilationTimes(0.0, 0.0, kv_cache_bytes=b) for b in measured]
    num_blocks = kv_cache_planning.finalize_extensible_kv_cache(
        vllm_config, configs, scheduler_config, times
    )
    sized = {0: num_blocks, **scheduler_config.pool_num_blocks}
    assert all(sized[p] <= reserved[p] for p in sized)
    # Rank 2 holds pools 3 and 4 and uses its measured budget.
    used2 = (sized[3] + sized[4]) * PAGE
    assert measured[2] - 2 * PAGE <= used2 <= measured[2]

    # The non-extensible path still checks at planning.
    vllm_config = _dsv41_vllm_config(262144, stage_pools=True)
    specs = _dsv41_pp3_specs()
    groups = kv_cache_planning.get_global_kv_cache_groups(vllm_config, specs)
    with pytest.raises(ValueError, match="larger than the available KV cache"):
        kv_cache_planning.get_kv_cache_configs(vllm_config, specs, planning, groups)
