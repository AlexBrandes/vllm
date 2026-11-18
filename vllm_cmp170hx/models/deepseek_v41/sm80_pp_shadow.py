# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 pipeline stages that cut inside a KV-sharing group.

DeepSeek-V4.1 shares attention state between layers: compressed layers read
the compressed KV (and index K) cache of the latest ``kv_source_layer_ids``
entry at or below them, reuse the top-k of the latest
``index_source_layer_ids`` entry, and index sources above
``candidate_source_layer_id`` filter through that layer's candidate blocks.
For DeepSeek-V4.1-Flash-1017 the KV groups are layers 2-7, 8-13, 14-19 and
20-39, so a pipeline stage boundary anywhere but 2, 8, 14 or 20 leaves a
stage whose layers read caches its own rank never writes (upstream refuses:
"PP splits inside a v4.1 kv-sharing group are not supported"). On three CMP
170HX cards (64 + 40 + 64 GB) the 20-layer group 20-39 does not fit on one
card at ~2.2 bpw, so the CMP layout (e.g. 16,9,15) must cut inside groups.

A *shadow source* makes such a cut work: the consuming stage builds its own
attention module for each source layer it needs (same prefix, so it is
registered in the forward context and gets its own KV cache like any local
layer), and the stage that runs the source layer ships that layer's
attention input (``[tokens, hidden]`` BF16, after the mHC pre-mix and the
attention norm) to the later stages in the pipeline's intermediate tensors.
Before its own layers run, the consuming stage replays each shadow's
:meth:`attend` on that input: the shadow writes its SWA, compressed KV and
index K caches, publishes its top-k and candidate blocks into the stage's
shared buffers, exactly as the source layer does on its own stage, and its
attention output is dropped (the source's ``wo_a``/``wo_b`` never run here).
Shadows run in ascending layer order, so a later shadow (an index source
such as 24) sees the candidate blocks and K cache of an earlier one (20),
and the stage's own layers see the last published top-k.

The design follows the "PP shadow KV" approach of zebgop-ops/dsv41reap-pp
(Apache-2.0), reimplemented for this plugin's attention and EXL3 loader.

Cost per shadow: one attention input projection set and attention region per
step (no output projection), one more layer's KV cache on the stage, the
source's attention input weights resident twice (its ``wo_a``/``wo_b`` are
dropped from the shadow), and ``hidden * 2`` bytes per token on each
pipeline hop between the source and its last consumer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

SHIP_KEY = "pp_shadow_x.{}"
_LAYER = re.compile(r"(?<=layers\.)\d+(?=\.)")


def _ratio(config, layer: int) -> int:
    ratios = getattr(config, "compress_ratios", None) or ()
    return int(ratios[layer]) if layer < len(ratios) else 0


def sources_read_by(config, layer: int) -> set[int]:
    """The source layers whose caches, top-k or candidates ``layer``'s
    attention reads (itself excluded)."""
    if layer >= config.num_hidden_layers or _ratio(config, layer) == 0:
        return set()
    kv = tuple(getattr(config, "kv_source_layer_ids", None) or ())
    index = tuple(getattr(config, "index_source_layer_ids", None) or ())
    candidate = getattr(config, "candidate_source_layer_id", -1)
    out = {max(s for s in kv if s <= layer), max(s for s in index if s <= layer)}
    if layer in index and 0 <= candidate < layer:
        out.add(candidate)
    out.discard(layer)
    return out


def shadow_sources(config, start: int, end: int) -> list[int]:
    """Source layers below ``start`` that the stage [start, end) must shadow
    (with the sources those shadows read in turn), ascending."""
    need: set[int] = set()
    for layer in range(start, end):
        need |= sources_read_by(config, layer)
    stack = [s for s in need if s < start]
    while stack:
        for s in sources_read_by(config, stack.pop()):
            if s < start and s not in need:
                need.add(s)
                stack.append(s)
    return sorted(s for s in need if s < start)


@dataclass(frozen=True)
class StagePlan:
    start: int
    end: int
    shadows: tuple[int, ...]
    # Source attention inputs this stage receives / sends (keys of
    # SHIP_KEY): received ones are relayed or used, sent ones are captured
    # here (local sources) or relayed (received ones).
    receive: tuple[int, ...]
    send: tuple[int, ...]
    capture: tuple[int, ...]


def stage_plans(config, pp_size: int) -> list[StagePlan]:
    from vllm.distributed.utils import get_pp_indices

    bounds = [
        get_pp_indices(config.num_hidden_layers, rank, pp_size)
        for rank in range(pp_size)
    ]
    shadows = [shadow_sources(config, lo, hi) for lo, hi in bounds]
    plans = []
    for rank, (lo, hi) in enumerate(bounds):
        later = {s for j in range(rank + 1, pp_size) for s in shadows[j]}
        here_or_later = later | set(shadows[rank])
        receive = sorted(s for s in here_or_later if s < lo)
        send = sorted(s for s in later if s < hi)
        capture = sorted(s for s in later if lo <= s < hi)
        plans.append(
            StagePlan(lo, hi, tuple(shadows[rank]), tuple(receive), tuple(send),
                      tuple(capture))
        )
    return plans


def current_plan(config) -> StagePlan | None:
    """This rank's plan, or None without pipeline parallelism or for a
    config without KV sharing (the DSpark drafter's)."""
    from vllm.distributed import get_pp_group

    group = get_pp_group()
    if group.world_size <= 1 or not getattr(config, "kv_source_layer_ids", None):
        return None
    return _plans(config, group.world_size)[group.rank_in_group]


def _plans(config, pp_size: int) -> list[StagePlan]:
    key = (
        config.num_hidden_layers,
        tuple(config.compress_ratios or ()),
        tuple(config.kv_source_layer_ids or ()),
        tuple(config.index_source_layer_ids or ()),
        getattr(config, "candidate_source_layer_id", -1),
        pp_size,
        _partition(),
    )
    return _cached_plans(key, config, pp_size)


def _partition() -> str | None:
    import vllm.envs as envs

    return envs.VLLM_PP_LAYER_PARTITION


_PLAN_CACHE: dict = {}


def _cached_plans(key, config, pp_size):
    if key not in _PLAN_CACHE:
        _PLAN_CACHE[key] = stage_plans(config, pp_size)
        for rank, plan in enumerate(_PLAN_CACHE[key]):
            if plan.shadows or plan.send:
                logger.info(
                    "DeepSeek-V4.1 PP stage %d (layers %d-%d): shadow sources %s, "
                    "receives %s, sends %s",
                    rank, plan.start, plan.end - 1, plan.shadows,
                    plan.receive, plan.send,
                )
    return _PLAN_CACHE[key]


# Shadows built for a model under construction, by id of its vllm_config:
# {layer: attention module}; the model adopts them (attach_shadows).
_BUILT: dict[int, dict[int, nn.Module]] = {}
_BUILDING: set[int] = set()


def ensure_shadows(
    attn_cls,
    vllm_config,
    prefix: str,
    layer_id: int,
    *,
    topk_indices_buffer,
    aux_stream_list,
    candidate_block_buffer,
) -> None:
    """Called by the sm_80 attention constructor before upstream's: when
    ``layer_id`` is this stage's first layer, build the stage's shadow source
    attention modules first, so the local consumers find their source
    caches in the forward context. The shadows share the stage's top-k and
    candidate buffers."""
    config = vllm_config.model_config.hf_text_config
    plan = current_plan(config)
    key = id(vllm_config)
    if plan is None or not plan.shadows or key in _BUILDING:
        return
    if layer_id != plan.start or key in _BUILT:
        return
    _BUILDING.add(key)
    try:
        built = {}
        for source in plan.shadows:
            shadow_prefix = _LAYER.sub(str(source), prefix, count=1)
            built[source] = attn_cls(
                vllm_config,
                prefix=shadow_prefix,
                topk_indices_buffer=topk_indices_buffer,
                aux_stream_list=aux_stream_list,
                candidate_block_buffer=candidate_block_buffer,
            )
        _BUILT[key] = built
    finally:
        _BUILDING.discard(key)
    logger.info(
        "DeepSeek-V4.1 PP: stage of layers %d-%d shadows source layers %s",
        plan.start, plan.end - 1, list(plan.shadows),
    )


def attach_shadows(model: nn.Module, vllm_config) -> None:
    """Give the (sm_80) backbone ``model`` its stage plan, the shadow
    modules built for it (``pp_shadow_sources``) and the capture buffers of
    the local sources later stages need."""
    config = model.config
    plan = current_plan(config)
    model.pp_plan = plan
    from vllm.model_executor.models.utils import PPMissingLayer

    built = _BUILT.pop(id(vllm_config), {})
    for shadow in built.values():
        # A shadow's output is dropped: its output projections never run
        # here, so they hold no weights (their tensors stay unused).
        shadow.wo_a = PPMissingLayer()
        shadow.wo_b = PPMissingLayer()
    model.pp_shadow_sources = nn.ModuleDict({str(s): m for s, m in built.items()})
    if plan is None:
        return
    missing = [s for s in plan.shadows if str(s) not in model.pp_shadow_sources]
    if missing:
        raise RuntimeError(f"DeepSeek-V4.1 PP shadow sources were not built: {missing}")
    max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    model.pp_capture_buffers = {}
    for source in plan.capture:
        layer = model.layers[source]
        buffer = torch.empty(
            (max_tokens, config.hidden_size),
            dtype=vllm_config.model_config.dtype,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        layer.attn.pp_capture_buffer = buffer
        model.pp_capture_buffers[source] = buffer


def owner_of(model: nn.Module, checkpoint_name: str):
    """(module, parameter name) of a shadow's parameter for a checkpoint
    tensor ``layers.<s>.attn.<rest>``; the module itself (no parameter name)
    for any other tensor of a shadowed attention (packed EXL3 tensors, which
    the shadow's own matrices load); None otherwise."""
    shadows = getattr(model, "pp_shadow_sources", None)
    if not shadows:
        return None
    match = re.match(r"layers\.(\d+)\.attn\.(.*)", checkpoint_name)
    if match is None or match[1] not in shadows:
        return None
    if match[2].startswith(("wo_a.", "wo_b.")):
        return None
    return shadows[match[1]], match[2]


def run_shadows(model: nn.Module, positions: torch.Tensor, intermediate_tensors) -> None:
    """Replay this stage's shadow sources on their shipped attention inputs
    (ascending), before the stage's own layers."""
    plan = getattr(model, "pp_plan", None)
    if plan is None or not plan.shadows:
        return
    num_tokens = positions.shape[0]
    for source in plan.shadows:
        x = intermediate_tensors[SHIP_KEY.format(source)][:num_tokens]
        model.pp_shadow_sources[str(source)].attend(positions, x)


def add_shipments(model: nn.Module, intermediate_tensors, received, num_tokens: int):
    """Add the source attention inputs later stages need to the outgoing
    intermediate tensors: captured here or relayed from ``received``."""
    plan = getattr(model, "pp_plan", None)
    if plan is None or not plan.send:
        return intermediate_tensors
    for source in plan.send:
        key = SHIP_KEY.format(source)
        if source in model.pp_capture_buffers:
            intermediate_tensors[key] = model.pp_capture_buffers[source][:num_tokens]
        else:
            intermediate_tensors[key] = received[key][:num_tokens]
    return intermediate_tensors


def receive_shapes(model: nn.Module) -> dict[str, int]:
    """Keys of the shipped source inputs this stage receives -> row width."""
    plan = getattr(model, "pp_plan", None)
    if plan is None:
        return {}
    return {SHIP_KEY.format(s): model.config.hidden_size for s in plan.receive}

