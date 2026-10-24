# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated hot routed experts under whole-expert EP (VLLM_EXL3_EXPERT_REPLICAS).

A routed-expert MoE call costs each rank about one weight read per distinct
local expert it serves, and every rank waits for the busiest one at the next
all-reduce. With a plan, every rank also holds copies of a few hot experts
per layer. Each MoE call of 1..MAX_ROWS rows then picks one owner rank per hot
expert it uses, and only that rank computes that expert's assignments.

The plan, ``{"version": 1, "ep_size": 4, "layers": {"layers.1.ffn.experts":
[e0, e1, ...]}}``, lists each layer's replicated experts by **checkpoint**
expert prefix, hottest first. Its SHA256 (VLLM_EXL3_EXPERT_REPLICAS_SHA256)
pins the same plan on every rank. Unlisted layers keep ordinary placement.

Owner rule, per call, from the call's top-k expert ids alone:

1. Count each rank's distinct used experts that are not replicated (each
   one served by its home rank).
2. The target is the larger of the busiest count from step 1 and the used
   distinct experts divided evenly over the ranks (rounded up). No placement
   can do better.
3. Take the replicated experts in plan order. Each one the call uses stays on
   its home rank while that rank's count is below the target; otherwise it
   goes to the rank with the fewest distinct experts so far (ties: the lowest
   rank). Either way that rank's count grows by one.

Every rank computes the same owners: the top-k ids are identical on every rank
(EP already requires that), and so are the plan and the home ranks. A
replicated expert leaves its home rank only when that rank is already at the
target, so a call whose used experts are already balanced keeps ordinary
placement, and the busiest rank never serves more distinct experts than the
target or than under ordinary placement. Each assignment is computed by
exactly one rank. A call of more rows keeps ordinary placement
(``expert_map`` unchanged).

The rule is applied by writing a per-call expert map: home experts map to
their local slot, the replicated experts this rank owns for the call map to
their slot (the home slot on their home rank, else a replica slot after the
home slots), and every other expert maps to -1. Router ids, weights and top-k
order do not change, and nor does any expert's arithmetic. **Which rank's
partial sum carries an expert does change**, so the MoE output after the
all-reduce is not bit-identical to ordinary placement whenever an expert
moves (an order-changing numerical change).

Replica slots add resident weight memory: one whole expert (gate, up and
down) per replicated expert that is not already local, per layer
(:func:`replica_bytes`).
"""

import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm.sm80.kernels.exl3.utils import _unique_object

logger = init_logger(__name__)

# Replicas serve MoE calls of 1..MAX_ROWS rows (decode, verify and draft
# graphs); larger calls keep ordinary placement. MAX_ROWS * top-k must fit
# BLOCK_ASSIGNMENTS (the kernel reads every assignment in one block).
MAX_ROWS = 64
BLOCK_ASSIGNMENTS = 512
MAX_HOT = 64


def validate_plan(plan: dict) -> tuple[int, dict[str, tuple[int, ...]]]:
    """The plan's EP size and, per checkpoint expert prefix, its replicated
    experts in plan order (distinct, nonnegative, at most MAX_HOT)."""
    if not isinstance(plan, dict) or set(plan) != {"version", "ep_size", "layers"}:
        raise ValueError("EXL3 expert replicas need version, ep_size and layers")
    if type(plan["version"]) is not int or plan["version"] != 1:
        raise ValueError("Unsupported EXL3 expert replica plan version")
    size = plan["ep_size"]
    if type(size) is not int or size < 2 or size & (size - 1):
        raise ValueError("EXL3 expert replicas require a power-of-two ep_size >= 2")
    if not isinstance(plan["layers"], dict) or not plan["layers"]:
        raise ValueError("EXL3 expert replicas need at least one layer")
    layers = {}
    for name, hot in plan["layers"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid EXL3 expert replica layer name")
        if (
            not isinstance(hot, list)
            or not 0 < len(hot) <= MAX_HOT
            or any(type(e) is not int or e < 0 for e in hot)
            or len(set(hot)) != len(hot)
        ):
            raise ValueError(f"Invalid EXL3 replicated expert list: {name}")
        layers[name] = tuple(hot)
    return size, layers


@cache
def load_plan(path: str, sha256: str) -> tuple[int, dict[str, tuple[int, ...]]]:
    """Read once per worker; the digest pins the same plan on every rank."""
    data = Path(path).read_bytes()
    if len(sha256) != 64 or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("EXL3 expert replica plan SHA256 mismatch or missing digest")
    return validate_plan(json.loads(data, object_pairs_hook=_unique_object))


@dataclass(frozen=True)
class LayerReplicas:
    """One MoE layer's replicas on one rank."""

    rank: int
    ep_size: int
    # Replicated experts in plan order.
    hot: tuple[int, ...]
    # Home rank of every global expert.
    home: tuple[int, ...]
    # Local slot of every global expert on this rank: its home slot, or for a
    # replicated expert held here as a copy, home_slots + its index in
    # ``copies``; -1 elsewhere.
    local: tuple[int, ...]
    # Replicated experts this rank holds as copies, in slot order.
    copies: tuple[int, ...]
    home_slots: int


def layer_replicas(
    hot: tuple[int, ...], home: tuple[int, ...], rank: int, ep_size: int
) -> LayerReplicas:
    """The replica layout of one layer on ``rank``: copies of the replicated
    experts whose home is another rank, after the home slots in plan order."""
    if any(e >= len(home) for e in hot):
        raise ValueError("EXL3 replicated expert id out of range")
    local = [-1] * len(home)
    slots = 0
    for expert, owner in enumerate(home):
        if owner == rank:
            local[expert] = slots
            slots += 1
    copies = tuple(e for e in hot if home[e] != rank)
    for index, expert in enumerate(copies):
        local[expert] = slots + index
    return LayerReplicas(rank, ep_size, tuple(hot), tuple(home), tuple(local), copies, slots)


def home_ranks(ep_size: int, global_experts: int) -> tuple[int, ...]:
    """Home rank of every expert under the linear EP placement."""
    from vllm.model_executor.layers.fused_moe.expert_map_manager import (
        determine_expert_map,
    )

    home = [-1] * global_experts
    for rank in range(ep_size):
        _, mapping, _ = determine_expert_map(ep_size, rank, global_experts)
        for expert in torch.nonzero(mapping >= 0).flatten().tolist():
            home[expert] = rank
    if min(home) < 0:
        raise ValueError("EXL3 expert replicas: an expert has no home rank")
    return tuple(home)


def plan_layer(layer, config, prefix: str) -> LayerReplicas | None:
    """This layer's replicas from VLLM_EXL3_EXPERT_REPLICAS, or None (off, or
    the layer is not in the plan). Called before the packed weights load."""
    from vllm import envs

    path = envs.VLLM_EXL3_EXPERT_REPLICAS
    digest = envs.VLLM_EXL3_EXPERT_REPLICAS_SHA256
    if not path:
        if digest:
            raise ValueError("EXL3 expert replica digest requires a plan file")
        return None
    if envs.VLLM_EXL3_EXPERT_PLACEMENT:
        raise ValueError(
            "VLLM_EXL3_EXPERT_REPLICAS and VLLM_EXL3_EXPERT_PLACEMENT cannot be "
            "combined: replicas assume the linear EP placement"
        )
    size, layers = load_plan(path, digest)
    parallel = layer.moe_config.moe_parallel_config
    manager = layer.expert_map_manager
    if (
        parallel.ep_size != size
        or parallel.tp_size != 1
        or parallel.dp_size != 1
        or parallel.enable_eplb
        or parallel.use_all2all_kernels
        or manager.num_fused_shared_experts
        or manager.expert_mask is not None
        or manager.routing_tables is not None
    ):
        raise ValueError(
            "EXL3 expert replicas require whole-expert EP without DP, EPLB, "
            "all-to-all dispatch or fused shared experts"
        )
    if getattr(config, "_checked_expert_replicas", None) != (path, digest):
        found = {name: set() for name in layers}
        for key in config.manifest.matrices:
            parts = key.rsplit(".", 2)
            if len(parts) == 3 and parts[0] in found and parts[1].isdigit():
                found[parts[0]].add(int(parts[1]))
        for name, hot in layers.items():
            if not found[name] or any(e not in found[name] for e in hot):
                raise ValueError(f"EXL3 expert replicas name no checkpoint experts: {name}")
        config._checked_expert_replicas = (path, digest)
    source = config.layout.expert_matrix(config, prefix, 0, "gate_proj")
    hot = layers.get(source.rsplit(".", 2)[0])
    if hot is None:
        return None
    if manager.expert_map is None:
        raise ValueError("EXL3 expert replicas require the canonical expert map")
    home = home_ranks(size, layer.global_num_experts)
    replicas = layer_replicas(hot, home, parallel.ep_rank, size)
    expected = [s if home[e] == parallel.ep_rank else -1 for e, s in enumerate(replicas.local)]
    if manager.expert_map.tolist() != expected:
        raise ValueError("EXL3 expert replicas require the linear EP placement")
    return replicas


def replica_bytes(replicas: LayerReplicas, expert_bytes: dict[int, int]) -> int:
    """Resident bytes the copies add on this rank (``expert_bytes``: the
    packed bytes of each global expert's three matrices)."""
    return sum(expert_bytes[e] for e in replicas.copies)


def assign_owners(ids: torch.Tensor, replicas: LayerReplicas) -> list[int]:
    """The owner rank of each replicated expert for one call (plan order; an
    unused one keeps its home rank). Reference for the kernel."""
    used = {int(e) for e in ids.flatten().tolist() if 0 <= e < len(replicas.home)}
    hot = set(replicas.hot)
    load = [0] * replicas.ep_size
    for expert in used - hot:
        load[replicas.home[expert]] += 1
    target = max(max(load), -(-len(used) // replicas.ep_size))
    owners = []
    for expert in replicas.hot:
        home = replicas.home[expert]
        if expert not in used:
            owners.append(home)
            continue
        if load[home] < target:
            owner = home
        else:
            owner = min(range(replicas.ep_size), key=lambda r: (load[r], r))
        load[owner] += 1
        owners.append(owner)
    return owners


def expert_map_reference(ids: torch.Tensor, replicas: LayerReplicas) -> torch.Tensor:
    """This rank's expert map for one call (int32, CPU). Reference for the
    kernel."""
    out = torch.tensor(replicas.local, dtype=torch.int32)
    for expert, owner in zip(replicas.hot, assign_owners(ids, replicas)):
        if owner != replicas.rank:
            out[expert] = -1
    return out


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    from vllm.triton_utils import tl, triton

    @triton.jit
    def replica_map_kernel(
        ids_ptr,
        n_ids,
        home_ptr,
        hot_of_ptr,
        hot_ptr,
        local_ptr,
        seen_ptr,
        owner_ptr,
        out_ptr,
        rank,
        n_experts,
        n_hot,
        BLOCK_E: tl.constexpr,
        BLOCK_A: tl.constexpr,
        RANKS: tl.constexpr,
    ):
        e = tl.arange(0, BLOCK_E)
        in_e = e < n_experts
        tl.store(seen_ptr + e, tl.zeros([BLOCK_E], tl.int32), mask=in_e)
        tl.debug_barrier()
        a = tl.arange(0, BLOCK_A)
        ids = tl.load(ids_ptr + a, mask=a < n_ids, other=-1).to(tl.int32)
        valid = (a < n_ids) & (ids >= 0) & (ids < n_experts)
        tl.store(seen_ptr + ids, tl.full([BLOCK_A], 1, tl.int32), mask=valid)
        tl.debug_barrier()
        seen = tl.load(seen_ptr + e, mask=in_e, other=0)
        home = tl.load(home_ptr + e, mask=in_e, other=0)
        hot_of = tl.load(hot_of_ptr + e, mask=in_e, other=-1)
        r = tl.arange(0, RANKS)
        cold = (seen > 0) & (hot_of < 0)
        load = tl.sum((cold[:, None] & (home[:, None] == r[None, :])).to(tl.int32), axis=0)
        used_total = tl.sum((seen > 0).to(tl.int32), axis=0)
        target = tl.maximum(tl.max(load, axis=0), (used_total + RANKS - 1) // RANKS)
        for j in range(0, n_hot):
            expert = tl.load(hot_ptr + j)
            used = tl.load(seen_ptr + expert)
            expert_home = tl.load(home_ptr + expert)
            home_load = tl.sum(tl.where(r == expert_home, load, 0), axis=0)
            # Fewest experts, then the lowest rank.
            fewest = tl.min(load * RANKS + r, axis=0) % RANKS
            owner = tl.where(home_load < target, expert_home, fewest)
            owner = tl.where(used > 0, owner, expert_home)
            load += ((r == owner) & (used > 0)).to(tl.int32)
            tl.store(owner_ptr + j, owner)
        tl.debug_barrier()
        is_hot = in_e & (hot_of >= 0)
        owner_e = tl.load(owner_ptr + tl.maximum(hot_of, 0), mask=is_hot, other=rank)
        local = tl.load(local_ptr + e, mask=in_e, other=-1)
        out = tl.where(is_hot & (owner_e != rank), -1, local)
        tl.store(out_ptr + e, out.to(out_ptr.dtype.element_ty), mask=in_e)

    _KERNEL = replica_map_kernel
    return _KERNEL


class ReplicaMap:
    """Device tables of one layer and its per-call expert map buffer."""

    def __init__(self, replicas: LayerReplicas, device, dtype: torch.dtype):
        experts = len(replicas.home)
        hot_of = [-1] * experts
        for index, expert in enumerate(replicas.hot):
            hot_of[expert] = index

        def table(values):
            return torch.tensor(values, dtype=torch.int32, device=device)

        self.replicas = replicas
        self.home = table(replicas.home)
        self.hot_of = table(hot_of)
        self.hot = table(replicas.hot)
        self.local = table(replicas.local)
        self.seen = torch.zeros(experts, dtype=torch.int32, device=device)
        self.owner = torch.zeros(len(replicas.hot), dtype=torch.int32, device=device)
        self.out = torch.empty(experts, dtype=dtype, device=device)
        self.block_e = triton_next_power_of_2(experts)

    def __call__(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """This rank's expert map for the call with ``topk_ids`` (written into
        the layer's own buffer, in stream order; graph-safe)."""
        n_ids = topk_ids.numel()
        if n_ids > BLOCK_ASSIGNMENTS or not topk_ids.is_contiguous():
            raise ValueError("EXL3 expert replicas: call too large or ids not contiguous")
        _kernel()[(1,)](
            topk_ids,
            n_ids,
            self.home,
            self.hot_of,
            self.hot,
            self.local,
            self.seen,
            self.owner,
            self.out,
            self.replicas.rank,
            len(self.replicas.home),
            len(self.replicas.hot),
            BLOCK_E=self.block_e,
            BLOCK_A=BLOCK_ASSIGNMENTS,
            RANKS=self.replicas.ep_size,
            num_warps=4,
        )
        return self.out


def triton_next_power_of_2(n: int) -> int:
    return 1 << max(0, n - 1).bit_length()


def log_layer(prefix: str, replicas: LayerReplicas, added_bytes: int) -> None:
    logger.info(
        "EXL3 expert replicas (VLLM_EXL3_EXPERT_REPLICAS): %s rank %d holds "
        "%d of %d replicated experts as copies (+%.1f MiB); calls of up to %d "
        "rows pick one owner rank per replicated expert",
        prefix,
        replicas.rank,
        len(replicas.copies),
        len(replicas.hot),
        added_bytes / 2**20,
        MAX_ROWS,
    )
