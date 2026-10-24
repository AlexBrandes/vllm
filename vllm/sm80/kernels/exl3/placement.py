# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Boot-time whole-expert permutations without changing logical routing IDs.

The plan maps checkpoint expert prefixes to one ordered list of logical expert
IDs per EP rank. Unlisted layers keep their original placement. Local capacity
is unchanged; replicas and runtime migration are deliberately not supported.
Only ownership changes: packed bytes, expert arithmetic and top-k order do not.
The grouping of expert results into rank partials DOES change, so complete MoE
outputs require an order-changing numerical check, not a byte-identity claim.
"""

import hashlib
import json
from functools import cache
from pathlib import Path

from vllm.sm80.kernels.exl3.utils import _unique_object


def validate_plan(plan: dict) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Validate a complete bijection in every listed layer, including capacity."""
    if not isinstance(plan, dict) or set(plan) != {"version", "ep_size", "layers"}:
        raise ValueError("EXL3 placement needs version, ep_size and layers")
    if type(plan["version"]) is not int or plan["version"] != 1:
        raise ValueError("Unsupported EXL3 placement version")
    size = plan["ep_size"]
    if type(size) is not int or size < 2:
        raise ValueError("EXL3 placement requires ep_size >= 2")
    if not isinstance(plan["layers"], dict) or not plan["layers"]:
        raise ValueError("EXL3 placement needs at least one layer")
    layers = {}
    for name, ranks in plan["layers"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid EXL3 placement layer name")
        if not isinstance(ranks, list) or len(ranks) != size:
            raise ValueError(f"Invalid EXL3 placement rank count: {name}")
        if any(not isinstance(ids, list) for ids in ranks):
            raise ValueError(f"Invalid EXL3 placement rank list: {name}")
        ids = [expert for rank in ranks for expert in rank]
        if not ids or any(type(expert) is not int for expert in ids):
            raise ValueError(f"Invalid EXL3 placement expert IDs: {name}")
        if sorted(ids) != list(range(len(ids))):
            raise ValueError(f"EXL3 placement must be a bijection: {name}")
        base, extra = divmod(len(ids), size)
        if [len(rank) for rank in ranks] != [base + (r < extra) for r in range(size)]:
            raise ValueError(f"EXL3 placement must preserve local capacity: {name}")
        layers[name] = tuple(tuple(rank) for rank in ranks)
    return layers


@cache
def load_plan(path: str, sha256: str):
    """Read once per worker; the digest pins identical maps on all ranks."""
    data = Path(path).read_bytes()
    if len(sha256) != 64 or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("EXL3 placement SHA256 mismatch or missing digest")
    plan = json.loads(data, object_pairs_hook=_unique_object)
    layers = validate_plan(plan)
    return plan["ep_size"], layers


def local_map(ranks: tuple[tuple[int, ...], ...], rank: int) -> list[int]:
    result = [-1] * sum(map(len, ranks))
    for local, expert in enumerate(ranks[rank]):
        result[expert] = local
    return result


def apply_placement(layer, config, prefix: str) -> None:
    """Install ownership before packed weights load; no work on the hot path."""
    from vllm import envs

    path = envs.VLLM_EXL3_EXPERT_PLACEMENT
    digest = envs.VLLM_EXL3_EXPERT_PLACEMENT_SHA256
    if not path:
        if digest:
            raise ValueError("EXL3 placement digest requires a placement file")
        return
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
            "EXL3 fixed placement requires whole-expert EP without DP, "
            "EPLB, all-to-all dispatch or fused shared experts"
        )
    # Check checkpoint names once per config, including the draft's layers.
    # Resolve through the layout only for runtime prefixes: checkpoint names
    # must not pass through the HF-to-vLLM mapping a second time.
    if getattr(config, "_checked_expert_placement", None) != (path, digest):
        found = {name: set() for name in layers}
        for key in config.manifest.matrices:
            parts = key.rsplit(".", 2)
            if len(parts) == 3 and parts[0] in found and parts[1].isdigit():
                found[parts[0]].add(int(parts[1]))
        for name, ranks in layers.items():
            if found[name] != set(range(sum(map(len, ranks)))):
                raise ValueError(f"EXL3 placement has no matching checkpoint experts: {name}")
        config._checked_expert_placement = (path, digest)
    source = config.layout.expert_matrix(config, prefix, 0, "gate_proj")
    ranks = layers.get(source.rsplit(".", 2)[0])
    if ranks is None:
        return
    if sum(map(len, ranks)) != layer.global_num_experts:
        raise ValueError("EXL3 placement expert count differs from the layer")
    if len(ranks[parallel.ep_rank]) != layer.local_num_experts:
        raise ValueError("EXL3 placement local capacity differs from the layer")
    if manager.expert_map is None or layer.expert_map is not manager.expert_map:
        raise ValueError("EXL3 placement requires the shared canonical expert map")
    import torch

    # Keep the existing buffer and all of its aliases, including the manager
    # used by the loader and the map later passed to the native expert kernels.
    mapping = local_map(ranks, parallel.ep_rank)
    layer.expert_map.copy_(
        torch.tensor(
            mapping,
            dtype=layer.expert_map.dtype,
            device=layer.expert_map.device,
        )
    )
