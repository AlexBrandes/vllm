#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pick a DeepSeek-V4.1 pipeline partition (VLLM_PP_LAYER_PARTITION) from the
real bytes of a converted EXL3 checkpoint and the dense W8 sidecar.

Reads only safetensors headers. Per stage it counts the resident weights the
plugin loads on that stage: its decoder layers (routed experts as stored;
dense matrices from the sidecar when one is given, else as stored), the
first stage's embedding, vision tower and aligner, the last stage's final
norm/head and the DSpark drafter (``mtp.*`` plus its own embedding copy under
PP), and the attention of every shadow source the stage replays
(models/deepseek_v41/sm80_pp_shadow.py). Engram tables stay on SSD.

    python3 tools/dsv41_pp_plan.py --model DIR [--w8 dense.safetensors] \
        --budgets-gb 63.5,37.5,63.5 [--top 10]

``--budgets-gb`` is the weight budget per stage in card order (visible memory
minus activations, graphs, KV and the Q4N-X startup allocation, measured at
the first boot). Prints the feasible partitions by smallest worst slack.
"""

import argparse
import json
import re
import struct
from collections import defaultdict
from pathlib import Path


def headers(path: Path) -> dict[str, int]:
    """tensor name -> stored bytes."""
    out = {}
    files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
    for file in files:
        with open(file, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            lo, hi = meta["data_offsets"]
            out[name] = hi - lo
    return out


def sources_read_by(cfg: dict, layer: int) -> set[int]:
    ratios = cfg.get("compress_ratios") or []
    if layer >= cfg["num_hidden_layers"] or layer >= len(ratios) or not ratios[layer]:
        return set()
    kv, index = cfg["kv_source_layer_ids"], cfg["index_source_layer_ids"]
    cand = cfg.get("candidate_source_layer_id", -1)
    out = {max(s for s in kv if s <= layer), max(s for s in index if s <= layer)}
    if layer in index and 0 <= cand < layer:
        out.add(cand)
    out.discard(layer)
    return out


def shadows(cfg: dict, start: int, end: int) -> set[int]:
    need = set()
    for layer in range(start, end):
        need |= sources_read_by(cfg, layer)
    stack = [s for s in need if s < start]
    while stack:
        for s in sources_read_by(cfg, stack.pop()):
            if s < start and s not in need:
                need.add(s)
                stack.append(s)
    return {s for s in need if s < start}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--w8", type=Path)
    ap.add_argument("--budgets-gb", required=True)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()
    config = json.loads((args.model / "config.json").read_text())
    cfg = config.get("text_config", config)
    layers = cfg["num_hidden_layers"]
    budgets = [float(x) * 1e9 for x in args.budgets_gb.split(",")]
    stored = headers(args.model)
    w8 = headers(args.w8) if args.w8 else {}
    covered = {n.rsplit(".", 1)[0] for n in w8 if n.endswith(".w8")}
    layer_bytes, attn_bytes = defaultdict(int), defaultdict(int)
    first = last = 0
    for name, size in stored.items():
        prefix = name.rsplit(".", 1)[0]
        if prefix in covered:
            continue  # served from the sidecar (counted below)
        m = re.match(r"layers\.(\d+)\.", name)
        if m:
            layer_bytes[int(m[1])] += size
            if ".attn." in name:
                attn_bytes[int(m[1])] += size
        elif name.startswith(("embed.", "vision.", "aligner.", "image_")):
            first += size
            if name.startswith("embed."):
                last += size  # the drafter's own embedding copy under PP
        else:  # head, norm, hc_head, mtp.*
            last += size
    for name, size in w8.items():
        prefix = name.rsplit(".", 1)[0]
        m = re.match(r"layers\.(\d+)\.", name)
        if m:
            layer_bytes[int(m[1])] += size
            if ".attn." in name:
                attn_bytes[int(m[1])] += size
        else:
            last += size
    stages = len(budgets)
    results = []

    def walk(prefix: list[int], left: int):
        if len(prefix) == stages - 1:
            if left > 0:
                parts = prefix + [left]
                start, loads = 0, []
                for i, n in enumerate(parts):
                    b = sum(layer_bytes[j] for j in range(start, start + n))
                    b += sum(attn_bytes[s] for s in shadows(cfg, start, start + n))
                    b += first if i == 0 else 0
                    b += last if i == stages - 1 else 0
                    loads.append(b)
                    start += n
                slack = [bud - ld for bud, ld in zip(budgets, loads)]
                results.append((min(slack), parts, loads, slack))
            return
        for n in range(1, left - (stages - 1 - len(prefix)) + 1):
            walk(prefix + [n], left - n)

    walk([], layers)
    results.sort(key=lambda r: -r[0])
    feasible = [r for r in results if r[0] >= 0]
    print(f"{len(feasible)} feasible of {len(results)} partitions")
    for worst, parts, loads, slack in (feasible or results)[: args.top]:
        print(
            ",".join(map(str, parts)),
            "loads GB", [round(x / 1e9, 2) for x in loads],
            "slack GB", [round(x / 1e9, 2) for x in slack],
        )


if __name__ == "__main__":
    main()
