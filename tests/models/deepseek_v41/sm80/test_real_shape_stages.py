# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build every pipeline stage of DeepSeek-V4.1-Flash-1017 at its real shapes
on the meta device (no memory) from a header-only EXL3 checkpoint, and report
the resident bytes each stage's modules hold (parameters and buffers,
shadow sources included), with the dense matrices served from a header-only
dense W8 sidecar when one is given.

Opt-in (writes sparse files of the real checkpoint's size into
``DSV41_REAL_SHAPE_DIR``, header plus holes):

    DSV41_REAL_SHAPE_DIR=/work/out/real DSV41_PP_PARTITION=16,9,15 \
    DSV41_ROUTED_25=0-6,32-39 [VLLM_EXL3_DENSE_W8=/w8/dense.safetensors] \
    pytest -s tests/models/deepseek_v41/sm80/test_real_shape_stages.py

Routed experts take bits_x2 5 (2.5 bpw) on the ``DSV41_ROUTED_25`` layers and
4 (2.0) elsewhere; the drafter's routed experts 5. Prints one JSON line per
stage; asserts only that every stage builds and loads its plan.
"""

import json
import os
import struct
from pathlib import Path

import pytest
import torch

from tests.models.deepseek_v41.sm80 import construct
from tests.models.deepseek_v41.sm80 import exl3_fixture as fixture
from tests.models.test_native_checkpoint_config_meta import (
    meta_platform,  # noqa: F401
)

ROOT = os.environ.get("DSV41_REAL_SHAPE_DIR")
pytestmark = pytest.mark.skipif(not ROOT, reason="set DSV41_REAL_SHAPE_DIR")


def _ranges(spec: str) -> set[int]:
    out = set()
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def _write_sparse(directory: Path, config: dict, routed_25: set[int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    tensors = fixture.checkpoint_tensors(config)
    header, offset = {}, 0
    for name, (dtype, shape) in sorted(tensors.items()):
        if ".ffn.experts." in name and name.endswith(".trellis"):
            layer = int(name.split(".")[1])
            bits_x2 = 5 if name.startswith("mtp.") or layer in routed_25 else 4
            shape = (*shape[:-1], 8 * bits_x2)
        size = fixture.tensor_bytes(dtype, tuple(shape))
        header[name] = {
            "dtype": {"I16": "I16", "I32": "I32", "F16": "F16", "BF16": "BF16",
                      "F32": "F32"}[dtype],
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path = directory / "model-00001-of-00001.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.truncate(8 + len(raw) + offset)
        # The codebook markers are read as values (mul1 scalars).
        marker = struct.pack("<I", fixture.MUL1)
        for name, entry in header.items():
            if name.endswith(".mul1"):
                os.pwrite(f.fileno(), marker, 8 + len(raw) + entry["data_offsets"][0])


def _bytes(module: torch.nn.Module) -> int:
    seen, total = set(), 0
    for t in list(module.parameters()) + list(module.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total


def _by_kind(model: torch.nn.Module) -> dict[str, float]:
    """Resident GB by a coarse tensor kind (routed experts, other layer
    tensors, embedding/head/vision, buffers), deduplicated."""
    import re

    out: dict[str, float] = {}
    seen = set()
    named = list(model.named_parameters()) + [
        (f"buffer:{n}", b) for n, b in model.named_buffers()
    ]
    for name, t in named:
        if id(t) in seen:
            continue
        seen.add(id(t))
        if name.startswith("buffer:"):
            kind = "buffer:" + re.sub(r"\.\d+\.", ".N.", name[7:]).split(".")[-1]
        elif ".exl3_experts." in name:
            kind = "routed"
        elif ".pp_shadow_sources." in name:
            kind = "shadow"
        elif ".layers." in name:
            kind = "layer-other"
        else:
            kind = ".".join(name.split(".")[:2])
        out[kind] = out.get(kind, 0) + t.numel() * t.element_size() / 1e9
    return {k: round(v, 3) for k, v in sorted(out.items(), key=lambda x: -x[1])[:12]}


def test_real_shape_stage_bytes(monkeypatch, meta_platform):  # noqa: F811
    from vllm_cmp170hx import plugin

    plugin._register_exl3()
    config = json.loads((fixture.FIXTURES / "config.json").read_text())
    config["engram_table_dir"] = "/nonexistent"
    directory = Path(ROOT)
    routed_25 = _ranges(os.environ.get("DSV41_ROUTED_25", "0-6,32-39"))
    if not (directory / "config.json").exists():
        _write_sparse(directory, config, routed_25)
    partition = os.environ.get("DSV41_PP_PARTITION", "16,9,15")
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    stages = len(partition.split(","))
    for stage in range(stages):
        construct.set_groups(monkeypatch, 1, 0, pp_size=stages, pp_rank=stage)
        construct.stub_engram_tables(monkeypatch, 1)
        target_config, draft_config = construct.vllm_configs(
            str(directory), 1, pp_size=stages
        )
        with torch.device("meta"):
            target = construct.build_model(target_config)
            draft = (
                construct.build_model(draft_config) if stage == stages - 1 else None
            )
        backbone = target.language_model.model
        shadows = {
            name: _bytes(module)
            for name, module in getattr(backbone, "pp_shadow_sources", {}).items()
        }
        print(
            json.dumps(
                {
                    "stage": stage,
                    "layers": [backbone.start_layer, backbone.end_layer - 1],
                    "target_gb": round(_bytes(target) / 1e9, 3),
                    "draft_gb": round(_bytes(draft) / 1e9, 3) if draft else 0,
                    "shadow_gb": {k: round(v / 1e9, 3) for k, v in shadows.items()},
                    "w8": bool(os.environ.get("VLLM_EXL3_DENSE_W8")),
                    "buffers_gb": round(
                        sum(b.numel() * b.element_size()
                            for b in {id(b): b for b in target.buffers()}.values())
                        / 1e9, 3,
                    ),
                    "by_kind_gb": _by_kind(target),
                }
            ),
            flush=True,
        )
