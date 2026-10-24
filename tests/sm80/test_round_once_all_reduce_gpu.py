# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP4 bits and time of the grouped round-once all-reduce (skip with fewer
than 4 GPUs).

Runs round_once_allreduce_bits.py on four ranks (torchrun) under the served
TP4 all-reduce settings and judges its JSON: on every row count and hidden
size the grouped all-reduce gives the correctly rounded exact sum on every
element and every rank; NCCL's BF16 ring misses it somewhere on prefill-sized
messages (the reason the path exists). The per-call times (grouped, BF16 ring,
FP32 ring) are printed for the record.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).with_name("round_once_allreduce_bits.py")
TREE = SCRIPT.parents[2]
TIMEOUT_S = 1200
SERVED_ENV = {
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "NCCL_P2P_LEVEL": "SYS",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
    "VLLM_CUSTOM_ALLREDUCE_ALGO": "2stage",
    "VLLM_CUSTOM_ALLREDUCE_FLAGS": "1",
    "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P": "1",
    "VLLM_HOST_ALLREDUCE": "1",
}

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        reason="needs 4 GPUs",
    ),
]


def test_grouped_round_once_bits_tp4(tmp_path):
    out = tmp_path / "round_once.json"
    env = dict(os.environ, **SERVED_ENV)
    env["VLLM_CACHE_ROOT"] = str(tmp_path / "vllm-cache")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(TREE), env.get("PYTHONPATH", "")) if p
    )
    argv = [
        sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node", "4", str(SCRIPT), "--out", str(out),
    ]  # fmt: skip
    proc = subprocess.run(
        argv, cwd=TREE, env=env, capture_output=True, text=True, timeout=TIMEOUT_S
    )
    log = (proc.stdout + proc.stderr)[-8000:]
    assert proc.returncode == 0, f"4-rank run exited {proc.returncode}:\n{log}"
    print(proc.stdout)
    data = json.loads(out.read_text())
    assert data["world"] == 4
    assert {c["rank"] for c in data["cases"]} == set(range(4))
    for c in data["cases"]:
        where = f"hidden {c['hidden']} rows {c['rows']} seed {c['seed']}"
        assert c["grouped_ne_exact"] == 0, f"{where}: not the correct rounding"
        assert c["ranks_agree"], f"{where}: ranks hold different bits"
        if c["seed"] == 0:
            assert math.isfinite(c["us_validation_overhead"])
            assert c["us_grouped_unchecked"] > 0
    assert data["equal"] is True
    large = [c for c in data["cases"] if c["rows"] >= 820]
    assert any(c["ring_ne_exact"] > 0 for c in large), "the ring rounded once?"


@pytest.mark.parametrize("invalid", ["span", "nonfinite"])
def test_unsupported_input_fails_normally_on_every_rank(tmp_path, invalid):
    out = tmp_path / "refusal.json"
    env = dict(os.environ, **SERVED_ENV, PYTHONPATH=str(TREE))
    proc = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                           "--nproc-per-node", "4", str(SCRIPT), "--out", str(out),
                           "--invalid", invalid], cwd=TREE, env=env, capture_output=True,
                          text=True, timeout=TIMEOUT_S)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-8000:]
    data = json.loads(out.read_text())
    assert {r["rank"] for r in data["records"]} == set(range(4))
    assert all(r["refused"] and r["cuda_context_healthy"] for r in data["records"])
