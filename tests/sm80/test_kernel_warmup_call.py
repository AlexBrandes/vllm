# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""kernel_warmup.py (upstream-owned) makes exactly one call into the sm80
package, vllm/sm80/kernels/warmup.py, which warms every sm_80 kernel family
before capture."""

import ast
from pathlib import Path
from types import SimpleNamespace

from vllm.model_executor.warmup import kernel_warmup as kw
from vllm.sm80.kernels import warmup as sm80_warmup


def test_one_call_into_the_sm80_package():
    tree = ast.parse(Path(kw.__file__).read_text())
    imports = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("vllm.sm80")
    ]
    assert [(n.module, [a.name for a in n.names]) for n in imports] == [
        ("vllm.sm80.kernels.warmup", ["warmup_from_worker"])
    ]
    src = Path(kw.__file__).read_text()
    assert src.count("sm80_warmup(") == 1
    assert "VLLM_AMPERE_" not in src and "VLLM_SM80_" not in src


def test_kernel_warmup_hands_the_capture_sizes_over(monkeypatch):
    seen = []
    monkeypatch.setattr(sm80_warmup, "warmup_from_worker",
                        lambda worker, sizes: seen.append((worker, sizes)))
    for name in ("qwen_triton_warmup", "qwen_vl_triton_warmup",
                 "mamba_triton_warmup"):
        monkeypatch.setattr(kw, name, lambda *a, **k: None)
    worker = SimpleNamespace(
        use_v2_model_runner=True,
        model_runner=SimpleNamespace(),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(enable_jit_warmup=False),
            model_config=None,
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 2])))
    try:
        kw.kernel_warmup(worker)
    except Exception:  # later upstream warmups need a real worker
        pass
    assert seen == [(worker, [1, 2])]
