# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance of the DeepSeek-V4.1 sm_80 mHC paths (the GLM-5.3-Flash
"32 vs 33 tokens" lesson: a reduction whose order depends on the call's
row count gave different logprobs for the same token in prefill chunks of
different sizes, up to 2.4 nats downstream).

For each path (torch reference formulas ``ops/sm80_mhc.py``; kernel A + torch
finish; kernel A + kernel B; kernel A + finish v2) and each op
(``mhc_pre_delayed``, ``mhc_fused_post_pre_delayed``), the first rows of a
call of ``n`` rows must equal, bit for bit, the same rows computed in calls
of other sizes (1..2048, across the decode/prefill thresholds), and the
paths are compared with each other (max abs difference, not bitwise).

Run in a GPU container (one card); prints one JSON line:

    python tests/models/deepseek_v41/sm80/mhc_batch_invariance_gpu.py
"""

import importlib
import json
import os
import sys

import torch

SIZES = (1, 2, 7, 8, 31, 32, 33, 63, 64, 65, 128, 511, 512, 513, 2048)
HC, HIDDEN, MIX = 4, 5120, (2 + 4) * 4


def _inputs(rows: int, generator: torch.Generator):
    dev = "cuda"
    residual = (torch.randn(rows, HC, HIDDEN, generator=generator) * 0.5).to(dev, torch.bfloat16)
    x = (torch.randn(rows, HIDDEN, generator=generator) * 0.5).to(dev, torch.bfloat16)
    fn = (torch.randn(MIX, HC * HIDDEN, generator=generator) * 0.02).to(dev)
    scale = torch.tensor([0.5, 0.7, 0.9], device=dev)
    base = (torch.randn(MIX, generator=generator) * 0.1).to(dev)
    pre_mix = torch.softmax(torch.randn(rows, HC, generator=generator), -1).to(dev)
    post = torch.rand(rows, HC, 1, generator=generator).to(dev) * 2
    comb = torch.softmax(torch.randn(rows, HC, HC, generator=generator), -1).to(dev)
    norm = (1 + 0.1 * torch.randn(HIDDEN, generator=generator)).to(dev, torch.bfloat16)
    return residual, x, fn, scale, base, pre_mix, post, comb, norm


def _mixes(fn, scale, base):
    # (fn, scale, base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, post_mult, iters)
    return fn, scale, base, 1e-20, 1e-6, 1e-6, 2.0, 20


def _run(module, op, data, rows):
    residual, x, fn, scale, base, pre_mix, post, comb, norm = data
    args = _mixes(fn, scale, base)
    if op == "pre":
        out = module.mhc_pre_delayed(
            residual[:rows], *args, pre_mix=pre_mix[:rows], norm_weight=norm, norm_eps=1e-20
        )
    else:
        out = module.mhc_fused_post_pre_delayed(
            x[:rows], residual[:rows], post[:rows], comb[:rows], *args,
            pre_mix=pre_mix[:rows], norm_weight=norm, norm_eps=1e-20,
        )
    return [t for t in out if isinstance(t, torch.Tensor) and t.numel()]


def _paths():
    """(name, module) for each mHC path, the fused module reloaded per
    finish switch (it reads them at import)."""
    from vllm_cmp170hx.models.deepseek_v41.ops import sm80_mhc

    yield "torch", sm80_mhc
    for name, finish, v2 in (
        ("kernelA+torch_finish", "0", "0"),
        ("kernelA+kernelB", "1", "0"),
        ("kernelA+finish_v2", "1", "1"),
    ):
        os.environ["VLLM_DEEPSEEK_V41_FUSED_MHC"] = "1"
        os.environ["VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH"] = finish
        os.environ["VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2"] = v2
        import vllm_cmp170hx.models.deepseek_v41.ops.sm80_mhc_fused as fused

        yield name, importlib.reload(fused)


def main() -> None:
    from vllm_cmp170hx import plugin

    plugin.register()
    generator = torch.Generator().manual_seed(0)
    data = _inputs(max(SIZES), generator)
    result = {}
    reference = {}
    for name, module in _paths():
        for op in ("pre", "post_pre"):
            try:
                full = [t.clone() for t in _run(module, op, data, max(SIZES))]
            except Exception as exc:  # noqa: BLE001
                result[f"{name}/{op}"] = f"error: {type(exc).__name__}: {exc}"[:300]
                continue
            mismatches = []
            for rows in SIZES:
                part = _run(module, op, data, rows)
                for i, (a, b) in enumerate(zip(part, full)):
                    if not torch.equal(a, b[:rows]):
                        diff = (a.float() - b[:rows].float()).abs().max().item()
                        mismatches.append([rows, i, diff])
            entry = {"batch_invariant": not mismatches, "mismatches": mismatches[:8]}
            if op in reference:
                entry["max_abs_vs_torch"] = [
                    (a.float() - b.float()).abs().max().item()
                    for a, b in zip(full, reference[op])
                ]
            else:
                reference[op] = full
            result[f"{name}/{op}"] = entry
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()
