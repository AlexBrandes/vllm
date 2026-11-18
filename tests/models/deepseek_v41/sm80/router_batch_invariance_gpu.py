# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch invariance of the DeepSeek-V4.1 router logits on sm_80 (the GLM
32/33-token lesson applied to expert selection): the logits of the same rows
in calls of 1..2048 rows, bitwise, for the served FP32 gate GEMM (cuBLAS,
the default), the FP64 gate GEMV (VLLM_DEEPSEEK_V41_GATE_GEMV, decode sizes)
and the tensor-core gate (VLLM_DEEPSEEK_V41_ROUTER_MMA, 1..2047 rows); plus
how often the top-6 selection of a row differs from the 2048-row call.

    python tests/models/deepseek_v41/sm80/router_batch_invariance_gpu.py
"""

import json
import os
import sys

import torch

SIZES = (1, 2, 4, 7, 8, 16, 31, 32, 33, 64, 128, 256, 512, 1024, 2047, 2048)


def main() -> None:
    from vllm_cmp170hx import plugin

    plugin.register()
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(2048, 5120, device="cuda", generator=gen).bfloat16()
    w = (torch.randn(384, 5120, device="cuda", generator=gen) * 0.02).bfloat16()
    w32 = w.float()
    bias = torch.randn(384, device="cuda", generator=gen) * 0.01

    def fp32_gemm(rows):
        return torch.mm(x[:rows].float(), w32.t())

    paths = {"fp32_gemm": fp32_gemm}
    try:
        from vllm_cmp170hx.models.deepseek_v41.ops import sm80_gate_gemv

        def gemv(rows):
            if not sm80_gate_gemv.serves(x[:rows], w32):
                return None
            return sm80_gate_gemv.gate_gemv(x[:rows], w32)

        paths["gate_gemv"] = gemv
    except Exception as exc:  # noqa: BLE001
        print("gemv unavailable", exc, file=sys.stderr)
    try:
        from vllm_cmp170hx.models.deepseek_v41.ops import sm80_router_mma

        def mma(rows):
            if not sm80_router_mma.serves(x[:rows]):
                return None
            logits, pending = sm80_router_mma.gate(x[:rows], w)
            torch.cuda.synchronize()
            return logits.clone()

        paths["router_mma_gate"] = mma
    except Exception as exc:  # noqa: BLE001
        print("mma unavailable", exc, file=sys.stderr)
    result = {}
    ref_full = fp32_gemm(2048)
    top_ref = (torch.sqrt(torch.nn.functional.softplus(ref_full)) + bias).topk(6, dim=-1)[1].sort(-1)[0]
    for name, fn in paths.items():
        outs = {}
        for rows in SIZES:
            try:
                out = fn(rows)
            except Exception as exc:  # noqa: BLE001
                out = None
                result[f"{name}/error"] = str(exc)[:200]
            if out is not None:
                outs[rows] = out.float()
        if not outs:
            continue
        biggest = max(outs)
        mism, sel = [], []
        for rows, out in outs.items():
            if not torch.equal(out, outs[biggest][:rows]):
                mism.append([rows, float((out - outs[biggest][:rows]).abs().max())])
            top = (torch.sqrt(torch.nn.functional.softplus(out)) + bias).topk(6, dim=-1)[1].sort(-1)[0]
            sel.append([rows, int((top != top_ref[:rows]).any(-1).sum())])
        result[name] = {
            "rows_served": sorted(outs),
            "batch_invariant_vs": biggest,
            "mismatches": mism,
            "rows_with_other_top6_than_fp32_2048": sel,
        }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()
