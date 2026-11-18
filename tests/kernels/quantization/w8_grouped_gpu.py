# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU check: the grouped W8 projection (one activation quantisation for all
groups, dense_w8._w8_grouped) equals per-group exl3_w8_linear calls bit for
bit, at DeepSeek-V4.1 wo_a geometry (8 groups, K 4096, N 1024), for decode
and prefill row counts; and times both. Prints one JSON line.

    VLLM_EXL3_DENSE_W8_DECODE=1 python tests/kernels/quantization/w8_grouped_gpu.py
"""

import json
import time

import torch


def main() -> None:
    from vllm_cmp170hx import plugin
    from vllm_cmp170hx.kernels.exl3 import dense_w8

    plugin.register()
    dense_w8.load_libs()
    g, k, n = 8, 4096, 1024
    gen = torch.Generator(device="cuda").manual_seed(0)
    w8s = [torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda", generator=gen) for _ in range(g)]
    sws = [torch.rand(n, device="cuda", generator=gen) * 1e-3 for _ in range(g)]
    suh = (torch.randint(0, 2, (k,), device="cuda", generator=gen) * 2 - 1).half()
    result = {}
    for t in (1, 2, 3, 4, 8, 16, 17, 32, 64, 128, 129, 512, 2048):
        x = torch.randn(t, g, k, device="cuda", generator=gen).bfloat16()
        ref = torch.cat(
            [torch.ops.vllm.exl3_w8_linear(x[:, i].contiguous(), w8s[i], sws[i], suh) for i in range(g)],
            dim=-1,
        )
        got = torch.ops.vllm.exl3_w8_grouped(x, w8s, sws, suh)
        equal = torch.equal(ref, got)
        times = []
        for fn in (
            lambda: torch.cat([torch.ops.vllm.exl3_w8_linear(x[:, i].contiguous(), w8s[i], sws[i], suh) for i in range(g)], dim=-1),
            lambda: torch.ops.vllm.exl3_w8_grouped(x, w8s, sws, suh),
        ):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / 20 * 1e6)
        result[t] = {"bitwise_equal": equal, "per_group_us": round(times[0], 1), "grouped_us": round(times[1], 1)}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
