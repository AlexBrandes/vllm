# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-level lab screen of packed KV records (``sm80_kv_pack``) against
BF16 rows at the served DeepSeek-V4.1 shapes (TP=1: 64 heads of 512, index
32 x 128, window 128, index top-k 512): the three readers and writers the
attention layer calls, each timed with CUDA events (median of ``--repeats``
after a warm call), BF16 and packed caches holding the same values. Lab
numbers rank only; serving numbers come from benchlib.

    python tests/models/deepseek_v41/sm80/kv_pack_bench_gpu.py [--contexts 16384,65536,131072]
"""

import argparse
import json

import torch

from vllm_cmp170hx.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_cmp170hx.models.deepseek_v41.ops import sm80_kv_pack as kvpack
from vllm_cmp170hx.models.deepseek_v41.ops.sm80_index_scores import index_scores_triton
from vllm_cmp170hx.models.deepseek_v41.ops.sm80_sparse_attn import sparse_attention_triton

BF16 = torch.bfloat16
DEV = "cuda"


def timed(fn, repeats: int) -> float:
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1000.0)
    out.sort()
    return round(out[len(out) // 2], 1)


def caches(states: int, entries: int, rec, dim: int, page: int, g):
    """A packed cache of ``states`` random records and the BF16 cache of their
    values, blocks of ``entries`` states, the packed pages ``page`` bytes."""
    blocks = -(-states // entries) + 1
    x = torch.randn(blocks * entries, dim, generator=g, device=DEV).to(BF16)
    packed = torch.zeros(blocks * page, dtype=torch.uint8, device=DEV).as_strided(
        (blocks, entries, rec.stride(dim)), (page, rec.stride(dim), 1))
    slots = torch.arange(blocks * entries, device=DEV)
    kvpack.store(packed, slots, x, rec)
    values = kvpack.unpack(packed.reshape(-1, packed.shape[-1]), rec, dim).to(BF16)
    return packed, values.view(blocks, entries, dim)


def bench_index(ctx: int, ratio: int, rows: int, repeats: int, g) -> dict:
    width = ctx // ratio
    entries = 512  # index keys per 36,864 B page
    packed, values = caches(width, entries, kvpack.MXFP4, 128, kvpack.PACKED_PAGE_BYTES, g)
    q = torch.randn(rows, 32, 128, generator=g, device=DEV).to(BF16)
    w = torch.randn(rows, 32, generator=g, device=DEV).to(BF16)
    n_vis = torch.full((rows,), width, device=DEV)
    req = torch.zeros(rows, dtype=torch.int64, device=DEV)
    bt = torch.arange(packed.shape[0], dtype=torch.int32, device=DEV)[None]
    a = index_scores_triton(q, w, n_vis, req, values, bt, width, 32)
    b = index_scores_triton(q, w, n_vis, req, packed, bt, width, 32)
    assert torch.equal(a, b)
    return {
        "bf16_us": timed(lambda: index_scores_triton(q, w, n_vis, req, values, bt, width, 32),
                         repeats),
        "packed_us": timed(lambda: index_scores_triton(q, w, n_vis, req, packed, bt, width, 32),
                           repeats),
    }


def bench_attention(ctx: int, rows: int, repeats: int, g) -> dict:
    swa_p, swa_v = caches(4096, 64, kvpack.MXFP8, 512, kvpack.PACKED_PAGE_BYTES, g)
    comp_states = ctx // 2
    comp_p, comp_v = caches(comp_states, 128, kvpack.NVFP4, 512, kvpack.PACKED_PAGE_BYTES, g)
    q = torch.randn(rows, 64, 512, generator=g, device=DEV).to(BF16)
    idx = torch.randint(0, 4096, (rows, 128), generator=g, device=DEV).to(torch.int32)
    local = torch.randint(0, comp_states, (rows, 512), generator=g, device=DEV)
    local = local.sort(dim=1).values.to(torch.int32)
    bt = torch.arange(comp_p.shape[0], dtype=torch.int32, device=DEV)[None]
    req = torch.zeros(rows, dtype=torch.int64, device=DEV)
    sink = torch.randn(64, generator=g, device=DEV)

    def run(swa, comp):
        return sparse_attention_triton(q, swa, idx, comp, bt, local, req, sink, 512**-0.5)

    assert torch.equal(run(swa_v, comp_v), run(swa_p, comp_p))
    return {"bf16_us": timed(lambda: run(swa_v, comp_v), repeats),
            "packed_us": timed(lambda: run(swa_p, comp_p), repeats)}


def bench_store(rows: int, repeats: int, g) -> dict:
    """The window write of a step: RoPE'd rows -> cache (BF16: qdq + indexed
    write; packed: one quantize-and-pack launch), and the compressed write."""
    freqs = core.rope_freqs(64, 0, 10000.0, 1.0, 32, 1, device=DEV)
    kv = torch.randn(rows, 512, generator=g, device=DEV).to(BF16)
    pos = torch.arange(rows, device=DEV)
    slots = torch.randperm(rows * 2, device=DEV)[:rows]
    swa_v = torch.zeros(rows * 2 // 64 + 1, 64, 512, dtype=BF16, device=DEV)
    swa_p = torch.zeros(rows * 2 // 64 + 1, 64, 528, dtype=torch.uint8, device=DEV)
    res = {
        "window_bf16_us": timed(lambda: core.store_window_kv(swa_v, slots, kv, pos, freqs),
                                repeats),
        "window_packed_us": timed(lambda: core.store_window_kv(swa_p, slots, kv, pos, freqs),
                                  repeats),
    }
    comp_v = torch.zeros(rows * 2 // 128 + 1, 128, 512, dtype=BF16, device=DEV)
    comp_p = torch.zeros(rows * 2 // 128 + 1, 128, 288, dtype=torch.uint8, device=DEV)
    rows_c = core.rope(kv, pos, freqs)
    res["comp_bf16_us"] = timed(
        lambda: core.store_rows(comp_v, slots, core.qdq_fp4(rows_c, 16, "e4m3")), repeats)
    res["comp_packed_us"] = timed(
        lambda: kvpack.store(comp_p, slots, rows_c, kvpack.NVFP4), repeats)
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="16384,65536,131072")
    parser.add_argument("--repeats", type=int, default=21)
    args = parser.parse_args()
    g = torch.Generator(device=DEV).manual_seed(0)
    result = {}
    for ctx in (int(c) for c in args.contexts.split(",")):
        for ratio in (2, 1):
            for rows in (1, 4, 256):
                result[f"index_ctx{ctx}_r{ratio}_rows{rows}"] = bench_index(
                    ctx, ratio, rows, args.repeats, g)
        for rows in (1, 4, 2048):
            result[f"attn_ctx{ctx}_rows{rows}"] = bench_attention(ctx, rows, args.repeats, g)
    for rows in (4, 2048):
        result[f"store_rows{rows}"] = bench_store(rows, args.repeats, g)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
