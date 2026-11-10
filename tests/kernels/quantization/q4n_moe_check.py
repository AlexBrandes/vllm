# SPDX-License-Identifier: Apache-2.0
"""Q4N (codebook 3) GLM routed-expert kernels: FP64 numerics gate and timing (campaign lab script, card-level).

Numerics: the Q4N tensor-core kernels (moe_out_decode_tc rows 1..8, moe_out_prefill 65/129 rows) are compared with
an FP64 oracle built from an independent Q4N decoder; the gate is relative to the shipped mul1 tensor-core kernels'
error against their own FP64 oracle on the same routes and inputs (mean <= 1.10x, max <= 1.25x), plus bitwise repeat.
Timing: one routed decode call at the served geometry (rows x top-8, distinct experts), Q4N vs mul1, synthetic
trellis words; diagnostic only.
"""
import json
import sys
from types import SimpleNamespace

import numpy as np
import torch

from tests.kernels.quantization import test_exl3_moe_prefill_glm as G
from vllm_cmp170hx.kernels.exl3 import ops as exl3_ops

Q4N_KEY = 5 + 64
MULT = 0x83DCD12D
ALPHA = 1.0 / 6.519202405202649


def q4n_core(trellis):
    """(k/16, n/16, 40) int16 Q4N ring -> FP64 core matrix (k, n), independent of the CUDA decoder."""
    tk, tn, h = trellis.shape
    halves = trellis.astype(np.int64) & 0xFFFF
    words = halves[..., 0::2] | (halves[..., 1::2] << 16)  # (tk, tn, 20) uint32 values
    bits = ((words[..., :, None] >> (31 - np.arange(32))) & 1).reshape(tk, tn, 640)
    core = np.zeros((tk * 16, tn * 16))
    for i in range(64):
        end = (i + 1) * 10
        st = np.zeros((tk, tn), dtype=np.int64)
        for b in range(end - 16, end):
            st = (st << 1) | (bits[..., b] if b >= 0 else 0)
        p = (st * MULT) & 0xFFFFFFFF
        lane, j = i >> 1, i & 1
        g, t = lane >> 2, lane & 3
        for c in range(4):
            byte = (p >> (8 * c)) & 0xFF
            v = ((byte >> 4) + (byte & 15) - 15).astype(np.float64) * ALPHA
            k = 2 * t + (c & 1) + 8 * (c >> 1)
            n = g + 8 * j
            core[k::16, n::16][: tk, : tn] = v
    return core


def q4n_dense(trellis, suh, svh, slab=512):
    from tests.kernels.quantization.test_exl3_kernels import _blocks
    padded_in, padded_out = trellis.shape[0] * 16, trellis.shape[1] * 16
    blocks = _blocks(slab)
    right = _blocks(padded_out) * svh.astype(np.float64)[None, :]
    out = np.empty((padded_in, padded_out))
    for lo in range(0, padded_in, slab):
        hi = lo + slab
        core = q4n_core(trellis[lo // 16: hi // 16])
        left = suh[lo:hi].astype(np.float64)[:, None] * blocks
        out[lo:hi] = left @ core @ right
    return out


def q4n_experts(seed=10157):
    device = torch.device("cuda")
    rng = np.random.default_rng(seed)
    owners, sets, dense = [], [], []
    for _ in range(G.SHARED + 1):
        ptrs, mats = [], []
        for k, n in ((G.GHIDDEN, G.GINTERMEDIATE), (G.GHIDDEN, G.GINTERMEDIATE), (G.GINTERMEDIATE, G.GHIDDEN)):
            trellis = rng.integers(-(2 ** 15), 2 ** 15, (k // 16, n // 16, 40)).astype(np.int16)
            suh = (rng.choice([-1.0, 1.0], k) * rng.uniform(0.5, 1.5, k)).astype(np.float16)
            svh = (rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n) / 64).astype(np.float16)
            tensors = [torch.from_numpy(v).to(device) for v in (trellis, suh, svh)]
            owners.append(tensors)
            ptrs.append([t.data_ptr() for t in tensors])
            mats.append(torch.from_numpy(q4n_dense(trellis, suh, svh)).to(device))
        sets.append(ptrs)
        dense.append(mats)
    rms = sorted(float(w[0].pow(2).mean().sqrt()) for w in dense)
    scale = 2.0 * G.LIMIT / (G.GHIDDEN ** 0.5 * rms[len(rms) // 2])
    meta = [[[G.GHIDDEN, G.GINTERMEDIATE, 5, 3], [G.GHIDDEN, G.GINTERMEDIATE, 5, 3],
             [G.GINTERMEDIATE, G.GHIDDEN, 5, 3]]] * G.GLOCAL
    return SimpleNamespace(
        owners=owners,
        pointers=torch.tensor([sets[s] for s in G.SET_OF], dtype=torch.int64, device=device),
        meta=torch.tensor(meta, dtype=torch.int64, device=device),
        expert_map=torch.arange(G.GGLOBAL, dtype=torch.int32, device=device),
        dense=dense, set_of=torch.from_numpy(G.SET_OF).to(device), scale=scale, bits_x2=Q4N_KEY)


def rel_errors(glm, op, scenarios, seed0):
    ec_sum = ec_n = 0.0
    ec_max = 0.0
    for idx, (scenario, rows) in enumerate(scenarios):
        x, ids, weights = G._case(glm, scenario, rows, seed0 + idx)
        out = op(glm, x, ids, weights)
        again = op(glm, x, ids, weights)
        assert torch.equal(out, again), f"{scenario}/{rows}: not bitwise repeatable"
        assert torch.isfinite(out).all()
        ref, _ = G._oracle(glm, x, ids, weights)
        e = (out.double() - ref).abs()
        ec_sum += float(e.sum()); ec_n += e.numel(); ec_max = max(ec_max, float(e.max()))
    return ec_sum / ec_n, ec_max


def distinct_tables(codebook, n_experts=64, seed=3):
    """n distinct experts (no shared memory between them) for timing; meta for 288 local experts."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    keep, rows = [], []
    for _ in range(n_experts):
        row = []
        for k, n in ((G.GHIDDEN, G.GINTERMEDIATE), (G.GHIDDEN, G.GINTERMEDIATE), (G.GINTERMEDIATE, G.GHIDDEN)):
            tr = torch.randint(-32768, 32768, (k // 16, n // 16, 40), generator=g, dtype=torch.int32).to(torch.int16).cuda()
            suh = torch.ones(k, dtype=torch.float16, device="cuda"); svh = torch.ones(n, dtype=torch.float16, device="cuda") / 64
            keep += [tr, suh, svh]; row.append([tr.data_ptr(), suh.data_ptr(), svh.data_ptr()])
        rows.append(row)
    rows += [rows[i % n_experts] for i in range(G.GLOCAL - n_experts)]
    meta = [[[G.GHIDDEN, G.GINTERMEDIATE, 5, codebook], [G.GHIDDEN, G.GINTERMEDIATE, 5, codebook],
             [G.GINTERMEDIATE, G.GHIDDEN, 5, codebook]]] * G.GLOCAL
    return keep, torch.tensor(rows, dtype=torch.int64, device="cuda"), torch.tensor(meta, dtype=torch.int64, device="cuda")


def timing_prefill(key, codebook, rows):
    """One moe_out_prefill call: rows tokens x top-8 over 64 distinct experts (uniform routing)."""
    keep, pointers, meta = distinct_tables(codebook)
    topk = 8
    g = torch.Generator().manual_seed(5)
    local_ids = torch.argsort(torch.rand(rows, 64, generator=g), dim=1)[:, :topk].cuda()
    weights = torch.full((rows, topk), 1.0 / topk, dtype=torch.float32, device="cuda")
    expert_map = torch.arange(G.GGLOBAL, dtype=torch.int32, device="cuda")
    x = (torch.randn(rows, G.GHIDDEN, device="cuda") * 0.05).to(torch.bfloat16)
    out = torch.empty_like(x)
    ws = torch.empty(exl3_ops.moe_workspace_size_prefill(rows, G.GHIDDEN, G.GINTERMEDIATE, topk, G.GLOCAL),
                     dtype=torch.float32, device="cuda")
    call = lambda: exl3_ops.moe_out_prefill(x, local_ids, weights, pointers, meta, expert_map,
                                            G.GINTERMEDIATE, key, out, ws)
    for _ in range(3):
        call()
    ts = []
    for _ in range(11):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); call(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    del keep
    return ts[len(ts) // 2]


def timing(key, codebook, rows):
    keep, pointers, meta = distinct_tables(codebook)
    topk = 8
    local_ids = torch.arange(rows * topk, dtype=torch.int64, device="cuda").view(rows, topk)
    weights = torch.full((rows, topk), 1.0 / topk, dtype=torch.float32, device="cuda")
    expert_map = torch.arange(G.GGLOBAL, dtype=torch.int32, device="cuda")
    x = (torch.randn(rows, G.GHIDDEN, device="cuda") * 0.05).to(torch.bfloat16)
    out = torch.empty_like(x)
    ws = torch.empty(exl3_ops.moe_workspace_size_decode_tc(8, G.GHIDDEN, G.GINTERMEDIATE, topk, G.GLOCAL),
                     dtype=torch.float32, device="cuda")
    call = lambda: exl3_ops.moe_out_decode_tc(x, local_ids, weights, pointers, meta, expert_map,
                                              G.GINTERMEDIATE, key, out, ws)
    for _ in range(5):
        call()
    flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    ts = []
    for _ in range(31):
        flush.zero_()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); call(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    del keep
    return ts[len(ts) // 2]


def main():
    exl3_ops.ensure_extension()
    res = {}
    mul1 = G.glm_experts()
    q4n = q4n_experts()
    dec = [("observed", 1), ("boundary", 2), ("uniform", 3), ("skew", 4), ("sparse", 5), ("uniform", 6), ("skew", 7), ("skew", 8)]
    pre = [("observed", 65), ("boundary", 129)]
    for name, glm in (("mul1", mul1), ("q4n", q4n)):
        res[name] = {
            "decode_mean_abs_err": None, "decode_max_abs_err": None,
        }
        m, mx = rel_errors(glm, G._d, dec, 700)
        res[name]["decode_mean_abs_err"], res[name]["decode_max_abs_err"] = m, mx
        m, mx = rel_errors(glm, G._p, pre, 100)
        res[name]["prefill_mean_abs_err"], res[name]["prefill_max_abs_err"] = m, mx
        key, cb = (5, 2) if name == "mul1" else (Q4N_KEY, 3)
        res[name]["decode_ms"] = {r: round(timing(key, cb, r), 4) for r in (1, 2, 4, 8)}
        res[name]["prefill_ms"] = {r: round(timing_prefill(key, cb, r), 3) for r in (512, 2048)}
        print(name, json.dumps(res[name]), flush=True)
    for phase in ("decode", "prefill"):
        mean_ratio = res["q4n"][f"{phase}_mean_abs_err"] / res["mul1"][f"{phase}_mean_abs_err"]
        max_ratio = res["q4n"][f"{phase}_max_abs_err"] / res["mul1"][f"{phase}_max_abs_err"]
        res[f"{phase}_gate"] = {"mean_ratio": round(mean_ratio, 4), "max_ratio": round(max_ratio, 4),
                                "pass": mean_ratio <= 1.10 and max_ratio <= 1.25}
    res["decode_speedup"] = {r: round(res["mul1"]["decode_ms"][r] / res["q4n"]["decode_ms"][r], 3) for r in (1, 2, 4, 8)}
    res["prefill_speedup"] = {r: round(res["mul1"]["prefill_ms"][r] / res["q4n"]["prefill_ms"][r], 3) for r in (512, 2048)}
    print(json.dumps({k: res[k] for k in ("decode_gate", "prefill_gate", "decode_speedup", "prefill_speedup")}, indent=1))
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "/out/q4n-moe-check.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
