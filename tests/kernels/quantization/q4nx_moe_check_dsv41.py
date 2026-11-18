# SPDX-License-Identifier: Apache-2.0
"""Q4N-X routed-expert kernels at DeepSeek-V4.1 geometry: FP64 numerics, repeatability and timing (campaign lab
script, card-level; no pytest needed).

Shapes (MOE_SHAPE): dsv41 = hidden 5120 / intermediate 2304, top-6 of 384 (target layers); dsv41-draft = the same
experts, top-3 of 128 (DSpark drafter mtp.0-2); glm = 4096 / 2048, top-8 of 288 (the reference family, run through
this same harness so the error scale is comparable). Q4N_BX2 = 4 (2.0 bpw) or 5 (2.5 bpw), codebook 4 (Q4N-X).
Kernel paths are process-static environment switches: FP16 (default: VLLM_EXL3_Q4N_PREFILL unset, one FP16
activation plane) or INT8 (VLLM_EXL3_Q4N_PREFILL=i8, VLLM_EXL3_Q4N_DECODE=i8, *_GEMM=t2c4); wide decode calls need
VLLM_EXL3_MOE_DECODE_TC_MAX_ROWS=64.

Numerics: moe_out_decode_tc (rows 1..64) and moe_out_prefill (65, 512, 2048 rows) against an FP64 oracle of an
independent numpy Q4N-X decoder (q4n_moe_check.q4n_core: suh, H128, codebook, H128, svh), the model's asymmetric
SwiGLU clamp (gate <= 10, |up| <= 10) and FP64 routing-weight sums. Inputs are scaled as in the GLM gate (gate
pre-activations ~ 2 x limit RMS, so the clamp is exercised). Routing weights are DeepSeek-V4.1's: sqrt(softplus)
scores of random logits, normalised over the top-k and multiplied by routed_scaling_factor 1.5 (GLM: sum 1, as its
gate). Five distinct weight sets (experts 0, E/2 - 1, E/2, E - 1 and one shared by the rest), so every scenario
mixes distinct experts; -1 / out-of-range ids test skipped routes; "skew" routes one expert twice per row and over
8 rows per expert (several 8-row tiles). Every call is run twice (bitwise repeat), and the decode schedule must give
each row bitwise the same result whether it is called alone or in a 64-row call.

Timing: one call on 384 (128, 288) distinct experts with random trellis words; decode rows x top-k distinct experts
per call (CUDA events, cache flushed), prefill uniform routing.

Usage: python q4nx_moe_check_dsv41.py OUT.json [--no-timing] [--dump DIR] (DIR: GLM bitwise reference outputs).
"""
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

try:  # the lab images have no pytest; the imported test modules only use its decorators
    import pytest  # noqa: F401
except ImportError:
    import types

    class _Any:
        def __getattr__(self, _name):
            return lambda *a, **k: (a[0] if len(a) == 1 and callable(a[0]) and not k else (lambda f: f))

    _pt = types.ModuleType("pytest")
    _pt.fixture = lambda *a, **k: (a[0] if len(a) == 1 and callable(a[0]) and not k else (lambda f: f))
    _pt.mark = _Any()
    _pt.raises = _pt.skip = _pt.importorskip = lambda *a, **k: None
    sys.modules["pytest"] = _pt

os.environ.setdefault("Q4N_CB", "4")
from tests.kernels.quantization import q4n_moe_check as Q  # noqa: E402  (Q4N_CB read at import)
from vllm_cmp170hx.kernels.exl3 import ops as exl3_ops  # noqa: E402

SHAPES = {
    # name: hidden, intermediate, top_k, experts, routed scaling (weights sum)
    "dsv41": (5120, 2304, 6, 384, 1.5),
    "dsv41-draft": (5120, 2304, 3, 128, 1.5),
    "glm": (4096, 2048, 8, 288, 1.0),
}
SHAPE = os.environ.get("MOE_SHAPE", "dsv41")
H, I, K, E, ROUTED_SCALE = SHAPES[SHAPE]
BX2 = int(os.environ.get("Q4N_BX2", "5"))
CB = 4
KEY = BX2 + 128
LIMIT = 10.0
OBSERVED = (0, E // 2 - 1, E // 2, E - 1)
SHARED = len(OBSERVED)
SET_OF = np.full(E, SHARED, dtype=np.int64)
SET_OF[list(OBSERVED)] = range(SHARED)
DECODE_ROWS = (1, 2, 4, 8, 16, 32, 64)
PREFILL_ROWS = (65, 512, 2048)
ORACLE_DEVICE = os.environ.get("ORACLE_DEVICE", "cuda")
assert Q.Q4N_CB == 4


def experts(seed=10157):
    """Five distinct expert weight sets (random trellis words, suh, svh) and their FP64 dense matrices; the dense
    matrices are cached under DENSE_CACHE (a pure function of shape, bits_x2 and seed)."""
    device = torch.device("cuda")
    rng = np.random.default_rng(seed)
    owners, sets, dense = [], [], []
    cache_dir = os.environ.get("DENSE_CACHE")
    for s_index in range(SHARED + 1):
        ptrs, mats = [], []
        for p_index, (k, n) in enumerate(((H, I), (H, I), (I, H))):
            trellis = rng.integers(-(2 ** 15), 2 ** 15, (k // 16, n // 16, 8 * BX2)).astype(np.int16)
            suh = (rng.choice([-1.0, 1.0], k) * rng.uniform(0.5, 1.5, k)).astype(np.float16)
            svh = (rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n) / 64).astype(np.float16)
            tensors = [torch.from_numpy(v).to(device) for v in (trellis, suh, svh)]
            owners.append(tensors)
            ptrs.append([t.data_ptr() for t in tensors])
            path = cache_dir and os.path.join(cache_dir, f"dense-{H}x{I}-bx{BX2}-s{seed}-{s_index}{p_index}.npy")
            if path and os.path.exists(path):
                mat = np.load(path)
            else:
                mat = Q.q4n_dense(trellis, suh, svh, slab=256)  # 2304 = 9 x 256 (not a multiple of 512)
                if path:
                    os.makedirs(cache_dir, exist_ok=True)
                    np.save(path, mat)
            mats.append(torch.from_numpy(mat).to(ORACLE_DEVICE))
        sets.append(ptrs)
        dense.append(mats)
    rms = sorted(float(w[0].pow(2).mean().sqrt()) for w in dense)
    scale = 2.0 * LIMIT / (H ** 0.5 * rms[len(rms) // 2])
    meta = [[[H, I, BX2, CB], [H, I, BX2, CB], [I, H, BX2, CB]]] * E
    return SimpleNamespace(
        owners=owners,
        pointers=torch.tensor([sets[s] for s in SET_OF], dtype=torch.int64, device=device),
        meta=torch.tensor(meta, dtype=torch.int64, device=device),
        expert_map=torch.arange(E, dtype=torch.int32, device=device),
        dense=dense, set_of=torch.from_numpy(SET_OF).to(ORACLE_DEVICE), scale=scale)


def routes(scenario, rows, gen):
    """(ids [rows, K] int64, weights [rows, K] FP32): DeepSeek-V4.1 sqrt(softplus) top-k weights x routed scale."""
    row = torch.arange(rows, dtype=torch.int64)
    ids = torch.empty((rows, K), dtype=torch.int64)
    for j in range(K):
        ids[:, j] = (5 + 37 * j + row * (j + 1)) % E  # distinct-ish varying experts
    if scenario == "observed":
        for j, e in enumerate(OBSERVED[: max(1, K - 2)]):
            ids[:, j] = e
        ids[:, K - 1] = -1
    elif scenario == "boundary":
        for j, e in enumerate(OBSERVED[1:][: max(1, K - 2)]):
            ids[:, j] = e
        ids[:, K - 1] = E  # out of range: skipped
    elif scenario == "sparse":
        ids[:, 2:] = -1
        ids[:, 0] = row % E
        if K > 1:
            ids[:, 1] = (7 * row + 3) % E
    elif scenario == "skew":
        # One expert twice per row and over 8 routes in total (several 8-row tiles of one expert).
        ids[:, 0] = 17
        if K > 1:
            ids[:, 1] = 17
        if K > 2:
            ids[:, 2] = 40
        if K > 3:
            ids[:, 3] = 40 + row % 3
        if K > 4:
            ids[:, 4] = OBSERVED[1]
        ids[:, K - 1] = -1 if K > 2 else ids[:, K - 1]
    elif scenario == "uniform":
        pass
    else:
        raise ValueError(scenario)
    logits = torch.randn(rows, E, generator=gen) * 2.0
    scores = torch.nn.functional.softplus(logits).sqrt()
    if scenario == "uniform":
        ids = scores.topk(K, dim=-1)[1]
    w = scores.gather(1, ids.clamp(0, E - 1))
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * ROUTED_SCALE
    return ids, w.float()


def case(ex, scenario, rows, seed):
    gen = torch.Generator().manual_seed(seed)
    ids, weights = routes(scenario, rows, gen)
    x = (torch.randn(rows, H, generator=gen) * ex.scale).to(torch.bfloat16)
    return x.cuda(), ids.to(torch.int32).cuda(), weights.cuda()


def run(op, size_fn, ex, x, ids, weights, ws_rows=None):
    out = torch.full_like(x, float("nan"))
    ws = torch.full((size_fn(ws_rows or x.shape[0], H, I, K, E),), float("nan"), dtype=torch.float32, device=x.device)
    op(x, ids, weights, ex.pointers, ex.meta, ex.expert_map, I, KEY, out, ws)
    return out


def dec(ex, x, ids, w):
    return run(exl3_ops.moe_out_decode_tc, exl3_ops.moe_workspace_size_decode_tc, ex, x, ids, w)


def pre(ex, x, ids, w):
    return run(exl3_ops.moe_out_prefill, exl3_ops.moe_workspace_size_prefill, ex, x, ids, w)


def oracle(ex, x, ids, weights):
    dev = ORACLE_DEVICE
    x64 = x.to(dev).double()
    ids = ids.to(dev).long()
    weights = weights.to(dev)
    out = torch.zeros_like(x64)
    table = ex.expert_map.to(dev).long()
    inside = (ids >= 0) & (ids < table.numel())
    local = torch.where(inside, table[ids.clamp(0, table.numel() - 1)], -1)
    for route in range(K):
        column = local[:, route]
        rows = torch.nonzero(column >= 0, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        for index, (gw, uw, dw) in enumerate(ex.dense):
            mine = ex.set_of[column[rows]] == index
            if not bool(mine.any()):
                continue
            picked = rows[mine]
            scaled = weights[picked, route].double()[:, None]
            gate = (x64[picked] @ gw).clamp(max=LIMIT)
            up = (x64[picked] @ uw).clamp(min=-LIMIT, max=LIMIT)
            act = gate / (1.0 + torch.exp(-gate)) * up
            out.index_add_(0, picked, act @ dw * scaled)
    return out


def errors(ex, op, cases, seed0, dump=None, tag=""):
    """Mean / max abs error vs the FP64 oracle, mean |oracle| (for relative error), bitwise repeat."""
    s = n = mx = ref_abs = 0.0
    per = {}
    for idx, (scenario, rows) in enumerate(cases):
        x, ids, w = case(ex, scenario, rows, seed0 + idx)
        out = op(ex, x, ids, w)
        again = op(ex, x, ids, w)
        assert torch.equal(out, again), f"{tag} {scenario}/{rows}: not bitwise repeatable"
        assert torch.isfinite(out).all(), f"{tag} {scenario}/{rows}: non-finite output"
        if dump is not None:
            torch.save(out.cpu(), os.path.join(dump, f"{tag}-{scenario}-{rows}.pt"))
        ref = oracle(ex, x, ids, w)
        e = (out.to(ref.device).double() - ref).abs()
        per[f"{scenario}/{rows}"] = round(float(e.mean()), 5)
        s += float(e.sum()); n += e.numel(); mx = max(mx, float(e.max())); ref_abs += float(ref.abs().sum())
    return {"mean_abs_err": s / n, "max_abs_err": mx, "mean_abs_ref": ref_abs / n, "rel_mean_err": s / ref_abs,
            "per_case_mean": per}


def row_invariance(ex, rows=64, seed=4242):
    """Decode: each row of a wide call is bitwise its 1-row call (and an 8-row slice's)."""
    x, ids, w = case(ex, "uniform", rows, seed)
    wide = dec(ex, x, ids, w)
    for lo in range(0, rows, 8):
        part = dec(ex, x[lo:lo + 8], ids[lo:lo + 8], w[lo:lo + 8])
        assert torch.equal(part, wide[lo:lo + 8]), f"8-row slice at {lo} differs from the {rows}-row call"
    for r in (0, 1, rows // 2, rows - 1):
        one = dec(ex, x[r:r + 1], ids[r:r + 1], w[r:r + 1])
        assert torch.equal(one, wide[r:r + 1]), f"row {r} alone differs from the {rows}-row call"
    return True


def distinct_tables(n_experts=E, seed=3):
    g = torch.Generator(device="cuda").manual_seed(seed)
    keep, rows = [], []
    for _ in range(n_experts):
        row = []
        for k, n in ((H, I), (H, I), (I, H)):
            tr = torch.randint(-32768, 32768, (k // 16, n // 16, 8 * BX2), generator=g, device="cuda",
                               dtype=torch.int32).to(torch.int16)
            suh = torch.ones(k, dtype=torch.float16, device="cuda")
            svh = torch.ones(n, dtype=torch.float16, device="cuda") / 64
            keep += [tr, suh, svh]
            row.append([tr.data_ptr(), suh.data_ptr(), svh.data_ptr()])
        rows.append(row)
    meta = [[[H, I, BX2, CB], [H, I, BX2, CB], [I, H, BX2, CB]]] * E
    return keep, torch.tensor(rows, dtype=torch.int64, device="cuda"), torch.tensor(meta, dtype=torch.int64,
                                                                                   device="cuda")


def timed(call, reps, flush=None):
    for _ in range(5):
        call()
    ts = []
    for _ in range(reps):
        if flush is not None:
            flush.zero_()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); call(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def timing(keep_ptrs_meta):
    _, pointers, meta = keep_ptrs_meta
    expert_map = torch.arange(E, dtype=torch.int32, device="cuda")
    res = {"decode_us": {}, "prefill_ms": {}}
    flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    for rows in (1, 4, 8, 16, 64):
        if rows > exl3_ops.MOE_DECODE_TC_MAX_ROWS:
            continue
        ids = (torch.arange(rows * K, dtype=torch.int64, device="cuda") % E).view(rows, K)
        w = torch.full((rows, K), ROUTED_SCALE / K, dtype=torch.float32, device="cuda")
        x = (torch.randn(rows, H, device="cuda") * 0.05).to(torch.bfloat16)
        out = torch.empty_like(x)
        ws = torch.empty(exl3_ops.moe_workspace_size_decode_tc(rows, H, I, K, E), dtype=torch.float32, device="cuda")
        call = lambda: exl3_ops.moe_out_decode_tc(x, ids, w, pointers, meta, expert_map, I, KEY, out, ws)  # noqa
        res["decode_us"][rows] = round(1000 * timed(call, 31, flush), 1)
    g = torch.Generator().manual_seed(5)
    for rows in (512, 2048):
        ids = torch.argsort(torch.rand(rows, E, generator=g), dim=1)[:, :K].cuda()
        w = torch.full((rows, K), ROUTED_SCALE / K, dtype=torch.float32, device="cuda")
        x = (torch.randn(rows, H, device="cuda") * 0.05).to(torch.bfloat16)
        out = torch.empty_like(x)
        ws = torch.empty(exl3_ops.moe_workspace_size_prefill(rows, H, I, K, E), dtype=torch.float32, device="cuda")
        call = lambda: exl3_ops.moe_out_prefill(x, ids, w, pointers, meta, expert_map, I, KEY, out, ws)  # noqa
        res["prefill_ms"][rows] = round(timed(call, 11), 3)
    return res


def main():
    out_path = sys.argv[1]
    no_timing = "--no-timing" in sys.argv
    dump = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else None
    if dump:
        os.makedirs(dump, exist_ok=True)
    exl3_ops.ensure_extension()
    env = {k: v for k, v in os.environ.items() if k.startswith("VLLM_EXL3")}
    props = torch.cuda.get_device_properties(0)
    res = {"shape": SHAPE, "hidden": H, "intermediate": I, "top_k": K, "experts": E, "bits_x2": BX2, "key": KEY,
           "routed_scale": ROUTED_SCALE, "env": env, "device": props.name, "sms": props.multi_processor_count,
           "decode_tc_max_rows": exl3_ops.MOE_DECODE_TC_MAX_ROWS}
    t0 = time.time()
    ex = experts()
    res["build_s"] = round(time.time() - t0, 1)
    print("experts built", res["build_s"], "s; input scale", ex.scale, flush=True)
    scen = ("observed", "boundary", "uniform", "skew", "sparse")
    dcases = [(scen[i % len(scen)], r) for i, r in enumerate(DECODE_ROWS) if r <= exl3_ops.MOE_DECODE_TC_MAX_ROWS]
    dcases += [("uniform", 64), ("skew", 16)] if exl3_ops.MOE_DECODE_TC_MAX_ROWS >= 64 else []
    pcases = [("observed", 65), ("skew", 512), ("uniform", 2048)]
    t0 = time.time()
    res["decode"] = errors(ex, dec, dcases, 700, dump, "decode")
    print("decode", json.dumps(res["decode"]), round(time.time() - t0, 1), "s", flush=True)
    t0 = time.time()
    res["prefill"] = errors(ex, pre, pcases, 100, dump, "prefill")
    print("prefill", json.dumps(res["prefill"]), round(time.time() - t0, 1), "s", flush=True)
    if exl3_ops.MOE_DECODE_TC_MAX_ROWS >= 64:
        res["decode_row_invariance_bitwise"] = row_invariance(ex)
    del ex
    torch.cuda.empty_cache()
    if not no_timing:
        tables = distinct_tables()
        res["timing"] = timing(tables)
        del tables
        print("timing", json.dumps(res["timing"]), flush=True)
    json.dump(res, open(out_path, "w"), indent=1, default=str)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
