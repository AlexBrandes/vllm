# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_80 sparse MLA decode v2 (``VLLM_AMPERE_SPARSE_MLA_DECODE_V2``).

With the flag set, ``triton_mla_sparse_fwd`` sends bf16 NoPE (512-wide) decode
calls with 16 or 64 query heads, 2048 or 2176 index slots and up to 64 rows on
sm_80 to ``triton_mla_sparse_decode_v2.py``; every other call, and every call
with the flag unset, keeps the shipped kernel.

CPU tests (``CUDA_VISIBLE_DEVICES=""``): the flag, the gate and its closed
reasons, the banners through the vLLM logger, and that the flag-off path never
reaches the new module. GPU tests (sm_80, skip otherwise): flag off bitwise
equal to the base tree's kernel when ``SMLA_DECODE_V2_BASE_TREE`` names one,
flag off unchanged by flag-on calls, accuracy against an FP64 reference no worse
than the shipped kernel's (per output and shape, absolute errors pooled over
layouts and seeds: mean <= 1.10x, max <= 1.25x, exact where it is exact),
strided views bitwise the dense result, gate-closed calls bitwise the shipped
kernel's, and CUDA-graph replay bitwise equal to eager, also after new query
and index contents, with no allocation growth.

    CUDA_VISIBLE_DEVICES="" pytest -q tests/kernels/test_ampere_sparse_mla_decode_v2.py
    pytest -q tests/kernels/test_ampere_sparse_mla_decode_v2.py
"""

import importlib.util
import logging
import os
import sys
import types

import pytest
import torch

from vllm.sm80.kernels.attention import triton_mla_sparse as tms

FLAG = "VLLM_AMPERE_SPARSE_MLA_DECODE_V2"
V2_MODULE = "vllm.sm80.kernels.attention.triton_mla_sparse_decode_v2"
DIM, WIDTH, TOPK, KPOOL, SM_SCALE, BLOCK = 512, 2176, 2048, 4, 0.0625, 1152
SERVED_T = (1, 2, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 40, 48, 56, 64)
OUTPUTS = ("out", "max_logits", "lse")
HEADS = (16, 64)
HAS_GPU = torch.cuda.is_available()
IS_SM80 = HAS_GPU and torch.cuda.get_device_capability(0) == (8, 0)
needs_sm80 = pytest.mark.skipif(not IS_SM80, reason="needs an sm_80 GPU")


# --------------------------------------------------------------------- CPU
def test_flag_declared_default_off(monkeypatch):
    from vllm import envs
    from vllm.sm80 import envs as fork_envs

    monkeypatch.delenv(FLAG, raising=False)
    assert FLAG in envs.environment_variables
    assert envs.environment_variables[FLAG]() is False
    monkeypatch.setenv(FLAG, "1")
    assert envs.environment_variables[FLAG]() is True
    assert f"    {FLAG}: bool = False\n" in open(fork_envs.__file__).read()


def _meta(T=4, heads=16, dim=DIM, width=WIDTH, dtype=torch.bfloat16, kv_heads=1):
    q = torch.empty(T, heads, dim, dtype=dtype, device="meta")
    kv = torch.empty(8192, kv_heads, dim, dtype=dtype, device="meta")
    idx = torch.empty(T, kv_heads, width, dtype=torch.int32, device="meta")
    return q, kv, idx


@pytest.fixture
def sm80(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_a, **_k: (8, 0))
    tms._sm80.cache_clear()
    yield
    tms._sm80.cache_clear()


@pytest.mark.parametrize("width", (2048, WIDTH))
@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("T", SERVED_T)
def test_gate_open_on_the_validated_configuration(monkeypatch, sm80, T, heads, width):
    monkeypatch.setenv(FLAG, "1")
    q, kv, idx = _meta(T, heads, width=width)
    assert tms._decode_v2_closed(q, kv, idx, 0, DIM) == ""
    assert tms._use_decode_v2(q, kv, idx, 0, DIM) is True


@pytest.mark.parametrize("case,expect", [
    ("rows65", "query rows"),
    ("rows0", "query rows"),
    ("heads32", "query heads"),
    ("heads8", "query heads"),
    ("fp16", "bfloat16"),
    ("dpe64", "NoPE"),
    ("dv256", "NoPE"),
    ("kvheads2", "NoPE"),
    ("width1024", "index width"),
    ("strided", "contiguous"),
    ("sm86", "sm_80"),
])
def test_gate_closed_reasons(monkeypatch, sm80, case, expect):
    monkeypatch.setenv(FLAG, "1")
    dpe, d_v = 0, DIM
    q, kv, idx = _meta()
    if case == "rows65":
        q, kv, idx = _meta(T=65)
    elif case == "rows0":
        q, kv, idx = _meta(T=0)
    elif case == "heads32":
        q, kv, idx = _meta(heads=32)
    elif case == "heads8":
        q, kv, idx = _meta(heads=8)
    elif case == "fp16":
        q, kv, idx = _meta(dtype=torch.float16)
    elif case == "dpe64":
        q, kv, idx = _meta(dim=576)
        dpe = 64
    elif case == "dv256":
        d_v = 256
    elif case == "kvheads2":
        q, kv, idx = _meta(kv_heads=2)
    elif case == "width1024":
        q, kv, idx = _meta(width=1024)
    elif case == "strided":
        q = torch.empty(4, 16, 2 * DIM, dtype=torch.bfloat16, device="meta")[..., ::2]
    elif case == "sm86":
        monkeypatch.setattr(torch.cuda, "get_device_capability",
                            lambda *_a, **_k: (8, 6))
        tms._sm80.cache_clear()
    why = tms._decode_v2_closed(q, kv, idx, dpe, d_v)
    assert expect in why, why
    assert tms._use_decode_v2(q, kv, idx, dpe, d_v) is False


def test_flag_off_never_looks_at_the_device(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)

    def boom(*_a, **_k):
        raise AssertionError("the flag-off path must not look at the device")

    monkeypatch.setattr(torch.cuda, "get_device_capability", boom)
    tms._sm80.cache_clear()
    q, kv, idx = _meta()
    assert tms._use_decode_v2(q, kv, idx, 0, DIM) is False


def _banners(fn):
    import vllm.logger as vlog
    from vllm.logger import init_logger

    seen = []

    class Rec(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    logger = init_logger(tms.__name__)
    h = Rec(level=logging.DEBUG)
    old = logger.level
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    vlog._print_info_once.cache_clear()
    try:
        out = fn()
    finally:
        logger.removeHandler(h)
        logger.setLevel(old)
    return out, seen


@pytest.mark.parametrize("flag,T,want,msg", [
    (None, 4, False, None),
    ("0", 4, False, None),
    ("1", 4, True, f"sm_80 sparse MLA decode v2 active for <= 64 query rows [{FLAG}=1]"),
    ("1", 65, False, f"{FLAG}=1 but sparse MLA decode v2 is off for this call"),
])
def test_banners_through_the_vllm_logger(monkeypatch, sm80, flag, T, want, msg):
    if flag is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, flag)
    q, kv, idx = _meta(T=T)
    got, seen = _banners(lambda: tms._use_decode_v2(q, kv, idx, 0, DIM))
    assert got is want
    if msg is None:
        assert seen == []
    else:
        assert any(msg in m for m in seen), seen


class _Reached(Exception):
    pass


def test_flag_off_keeps_the_shipped_path(monkeypatch, sm80):
    """Flag off: the shipped schedule runs and the new module is never called;
    flag on with the gate open: the new module gets the call unchanged."""
    calls = []
    fake = types.ModuleType(V2_MODULE)

    def fake_fwd(q, kv, indices, sm_scale, d_v=512, block_dpe=None, out=None):
        calls.append((q.shape, sm_scale, d_v, block_dpe, out))
        return "v2"

    fake.triton_mla_sparse_fwd = fake_fwd
    monkeypatch.setitem(sys.modules, V2_MODULE, fake)

    def shipped(*_a, **_k):
        raise _Reached

    monkeypatch.setattr(tms, "_pick_config", shipped)
    q, kv, idx = _meta()
    monkeypatch.delenv(FLAG, raising=False)
    with pytest.raises(_Reached):
        tms.triton_mla_sparse_fwd(q, kv, idx, SM_SCALE, d_v=DIM, block_dpe=0)
    assert calls == []
    monkeypatch.setenv(FLAG, "1")
    assert tms.triton_mla_sparse_fwd(q, kv, idx, SM_SCALE, d_v=DIM) == "v2"
    assert calls == [(q.shape, SM_SCALE, DIM, 0, None)]
    q65, kv65, idx65 = _meta(T=65)
    with pytest.raises(_Reached):
        tms.triton_mla_sparse_fwd(q65, kv65, idx65, SM_SCALE, d_v=DIM, block_dpe=0)
    assert len(calls) == 1


# ------------------------------------------------------------ GPU: inputs
def _rows_short(ctx, t_req):
    """Batch max seq_len <= 2048: [0 .. pos] then -1 (one row per draft position)."""
    rows = []
    for j in range(t_req):
        pos = ctx - t_req + j
        r = torch.full((WIDTH,), -1, dtype=torch.int64)
        r[:pos + 1] = torch.arange(pos + 1)
        rows.append(r)
    return rows


def _rows_pools(ctx, t_req, gen, width=WIDTH):
    """Batch max seq_len > 2048: up to 512 sorted complete pools of 4 tokens in
    [0, 2048) (a -1 gap when the row has fewer), its incomplete tail pool at
    2048.., then -1."""
    rows = []
    for j in range(t_req):
        pos = ctx - t_req + j
        full = (pos + 1) // KPOOL
        npools = min(full, TOPK // KPOOL)
        pick = torch.randperm(full, generator=gen)[:npools].sort().values
        r = torch.full((width,), -1, dtype=torch.int64)
        toks = (pick[:, None] * KPOOL + torch.arange(KPOOL)).flatten()
        r[:toks.numel()] = toks
        tail = (pos + 1) - full * KPOOL
        if width > TOPK and tail:
            r[TOPK:TOPK + tail] = full * KPOOL + torch.arange(tail)
        rows.append(r)
    return rows


def make_case(layout, T, heads, seed, width=WIDTH, device="cuda"):
    """q [T, heads, 512], kv [R, 1, 512] (NaN in every row the indices do not
    name), indices [T, 1, width] over a paged cache (1152-token blocks)."""
    gen = torch.Generator().manual_seed(seed)
    reqs = []  # (ctx, rows)
    left = T
    while left:
        t_req = min(4, left)
        if layout == "short":
            ctx = int(torch.randint(280, 421, (1,), generator=gen))
            rows = _rows_short(ctx, t_req)
        elif layout == "kpool":
            ctx = int(torch.randint(8192, 32768, (1,), generator=gen))
            rows = _rows_pools(ctx, t_req, gen, width)
        elif layout == "gap":  # a short request in a long batch: -1 gap before the tail
            ctx = int(torch.randint(300, 1500, (1,), generator=gen))
            rows = _rows_pools(ctx, t_req, gen, width)
        elif layout == "masked":  # every second request has no valid entry
            ctx = int(torch.randint(280, 421, (1,), generator=gen))
            rows = _rows_short(ctx, t_req)
            if len(reqs) % 2:
                rows = [torch.full((width,), -1, dtype=torch.int64) for _ in rows]
        else:
            raise ValueError(layout)
        reqs.append((ctx, [r[:width] for r in rows]))
        left -= t_req
    nblocks = sum(-(-ctx // BLOCK) for ctx, _ in reqs) + 2
    perm = torch.randperm(nblocks, generator=gen)
    idx, b0 = [], 0
    for ctx, rows in reqs:
        table = perm[b0:b0 + -(-ctx // BLOCK)]
        b0 += table.numel()
        for r in rows:
            ok = r >= 0
            phys = torch.where(ok, table[r.clamp_min(0) // BLOCK] * BLOCK + r.clamp_min(0) % BLOCK,
                               torch.full_like(r, -1))
            idx.append(phys)
    indices = torch.stack(idx).to(torch.int32).view(T, 1, width)
    kv = torch.full((nblocks * BLOCK, 1, DIM), float("nan"), dtype=torch.bfloat16)
    named = indices[indices >= 0].long().unique()
    kv[named, 0] = (torch.randn(named.numel(), DIM, generator=gen) * 0.6).to(torch.bfloat16)
    q = (torch.randn(T, heads, DIM, generator=gen)).to(torch.bfloat16)
    return q.to(device), kv.to(device), indices.to(device)


def reference_fp64(q, kv, indices):
    """(out, max_logits, lse) in fp64 and the live rows: logits in fp64,
    softmax with the row max, P rounded to bf16 before P.V."""
    T, H, _ = q.shape
    out = torch.zeros(T, H, DIM, dtype=torch.float64, device=q.device)
    mx = torch.full((T, H), float("-inf"), dtype=torch.float64, device=q.device)
    lse = torch.full((T, H), float("-inf"), dtype=torch.float64, device=q.device)
    idx = indices[:, 0, :].long()
    valid = (idx >= 0) & (idx < kv.shape[0])
    for t in range(T):
        sel = idx[t][valid[t]]
        if sel.numel() == 0:
            continue
        rows = kv[sel, 0, :].double()
        s = (q[t].double() @ rows.t()) * SM_SCALE
        m = s.max(1).values
        e = torch.exp(s - m[:, None])
        p = e.to(torch.bfloat16).double()
        den = e.sum(1)
        out[t] = (p @ rows) / den[:, None]
        mx[t] = m
        lse[t] = m + torch.log(den)
    return (out, mx, lse), valid.any(1)


def _run(monkeypatch, flag, q, kv, idx, dpe=0, out=None):
    monkeypatch.setenv(FLAG, flag)
    return tms.triton_mla_sparse_fwd(q, kv, idx, SM_SCALE, d_v=DIM, block_dpe=dpe, out=out)


def _clone(y):
    return tuple(t.clone() for t in y)


def _equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


# ------------------------------------------------------------- GPU: tests
@needs_sm80
@pytest.mark.skipif(not os.environ.get("SMLA_DECODE_V2_BASE_TREE"),
                    reason="SMLA_DECODE_V2_BASE_TREE (a checkout of the base) not set")
def test_flag_off_bitwise_equals_the_base_tree(monkeypatch):
    path = os.path.join(os.environ["SMLA_DECODE_V2_BASE_TREE"],
                        "vllm/sm80/kernels/attention/triton_mla_sparse.py")
    spec = importlib.util.spec_from_file_location("base_triton_mla_sparse", path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    for heads in HEADS:
        for T in SERVED_T:
            for layout in ("short", "kpool"):
                q, kv, idx = make_case(layout, T, heads, seed=T * 7 + heads)
                y = _clone(_run(monkeypatch, "0", q, kv, idx))
                yb = base.triton_mla_sparse_fwd(q, kv, idx, SM_SCALE, d_v=DIM, block_dpe=0)
                assert _equal(y, yb), (layout, T, heads)


@needs_sm80
@pytest.mark.parametrize("heads", HEADS)
def test_flag_off_unchanged_by_flag_on_calls(monkeypatch, heads):
    for T in (1, 4, 8, 16, 32, 64):
        q, kv, idx = make_case("short", T, heads, seed=100 + T)
        y0 = _clone(_run(monkeypatch, "0", q, kv, idx))
        _run(monkeypatch, "1", q, kv, idx)
        assert _equal(_run(monkeypatch, "0", q, kv, idx), y0), T


@needs_sm80
@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("T", SERVED_T)
@pytest.mark.parametrize("width", (2048, WIDTH))
def test_flag_on_accuracy_vs_fp64(monkeypatch, width, T, heads):
    """Per output and shape, the absolute errors against FP64 on live rows,
    pooled over four layouts and two seeds: the new kernel's max <= 1.25x and
    mean <= 1.10x the shipped kernel's, exact where the shipped kernel's pooled
    error is zero, every value finite. (Generated inputs; the served-shape
    verdict is the comparison on captured inputs.)"""
    pools = {name: {"new": [0.0, 0.0], "old": [0.0, 0.0], "n": 0} for name in OUTPUTS}
    for layout in ("short", "kpool", "gap", "masked"):
        for seed in range(2):
            q, kv, idx = make_case(layout, T, heads, seed=1000 * T + 10 * heads + seed,
                                   width=width)
            y1 = _clone(_run(monkeypatch, "1", q, kv, idx))
            assert _equal(_run(monkeypatch, "1", q, kv, idx), y1), "not deterministic"
            buf = torch.full_like(y1[0], float("nan"))
            y3 = _run(monkeypatch, "1", q, kv, idx, out=buf)
            assert y3[0].data_ptr() == buf.data_ptr() and _equal(y3, y1)
            y0 = _clone(_run(monkeypatch, "0", q, kv, idx))
            refs, live = reference_fp64(q, kv, idx)
            for t in y1:
                assert bool(torch.isfinite(t.float()).all()), (layout, seed)
            dead = ~live
            if bool(dead.any()):
                assert bool((y1[0][dead] == 0).all())
                assert bool((y1[1][dead] < -1e20).all()) and bool((y1[2][dead] < -1e20).all())
            if not bool(live.any()):
                continue
            for name, a1, a0, r in zip(OUTPUTS, y1, y0, refs):
                p = pools[name]
                for key, y in (("new", a1), ("old", a0)):
                    d = (y.double() - r)[live].abs()
                    p[key][0] = max(p[key][0], float(d.max()))
                    p[key][1] += float(d.sum())
                p["n"] += int(live.sum()) * (r.numel() // r.shape[0])
    for name, p in pools.items():
        (nm, ns), (om, os_) = p["new"], p["old"]
        assert p["n"] > 0, name
        if om == 0.0:
            assert nm == 0.0, (name, nm)
            continue
        assert nm <= 1.25 * om, (name, "max", nm, om)
        assert ns <= 1.10 * os_, (name, "mean", ns / p["n"], os_ / p["n"])


@needs_sm80
@pytest.mark.parametrize("heads", HEADS)
def test_flag_on_strided_views(monkeypatch, heads):
    """Row and head strides other than the dense ones (a wider query buffer, a
    padded cache row, an index buffer with spare columns, a strided out=):
    bitwise the result on dense copies."""
    for T in (1, 5, 16, 64):
        q, kv, idx = make_case("kpool", T, heads, seed=300 + T)
        y = _clone(_run(monkeypatch, "1", q, kv, idx))
        qw = torch.zeros(T, heads, 2 * DIM, dtype=q.dtype, device=q.device)
        qw[..., :DIM] = q
        kvw = torch.zeros(kv.shape[0], 1, DIM + 64, dtype=kv.dtype, device=kv.device)
        kvw[..., :DIM] = kv
        iw = torch.full((T, 1, 2 * WIDTH), -1, dtype=idx.dtype, device=idx.device)
        iw[..., :WIDTH] = idx
        ow = torch.full((T, heads, 2 * DIM), float("nan"), dtype=q.dtype, device=q.device)
        qs, kvs, ids, os_ = qw[..., :DIM], kvw[..., :DIM], iw[..., :WIDTH], ow[..., :DIM]
        assert tms._decode_v2_closed(qs, kvs, ids, 0, DIM, os_) == ""
        ys = _run(monkeypatch, "1", qs, kvs, ids, out=os_)
        assert ys[0].data_ptr() == os_.data_ptr()
        assert _equal(ys, y), T


@needs_sm80
@pytest.mark.parametrize("case", ["rows65", "heads32", "dpe64", "fp16"])
def test_gate_closed_calls_are_the_shipped_kernel(monkeypatch, case):
    T, heads, dpe = 4, 16, 0
    if case == "rows65":
        T = 65
    elif case == "heads32":
        heads = 32
    q, kv, idx = make_case("short", T, heads, seed=7)
    if case == "dpe64":
        dpe = 64
        q = torch.cat([q, q[..., :64]], dim=-1).contiguous()
        kv = torch.cat([kv, kv[..., :64]], dim=-1).contiguous()
    elif case == "fp16":
        kv = torch.nan_to_num(kv.float()).half()
        q = q.half()
    y0 = _clone(_run(monkeypatch, "0", q, kv, idx, dpe=dpe))
    y1 = _run(monkeypatch, "1", q, kv, idx, dpe=dpe)
    assert _equal(y1, y0)


def _new_contents(q, idx, gen):
    """Other queries, and each row's valid entries moved to the end of the row."""
    q2 = torch.randn(q.shape, generator=gen).to(q.dtype).to(q.device)
    rows = idx[:, 0, :].cpu()
    out = torch.full_like(rows, -1)
    for t in range(rows.shape[0]):
        v = rows[t][rows[t] >= 0]
        if v.numel():
            out[t, -v.numel():] = v
    return q2, out.view_as(idx).to(idx.device)


@needs_sm80
@pytest.mark.parametrize("T,heads", [(1, 16), (4, 16), (8, 16), (16, 16), (32, 16),
                                     (64, 16), (4, 64), (8, 64), (16, 64), (32, 64)])
@pytest.mark.parametrize("width", (2048, WIDTH))
def test_cuda_graph_replay_equals_eager(monkeypatch, width, T, heads):
    q, kv, idx = make_case("short", T, heads, seed=500 + T, width=width)
    monkeypatch.setenv(FLAG, "1")

    def call():
        return tms.triton_mla_sparse_fwd(q, kv, idx, SM_SCALE, d_v=DIM, block_dpe=0)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        eager = _clone(call())
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    # what capturing one in-place op costs, to separate torch's own allocation
    scratch = torch.zeros(16, device="cuda")
    g0 = torch.cuda.CUDAGraph()
    m0 = torch.cuda.memory_allocated()
    with torch.cuda.graph(g0):
        scratch.add_(1)
    overhead = torch.cuda.memory_allocated() - m0

    g = torch.cuda.CUDAGraph()
    m0 = torch.cuda.memory_allocated()
    with torch.cuda.graph(g):
        outs = call()
    own = sum(-(-t.untyped_storage().nbytes() // 512) * 512
              for t in {t.untyped_storage().data_ptr(): t for t in outs}.values())
    assert torch.cuda.memory_allocated() - m0 <= own + overhead

    gen = torch.Generator().manual_seed(T)
    q2, idx2 = _new_contents(q, idx, gen)
    torch.cuda.synchronize()
    m1 = torch.cuda.memory_allocated()
    for t in outs:
        t.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert _equal(outs, eager)
    assert torch.cuda.memory_allocated() == m1, "replay allocated"
    # new contents in the captured buffers: the replay must follow them
    q.copy_(q2)
    idx.copy_(idx2)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == m1, "replay allocated"
    fresh = call()
    assert _equal(outs, fresh)
    del fresh
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == m1, "an eager call left an allocation behind"
