# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Gluon thin GEMM (VLLM_THIN_GEMM_GLUON).

vllm/sm80/kernels/thin_gemm/gluon.py (dispatch, gate, warmup) and
thin_gemm_gluon.py (the kernels and the measured schedule).

CPU (runs anywhere):
    CUDA_VISIBLE_DEVICES="" pytest -q tests/kernels/test_ampere_thin_gemm_gluon.py

  off path: both flags default off; with VLLM_THIN_GEMM_GLUON unset the
  package binds the shipped ``ampere_thin_gemm`` itself and never imports
  gluon.py (checked in a fresh interpreter, flag unset and set); the warmup
  never imports it either. Dispatch: exactly the seven (N, K) of the table at
  M <= 32 with the gate open go to the Gluon module, everything else (other
  shapes, more rows, bias, 3-D input, gate closed) to the wrapped function.
  Layout: a table call reaches the kernels only with a unit K stride, a row
  stride that is a multiple of 16 elements and 16-byte aligned operands (what
  the warmup compiled); a row stride of 8 mod 16, an unaligned base or a K
  stride of 2 takes cuBLAS, with one warning; an unaligned weight too.
  Gate: 74 SMs, compilation mode NONE, CUDA graphs on every step
  (enforce_eager and graph modes with eager steps close it), Gluon ampere
  API; a warmup failure fails the boot; VLLM_THIN_GEMM_CONCAT with the flag
  raises; the banner and the gate-closed line through vLLM's own logger,
  the shipped warmup's count line. Warmup split:
  the Gluon module warms its shapes at M <= 32, the shipped kernel the rest.
  Schedule: the table holds every served row count for every (N, K), its rows
  are the measured ones (pinned), an unserved M takes the next served row;
  the forced-schedule diagnostic; the workspace lane is shared with the
  shipped module; the kernel module imports only torch and triton.

GPU (sm_80 with 74 SMs, skipped otherwise): every (N, K) at every M 1..32
against an FP64 recomputation, pooled per (N, K, M) over seeds, with cuBLAS
(F.linear) as the error budget: mean <= 1.10x, max <= 1.25x, BF16 rounding
ties forgiven (<= 0.1 %), exact where cuBLAS is exact; bitwise run to run and
out= == out=None; the views the layout check admits (row stride K + 16, rows
at a whole-row offset) give the contiguous call's result bit for bit, the
ones it refuses give F.linear's through the dispatch; CUDA-graph replay bitwise equal
to eager with no allocation in capture or replay; split-K arrival counters
back at zero after every call; the forced schedule makes each row's result
independent of the call's row count; the dispatch end to end through the
warmup gate.
"""

import contextlib
import hashlib
import logging
import os
import subprocess
import sys
import types

import pytest
import torch

from vllm import envs

GLUON = "vllm.sm80.kernels.thin_gemm.gluon"
KERNEL = "vllm.sm80.kernels.thin_gemm.thin_gemm_gluon"
BF16 = torch.bfloat16
NK = [(336, 10240), (320, 10240), (10240, 320), (512, 2560), (1, 2560), (48, 2560), (12800, 2560)]


def _gluon():
    from vllm.sm80.kernels.thin_gemm import gluon

    return gluon


def _kernel():
    from vllm.sm80.kernels.thin_gemm import thin_gemm_gluon

    return thin_gemm_gluon


@pytest.fixture
def g(monkeypatch):
    """gluon.py with a fresh impl() cache and the gate closed after the test."""
    gl = _gluon()
    gl.impl.cache_clear()
    monkeypatch.setattr(gl, "_LIVE", [False])
    yield gl
    # a test may have replaced impl; the next setup clears the real one
    getattr(gl.impl, "cache_clear", lambda: None)()
    _kernel()._FORCE_M[0] = 0


def _worker(mode_name="NONE", device="cpu", eager=False, graphs="PIECEWISE"):
    from vllm.config import CompilationMode, CUDAGraphMode

    cc = types.SimpleNamespace(mode=getattr(CompilationMode, mode_name),
                               cudagraph_mode=getattr(CUDAGraphMode, graphs))
    return types.SimpleNamespace(
        vllm_config=types.SimpleNamespace(
            compilation_config=cc,
            model_config=types.SimpleNamespace(enforce_eager=eager)),
        device=device,
    )


# ------------------------------------------------------------------ CPU ----
def test_flags_default_off():
    assert envs.VLLM_THIN_GEMM_GLUON is False
    assert envs.VLLM_THIN_GEMM_GLUON_FORCE_M == 0


_BINDING = (
    "import sys, vllm.sm80.kernels.thin_gemm as t; "
    "print(t.ampere_thin_gemm.__module__, hasattr(t.ampere_thin_gemm, 'shipped'), "
    "'vllm.sm80.kernels.thin_gemm.gluon' in sys.modules)"
)


@pytest.mark.parametrize("flag,expect", [
    (None, "vllm.sm80.kernels.thin_gemm False False"),
    ("0", "vllm.sm80.kernels.thin_gemm False False"),
    ("1", "vllm.sm80.kernels.thin_gemm.gluon True True"),
])
def test_package_binding_in_a_fresh_interpreter(flag, expect):
    """Unset (or 0), the package binds the shipped function itself and never
    imports gluon.py; set, it binds gluon.py's wrapper around it."""
    env = {k: v for k, v in os.environ.items() if k != "VLLM_THIN_GEMM_GLUON"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    if flag is not None:
        env["VLLM_THIN_GEMM_GLUON"] = flag
    out = subprocess.run([sys.executable, "-c", _BINDING], env=env, check=True,
                         capture_output=True, text=True).stdout.strip().splitlines()[-1]
    assert out == expect


def test_concat_and_gluon_raise_in_a_fresh_interpreter():
    """Both flags set: importing the package raises (the concat groups pair
    only the shipped thin GEMM's calls)."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", VLLM_THIN_GEMM_GLUON="1",
               VLLM_THIN_GEMM_CONCAT="1")
    run = subprocess.run([sys.executable, "-c", _BINDING], env=env,
                         capture_output=True, text=True)
    assert run.returncode != 0
    assert ("VLLM_THIN_GEMM_GLUON and VLLM_THIN_GEMM_CONCAT are both set"
            in run.stderr), run.stderr[-2000:]


def test_concat_and_gluon_do_not_combine(g, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM_CONCAT", True, raising=False)
    with pytest.raises(ValueError, match="VLLM_THIN_GEMM_CONCAT are both set"):
        g.wrap_dispatch(lambda layer, x, w, bias=None: None)


def test_warmup_never_imports_gluon_when_off(monkeypatch):
    from vllm.sm80.kernels.thin_gemm import warmup as wu

    monkeypatch.setattr(envs, "VLLM_THIN_GEMM_GLUON", False, raising=False)
    # restored at teardown, so later tests patch the module the code imports
    monkeypatch.delitem(sys.modules, GLUON, raising=False)
    import vllm.sm80.kernels.thin_gemm as tg

    monkeypatch.setattr(tg, "use_ampere_thin_gemm", lambda: False)
    wu.warmup_ampere_thin_gemm(_worker(), [1, 4, 32])
    monkeypatch.setattr(tg, "use_ampere_thin_gemm", lambda: True)
    seen = []
    monkeypatch.setattr(tg, "impl", lambda: types.SimpleNamespace(
        warmup=lambda nk, ms, device=None: seen.append((tuple(nk), tuple(ms)))))
    monkeypatch.setattr(wu, "_discover_nk", lambda worker: {(336, 10240)})
    wu.warmup_ampere_thin_gemm(_worker(), [1, 4, 32])
    assert GLUON not in sys.modules
    assert (336, 10240) in seen[-1][0]


def test_kernel_module_imports_only_torch_and_triton():
    """The vendored file stays runnable outside vLLM (the standalone gate
    and benchmark load it as their candidate)."""
    import ast

    src = open(_kernel().__file__).read()
    roots = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add(node.module.split(".")[0])
    assert roots <= {"contextlib", "torch", "triton"}, roots


# The schedule rows as measured (a change here is a change nothing measured).
_TABLE_SHA256 = "872d2944c741f54b055c0b233bbded72aa0de5f8cffde306ee8794a050472ba4"


def _table_digest(table):
    return hashlib.sha256(repr(sorted(table.items())).encode()).hexdigest()


def test_table_is_the_measured_one():
    k = _kernel()
    assert _table_digest(k._CONFIG_OVERRIDES) == _TABLE_SHA256
    assert k._SERVED_M == _gluon().SERVED_M
    assert _gluon().MAX_M == max(k._SERVED_M)


def test_shapes_are_the_table(g):
    k = _kernel()
    table_nk = {(n, kk) for n, kk, _ in k._CONFIG_OVERRIDES}
    assert g.SHAPES == frozenset(NK)
    assert g.SHAPES == table_nk | {(n, kk) for n, kk in NK if n <= k._GEMV_MAX_N}
    for n, kk in table_nk:
        for m in k._SERVED_M:
            assert (n, kk, m) in k._CONFIG_OVERRIDES, (n, kk, m)


def test_unserved_rows_take_the_next_served_row(monkeypatch):
    k = _kernel()
    monkeypatch.setattr(k, "_num_sms_cache", 74)
    for (n, kk, m), row in k._CONFIG_OVERRIDES.items():
        lo = max([s for s in k._SERVED_M if s < m], default=0)
        for mm in range(lo + 1, m + 1):
            assert k._table_row(mm, n, kk) == row, (n, kk, mm)
            bm, bn, bk, sk, warps, stages = k._select_config(mm, n, kk)
            assert (bn, bk, sk, warps) == row[:4]
            assert bm == max(16, 1 << (mm - 1).bit_length())
    assert k._table_row(33, 336, 10240) is None


def test_forced_schedule(g, monkeypatch):
    k = _kernel()
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM_GLUON_FORCE_M", 24, raising=False)
    assert g.force_m() == 24
    assert g.impl()._FORCE_M[0] == 24
    for n, kk in NK:
        if n == 1:
            continue
        for m in range(1, 25):
            assert k._select_config(m, n, kk) == k._select_config(24, n, kk)
            assert k._gluon_schedule(m, n, kk) == k._gluon_schedule(24, n, kk)
        assert k._table_row(32, n, kk) == k._CONFIG_OVERRIDES[(n, kk, 32)]
    k._FORCE_M[0] = 0
    assert k._select_config(1, 336, 10240)[0] == 16
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM_GLUON_FORCE_M", 20, raising=False)
    assert g.force_m() == 0


def test_workspace_lane_is_shared_with_the_shipped_module(g):
    from vllm.sm80.kernels.thin_gemm import impl as shipped

    k = g.impl()
    assert k._LANE is shipped()._LANE
    assert k._lane_key(0, 7) == (0, 7)
    with shipped().workspace_lane(1):
        assert k._lane_key(0, 7) == ("lane", 1, 0, 7)
    assert k._lane_key(0, 7) == (0, 7)


def _route(g, monkeypatch, live=True):
    calls = []
    monkeypatch.setattr(g, "impl", lambda: types.SimpleNamespace(
        thin_gemm=lambda x, w: calls.append(("gluon", x.shape[0], tuple(w.shape))) or "gluon"))
    g._LIVE[0] = live

    def shipped(layer, x, w, bias=None):
        calls.append(("shipped", x.shape[0], tuple(w.shape)))
        return "shipped"

    return g.wrap_dispatch(shipped), calls


def test_dispatch_sends_exactly_the_table(g, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM_MAX_TOKENS", 64, raising=False)
    fn, calls = _route(g, monkeypatch)
    for n, kk in NK:
        w = torch.empty(n, kk, dtype=BF16)
        for m in (1, 3, 4, 17, 24, 31, 32):
            assert fn(None, torch.empty(m, kk, dtype=BF16), w) == "gluon", (n, kk, m)
        for m in (33, 40, 64):
            assert fn(None, torch.empty(m, kk, dtype=BF16), w) == "shipped", (n, kk, m)
        assert fn(None, torch.empty(4, kk, dtype=BF16), w,
                  torch.empty(n, dtype=BF16)) == "shipped"
        assert fn(None, torch.empty(1, 4, kk, dtype=BF16), w) == "shipped"
        assert fn(None, torch.empty(4, kk, dtype=torch.float16),
                  w.to(torch.float16)) == "shipped"
    for n, kk in ((4096, 4096), (336, 2560), (1, 10240), (288, 4096)):
        w = torch.empty(n, kk, dtype=BF16)
        assert fn(None, torch.empty(4, kk, dtype=BF16), w) == "shipped", (n, kk)
    assert fn.shipped is not None


# Views of x: what the layout check admits and what it refuses.
_ADMITTED = ("row_pad16", "row_offset")
_REFUSED = ("row_pad24", "unaligned", "k_strided")


def _layout(kind, x):
    """``x`` [M, K] copied into a view of the named layout (same values)."""
    m, k = x.shape
    nan = dict(fill_value=float("nan"), dtype=x.dtype, device=x.device)
    if kind in ("row_pad16", "row_pad24"):
        xa = torch.full((m, k + int(kind[-2:])), **nan)[:, :k]
    elif kind == "row_offset":
        xa = torch.full((m + 1, k), **nan)[1:]
    elif kind == "unaligned":
        xa = torch.empty(m * k + 1, dtype=x.dtype, device=x.device)[1:].view(m, k)
    elif kind == "k_strided":
        xa = torch.empty(m, k, 2, dtype=x.dtype, device=x.device)[..., 0]
    else:
        raise ValueError(kind)
    xa.copy_(x)
    return xa


@pytest.mark.parametrize("kind", _ADMITTED + _REFUSED)
def test_dispatch_layout_gate(g, monkeypatch, caplog, kind):
    """A table call runs the kernels only in the layout they were compiled
    for; any other layout takes cuBLAS (F.linear), never a new variant."""
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM_MAX_TOKENS", 32, raising=False)
    fn, calls = _route(g, monkeypatch)
    w = torch.randn(48, 2560).to(BF16)
    xa = _layout(kind, torch.randn(4, 2560).to(BF16))
    assert g.gluon_layout_ok(xa, w) is (kind in _ADMITTED)
    with _vllm_log(caplog):
        out = fn(None, xa, w)
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    if kind in _ADMITTED:
        assert out == "gluon" and calls == [("gluon", 4, (48, 2560))]
        assert warned == []
    else:
        assert calls == []
        assert torch.equal(out, torch.nn.functional.linear(xa, w))
        # (the vllm logger's handler and pytest's both see each record)
        assert set(warned) == {
            f"Gluon thin GEMM: a (N, K) = (48, 2560) call with x strides {tuple(xa.stride())} "
            f"and base offset {xa.data_ptr() % 16} mod 16 bytes is not in the layout its "
            "kernels were compiled for; such calls take cuBLAS"}


def test_layout_gate_checks_the_weight(g):
    x = torch.empty(4, 2560, dtype=BF16)
    assert g.gluon_layout_ok(x, torch.empty(48, 2560, dtype=BF16))
    unaligned = torch.empty(48 * 2560 + 1, dtype=BF16)[1:].view(48, 2560)
    assert not g.gluon_layout_ok(x, unaligned)
    padded = torch.empty(48, 2568, dtype=BF16)[:, :2560]
    assert not g.gluon_layout_ok(x, padded)


def test_dispatch_follows_the_shipped_bound(g, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM_MAX_TOKENS", 8, raising=False)
    fn, _ = _route(g, monkeypatch)
    w = torch.empty(512, 2560, dtype=BF16)
    assert fn(None, torch.empty(8, 2560, dtype=BF16), w) == "gluon"
    assert fn(None, torch.empty(16, 2560, dtype=BF16), w) == "shipped"


def test_dispatch_gate_closed_is_the_shipped_path(g, monkeypatch):
    fn, calls = _route(g, monkeypatch, live=False)
    w = torch.empty(512, 2560, dtype=BF16)
    assert fn(None, torch.empty(4, 2560, dtype=BF16), w) == "shipped"
    assert calls == [("shipped", 4, (512, 2560))]


def _sms(monkeypatch, n):
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: types.SimpleNamespace(multi_processor_count=n))


def test_gate_reasons(g, monkeypatch):
    _sms(monkeypatch, 74)
    assert g.closed_reason("cpu", _worker().vllm_config) == ""
    assert (g.closed_reason("cpu", _worker("VLLM_COMPILE").vllm_config)
            == "compilation mode is not NONE")
    for graphs in ("PIECEWISE", "FULL", "FULL_AND_PIECEWISE"):
        assert g.closed_reason("cpu", _worker(graphs=graphs).vllm_config) == "", graphs
    assert (g.closed_reason("cpu", _worker(eager=True, graphs="NONE").vllm_config)
            == "enforce_eager runs every step eagerly (unpadded row counts)")
    for graphs in ("NONE", "FULL_DECODE_ONLY"):
        assert (g.closed_reason("cpu", _worker(graphs=graphs).vllm_config)
                == f"cudagraph_mode {graphs} runs steps eagerly (unpadded row counts)")
    _sms(monkeypatch, 70)
    assert (g.closed_reason("cpu", _worker().vllm_config)
            == "70 SMs, the schedule is measured at 74")
    _sms(monkeypatch, 74)
    monkeypatch.setitem(sys.modules, "triton.experimental.gluon.language.nvidia.ampere", None)
    reason = g.closed_reason("cpu", _worker().vllm_config)
    assert reason.startswith("Triton Gluon ampere API missing ("), reason
    assert "triton.experimental.gluon.language.nvidia.ampere" in reason, reason


@contextlib.contextmanager
def _vllm_log(caplog):
    import vllm.logger as vllm_logger

    root = logging.getLogger("vllm")
    root.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="vllm")
    for once in (vllm_logger._print_info_once, vllm_logger._print_warning_once):
        once.cache_clear()
    try:
        yield
    finally:
        root.removeHandler(caplog.handler)


def _fake_kernels(g, monkeypatch, fail=False):
    seen = []

    def warm(name):
        def run(shapes, ms, device=None):
            if fail and name == "gluon":
                raise RuntimeError("compile error")
            seen.append((name, tuple(shapes), tuple(ms)))
        return run

    fake = types.SimpleNamespace(warmup=warm("gluon"), _FORCE_M=[0])
    monkeypatch.setattr(g, "impl", lambda: fake)
    monkeypatch.setattr(g, "shipped_impl",
                        lambda: types.SimpleNamespace(warmup=warm("shipped")))
    return seen


def test_warmup_banner_and_split_through_the_vllm_logger(g, monkeypatch, caplog):
    _sms(monkeypatch, 74)
    seen = _fake_kernels(g, monkeypatch)
    with _vllm_log(caplog):
        served = g.warmup_table(_worker(), [1, 2, 4, 32, 40])
    assert served == g.SHAPES and g._LIVE[0] is True
    assert seen == [("gluon", tuple(sorted(g.SHAPES)), (1, 2, 4, 32)),
                    ("shipped", tuple(sorted(g.SHAPES)), (40,))]
    lines = [(r.levelno, r.name, r.getMessage()) for r in caplog.records]
    assert (logging.INFO, GLUON,
            "Gluon thin GEMM active [VLLM_THIN_GEMM_GLUON=1]: 7 (N, K) x M in "
            "(1, 2, 4, 32) (up to 32 rows)") in lines


def test_warmup_gate_closed_lines(g, monkeypatch, caplog):
    seen = _fake_kernels(g, monkeypatch)
    _sms(monkeypatch, 74)
    with _vllm_log(caplog):
        assert g.warmup_table(_worker("VLLM_COMPILE"), [4]) == frozenset()
        assert g.warmup_table(_worker(eager=True, graphs="NONE"), [4]) == frozenset()
        _sms(monkeypatch, 70)
        assert g.warmup_table(_worker(), [4]) == frozenset()
    assert g._LIVE[0] is False and seen == []
    lines = [(r.levelno, r.getMessage()) for r in caplog.records]
    tail = "; the VLLM_THIN_GEMM path serves every call"
    assert (logging.WARNING, "VLLM_THIN_GEMM_GLUON set but off: compilation mode "
            "is not NONE" + tail) in lines
    assert (logging.WARNING, "VLLM_THIN_GEMM_GLUON set but off: enforce_eager runs "
            "every step eagerly (unpadded row counts)" + tail) in lines
    assert (logging.WARNING, "VLLM_THIN_GEMM_GLUON set but off: 70 SMs, the "
            "schedule is measured at 74" + tail) in lines


def test_warmup_failure_fails_the_boot(g, monkeypatch):
    """Every rank must serve the same kernel: a warmup failure with the flag
    set raises instead of leaving this rank on the shipped path."""
    _sms(monkeypatch, 74)
    _fake_kernels(g, monkeypatch, fail=True)
    said = []
    monkeypatch.setattr(g, "set_but_off", said.append)
    with pytest.raises(RuntimeError, match=r"VLLM_THIN_GEMM_GLUON=1: the Gluon thin GEMM "
                       r"warmup failed \(RuntimeError: compile error\)"):
        g.warmup_table(_worker(), [4])
    assert g._LIVE[0] is False
    assert said == []


def test_warmup_py_hands_the_table_to_gluon(g, monkeypatch, caplog):
    from vllm.sm80.kernels.thin_gemm import warmup as wu
    import vllm.sm80.kernels.thin_gemm as tg

    monkeypatch.setattr(envs, "VLLM_THIN_GEMM_GLUON", True, raising=False)
    monkeypatch.setattr(tg, "use_ampere_thin_gemm", lambda: True)
    seen = []
    monkeypatch.setattr(tg, "impl", lambda: types.SimpleNamespace(
        warmup=lambda nk, ms, device=None: seen.append(("shipped", tuple(nk)))))
    monkeypatch.setattr(wu, "_discover_nk", lambda worker: {(336, 10240), (4096, 4096)})
    monkeypatch.setattr(g, "warmup_table",
                        lambda worker, ms: seen.append(("gluon", tuple(ms))) or g.SHAPES)
    with _vllm_log(caplog):
        wu.warmup_ampere_thin_gemm(_worker(), [1, 4, 32])
    assert seen[0] == ("gluon", (1, 2, 3, 4, 8, 32))
    assert (4096, 4096) in seen[1][1]
    assert not set(seen[1][1]) & g.SHAPES
    # 2 discovered + 18 from the backstop (its 19 shapes but (4096, 4096));
    # the shipped kernel warms them without the Gluon-owned (336, 10240).
    assert ("Warmed up the sm_80 thin-M BF16 GEMM: 19 shapes (2 discovered on "
            "the live model, 18 from the shape-table backstop) x M in "
            "[1, 2, 3, 4, 8, 32] (bound 32).") in [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize("thin_gemm,reason", [
    (False, "VLLM_THIN_GEMM is off"), (True, "not an sm_80 device")])
def test_warmup_py_flag_set_but_dispatch_off(g, monkeypatch, thin_gemm, reason):
    from vllm.sm80.kernels.thin_gemm import warmup as wu
    import vllm.sm80.kernels.thin_gemm as tg

    monkeypatch.setattr(envs, "VLLM_THIN_GEMM_GLUON", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM", thin_gemm, raising=False)
    monkeypatch.setattr(tg, "use_ampere_thin_gemm", lambda: False)
    said = []
    monkeypatch.setattr(g, "set_but_off", said.append)
    wu.warmup_ampere_thin_gemm(_worker(), [4])
    assert said == [reason]


# ------------------------------------------------------------------ GPU ----
def _gpu_ok():
    return (torch.cuda.is_available()
            and torch.cuda.get_device_capability(0) == (8, 0)
            and torch.cuda.get_device_properties(0).multi_processor_count == 74)


gpu = pytest.mark.skipif(not _gpu_ok(), reason="needs an sm_80 GPU with 74 SMs")

# Input scales of the served calls (x std, w std) per (N, K).
_SCALE = {(336, 10240): (4.8, 0.036), (320, 10240): (4.8, 0.036),
          (10240, 320): (1.0, 0.05), (512, 2560): (1.0, 0.02), (1, 2560): (1.0, 0.02),
          (48, 2560): (1.0, 0.02), (12800, 2560): (1.0, 0.02)}


def _inputs(seed, m, n, k):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    xs, ws = _SCALE[(n, k)]
    x = (torch.randn(m, k, generator=gen, device="cuda") * xs).to(BF16)
    w = (torch.randn(n, k, generator=gen, device="cuda") * ws).to(BF16)
    return x, w


def _pool():
    from tests.kernels.quantization.test_exl3_dense_splitk import _Pool

    return _Pool()


def _ref(x, w):
    return x.double() @ w.double().T


def _bitwise(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@gpu
@pytest.mark.parametrize("nk", NK, ids=[f"{n}x{k}" for n, k in NK])
def test_gpu_fp64_gate_every_row_count(nk):
    k_mod = _kernel()
    n, k = nk
    for m in range(1, 33):
        pool = _pool()
        for seed in range(max(2, min(8, 4096 // (m * n) + 1))):
            x, w = _inputs(1000 * m + seed, m, n, k)
            c = k_mod.thin_gemm(x, w)
            again = torch.empty_like(c)
            k_mod.thin_gemm(x, w, out=again)
            assert _bitwise(c, again), (n, k, m, "run to run / out=")
            pool.add(c, torch.nn.functional.linear(x, w), _ref(x, w))
        pool.check(f"{n}x{k} M{m}")


@gpu
@pytest.mark.parametrize("nk", NK, ids=[f"{n}x{k}" for n, k in NK])
@pytest.mark.parametrize("kind", _ADMITTED)
def test_gpu_admitted_layouts(nk, kind):
    """Views the layout check admits run the warmed kernels: bit for bit the
    contiguous call's result."""
    k_mod = _kernel()
    n, k = nk
    for m in (1, 4, 17, 24, 32):
        x, w = _inputs(7 * m, m, n, k)
        xa = _layout(kind, x)
        assert _gluon().gluon_layout_ok(xa, w), (n, k, m, kind)
        assert _bitwise(k_mod.thin_gemm(xa, w), k_mod.thin_gemm(x, w)), (n, k, m, kind)


@gpu
@pytest.mark.parametrize("kind", _REFUSED)
def test_gpu_refused_layouts_take_cublas(g, monkeypatch, kind):
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM_MAX_TOKENS", 32, raising=False)

    def no_kernel(x, w):
        raise AssertionError("a refused layout reached the Gluon kernels")

    monkeypatch.setattr(g, "impl", lambda: types.SimpleNamespace(thin_gemm=no_kernel))
    g._LIVE[0] = True
    fn = g.wrap_dispatch(lambda layer, x, w, bias=None: pytest.fail("left the table path"))
    for n, k in NK:
        for m in (1, 4, 17, 24, 32):
            x, w = _inputs(7 * m, m, n, k)
            xa = _layout(kind, x)
            assert _bitwise(fn(None, xa, w), torch.nn.functional.linear(xa, w)), (n, k, m, kind)


@gpu
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 24, 32])
def test_gpu_graph_replay_bitwise_no_allocation(m):
    k_mod = _kernel()
    k_mod.warmup(NK, [m])
    cases = [(_inputs(m + i, m, n, k), n) for i, (n, k) in enumerate(NK)]
    outs = [torch.empty(m, n, dtype=BF16, device="cuda") for (_, n) in cases]
    eager = [k_mod.thin_gemm(x, w) for (x, w), _ in cases]
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    with torch.cuda.graph(graph, stream=stream):
        for ((x, w), _), o in zip(cases, outs):
            k_mod.thin_gemm(x, w, out=o)
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    used = torch.cuda.memory_allocated()
    for _ in range(3):
        for o in outs:
            o.zero_()
        graph.replay()
        torch.cuda.synchronize()
        for e, o in zip(eager, outs):
            assert _bitwise(e, o), m
    assert torch.cuda.memory_allocated() == used


@gpu
def test_gpu_arrival_counters_back_at_zero():
    k_mod = _kernel()
    for n, k in NK:
        for m in (1, 2, 4, 8, 16, 24, 32):
            k_mod.thin_gemm(*_inputs(m, m, n, k))
    torch.cuda.synchronize()
    locks = [v for key, v in k_mod._WORKSPACE.items() if "lock" in key]
    assert locks and all(int(v.abs().sum()) == 0 for v in locks)


@gpu
@pytest.mark.parametrize("force", [8, 24, 32])
def test_gpu_forced_schedule_rows_independent_of_row_count(force):
    k_mod = _kernel()
    k_mod._FORCE_M[0] = force
    try:
        for n, k in NK:
            x, w = _inputs(force, force, n, k)
            full = k_mod.thin_gemm(x, w)
            for m in (1, 3, force // 2, force - 1):
                if m < 1:
                    continue
                part = k_mod.thin_gemm(x[:m].contiguous(), w)
                assert _bitwise(part, full[:m]), (n, k, m, force)
    finally:
        k_mod._FORCE_M[0] = 0


@gpu
def test_gpu_dispatch_end_to_end(g, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_THIN_GEMM_MAX_TOKENS", 64, raising=False)
    shipped_calls = []

    def shipped(layer, x, w, bias=None):
        shipped_calls.append(x.shape[0])
        return torch.nn.functional.linear(x, w, bias)

    fn = g.wrap_dispatch(shipped)
    device = torch.device("cuda", torch.cuda.current_device())
    assert g.warmup_table(_worker(device=device), [1, 2, 4, 8, 16, 24, 32, 40]) == g.SHAPES
    for n, k in NK:
        for m in (1, 5, 24, 32):
            x, w = _inputs(m, m, n, k)
            assert _bitwise(fn(None, x, w), g.impl().thin_gemm(x, w)), (n, k, m)
        x, w = _inputs(40, 40, n, k)
        fn(None, x, w)
    assert shipped_calls == [40] * len(NK)
