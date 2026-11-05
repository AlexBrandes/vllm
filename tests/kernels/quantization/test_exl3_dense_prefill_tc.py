# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 dense prefill on the tensor cores (VLLM_EXL3_DENSE_PREFILL_TC,
csrc/exl3/dense_prefill_tc.cu, vllm_cmp170hx/kernels/exl3/dense_prefill_tc.py).

CPU: the switch and its schedule diagnostic default, the gate, the one-time
resolution with the active and gate-closed log lines (through vLLM's own
logger), resolution once per process by ensure_extension (with the gate open
through the real resolve and warm-up, which never call back into
ensure_extension), and _linear's operator per row chunk: with the switch
off exactly the base's operator;
with it on, linear_out_prefill only in prefill steps for launches the
operator covers and split-K does not split, never in a decode step of more
than 64 tokens, during a graph capture or without a forward context. The
library's host rule (linear_prefill_applies) and the served launches of the
3460 / 1906-token prefill chunks: the k/v 482-row tail keeps split-K.

GPU (SM 8.0, 74 multiprocessors): linear_out_prefill against an FP64
reference within linear_out's error (pooled mean <= 1.10x, max <= 1.25x,
BF16 rounding ties forgiven, exact where linear_out is exact) on whole-tile
and stream-K schedules; the k split ran (kernel names); bitwise
deterministic, also while another stream keeps multiprocessors busy (the
split tiles' last-arriving block changes, the sum order does not); the
whole-tile diagnostic; refusals; _linear end to end (off: linear_out; on:
each chunk the operator of its launch); graph replay bitwise like eager with
no allocator event and a NaN-filled workspace (the fixup counters are zeroed
inside the call); ensure_extension with the switch set at TP2 opens the gate
and warms up once.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.forward_context as forward_context
from tests.kernels.quantization.test_exl3_dense_splitk import (
    _allocations,
    _bitwise,
    _extension_or_skip,
    _inputs,
    _matrix,
    _Pool,
    _run,
    _run_core,
    needs_sm80,
)
from vllm_cmp170hx.kernels.exl3 import dense_prefill_tc as tc
from vllm_cmp170hx.kernels.exl3 import ops as exl3_ops

SMS = 74
DECODE_QUERY_LEN = 4  # one sampled token plus three drafts under verification
# (padded_in, padded_out, bits_x2) of the rank's served dense matrices.
KV, GATE, INDEXER, FC, DOWN, OUT, Z, QKV, Q = (
    (2560, 256, 14),
    (2560, 640, 14),
    (2560, 640, 10),
    (2560, 1280, 12),
    (640, 2560, 14),
    (3072, 2560, 14),
    (2560, 3072, 14),
    (2560, 5120, 14),
    (2560, 6144, 14),
)
# Every launch of the 8,826-token reference prompt in 3460 / 1906-token chunks
# (row chunk floor(8 Mi / (K + N))) -> the path with split-K on.
SERVED_LAUNCHES = [
    (KV, 2978, "tc"),
    (KV, 482, "splitk"),
    (KV, 1906, "tc"),
    (GATE, 2621, "tc"),
    (GATE, 839, "tc"),
    (GATE, 1906, "tc"),
    (INDEXER, 2621, "tc"),
    (INDEXER, 839, "tc"),
    (INDEXER, 1906, "tc"),
    (FC, 2184, "tc"),
    (FC, 1276, "tc"),
    (FC, 1906, "tc"),
    (FC, 736, "tc"),
    (FC, 1072, "tc"),
    (DOWN, 2621, "tc"),
    (DOWN, 839, "tc"),
    (DOWN, 1906, "tc"),
    (OUT, 1489, "tc"),
    (OUT, 482, "tc"),
    (OUT, 417, "tc"),
    (Z, 1489, "tc"),
    (Z, 482, "tc"),
    (Z, 417, "tc"),
    (QKV, 1092, "tc"),
    (QKV, 184, "tc"),
    (QKV, 814, "tc"),
    (Q, 963, "tc"),
    (Q, 571, "tc"),
    (Q, 943, "tc"),
]


def _attention(max_query_len):
    return SimpleNamespace(max_query_len=max_query_len)


def _step(monkeypatch, max_query_len):
    """A forward context whose scheduler gave its longest request
    max_query_len tokens (None: no forward context)."""
    context = (
        None
        if max_query_len is None
        else SimpleNamespace(
            attn_metadata={"layers.3.self_attn": _attention(max_query_len)}
        )
    )
    monkeypatch.setattr(forward_context, "_forward_context", context)


# ---------------------------------------------------------------- CPU tests


def test_switch_and_schedule_default_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_EXL3_DENSE_PREFILL_TC", raising=False)
    monkeypatch.delenv("VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE", raising=False)
    assert envs.VLLM_EXL3_DENSE_PREFILL_TC is False
    assert envs.VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE == "planned"
    monkeypatch.setenv("VLLM_EXL3_DENSE_PREFILL_TC", "1")
    monkeypatch.setenv("VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE", "whole")
    assert envs.VLLM_EXL3_DENSE_PREFILL_TC is True
    assert envs.VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE == "whole"
    monkeypatch.setenv("VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE", "split")
    with pytest.raises(ValueError):
        _ = envs.VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE
    assert set(tc.SCHEDULES) == {"planned", "whole"}


def _config(tp=2, pp=1, method="mtp", parallel_drafting=False):
    speculative = (
        None
        if method is None
        else SimpleNamespace(method=method, parallel_drafting=parallel_drafting)
    )
    return SimpleNamespace(
        num_speculative_tokens=0 if method is None else 3,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=pp
        ),
        speculative_config=speculative,
    )


def test_gate_opens_only_on_the_validated_configuration():
    config = _config()
    assert tc.closed_reason((8, 0), 74, config) is None
    assert tc.closed_reason((8, 0), 74, _config(method=None)) is None
    assert tc.closed_reason((8, 0), 74, _config(method="eagle3")) is None
    for capability, sms, candidate in (
        ((8, 6), 74, config),
        (None, None, config),
        ((8, 0), 70, config),
        ((8, 0), 74, None),
        ((8, 0), 74, _config(tp=1)),
        ((8, 0), 74, _config(pp=2)),
        ((8, 0), 74, _config(method="dflash")),
        ((8, 0), 74, _config(parallel_drafting=True)),
    ):
        assert tc.closed_reason(capability, sms, candidate) is not None


def _glm_config():
    config = _config(tp=1, pp=2, method="dflash", parallel_drafting=True)
    config.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_text_config=SimpleNamespace(model_type="glm5_next_text", hidden_size=4096),
    )
    return config


def test_glm_qkv_gate_rejects_unchecked_model_dtype_and_drafter():
    config = _glm_config()
    assert tc.closed_reason((8, 0), SMS, config) is None
    for change in ("model", "width", "dtype", "drafter", "depth"):
        config = _glm_config()
        if change == "model":
            config.model_config.hf_text_config.model_type = "other"
        elif change == "width":
            config.model_config.hf_text_config.hidden_size = 2560
        elif change == "dtype":
            config.model_config.dtype = torch.float16
        elif change == "drafter":
            config.speculative_config.method = "mtp"
        else:
            config.num_speculative_tokens = 9  # a depth whose 10-row verify exceeds 8
        assert tc.closed_reason((8, 0), SMS, config) is not None


def test_glm_gate_admits_retained_dflash_depths():
    """Opt-in eligibility for the checked DFlash depths: 3, 5 and 7 (verify rows
    4, 6 and 8, all within the small-row schedule's 1..8 bound) open the GLM
    gate; every other depth, including an even one and one whose verify would
    exceed 8 rows, keeps it closed. Eligibility only -- each depth is qualified
    separately before measurement."""
    assert set(tc.GLM_DFLASH_DEPTHS) == {3, 5, 7}
    for depth in tc.GLM_DFLASH_DEPTHS:
        config = _glm_config()
        config.num_speculative_tokens = depth
        assert tc.closed_reason((8, 0), SMS, config) is None, depth
        assert tc.decode_closed_reason((8, 0), SMS, config) is None, depth
    for depth in (0, 1, 2, 4, 6, 8, 9, 11):
        config = _glm_config()
        config.num_speculative_tokens = depth
        assert tc.closed_reason((8, 0), SMS, config) is not None, depth
        assert tc.decode_closed_reason((8, 0), SMS, config) is not None, depth


def test_only_validated_matrices_are_taken():
    _extension_or_skip()
    validated = tc.VALIDATED_MATRICES[(2, 1)]
    for padded_in, padded_out, bits_x2 in validated:
        assert tc.takes_launch(
            1489, padded_in, padded_out, bits_x2, 2, False, SMS, validated
        )
        assert not tc.takes_launch(
            1489, padded_in, padded_out, bits_x2, 1, False, SMS, validated
        )
    # Full-width matrices of a one-card layout, and a validated shape stored
    # at another rate: the operator covers them, the gate does not.
    for padded_in, padded_out, bits_x2 in (
        (2560, 512, 14),
        (2560, 12288, 14),
        (2560, 2560, 14),
        (2560, 6144, 12),
    ):
        assert tc.linear_prefill_applies(2730, padded_in, padded_out, bits_x2, 2)
        assert not tc.takes_launch(
            2730, padded_in, padded_out, bits_x2, 2, False, SMS, validated
        )


GLM_DENSE_FAMILIES = (
    (4096, 24576),  # attention qkv
    (1536, 16384),  # attention q_b
    (8192, 4096),  # attention out, 8K-K layers
    (16384, 4096),  # attention out, 16K-K layers
    (4096, 12288),  # dense-layer mlp gate/up
    (12288, 4096),  # dense-layer mlp down
    (4096, 2048),  # shared-expert gate/up
    (2048, 4096),  # shared-expert down
    (1536, 4096),  # sparse-attention indexer wq_b
)


def test_glm_dense_family_set_is_explicit():
    """The served set is the fail-closed contract: every listed family is
    covered, nothing else is. CPU-only; the operator's agreement is the GPU
    test below."""
    glm = tc.VALIDATED_MATRICES[(1, 2)]
    assert {b for _, _, b in glm} == {12}
    assert set(GLM_DENSE_FAMILIES) <= {(i, o) for i, o, _ in glm}
    for padded_in, padded_out in (
        (4096, 154880),
        (4096, 1536),
        (4096, 512),
        (1024, 1024),
    ):
        assert (padded_in, padded_out, 12) not in glm


def test_glm_dense_launch_boundaries_and_layout_isolation():
    _extension_or_skip()
    glm = tc.VALIDATED_MATRICES[(1, 2)]
    qwen = tc.VALIDATED_MATRICES[(2, 1)]
    for padded_in, padded_out in GLM_DENSE_FAMILIES:
        assert not tc.takes_launch(64, padded_in, padded_out, 12, 2, False, SMS, glm), (
            padded_in,
            padded_out,
        )
        assert tc.takes_launch(65, padded_in, padded_out, 12, 2, False, SMS, glm)
        assert tc.takes_launch(292, padded_in, padded_out, 12, 2, False, SMS, glm)
        assert not tc.takes_launch(
            292, padded_in, padded_out, 12, 2, False, SMS, qwen
        ), (padded_in, padded_out)
    # The head, the narrow projections and any other rate stay on today's
    # kernels even though the operator itself may cover them.
    for padded_in, padded_out, bits_x2 in (
        (4096, 154880, 12),
        (4096, 1536, 12),
        (4096, 512, 12),
        (4096, 1024, 12),
        (4096, 24576, 10),
        (4096, 24576, 14),
    ):
        assert not tc.takes_launch(
            292, padded_in, padded_out, bits_x2, 2, False, SMS, glm
        )


def test_glm_warmup_launches_once_per_covered_family(monkeypatch):
    """Every GLM family is shape-specialized: each needs its own warm-up
    launch before capture, or a request's first prefill loads it."""
    calls = []
    monkeypatch.setattr(tc, "_require_explicit_prefill", True)
    monkeypatch.setattr(tc, "_served_matrices", tc.VALIDATED_MATRICES[(1, 2)])
    monkeypatch.setattr(
        tc,
        "torch",
        SimpleNamespace(
            device=lambda kind, index: "cpu",
            bfloat16="bfloat16",
            float16="float16",
            float32="float32",
            int16="int16",
            ones=lambda shape, dtype=None, device=None: shape,
            empty=lambda shape, dtype=None, device=None: shape,
            zeros=lambda shape, dtype=None, device=None: shape,
            cuda=SimpleNamespace(current_device=lambda: 0, synchronize=lambda d: None),
            ops=SimpleNamespace(
                _exl3_C=SimpleNamespace(linear_workspace_size=lambda *args: 1)
            ),
        ),
    )
    monkeypatch.setattr(
        tc,
        "linear_out_prefill",
        lambda x, trellis, suh, svh, bits, codebook, i, o, out, ws: calls.append(
            (i, o, bits)
        ),
    )
    tc._warmup(SMS)
    assert calls == sorted(tc.VALIDATED_MATRICES[(1, 2)])


def _device(monkeypatch, capability, sms=SMS, config=True, **layout):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: capability is not None)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device=None: SimpleNamespace(multi_processor_count=sms),
    )
    import vllm.config

    value = _config(**layout) if config else None
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: value)


def test_a_stale_library_is_refused_with_the_rebuild_hint(monkeypatch):
    monkeypatch.setattr(tc, "active", False)
    monkeypatch.setenv("VLLM_EXL3_DENSE_PREFILL_TC", "1")
    _device(monkeypatch, (8, 0))

    def missing(name, overload):
        raise RuntimeError(f"no schema {name}")

    monkeypatch.setattr(torch._C, "_dispatch_find_schema_or_throw", missing)
    with pytest.raises(RuntimeError, match="build_standalone.py"):
        tc.resolve()
    assert tc.active is False


def test_switch_is_resolved_once_by_ensure_extension(monkeypatch):
    monkeypatch.setattr(exl3_ops, "_loaded", False)
    monkeypatch.setattr(exl3_ops, "_fakes_registered", False)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", False)
    monkeypatch.setattr(exl3_ops.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(exl3_ops, "_validate_extension", lambda module: None)
    monkeypatch.setattr(torch.library, "register_fake", lambda name: lambda fn: fn)
    monkeypatch.setattr(exl3_ops, "_resolve_dense_splitk", lambda: False)
    resolved = []
    monkeypatch.setattr(tc, "resolve", lambda: resolved.append(1))
    exl3_ops.ensure_extension()
    exl3_ops.ensure_extension()
    assert resolved == [1], "resolved once per process"


def test_ensure_extension_runs_the_open_gate_warmup_once(monkeypatch):
    """A process's first EXL3 use with the switch set and the gate open (TP2):
    ensure_extension runs the real resolve and warm-up once, and nothing they
    run calls back into ensure_extension, which marks the library loaded only
    after resolve returns. The library and the device are stood in for; the
    warm-up buffers live on the CPU."""
    monkeypatch.setattr(exl3_ops, "_loaded", False)
    monkeypatch.setattr(exl3_ops, "_fakes_registered", False)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", False)
    monkeypatch.setattr(exl3_ops.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(exl3_ops, "_validate_extension", lambda module: None)
    monkeypatch.setattr(exl3_ops, "_resolve_dense_splitk", lambda: False)
    for name in (
        "active",
        "_prefill_active",
        "decode_active",
        "_decode_served",
        "_served_matrices",
        "_require_explicit_prefill",
        "_max_decode_query_len",
        "_sms",
    ):
        monkeypatch.setattr(tc, name, getattr(tc, name))
    monkeypatch.setattr(tc, "_fake_registered", False)
    fakes = []
    monkeypatch.setattr(
        torch.library, "register_fake", lambda name: lambda fn: fakes.append(name)
    )
    monkeypatch.setattr(
        torch._C, "_dispatch_find_schema_or_throw", lambda name, overload: None
    )
    monkeypatch.setenv("VLLM_EXL3_DENSE_PREFILL_TC", "1")
    monkeypatch.delenv("VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE", raising=False)
    _device(monkeypatch, (8, 0))
    calls = []

    def workspace_size(*shape):
        calls.append(("workspace", shape))
        return 16

    def prefill(x, trellis, suh, svh, bits_x2, *rest):
        calls.append(("prefill", tuple(x.shape), bits_x2))

    library = SimpleNamespace(
        set_prefill_schedule=lambda mode: calls.append(("schedule", mode)),
        linear_workspace_size=workspace_size,
        linear_out_prefill=prefill,
    )
    monkeypatch.setattr(torch.ops, "_exl3_C", library)
    for name in ("ones", "empty", "zeros"):
        make = getattr(torch, name)
        monkeypatch.setattr(
            torch, name, lambda *a, _make=make, device=None, **kw: _make(*a, **kw)
        )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    # A cached size would hide a call back into ensure_extension.
    exl3_ops.linear_workspace_size.cache_clear()
    resolve, resolved = tc.resolve, []
    monkeypatch.setattr(tc, "resolve", lambda: resolved.append(1) or resolve())
    exl3_ops.ensure_extension()
    exl3_ops.ensure_extension()
    assert resolved == [1], "resolved once per process"
    assert exl3_ops._loaded and tc.active and tc._sms == SMS
    rows, padded_in, padded_out = tc.WARMUP_LAUNCH
    assert calls == [
        ("schedule", 0),
        ("workspace", (rows, padded_in, padded_out)),
        *(("prefill", (rows, padded_in), bits_x2) for bits_x2 in (10, 12, 14)),
    ]
    assert fakes[-1] == "_exl3_C::linear_out_prefill"


def test_ensure_extension_retries_resolve_after_it_raised(monkeypatch):
    """A resolve that raised leaves the library unloaded; the next
    ensure_extension reaches resolve again instead of failing on the base fake
    implementations it registered the first time (torch refuses a second
    registration). The library is stood in for."""
    monkeypatch.setattr(exl3_ops, "_loaded", False)
    monkeypatch.setattr(exl3_ops, "_fakes_registered", False, raising=False)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", False)
    monkeypatch.setattr(exl3_ops.importlib, "import_module", lambda name: object())
    monkeypatch.setattr(exl3_ops, "_validate_extension", lambda module: None)
    monkeypatch.setattr(exl3_ops, "_resolve_dense_splitk", lambda: False)
    fakes = []

    def register_fake(name):
        if name in fakes:
            raise RuntimeError(f"{name} already has a fake impl registered")
        return lambda fn: fakes.append(name)

    monkeypatch.setattr(torch.library, "register_fake", register_fake)
    resolved = []

    def resolve():
        resolved.append(1)
        if len(resolved) == 1:
            raise RuntimeError("warm-up failed")

    monkeypatch.setattr(tc, "resolve", resolve)
    with pytest.raises(RuntimeError, match="warm-up failed"):
        exl3_ops.ensure_extension()
    assert not exl3_ops._loaded
    exl3_ops.ensure_extension()
    assert resolved == [1, 1] and exl3_ops._loaded
    assert "_exl3_C::linear_out" in fakes and len(fakes) == len(set(fakes))


def _recorders(monkeypatch):
    """Replace the three operators by recorders; returns the call list."""
    calls = []
    for module, name in (
        (exl3_ops, "linear_out"),
        (exl3_ops, "linear_out_splitk"),
        (tc, "linear_out_prefill"),
    ):
        monkeypatch.setattr(
            module, name, lambda *args, _name=name: calls.append((_name, args))
        )
    return calls


def _linear(monkeypatch, rows, matrix, row_chunk, splitk):
    padded_in, padded_out, bits_x2 = matrix
    workspace = torch.empty(16)
    monkeypatch.setattr(exl3_ops, "_linear_scratch", lambda device: workspace)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", splitk)
    trellis = torch.zeros(
        padded_in // 16, padded_out // 16, 8 * bits_x2, dtype=torch.int16
    )
    x = torch.zeros(rows, padded_in, dtype=torch.bfloat16)
    out = exl3_ops._linear(x, trellis, "suh", "svh", bits_x2, 2, padded_out, row_chunk)
    assert out.shape == (rows, padded_out)
    return trellis, workspace


@pytest.mark.parametrize("splitk", [False, True])
@pytest.mark.parametrize("max_query_len", [None, DECODE_QUERY_LEN, 3460])
def test_switch_off_keeps_the_base_operator(monkeypatch, splitk, max_query_len):
    calls = _recorders(monkeypatch)
    monkeypatch.setattr(tc, "active", False)
    monkeypatch.setattr(tc, "route", lambda *a: pytest.fail("route called while off"))
    _step(monkeypatch, max_query_len)
    trellis, workspace = _linear(monkeypatch, 3460, KV, 2978, splitk)
    expected = "linear_out_splitk" if splitk else "linear_out"
    assert [name for name, _ in calls] == [expected] * 2
    for _, args in calls:
        assert args[1] is trellis and args[9] is workspace
    assert [args[0].shape[0] for _, args in calls] == [2978, 482]


def _on(monkeypatch):
    """The prefill tensor-core path only: route dispatches the prefill sub-path,
    the small-row decode path stays closed."""
    monkeypatch.setattr(tc, "active", True)
    monkeypatch.setattr(tc, "_prefill_active", True)
    monkeypatch.setattr(tc, "decode_active", False)
    monkeypatch.setattr(tc, "_decode_served", frozenset())
    monkeypatch.setattr(tc, "_max_decode_query_len", DECODE_QUERY_LEN)
    monkeypatch.setattr(tc, "_sms", SMS)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


@pytest.mark.parametrize("splitk", [False, True])
@pytest.mark.parametrize(
    "max_query_len,rows,matrix",
    [
        # Decode steps of more than 64 tokens: 32 requests x 4 verify rows, and
        # the drafter's hidden-state projection at 4 x 64 verify rows.
        (DECODE_QUERY_LEN, 128, Q),
        (DECODE_QUERY_LEN, 256, FC),
        (1, 3460, OUT),
        # No forward context (profiling, tests, captures in piecewise mode).
        (None, 3460, OUT),
        # A prefill step's call of at most 64 rows.
        (3460, 64, Q),
    ],
)
def test_decode_steps_and_small_calls_keep_the_base_operator(
    monkeypatch, splitk, max_query_len, rows, matrix
):
    _extension_or_skip()
    calls = _recorders(monkeypatch)
    _on(monkeypatch)
    _step(monkeypatch, max_query_len)
    _linear(monkeypatch, rows, matrix, 4096, splitk)
    expected = "linear_out_splitk" if splitk else "linear_out"
    assert [name for name, _ in calls] == [expected]


def test_graph_capture_keeps_the_base_operator(monkeypatch):
    _extension_or_skip()
    calls = _recorders(monkeypatch)
    _on(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _step(monkeypatch, 3460)
    _linear(monkeypatch, 1489, OUT, 1489, True)
    assert [name for name, _ in calls] == ["linear_out_splitk"]


@pytest.mark.parametrize(
    "splitk,expected",
    [
        # The k/v 482-row tail is a split-K launch: it keeps split-K.
        (True, ["linear_out_prefill", "linear_out_splitk"]),
        # 8 output tiles: below the operator's 16, so linear_out.
        (False, ["linear_out_prefill", "linear_out"]),
    ],
)
def test_prefill_step_routes_each_chunk_by_its_launch(monkeypatch, splitk, expected):
    _extension_or_skip()
    calls = _recorders(monkeypatch)
    _on(monkeypatch)
    _step(monkeypatch, 3460)
    trellis, workspace = _linear(monkeypatch, 3460, KV, 2978, splitk)
    assert [name for name, _ in calls] == expected
    assert [args[0].shape[0] for _, args in calls] == [2978, 482]
    for _, args in calls:
        assert args[1] is trellis and args[2:8] == ("suh", "svh", 14, 2, 2560, 256)
        assert args[8].shape == (args[0].shape[0], 256) and args[9] is workspace


def test_a_misaligned_trellis_keeps_the_base_operator(monkeypatch):
    _extension_or_skip()
    calls = _recorders(monkeypatch)
    _on(monkeypatch)
    _step(monkeypatch, 3460)
    padded_in, padded_out, bits_x2 = OUT
    base = torch.zeros(
        padded_in * padded_out // 256 * 8 * bits_x2 + 1, dtype=torch.int16
    )
    trellis = base[1:].view(padded_in // 16, padded_out // 16, 8 * bits_x2)
    monkeypatch.setattr(exl3_ops, "_linear_scratch", lambda device: torch.empty(16))
    monkeypatch.setattr(exl3_ops, "_dense_splitk", False)
    x = torch.zeros(1489, padded_in, dtype=torch.bfloat16)
    exl3_ops._linear(x, trellis, "suh", "svh", bits_x2, 2, padded_out, 1489)
    assert [name for name, _ in calls] == ["linear_out"]


def test_operator_rule_and_served_launches():
    _extension_or_skip()
    assert set(tc.OPERATORS) <= {
        name.split("::")[1].split(".")[0]
        for name in torch._C._dispatch_get_all_op_names()
        if name.startswith("_exl3_C::")
    }
    applies = tc.linear_prefill_applies
    # More than 64 rows, at least 16 output tiles, the covered formats.
    assert not applies(64, 2560, 2560, 14, 2)
    assert applies(65, 2560, 2048, 14, 2)  # 16 tiles
    assert not applies(65, 2560, 1920, 14, 2)  # 15 tiles
    assert not applies(1906, 2560, 128, 14, 2)  # 15 tiles
    assert applies(1921, 2560, 128, 14, 2)  # 16 tiles
    for bits_x2, codebook in ((10, 2), (12, 2), (14, 2)):
        assert applies(1489, 3072, 2560, bits_x2, codebook)
    for bits_x2, codebook in ((8, 2), (16, 2), (14, 1), (4, 1)):
        assert not applies(1489, 3072, 2560, bits_x2, codebook)
    with pytest.raises(RuntimeError):
        applies(1489, 3000, 2560, 14, 2)
    for (padded_in, padded_out, bits_x2), rows, path in SERVED_LAUNCHES:
        on = tc.takes_launch(
            rows,
            padded_in,
            padded_out,
            bits_x2,
            2,
            True,
            SMS,
            tc.VALIDATED_MATRICES[(2, 1)],
        )
        off = tc.takes_launch(
            rows,
            padded_in,
            padded_out,
            bits_x2,
            2,
            False,
            SMS,
            tc.VALIDATED_MATRICES[(2, 1)],
        )
        split = exl3_ops.linear_splitk_splits(rows, padded_out, SMS, bits_x2, 2) == 2
        got = "tc" if on else ("splitk" if split else "linear_out")
        assert got == path, (padded_in, padded_out, rows)
        # With split-K off every launch split-K does not take is unchanged.
        assert off == (
            on or (split and applies(rows, padded_in, padded_out, bits_x2, 2))
        )
    # The k/v tail without split-K: 8 tiles, so linear_out.
    assert not tc.takes_launch(
        482, 2560, 256, 14, 2, False, SMS, tc.VALIDATED_MATRICES[(2, 1)]
    )
    # Row chunks are the base's (the operator fits linear_workspace_size).
    assert exl3_ops.linear_row_chunk(2560, 256) == 2978
    assert exl3_ops.linear_row_chunk(3072, 2560) == 1489


# ---------------------------------------------------------------- GPU tests


def _gpu_or_skip():
    _extension_or_skip()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    if sms != SMS:
        pytest.skip(f"{sms} multiprocessors: the schedule was validated at {SMS}")


def _prefill_kernels(fn):
    """Names of the prefill GEMM kernels fn launched."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return {e.name for e in prof.events() if "prefill_gemm" in e.name}


def _split_ran(names):
    return any(", true>" in n or ",true>" in n or "(bool)1>" in n for n in names)


@needs_sm80
@pytest.mark.parametrize(
    "padded_in,padded_out,bits_x2,logical_in,logical_out,rows,split",
    [
        (2560, 256, 14, 2560, 256, 1906, False),  # k/v, 30 tiles: one wave
        (2560, 640, 10, 2560, 640, 839, False),  # indexer q/k, 35 tiles
        (3072, 2560, 14, 3072, 2560, 482, True),  # out/o_proj tail, 80 tiles
        (2560, 1280, 12, 2560, 1280, 1276, True),  # drafter fc, 100 tiles
        # Column-band splits (the whole-tile rectangle spans every row block).
        (2560, 3072, 14, 2560, 3072, 1300, True),  # linear-attention z, 264 tiles
        (2560, 640, 14, 2560, 640, 2200, True),  # shared gate/up, 90 tiles
        # q_proj, 74 tiles: 37 split blocks, as many as the workspace holds.
        (2560, 6144, 14, 2560, 6144, 200, True),
        (2560, 2048, 14, 2500, 1999, 200, False),  # cropped edges, 32 tiles
        # GLM 6-bit dense families at the first legal launch; their stream-K
        # schedule is observed rather than predicted (split=None).
        (4096, 2048, 12, 4096, 2048, 65, None),  # shared gate/up, 16 tiles
        (2048, 4096, 12, 2048, 4096, 65, None),  # shared down, 32 tiles
        (4096, 12288, 12, 4096, 12288, 65, None),  # dense-layer gate/up, 96 tiles
        (12288, 4096, 12, 12288, 4096, 65, None),  # dense-layer down, 32 tiles
        (1536, 16384, 12, 1536, 16384, 65, None),  # q_b, 128 tiles
    ],
)
def test_prefill_holds_linear_outs_error_against_fp64(
    padded_in, padded_out, bits_x2, logical_in, logical_out, rows, split
):
    from tests.kernels.quantization.test_exl3_kernels import effective_weight

    _gpu_or_skip()
    assert tc.linear_prefill_applies(rows, padded_in, padded_out, bits_x2, 2)
    m = _matrix(17 * bits_x2 + padded_out, padded_in, padded_out, bits_x2)
    weight = torch.from_numpy(effective_weight(*m["numpy"], bits_x2, 2)).cuda()
    weight = weight[:logical_in, :logical_out]
    pool = _Pool()
    for seed in range(2):
        x = _inputs(1000 + rows + seed, rows, logical_in)
        cand = _run(tc.linear_out_prefill, m, x, bits_x2, logical_out)
        names = _prefill_kernels(
            lambda x=x: _run(tc.linear_out_prefill, m, x, bits_x2, logical_out)
        )
        assert names, "the prefill GEMM did not run"
        if split is not None:
            assert _split_ran(names) == split, sorted(names)
        _bitwise(
            _run(tc.linear_out_prefill, m, x, bits_x2, logical_out), cand, "repeat"
        )
        inc = _run(exl3_ops.linear_out, m, x, bits_x2, logical_out)
        pool.add(cand, inc, x.double() @ weight)
    pool.check(f"{padded_in}x{padded_out} bits_x2={bits_x2} rows {rows}")


@needs_sm80
@pytest.mark.parametrize(
    "padded_in,padded_out,rows",
    [
        (3072, 2560, 482),  # column band of 2 x 4 tiles
        (2560, 3072, 1300),  # column band of 4 x 11 tiles
        (2560, 640, 2200),  # column band of 1 x 18 tiles
        (2560, 6144, 200),  # column band of 11 x 2 tiles, 37 split blocks
        (2560, 1280, 1276),  # row band of 10 x 3 tiles
    ],
)
def test_stream_k_is_order_independent(padded_in, padded_out, rows):
    """A split tile's value is the sum of its segments' partials in k order,
    whichever block arrives last (the counter only elects the block that adds
    them). Busy multiprocessors change which block that is: the result must
    not change."""
    _gpu_or_skip()
    bits_x2 = 12 if padded_out == 1280 else 14
    m = _matrix(41 + rows, padded_in, padded_out, bits_x2)
    x = _inputs(42 + rows, rows, padded_in)

    def call():
        return _run(tc.linear_out_prefill, m, x, bits_x2, padded_out)

    assert _split_ran(_prefill_kernels(call))
    reference = call()
    for _ in range(10):
        _bitwise(call(), reference, "repeat")
    side = torch.cuda.Stream()
    a = torch.randn(2048, 2048, device="cuda")
    for delay in range(6):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(delay):
                a = a @ a / 2048
        out = call()
        torch.cuda.synchronize()
        _bitwise(out, reference, f"with {delay} matmuls beside it")


@needs_sm80
def test_whole_tile_diagnostic_runs_no_split():
    _gpu_or_skip()
    m = _matrix(51, 3072, 2560, 14)
    x = _inputs(52, 482, 3072)
    planned = _run(tc.linear_out_prefill, m, x, 14, 2560)
    try:
        tc.set_prefill_schedule(1)
        names = _prefill_kernels(lambda: _run(tc.linear_out_prefill, m, x, 14, 2560))
        whole = _run(tc.linear_out_prefill, m, x, 14, 2560)
    finally:
        tc.set_prefill_schedule(0)
    assert names and not _split_ran(names), sorted(names)
    # Only the split tiles' FP32 sums differ; the BF16 outputs stay close.
    assert (
        whole.float() - planned.float()
    ).abs().max() <= 2.0**-6 * planned.abs().max()
    with pytest.raises(RuntimeError):
        tc.set_prefill_schedule(2)


@needs_sm80
def test_uncovered_launches_are_refused():
    _gpu_or_skip()
    m = _matrix(61, 2560, 256, 14)
    for rows in (64, 482):  # the decode bound; 8 output tiles
        with pytest.raises(RuntimeError, match="does not cover"):
            _run(tc.linear_out_prefill, m, _inputs(rows, rows, 2560), 14, 256)
    unaligned = _matrix(62, 3072, 2560, 14, unaligned=True)
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        _run(tc.linear_out_prefill, unaligned, _inputs(63, 482, 3072), 14, 2560)
    m = _matrix(64, 3072, 2560, 8)
    with pytest.raises(RuntimeError, match="does not cover"):
        _run(tc.linear_out_prefill, m, _inputs(65, 482, 3072), 8, 2560)


@needs_sm80
def test_ensure_extension_opens_the_gate_and_warms_up_on_the_card(monkeypatch):
    """A process's first EXL3 use with the switch set at TP2, on the card: the
    real library, gate and warm-up, resolved once without calling back into
    ensure_extension."""
    _gpu_or_skip()
    import vllm.config

    monkeypatch.setattr(exl3_ops, "_loaded", False)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", exl3_ops._dense_splitk)
    for name in (
        "active",
        "_prefill_active",
        "decode_active",
        "_decode_served",
        "_served_matrices",
        "_require_explicit_prefill",
        "_max_decode_query_len",
        "_sms",
        "_fake_registered",
    ):
        monkeypatch.setattr(tc, name, getattr(tc, name))
    # This process registered the operators' fake implementations already.
    monkeypatch.setattr(torch.library, "register_fake", lambda name: lambda fn: fn)
    monkeypatch.setenv("VLLM_EXL3_DENSE_PREFILL_TC", "1")
    monkeypatch.delenv("VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE", raising=False)
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", _config)
    exl3_ops.linear_workspace_size.cache_clear()
    resolve, resolved = tc.resolve, []
    monkeypatch.setattr(tc, "resolve", lambda: resolved.append(1) or resolve())
    exl3_ops.ensure_extension()
    exl3_ops.ensure_extension()
    assert resolved == [1] and exl3_ops._loaded
    assert tc.active and tc._sms == SMS
    assert tc._max_decode_query_len == DECODE_QUERY_LEN


@needs_sm80
@pytest.mark.parametrize("splitk", [False, True])
def test_linear_output_is_each_launchs_operator_bitwise(monkeypatch, splitk):
    _gpu_or_skip()
    m = _matrix(71, 2560, 256, 14)
    x = _inputs(72, 3460, 2560)
    scratch = torch.empty(exl3_ops.LINEAR_SCRATCH_ELEMENTS, device=x.device)
    monkeypatch.setattr(exl3_ops, "_linear_scratch", lambda device: scratch)
    monkeypatch.setattr(exl3_ops, "_dense_splitk", splitk)
    default = exl3_ops.linear_out_splitk if splitk else exl3_ops.linear_out
    chunk = exl3_ops.linear_row_chunk(2560, 256)
    assert chunk == 2978

    def linear():
        return exl3_ops._linear(x, m["trellis"], m["suh"], m["svh"], 14, 2, 256, chunk)

    monkeypatch.setattr(tc, "active", False)
    _step(monkeypatch, 3460)
    off = linear()
    for start in (0, chunk):
        _bitwise(
            off[start : start + chunk],
            _run(default, m, x[start : start + chunk], 14, 256),
            "off",
        )
    _on(monkeypatch)
    on = linear()
    _bitwise(
        on[:chunk], _run(tc.linear_out_prefill, m, x[:chunk], 14, 256), "2978 rows"
    )
    _bitwise(on[chunk:], off[chunk:], "the 482-row tail keeps the base operator")
    _step(monkeypatch, DECODE_QUERY_LEN)
    _bitwise(linear(), off, "decode step")


@needs_sm80
def test_prefill_graph_replay_is_bitwise_eager_without_allocations():
    _gpu_or_skip()
    rows = 482
    m = _matrix(81, 3072, 2560, 14)
    x = _inputs(82, rows, 3072)
    out = torch.empty(rows, 2560, dtype=torch.bfloat16, device=x.device)
    workspace = torch.empty(
        exl3_ops.linear_workspace_size(rows, 3072, 2560), device=x.device
    )

    def call():
        tc.linear_out_prefill(
            x, m["trellis"], m["suh"], m["svh"], 14, 2, 3072, 2560, out, workspace
        )

    assert _split_ran(_prefill_kernels(call)), "the captured call must split k"
    torch.cuda.synchronize()
    before = _allocations()
    call()
    torch.cuda.synchronize()
    assert _allocations() == before, "eager call allocated"
    eager = out.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    empty = torch.cuda.CUDAGraph()
    before = _allocations()
    with torch.cuda.graph(empty, stream=stream):
        pass
    calibration = _allocations() - before
    graph = torch.cuda.CUDAGraph()
    before = _allocations()
    with torch.cuda.graph(graph, stream=stream):
        call()
    assert _allocations() - before <= calibration, "capture allocated"
    for _ in range(2):
        # NaN bits in the counters too: the call zeroes them itself.
        out.fill_(float("nan"))
        workspace.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _bitwise(out, eager, "replay vs eager")
    x.copy_(_inputs(83, rows, 3072))
    graph.replay()
    torch.cuda.synchronize()
    _bitwise(out, _run(tc.linear_out_prefill, m, x, 14, 2560), "new inputs")


def test_split_witness_matches_the_profiler_name_style():
    # The profiler demangles template arguments as true/false; cuobjdump as (bool)1.
    assert _split_ran({"void exl3::x::prefill_gemm<14, 2, __nv_bfloat16, true>(A, B*)"})
    assert not _split_ran(
        {"void exl3::x::prefill_gemm<14, 2, __nv_bfloat16, false>(A, B*)"}
    )
    assert _run_core is not None  # shared helper kept importable


# -------------------------------------------------------- small-row decode tests
# VLLM_EXL3_DENSE_DECODE_TC: the small-row tensor-core decode schedule for the GLM
# QKV family, decode steps of 1..8 rows. CPU: the gate (GLM (1,2) only, DFlash
# depths 3/5/7), the fail-closed family set, the per-launch gate (1..8 rows, codebook
# 2, split-K deferral), and _linear's operator per launch (the small-row decode path
# fires in a decode step and under a decode graph capture, keeps the base for other
# row counts, families and clean prefill steps). Host: linear_prefill_applies accepts
# 1..8 rows only while the native decode schedule is on. GPU: linear_out_prefill at
# 1..8 rows holds linear_out's error against FP64, is bitwise deterministic, and a
# row's bits are independent of the launch's row count (whole tile, no stream-K).


QKV_DECODE = (4096, 24576, 12)  # the measured family; small below to spare host memory
SMALL_DECODE = (1024, 256, 12)  # a stand-in family for the routing plumbing


def _explicit_step(monkeypatch, prefill, decode, spec):
    """A forward context with hybrid token counts (the GLM explicit-prefill form)."""
    layer = SimpleNamespace(
        num_prefill_tokens=prefill,
        num_decode_tokens=decode,
        num_spec_decode_tokens=spec,
    )
    monkeypatch.setattr(
        forward_context,
        "_forward_context",
        SimpleNamespace(attn_metadata={"layers.3.self_attn": layer}),
    )


def _on_decode(monkeypatch, served=frozenset({QKV_DECODE})):
    """The small-row decode path only: route dispatches the decode sub-path, the
    prefill path stays closed, and the operator's applies rule is stubbed covered so
    the routing decision is exercised without a live native schedule."""
    monkeypatch.setattr(tc, "active", True)
    monkeypatch.setattr(tc, "_prefill_active", False)
    monkeypatch.setattr(tc, "decode_active", True)
    monkeypatch.setattr(tc, "_decode_served", frozenset(served))
    monkeypatch.setattr(tc, "_require_explicit_prefill", True)
    monkeypatch.setattr(tc, "_max_decode_query_len", DECODE_QUERY_LEN)
    monkeypatch.setattr(tc, "_sms", SMS)
    monkeypatch.setattr(tc, "linear_prefill_applies", lambda *a: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def test_decode_gate_requires_the_glm_layout():
    assert tc.decode_closed_reason((8, 0), SMS, _glm_config()) is None
    # The Qwen (2, 1) layout the prefill path also serves is not a decode layout.
    assert tc.decode_closed_reason((8, 0), SMS, _config()) is not None
    # A non-GLM (1, 2) model is rejected by the shared GLM checks.
    not_glm = _config(tp=1, pp=2, method="dflash")
    assert tc.decode_closed_reason((8, 0), SMS, not_glm) is not None
    for capability, sms in (((8, 6), SMS), ((8, 0), 70), (None, None)):
        assert tc.decode_closed_reason(capability, sms, _glm_config()) is not None
    assert tc.decode_closed_reason((8, 0), SMS, None) is not None


def test_decode_launch_gate_is_narrow(monkeypatch):
    tc.takes_decode_launch.cache_clear()
    # Pretend the operator covers any 12-bit codebook-2 shape at any row count; the
    # Python gate is what is under test, and split-K is stubbed to never split.
    monkeypatch.setattr(
        tc, "linear_prefill_applies", lambda r, i, o, b, c: b == 12 and c == 2
    )
    monkeypatch.setattr(exl3_ops, "linear_splitk_splits", lambda *a: 1)
    served = frozenset({QKV_DECODE})
    take = tc.takes_decode_launch
    for rows in (1, 4, 8):
        assert take(rows, *QKV_DECODE, 2, False, SMS, served)
    for rows in (0, 9, 64, 1489):
        assert not take(rows, *QKV_DECODE, 2, False, SMS, served)
    # Codebook, family and rate are checked before the operator is asked.
    assert not take(4, *QKV_DECODE[:2], 12, 1, False, SMS, served)
    assert not take(4, 4096, 12288, 12, 2, False, SMS, served)
    assert not take(4, 4096, 24576, 10, 2, False, SMS, served)
    # With split-K on, a launch split-K would split keeps split-K.
    monkeypatch.setattr(exl3_ops, "linear_splitk_splits", lambda *a: 2)
    tc.takes_decode_launch.cache_clear()
    assert not take(4, *QKV_DECODE, 2, True, SMS, served)


@pytest.mark.parametrize("rows", [1, 4, 8])
def test_decode_step_routes_small_calls(monkeypatch, rows):
    calls = _recorders(monkeypatch)
    _on_decode(monkeypatch, served={SMALL_DECODE})
    tc.takes_decode_launch.cache_clear()
    _explicit_step(monkeypatch, 0, 1, 3)
    _linear(monkeypatch, rows, SMALL_DECODE, 16, False)
    assert [name for name, _ in calls] == ["linear_out_prefill"]


@pytest.mark.parametrize("rows", [9, 64])
def test_decode_step_larger_calls_keep_the_base_operator(monkeypatch, rows):
    calls = _recorders(monkeypatch)
    _on_decode(monkeypatch, served={SMALL_DECODE})
    tc.takes_decode_launch.cache_clear()
    _explicit_step(monkeypatch, 0, 1, 3)
    _linear(monkeypatch, rows, SMALL_DECODE, 128, False)
    assert [name for name, _ in calls] == ["linear_out"]


def test_decode_step_clean_prefill_and_uncovered_keep_the_base(monkeypatch):
    calls = _recorders(monkeypatch)
    _on_decode(monkeypatch, served={SMALL_DECODE})
    tc.takes_decode_launch.cache_clear()
    # A clean prefill step (prefill > 0, no decode, no speculative) keeps the base.
    _explicit_step(monkeypatch, 100, 0, 0)
    _linear(monkeypatch, 4, SMALL_DECODE, 16, False)
    # A decode step of an uncovered family keeps the base.
    _explicit_step(monkeypatch, 0, 1, 3)
    _linear(monkeypatch, 4, (1024, 384, 12), 16, False)
    assert [name for name, _ in calls] == ["linear_out", "linear_out"]


def test_decode_graph_capture_still_routes(monkeypatch):
    """Full decode graph capture retains explicit homogeneous-decode metadata."""
    calls = _recorders(monkeypatch)
    _on_decode(monkeypatch, served={SMALL_DECODE})
    tc.takes_decode_launch.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _explicit_step(monkeypatch, 0, 1, 3)
    _linear(monkeypatch, 4, SMALL_DECODE, 16, False)
    assert [name for name, _ in calls] == ["linear_out_prefill"]


@pytest.mark.parametrize("counts", [(1, 1, 2), (0, 0, 0), (0, None, 4), (-1, 1, 3)])
def test_decode_mixed_or_invalid_phase_keeps_base(monkeypatch, counts):
    calls = _recorders(monkeypatch)
    _on_decode(monkeypatch, served={SMALL_DECODE})
    _explicit_step(monkeypatch, *counts)
    _linear(monkeypatch, 4, SMALL_DECODE, 16, False)
    assert [name for name, _ in calls] == ["linear_out"]


def test_decode_unknown_capture_phase_keeps_base(monkeypatch):
    calls = _recorders(monkeypatch)
    _on_decode(monkeypatch, served={SMALL_DECODE})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    _step(monkeypatch, None)
    _linear(monkeypatch, 4, SMALL_DECODE, 16, False)
    assert [name for name, _ in calls] == ["linear_out"]


def test_switch_off_keeps_base_operator_for_decode(monkeypatch):
    """The disabled decode switch preserves the existing operator."""
    calls = _recorders(monkeypatch)
    monkeypatch.setattr(tc, "active", False)
    _step(monkeypatch, DECODE_QUERY_LEN)
    _linear(monkeypatch, 4, SMALL_DECODE, 16, False)
    assert [name for name, _ in calls] == ["linear_out"]


def test_native_applies_small_rows_only_when_decode_enabled():
    """Host arithmetic: linear_prefill_applies accepts 1..8 rows only while the native
    decode schedule is on; 9..64 stay the shipped bound, >64 is the prefill schedule's.
    The native process flag is restored."""
    _extension_or_skip()
    applies = tc.linear_prefill_applies
    try:
        tc.set_prefill_decode(False)
        for rows in (1, 4, 8):
            assert not applies(rows, *QKV_DECODE, 2)
        tc.set_prefill_decode(True)
        for rows in range(1, 9):
            assert applies(rows, *QKV_DECODE, 2), rows
        # 9..64 keep the shipped decode bound; only >64 is the prefill schedule's.
        for rows in (9, 64):
            assert not applies(rows, *QKV_DECODE, 2)
        assert applies(65, *QKV_DECODE, 2)
        # A decode launch must still be a covered format. The native predicate is
        # shape-generic for the GLM 6-bit codebook-2 format; which matrices take
        # it is the Python gate's validated set (DECODE_VALIDATED_MATRICES, or
        # DECODE_LAB_ALL_MATRICES under VLLM_EXL3_DENSE_DECODE_TC_FAMILIES=all).
        assert not applies(4, 4096, 24576, 10, 2)
        assert not applies(4, 4096, 24576, 12, 1)
        assert applies(4, 4096, 12288, 12, 2)
        assert applies(4, 4096, 2048, 12, 2)
    finally:
        tc.set_prefill_decode(False)


# The lab-widened decode families (shared-expert gate/up and down, attention out,
# indexer wq_b) besides QKV; the LM head shares QKV's k and only widens n.
DECODE_ERROR_SHAPES = [
    QKV_DECODE,
    (4096, 2048, 12),
    (2048, 4096, 12),
    (8192, 4096, 12),
    (1536, 4096, 12),
]


@needs_sm80
@pytest.mark.parametrize("shape", DECODE_ERROR_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("rows", [1, 6, 8])
def test_decode_holds_linear_outs_error_against_fp64(rows, shape):
    from tests.kernels.quantization.test_exl3_kernels import effective_weight

    _gpu_or_skip()
    padded_in, padded_out, bits_x2 = shape
    try:
        tc.set_prefill_decode(True)
        assert tc.linear_prefill_applies(rows, *shape, 2)
        m = _matrix(17 * bits_x2 + rows, padded_in, padded_out, bits_x2)
        weight = torch.from_numpy(effective_weight(*m["numpy"], bits_x2, 2)).cuda()
        pool = _Pool()
        for seed in range(2):
            x = _inputs(1000 + rows + seed, rows, padded_in)
            cand = _run(tc.linear_out_prefill, m, x, bits_x2, padded_out)
            _bitwise(
                _run(tc.linear_out_prefill, m, x, bits_x2, padded_out),
                cand,
                "repeat",
            )
            inc = _run(exl3_ops.linear_out, m, x, bits_x2, padded_out)
            pool.add(cand, inc, x.double() @ weight)
        pool.check(f"{padded_in}x{padded_out} bits_x2={bits_x2} decode rows {rows}")
    finally:
        tc.set_prefill_decode(False)


# Narrow decode families the ordered split-K is for (lab, set_decode_splitk).
DECODE_SPLITK_SHAPES = [(4096, 2048, 12), (2048, 4096, 12), (8192, 4096, 12), (4096, 512, 12)]


@needs_sm80
@pytest.mark.parametrize("shape", DECODE_SPLITK_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_decode_splitk_engages_and_holds_linear_outs_error(shape):
    """Ordered split-K with a serving-sized workspace (2,048 rows): the partials
    region past the activation scratch is written (the split engaged), repeats are
    bitwise identical, and the error against FP64 passes linear_out's gate."""
    from tests.kernels.quantization.test_exl3_kernels import effective_weight

    _gpu_or_skip()
    padded_in, padded_out, bits_x2 = shape
    rows = 4
    m = _matrix(31 * bits_x2 + padded_in, padded_in, padded_out, bits_x2)
    weight = torch.from_numpy(effective_weight(*m["numpy"], bits_x2, 2)).cuda()
    big = torch.empty(
        exl3_ops.linear_workspace_size(2048, padded_in, padded_out),
        dtype=torch.float32,
        device="cuda",
    )
    used = (rows * padded_in + rows * (padded_in // 128) + 3) // 4 * 4

    def run(x, splitk):
        tc.set_decode_splitk(splitk)
        big.fill_(float("nan"))
        out = torch.full(
            (rows, padded_out), float("nan"), dtype=torch.bfloat16, device="cuda"
        )
        tc.linear_out_prefill(
            x, m["trellis"], m["suh"], m["svh"], bits_x2, 2,
            padded_in, padded_out, out, big,
        )
        torch.cuda.synchronize()
        return out, bool(torch.isfinite(big[used:]).any())

    try:
        tc.set_prefill_decode(True)
        pool = _Pool()
        for seed in range(2):
            x = _inputs(2000 + seed, rows, padded_in)
            _, whole_wrote = run(x, False)
            assert not whole_wrote, "whole-k decode must not touch the partials region"
            split, split_wrote = run(x, True)
            assert split_wrote, "split-K did not engage with a serving-sized workspace"
            again, _ = run(x, True)
            _bitwise(again, split, "split-K repeat")
            inc = _run(exl3_ops.linear_out, m, x, bits_x2, padded_out)
            pool.add(split, inc, x.double() @ weight)
        pool.check(f"{padded_in}x{padded_out} split-K decode rows {rows}")
    finally:
        tc.set_decode_splitk(False)
        tc.set_prefill_decode(False)


@needs_sm80
def test_decode_row_bits_are_row_count_independent():
    """Whole tiles with no stream-K: a row's output is that column's whole-k sum, so
    it is bitwise identical whether the launch had one row or eight, and repeat calls
    are bitwise identical. This is what separates the decode schedule from the prefill
    path, whose split tiles make a row depend on the row count and the schedule."""
    _gpu_or_skip()
    padded_in, padded_out, bits_x2 = QKV_DECODE
    m = _matrix(7, padded_in, padded_out, bits_x2)
    try:
        tc.set_prefill_decode(True)
        wide = _run(
            tc.linear_out_prefill, m, _inputs(7, 8, padded_in), bits_x2, padded_out
        )
        for rows in (1, 3, 5, 8):
            narrow = _run(
                tc.linear_out_prefill,
                m,
                _inputs(7, rows, padded_in),
                bits_x2,
                padded_out,
            )
            _bitwise(narrow, wide[:rows], f"rows {rows} vs the 8-row launch")
        _bitwise(
            _run(
                tc.linear_out_prefill, m, _inputs(7, 8, padded_in), bits_x2, padded_out
            ),
            wide,
            "repeat",
        )
    finally:
        tc.set_prefill_decode(False)


def test_dense_decode_opt_in_preserves_small_prefill_tail(monkeypatch):
    calls = _recorders(monkeypatch)
    _on_decode(monkeypatch, served={SMALL_DECODE})
    monkeypatch.setattr(tc, "_prefill_active", True)
    monkeypatch.setattr(tc, "_served_matrices", frozenset({SMALL_DECODE}))
    tc.takes_launch.cache_clear()
    _explicit_step(monkeypatch, 69, 0, 0)
    _linear(monkeypatch, 69, SMALL_DECODE, 65, False)
    assert [(name, args[0].shape[0]) for name, args in calls] == [
        ("linear_out_prefill", 65),
        ("linear_out", 4),
    ]
