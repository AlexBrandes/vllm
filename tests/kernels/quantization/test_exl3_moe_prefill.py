# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 MoE prefill kernel (VLLM_EXL3_MOE_PREFILL_TC, csrc/exl3/moe_prefill.cu).

CPU: opt-in and supported-layout guards; every local expert's trellis must
be 16-byte aligned before the prefill kernel is eligible.

GPU (SM 8.0 only): at the served shape (hidden 2560, intermediate 640, top-10,
256 of 512 experts local, bits_x2 10 / codebook 2), the off path is bitwise
moe_out; moe_out_prefill is held to moe_out's error against an FP64 reference
(pooled mean <= 1.10x, max <= 1.25x, BF16 rounding ties forgiven, exact where
moe_out is exact) over awkward routings of 65..1001 rows; it is deterministic;
it refuses calls it does not cover; and a captured call replays bitwise like
the eager call without any allocator event. Every weight set's trellis is
16-byte aligned, as the gate requires.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.forward_context as forward_context
from tests.kernels.quantization.test_exl3_moe_v2 import (
    GLOBAL,
    HIDDEN,
    INTERMEDIATE,
    LOCAL,
    SERVED_META,
    TOPK,
    _apply,
    _bitwise,
    _case,
    _Fp64Pool,
    _reference,
    _v1,
    needs_sm80,
    served_experts,
)
from vllm_cmp170hx.kernels.exl3 import ops as exl3_ops

# One sampled token plus three draft tokens under verification.
DECODE_LEN = 4


def _phase(monkeypatch, phase):
    """Forward context of a step of the given scheduler phase ("prefill",
    "decode") or none at all; the attention metadata carries the longest
    request's scheduled tokens, as the attention backends build it."""
    if phase is None:
        monkeypatch.setattr(forward_context, "_forward_context", None)
        return
    query_len = 309 if phase == "prefill" else DECODE_LEN
    monkeypatch.setattr(
        forward_context,
        "_forward_context",
        SimpleNamespace(
            attn_metadata={
                "layers.3.self_attn": SimpleNamespace(max_query_len=query_len)
            }
        ),
    )


# ---------------------------------------------------------------- CPU tests


def test_switch_defaults_off(monkeypatch):
    from vllm import envs

    monkeypatch.delenv("VLLM_EXL3_MOE_PREFILL_TC", raising=False)
    assert envs.VLLM_EXL3_MOE_PREFILL_TC is False
    monkeypatch.setenv("VLLM_EXL3_MOE_PREFILL_TC", "1")
    assert envs.VLLM_EXL3_MOE_PREFILL_TC is True


def _method(monkeypatch, warmed, num_speculative_tokens=3):
    import vllm.config
    from vllm_cmp170hx.kernels.exl3.quant import Exl3MoEMethod

    method = object.__new__(Exl3MoEMethod)
    method.moe = SimpleNamespace(in_dtype=torch.bfloat16)
    method.hidden_size = HIDDEN
    method.intermediate_size = INTERMEDIATE
    method.local_experts = LOCAL
    method.global_expert_ids = list(range(0, GLOBAL, 2))
    layer = SimpleNamespace(
        exl3_ptrs=torch.zeros(LOCAL, 3, 3, dtype=torch.int64),
        exl3_meta=torch.tensor(SERVED_META),
        exl3_expert_map=torch.zeros(GLOBAL, dtype=torch.int32),
        top_k=TOPK,
        global_num_experts=GLOBAL,
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (8, 0))
    monkeypatch.setattr(
        exl3_ops, "moe_workspace_size_prefill", lambda rows, *args: 1000 * rows
    )
    monkeypatch.setattr(
        exl3_ops, "warmup_moe_prefill", lambda *args: warmed.append(args)
    )
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: _config(num_speculative_tokens=num_speculative_tokens),
    )
    return method, layer


def _config(num_speculative_tokens=3, tp=2, pp=1, spec=None, graph_max=None):
    if spec is None and num_speculative_tokens:
        spec = SimpleNamespace(method="mtp", parallel_drafting=False)
    return SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=pp
        ),
        speculative_config=spec,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=graph_max),
    )


@pytest.mark.parametrize(
    "config",
    [
        dict(tp=4),
        dict(pp=2),
        dict(spec=SimpleNamespace(method="dflash", parallel_drafting=True)),
        dict(spec=SimpleNamespace(method="eagle", parallel_drafting=True)),
    ],
)
def test_gate_closes_outside_the_validated_layout_and_drafters(monkeypatch, config):
    import vllm.config

    warmed = []
    method, layer = _method(monkeypatch, warmed)
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config", lambda: _config(**config)
    )
    assert method._moe_prefill(layer, SERVED_META, 3460, 7) == (0, 7, 0)
    assert not warmed


def test_gate_counts_the_speculative_draft(monkeypatch):
    method, layer = _method(monkeypatch, [], num_speculative_tokens=0)
    assert method._moe_prefill(layer, SERVED_META, 3460, 7)[0] == 1


@pytest.mark.parametrize("projection", [0, 1, 2])
def test_gate_closes_unless_every_expert_trellis_is_16_byte_aligned(
    monkeypatch, projection
):

    warmed = []
    method, layer = _method(monkeypatch, warmed)
    # One 1,024,000-byte trellis (or scale vector) per address; the scale
    # vectors' alignment does not matter.
    pointers = 0x7F0000000000 + 1024000 * torch.arange(LOCAL * 9).view(LOCAL, 3, 3)
    pointers[:, :, 1:] += 2
    layer.exl3_ptrs = pointers.clone()
    assert method._moe_prefill(layer, SERVED_META, 3460, 7) == (4, 3460000, 10)
    assert len(warmed) == 1
    pointers[37, projection, 0] += 2
    layer.exl3_ptrs = pointers
    assert method._moe_prefill(layer, SERVED_META, 3460, 7) == (0, 7, 0)
    assert len(warmed) == 1


def _experts(
    moe_v2_rows, prefill_decode_len, max_rows=8192, workspace=0, expert_map="layer map"
):
    """Exl3Experts without its constructor (see test_exl3_moe_v2._experts)."""
    from vllm_cmp170hx.kernels.exl3.quant import Exl3Experts

    experts = object.__new__(Exl3Experts)
    experts.intermediate_size = INTERMEDIATE
    experts.expert_map = expert_map
    experts.workspace_elements = workspace
    experts.max_rows = max_rows
    experts.moe_v2_rows = moe_v2_rows
    experts.prefill_decode_len = prefill_decode_len
    # The Qwen family: the row gate is implied by the shape, and it serves
    # calls of 65..3460 rows (the GLM family sets both attributes differently).
    experts.prefill_requires_explicit = False
    experts.prefill_max_rows = 3460
    return experts


# ---------------------------------------------------------------- GPU tests


@pytest.fixture(scope="module")
def served():
    """The moe_out_v2 tests' expert sets with every trellis 16-byte aligned:
    moe_out_prefill requires it (its gate keeps moe_out otherwise). The
    "single" scenario still routes one assignment to one expert."""
    return served_experts(None)


QWEN_BITS = 10


def _p(served, x, ids, weights):
    out = torch.full_like(x, float("nan"))
    workspace = torch.full(
        (
            exl3_ops.moe_workspace_size_prefill(
                x.shape[0], HIDDEN, INTERMEDIATE, TOPK, LOCAL
            ),
        ),
        float("nan"),
        device=x.device,
    )
    exl3_ops.moe_out_prefill(
        x,
        ids,
        weights,
        served.pointers,
        served.meta,
        served.expert_map,
        INTERMEDIATE,
        QWEN_BITS,
        out,
        workspace,
    )
    return out


@needs_sm80
@pytest.mark.parametrize("rows", [65, 1906])
def test_off_path_is_moe_out(served, monkeypatch, rows):
    _phase(monkeypatch, "prefill")
    x, ids, weights = _case(served, "uniform", rows, rows)
    expected = _v1(served, x, ids, weights)
    size = exl3_ops.moe_workspace_size(rows, HIDDEN, INTERMEDIATE, TOPK, LOCAL)
    for moe_v2_rows in (0, 64):
        experts = _experts(moe_v2_rows, 0, workspace=size, expert_map=served.expert_map)
        out = torch.full_like(x, float("nan"))
        workspace2 = torch.empty(2 * size, dtype=torch.bfloat16, device=x.device)
        _apply(experts, x, ids, weights, served.pointers, served.meta, out, workspace2)
        _bitwise(out, expected, f"off path rows={rows} moe_v2_rows={moe_v2_rows}")


@needs_sm80
def test_on_path_takes_the_prefill_op_only_in_prefill_steps(served, monkeypatch):
    rows = 97
    x, ids, weights = _case(served, "uniform", rows, 5)
    size = max(
        exl3_ops.moe_workspace_size(rows, HIDDEN, INTERMEDIATE, TOPK, LOCAL),
        exl3_ops.moe_workspace_size_prefill(rows, HIDDEN, INTERMEDIATE, TOPK, LOCAL),
    )
    experts = _experts(64, DECODE_LEN, workspace=size, expert_map=served.expert_map)
    for phase, expected in (
        ("prefill", _p(served, x, ids, weights)),
        ("decode", _v1(served, x, ids, weights)),
    ):
        _phase(monkeypatch, phase)
        out = torch.full_like(x, float("nan"))
        workspace2 = torch.empty(2 * size, dtype=torch.bfloat16, device=x.device)
        _apply(experts, x, ids, weights, served.pointers, served.meta, out, workspace2)
        _bitwise(out, expected, f"{phase} step, {rows} rows")


SCENARIOS = [
    ("uniform", 65),
    ("uniform", 97),
    ("uniform", 129),
    ("uniform", 255),
    ("uniform", 511),
    ("uniform", 1001),
    ("masked", 777),
    ("hot", 512),
    ("hot", 70),
    ("single", 100),
    ("nonlocal", 66),
    ("int64", 81),
]


@needs_sm80
def test_prefill_op_holds_moe_outs_error_against_fp64(served):
    pool = _Fp64Pool()
    for index, (scenario, rows) in enumerate(SCENARIOS):
        x, ids, weights = _case(served, scenario, rows, 100 + index)
        name = f"{scenario} rows={rows}"
        cand = _p(served, x, ids, weights)
        inc = _v1(served, x, ids, weights)
        ref = _reference(served, x, ids, weights)
        if scenario in ("nonlocal", "single"):
            # Rows without a local route are exactly zero in both.
            local = ((ids >= 0) & (ids < GLOBAL) & (ids % 2 == 0)).any(-1)
            assert (cand[~local] == 0).all() and (inc[~local] == 0).all(), name
        # Deterministic: a second call is bitwise the first.
        _bitwise(_p(served, x, ids, weights), cand, f"{name}: repeat")
        pool.add(cand, inc, ref, name)
    pool.check("moe_out_prefill")


@needs_sm80
def test_prefill_op_refuses_calls_it_does_not_cover(served):
    x, ids, weights = _case(served, "uniform", 65, 3)
    out = torch.empty_like(x)
    args = (served.pointers, served.meta, served.expert_map)
    small = torch.empty(
        exl3_ops.moe_workspace_size_prefill(65, HIDDEN, INTERMEDIATE, TOPK, LOCAL) - 1,
        device=x.device,
    )
    with pytest.raises(RuntimeError):
        exl3_ops.moe_out_prefill(
            x, ids, weights, *args, INTERMEDIATE, QWEN_BITS, out, small
        )
    workspace = torch.empty(small.numel() + 1, device=x.device)
    with pytest.raises(RuntimeError):
        exl3_ops.moe_out_prefill(
            x[:64],
            ids[:64],
            weights[:64],
            *args,
            INTERMEDIATE,
            QWEN_BITS,
            out[:64],
            workspace,
        )
    with pytest.raises(RuntimeError):
        exl3_ops.moe_out_prefill(x, ids, weights, *args, 512, QWEN_BITS, out, workspace)
    # Zero rows: nothing to do.
    exl3_ops.moe_out_prefill(
        x[:0], ids[:0], weights[:0], *args, INTERMEDIATE, QWEN_BITS, out[:0], workspace
    )


@needs_sm80
@pytest.mark.parametrize("rows", [65, 1906])
def test_prefill_graph_replay_is_bitwise_eager_without_allocations(served, rows):
    exl3_ops.warmup_moe_prefill(
        served.pointers,
        served.meta,
        served.expert_map,
        list(range(0, GLOBAL, 2)),
        HIDDEN,
        INTERMEDIATE,
        TOPK,
        QWEN_BITS,
    )
    x, ids, weights = _case(served, "uniform", rows, 21)
    out = torch.empty_like(x)
    workspace = torch.empty(
        exl3_ops.moe_workspace_size_prefill(rows, HIDDEN, INTERMEDIATE, TOPK, LOCAL),
        device=x.device,
    )
    args = (served.pointers, served.meta, served.expert_map, INTERMEDIATE, QWEN_BITS)

    def call():
        exl3_ops.moe_out_prefill(x, ids, weights, *args, out, workspace)

    def allocations():
        return torch.cuda.memory_stats()["allocation.all.allocated"]

    before = allocations()
    call()
    torch.cuda.synchronize()
    assert allocations() == before, "eager call allocated"
    eager = out.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()  # warm on the capture stream
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    empty = torch.cuda.CUDAGraph()
    before = allocations()
    with torch.cuda.graph(empty, stream=stream):
        pass
    calibration = allocations() - before
    graph = torch.cuda.CUDAGraph()
    before = allocations()
    with torch.cuda.graph(graph, stream=stream):
        call()
    assert allocations() - before <= calibration, "capture allocated"
    out.fill_(float("nan"))
    workspace.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _bitwise(out, eager, "replay vs eager")
    # New inputs in the captured buffers: the replay recomputes them.
    x2, ids2, weights2 = _case(served, "masked", rows, 22)
    x.copy_(x2)
    ids.copy_(ids2)
    weights.copy_(weights2)
    graph.replay()
    torch.cuda.synchronize()
    _bitwise(out, _p(served, x2, ids2, weights2), "replay with new inputs")
