# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded GLM prefill checks, not whole-model or graph/cache certification.

The exact family is hidden4096/intermediate2048, top8,288 experts, bits_x2=5,
codebook2, BF16 activations, and SwiGLU limit10. The existing Qwen FP64-relative
error gate is unchanged. Five distinct weight sets distinguish experts0,255,
256,287 and the remaining entries without materializing288 FP64 expert sets.
The module fixture is deliberately shared across selected cases.

Decode coverage: the same file gates the small-row schedule (operator
moe_out_decode_tc, route tiles of 8 rows) against the identical FP64-relative
gate, including skewed routing that splits one expert across several tiles. The
gate/up role modes (VLLM_EXL3_MOE_DECODE_TC_MODE: full, down, gate_down, up_down)
are process-static, so the role-mode tests spawn one subprocess per mode; each
qualifies the same error gate, and invalid modes fail closed.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.kernels.quantization.test_exl3_moe_v2 import (
    _bitwise,
    _extension_or_skip,
    _Fp64Pool,
    needs_sm80,
)
from vllm_cmp170hx.kernels.exl3 import ops as exl3_ops

GHIDDEN, GINTERMEDIATE, GTOPK = 4096, 2048, 8
GLOCAL, GGLOBAL = 288, 288
GBITS_X2, GCODEBOOK = 5, 2
LIMIT = 10.0


def _glm_meta(bits_x2):
    return [
        [
            [GHIDDEN, GINTERMEDIATE, bits_x2, GCODEBOOK],
            [GHIDDEN, GINTERMEDIATE, bits_x2, GCODEBOOK],
            [GINTERMEDIATE, GHIDDEN, bits_x2, GCODEBOOK],
        ]
    ] * GLOCAL


OBSERVED = (0, 255, 256, 287)
SHARED = len(OBSERVED)
SET_OF = np.full(GLOCAL, SHARED, dtype=np.int64)
SET_OF[list(OBSERVED)] = range(len(OBSERVED))


def _dense(trellis, suh, svh, bits_x2, codebook, slab=512):
    """The existing independent FP64 decoder in H128-aligned row slabs."""
    from tests.kernels.quantization.test_exl3_kernels import _blocks, core_weight

    padded_in, padded_out = trellis.shape[0] * 16, trellis.shape[1] * 16
    blocks = _blocks(slab)
    right = _blocks(padded_out) * svh.astype(np.float64)[None, :]
    out = np.empty((padded_in, padded_out))
    for lo in range(0, padded_in, slab):
        hi = lo + slab
        core = core_weight(trellis[lo // 16 : hi // 16], bits_x2, codebook)
        left = suh[lo:hi].astype(np.float64)[:, None] * blocks
        out[lo:hi] = left @ core @ right
    return out


def glm_experts(seed=10157, bits_x2=GBITS_X2):
    _extension_or_skip()
    device = torch.device("cuda")
    rng = np.random.default_rng(seed)
    owners, sets, dense = [], [], []
    for _ in range(SHARED + 1):
        ptrs, mats = [], []
        for k, n in (
            (GHIDDEN, GINTERMEDIATE),
            (GHIDDEN, GINTERMEDIATE),
            (GINTERMEDIATE, GHIDDEN),
        ):
            trellis = rng.integers(
                -(2**15), 2**15, (k // 16, n // 16, 8 * bits_x2)
            ).astype(np.int16)
            suh = (rng.choice([-1.0, 1.0], k) * rng.uniform(0.5, 1.5, k)).astype(
                np.float16
            )
            svh = (rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n) / 64).astype(
                np.float16
            )
            tensors = [torch.from_numpy(v).to(device) for v in (trellis, suh, svh)]
            owners.append(tensors)
            ptrs.append([t.data_ptr() for t in tensors])
            mats.append(
                torch.from_numpy(_dense(trellis, suh, svh, bits_x2, GCODEBOOK)).to(
                    device
                )
            )
        sets.append(ptrs)
        dense.append(mats)
    rms = sorted(float(weights[0].pow(2).mean().sqrt()) for weights in dense)
    scale = 2.0 * LIMIT / (GHIDDEN**0.5 * rms[len(rms) // 2])
    return SimpleNamespace(
        owners=owners,
        pointers=torch.tensor(
            [sets[s] for s in SET_OF], dtype=torch.int64, device=device
        ),
        meta=torch.tensor(_glm_meta(bits_x2), dtype=torch.int64, device=device),
        expert_map=torch.arange(GGLOBAL, dtype=torch.int32, device=device),
        dense=dense,
        set_of=torch.from_numpy(SET_OF).to(device),
        scale=scale,
        bits_x2=bits_x2,
    )


def _routes(scenario, rows, generator):
    row = torch.arange(rows, dtype=torch.int64)
    fixed = lambda value: torch.full((rows,), value, dtype=torch.int64)
    if scenario == "observed":
        ids = torch.stack(
            (
                fixed(0),
                fixed(255),
                fixed(256),
                fixed(287),
                5 + row % 70,
                80 + row % 40,
                200 + row % 50,
                fixed(-1),
            ),
            1,
        )
    elif scenario == "boundary":
        ids = torch.stack(
            (
                fixed(255),
                fixed(256),
                fixed(287),
                1 + row % 80,
                90 + row % 60,
                160 + row // 2 % 60,
                225 + row // 3 % 20,
                fixed(GGLOBAL),
            ),
            1,
        )
    elif scenario == "sparse":
        ids = torch.full((rows, GTOPK), -1, dtype=torch.int64)
        ids[:, 0] = row % GGLOBAL
        ids[:, 1] = (7 * row + 3) % GGLOBAL
    elif scenario == "skew":
        # A decode-schedule edge: experts with more routes than one 8-row
        # tile (the 1..8-row range must split them across tiles without
        # moving any row's sums).
        ids = torch.stack(
            (
                fixed(17),
                fixed(17),
                fixed(40),
                40 + row % 3,
                fixed(255),
                fixed(255),
                (3 * row) % GGLOBAL,
                fixed(-1),
            ),
            1,
        )
    elif scenario == "uniform":
        ids = torch.argsort(torch.rand(rows, GGLOBAL, generator=generator), dim=1)[
            :, :GTOPK
        ]
    else:
        raise ValueError(scenario)
    weights = torch.rand(rows, GTOPK, generator=generator) + 0.01
    return ids, weights / weights.sum(-1, keepdim=True)


def _case(glm, scenario, rows, seed):
    generator = torch.Generator().manual_seed(seed)
    ids, weights = _routes(scenario, rows, generator)
    x = (torch.randn(rows, GHIDDEN, generator=generator) * glm.scale).to(torch.bfloat16)
    return x.cuda(), ids.to(torch.int32).cuda(), weights.cuda()


def _run(op, glm, x, ids, weights, size_fn, limit=None):
    out = torch.full_like(x, float("nan"))
    workspace = torch.full(
        (size_fn(x.shape[0], GHIDDEN, GINTERMEDIATE, GTOPK, GLOCAL),),
        float("nan"),
        dtype=torch.float32,
        device=x.device,
    )
    tail = (
        [glm.bits_x2, out, workspace]
        if op in (exl3_ops.moe_out_prefill, exl3_ops.moe_out_decode_tc)
        else [out, workspace]
    )
    if limit is not None:
        tail.append(limit)
    op(x, ids, weights, glm.pointers, glm.meta, glm.expert_map, GINTERMEDIATE, *tail)
    return out


def _p(glm, x, ids, weights):
    return _run(
        exl3_ops.moe_out_prefill,
        glm,
        x,
        ids,
        weights,
        exl3_ops.moe_workspace_size_prefill,
    )


def _v1(glm, x, ids, weights):
    return _run(
        exl3_ops.moe_out, glm, x, ids, weights, exl3_ops.moe_workspace_size, LIMIT
    )


def _d(glm, x, ids, weights):
    return _run(
        exl3_ops.moe_out_decode_tc,
        glm,
        x,
        ids,
        weights,
        exl3_ops.moe_workspace_size_decode_tc,
    )


def _oracle(glm, x, ids, weights):
    """FP64 reference, preserving the model's asymmetric gate/up clipping."""
    x64 = x.double()
    out = torch.zeros_like(x64)
    table = glm.expert_map.to(torch.int64)
    inside = (ids >= 0) & (ids < table.numel())
    # torch.where evaluates both operands: never index padding/out-of-range IDs.
    local = torch.where(inside, table[ids.clamp(0, table.numel() - 1)], -1)
    clipped = [0, 0, int((local >= 0).sum())]
    for route in range(GTOPK):
        column = local[:, route]
        rows = torch.nonzero(column >= 0, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        for index, (gate_weight, up_weight, down_weight) in enumerate(glm.dense):
            mine = glm.set_of[column[rows]] == index
            if not bool(mine.any()):
                continue
            picked = rows[mine]
            scaled = weights[rows, route][mine].double()[:, None]
            gate = x64[picked] @ gate_weight
            up = x64[picked] @ up_weight
            clipped[0] += int((gate > LIMIT).sum())
            clipped[1] += int((up.abs() > LIMIT).sum())
            gate = gate.clamp(max=LIMIT)
            up = up.clamp(min=-LIMIT, max=LIMIT)
            act = gate / (1.0 + torch.exp(-gate)) * up
            out.index_add_(0, picked, act @ down_weight * scaled)
    return out, clipped


def test_glm_workspace_rejects_mixed_families():
    _extension_or_skip()
    for hidden, intermediate in ((3072, 2048), (4096, 640), (2560, 2048)):
        with pytest.raises(RuntimeError):
            exl3_ops.moe_workspace_size_prefill(65, hidden, intermediate, GTOPK, GLOCAL)


def test_glm_decode_tc_workspace_rejects_unserved_shapes():
    _extension_or_skip()
    # The Qwen prefill shape included: the small-row schedule serves GLM only.
    for hidden, intermediate in ((2560, 640), (3072, 2048), (4096, 640)):
        with pytest.raises(RuntimeError):
            exl3_ops.moe_workspace_size_decode_tc(
                8, hidden, intermediate, GTOPK, GLOCAL
            )


@pytest.fixture(scope="module")
def glm():
    return glm_experts()


def _pool(glm, scenarios, name):
    pool = _Fp64Pool()
    clipped = [0, 0, 0]
    for index, (scenario, rows) in enumerate(scenarios):
        x, ids, weights = _case(glm, scenario, rows, 100 + index)
        label = f"{name}: {scenario} rows={rows}"
        candidate = _p(glm, x, ids, weights)
        _bitwise(_p(glm, x, ids, weights), candidate, f"{label}: repeat")
        reference, counts = _oracle(glm, x, ids, weights)
        clipped = [a + b for a, b in zip(clipped, counts)]
        pool.add(candidate, _v1(glm, x, ids, weights), reference, label)
    gate, up, routes = clipped
    assert gate >= 0.05 * routes * GINTERMEDIATE
    assert up >= 0.05 * routes * GINTERMEDIATE
    pool.check(f"moe_out_prefill {name}")


@needs_sm80
def test_glm_prefill_clipped_boundary_numerics(glm):
    _pool(glm, [("observed", 65), ("boundary", 129)], "GLM clipped boundary")


@needs_sm80
def test_glm_prefill_larger_numerics(glm):
    _pool(glm, [("uniform", 512), ("sparse", 2048)], "GLM larger prefill")


@needs_sm80
def test_glm_prefill_empty_routes_and_workspace_bounds(glm):
    x, ids, weights = _case(glm, "uniform", 65, 11)
    ids.fill_(-1)
    ids[33:, 0] = GGLOBAL
    ids[33:, 1] = GGLOBAL + 7
    assert (_p(glm, x, ids, weights) == 0).all()
    x, ids, weights = _case(glm, "boundary", 65, 3)
    out = torch.empty_like(x)
    args = (glm.pointers, glm.meta, glm.expert_map)
    elements = exl3_ops.moe_workspace_size_prefill(
        65, GHIDDEN, GINTERMEDIATE, GTOPK, GLOCAL
    )
    workspace = torch.empty(elements, dtype=torch.float32, device=x.device)
    with pytest.raises(RuntimeError):
        exl3_ops.moe_out_prefill(
            x[:64],
            ids[:64],
            weights[:64],
            *args,
            GINTERMEDIATE,
            glm.bits_x2,
            out[:64],
            workspace,
        )


@pytest.fixture(scope="module")
def glm35():
    return glm_experts(seed=10158, bits_x2=7)


@needs_sm80
def test_glm35_prefill_numerics(glm35):
    """The 3.5-bit layer family: same TC path, same FP64-relative gates."""
    _pool(glm35, [("observed", 65), ("uniform", 512), ("sparse", 1024)], "GLM 3.5-bit")


def _decode_pool(glm, scenarios, name):
    """The file's FP64-relative gate, on the small-row schedule."""
    pool = _Fp64Pool()
    clipped = [0, 0, 0]
    for index, (scenario, rows) in enumerate(scenarios):
        x, ids, weights = _case(glm, scenario, rows, 700 + index)
        label = f"{name}: {scenario} rows={rows}"
        candidate = _d(glm, x, ids, weights)
        _bitwise(_d(glm, x, ids, weights), candidate, f"{label}: repeat")
        reference, counts = _oracle(glm, x, ids, weights)
        clipped = [a + b for a, b in zip(clipped, counts)]
        pool.add(candidate, _v1(glm, x, ids, weights), reference, label)
    gate, up, routes = clipped
    assert gate >= 0.05 * routes * GINTERMEDIATE
    assert up >= 0.05 * routes * GINTERMEDIATE
    pool.check(f"moe_out_decode_tc {name}")


@needs_sm80
def test_glm_decode_tc_small_row_numerics(glm):
    """Every row count 1..8, including experts split across 8-row tiles."""
    _decode_pool(
        glm,
        [
            ("observed", 1),
            ("boundary", 2),
            ("uniform", 3),
            ("skew", 4),
            ("sparse", 5),
            ("uniform", 6),
            ("skew", 7),
            ("skew", 8),
        ],
        "GLM decode",
    )


@needs_sm80
def test_glm_decode_tc_row_order_invariance(glm):
    """A row's result never depends on the row order or on the order inside
    an expert: permuting the rows regroups the 8-row tiles, the sums stay."""
    x, ids, weights = _case(glm, "skew", 8, 5)
    out = _d(glm, x, ids, weights)
    perm = torch.tensor([7, 0, 5, 2, 6, 1, 4, 3], device=x.device)
    again = _d(glm, x[perm], ids[perm], weights[perm])
    _bitwise(again, out[perm], "decode row order")


@needs_sm80
def test_glm_decode_tc_empty_routes_and_bounds(glm):
    x, ids, weights = _case(glm, "uniform", 8, 12)
    ids.fill_(-1)
    assert (_d(glm, x, ids, weights) == 0).all()
    ids.fill_(GGLOBAL)
    assert (_d(glm, x, ids, weights) == 0).all()
    # int64 route ids reach the same schedule and the same bits.
    x, ids, weights = _case(glm, "boundary", 8, 14)
    out32 = _d(glm, x, ids, weights)
    args = (glm.pointers, glm.meta, glm.expert_map)
    workspace = torch.empty(
        exl3_ops.moe_workspace_size_decode_tc(8, GHIDDEN, GINTERMEDIATE, GTOPK, GLOCAL),
        dtype=torch.float32,
        device=x.device,
    )
    out64 = torch.empty_like(x)
    exl3_ops.moe_out_decode_tc(
        x,
        ids.to(torch.int64),
        weights,
        *args,
        GINTERMEDIATE,
        glm.bits_x2,
        out64,
        workspace,
    )
    _bitwise(out64, out32, "decode int64 ids")
    # Row and workspace bounds.
    x9, ids9, weights9 = _case(glm, "uniform", 9, 15)
    with pytest.raises(RuntimeError):
        _d(glm, x9, ids9, weights9)
    x, ids, weights = _case(glm, "boundary", 8, 3)
    out = torch.empty_like(x)
    small = torch.empty(
        exl3_ops.moe_workspace_size_decode_tc(7, GHIDDEN, GINTERMEDIATE, GTOPK, GLOCAL),
        dtype=torch.float32,
        device=x.device,
    )
    with pytest.raises(RuntimeError):
        exl3_ops.moe_out_decode_tc(
            x, ids, weights, *args, GINTERMEDIATE, glm.bits_x2, out, small
        )


@needs_sm80
def test_glm_decode_tc_refuses_unserved_formats(glm):
    x, ids, weights = _case(glm, "uniform", 4, 17)
    out = torch.empty_like(x)
    workspace = torch.empty(
        exl3_ops.moe_workspace_size_decode_tc(4, GHIDDEN, GINTERMEDIATE, GTOPK, GLOCAL),
        dtype=torch.float32,
        device=x.device,
    )
    args = (glm.pointers, glm.meta, glm.expert_map)
    for bits in (4, 10):
        with pytest.raises(RuntimeError):
            exl3_ops.moe_out_decode_tc(
                x, ids, weights, *args, GINTERMEDIATE, bits, out, workspace
            )


@needs_sm80
def test_glm35_decode_tc_numerics(glm35):
    """The 3.5-bit layer family on the small-row schedule, same gates."""
    _decode_pool(glm35, [("observed", 3), ("skew", 8)], "GLM 3.5-bit decode")


# ---------------------------------------------------------------------------
# Decode gate/up role modes (VLLM_EXL3_MOE_DECODE_TC_MODE): full / down /
# gate_down / up_down. The native role mode is process-static (read once), so a
# process cannot switch modes mid-flight: the driver spawns one pytest worker
# per mode. Each worker checks the same FP64-relative error gate as the full
# schedule, including changed routing and clipping boundaries. Different
# schedules are allowed to produce identical rounded outputs.
# ---------------------------------------------------------------------------

ROLE_MODE = os.environ.get("VLLM_EXL3_MOE_DECODE_TC_MODE")
ROLE_CASES = [("observed", 1), ("boundary", 4), ("skew", 8), ("sparse", 5)]


@needs_sm80
def test_glm_decode_tc_role_mode(glm):
    """Subprocess worker for one static gate/up role mode (see the driver)."""
    pool = _Fp64Pool()
    clipped = [0, 0, 0]
    for index, (scenario, rows) in enumerate(ROLE_CASES):
        x, ids, weights = _case(glm, scenario, rows, 900 + index)
        label = f"role {ROLE_MODE or 'full'}: {scenario} rows={rows}"
        candidate = _d(glm, x, ids, weights)
        _bitwise(_d(glm, x, ids, weights), candidate, f"{label}: repeat")
        reference, counts = _oracle(glm, x, ids, weights)
        clipped = [a + b for a, b in zip(clipped, counts)]
        pool.add(candidate, _v1(glm, x, ids, weights), reference, label)
    gate, up, routes = clipped
    assert gate >= 0.05 * routes * GINTERMEDIATE
    assert up >= 0.05 * routes * GINTERMEDIATE
    pool.check(f"moe_out_decode_tc role {ROLE_MODE or 'full'}")


def _spawn_role(mode):
    """Run the worker test in a fresh process under one static role mode."""
    env = dict(os.environ)
    env["VLLM_EXL3_MOE_DECODE_TC_MODE"] = mode
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{__file__}::test_glm_decode_tc_role_mode",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result


@needs_sm80
def test_glm_decode_tc_role_modes_qualify_and_fail_closed():
    """Each mode meets the numerical gate; invalid and obsolete modes fail."""
    for mode in ("full", "down", "gate_down", "up_down"):
        result = _spawn_role(mode)
        assert result.returncode == 0, (
            f"mode {mode}:\n{result.stdout[-3000:]}\n{result.stderr[-1500:]}"
        )
    # An unsupported mode must fail loudly, never fall back to the full schedule.
    result = _spawn_role("gate")
    assert result.returncode != 0
    assert "VLLM_EXL3_MOE_DECODE_TC_MODE" in result.stdout + result.stderr
    # The obsolete VLLM_EXL3_MOE_DECODE_TC_DOWN_ONLY is refused, not aliased.
    env = dict(os.environ)
    env["VLLM_EXL3_MOE_DECODE_TC_MODE"] = ""
    env["VLLM_EXL3_MOE_DECODE_TC_DOWN_ONLY"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{__file__}::test_glm_decode_tc_role_mode",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode != 0
    assert "VLLM_EXL3_MOE_DECODE_TC_DOWN_ONLY" in result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Decode glue kernels (VLLM_EXL3_MOE_DECODE_GLUE): block (default; one block per
# operand task, one warp per 128-block in the combine) and warp (the original
# warp-per-task kernels). They must agree bit for bit: same per-element FP32
# operations, an order-independent row peak, routes combined in route order.
# Process-static, so the driver spawns one worker per setting; the worker also
# covers the role modes' down-only path and non-local experts.
# ---------------------------------------------------------------------------

GLUE_OUT = os.environ.get("EXL3_TEST_GLUE_OUT")
GLUE_CASES = [("observed", 1), ("boundary", 2), ("uniform", 3), ("skew", 4),
              ("sparse", 5), ("uniform", 8), ("skew", 8)]


@needs_sm80
def test_glm_decode_tc_glue_worker(glm):
    """Subprocess worker: saves decode outputs under one glue setting."""
    if GLUE_OUT is None:
        pytest.skip("driven by test_glm_decode_tc_glue_bitwise")
    outputs = {}
    for index, (scenario, rows) in enumerate(GLUE_CASES):
        x, ids, weights = _case(glm, scenario, rows, 1300 + index)
        outputs[f"{scenario}/{rows}/{index}"] = _d(glm, x, ids, weights).cpu()
    remote = glm.expert_map.clone()
    remote[1::2] = -1
    split = SimpleNamespace(**{**vars(glm), "expert_map": remote})
    for index, rows in enumerate((3, 8)):
        x, ids, weights = _case(glm, "uniform", rows, 1350 + index)
        outputs[f"remote/{rows}/{index}"] = _d(split, x, ids, weights).cpu()
    torch.save(outputs, GLUE_OUT)


def _spawn_glue(glue, out, mode="full", stages="3"):
    env = dict(os.environ)
    env["VLLM_EXL3_MOE_DECODE_GLUE"] = glue
    env["VLLM_EXL3_MOE_DECODE_TC_MODE"] = mode
    env["VLLM_EXL3_MOE_DECODE_TC_STAGES"] = stages
    env["EXL3_TEST_GLUE_OUT"] = str(out)
    return subprocess.run(
        [sys.executable, "-m", "pytest",
         f"{__file__}::test_glm_decode_tc_glue_worker", "-q", "-p", "no:cacheprovider"],
        env=env, capture_output=True, text=True, timeout=300,
    )


@needs_sm80
def test_glm_decode_tc_glue_bitwise(tmp_path):
    """Block and warp glue kernels produce identical bits, full and down-only;
    every lab GEMM pipeline depth (VLLM_EXL3_MOE_DECODE_TC_STAGES) too."""
    reference = None
    for mode in ("full", "down"):
        saved = {}
        for glue in ("warp", "block"):
            path = tmp_path / f"{mode}-{glue}.pt"
            result = _spawn_glue(glue, path, mode)
            assert result.returncode == 0, (
                f"{mode}/{glue}:\n{result.stdout[-3000:]}\n{result.stderr[-1500:]}"
            )
            saved[glue] = torch.load(path)
        assert saved["warp"].keys() == saved["block"].keys()
        for key, value in saved["warp"].items():
            _bitwise(saved["block"][key], value, f"glue {mode}: {key}")
        if mode == "full":
            reference = saved["warp"]
    for stages in ("4", "6", "8"):
        path = tmp_path / f"stages-{stages}.pt"
        result = _spawn_glue("block", path, stages=stages)
        assert result.returncode == 0, (
            f"stages {stages}:\n{result.stdout[-3000:]}\n{result.stderr[-1500:]}"
        )
        for key, value in torch.load(path).items():
            _bitwise(value, reference[key], f"stages {stages}: {key}")
    result = _spawn_glue("block", tmp_path / "bad-stages.pt", stages="5")
    assert result.returncode != 0
    assert "VLLM_EXL3_MOE_DECODE_TC_STAGES" in result.stdout + result.stderr
    result = _spawn_glue("tile", tmp_path / "bad.pt")
    assert result.returncode != 0
    assert "VLLM_EXL3_MOE_DECODE_GLUE" in result.stdout + result.stderr
