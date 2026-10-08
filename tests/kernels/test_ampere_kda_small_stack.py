# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the small-stack RecoverSSM verify configs (VLLM_AMPERE_KDA_SMALL_STACK,
vllm/models/glm5next/nvidia/ops/kda_decode.py).

CPU part: the switch, read once at the first launch; the block configs it
changes (nseq 7-8 at 16 heads, every RecoverSSM verify, the commit fold's
included) and the ones it leaves; the banner and the "set but off" lines; the
verify's stack per thread compiled for sm_80 (Triton's own toolchain, no GPU)
with and without the switch, today's verify and the fold's.
GPU part (skipped without CUDA): with the switch against without, on the same
inputs, bitwise (outputs normed and SKIP_NORM, records, conv window, state
slot), today's verify and the fold's, for nseq 1..8 and T 1..8.

    python tests/kernels/test_ampere_kda_small_stack.py           # CPU part
    CUDA_VISIBLE_DEVICES=<gpu> python tests/kernels/test_ampere_kda_small_stack.py
"""

import os
import re
import subprocess
import sys
import tempfile
import types

try:
    import pytest
except ImportError:
    pytest = None
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

H, D = 16, 128


def _fresh_switch(value):
    """Set the switch and forget the value read at the first launch."""
    from vllm.models.glm5next.nvidia.ops import kda_decode as k2

    if value is None:
        os.environ.pop("VLLM_AMPERE_KDA_SMALL_STACK", None)
    else:
        os.environ["VLLM_AMPERE_KDA_SMALL_STACK"] = value
    k2._SMALL_STACK = None
    return k2


def _launch_config(k2, nseq, recover, fold=False):
    """The (BV, num_warps) kda_decode_v2 launches with (its launch replaced)."""
    seen = []

    class _Grid:
        def __getitem__(self, grid):
            return lambda *a, **kw: seen.append((kw["BV"], kw["num_warps"]))

    T = 4
    M = nseq * T
    f32, bf16, i32 = torch.float32, torch.bfloat16, torch.int32
    qkv = torch.zeros(M, 3 * H * D, dtype=bf16)
    records = tuple(torch.zeros(nseq, k2._WS_T, H, D, dtype=f32) for _ in range(3)) \
        if recover else None
    orig, k2._kda_decode_v2_kernel = k2._kda_decode_v2_kernel, _Grid()
    orig_ctr, k2._counter = k2._counter, lambda device, heads: (
        torch.zeros(2 * k2._CTR_SLOTS, dtype=i32), torch.zeros(1, dtype=f32))
    try:
        k2.kda_decode_v2(
            qkv, torch.zeros(1, M, H, dtype=bf16), torch.zeros(M, 128, dtype=bf16),
            torch.zeros(M, 128, dtype=bf16), torch.zeros(H * D, 128, dtype=bf16),
            torch.zeros(H * D, 128, dtype=bf16), torch.zeros(4, 3 * H * D, 10, dtype=bf16),
            torch.zeros(3 * H * D, 4, dtype=f32), None, torch.zeros(D, dtype=bf16),
            torch.zeros(4, H, D, D, dtype=f32), torch.zeros(nseq, dtype=i32),
            torch.zeros(nseq, 8, dtype=i32), torch.ones(nseq, dtype=i32),
            torch.arange(nseq + 1, dtype=i32) * T, T, torch.zeros(H, dtype=f32),
            torch.zeros(H * D, dtype=f32), out=torch.empty(1, M, H, D, dtype=bf16),
            records=records,
            fold=(torch.zeros(nseq, dtype=i32), torch.arange(nseq, dtype=i32)) if fold else None)
    finally:
        k2._kda_decode_v2_kernel, k2._counter = orig, orig_ctr
    return seen[0]


def test_switch_and_configs():
    """Off unless VLLM_AMPERE_KDA_SMALL_STACK=1, read once at the first launch.
    On: RecoverSSM verifies at nseq 7-8, the commit fold's included, take one
    8-row warp per CTA (BV 8: the 32-row warp's layouts, four times the CTAs);
    every other nseq and every non-RecoverSSM launch keeps its config. Off:
    today's configs."""
    from vllm import envs

    assert "VLLM_AMPERE_KDA_SMALL_STACK" in envs.environment_variables
    saved = os.environ.pop("VLLM_AMPERE_KDA_SMALL_STACK", None)
    try:
        k2 = _fresh_switch(None)
        today = {n: k2._select(n, H, D) for n in range(1, 9)}
        assert today[7] == today[8] == (32, 1) and today[6] == (32, 2)
        for n in range(1, 9):
            assert _launch_config(k2, n, True) == today[n]
            assert _launch_config(k2, n, True, fold=True) == today[n]
        k2 = _fresh_switch("1")
        assert k2.SMALL_STACK_ROWS_MAX == 8
        for n in range(1, 9):
            want = (8, 1) if n >= 7 else today[n]
            assert _launch_config(k2, n, True) == want, n
            assert _launch_config(k2, n, True, fold=True) == want, n
            assert _launch_config(k2, n, False) == today[n], n
        os.environ["VLLM_AMPERE_KDA_SMALL_STACK"] = "0"      # read once: stays on
        assert _launch_config(k2, 8, True) == (8, 1)
    finally:
        _fresh_switch(saved)


def test_banner_and_set_but_off():
    """The warmup names the switch when it is on and a verify config changes,
    and says why it is off when the fused verify is not used or no batch size
    up to max_num_seqs has a 32-row warp."""
    from vllm.models.glm5next.nvidia.ops import kda_decode as k2

    saved = os.environ.pop("VLLM_AMPERE_KDA_SMALL_STACK", None)
    saved_dk = os.environ.get("VLLM_AMPERE_DECODE_KERNELS")
    msgs = []
    orig_info, orig_once = k2.logger.info, k2.logger.info_once
    k2.logger.info_once = lambda msg, *a, **k: msgs.append(msg % a if a else msg)
    k2.logger.info = lambda *a, **k: None
    orig_warm, k2.warmup = k2.warmup, lambda *a, **k: None
    layer = types.SimpleNamespace(_conv_state_dim_first=True, _kda_v2=True, local_num_heads=16,
                                  head_dim=128, num_spec=7, _kda_step_tile=True,
                                  _kda_wide=False, _kda_recover=True)
    worker = types.SimpleNamespace(vllm_config=types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(max_num_seqs=8)))
    orig_find, k2.find_module = k2.find_module, lambda model, attr: model
    try:
        _fresh_switch("1")
        k2._warmup_kda_v2(worker, layer, "cpu", [8])
        worker.vllm_config.scheduler_config.max_num_seqs = 6
        k2._warmup_kda_v2(worker, layer, "cpu", [8])
        k2._warmup_kda_v2(worker, None, "cpu", [8])
        os.environ["VLLM_AMPERE_DECODE_KERNELS"] = "0"
        k2.warmup_from_worker(worker, [8])
    finally:
        k2.logger.info, k2.logger.info_once = orig_info, orig_once
        k2.warmup, k2.find_module = orig_warm, orig_find
        if saved_dk is None:
            os.environ.pop("VLLM_AMPERE_DECODE_KERNELS", None)
        else:
            os.environ["VLLM_AMPERE_DECODE_KERNELS"] = saved_dk
        _fresh_switch(saved)
    assert any("small-stack configs active (VLLM_AMPERE_KDA_SMALL_STACK=1)" in m
               for m in msgs), msgs
    assert any("VLLM_AMPERE_KDA_SMALL_STACK=1 set but off" in m
               and "no verify batch up to max_num_seqs 6" in m for m in msgs), msgs
    assert any("VLLM_AMPERE_KDA_SMALL_STACK=1 set but off" in m and "not used by this model" in m
               for m in msgs), msgs
    assert any("VLLM_AMPERE_KDA_SMALL_STACK=1 set but off" in m
               and "VLLM_AMPERE_DECODE_KERNELS is off" in m for m in msgs), msgs


def _verify_stack(nseq, T, skip, small_stack, fold):
    """Stack bytes per thread of the RecoverSSM verify compiled for sm_80 on
    the CPU, with the specialisation a served launch gets (pointers and the
    divisible strides 16-aligned, the unit strides constant)."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.experimental.gluon._runtime import GluonASTSource

    from vllm.models.glm5next.nvidia.ops import kda_decode as k2

    KA, CK = 128, 4
    ptr = {"x": "*bf16", "fa": "*bf16", "ga": "*bf16", "wf": "*bf16", "wg": "*bf16",
           "conv_state": "*bf16", "conv_w": "*fp32", "beta": "*bf16", "norm_w": "*bf16",
           "out": "*bf16", "rec": "*fp32", "cu_seqlens": "*i32", "conv_idx": "*i32",
           "ssm_idx": "*i32", "num_acc": "*i32", "a_log": "*fp32", "g_bias": "*fp32",
           "ctr": "*i32", "gctr": "*i32", "ws": "*fp32", "rc": "*fp32", "rk": "*fp32",
           "re": "*fp32", "fl": "*i32", "rr": "*i32", "scale": "fp32", "eps": "fp32"}
    div16 = {"stride_x_t", "stride_fa_t", "stride_ga_t", "stride_wf", "stride_wg",
             "stride_beta_t", "stride_out_t", "stride_cs_slot", "stride_cs_tok",
             "stride_rec_slot"}
    fn = k2._kda_decode_v2_kernel
    bv, nw = k2._select(nseq, H, D, small_stack=small_stack)
    const = dict(conv_b=None, H=H, D=D, KA=KA, CONV_K=CK, PROJ=H * D, LOWER_BOUND=-5.0,
                 HAS_BIAS=False, L2_EPS=1e-6, BV=bv, NV=D // bv,
                 BS=triton.next_power_of_2(CK - 2 + T), BT=triton.next_power_of_2(T),
                 WS_T=8, DC=32, SW=8, NW=nw, RECOVER=True, SKIP_NORM=skip, ST_CACHE="",
                 FOLD=fold, stride_cs_chan=1, stride_sidx_tok=1)
    sig = {a: ("constexpr" if a in const else ptr.get(a, "i32")) for a in fn.arg_names}
    attrs = {(i,): [["tt.divisibility", 16]] for i, a in enumerate(fn.arg_names)
             if a not in const and (ptr.get(a, "").startswith("*") or a in div16)}
    k = triton.compile(GluonASTSource(fn, sig, const, attrs),
                       target=GPUTarget("cuda", 80, 32), options={"num_warps": nw})
    cu = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin",
                      "cuobjdump")
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(k.asm["cubin"])
        f.flush()
        out = subprocess.run([cu, "--dump-resource-usage", f.name], capture_output=True,
                             text=True, check=True).stdout
    return int(re.search(r"STACK:(\d+)", out).group(1))


def test_verify_stack():
    """Compiled for sm_80, the largest-stack verify plans (nseq 7-8: T 5
    SKIP_NORM, T 7 normed), today's verify and the fold's, stay under the
    driver's default 1,024 B of stack per thread with the small-stack config,
    against more than 2,000 B without it."""
    for fold in (False, True):
        for nseq, T, skip in ((8, 5, True), (7, 7, False)):
            today = _verify_stack(nseq, T, skip, False, fold)
            small = _verify_stack(nseq, T, skip, True, fold)
            print(f"  {'fold' if fold else 'off'} n{nseq}_t{T}_{'skip' if skip else 'norm'}: "
                  f"stack {today} B -> {small} B")
            assert today > 2000 and small < 1024, (fold, nseq, T, skip, today, small)


# ------------------------------------------------------------------ GPU
def gpu_test_small_stack_bitwise():
    """Real kernel, one card: with the switch against without, on the same
    inputs, bitwise: outputs (normed and SKIP_NORM), records, the conv window
    and the state slot, for nseq 1..8, T 1..8 and n accepted 1, T // 2, T.
    Today's verify: verify k -> commit(n) -> verify k + 1. The fold's: FOLD
    verify k (nothing owed) -> FOLD verify k + 1 owing n."""
    from tests.kernels.test_ampere_kda_recover import _context, _inputs
    from vllm.sm80.kernels.kda import recover as kr

    def v2(k2, p, conv, rec, ssm, acc, T, records, fold=None, skip=False):
        nseq = ssm.shape[0]
        out = torch.empty(1, nseq * T, H, D, device="cuda", dtype=torch.bfloat16)
        return k2.kda_decode_v2(p.qkv, p.beta, p.fa, p.ga, p.wf, p.wg, conv.transpose(-1, -2),
                                p.cw, None, p.nw, rec, ssm[:, 0], ssm, acc, p.qsl, 8, p.al,
                                p.gb, out=out, records=records, skip_norm=skip, fold=fold)

    def run(small, fold, nseq, T, n, skip):
        k2 = _fresh_switch("1" if small else None)
        k2.warmup(plans=((nseq, T),), recover=True, width=8, fold=fold)
        nslot = 2 + nseq * 8
        p = _inputs(nseq, T, 7000 + 10 * T + nseq, nslot)
        p2 = _inputs(nseq, T, 9000 + 10 * T + nseq, nslot)
        p.qsl = p2.qsl = torch.arange(0, nseq + 1, device="cuda", dtype=torch.int32) * T
        ssm = torch.tensor([[1 + 8 * s] for s in range(nseq)], device="cuda",
                           dtype=torch.int32)
        ones = torch.ones(nseq, device="cuda", dtype=torch.int32)
        pool = torch.full((1, 3, 8, kr.WS_T, H, D), float("nan"), device="cuda")
        rec, conv = p.rec.clone(), p.conv.clone()
        recs = kr.layer_records(pool, 0)
        if fold:
            slots = torch.tensor([(3 * s + 5) % 8 for s in range(nseq)], device="cuda",
                                 dtype=torch.int32)
            zero = torch.zeros(nseq, device="cuda", dtype=torch.int32)
            v2(k2, p, conv, rec, ssm, ones, T, recs, fold=(zero, slots))
            owed = torch.full((nseq,), n, device="cuda", dtype=torch.int32)
            y = v2(k2, p2, conv, rec, ssm, ones, T, recs, fold=(owed, slots), skip=skip)
        else:
            v2(k2, p, conv, rec, ssm, ones, T, recs)
            ctx = _context(conv, rec, pool)
            ctx.commit(torch.full((nseq,), n, device="cuda", dtype=torch.int32),
                       ssm[:, 0].contiguous(), p.qsl)
            y = v2(k2, p2, conv, rec, ssm, ones, T, recs, skip=skip)
        torch.cuda.synchronize()
        rows = [1 + 8 * s for s in range(nseq)]
        return (y.view(torch.int16), pool.view(torch.int32), conv[rows].view(torch.int16),
                rec[rows].view(torch.int32))

    saved = os.environ.pop("VLLM_AMPERE_KDA_SMALL_STACK", None)
    nfail = checks = 0
    try:
        for fold in (False, True):
            for nseq in range(1, 9):
                for T in range(1, 9):
                    for n in sorted({1, max(1, T // 2), T}):
                        for skip in (False, True):
                            a = run(False, fold, nseq, T, n, skip)
                            b = run(True, fold, nseq, T, n, skip)
                            checks += 1
                            if not all(torch.equal(x, y) for x, y in zip(a, b)):
                                nfail += 1
                                print(f"  FAIL fold={fold} nseq={nseq} T={T} n={n} skip={skip}")
    finally:
        _fresh_switch(saved)
    print(f"  {checks - nfail}/{checks} bitwise")
    assert nfail == 0


CPU_TESTS = (test_switch_and_configs, test_banner_and_set_but_off, test_verify_stack)
GPU_TESTS = (gpu_test_small_stack_bitwise,)

if pytest is not None:
    _needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an sm_80 GPU")
    test_gpu_small_stack_bitwise = _needs_gpu(gpu_test_small_stack_bitwise)


def _main():
    import traceback

    tests = list(CPU_TESTS)
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" and torch.cuda.is_available():
        tests += list(GPU_TESTS)
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}", flush=True)
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}", flush=True)
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
