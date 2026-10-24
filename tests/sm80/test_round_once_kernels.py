# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from pathlib import Path
import subprocess
import sys

import torch


def interpreted_cases():
    from tests.sm80.test_round_once_all_reduce import exact_sum, reference_round
    from vllm.sm80.distributed import round_once_kernels as kernels
    for dtype in (torch.bfloat16, torch.float16):
        for world in (1, 2, 4, 8):
            span = (45 if dtype == torch.bfloat16 else 42) - (world - 1).bit_length()
            parts = torch.randint(-400, 400, (world, 517), generator=torch.Generator().manual_seed(41)).to(dtype)
            parts[:, :4] = -0.0
            if world == 4 and dtype == torch.bfloat16:
                parts[:, 4] = torch.tensor([1, 2.0**-8, 2.0**-30, 0], dtype=dtype)
                parts[:, 5] = torch.tensor([1, 2.0**-43, -1, 0], dtype=dtype)
            # Triton's interpreter truncates BF16 stores and mishandles BF16
            # subnormal loads. FP32 staging is lossless for both input formats;
            # exercise the fused arithmetic there and RNE-narrow using torch.
            # AOT checks the native narrowing instruction and absence of FTZ;
            # the GPU bits fixture also includes native BF16 subnormal inputs.
            out = torch.empty(517, dtype=torch.float32)
            kernels._sum_round[(3,)](parts.float(), out, 517, world, 2.0**span, True, 256)
            out = out.to(dtype)
            want = reference_round(exact_sum(parts), dtype)
            assert torch.equal(out.view(torch.int16), want.view(torch.int16)), (dtype, world)
            flag = torch.zeros((), dtype=torch.int32)
            kernels._record_error[(1,)](out, flag, 517, 1024)
            assert flag.item() == 0
    for dtype, tiny in ((torch.bfloat16, 2.0**-133), (torch.float16, 2.0**-24)):
        parts = torch.tensor([[tiny, torch.finfo(dtype).max]] * 4, dtype=dtype)
        out = torch.empty(2, dtype=torch.float32)
        span = 43 if dtype == torch.bfloat16 else 40
        kernels._sum_round[(1,)](parts.float(), out, 2, 4, 2.0**span, True, 256)
        out = out.to(dtype)
        assert out[0].item() == 4 * tiny and torch.isposinf(out[1]), (dtype, out.tolist())
        flag = torch.zeros((), dtype=torch.int32)
        kernels._record_error[(1,)](out, flag, 2, 1024)
        assert flag.item() == 0, "finite sums may correctly round to infinity"
    for bad in (2.0**-60, float("nan"), float("inf")):
        parts = torch.zeros((4, 517), dtype=torch.bfloat16)
        parts[:, 400] = torch.tensor([1, bad, -1, 0], dtype=torch.bfloat16)
        out = torch.empty(517, dtype=torch.float32)
        kernels._sum_round[(3,)](parts.float(), out, 517, 4, 2.0**43, True, 256)
        out = out.to(parts.dtype)
        assert torch.isnan(out[400]), "unsupported input must travel as poison"
        flag = torch.zeros((), dtype=torch.int32)
        kernels._record_error[(1,)](out, flag, 517, 1024)
        assert flag.item() == 1
        out.zero_()
        kernels._record_error[(1,)](out, flag, 517, 1024)
        assert flag.item() == 1, "a later valid call must not erase a pending failure"


def test_fused_rounding_poison_and_sticky_errors_on_cpu():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TRITON_INTERPRET="1", PYTHONPATH=str(root))
    done = subprocess.run([sys.executable, "-c", "from tests.sm80.test_round_once_kernels import interpreted_cases; interpreted_cases()"],
                          cwd=root, env=env, capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, (done.stdout + done.stderr)[-8000:]


def test_guarded_kernels_compile_without_device_traps():
    import re
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from vllm.sm80.distributed import round_once_kernels as kernels
    target = GPUTarget("cuda", 80, 32)
    def build(kernel, types, constants):
        signature = {p: types.get(p, "constexpr") for p in kernel.arg_names}
        compiled = triton.compile(ASTSource(kernel, signature, constexprs=constants), target=target)
        assert compiled.asm["cubin"]
        assert "trap;" not in compiled.asm["ptx"] and "__assertfail" not in compiled.asm["ptx"]
        assert ".ftz" not in compiled.asm["ptx"]
        if kernel is kernels._sum_round:
            target_dtype = "bf16" if types["Out"] == "*bf16" else "f16"
            assert re.search(r"cvt\.rn\." + target_dtype + r"(?:x2)?\.f32", compiled.asm["ptx"])
    for dtype, span in (("bf16", 43), ("fp16", 40)):
        for checked in (False, True):
            build(kernels._sum_round, {"Parts": "*" + dtype, "Out": "*" + dtype},
                  dict(N=517, WORLD=4, SPAN=2.0**span, CHECK=checked, BLOCK=256))
        build(kernels._record_error, {"Values": "*" + dtype, "Error": "*i32"}, dict(N=517, BLOCK=1024))
