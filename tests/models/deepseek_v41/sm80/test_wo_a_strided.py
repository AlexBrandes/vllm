# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The grouped wo_a with its groups' rows read in place
(VLLM_DEEPSEEK_V41_WO_A_STRIDED).

On CPU a float64 matmul stands in for the GEMM: the method hands each group's
rows as a view of the input (no copy kernel), and the outputs are the copy
path's bytes. On CUDA the real Marlin W8A16 GEMM of a loaded MXFP8 layer gives
the copy path's bytes for 1 to 48 rows and the deployed group counts.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.sm80.layers.grouped_linear import GroupedLinearMethod

BF16 = torch.bfloat16
# Kernels that copy a tensor (``Tensor.contiguous`` dispatches ``clone``).
COPIES = {"clone", "copy_", "_to_copy"}


class _Ops(TorchDispatchMode):
    """The aten ops dispatched inside the block."""

    def __init__(self):
        super().__init__()
        self.names = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.names.append(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


class _Float64Linear(LinearMethodBase):
    """Records each input it is applied to; computes in float64 (outside any
    op recording: it stands in for the GEMM kernel) so the bytes depend only
    on the values."""

    def __init__(self):
        self.inputs = []

    def create_weights(self, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, bias=None):
        self.inputs.append(x)
        with _disable_current_modes():
            return (x.double() @ layer.weight.double().t()).to(x.dtype)


def _grouped(groups, rank_out, k, strided=None):
    generator = torch.Generator().manual_seed(groups)
    layer = nn.Module()
    weight = nn.Parameter(
        torch.randn(groups * rank_out, k, generator=generator).to(BF16),
        requires_grad=False,
    )
    weight.output_dim = 0
    layer.register_parameter("weight", weight)
    layer.output_size_per_partition = groups * rank_out
    layer.input_size_per_partition = k
    inner = _Float64Linear()
    layer.quant_method = inner
    if strided is None:
        GroupedLinearMethod.adopt(layer, groups)
    else:
        GroupedLinearMethod.adopt(layer, groups, strided_input=strided)
    layer.quant_method.process_weights_after_loading(layer)
    return layer, inner


def _attn_out(rows, groups, k):
    """The attention output as ``_wo_a_proj`` sees it: [T, groups * k] viewed
    as [T, groups, k]."""
    generator = torch.Generator().manual_seed(rows)
    return (
        torch.randn(rows, groups * k, generator=generator)
        .to(BF16)
        .view(rows, groups, k)
    )


def _project(layer, x):
    with _Ops() as ops:
        out = layer.quant_method.apply_grouped(layer, x)
    return out, ops.names


@pytest.mark.parametrize("groups", [2, 4])
@pytest.mark.parametrize("rows", [1, 6, 48])
def test_strided_input_reads_the_rows_in_place(groups, rows):
    rank_out, k = 16, 64
    copy_layer, copy_inner = _grouped(groups, rank_out, k)
    layer, inner = _grouped(groups, rank_out, k, strided=True)
    x = _attn_out(rows, groups, k)

    want, copy_ops = _project(copy_layer, x)
    got, ops = _project(layer, x)

    # The copy path: one copy kernel per group (a single row is already
    # contiguous, so no copy there).
    copies = groups if rows > 1 else 0
    assert sum(name in COPIES for name in copy_ops) == copies
    assert all(a.is_contiguous() for a in copy_inner.inputs)
    # Strided: no copy kernel; each group's rows are a view of x.
    assert not [name for name in ops if name in COPIES]
    assert len(inner.inputs) == groups
    for g, a in enumerate(inner.inputs):
        assert a.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        assert a.data_ptr() == x[:, g].data_ptr()
        assert tuple(a.shape) == (rows, k)
        assert a.stride() == (groups * k, 1)
    # Same values in, same bytes out, in one [T, groups * output] tensor.
    assert ops.count("cat") == copy_ops.count("cat") == 1
    assert got.shape == want.shape == (rows, groups * rank_out)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_default_adoption_keeps_the_copy_path():
    """Other adopters (and the switch off) keep the contiguous copies."""
    layer, inner = _grouped(2, 16, 64)
    assert layer.quant_method.strided_input is False
    x = _attn_out(3, 2, 64)
    _, ops = _project(layer, x)
    assert sum(name in COPIES for name in ops) == 2
    assert all(
        a.is_contiguous() and a.data_ptr() != x[:, g].data_ptr()
        for g, a in enumerate(inner.inputs)
    )


def test_switch_default(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_DEEPSEEK_V41_WO_A_STRIDED", raising=False)
    assert envs.VLLM_DEEPSEEK_V41_WO_A_STRIDED is False
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_WO_A_STRIDED", "1")
    assert envs.VLLM_DEEPSEEK_V41_WO_A_STRIDED is True


def _attention(wo_a_method):
    from vllm.models.deepseek_v41.nvidia import sm80_attention as sm80

    layer = object.__new__(sm80.DeepseekV41Sm80Attention)
    nn.Module.__init__(layer)
    object.__setattr__(layer, "wo_a", SimpleNamespace(quant_method=wo_a_method))
    return layer


def test_dev_ab_switch_sets_every_grouped_wo_a(monkeypatch):
    import vllm.envs as envs
    from vllm.sm80 import dev_ab

    monkeypatch.setenv("VLLM_DEEPSEEK_V41_WO_A_STRIDED", "0")
    switch = dev_ab.SWITCHES["dsv41_wo_a_strided"]
    assert switch.env == "VLLM_DEEPSEEK_V41_WO_A_STRIDED"
    assert switch.graph_bound and switch.output_identical
    target = [GroupedLinearMethod(object(), 2) for _ in range(2)]
    draft = GroupedLinearMethod(object(), 2)
    shared = _attention(target[0])
    runner = SimpleNamespace(
        model=nn.ModuleList([shared, _attention(target[1])]),
        speculator=SimpleNamespace(model=nn.ModuleList([shared, _attention(draft)])),
    )
    assert switch.unavailable(runner) is None and switch.current(runner) is False
    switch.apply(runner, True)
    assert [m.strided_input for m in (*target, draft)] == [True] * 3
    assert switch.current(runner) is True
    assert envs.VLLM_DEEPSEEK_V41_WO_A_STRIDED is True
    switch.apply(runner, False)
    assert [m.strided_input for m in (*target, draft)] == [False] * 3
    assert envs.VLLM_DEEPSEEK_V41_WO_A_STRIDED is False
    # A model without a grouped wo_a (EXL3 or BF16 wo_a, another model).
    plain = SimpleNamespace(
        model=nn.ModuleList([_attention(object()), nn.Linear(2, 2)]), speculator=None
    )
    assert "no grouped" in switch.unavailable(plain)


# -- CUDA: the real Marlin W8A16 GEMM ---------------------------------------------


def _marlin_wo_a(monkeypatch, groups, rank_out, k):
    """A loaded MXFP8 wo_a of ``groups`` local groups on Marlin, as the hybrid
    checkpoint's (adopted with the copy path; the arm is the attribute)."""
    from tests.models.deepseek_v41.sm80.construct import set_groups
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.quantization.modelopt import (
        CkptCtx,
        ModelOptLinearMethod,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
        kMxfp8Static,
    )

    generator = torch.Generator().manual_seed(groups)
    weight = (torch.rand(groups * rank_out, k, generator=generator) * 2 - 1).to(
        torch.float8_e4m3fn
    )
    scale = torch.randint(
        118, 130, (groups * rank_out // 32, k // 32), generator=generator
    ).to(torch.uint8)
    layer = nn.Module()
    layer.register_parameter("bias", None)
    method = ModelOptLinearMethod(
        QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic),
        CkptCtx(scale_block_size=(32, 32)),
    )
    method.input_dtype = method.out_dtype = BF16

    def load(param, loaded, *args):
        param.data.copy_(loaded)

    set_groups(monkeypatch, 1, 0)
    with torch.device("cuda"):
        method.create_weights(
            layer, k, [groups * rank_out], k, groups * rank_out, BF16,
            weight_loader=load,
        )  # fmt: skip
    assert type(method.kernel).__name__ == "MarlinMxfp8LinearKernel"
    layer.weight.weight_loader(layer.weight, weight.cuda())
    layer.weight_scale.weight_loader(
        layer.weight_scale, scale.cuda().view(torch.float8_e8m0fnu)
    )
    layer.quant_method = method
    GroupedLinearMethod.adopt(layer, groups)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Marlin runs on CUDA")
@pytest.mark.parametrize("groups", [2, 4])
def test_marlin_strided_rows_are_the_copy_path_bytes(monkeypatch, groups):
    """DeepSeek-V4.1 wo_a per group: 4096 inputs (8 heads of 512) to
    o_lora_rank 1024; 2 local groups at TP4 (o_groups 8), 4 at TP2."""
    rank_out, k = 1024, 4096
    layer = _marlin_wo_a(monkeypatch, groups, rank_out, k)
    method = layer.quant_method
    generator = torch.Generator(device="cuda").manual_seed(7)
    # Decode/verify sizes, then prefill sizes past Marlin's second launch
    # (65+ rows: A_ptr advanced by lda) and its parallel row slices (128+).
    for rows in [*range(1, 49), 64, 65, 127, 128, 200, 1000, 2048, 8192, 8193]:
        x = torch.randn(rows, groups * k, generator=generator, device="cuda").to(BF16)
        x = x.view(rows, groups, k)
        method.strided_input = False
        want = method.apply_grouped(layer, x)
        method.strided_input = True
        with _Ops() as ops:
            got = method.apply_grouped(layer, x)
        assert not [name for name in ops.names if name in COPIES], rows
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), rows
    torch.cuda.synchronize()
