# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of Qwen4Exp router precision and MTP hidden normalization."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor import parameter as parameter_module
from vllm.model_executor.layers import linear as linear_module
from vllm.model_executor.layers.fused_moe.router import gate_linear
from vllm.model_executor.models import qwen3_next
from vllm.models.qwen4_exp.nvidia import model as model_module
from vllm.models.qwen4_exp.nvidia import mtp as mtp_module
from vllm.models.qwen4_exp.nvidia import precision as precision_module


@pytest.mark.parametrize("name", ["VLLM_QWEN4_ROUTER_FP32", "VLLM_QWEN4_MTP_PER_STREAM_NORM"])
def test_switch_defaults_and_environment(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    assert envs.environment_variables[name]() is False
    monkeypatch.setenv(name, "1")
    assert envs.environment_variables[name]() is True
    monkeypatch.setenv(name, "0")
    assert envs.environment_variables[name]() is False


class _GateConsumer(nn.Module):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate

    def forward(self, hidden_states, router_logits):
        self.last_logits = self.gate(hidden_states)[0]
        return hidden_states


@pytest.mark.parametrize("enabled", [False, True])
def test_router_uses_bf16_operands_and_fp32_output(monkeypatch, enabled, default_vllm_config):
    monkeypatch.setenv("VLLM_QWEN4_ROUTER_FP32", str(int(enabled)))
    # Simulate the device capability, not the GEMM result. The cuBLAS call is
    # executed below as CPU FP32 arithmetic on its actual BF16 operands.
    platform = gate_linear.current_platform
    monkeypatch.setattr(platform, "is_cuda", lambda: True)
    monkeypatch.setattr(platform, "is_rocm", lambda: False)
    monkeypatch.setattr(platform, "is_device_capability", lambda *args: False)
    monkeypatch.setattr(platform, "is_device_capability_family", lambda *args: False)
    monkeypatch.setattr(qwen3_next, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(linear_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear_module, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(qwen3_next, "get_ep_group", lambda: SimpleNamespace(device_group=SimpleNamespace(size=lambda: 2)))
    monkeypatch.setattr(qwen3_next, "FusedMoEFactory", lambda **kw: _GateConsumer(kw["gate"]))
    config = SimpleNamespace(
        hidden_size=2560,
        num_experts=512,
        shared_expert_intermediate_size=0,
        num_experts_per_tok=10,
        moe_intermediate_size=640,
        hidden_act="silu",
    )
    parallel = SimpleNamespace(use_sequence_parallel_moe=False, enable_expert_parallel=True,
                               enable_eplb=False, eplb_config=SimpleNamespace(num_redundant_experts=0))
    cfg = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=config), parallel_config=parallel, quant_config=None)
    banners = []
    monkeypatch.setattr(precision_module, "logger", SimpleNamespace(info_once=lambda *args: banners.append(args)))
    with torch.device("cpu"):
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            block = model_module.Qwen4ExpSparseMoeBlock(cfg, "model.layers.0.mlp")
        finally:
            torch.set_default_dtype(old_dtype)
    assert block.gate is block.experts.gate
    assert block.gate.weight.dtype == torch.bfloat16
    assert block.gate.out_dtype == (torch.float32 if enabled else None)
    assert block.gate.allow_cublas_router_gemm is enabled
    generator = torch.Generator().manual_seed(117)
    x = torch.randn(3, 2560, generator=generator).to(torch.bfloat16)
    block.gate.weight.data.copy_(
        torch.randn(512, 2560, generator=generator).to(torch.bfloat16)
    )
    calls = []
    real_mm = torch.mm

    def cpu_mm(a, b, *, out_dtype=None):
        calls.append((a.dtype, b.dtype, out_dtype))
        assert a.dtype == b.dtype == torch.bfloat16
        assert out_dtype == torch.float32
        return real_mm(a.float(), b.float())

    # The legacy fallback uses F.linear; the candidate must use the out_dtype
    # GEMM instead, not round BF16 logits and cast them back to FP32.
    monkeypatch.setattr(torch, "mm", cpu_mm)
    block(x)
    logits = block.experts.last_logits
    reference = real_mm(x.float(), block.gate.weight.float().T)
    if enabled:
        assert logits.dtype == torch.float32
        torch.testing.assert_close(logits, reference, atol=0, rtol=0)
        assert len(calls) == 1
        assert not torch.equal(logits, reference.bfloat16().float())
    else:
        assert logits.dtype == torch.bfloat16
        assert calls == []
    assert banners and banners[0][1:] == (("FP32" if enabled else "model dtype"), int(enabled))


class _DraftLayer(nn.Module):
    def forward(self, **kwargs):
        x = kwargs["hidden_states"]
        return x, x, torch.zeros(x.shape[0], 4)


class _Mixer(nn.Module):
    def combine_and_mix(self, hidden, block, injection):
        return hidden, hidden.unflatten(-1, (4, 2560)).mean(-2), None


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mtp_hidden_norm_selects_real_forward_path(monkeypatch, enabled, dtype, default_vllm_config):
    monkeypatch.setenv("VLLM_QWEN4_MTP_PER_STREAM_NORM", str(int(enabled)))
    config = SimpleNamespace(vocab_size=8, hidden_size=2560, hc_count=4, num_hidden_layers=48,
                             mtp_num_hidden_layers=1, rms_norm_eps=1e-6, hc_lowrank=4)
    parallel = SimpleNamespace(tensor_parallel_size=1, enable_hc_sp=False, use_sequence_parallel_moe=False)
    cfg = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=config), parallel_config=parallel,
                          quant_config=None)
    monkeypatch.setattr(mtp_module, "_make_draft_vllm_config", lambda *args: cfg)
    monkeypatch.setattr(mtp_module, "set_current_vllm_config", lambda *args, **kw: nullcontext())
    monkeypatch.setattr(mtp_module, "ColumnParallelLinear", lambda *args, **kw: nn.Identity())
    monkeypatch.setattr(mtp_module, "VocabParallelEmbedding", lambda *args, **kw: nn.Embedding(8, 2560))
    monkeypatch.setattr(mtp_module, "Qwen4ExpDecoderLayer", lambda *args, **kw: _DraftLayer())
    monkeypatch.setattr(mtp_module, "is_model_fused_shared_expert_compatible", lambda *args: False)
    monkeypatch.setattr(mtp_module, "GatedResidual", lambda *args, **kw: _Mixer())
    monkeypatch.setattr(mtp_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True))
    banners = []
    monkeypatch.setattr(precision_module, "logger", SimpleNamespace(info_once=lambda *args: banners.append(args)))
    predictor = mtp_module.Qwen4ExpMultiTokenPredictor(vllm_config=cfg).to(dtype)
    # Different stream scales distinguish one stack RMS from four stream RMSs.
    x = torch.linspace(-1.0, 1.0, 2560).repeat(2, 4, 1)
    x = (x * torch.tensor([0.125, 1.0, 4.0, 16.0])[None, :, None]).to(dtype)
    predictor.pre_fc_norm_hidden.weight.data.copy_(torch.linspace(-0.25, 0.5, 10240).to(dtype))
    x32 = x.float()
    variance = x32.square().mean(-1, keepdim=True) if enabled else x32.square().mean((-2, -1), keepdim=True)
    affine = 1.0 + predictor.pre_fc_norm_hidden.weight.float().view(4, 2560)
    expected = (x32 * torch.rsqrt(variance + config.rms_norm_eps) * affine).to(dtype).flatten(-2)
    _, got = predictor.forward(input_ids=None, positions=torch.arange(2), hidden_states=x.flatten(-2),
                               inputs_embeds=torch.zeros(2, 2560, dtype=dtype))
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert predictor.pre_fc_norm_hidden.weight.shape == (10240,)
    assert got.dtype == dtype and got.shape == (2, 10240)
    assert predictor.hidden_norm.per_stream is enabled
    assert banners and banners[0][1:] == (("per-stream" if enabled else "flat"), int(enabled))
    # The selection is fixed at construction, as it must be for graph replay.
    monkeypatch.setenv("VLLM_QWEN4_MTP_PER_STREAM_NORM", str(int(not enabled)))
    _, repeated = predictor.forward(input_ids=None, positions=torch.arange(2), hidden_states=x.flatten(-2),
                                    inputs_embeds=torch.zeros(2, 2560, dtype=dtype))
    torch.testing.assert_close(repeated, got, rtol=0, atol=0)
