# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EXL3 behind the sm80 plugin: registrations, platform and launch options."""

from types import SimpleNamespace
from typing import get_args

import pytest
from torch import nn

import vllm.config
from vllm.config.load import LoadConfig
from vllm.model_executor.layers import quantization
from vllm.model_executor.model_loader import get_model_loader
from vllm.platforms import current_platform
from vllm.sm80 import plugin
from vllm.sm80.kernels.exl3 import quant as exl3
from vllm.sm80.kernels.exl3.loader import Exl3ModelLoader


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(plugin, "_registered", False)
    plugin.register()
    plugin.register()


def test_upstream_tables_do_not_name_exl3():
    assert "exl3" not in get_args(quantization.QuantizationMethods)


def test_register_adds_the_method_and_the_load_format(registered):
    assert quantization.QUANTIZATION_METHODS.count("exl3") == 1
    assert quantization.get_quantization_config("exl3") is exl3.Exl3Config
    assert type(get_model_loader(LoadConfig(load_format="exl3"))) is Exl3ModelLoader


@pytest.mark.parametrize("is_cuda", [True, False])
def test_exl3_checkpoints_are_refused_off_cuda(monkeypatch, is_cuda):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: is_cuda)
    override = exl3.Exl3Config.override_quantization_method
    assert override({"quant_method": "gptq"}, None) is None
    if is_cuda:
        assert override({"quant_method": "exl3"}, None) is None
        assert override({"quant_method": "exl3"}, "exl3") is None
    else:
        for user_quant in (None, "exl3"):
            with pytest.raises(ValueError, match="NVIDIA CUDA platform"):
                override({"quant_method": "exl3"}, user_quant)


def _config(load_format="exl3", mode="data", runner_type="generate", vision=True):
    multimodal = SimpleNamespace(mm_encoder_tp_mode=mode) if vision else None
    return SimpleNamespace(
        load_config=SimpleNamespace(load_format=load_format),
        model_config=SimpleNamespace(
            multimodal_config=multimodal, runner_type=runner_type
        ),
    )


def test_launch_options_are_required():
    verify = exl3._verify_launch_options
    for accepted in (
        None,
        _config(),
        _config(vision=False),
        _config(mode="weights", runner_type="draft"),
    ):
        verify(accepted)
    with pytest.raises(ValueError, match="--load-format exl3, not 'auto'"):
        verify(_config(load_format="auto"))
    with pytest.raises(ValueError, match="--mm-encoder-tp-mode data, not 'weights'"):
        verify(_config(mode="weights"))


def test_method_selection_checks_the_current_config(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3", "1")
    config = exl3.Exl3Config()
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config_or_none",
        lambda: _config(load_format="safetensors"),
    )
    with pytest.raises(ValueError, match="--load-format exl3"):
        config.get_quant_method(nn.Linear(1, 1), "model.layers.0.mlp.down_proj")
