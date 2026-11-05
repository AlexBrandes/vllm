# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Construct GLM-5.3-Flash EXL3 from the real checkpoint headers on CPU."""

import json
import os
import socket
import traceback
from collections import Counter
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

CHECKPOINT = os.environ.get("VLLM_TEST_EXL3_GLM_CHECKPOINT")


def _params_on_meta(module, name, param, register):
    register(module, name, param)
    if param is not None and param.device.type != "meta":
        moved = torch.nn.Parameter(param.to("meta"), requires_grad=param.requires_grad)
        moved.__dict__.update(param.__dict__)
        module._parameters[name] = moved


def _packed_owners(model):
    from vllm_cmp170hx.kernels.exl3.quant import Exl3LinearMethod, Exl3MoEMethod

    owners = []
    modules = {}
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        if isinstance(method, Exl3LinearMethod):
            for weight in module.exl3_weights:
                owners.append(weight.matrix.prefix)
                modules[weight.matrix.prefix] = name
        elif isinstance(method, Exl3MoEMethod):
            for expert in module.exl3_experts:
                for weight in expert:
                    owners.append(weight.matrix.prefix)
                    modules[weight.matrix.prefix] = name
    return owners, modules


def _build(port, checkpoint, output, draft):
    os.environ.update(
        HF_HUB_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        VLLM_DISABLE_SHARED_EXPERTS_STREAM="1",
    )
    try:
        from vllm.config import set_current_vllm_config
        from vllm.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.engine.arg_utils import EngineArgs
        from vllm.model_executor.model_loader.utils import initialize_model
        from vllm.platforms import current_platform
        from vllm.platforms.interface import DeviceCapability, PlatformEnum
        from vllm_cmp170hx import guard as sm80_guard
        from vllm_cmp170hx.plugin import register as register_sm80
        from vllm.utils.torch_utils import set_default_torch_dtype
        from vllm.v1.attention.backends.mla.prefill.cpu_sdpa import (
            CPUSDPAMLAPrefillBackend,
        )

        platform = type(current_platform)
        platform.device_type = "meta"
        platform._enum = PlatformEnum.CUDA
        platform.get_device_capability = classmethod(
            lambda cls, device_id=0: DeviceCapability(8, 0)
        )
        platform.current_device = classmethod(lambda cls: torch.device("meta"))
        platform.get_device_communicator_cls = classmethod(
            lambda cls: "vllm.distributed.device_communicators."
            "base_device_communicator.DeviceCommunicatorBase"
        )
        platform.get_attn_backend_cls = classmethod(
            lambda cls, selected_backend, attn_selector_config, num_heads=None: (
                "vllm.v1.attention.backends.mla.cpu_mla.CPUMLABackend"
            )
        )
        platform.get_mla_prefill_backend_cls = classmethod(
            lambda cls, mla_selector_config: CPUSDPAMLAPrefillBackend
        )
        register_sm80()
        sm80_guard.check_plugin_loaded = lambda: None
        torch.ops._C.cutlass_scaled_mm_supports_fp8 = lambda capability: False
        torch.ops._C.silu_and_mul_with_clamp = lambda *_args, **_kwargs: None
        register = torch.nn.Module.register_parameter
        torch.nn.Module.register_parameter = lambda module, name, param: (
            _params_on_meta(module, name, param, register)
        )
        vllm_config = EngineArgs(
            model=checkpoint,
            distributed_executor_backend="external_launcher",
            max_model_len=32768,
            max_num_batched_tokens=256,
            enforce_eager=True,
            load_format="exl3",
            mm_encoder_tp_mode="data",
            speculative_config=(
                {"method": "mtp", "num_speculative_tokens": 3} if draft else None
            ),
        ).create_engine_config()
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1)
            with set_default_torch_dtype(vllm_config.model_config.dtype):
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=(
                        vllm_config.speculative_config.draft_model_config
                        if draft
                        else None
                    ),
                )
        if draft:
            owners, modules = _packed_owners(model)
            Path(output).write_text(json.dumps({"owners": owners, "modules": modules}))
            return
        backbone = model.language_model.model
        kda = backbone.layers[0].self_attn
        conv_shape = tuple(kda.q_conv1d.weight.shape)
        captured_conv = {}
        for branch in ("q", "k", "v"):
            param = getattr(kda, f"{branch}_conv1d").weight

            def capture_conv(_param, tensor, branch=branch):
                captured_conv[branch] = {
                    "shape": list(tensor.shape),
                    "value": tensor.flatten()[0].item(),
                }

            param.weight_loader = capture_conv
        fused_conv = torch.cat(
            [
                torch.full(conv_shape, value, dtype=torch.float32)
                for value in (1.0, 2.0, 3.0)
            ]
        )
        loaded_conv = backbone.load_weights(
            [("layers.0.self_attn.conv1d.weight", fused_conv)]
        )
        owners, modules = _packed_owners(model)
        Path(output).write_text(
            json.dumps(
                {
                    "owners": owners,
                    "modules": modules,
                    "conv_shape": list(conv_shape),
                    "captured_conv": captured_conv,
                    "loaded_conv": sorted(loaded_conv),
                }
            )
        )
    except BaseException:
        Path(output + ".error").write_text(traceback.format_exc())
        raise


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _construct(tmp_path_factory, *, draft):
    if CHECKPOINT is None:
        pytest.skip("VLLM_TEST_EXL3_GLM_CHECKPOINT is not set")
    checkpoint = Path(CHECKPOINT)
    if not (checkpoint / "quantization_config.json").is_file():
        pytest.fail(f"Not an EXL3 checkpoint: {checkpoint}")
    output = str(tmp_path_factory.mktemp("glm_exl3") / "owners.json")
    process = mp.get_context("spawn").Process(
        target=_build, args=(_free_port(), str(checkpoint), output, draft)
    )
    process.start()
    process.join(timeout=1800)
    error = Path(output + ".error")
    if error.exists():
        pytest.fail(error.read_text())
    if process.is_alive():
        process.kill()
        pytest.fail("GLM EXL3 construction timed out")
    assert process.exitcode == 0
    return json.loads(Path(output).read_text())


@pytest.fixture(scope="module")
def constructed(tmp_path_factory):
    return _construct(tmp_path_factory, draft=False)


@pytest.fixture(scope="module")
def constructed_draft(tmp_path_factory):
    return _construct(tmp_path_factory, draft=True)


def test_every_draft_matrix_has_exactly_one_runtime_owner(constructed_draft):
    from vllm_cmp170hx.kernels.exl3.utils import Exl3Manifest

    manifest = Exl3Manifest(CHECKPOINT)
    expected = {
        prefix
        for prefix in manifest.matrices
        if prefix.startswith("model.language_model.layers.45.")
    }
    counts = Counter(constructed_draft["owners"])
    assert set(counts) == expected
    assert all(count == 1 for count in counts.values())


def test_every_target_matrix_has_exactly_one_runtime_owner(constructed):
    from vllm_cmp170hx.kernels.exl3.utils import Exl3Manifest

    manifest = Exl3Manifest(CHECKPOINT)
    expected = {
        prefix
        for prefix in manifest.matrices
        if not prefix.startswith("model.language_model.layers.45.")
    }
    counts = Counter(constructed["owners"])
    assert set(counts) == expected
    assert all(count == 1 for count in counts.values())


def test_glm_specific_fusions_resolve_to_packed_sources(constructed):
    modules = constructed["modules"]
    assert modules["model.language_model.layers.0.self_attn.qkv_proj"].endswith(
        "layers.0.self_attn.qkv_proj"
    )
    assert modules["model.language_model.layers.3.self_attn.q_a_proj"].endswith(
        "layers.3.self_attn.fused_qkv_a_proj"
    )
    assert modules["model.visual.blocks.0.attn.q_proj"].endswith(
        "visual.blocks.0.attn.qkv"
    )
    assert modules["lm_head"].endswith("language_model.lm_head")


def test_fused_kda_convolution_loads_each_runtime_branch(constructed):
    assert constructed["captured_conv"] == {
        "q": {"shape": constructed["conv_shape"], "value": 1.0},
        "k": {"shape": constructed["conv_shape"], "value": 2.0},
        "v": {"shape": constructed["conv_shape"], "value": 3.0},
    }
    assert constructed["loaded_conv"] == [
        "layers.0.self_attn.k_conv1d.weight",
        "layers.0.self_attn.q_conv1d.weight",
        "layers.0.self_attn.v_conv1d.weight",
    ]
