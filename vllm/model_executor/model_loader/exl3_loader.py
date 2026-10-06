# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load EXL3 constituents without materializing unassigned packed tensors."""

from contextlib import ExitStack

import torch
from safetensors import safe_open

from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.exl3 import (
    Exl3LinearMethod,
    Exl3MoEMethod,
)
from vllm.model_executor.layers.quantization.exl3_utils import (
    PLE_AUXILIARIES,
    Exl3LoadLedger,
)

logger = init_logger(__name__)


def load_exl3_weights(model: torch.nn.Module) -> set[str]:
    from vllm.model_executor.layers.quantization.exl3_ops import ensure_extension

    ensure_extension()
    config = get_current_vllm_config()
    quant = (
        model.quant_config if hasattr(model, "quant_config") else config.quant_config
    )
    manifest = quant.manifest
    if manifest is None:
        raise ValueError("EXL3 checkpoint manifest was not initialized")
    if config.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("EXL3 loading currently requires pipeline parallel size one")
    ledger = Exl3LoadLedger(manifest)
    is_draft = model.__class__.__name__ == "Qwen4ExpMTP"
    loaded = set()
    file_backed_parameters = set()
    modules = dict(model.named_modules())
    shared = getattr(model, "exl3_shared_source_keys", set())
    for name in manifest.tensors:
        if name in ledger.entries:
            continue
        if name in shared:
            ledger.mark(name, "unused", "shared target embedding/head owner")
        elif name.startswith("mtp.") and not is_draft:
            ledger.mark(name, "unused", "native MTP model owner")
        elif is_draft and not name.startswith("mtp."):
            if name.startswith(("lm_head.", "model.language_model.embed_tokens.")):
                continue
            ledger.mark(name, "unused", "target model owner")

    for name, module in modules.items():
        setup = getattr(module, "setup_exl3_file", None)
        if setup is None:
            continue
        source = quant.source_prefix(f"{name}.ngram_embedding")
        table = manifest.ple_tables.get(source)
        if table is None:
            raise ValueError(f"Missing EXL3 PLE source for module: {name}")
        expected = {table.name} | {f"{source}.{suffix}" for suffix in PLE_AUXILIARIES}
        consumed = setup(str(table.path), table.name)
        if consumed != expected:
            raise ValueError(f"Incomplete or unexpected EXL3 PLE source join: {source}")
        for key in sorted(consumed):
            ledger.mark(key, "file-backed" if key == table.name else "consumed", name)
        file_backed_parameters.add(f"{name}.ngram_embedding.weight")

    with ExitStack() as stack:
        files = {
            path: stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
            for path in sorted({header.path for header in manifest.tensors.values()})
        }
        for name, module in modules.items():
            method = getattr(module, "quant_method", None)
            if isinstance(method, Exl3LinearMethod):
                bias_offset = 0
                for weight in module.exl3_weights:
                    weight.load(files, ledger)
                    bias_key = f"{weight.matrix.prefix}.bias"
                    if bias_key in manifest.tensors:
                        bias = getattr(module, "bias", None)
                        if bias is None:
                            raise ValueError(
                                f"EXL3 checkpoint bias has no owner: {bias_key}"
                            )
                        header = manifest.tensors[bias_key]
                        if (
                            header.dtype not in ("F16", "BF16", "F32")
                            or len(header.shape) != 1
                        ):
                            raise ValueError(f"Invalid EXL3 bias: {bias_key}")
                        if (
                            header.shape[0]
                            not in (
                                weight.matrix.padded_out,
                                sum(s.stop - s.start for s in weight.output_slices),
                            )
                            and header.shape[0] != module.output_size
                        ):
                            raise ValueError(f"EXL3 bias geometry mismatch: {bias_key}")
                        data = files[header.path].get_slice(bias_key)
                        remaining = weight.logical_out
                        for part in weight.output_slices:
                            width = min(part.stop - part.start, remaining)
                            bias.data[bias_offset : bias_offset + width].copy_(
                                data[part.start : part.start + width]
                            )
                            bias_offset += width
                            remaining -= width
                        ledger.mark(bias_key, "consumed", name)
                        loaded.add(f"{name}.bias")
                    elif getattr(module, "bias", None) is not None:
                        raise ValueError(f"Missing EXL3 bias: {bias_key}")
                loaded.update(f"{name}.{key}" for key, _ in module.named_parameters())
            elif isinstance(method, Exl3MoEMethod):
                for key in method.nonlocal_keys:
                    ledger.mark(key, "unused", "expert assigned to another EP rank")
                for expert in module.exl3_experts:
                    for weight in expert:
                        weight.load(files, ledger)
                loaded.update(f"{name}.{key}" for key, _ in module.named_parameters())

        for name, header in manifest.tensors.items():
            if name in ledger.entries:
                continue
            if name.endswith((".trellis", ".suh", ".svh", ".mcg", ".mul1")):
                raise ValueError(f"EXL3 packed tensor has no runtime owner: {name}")
            tensor = files[header.path].get_tensor(name)
            consumed = model.load_weights(iter([(name, tensor)]))
            if not consumed:
                raise ValueError(f"EXL3 plain tensor has no runtime owner: {name}")
            loaded.update(consumed)
            ledger.mark(name, "consumed", ",".join(sorted(consumed)))
        ledger.finish()
    missing = {
        name
        for name, _ in model.named_parameters()
        if name not in loaded and name not in file_backed_parameters
    }
    if missing:
        raise ValueError(f"Unloaded EXL3 runtime parameters: {sorted(missing)[:16]}")
    for name, parameter in model.named_parameters():
        if not parameter.is_cuda:
            raise ValueError(f"EXL3 parameter must remain CUDA-owned: {name}")
    model.exl3_load_ledger = ledger
    logger.info(
        "EXL3 packed format, BF16 activations; PLE placement=%s; TP rank=%d/%d, EP=%s; "
        "whole routed experts, replicated shared experts and vision",
        quant.ple_placement,
        get_tensor_model_parallel_rank(),
        config.parallel_config.tensor_parallel_size,
        config.parallel_config.enable_expert_parallel,
    )
    assignments = {
        tuple(module.quant_method.global_expert_ids)
        for module in modules.values()
        if isinstance(getattr(module, "quant_method", None), Exl3MoEMethod)
    }
    for assignment in sorted(assignments):
        logger.info("EXL3 local routed expert IDs: %s", assignment)
    return loaded
