# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lab (default off): weight-only Marlin quantization of the DFlash drafter's BF16 linears
(VLLM_DFLASH_DRAFT_QUANT = "w8" | "w4").

w8: uint8b128 with one scale per output channel; w4: uint4b8 with 128-row groups. Round-to-
nearest from the BF16 checkpoint after loading (vLLM's marlin_quantize), BF16 activations and
scales. Draft quality only changes acceptance: every emitted token is still verified by the
target. Linears whose shape Marlin cannot take keep BF16.
"""

import os

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

MODE = os.environ.get("VLLM_DFLASH_DRAFT_QUANT", "").strip().lower()
_registered = False
_workspaces: dict[int, torch.Tensor] = {}


def enabled() -> bool:
    return MODE in ("w8", "w4")


def _workspace(device):
    ws = _workspaces.get(device.index)
    if ws is None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new,
        )

        ws = marlin_make_workspace_new(device, 4)
        _workspaces[device.index] = ws
    return ws


def _qtype():
    from vllm.scalar_type import scalar_types

    return scalar_types.uint8b128 if MODE == "w8" else scalar_types.uint4b8


def _linear(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, n: int, k: int) -> torch.Tensor:
    from vllm import _custom_ops as ops

    m = x.shape[0]
    if m == 0:
        return x.new_empty((0, n))
    return ops.marlin_gemm(x, None, q, None, s, None, None, None, _workspace(x.device),
                           _qtype(), m, n, k, False, True, False)


def _linear_fake(x, q, s, n, k):
    return x.new_empty((x.shape[0], n))


def _register():
    global _registered
    if not _registered:
        from vllm.utils.torch_utils import direct_register_custom_op

        direct_register_custom_op(op_name="dflash_marlin_linear", op_func=_linear,
                                  mutates_args=[], fake_impl=_linear_fake)
        _registered = True


class MarlinDraftLinearMethod:
    """Stand-in quant_method for an already-loaded BF16 linear."""

    def __init__(self, n: int, k: int):
        self.n, self.k = n, k

    def apply(self, layer, x, bias=None):
        shape = x.shape[:-1]
        out = torch.ops.vllm.dflash_marlin_linear(
            x.reshape(-1, x.shape[-1]).contiguous(), layer.marlin_q, layer.marlin_s,
            self.n, self.k)
        if bias is not None:
            out = out + bias
        return out.reshape(*shape, self.n)

    def process_weights_after_loading(self, layer):
        return None


def quantize_drafter(model: nn.Module) -> None:
    if not enabled():
        return
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        marlin_quantize,
    )

    _register()
    group = -1 if MODE == "w8" else 128
    done, kept, before, after = [], [], 0, 0
    for name, mod in model.named_modules():
        if not isinstance(mod, LinearBase) or not isinstance(
            getattr(mod, "quant_method", None), UnquantizedLinearMethod
        ):
            continue
        w = getattr(mod, "weight", None)
        if w is None or w.dim() != 2 or w.dtype != torch.bfloat16:
            continue
        n, k = w.shape
        if k % 128 or n % 64:
            kept.append(name)
            continue
        _, q, s = marlin_quantize(w.data.t().contiguous(), _qtype(), group)
        before += w.numel() * 2
        mod.marlin_q = nn.Parameter(q, requires_grad=False)
        mod.marlin_s = nn.Parameter(s.to(torch.bfloat16), requires_grad=False)
        after += q.numel() * q.element_size() + s.numel() * 2
        mod.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16, device=w.device),
                                  requires_grad=False)
        mod.quant_method = MarlinDraftLinearMethod(n, k)
        done.append(name)
    torch.cuda.empty_cache()
    logger.info(
        "DFlash drafter quantized to %s (Marlin): %d linears, %.2f -> %.2f GB; kept BF16: %s",
        MODE, len(done), before / 1e9, after / 1e9, kept or "none",
    )
