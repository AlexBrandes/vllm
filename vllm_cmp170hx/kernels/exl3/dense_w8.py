# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lab (default off): serve dense EXL3 matrices from an INT8 sidecar built from the original
checkpoint (VLLM_EXL3_DENSE_W8=<sidecar .safetensors>).

The sidecar (runtime/glm53-exl3-campaign/decode-int8-20261110/convert_dense.py) holds, per EXL3
matrix prefix:
  <prefix>.w8  int8 [N, K]  original weight in EXL3's rotated input space
                            (x W^T = H128(x * suh) W_r^T), one scale per output row
  <prefix>.sw  fp32 [N]
  <prefix>.suh fp16 [K]     (rotated matrices only)
  lm_head.w8 / lm_head.sw   unrotated (no suh)
Rotated matrices run W8A8: a fused H128(x * suh) + per-token int8 quant (exl3_i8_lab.act_int8), then
vLLM's cutlass_scaled_mm for <= VLLM_EXL3_DENSE_W8_SMALL_M rows (default 128) or the 128x256 CUTLASS
kernel (cmp_i8mm.scaled_mm_out) above. The LM head runs Marlin W8A16 (uint8b128, per-channel FP16
scales, FP16 activations) so the logits take no activation rounding
(VLLM_EXL3_DENSE_W8_LM_HEAD=0 keeps the EXL3 head). Matrices absent from the sidecar keep EXL3.
Lab libraries come from VLLM_EXL3_ONCE_LIB_DIR (default /opt/exl3-once), as for dense_prefill_once.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

PATH = os.environ.get("VLLM_EXL3_DENSE_W8", "").strip()
LM_HEAD = os.environ.get("VLLM_EXL3_DENSE_W8_LM_HEAD", "1") == "1"
SMALL_M = int(os.environ.get("VLLM_EXL3_DENSE_W8_SMALL_M", "128"))
LIB_DIR = os.environ.get("VLLM_EXL3_ONCE_LIB_DIR", "/opt/exl3-once")

_file = None
_keys: frozenset[str] = frozenset()
_libs = False
_workspaces: dict[int, torch.Tensor] = {}
_registered = False
_logged: set[str] = set()


def enabled() -> bool:
    return bool(PATH)


def _open():
    global _file, _keys
    if _file is None:
        from safetensors import safe_open

        _file = safe_open(PATH, "pt", device="cpu")
        _keys = frozenset(_file.keys())
        logger.info_once(
            "EXL3 dense W8 sidecar %s: %d matrices (LM head %s)",
            PATH,
            sum(1 for k in _keys if k.endswith(".w8")),
            "on" if LM_HEAD else "off",
            scope="process",
        )
    return _file


def covers(prefix: str) -> bool:
    if not enabled() or (prefix == "lm_head" and not LM_HEAD):
        return False
    _open()
    return f"{prefix}.w8" in _keys


def tensor(prefix: str, suffix: str) -> torch.Tensor:
    return _open().get_tensor(f"{prefix}.{suffix}")


def load_libs() -> None:
    global _libs
    if not _libs:
        torch.ops.load_library(os.path.join(LIB_DIR, "exl3_i8_lab.so"))
        torch.ops.load_library(os.path.join(LIB_DIR, "cmp_i8mm.so"))
        _libs = True
    register_ops()


def marlin_pack_lm_head(w8: torch.Tensor, sw: torch.Tensor):
    """int8 [N, K] + fp32 [N] -> Marlin uint8b128 weight and permuted FP16 channel scales."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )

    n, k = w8.shape
    q = (w8.t().to(torch.int32) + 128).contiguous()  # [K, N] in 0..255
    q = q.view(k // 4, 4, n)
    packed = (
        q[:, 0] | (q[:, 1] << 8) | (q[:, 2] << 16) | (q[:, 3] << 24)
    ).contiguous()  # GPTQ layout: 4 consecutive k per int32, low bits first
    marlin_q = ops.gptq_marlin_repack(packed, k, n, 8)
    scales = marlin_permute_scales(sw.view(1, n).half(), k, n, -1)
    return marlin_q, scales


def _workspace(device: torch.device) -> torch.Tensor:
    ws = _workspaces.get(device.index)
    if ws is None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new,
        )

        ws = marlin_make_workspace_new(device, 4)
        _workspaces[device.index] = ws
    return ws


def _w8_linear(x: torch.Tensor, w8: torch.Tensor, sw: torch.Tensor,
               suh: torch.Tensor) -> torch.Tensor:
    m, k = x.shape[0], w8.shape[1]
    n = w8.shape[0]
    if m == 0:
        return x.new_empty((0, n))
    xq = torch.empty((m, k), dtype=torch.int8, device=x.device)
    sx = torch.empty((m,), dtype=torch.float32, device=x.device)
    torch.ops.exl3_i8_lab.act_int8(x, suh, xq, sx)
    _log(f"{'decode' if m <= SMALL_M else 'prefill'} k={k} n={n}", m)
    if m <= SMALL_M:
        from vllm import _custom_ops as ops

        return ops.cutlass_scaled_mm(
            xq, w8.t(), sx.view(-1, 1), sw.view(1, -1), torch.bfloat16
        )
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    torch.ops.cmp_i8mm.scaled_mm_out(out, xq, w8, sx, sw)
    return out


def _w8_linear_fake(x, w8, sw, suh):
    return x.new_empty((x.shape[0], w8.shape[0]))


def _w8_lm_head(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, n: int,
                k: int) -> torch.Tensor:
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types

    m = x.shape[0]
    if m == 0:
        return x.new_empty((0, n))
    _log(f"lm_head k={k} n={n}", m)
    out = ops.marlin_gemm(
        x.half(), None, q, None, s, None, None, None, _workspace(x.device),
        scalar_types.uint8b128, m, n, k, False, True, False,
    )
    return out.to(torch.bfloat16)


def _w8_lm_head_fake(x, q, s, n, k):
    return x.new_empty((x.shape[0], n))


def register_ops() -> None:
    global _registered
    if _registered:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(op_name="exl3_w8_linear", op_func=_w8_linear,
                              mutates_args=[], fake_impl=_w8_linear_fake)
    direct_register_custom_op(op_name="exl3_w8_lm_head", op_func=_w8_lm_head,
                              mutates_args=[], fake_impl=_w8_lm_head_fake)
    _registered = True


def _log(key: str, rows: int) -> None:
    if key not in _logged and not torch.cuda.is_current_stream_capturing():
        _logged.add(key)
        logger.info("EXL3 dense W8 dispatched: %s rows=%d", key, rows)
