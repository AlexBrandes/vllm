# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lab (default off): serve dense EXL3 matrices from an INT8 sidecar built from the original
checkpoint (VLLM_EXL3_DENSE_W8=<sidecar .safetensors>).

The sidecar (built by tools/convert_dense.py in github.com/AlexBrandes/cmp170hx-glm53-flash-exl3)
holds, per EXL3 matrix prefix:
  <prefix>.w8  int8 [N, K]  original weight in EXL3's rotated input space
                            (x W^T = H128(x * suh) W_r^T), one scale per output row
  <prefix>.sw  fp32 [N]
  <prefix>.suh fp16 [K]     (rotated matrices only)
  lm_head.w8 / lm_head.sw   unrotated (no suh); DeepSeek-V4.1: head.w8 / head.sw
Rotated matrices run W8A8: a fused H128(x * suh) + per-token int8 quant (exl3_i8_lab.act_int8), then
vLLM's cutlass_scaled_mm for <= VLLM_EXL3_DENSE_W8_SMALL_M rows (default 128) or the 128x256 CUTLASS
kernel (cmp_i8mm.scaled_mm_out) above. The LM head runs Marlin W8A16 (uint8b128, per-channel FP16
scales, FP16 activations) so the logits take no activation rounding
(VLLM_EXL3_DENSE_W8_LM_HEAD=0 keeps the EXL3 head). Matrices absent from the sidecar keep EXL3.
Lab (default off): VLLM_EXL3_DENSE_W8_DECODE=1 runs calls of <= 16 rows (N % 16, K % 128) through
exl3_i8_lab.act_int8_dec (the same quantisation with a wider block) and exl3_i8_lab.w8_decode_mm, a skinny
W8A8 GEMM that streams the row-major sidecar in contiguous runs; both equal the default path bit for bit
(card 4, 8 GLM shapes, 1..16 rows) at 1.17-1.40x its speed per call.
Lab libraries come from VLLM_EXL3_ONCE_LIB_DIR (default /opt/exl3-once), as for dense_prefill_once.
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

PATH = os.environ.get("VLLM_EXL3_DENSE_W8", "").strip()
LM_HEAD = os.environ.get("VLLM_EXL3_DENSE_W8_LM_HEAD", "1") == "1"
SMALL_M = int(os.environ.get("VLLM_EXL3_DENSE_W8_SMALL_M", "128"))
DECODE = os.environ.get("VLLM_EXL3_DENSE_W8_DECODE", "0") == "1"
# Grouped projections whose groups share one suh run one activation
# quantisation for all groups (_w8_grouped); 0 keeps one call per group.
GROUPED = os.environ.get("VLLM_EXL3_DENSE_W8_GROUPED", "1") == "1"
LIB_DIR = os.environ.get("VLLM_EXL3_ONCE_LIB_DIR", "/opt/exl3-once")

_file = None
_keys: frozenset[str] = frozenset()
_libs = False
_workspaces: dict[int, torch.Tensor] = {}
_registered = False
_logged: set[str] = set()


def enabled() -> bool:
    return bool(PATH)


# EXL3 matrix prefixes of the LM head: GLM/Qwen checkpoints keep "lm_head",
# DeepSeek-V4.1's keep DeepSeek's own name "head".
HEAD_PREFIXES = ("lm_head", "head")


def is_head(prefix: str) -> bool:
    return prefix in HEAD_PREFIXES


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
    if not enabled() or (is_head(prefix) and not LM_HEAD):
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
    if DECODE and m <= 16 and n % 16 == 0 and k % 128 == 0 and w8.is_contiguous():
        torch.ops.exl3_i8_lab.act_int8_dec(x, suh, xq, sx)
        _log(f"decode-w8d k={k} n={n}", m)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        torch.ops.exl3_i8_lab.w8_decode_mm(out, xq, w8, sx, sw)
        return out
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


# One launch for every wo_a group (w8_decode_grouped_mm; each group's arithmetic is w8_decode_mm's, so the output
# is bit for bit the per-group loop's) instead of G launches + G column copies. Card 1, 6 rows, wo_a 8 x [1024,
# 4096]: 96.6 -> 25.4 us per layer. VLLM_EXL3_DENSE_W8_GROUPED=0 restores the loop.
GROUPED = os.environ.get("VLLM_EXL3_DENSE_W8_GROUPED", "1") == "1"
# VLLM_EXL3_DENSE_W8_GROUPED_ACT=1: the grouped projection's activation quantisation reads x [T, G, K] in place
# (exl3_i8_lab.act_int8_dec_grouped, register-resident) instead of a transposed copy + act_int8; same bytes.
GROUPED_ACT = os.environ.get("VLLM_EXL3_DENSE_W8_GROUPED_ACT", "0") == "1"
_tables: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}


def _group_tables(w8s: list[torch.Tensor], sws: list[torch.Tensor]):
    """Device int64 tables of the groups' weight / scale pointers, built once per layer (outside CUDA graph
    capture; None while capturing an unseen layer, which then takes the per-group loop)."""
    key = tuple(w.data_ptr() for w in w8s) + tuple(s.data_ptr() for s in sws)
    tab = _tables.get(key)
    if tab is None:
        if not all(w.is_contiguous() for w in w8s) or not all(s.is_contiguous() for s in sws):
            return None
        if torch.cuda.is_current_stream_capturing():
            return None
        dev = w8s[0].device
        tab = (torch.tensor([w.data_ptr() for w in w8s], dtype=torch.int64, device=dev),
               torch.tensor([s.data_ptr() for s in sws], dtype=torch.int64, device=dev))
        _tables[key] = tab
    return tab


def _w8_grouped(x: torch.Tensor, w8s: list[torch.Tensor], sws: list[torch.Tensor],
                suh: torch.Tensor) -> torch.Tensor:
    """A block-diagonal projection whose groups share one input sign vector
    (DeepSeek-V4.1 ``wo_a``: 8 groups, sidecar built with share-suh): ``x``
    [T, G, K] -> [T, G * N]. One activation quantisation covers every group's
    rows (group-major, so each group's rows stay contiguous), then each group
    takes the GEMM :func:`_w8_linear` would pick for T rows. The quantisation
    is per row, so every group's operand equals its separate call's bit for
    bit (as do act_int8 and act_int8_dec)."""
    t, g, k = x.shape
    n = w8s[0].shape[0]
    if t == 0:
        return x.new_empty((0, g * n))
    m = g * t
    xq = torch.empty((m, k), dtype=torch.int8, device=x.device)
    sx = torch.empty((m,), dtype=torch.float32, device=x.device)
    if (GROUPED_ACT and DECODE and GROUPED and t <= 16 and k % 128 == 0 and x.is_contiguous()
            and hasattr(torch.ops.exl3_i8_lab, "act_int8_dec_grouped")):
        # Group-major rows read in place (no transposed copy); the same per-row quantisation.
        torch.ops.exl3_i8_lab.act_int8_dec_grouped(x, suh, xq, sx)
        xs = None
    else:
        xs = x.transpose(0, 1).reshape(m, k).contiguous()
    if xs is None:
        pass
    elif DECODE and m <= 16 and k % 128 == 0:
        torch.ops.exl3_i8_lab.act_int8_dec(xs, suh, xq, sx)
    else:
        torch.ops.exl3_i8_lab.act_int8(xs, suh, xq, sx)
    out = torch.empty((t, g * n), dtype=torch.bfloat16, device=x.device)
    if DECODE and GROUPED and t <= 16 and n % 16 == 0 and k % 128 == 0:
        tables = _group_tables(w8s, sws)
        if tables is not None:
            torch.ops.exl3_i8_lab.w8_decode_grouped_mm(out, xq, tables[0], tables[1], sx, n, k)
            _log(f"grouped-1launch g={g} k={k} n={n}", t)
            return out
    for i in range(g):
        xq_i, sx_i = xq[i * t : (i + 1) * t], sx[i * t : (i + 1) * t]
        w8, sw = w8s[i], sws[i]
        if DECODE and t <= 16 and n % 16 == 0 and k % 128 == 0 and w8.is_contiguous():
            o = torch.empty((t, n), dtype=torch.bfloat16, device=x.device)
            torch.ops.exl3_i8_lab.w8_decode_mm(o, xq_i, w8, sx_i, sw)
        elif t <= SMALL_M:
            from vllm import _custom_ops as ops

            o = ops.cutlass_scaled_mm(
                xq_i, w8.t(), sx_i.view(-1, 1), sw.view(1, -1), torch.bfloat16
            )
        else:
            o = torch.empty((t, n), dtype=torch.bfloat16, device=x.device)
            torch.ops.cmp_i8mm.scaled_mm_out(o, xq_i, w8, sx_i, sw)
        out[:, i * n : (i + 1) * n].copy_(o)
    _log(f"grouped g={g} k={k} n={n}", t)
    return out


def _w8_grouped_fake(x, w8s, sws, suh):
    return x.new_empty((x.shape[0], x.shape[1] * w8s[0].shape[0]))


def _w8_linear_fake(x, w8, sw, suh):
    return x.new_empty((x.shape[0], w8.shape[0]))


# Batch invariance (VLLM_EXL3_DENSE_W8_LM_HEAD_ROWS, default 64; 0 = one call of m rows as before): Marlin picks its
# tiling and k-partition from size_m, so a row's logits depended on how many rows shared the call
# (tests/verify/batch_invariance/w8_m_sweep.py: 1-2 bf16 ulps below 1024 rows). The head now always runs as Marlin
# calls of exactly LM_HEAD_ROWS rows (the last one zero-padded), so every row takes the same configuration whatever
# the step's row count; the sweep's shifted-batch test checks the in-call row position does not matter either.
LM_HEAD_ROWS = int(os.environ.get("VLLM_EXL3_DENSE_W8_LM_HEAD_ROWS", "64"))


def _w8_lm_head(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, n: int,
                k: int) -> torch.Tensor:
    from vllm import _custom_ops as ops
    from vllm.scalar_type import scalar_types

    m = x.shape[0]
    if m == 0:
        return x.new_empty((0, n))
    _log(f"lm_head k={k} n={n}", m)
    rows = LM_HEAD_ROWS
    if rows <= 0:
        out = ops.marlin_gemm(
            x.half(), None, q, None, s, None, None, None, _workspace(x.device),
            scalar_types.uint8b128, m, n, k, False, True, False,
        )
        return out.to(torch.bfloat16)
    xpad, ypad = _lm_head_buffers(x.device, rows, n, k)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    for lo in range(0, m, rows):
        hi = min(m, lo + rows)
        # Rows are independent in the GEMM, so stale padding rows only change the discarded outputs.
        xpad[: hi - lo].copy_(x[lo:hi])
        ops.marlin_gemm(
            xpad, ypad, q, None, s, None, None, None, _workspace(x.device),
            scalar_types.uint8b128, rows, n, k, False, True, False,
        )
        out[lo:hi] = ypad[: hi - lo]
    return out


_lm_bufs: dict = {}


def _lm_head_buffers(device: torch.device, rows: int, n: int, k: int):
    """Persistent FP16 input/output staging for the fixed-shape head calls (one pair per device and shape), so the
    head allocates nothing per call: per-call buffers were charged once per captured CUDA-graph size and cost the
    first stack12 screen 4.6 % of its KV cache."""
    key = (device.index, rows, n, k)
    bufs = _lm_bufs.get(key)
    if bufs is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("the LM head staging buffers are allocated by its first call, which must run outside a CUDA graph capture")
        bufs = (torch.zeros((rows, k), dtype=torch.float16, device=device),
                torch.empty((rows, n), dtype=torch.float16, device=device))
        _lm_bufs[key] = bufs
    return bufs


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
    direct_register_custom_op(op_name="exl3_w8_grouped", op_func=_w8_grouped,
                              mutates_args=[], fake_impl=_w8_grouped_fake)
    _registered = True


def _log(key: str, rows: int) -> None:
    if key not in _logged and not torch.cuda.is_current_stream_capturing():
        _logged.add(key)
        logger.info("EXL3 dense W8 dispatched: %s rows=%d", key, rows)
