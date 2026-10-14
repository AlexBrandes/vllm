# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-core dense EXL3 linear for prefill steps (VLLM_EXL3_DENSE_PREFILL_TC).

The operator ``linear_out_prefill`` (csrc/exl3/dense_prefill_tc.cu) computes
what ``linear_out`` computes, on the tensor cores: exact integer-grid main
products, FP32 sums in k order, and, for the last part-empty wave of tiles,
a k split whose partial tiles the last arriving block adds in k order
(arithmetic in csrc/exl3/dense_prefill_tc.h). Its results differ from
``linear_out``'s in the last bits.

``ops._linear`` asks ``route`` for the operator of each call. A row chunk of
the call takes ``linear_out_prefill`` only when all of these hold:
- the flag is set and ``resolve`` found the gate open (sm_80 with 74
  multiprocessors, a vLLM config to read the speculative width from);
- the running step is a prefill step by the scheduler (``step_phase``: some
  request has more than 1 + num_speculative_tokens tokens scheduled), never
  judged from a row count, and the stream is not capturing a CUDA graph;
- the launch has more than 64 rows, at least 16 output tiles of 128 x 128, a
  covered format (bits_x2 10/12/14, codebook 2) and a 16-byte aligned trellis;
- with VLLM_EXL3_DENSE_SPLITK on, ``linear_out_splitk`` would not split the
  launch's K: every launch split-K splits keeps split-K.
Every other launch runs the operator ``_linear`` picked before (linear_out or
linear_out_splitk), unchanged. Row chunks, the workspace and its size are
``_linear``'s; the operator places its stream-K partial tiles and their
counters inside ``linear_workspace_size`` of the launch and zeroes the
counters on every call.
"""

import functools

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.sm80.kernels.exl3 import step_phase

logger = init_logger(__name__)

# Launches of at most this many rows keep today's kernels (the decode bound).
MAX_SHIPPED_ROWS = 64
# The multiprocessor count the schedule was validated at.
VALIDATED_SMS = 74
# VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE -> set_prefill_schedule mode.
SCHEDULES = {"planned": 0, "whole": 1}
# The library operators this path needs (csrc/exl3/dense_prefill_tc.cu).
OPERATORS = ("linear_out_prefill", "linear_prefill_applies", "set_prefill_schedule")

# Fixed by resolve(), before any launch or graph capture.
active = False
_max_decode_query_len = 0
_sms = 0
_fake_registered = False


def closed_reason(
    capability: tuple[int, int] | None, sms: int | None, vllm_config
) -> str | None:
    """Why linear_out_prefill must not serve this process, or None."""
    if capability != (8, 0):
        return f"device capability {capability}, not (8, 0)"
    if sms != VALIDATED_SMS:
        return f"{sms} multiprocessors, not {VALIDATED_SMS}"
    if vllm_config is None:
        return "no vLLM config is set, so the decode query length is unknown"
    return None


def resolve() -> None:
    """VLLM_EXL3_DENSE_PREFILL_TC for this process, with its log line.

    Called once by ops.ensure_extension, after the library loaded."""
    global active, _max_decode_query_len, _sms, _fake_registered
    if not envs.VLLM_EXL3_DENSE_PREFILL_TC:
        return
    from vllm.config import get_current_vllm_config_or_none

    capability, sms = None, None
    if torch.cuda.is_available():
        capability = tuple(torch.cuda.get_device_capability())
        sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    vllm_config = get_current_vllm_config_or_none()
    reason = closed_reason(capability, sms, vllm_config)
    if reason is not None:
        logger.info_once(
            "VLLM_EXL3_DENSE_PREFILL_TC is set but the gate is closed: %s; every "
            "dense EXL3 launch keeps today's kernels.",
            reason,
            scope="process",
        )
        return
    schedule = envs.VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE
    for name in OPERATORS:
        try:
            torch._C._dispatch_find_schema_or_throw(f"_exl3_C::{name}", "")
        except RuntimeError as exc:
            raise RuntimeError(
                f"VLLM_EXL3_DENSE_PREFILL_TC needs _exl3_C::{name}: rebuild the "
                "EXL3 extension from this tree (python "
                "csrc/exl3/build_standalone.py --out vllm), or unset the flag. "
                f"Detail: {exc}"
            ) from exc
    if not _fake_registered:
        from vllm.sm80.kernels.exl3.ops import _linear_fake

        torch.library.register_fake("_exl3_C::linear_out_prefill")(_linear_fake)
        _fake_registered = True
    set_prefill_schedule(SCHEDULES[schedule])
    _max_decode_query_len = step_phase.max_decode_query_len(vllm_config)
    _sms = sms
    logger.info_once(
        "EXL3 dense prefill tensor-core path active (VLLM_EXL3_DENSE_PREFILL_TC): "
        "in prefill steps (a request with more than %d tokens scheduled), dense "
        "launches of more than %d rows and at least 16 output tiles (bits_x2 "
        "10/12/14, codebook 2) take linear_out_prefill, schedule %s; split-K "
        "launches, decode steps and graph captures keep today's kernels.",
        _max_decode_query_len,
        MAX_SHIPPED_ROWS,
        schedule,
        scope="process",
    )
    active = True


def linear_prefill_applies(
    rows: int, padded_in: int, padded_out: int, bits_x2: int, codebook: int
) -> bool:
    """Whether linear_out_prefill covers this launch shape (host arithmetic)."""
    return torch.ops._exl3_C.linear_prefill_applies(
        rows, padded_in, padded_out, bits_x2, codebook
    )


def set_prefill_schedule(mode: int) -> None:
    """The operator's schedule for this process: 0 planned, 1 whole tiles."""
    torch.ops._exl3_C.set_prefill_schedule(mode)


@functools.cache
def takes_launch(
    rows: int,
    padded_in: int,
    padded_out: int,
    bits_x2: int,
    codebook: int,
    splitk: bool,
    sms: int,
) -> bool:
    """Whether a prefill-step launch of this shape goes to linear_out_prefill:
    the operator covers it and, with split-K on, split-K does not split it."""
    from vllm.sm80.kernels.exl3.ops import linear_splitk_splits

    if splitk and linear_splitk_splits(rows, padded_out, sms, bits_x2, codebook) == 2:
        return False
    return linear_prefill_applies(rows, padded_in, padded_out, bits_x2, codebook)


def linear_out_prefill(
    x,
    trellis,
    suh,
    svh,
    bits_x2: int,
    codebook: int,
    logical_in: int,
    logical_out: int,
    out,
    workspace,
) -> None:
    """linear_out's contract for the launches linear_prefill_applies covers."""
    torch.ops._exl3_C.linear_out_prefill(
        x,
        trellis,
        suh,
        svh,
        bits_x2,
        codebook,
        logical_in,
        logical_out,
        out,
        workspace,
    )


def _launch(
    default,
    splitk: bool,
    x,
    trellis,
    suh,
    svh,
    bits_x2: int,
    codebook: int,
    logical_in: int,
    logical_out: int,
    out,
    workspace,
) -> None:
    padded_in, padded_out = trellis.shape[0] * 16, trellis.shape[1] * 16
    operator = default
    if trellis.data_ptr() % 16 == 0 and takes_launch(
        x.shape[0], padded_in, padded_out, bits_x2, codebook, splitk, _sms
    ):
        operator = linear_out_prefill
    operator(
        x, trellis, suh, svh, bits_x2, codebook, logical_in, logical_out, out, workspace
    )


def route(default, rows: int, splitk: bool):
    """The operator _linear calls for each row chunk of a call of `rows` rows:
    `default` itself outside prefill steps, during graph capture and for calls
    of at most MAX_SHIPPED_ROWS rows; otherwise a per-launch choice between
    linear_out_prefill and `default`."""
    if (
        rows <= MAX_SHIPPED_ROWS
        or torch.cuda.is_current_stream_capturing()
        or not step_phase.prefill_step(_max_decode_query_len)
    ):
        return default
    return functools.partial(_launch, default, splitk)
