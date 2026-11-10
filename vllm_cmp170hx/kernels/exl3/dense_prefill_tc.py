# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor-core dense EXL3 linear: prefill steps (VLLM_EXL3_DENSE_PREFILL_TC)
and the small-row GLM QKV decode schedule (VLLM_EXL3_DENSE_DECODE_TC).

The operator ``linear_out_prefill`` (csrc/exl3/dense_prefill_tc.cu) computes
what ``linear_out`` computes, on the tensor cores: exact integer-grid main
products, FP32 sums in k order, and, for the last part-empty wave of tiles,
a k split whose partial tiles the last arriving block adds in k order
(arithmetic in csrc/exl3/dense_prefill_tc.h). Its results differ from
``linear_out``'s in the last bits, and a row's bits depend on the row count
of its launch and the row's position in it (which tiles the schedule
splits).

``ops._linear`` asks ``route`` for the operator of each call. A row chunk of
the call takes ``linear_out_prefill`` only when all of these hold:
- the flag is set and ``resolve`` found a covered model/layout on sm_80 with
  74 multiprocessors: Qwen TP2/PP1 with its nonparallel drafters, or the
  GLM5Next TP1/PP2 DFlash dense families (draft depths 3, 5 or 7);
- the running step is prefill by scheduler metadata, never merely its row
  count, and the stream is not capturing a CUDA graph. GLM requires explicit
  homogeneous prefill counts; parallel verification and mixed batches fall back;
- the matrix belongs to the resolved layout's VALIDATED_MATRICES set
  (padded_in, padded_out, bits_x2; codebook 2), and the launch has more than
  64 rows, at least 16 output tiles of 128 x 128 and a 16-byte aligned trellis;
- with VLLM_EXL3_DENSE_SPLITK on, ``linear_out_splitk`` would not split the
  launch's K: every launch split-K splits keeps split-K.
Every other launch runs the operator ``_linear`` picked before (linear_out or
linear_out_splitk), unchanged. Row chunks, the workspace and its size are
``_linear``'s; the operator places its stream-K partial tiles and their
counters inside ``linear_workspace_size`` of the launch and zeroes the
counters on every call.

A second, independent switch (VLLM_EXL3_DENSE_DECODE_TC) opens the small-row
decode schedule: the exact GLM PP2 DFlash layout (draft depths 3, 5 or 7), its
measured QKV family
(padded_in 4096, padded_out 24576, bits_x2 12, codebook 2), and decode-step
launches of 1..8 rows -- one MMA row group, whole tiles, no stream-K -- on the
same codec and numerical policy, reusing the ``linear_out_prefill`` operator and
``linear_workspace_size`` unchanged. Prefill steps, larger calls and every other
matrix keep their kernels (see ``decode_closed_reason``, ``route`` and
``csrc/exl3/dense_prefill_tc.h``).
"""

import functools
import os

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm_cmp170hx.kernels.exl3 import step_phase

logger = init_logger(__name__)

# Launches of at most this many rows keep today's kernels (the decode bound).
MAX_SHIPPED_ROWS = 64
# The multiprocessor count the schedule was validated at.
VALIDATED_SMS = 74
# Per-layout matrix sets, using the Qwen PP2 port's explicit set selection.
# TP2 retains its original Qwen shapes. GLM PP2 covers its 6-bit dense
# families: every text matrix of this artifact whose padded_out is at least
# 16 tiles of 128 and whose padded_in is trellis-aligned. The LM head, the
# narrow projections (q_a 1536, kv_a 512) and routed experts stay out.
VALIDATED_MATRICES = {
    (2, 1): frozenset(
        {
            (2560, 256, 14),  # attention k/v
            (2560, 640, 14),  # shared expert gate/up
            (2560, 640, 10),  # sparse-attention indexer q/k
            (2560, 1280, 12),  # drafter embedding / hidden-state projections
            (640, 2560, 14),  # shared expert down
            (3072, 2560, 14),  # output projections
            (2560, 3072, 14),  # linear-attention z
            (2560, 5120, 14),  # linear-attention qkv
            (2560, 6144, 14),  # attention q (with output gate)
        }
    ),
    (1, 2): frozenset(
        {
            (4096, 24576, 12),  # attention qkv (the measured family)
            (1536, 16384, 12),  # attention q_b
            (8192, 4096, 12),  # attention out, 8K-K layers
            (16384, 4096, 12),  # attention out, 16K-K layers
            (4096, 12288, 12),  # dense-layer mlp gate/up
            (12288, 4096, 12),  # dense-layer mlp down
            (4096, 2048, 12),  # shared-expert gate/up
            (2048, 4096, 12),  # shared-expert down
            (1536, 4096, 12),  # sparse-attention indexer wq_b
        }
    ),
}
VALIDATED_CODEBOOK = 2
# Speculative methods whose decode steps step_phase reads as decode steps.
COVERED_DRAFTERS = ("mtp", "eagle", "eagle3")
# GLM DFlash draft depths the dense tensor-core gates admit: an odd depth d gives a
# 1 + d-token verified decode step (4, 6, or 8 rows for a single request), all within
# the small-row schedule's 1..8-row native bound. Any other depth -- including a depth
# whose verify would exceed 8 rows -- keeps today's kernels. Opt-in eligibility only;
# each retained depth is qualified separately before it is measured.
GLM_DFLASH_DEPTHS = (3, 5, 7)
# VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE -> set_prefill_schedule mode.
SCHEDULES = {"planned": 0, "whole": 1}
# The library operators this path needs (csrc/exl3/dense_prefill_tc.cu).
OPERATORS = ("linear_out_prefill", "linear_prefill_applies", "set_prefill_schedule")
# A warm-up launch per covered format: (rows, padded_in, padded_out). Its plan
# runs both kernels (whole tiles and the k split) at 74 multiprocessors. The
# explicit-prefill GLM layout warms every covered (padded_in, padded_out,
# bits_x2) instead: its families are shape-specialized kernel variants.
WARMUP_LAUNCH = (482, 2560, 2560)

# Whole-K, one MMA row group; the codec and numerical policy match prefill.
MAX_DECODE_ROWS = 8
# Shipped default: only GLM QKV is covered. Mixed and unknown phases retain
# their kernels.
DECODE_VALIDATED_MATRICES = {
    (1, 2): frozenset(
        {
            (4096, 24576, 12),  # attention qkv (the measured family)
        }
    ),
}
# Lab widening (VLLM_EXL3_DENSE_DECODE_TC_FAMILIES=all, default "qkv"): every
# GLM PP2 dense 6-bit text family of the prefill set plus the narrow MLA
# projections and the LM head. The decode GEMM is shape-generic (one block per
# 128 output columns, whole k, the same exact-grid arithmetic as QKV); shapes
# that do not occur in the artifact are simply never matched.
DECODE_LAB_ALL_MATRICES = DECODE_VALIDATED_MATRICES[(1, 2)] | frozenset(
    {
        (1536, 16384, 12),  # attention q_b
        (8192, 4096, 12),  # attention out, 8K-K layers
        (16384, 4096, 12),  # attention out, 16K-K layers
        (4096, 12288, 12),  # dense-layer mlp gate/up
        (12288, 4096, 12),  # dense-layer mlp down
        (4096, 2048, 12),  # shared-expert gate/up
        (2048, 4096, 12),  # shared-expert down
        (1536, 4096, 12),  # sparse-attention indexer wq_b
        (4096, 1536, 12),  # attention q_a
        (4096, 512, 12),  # attention kv_a
        (4096, 154880, 12),  # LM head (shared with the DFlash draft)
    }
)
# The prefill operator also dispatches the enabled small-row schedule.
DECODE_OPERATORS = (
    "linear_out_prefill",
    "linear_prefill_applies",
    "set_prefill_decode",
)
DECODE_CODEBOOK = VALIDATED_CODEBOOK
# Warm the largest covered row count before capture.
DECODE_WARMUP_ROWS = MAX_DECODE_ROWS

# Fixed before capture. ops._linear consults `active`; route chooses the phase.
active = False
_prefill_active = False
decode_active = False
_max_decode_query_len = 0
_sms = 0
_fake_registered = False
_served_matrices = VALIDATED_MATRICES[(2, 1)]
_require_explicit_prefill = False
_prefill_dispatch_logged = False
_decode_served = frozenset()
_decode_dispatch_logged: set[tuple[int, int]] = set()


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
    parallel = vllm_config.parallel_config
    layout = (parallel.tensor_parallel_size, parallel.pipeline_parallel_size)
    if layout not in VALIDATED_MATRICES:
        return f"unvalidated (tensor, pipeline) parallel size {layout}"
    speculative = vllm_config.speculative_config
    if layout == (1, 2):
        model = getattr(vllm_config, "model_config", None)
        text = getattr(model, "hf_text_config", None)
        if (
            getattr(text, "model_type", None) != "glm5_next_text"
            or getattr(text, "hidden_size", None) != 4096
            or getattr(model, "dtype", None) != torch.bfloat16
        ):
            return (
                "PP2 prefill is limited to the BF16-activation GLM5Next dense families"
            )
        if (
            speculative is None
            or speculative.method != "dflash"
            or vllm_config.num_speculative_tokens not in GLM_DFLASH_DEPTHS
        ):
            return (
                "GLM5Next prefill is limited to the checked DFlash depths "
                f"{GLM_DFLASH_DEPTHS}"
            )
        return None
    if speculative is not None and (
        speculative.method not in COVERED_DRAFTERS
        or getattr(speculative, "parallel_drafting", False)
    ):
        return (
            f"speculative method {speculative.method!r} (parallel drafting "
            f"{bool(getattr(speculative, 'parallel_drafting', False))}): its "
            "decode steps can schedule more than 1 + num_speculative_tokens "
            "tokens per request"
        )
    return None


def decode_closed_reason(
    capability: tuple[int, int] | None, sms: int | None, vllm_config
) -> str | None:
    """Why the small-row decode schedule must not serve this process, or None.

    The decode schedule serves the GLM PP2 layout only, and its gate is the
    prefill GLM gate -- a BF16-activation GLM5Next (tensor, pipeline) (1, 2)
    DFlash process (draft depths 3, 5 or 7) on sm_80 with 74 multiprocessors --
    pinned to (1, 2): the Qwen (2, 1) layout the prefill path also serves is
    not a decode-schedule layout."""
    if vllm_config is None:
        return "no vLLM config is set, so the layout is unknown"
    parallel = vllm_config.parallel_config
    layout = (parallel.tensor_parallel_size, parallel.pipeline_parallel_size)
    if layout != (1, 2):
        return (
            f"the small-row decode schedule serves the GLM (tensor, pipeline) "
            f"layout (1, 2), not {layout}"
        )
    return closed_reason(capability, sms, vllm_config)


def _warmup(sms: int) -> None:
    """Run linear_out_prefill once per covered format before any capture, so
    no request's first prefill loads the kernels or sets their attributes.
    The temporary buffers are freed again. ops.ensure_extension runs this
    before it marks the library loaded, so the workspace size comes from the
    library itself: ops.linear_workspace_size would call ensure_extension."""
    device = torch.device("cuda", torch.cuda.current_device())
    if _require_explicit_prefill:
        # One launch per covered GLM family: every shape is its own variant.
        groups = [
            ((292, padded_in, padded_out), (bits_x2,))
            for padded_in, padded_out, bits_x2 in sorted(_served_matrices)
        ]
    else:
        rows, padded_in, padded_out = WARMUP_LAUNCH
        groups = [
            (
                (rows, padded_in, padded_out),
                tuple(sorted({b for _, _, b in _served_matrices})),
            )
        ]
    for (rows, padded_in, padded_out), formats in groups:
        x = torch.ones((rows, padded_in), dtype=torch.bfloat16, device=device)
        out = torch.empty((rows, padded_out), dtype=torch.bfloat16, device=device)
        workspace = torch.empty(
            torch.ops._exl3_C.linear_workspace_size(rows, padded_in, padded_out),
            dtype=torch.float32,
            device=device,
        )
        suh = torch.ones(padded_in, dtype=torch.float16, device=device)
        svh = torch.ones(padded_out, dtype=torch.float16, device=device)
        for bits_x2 in formats:
            trellis = torch.zeros(
                (padded_in // 16, padded_out // 16, 8 * bits_x2),
                dtype=torch.int16,
                device=device,
            )
            linear_out_prefill(
                x,
                trellis,
                suh,
                svh,
                bits_x2,
                VALIDATED_CODEBOOK,
                padded_in,
                padded_out,
                out,
                workspace,
            )
            del trellis
        del x, out, workspace, suh, svh
    torch.cuda.synchronize(device)


def _warmup_decode(sms: int) -> None:
    """Run decode_gemm once per served decode format before any capture, so no
    request's first decode step loads the kernel or sets its attributes. The
    native decode schedule must already be on (set_prefill_decode) for these
    1..MAX_DECODE_ROWS launches to be covered. ops.ensure_extension runs this
    before it marks the library loaded, so the workspace size comes from the
    library itself: ops.linear_workspace_size would call ensure_extension."""
    device = torch.device("cuda", torch.cuda.current_device())
    rows = DECODE_WARMUP_ROWS
    for padded_in, padded_out, bits_x2 in sorted(_decode_served):
        x = torch.ones((rows, padded_in), dtype=torch.bfloat16, device=device)
        out = torch.empty((rows, padded_out), dtype=torch.bfloat16, device=device)
        workspace = torch.empty(
            torch.ops._exl3_C.linear_workspace_size(rows, padded_in, padded_out),
            dtype=torch.float32,
            device=device,
        )
        suh = torch.ones(padded_in, dtype=torch.float16, device=device)
        svh = torch.ones(padded_out, dtype=torch.float16, device=device)
        trellis = torch.zeros(
            (padded_in // 16, padded_out // 16, 8 * bits_x2),
            dtype=torch.int16,
            device=device,
        )
        linear_out_prefill(
            x,
            trellis,
            suh,
            svh,
            bits_x2,
            DECODE_CODEBOOK,
            padded_in,
            padded_out,
            out,
            workspace,
        )
        del x, out, workspace, suh, svh, trellis
    torch.cuda.synchronize(device)


def resolve() -> None:
    """Both dense tensor-core switches for this process, each with its log line.

    VLLM_EXL3_DENSE_PREFILL_TC (prefill steps) and VLLM_EXL3_DENSE_DECODE_TC
    (the small-row decode schedule) are independent gates: each resolves its
    own configuration and warms its own kernels, and either may be set alone.
    `active` is set once, true when either path opened, so ops._linear routes
    every launch through route(). Called once by ops.ensure_extension, after
    the library loaded and before ensure_extension returns: nothing here may
    call an ops function that calls ensure_extension."""
    global active
    _resolve_prefill_tc()
    _resolve_decode_tc()
    active = _prefill_active or decode_active


def _resolve_prefill_tc() -> None:
    """VLLM_EXL3_DENSE_PREFILL_TC for this process, with its log line."""
    global _prefill_active, _max_decode_query_len, _sms, _fake_registered
    global _served_matrices, _require_explicit_prefill
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
    parallel = vllm_config.parallel_config
    layout = (parallel.tensor_parallel_size, parallel.pipeline_parallel_size)
    _served_matrices = VALIDATED_MATRICES[layout]
    _require_explicit_prefill = layout == (1, 2)
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
        from vllm_cmp170hx.kernels.exl3.ops import _linear_fake

        torch.library.register_fake("_exl3_C::linear_out_prefill")(_linear_fake)
        _fake_registered = True
    set_prefill_schedule(SCHEDULES[schedule])
    _warmup(sms)
    _max_decode_query_len = step_phase.max_decode_query_len(vllm_config)
    _sms = sms
    logger.info_once(
        "EXL3 dense prefill tensor-core path active (VLLM_EXL3_DENSE_PREFILL_TC): "
        "TP%d PP%d, %d checked matrices, explicit-prefill metadata=%s; "
        "launches above %d rows use schedule %s. Decode, mixed/capture "
        "steps and uncovered matrices keep the existing kernels.",
        layout[0],
        layout[1],
        len(_served_matrices),
        _require_explicit_prefill,
        MAX_SHIPPED_ROWS,
        schedule,
        scope="process",
    )
    _prefill_active = True
    from vllm_cmp170hx.kernels.exl3 import dense_prefill_once

    if dense_prefill_once.enabled():
        dense_prefill_once.preallocate(torch.device("cuda", torch.cuda.current_device()))


def once_takes(x, trellis, bits_x2: int, codebook: int, logical_out: int) -> bool:
    """Lab: whether this whole _linear call takes the decode-once prefill path
    (dense_prefill_once; VLLM_EXL3_DENSE_PREFILL_ONCE). Same phase gate and
    matrix set as the tensor-core prefill operator; the call is not split into
    _linear's row chunks."""
    from vllm_cmp170hx.kernels.exl3 import dense_prefill_once

    if not _prefill_active or not dense_prefill_once.enabled():
        return False
    padded_in, padded_out = trellis.shape[0] * 16, trellis.shape[1] * 16
    if (
        codebook != VALIDATED_CODEBOOK
        or (padded_in, padded_out, bits_x2) not in _served_matrices
        or not dense_prefill_once.takes(x.shape[0], x.shape[1], trellis, logical_out)
        or torch.cuda.is_current_stream_capturing()
    ):
        return False
    return step_phase.prefill_step(
        _max_decode_query_len, require_explicit=_require_explicit_prefill
    )


def _resolve_decode_tc() -> None:
    """VLLM_EXL3_DENSE_DECODE_TC for this process, with its log line.

    Opens the small-row tensor-core decode schedule: the exact GLM PP2 DFlash
    layout, the QKV family, decode steps of 1..MAX_DECODE_ROWS rows. Prefill
    steps, larger calls and uncovered matrices keep their kernels."""
    global decode_active, _max_decode_query_len, _sms, _fake_registered
    global _decode_served, _require_explicit_prefill
    if not envs.VLLM_EXL3_DENSE_DECODE_TC:
        return
    from vllm.config import get_current_vllm_config_or_none

    capability, sms = None, None
    if torch.cuda.is_available():
        capability = tuple(torch.cuda.get_device_capability())
        sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    vllm_config = get_current_vllm_config_or_none()
    reason = decode_closed_reason(capability, sms, vllm_config)
    if reason is not None:
        logger.info_once(
            "VLLM_EXL3_DENSE_DECODE_TC is set but the gate is closed: %s; every "
            "dense EXL3 decode launch keeps today's kernels.",
            reason,
            scope="process",
        )
        return
    for name in DECODE_OPERATORS:
        try:
            torch._C._dispatch_find_schema_or_throw(f"_exl3_C::{name}", "")
        except RuntimeError as exc:
            raise RuntimeError(
                f"VLLM_EXL3_DENSE_DECODE_TC needs _exl3_C::{name}: rebuild the "
                "EXL3 extension from this tree (python "
                "csrc/exl3/build_standalone.py --out vllm), or unset the flag. "
                f"Detail: {exc}"
            ) from exc
    if not _fake_registered:
        from vllm_cmp170hx.kernels.exl3.ops import _linear_fake

        torch.library.register_fake("_exl3_C::linear_out_prefill")(_linear_fake)
        _fake_registered = True
    families = os.environ.get("VLLM_EXL3_DENSE_DECODE_TC_FAMILIES", "qkv")
    if families not in ("qkv", "all"):
        raise ValueError(
            "VLLM_EXL3_DENSE_DECODE_TC_FAMILIES must be 'qkv' or 'all', "
            f"not {families!r}"
        )
    _decode_served = (
        DECODE_LAB_ALL_MATRICES
        if families == "all"
        else DECODE_VALIDATED_MATRICES[(1, 2)]
    )
    # Enable the native schedule before the warm-up: the 1..8-row warm-up
    # launches are covered only while it is on. Decode is whole-tile, so it
    # ignores set_prefill_schedule (the prefill path's stream-K diagnostic).
    set_prefill_decode(True)
    _warmup_decode(sms)
    _max_decode_query_len = step_phase.max_decode_query_len(vllm_config)
    _sms = sms
    _require_explicit_prefill = True  # the GLM PP2 layout's decode-phase signal
    logger.info_once(
        "EXL3 dense small-row tensor-core decode path active "
        "(VLLM_EXL3_DENSE_DECODE_TC): TP1 PP2, %d checked matrix families; "
        "decode-step launches of 1..%d rows use the tensor-core decode GEMM. "
        "Prefill, mixed and larger calls and uncovered matrices keep the "
        "existing kernels.",
        len(_decode_served),
        MAX_DECODE_ROWS,
        scope="process",
    )
    decode_active = True


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


def set_decode_splitk(enable: bool) -> None:
    """Lab: turn the ordered split-K of narrow small-row decode launches on or
    off for this process (VLLM_EXL3_DENSE_DECODE_SPLITK sets the initial state).
    Splits engage only when the caller's workspace tensor has room for the
    FP32 partials past the activation scratch."""
    torch.ops._exl3_C.set_decode_splitk(bool(enable))


def set_prefill_decode(enable: bool) -> None:
    """Turn the native small-row decode schedule on or off for this process.

    While off, linear_prefill_applies refuses every launch of at most
    MAX_SHIPPED_ROWS rows and linear_out_prefill is the prefill operator only."""
    torch.ops._exl3_C.set_prefill_decode(bool(enable))


@functools.cache
def takes_launch(
    rows: int,
    padded_in: int,
    padded_out: int,
    bits_x2: int,
    codebook: int,
    splitk: bool,
    sms: int,
    validated: frozenset[tuple[int, int, int]],
) -> bool:
    """Whether a prefill-step launch of this shape goes to linear_out_prefill:
    a validated matrix the operator covers at this row count and, with
    split-K on, split-K does not split it."""
    from vllm_cmp170hx.kernels.exl3.ops import linear_splitk_splits

    # The shared native predicate also admits small rows under the decode opt-in.
    if (
        rows <= MAX_SHIPPED_ROWS
        or codebook != VALIDATED_CODEBOOK
        or (padded_in, padded_out, bits_x2) not in validated
    ):
        return False
    if splitk and linear_splitk_splits(rows, padded_out, sms, bits_x2, codebook) == 2:
        return False
    return linear_prefill_applies(rows, padded_in, padded_out, bits_x2, codebook)


@functools.cache
def takes_decode_launch(
    rows: int,
    padded_in: int,
    padded_out: int,
    bits_x2: int,
    codebook: int,
    splitk: bool,
    sms: int,
    validated: frozenset[tuple[int, int, int]],
) -> bool:
    """Whether a decode-step launch of this shape goes to the small-row
    tensor-core decode GEMM: a validated decode matrix at 1..MAX_DECODE_ROWS
    rows the operator covers and, with split-K on, split-K does not split."""
    from vllm_cmp170hx.kernels.exl3.ops import linear_splitk_splits

    if (
        not 1 <= rows <= MAX_DECODE_ROWS
        or codebook != DECODE_CODEBOOK
        or (padded_in, padded_out, bits_x2) not in validated
    ):
        return False
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
    global _prefill_dispatch_logged
    operator = default
    if trellis.data_ptr() % 16 == 0 and takes_launch(
        x.shape[0],
        padded_in,
        padded_out,
        bits_x2,
        codebook,
        splitk,
        _sms,
        _served_matrices,
    ):
        operator = linear_out_prefill
    operator(
        x, trellis, suh, svh, bits_x2, codebook, logical_in, logical_out, out, workspace
    )

    if operator is linear_out_prefill and not _prefill_dispatch_logged:
        logger.info(
            "EXL3 dense tensor-core prefill dispatched: rows=%d, in=%d, out=%d",
            x.shape[0],
            padded_in,
            padded_out,
        )
        _prefill_dispatch_logged = True


def _launch_decode(
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
    if trellis.data_ptr() % 16 == 0 and takes_decode_launch(
        x.shape[0],
        padded_in,
        padded_out,
        bits_x2,
        codebook,
        splitk,
        _sms,
        _decode_served,
    ):
        operator = linear_out_prefill
    operator(
        x, trellis, suh, svh, bits_x2, codebook, logical_in, logical_out, out, workspace
    )

    # Engagement evidence: one line per served shape (not just the first), so a
    # screen can show which families actually took the decode GEMM.
    if operator is linear_out_prefill and (padded_in, padded_out) not in (
        _decode_dispatch_logged
    ):
        logger.info(
            "EXL3 dense tensor-core decode dispatched: rows=%d, in=%d, out=%d",
            x.shape[0],
            padded_in,
            padded_out,
        )
        _decode_dispatch_logged.add((padded_in, padded_out))


def route(default, rows: int, splitk: bool):
    """The operator _linear calls for each row chunk of a call of `rows` rows.

    Decode: a launch of 1..MAX_DECODE_ROWS rows with explicit pure-decode
    metadata takes the small-row schedule for a covered matrix, including full
    decode graph capture. Unknown and mixed phases keep `default`.
    Prefill (unchanged): a launch above MAX_SHIPPED_ROWS rows in a prefill step,
    never during capture, is chosen per launch between linear_out_prefill and
    `default`.
    Every other launch -- both switches closed, other row counts, phases or
    matrices -- runs `default` itself."""
    if decode_active and 1 <= rows <= MAX_DECODE_ROWS:
        if not step_phase.decode_step():
            return default
        return functools.partial(_launch_decode, default, splitk)
    if _prefill_active:
        if (
            rows <= MAX_SHIPPED_ROWS
            or torch.cuda.is_current_stream_capturing()
            or not step_phase.prefill_step(
                _max_decode_query_len, require_explicit=_require_explicit_prefill
            )
        ):
            return default
        return functools.partial(_launch, default, splitk)
    return default
