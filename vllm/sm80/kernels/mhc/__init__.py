# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_80 mHC kernels called from the shared TileLang mHC ops.

  ``decode``             the fused mHC post + pre sublayer boundary at decode
                         sizes, 90 calls/step (``VLLM_GLM5_DECODE_MHC_V2``)
  ``prenorm``            the pre-norm projection GEMM at prefill sizes
                         (``VLLM_GLM5_PREFILL_KERNELS``)
  ``prenorm_workspace``  the prefill scratch owner ``prenorm`` uses
  ``warmup``             compiles ``decode`` before CUDA graph capture

The gates below are evaluated on the host at every call site. The decode gate
needs ``VLLM_AMPERE_DECODE_KERNELS`` and ``VLLM_GLM5_DECODE_MHC_V2``, and with
either unset no kernel module is imported and the call site keeps the TileLang
path. The prefill pre-norm gates are called from
``vllm/model_executor/kernels/mhc/tilelang.py::_hc_prenorm_gemm_outputs``.
"""

import torch

from vllm.sm80.kernels._platform import _is_sm80

__all__ = [
    "ampere_prenorm_offsets_fit",
    "use_ampere_mhc_decode_v3",
    "use_ampere_prefill_prenorm",
]

# Sentinel for "the caller has a norm_weight tensor". The mHC gate only needs
# to know whether it is None, and the artifact-link script calls the gate with
# three positional arguments, so the parameter cannot be required.
_PRESENT = object()


def use_ampere_mhc_decode_v3(
    num_tokens: int,
    hc_mult: int,
    hidden_size: int,
    *,
    norm_weight: object | None = _PRESENT,
) -> bool:
    """Gate for vllm/sm80/kernels/mhc/decode.py::mhc_fused_post_pre.

    When open it replaces TileLang for 1 <= M <=
    ``VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS`` (32). Measured us/call for v2, the
    kernel pair v3 refines, graph replay, one CMP 170HX, against the faster of
    the earlier v1 kernel and TileLang in the same run (best of 3):

        M     v1      TileLang   v2     vs faster   vs the path then in service
        4    12.71    12.94     7.76     1.64x       1.64x (v1)
        8    15.22    16.22     8.27     1.84x       1.84x (v1)
       16    20.28    18.15     8.91     2.04x       2.04x (TileLang)
       32    22.66    25.66    10.66     2.13x       2.41x (TileLang)

    Only the hc == 4, hidden % 1024 == 0 path (two Gluon kernels) was measured
    and validated, so the gate admits nothing else. Resolved on the host, per
    call, outside any graph; ``norm_weight`` is only inspected for None-ness
    (the kernel fuses the input RMSNorm, so a call without it falls through to
    TileLang).
    """
    from vllm import envs

    if not envs.VLLM_AMPERE_DECODE_KERNELS or not envs.VLLM_GLM5_DECODE_MHC_V2:
        return False
    if num_tokens < 1 or num_tokens > envs.VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS:
        return False
    if norm_weight is None:
        return False
    if hc_mult != 4 or hidden_size <= 0 or hidden_size % 1024:
        return False
    return _is_sm80()


def use_ampere_prefill_prenorm(num_tokens: int) -> bool:
    """Gate for vllm/sm80/kernels/mhc/prenorm.py.

    Resolved on the host, per call, outside any graph. At decode sizes the
    upstream TileLang kernel is the one that was tuned, so the threshold keeps
    every captured CUDA graph on it.

    COVERAGE. This gates `_hc_prenorm_gemm_outputs`, which is the choke point
    for `mhc_pre_tilelang`, `mhc_pre_broadcast_tilelang` and
    `mhc_fused_post_pre_tilelang` -- the last of which is what GLM-5.3-Flash
    actually dispatches (confirmed in the trace: `vllm::mhc_fused_post_pre_
    tilelang`, 90 calls per 1152-token chunk). The two *delayed* variants,
    `mhc_pre_delayed_tilelang` and `mhc_fused_post_pre_delayed_tilelang`, call
    `_HC_PRENORM_GEMM_TILELANG_KERNEL` inline and are deliberately left on the
    upstream kernel: this deployment does not take them. If that ever changes,
    extend the hook there too rather than assuming it is already covered.
    """
    from vllm import envs
    from vllm.platforms import current_platform

    if not envs.VLLM_GLM5_PREFILL_KERNELS:
        return False
    if num_tokens < envs.VLLM_GLM5_PREFILL_MIN_TOKENS:
        return False
    return current_platform.is_cuda() and current_platform.is_device_capability(80)


def ampere_prenorm_offsets_fit(x: torch.Tensor, fn: torch.Tensor) -> bool:
    """The sm_80 kernel only where its int32 offsets hold; else TileLang."""
    from vllm.sm80.kernels.mhc.prenorm import offsets_fit

    if offsets_fit(x, fn):
        return True
    from vllm.logger import init_logger

    init_logger(__name__).warning_once(
        "sm_80 prefill mHC pre-norm: a step spanning 2**31 or more elements "
        "would overflow the kernel's int32 offsets; such steps use the "
        "TileLang kernel instead."
    )
    return False
