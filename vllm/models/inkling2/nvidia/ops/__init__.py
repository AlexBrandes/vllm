# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools


@functools.cache
def use_sm100_kernels() -> bool:
    """Whether to run the Blackwell kernels (tml_fa4, tcgen05 attention,
    TRT-LLM-gen expert GEMMs, e2m1 conversions). Hopper uses Humming experts;
    SM12x uses Triton experts and does not support tcgen05."""
    from vllm.platforms import current_platform

    return current_platform.is_device_capability_family(100)


@functools.cache
def use_sm12x_path() -> bool:
    """Whether to take the SM12x code path: Triton experts, Triton relative
    attention, no TMA, no native conv/mix kernels. HyperInkling also routes
    Ampere (SM8x) here; what SM8x lacks beyond that (FP8/e2m1 hardware) is
    checked separately with `is_sm8x()`."""
    from vllm.platforms import current_platform

    return current_platform.is_device_capability_family(120) or is_sm8x()


@functools.cache
def is_sm8x() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_device_capability_family(80)


# Programmatic dependent launch (griddepcontrol) is SM90+. `PDL` gates the
# launches, and the two helpers below compile the in-kernel waits away on SM8x.
PDL: bool = not is_sm8x()

from vllm.triton_utils import tl, triton  # noqa: E402

_PDL = tl.constexpr(PDL)


@triton.jit
def gdc_wait():
    if _PDL:
        tl.extra.cuda.gdc_wait()


@triton.jit
def gdc_launch_dependents():
    if _PDL:
        tl.extra.cuda.gdc_launch_dependents()
