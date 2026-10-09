# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_80 Marlin MoE paths called from the shared Marlin experts.

  ``extension``    loads and validates the optional compiled library
                   ``vllm._ampere_marlin_C`` (sources in
                   csrc/libtorch_stable/moe/ampere_marlin)
  ``decode``       the compiled Marlin decode (``VLLM_AMPERE_MARLIN_DECODE_CUDA``)
  ``prefill``      the split-block Marlin prefill (``VLLM_AMPERE_MARLIN_PREFILL_WIDE``,
                   ``VLLM_AMPERE_MARLIN_PREFILL_NARROW``) and its compiled PP
                   tiles (``VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED``)
  ``split_align``  the block lists ``prefill`` launches over
"""
