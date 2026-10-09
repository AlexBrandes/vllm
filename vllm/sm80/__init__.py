# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Code the CMP 170HX (sm_80) fork adds to vLLM and that more than one model
can reach.

- ``kernels``: kernels dispatched from shared vLLM ops, one subpackage per
  family (attention, indexer, kda, marlin_moe, mhc, moe, thin_gemm).
- ``attention``: the Triton sparse-MLA attention backend.
- ``distributed``: the host-staged and the flags-in-data custom all-reduce,
  and the opt-in PCIe peer-to-peer topology leg (``pcie_p2p``).
- ``fla``: autotune pins for the vendored flash-linear-attention kernels.
- ``spec_decode``: adaptive draft depth and the PP drafter fc fold.
- ``worker``: the PP draft tail and packed hop, and the memory attribution
  and state-index diagnostics.
- ``envs``: the fork's environment variables; ``vllm/envs.py`` registers
  them at the end of its own import.
- ``plugin`` and ``guard``: the ``vllm.general_plugins`` entry point
  (distribution in ``plugins/sm80``) and the startup check that it loaded.
- ``platform``: the fork's CUDA platform, selected by the
  ``vllm.platform_plugins`` entry point of the same distribution; the guard
  also checks that it is the platform in use on CUDA.

Kernels only one model calls live under ``vllm/models/<model>/nvidia/ops``.
Importing this package does no work: the upstream files that use a module
here import that module directly.
"""
