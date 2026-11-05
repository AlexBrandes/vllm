# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Code the CMP 170HX (sm_80) fork adds to vLLM.

- ``kernels``: kernels dispatched from shared vLLM ops, one subpackage per
  family (attention, indexer, kda, marlin_moe, mhc, moe, thin_gemm), and
  ``exl3``, the EXL3 quantization method and load format.
- ``attention``: the Triton sparse-MLA attention backend.
- ``distributed``: the host-staged and the flags-in-data custom all-reduce,
  and the opt-in PCIe peer-to-peer topology leg (``pcie_p2p``).
- ``fla``: autotune pins for the vendored flash-linear-attention kernels.
- ``spec_decode``: adaptive draft depth and the PP drafter fc fold.
- ``worker``: the PP handler and packed hop, and the memory attribution
  and state-index diagnostics.
- ``kv_planning``: KV cache planning rules: packed-group refit for MLA +
  Mamba hybrids (DFlash drafter fit, PP-aware state groups), speculative
  scratch in KV capacity, and the scheduler's eagle-group flags.
- ``failed_inputs``: fails only the requests whose model inputs a worker
  could not produce (an EXL3 PLE row that storage failed to read).
- ``envs``: the fork's environment variables; the plugin's entry points
  register them (``plugin.register_fork_env_vars``).
- ``sched``: the fork scheduler (fair chunked prefill, PP decode spreading,
  homogeneous PP steps, adaptive draft depth) and ``sched.prefill_budget``
  (VLLM_MIXED_PREFILL_CHUNK), which the vLLM branch's scheduler hooks call.
- ``ple``: Qwen4Exp PLE row reads the vLLM branch's model hooks call
  (``disk_reader``: VLLM_PLE_DISK_READ_THREADS).
- ``plugin`` and ``guard``: the ``vllm.general_plugins`` entry point
  ``cmp170hx``, which registers the GLM-5.3 model code and ``kernels.exl3``,
  and the startup check that it loaded.
- ``platform``: the fork's CUDA platform, selected by the
  ``vllm.platform_plugins`` entry point of the same distribution; the guard
  also checks that it is the platform in use on CUDA.

Kernels only one model calls live under ``vllm_cmp170hx/models/<model>/ops``.
Importing this package does no work beyond placing its loggers: the
upstream files that use a module here import that module directly, and
``plugin`` imports the EXL3 modules it registers.
"""

import logging as _logging

# vLLM configures only the "vllm" logger (handler, format, level). The
# package's loggers ("vllm_cmp170hx.*", from vllm.logger.init_logger(__name__))
# get it as their ancestor, so their records reach vLLM's handler at vLLM's
# level.
_logging.getLogger(__name__).parent = _logging.getLogger("vllm")
