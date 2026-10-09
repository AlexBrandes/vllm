# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernels tuned for NVIDIA sm_80 parts (GA100 class, e.g. the CMP 170HX).

Code dispatched from shared vLLM ops with shape gates any sm_80 model could
hit lives here, one subpackage per kernel family. Kernels only one model
calls live in that model's ``vllm/models/<model>/nvidia/ops/`` instead.

Importing this package does no work: every family is opt-in behind its own
switch, and its kernel modules are imported only when that switch is on.
"""
