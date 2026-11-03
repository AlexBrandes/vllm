# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fork's vLLM general plugin (``vllm.general_plugins`` entry point
``cmp170hx``).

vLLM calls :func:`register` in every process that loads general plugins: the
API server, the engine core and each worker. It may be called more than once
in one process and must stay idempotent. It records that it ran, so
``vllm_cmp170hx.guard`` can tell a loaded plugin from a missing one. The
fork's op, layer and model registrations are added here as they move out of
upstream files.
"""

_registered = False


def register() -> None:
    """Entry point: register the fork's code with vLLM (idempotent)."""
    global _registered
    _registered = True


def is_registered() -> bool:
    """Whether :func:`register` ran in this process."""
    return _registered
