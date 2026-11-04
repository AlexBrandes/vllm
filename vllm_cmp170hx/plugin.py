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
_env_vars_registered = False


def register() -> None:
    """Entry point: register the fork's code with vLLM (idempotent)."""
    global _registered
    register_env_vars()
    _registered = True


def register_env_vars() -> None:
    """Add the fork's variables (``vllm_cmp170hx/envs.py``) to vLLM's table
    ``vllm.envs.environment_variables``, which ``vllm.envs.__getattr__`` reads
    lazily. :func:`register` calls it first, before engine and model code
    (``validate_environ`` runs after it). Fork modules that read a variable
    call it too, for processes where the entry point did not run. Once per
    process."""
    global _env_vars_registered
    if _env_vars_registered:
        return
    import vllm.envs as envs
    from vllm_cmp170hx.envs import register_env_vars as register_table

    register_table(envs.environment_variables)
    _env_vars_registered = True


def is_registered() -> bool:
    """Whether :func:`register` ran in this process."""
    return _registered
