# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment variables of the CMP 170HX fork.

The plugin's entry point calls :func:`register_env_vars` first
(``vllm_cmp170hx.plugin.register``), so every name below is an ordinary vLLM
environment variable in every process that loaded the plugin: ``envs.X``,
``envs.is_set`` and the unknown-variable check in
``Platform.validate_environ`` all see it.
"""

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    VLLM_MIXED_PREFILL_CHUNK: int = 0

FORK_ENV_VARS: dict[str, Callable[[], Any]] = {
    # While any request is decoding, cap each step's prefill tokens at this
    # many per prompt awaiting prefill, at most one Mamba block, spent
    # oldest-first (vllm_cmp170hx.sched.prefill_budget). 0 = off.
    "VLLM_MIXED_PREFILL_CHUNK": lambda: int(
        os.getenv("VLLM_MIXED_PREFILL_CHUNK", "0") or 0
    ),
}


def register_env_vars(environment_variables: dict[str, Callable[[], Any]]) -> None:
    """Add the fork's variables to vLLM's table (idempotent).

    Raises if upstream vLLM declares one of the fork's names itself, so a
    rebase that brings such a name cannot silently swap the reader.
    """
    for name, getter in FORK_ENV_VARS.items():
        existing = environment_variables.get(name)
        if existing is not None and existing is not getter:
            raise ValueError(
                f"{name} is declared by vLLM and by vllm_cmp170hx/envs.py; "
                "keep one declaration"
            )
    environment_variables.update(FORK_ENV_VARS)
