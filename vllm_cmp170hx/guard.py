# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-sm80 (Morrowmake), vllm_sm80/guard.py.
"""Startup check that the fork's vLLM plugin is loaded.

vLLM logs and then ignores an entry-point plugin that fails to load, and
``VLLM_PLUGINS`` can leave a plugin out (``vllm.plugins.load_plugins_by_group``).
Anything the plugin registers would then quietly fall back to upstream
behaviour. ``VllmConfig.__post_init__`` calls :func:`check_plugin_loaded`, which
turns that case into an error at startup.
"""

PLUGIN_NAME = "cmp170hx"
DISTRIBUTION = "vllm-cmp170hx"


def check_plugin_loaded() -> None:
    """Raise if this process loaded vLLM's general plugins but not ``cmp170hx``.

    A process that never loaded general plugins (unit tests that build configs
    directly, offline scripts) is not checked: there is nothing to compare with.
    """
    import vllm.plugins

    if not vllm.plugins.plugins_loaded:
        return

    from vllm_cmp170hx import plugin

    if plugin.is_registered():
        return
    raise RuntimeError(
        f"vLLM loaded its general plugins, but not the fork's plugin "
        f"{PLUGIN_NAME!r}. Install the {DISTRIBUTION} distribution into the "
        f"environment that runs vLLM (pip install --no-deps . from the plugin's "
        f"source tree). If VLLM_PLUGINS is set, it must include {PLUGIN_NAME!r}. "
        f"If the distribution is installed, the log above has "
        f"'Failed to load plugin {PLUGIN_NAME}' with the reason."
    )
