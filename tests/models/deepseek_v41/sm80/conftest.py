# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixtures these tests took from vLLM's tests/conftest.py when they lived in
the vLLM tree (vllm-sm80); the plugin repository carries its own copy."""

import pytest


@pytest.fixture
def default_vllm_config():
    """A default VllmConfig as the current config (vLLM's fixture)."""
    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    with set_current_vllm_config(config):
        yield config
