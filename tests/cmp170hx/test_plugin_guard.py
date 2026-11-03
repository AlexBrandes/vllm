# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm-sm80 (Morrowmake), tests/sm80/test_plugin_guard.py.
"""The fork's plugin entry point and the startup guard (vllm_cmp170hx/guard.py)."""

import os
import pathlib
import subprocess
import sys
import tomllib

import pytest

import vllm.plugins
from vllm_cmp170hx import guard, plugin

REPO = pathlib.Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"


def _entry_point() -> str:
    data = tomllib.loads(PYPROJECT.read_text())
    entry_points = data["project"]["entry-points"]
    return entry_points[vllm.plugins.DEFAULT_PLUGINS_GROUP][guard.PLUGIN_NAME]


def test_distribution_declares_the_register_entry_point():
    data = tomllib.loads(PYPROJECT.read_text())
    assert data["project"]["name"] == guard.DISTRIBUTION
    assert _entry_point() == "vllm_cmp170hx.plugin:register"


def test_distribution_declares_no_platform_plugin():
    """The fork runs on upstream's CUDA platform."""
    data = tomllib.loads(PYPROJECT.read_text())
    assert vllm.plugins.PLATFORM_PLUGINS_GROUP not in data["project"]["entry-points"]


def test_guard_ignores_a_process_without_plugins(monkeypatch):
    monkeypatch.setattr(vllm.plugins, "plugins_loaded", False)
    monkeypatch.setattr(plugin, "_registered", False)
    guard.check_plugin_loaded()


def test_guard_passes_when_the_plugin_ran(monkeypatch):
    monkeypatch.setattr(vllm.plugins, "plugins_loaded", True)
    monkeypatch.setattr(plugin, "_registered", False)
    plugin.register()
    plugin.register()
    guard.check_plugin_loaded()


def test_guard_raises_when_plugins_loaded_without_ours(monkeypatch):
    monkeypatch.setattr(vllm.plugins, "plugins_loaded", True)
    monkeypatch.setattr(plugin, "_registered", False)
    with pytest.raises(RuntimeError, match="not the fork's plugin 'cmp170hx'"):
        guard.check_plugin_loaded()


def _write_metadata(site: pathlib.Path) -> None:
    """The metadata pip writes for the distribution, without running pip."""
    info = site / "vllm_cmp170hx-0.1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {guard.DISTRIBUTION}\nVersion: 0.1.0\n"
    )
    (info / "entry_points.txt").write_text(
        f"[{vllm.plugins.DEFAULT_PLUGINS_GROUP}]\n"
        f"{guard.PLUGIN_NAME} = {_entry_point()}\n"
    )


# A CPU device config, so the child starts on a host without a GPU too; the
# guard runs first in VllmConfig.__post_init__ either way.
CHILD = """
import vllm.plugins
from vllm.config import DeviceConfig, VllmConfig
vllm.plugins.load_general_plugins()
VllmConfig(device_config=DeviceConfig(device="cpu"))
print("STARTED")
"""


def _has_metadata(path: str) -> bool:
    return any(pathlib.Path(path or ".").glob("vllm_cmp170hx-*.dist-info"))


def _start(tmp_path, with_metadata: bool, plugins: str | None = None):
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    if with_metadata:
        _write_metadata(site_dir)
    # The package stays importable when its metadata is hidden: the path
    # entries dropped below may hold both (an assembled engine's plugin).
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "vllm_cmp170hx").symlink_to(REPO / "vllm_cmp170hx")
    env = {
        k: v for k, v in os.environ.items() if k not in ("VLLM_PLUGINS", "PYTHONPATH")
    }
    # The child sees the plugin metadata only when this test writes it.
    paths = [p for p in sys.path if not _has_metadata(p)]
    env["PYTHONPATH"] = os.pathsep.join([str(site_dir), str(package_dir), *paths])
    env["CUDA_VISIBLE_DEVICES"] = ""
    if plugins is not None:
        env["VLLM_PLUGINS"] = plugins
    return subprocess.run(
        [sys.executable, "-c", CHILD],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_config_starts_with_the_plugin_installed(tmp_path):
    proc = _start(tmp_path, with_metadata=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "STARTED" in proc.stdout


@pytest.mark.parametrize(
    "with_metadata,plugins",
    [(False, None), (True, "some_other_plugin")],
    ids=["not-installed", "filtered-by-VLLM_PLUGINS"],
)
def test_config_fails_loudly_without_the_plugin(tmp_path, with_metadata, plugins):
    proc = _start(tmp_path, with_metadata=with_metadata, plugins=plugins)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "STARTED" not in proc.stdout
    assert "not the fork's plugin 'cmp170hx'" in out
