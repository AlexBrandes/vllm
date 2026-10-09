# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""setup.py must load vLLM's environment variables without an importable vllm.

setup.py loads vllm/envs.py by file path, and vllm/envs.py registers the fork's
variables from vllm/sm80/envs.py at the end of its import. An isolated PEP 517
build does not put the source tree on sys.path, so setup.py loads
vllm/sm80/envs.py by path first. This runs setup.py's own loading statements in
a clean interpreter (-I: no PYTHONPATH, no script or working directory on
sys.path) and checks that the fork's variables are registered and that every
variable setup.py reads resolves, without importing the vllm package.
"""

import ast
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SETUP = REPO / "setup.py"


def _loading_statements() -> str:
    """load_module_from_path, ROOT_DIR and every statement that loads an envs.py."""
    source = SETUP.read_text(encoding="utf-8")
    keep = []
    for node in ast.parse(source).body:
        text = ast.get_source_segment(source, node)
        if isinstance(node, ast.FunctionDef) and node.name == "load_module_from_path":
            keep.append(text)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ROOT_DIR" for t in node.targets
        ):
            keep.append(text)
        elif "load_module_from_path(" in text and '"envs.py"' in text:
            keep.append(text)
    return "\n".join(keep)


def test_setup_loads_fork_env_vars_without_vllm_package(tmp_path):
    statements = _loading_statements()
    read = sorted(set(re.findall(r"\benvs\.(VLLM_[A-Z0-9_]+)", SETUP.read_text())))
    assert "VLLM_AMPERE_MARLIN_BUILD" in read, read
    script = tmp_path / "load_envs.py"
    script.write_text(
        "import importlib.util\nimport os\nimport sys\nfrom pathlib import Path\n"
        f"__file__ = {str(SETUP)!r}\n"
        f"READ = {read!r}\n"
        + statements
        + textwrap.dedent(
            """
            from vllm.sm80.envs import FORK_ENV_VARS
            assert "vllm" not in sys.modules, sorted(sys.modules)
            assert set(FORK_ENV_VARS) <= set(envs.environment_variables)
            for name in READ:
                getattr(envs, name)
            print("AMPERE_MARLIN_BUILD", envs.VLLM_AMPERE_MARLIN_BUILD)
            """
        )
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("VLLM_", "PYTHON"))
    }
    env["VLLM_AMPERE_MARLIN_BUILD"] = "1"
    proc = subprocess.run(
        [sys.executable, "-I", str(script)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "AMPERE_MARLIN_BUILD True" in proc.stdout, proc.stdout + proc.stderr
