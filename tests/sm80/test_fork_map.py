# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FORK.md must match what tools/sm80/fork_map.py generates from git.

Fix a failure with: python tools/sm80/fork_map.py --write
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "sm80" / "fork_map.py"
FORK_MD = REPO / "FORK.md"


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    )


def test_fork_map_is_current():
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    if _git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        pytest.skip(f"{REPO} is not a git checkout")
    assert FORK_MD.is_file(), (
        "FORK.md is missing. Run: python tools/sm80/fork_map.py --base <sha> --write"
    )
    match = re.search(
        r"^Upstream base: `?([0-9a-f]{7,40})`?[ \t]*$",
        FORK_MD.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, "FORK.md has no 'Upstream base: <sha>' line"
    base = match.group(1)
    if _git("cat-file", "-e", f"{base}^{{commit}}").returncode != 0:
        pytest.skip(f"upstream base {base} is not in this clone (shallow or not fetched)")

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=300,
    )
    assert proc.returncode == 0, (
        "FORK.md is stale. Run: python tools/sm80/fork_map.py --write\n"
        f"{proc.stdout}{proc.stderr}"
    )
