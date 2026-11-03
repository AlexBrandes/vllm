# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assemble an engine directory from the vLLM branch and plugin commits.

    python tools/cmp170hx/assemble.py <dir> --vllm <ref> --plugin <ref>
        [--base <commit>] [--natives-from <vllm package dir>]
    python tools/cmp170hx/assemble.py <dir> --remove

Writes the layout the tests recognize as an assembled engine:

    <dir>/engine         the vLLM branch commit (upstream plus the fork's
                         commits), a detached git worktree
    <dir>/plugin         the plugin commit, a detached git worktree, with the
                         distribution's entry-point metadata beside the package
    <dir>/assembly.json  the upstream, vLLM branch and plugin commits

vLLM then runs with PYTHONPATH=<dir>/engine:<dir>/plugin. A source tree has no
compiled extensions: --natives-from copies the *.so files of a vLLM build of
the same upstream commit (its installed ``vllm`` package directory) into
<dir>/engine/vllm.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fork_map import upstream_base  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "engine-assembly/1"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def write_metadata(plugin: Path) -> None:
    """The dist-info pip would write for the plugin: name, version and the
    entry points, so vLLM finds the plugin on PYTHONPATH."""
    project = tomllib.loads((plugin / "pyproject.toml").read_text())["project"]
    name, version = project["name"], project["version"]
    info = plugin / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )
    lines = []
    for group, entries in project["entry-points"].items():
        lines.append(f"[{group}]")
        lines += [f"{key} = {value}" for key, value in entries.items()]
        lines.append("")
    (info / "entry_points.txt").write_text("\n".join(lines))


def copy_natives(source: Path, engine: Path) -> int:
    count = 0
    for so in sorted(source.rglob("*.so")):
        target = engine / "vllm" / so.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(so, target)
        count += 1
    return count


def assemble(
    out: Path, vllm: str, plugin: str, base: str | None, natives: Path | None
) -> None:
    pins = {
        "schema": SCHEMA,
        "upstream_commit": upstream_base(vllm, base),
        "vllm_commit": git("rev-parse", vllm),
        "plugin_commit": git("rev-parse", plugin),
    }
    out.mkdir(parents=True)
    git("worktree", "add", "--detach", str(out / "engine"), pins["vllm_commit"])
    git("worktree", "add", "--detach", str(out / "plugin"), pins["plugin_commit"])
    write_metadata(out / "plugin")
    if natives is not None:
        print(f"copied {copy_natives(natives, out / 'engine')} extensions")
    (out / "assembly.json").write_text(json.dumps(pins, indent=2) + "\n")
    print(json.dumps(pins, indent=2))
    print(f"PYTHONPATH={out / 'engine'}:{out / 'plugin'}")


def remove(out: Path) -> None:
    for name in ("engine", "plugin"):
        if (out / name).exists():
            git("worktree", "remove", "--force", str(out / name))
    (out / "assembly.json").unlink(missing_ok=True)
    out.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dir", type=Path)
    parser.add_argument("--vllm", help="the vLLM branch or commit")
    parser.add_argument("--base", help="its upstream commit (default: from its name)")
    parser.add_argument("--plugin", help="the plugin branch or commit")
    parser.add_argument("--natives-from", type=Path)
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    out = args.dir.absolute()
    if args.remove:
        remove(out)
    elif args.vllm and args.plugin:
        assemble(out, args.vllm, args.plugin, args.base, args.natives_from)
    else:
        parser.error("--vllm and --plugin are required (or --remove)")


if __name__ == "__main__":
    main()
