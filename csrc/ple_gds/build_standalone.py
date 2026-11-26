# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the optional PLE reader with the engine's Python and CUDA toolkit."""

import argparse
import hashlib
import os
import subprocess
import sysconfig
import tempfile
import urllib.request
from pathlib import Path

REVISION = "4d229f1438486fa759727f128fc28f3609423dbb"
SOURCE = (
    "https://raw.githubusercontent.com/nguyenthimy2022kg-alt/"
    f"Qwen-Flash-SM80-170HX/{REVISION}/csrc/ple_gds/ple_gds_persistent.cpp"
)
SHA256 = "0b3f4fbecf0e9f74eedb9f9f1a6a4d3781bde2b5110eb7d0ae1a48db3c7b9f19"
ROOT = Path(__file__).resolve().parents[2]


def adapt(source: bytes) -> str:
    if hashlib.sha256(source).hexdigest() != SHA256:
        raise ValueError(f"Expected PLE reader source at {REVISION}")
    text = source.decode()
    # The 160-byte rows tile exactly every five 4 KiB pages. Each row
    # must fit in one planned range and one registered worker lane.
    text = text.replace(
        "if (gap <= max_gap_pages && extended_size <= effective_max_range) {",
        "if (page % 5 != 0 && gap <= max_gap_pages && "
        "extended_size <= effective_max_range) {",
    )
    # cuFile registers staging alone. The GPU gather writes output bytes.
    text = text.replace(
        "if (pointer & 4095U)",
        'if (std::strcmp(name, "staging") == 0 && (pointer & 4095U))',
    )
    return (
        "// SPDX-License-Identifier: Apache-2.0\n"
        "// Copyright 2026 Qwen-Flash-SM80 contributors.\n"
        "// Adapted for the vLLM CMP 170HX plugin.\n" + text
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="local copy of the pinned source")
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument("--cufile", type=Path, required=True)
    parser.add_argument("--cufile-lib", type=Path, help="cuFile library directory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "vllm_cmp170hx/ple")
    args = parser.parse_args()
    if args.source:
        text = adapt(args.source.read_bytes())
    else:
        with urllib.request.urlopen(SOURCE, timeout=30) as response:
            text = adapt(response.read())

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    output = output_dir / ("_persistent_native" + suffix)
    cufile_lib = (args.cufile_lib or args.cufile / "lib").resolve()
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        source = Path(tmp) / "persistent.cpp"
        source.write_text(text)
        library = Path(tmp) / output.name
        subprocess.run(
            [
                os.environ.get("CXX", "g++"), "-std=c++17", "-O3", "-pthread",
                "-fPIC", "-shared", f"-I{sysconfig.get_path('include')}",
                f"-I{args.cuda / 'include'}", f"-I{args.cufile / 'include'}",
                str(source), f"-L{args.cuda / 'lib64/stubs'}", f"-L{cufile_lib}",
                f"-Wl,-rpath,{cufile_lib}", "-l:libcufile.so.0", "-lcuda",
                "-o", str(library),
            ],
            check=True,
        )
        library.replace(output)
    print(output)


if __name__ == "__main__":
    main()
