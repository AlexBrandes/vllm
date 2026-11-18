# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The EXL3 manifest admits Q4N-X matrices at 2.0 / 2.5 / 3.0 / 3.5 bpw
(bits_x2 4-7) and Q4N at 2.5 / 3.0 only (CPU, header-only checkpoints)."""

import json
import struct

import pytest

from vllm_cmp170hx.kernels.exl3.utils import _MARKERS, Exl3Manifest


def _write(path, marker: str, bits_x2: int, k: int = 256, n: int = 512) -> None:
    tensors = {
        "w.trellis": ("I16", [k // 16, n // 16, 8 * bits_x2]),
        "w.suh": ("F16", [k]),
        "w.svh": ("F16", [n]),
        f"w.{marker}": ("I32", []),
    }
    header, offset, sizes = {}, 0, {"I16": 2, "F16": 2, "I32": 4}
    for name, (dtype, shape) in tensors.items():
        size = sizes[dtype]
        for d in shape:
            size *= d
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    data = bytearray(offset)
    start = header[f"w.{marker}"]["data_offsets"][0]
    data[start : start + 4] = struct.pack("<I", _MARKERS[marker][1])
    (path / "config.json").write_text("{}")
    with open(path / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + bytes(data))


@pytest.mark.parametrize(
    "marker,bits_x2,ok",
    [("q4nx", 4, True), ("q4nx", 5, True), ("q4nx", 6, True), ("q4nx", 7, True),
     ("q4nx", 3, False), ("q4nx", 8, False), ("q4n", 4, False), ("q4n", 5, True)],
)
def test_q4n_widths(tmp_path, marker, bits_x2, ok):
    _write(tmp_path, marker, bits_x2)
    if ok:
        manifest = Exl3Manifest(str(tmp_path))
        assert manifest.matrices["w"].bits_x2 == bits_x2
    else:
        with pytest.raises(ValueError, match="Q4N bitrate"):
            Exl3Manifest(str(tmp_path))
