# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout of the compact FP8 PLE artifact supported by the GDS reader."""

import json
import stat
from pathlib import Path

NUM_SHARDS = 128
ROWS_PER_SHARD = 2_500_012
ROWS = NUM_SHARDS * ROWS_PER_SHARD
ROW_BYTES = 160
PAGE = 4096
BLOCK_ROWS = 256
BLOCK_BYTES = BLOCK_ROWS * ROW_BYTES
PAYLOAD = ROWS * ROW_BYTES
SCALE = b"\x51\x39"  # BF16 little-endian
DATA_SHA256 = "9753c76816abd538fef53359d4315b00d0fae0d38f6cbab19ae9baaec1f748ad"


def artifact_path(directory: str | Path) -> Path:
    directory = Path(directory)
    data = directory / "ple-gds-data.bin"
    metadata = directory / "ple-gds-metadata.json"
    for path, kind in ((directory, stat.S_ISDIR), (data, stat.S_ISREG),
                       (metadata, stat.S_ISREG)):
        mode = path.lstat().st_mode
        if mode & 0o222 or not kind(mode):
            raise ValueError(f"Expected read-only PLE GDS artifact at {path}")
    manifest = json.loads(metadata.read_text())
    layout = dict(
        schema="ple-gds-compact-v1", format="component-block-stride-v1",
        row_count=ROWS, alignment=PAGE, block_rows=BLOCK_ROWS, data_file=data.name,
        data_size=PAYLOAD + PAGE, data_sha256=DATA_SHA256,
        components=[dict(
            name="ngram_embedding", dtype="F8_E4M3", row_bytes=ROW_BYTES,
            base_offset=0, block_stride=BLOCK_BYTES, block_payload_bytes=BLOCK_BYTES,
            region_size=PAYLOAD, tail_payload_bytes=BLOCK_BYTES, tail_rows=BLOCK_ROWS,
            block_count=ROWS // BLOCK_ROWS,
        )],
        globals=[dict(name="weight_scale", dtype="BF16", offset=PAYLOAD, size=2)],
    )
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != value for key, value in layout.items()
    ) or data.stat().st_size != PAYLOAD + PAGE:
        raise ValueError(f"Unsupported PLE GDS artifact layout at {data}")
    with data.open("rb", buffering=0) as file:
        file.seek(PAYLOAD)
        if file.read(len(SCALE)) != SCALE:
            raise ValueError(f"Invalid PLE GDS BF16 scale in {data}")
    return data
