# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Describe or verify the compact GDS artifact for the FP8 PLE checkpoint."""

import argparse
import hashlib
import json
import mmap
import os
import struct
from pathlib import Path

from vllm_cmp170hx.ple.gds_artifact import (
    DATA_SHA256, NUM_SHARDS, PAGE, PAYLOAD, ROW_BYTES, ROWS, ROWS_PER_SHARD,
    SCALE, artifact_path,
)

PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
CHUNK = 8 * 1024 * 1024


def source_spec(checkpoint: Path) -> dict:
    with checkpoint.open("rb") as file:
        header_size = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_size))
        base = file.tell()

        def tensor(name, dtype, shape, size):
            entry = header[PREFIX + name]
            start, end = entry["data_offsets"]
            if (
                entry["dtype"] != dtype or entry["shape"] != shape
                or start < 0 or end - start != size
                or base + end > checkpoint.stat().st_size
            ):
                raise ValueError(f"Unexpected safetensors entry for {name}")
            return dict(
                path=str(checkpoint.resolve()), offset=base + start,
                nbytes=size, dtype=dtype, shape=shape, tensor=name,
            )

        shards = [
            dict(
                tensor(f"shard_{index}.weight", "F8_E4M3",
                       [ROWS_PER_SHARD, ROW_BYTES], ROWS_PER_SHARD * ROW_BYTES),
                row_start=index * ROWS_PER_SHARD,
                rows=ROWS_PER_SHARD,
                row_bytes=ROW_BYTES,
            )
            for index in range(NUM_SHARDS)
        ]
        scale_entry = header[PREFIX + "weight_scale"]
        scale_shape = scale_entry["shape"]
        if scale_shape not in ([], [1]):
            raise ValueError("Expected one BF16 PLE scale")
        scale = dict(tensor("weight_scale", "BF16", scale_shape, len(SCALE)),
                     rows=1, row_bytes=len(SCALE))
        file.seek(scale["offset"])
        if file.read(len(SCALE)) != SCALE:
            raise ValueError("Unsupported checkpoint PLE scale")
    return dict(
        schema="ple-gds-source-v1", row_count=ROWS,
        components=[
            dict(name="ngram_embedding", dtype="F8_E4M3", shards=shards),
            dict(name="weight_scale", dtype="BF16", shards=[scale], **{"global": True}),
        ],
    )


def digest_ranges(path, ranges) -> str:
    """Hash with O_DIRECT so verification leaves the page cache cold."""
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        with mmap.mmap(-1, CHUNK + PAGE) as buffer, memoryview(buffer) as view:
            for begin, length in ranges:
                for offset in range(begin, begin + length, CHUNK):
                    used = min(CHUNK, begin + length - offset)
                    aligned = offset // PAGE * PAGE
                    delta = offset - aligned
                    size = (used + delta + PAGE - 1) // PAGE * PAGE
                    if os.preadv(fd, [view[:size]], aligned) < used + delta:
                        raise OSError(f"Short O_DIRECT read at {path}:{offset}")
                    digest.update(view[delta : delta + used])
    finally:
        os.close(fd)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--verify", type=Path, metavar="ARTIFACT_DIR")
    args = parser.parse_args()
    spec = source_spec(args.checkpoint)
    if args.verify is None:
        print(json.dumps(spec, indent=2))
        return
    artifact = artifact_path(args.verify)
    if digest_ranges(artifact, [(0, PAYLOAD + PAGE)]) != DATA_SHA256:
        raise ValueError("PLE GDS artifact SHA-256 mismatch")
    ranges = [(s["offset"], s["nbytes"]) for s in spec["components"][0]["shards"]]
    payload_digest = digest_ranges(artifact, [(0, PAYLOAD)])
    if payload_digest != digest_ranges(args.checkpoint, ranges):
        raise ValueError("PLE GDS rows differ from the checkpoint")
    print(f"Verified {PAYLOAD} FP8 bytes and the BF16 scale")


if __name__ == "__main__":
    main()
