# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import tempfile
import unittest
from pathlib import Path

from vllm_cmp170hx.ple.gds_artifact import (
    BLOCK_BYTES, BLOCK_ROWS, DATA_SHA256, PAGE, PAYLOAD, ROW_BYTES, ROWS, SCALE,
    artifact_path,
)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name) / "generation"
        self.directory.mkdir()
        self.addCleanup(self.directory.chmod, 0o755)
        self.data = self.directory / "ple-gds-data.bin"
        self.metadata = self.directory / "ple-gds-metadata.json"
        self.manifest = {
            "schema": "ple-gds-compact-v1",
            "format": "component-block-stride-v1",
            "row_count": ROWS,
            "alignment": PAGE,
            "block_rows": BLOCK_ROWS,
            "data_file": self.data.name,
            "data_size": PAYLOAD + PAGE,
            "data_sha256": DATA_SHA256,
            "components": [dict(
                name="ngram_embedding", dtype="F8_E4M3", row_bytes=ROW_BYTES,
                base_offset=0, block_stride=BLOCK_BYTES, block_payload_bytes=BLOCK_BYTES,
                region_size=PAYLOAD, tail_payload_bytes=BLOCK_BYTES, tail_rows=BLOCK_ROWS,
                block_count=ROWS // BLOCK_ROWS,
            )],
            "globals": [dict(name="weight_scale", dtype="BF16", offset=PAYLOAD, size=2)],
        }

    def write(self, scale=SCALE, size=PAYLOAD + PAGE):
        with self.data.open("wb") as file:
            file.seek(PAYLOAD)
            file.write(scale)
            file.truncate(size)
        self.metadata.write_text(json.dumps(self.manifest))
        self.metadata.chmod(0o444)
        self.data.chmod(0o444)
        self.directory.chmod(0o555)

    def test_layout(self):
        self.write()
        self.assertEqual(artifact_path(self.directory), self.data)

    def test_rejects_unexpected_identity(self):
        self.manifest["data_sha256"] = "0" * 64
        self.write()
        with self.assertRaisesRegex(ValueError, "layout"):
            artifact_path(self.directory)

    def test_rejects_short_artifact(self):
        self.write(size=PAYLOAD)
        with self.assertRaisesRegex(ValueError, "layout"):
            artifact_path(self.directory)

    def test_rejects_wrong_scale(self):
        self.write(scale=b"\0\0")
        with self.assertRaisesRegex(ValueError, "scale"):
            artifact_path(self.directory)

    def test_rejects_writable_payload(self):
        self.write()
        self.data.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "read-only"):
            artifact_path(self.directory)
