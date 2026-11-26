# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage compact FP8 PLE rows directly into the existing CUDA graph input."""

import os
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm_cmp170hx.ple.gds_artifact import (
    BLOCK_BYTES, BLOCK_ROWS, NUM_SHARDS, PAGE, ROW_BYTES, ROWS, ROWS_PER_SHARD,
    artifact_path,
)

logger = init_logger(__name__)


class GdsRows:
    def __init__(self, directory: Path, output: torch.Tensor) -> None:
        from vllm.utils.import_utils import import_pynvml

        try:
            from vllm_cmp170hx.ple._persistent_native import PersistentReader
        except ImportError as exc:
            raise RuntimeError(
                "Cannot load the PLE GDS extension. See docs/features/ple-gds.md."
            ) from exc
        if os.getenv("Q38_PLE_HOST_READ") == "1":
            raise ValueError("PLE GDS requires direct cuFile reads")
        if (
            not output.is_cuda or output.ndim < 2
            or output.dtype != torch.float8_e4m3fn
            or not output.is_contiguous() or output.shape[-1] != ROW_BYTES
            or output.numel() == 0
        ):
            raise ValueError("Unsupported PLE GDS output buffer")
        data = artifact_path(directory)
        self.output = output
        self.chunk_rows = min(output.numel() // ROW_BYTES, 4096)
        workers = 16
        # At most two pages per row plus one tile of lane imbalance.
        pages_per_lane = (2 * self.chunk_rows + workers - 1) // workers + 5
        capacity = pages_per_lane * PAGE * workers
        with torch.cuda.device(output.device):
            self.workspace = torch.empty(
                capacity + PAGE - 1, dtype=torch.uint8, device=output.device
            )
            ptr = self.workspace.data_ptr()
            aligned = (ptr + PAGE - 1) // PAGE * PAGE
            uuid = "GPU-" + str(
                torch.cuda.get_device_properties(output.device).uuid
            ).removeprefix("GPU-")
            pynvml = import_pynvml()
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
                bdf = pynvml.nvmlDeviceGetPciInfo(handle).busId
            finally:
                pynvml.nvmlShutdown()
            if isinstance(bdf, bytes):
                bdf = bdf.decode("ascii")
            self.native = PersistentReader(
                str(data), [aligned], capacity, [output.data_ptr()], output.numel(),
                uuid, bdf, self.chunk_rows, 128, workers, False, True,
            )

    def __del__(self) -> None:
        native = getattr(self, "native", None)
        if native is not None:
            native.close()

    def read(self, ids: torch.Tensor) -> None:
        if (
            ids.device.type != "cpu" or ids.dtype != torch.int64 or ids.ndim != 1
            or not ids.is_contiguous() or ids.numel() * ROW_BYTES > self.output.numel()
        ):
            raise ValueError("Invalid PLE GDS row IDs")
        stream = torch.cuda.current_stream(self.output.device).cuda_stream
        rows = ids.numpy()
        for start in range(0, len(rows), self.chunk_rows):
            chunk = rows[start : start + self.chunk_rows]
            self.native.read_ids_at(
                0, chunk, ROWS, 0, BLOCK_ROWS, BLOCK_BYTES, ROW_BYTES, PAGE,
                0, 5 * PAGE, len(chunk) * ROW_BYTES, start * ROW_BYTES, stream,
                "sync_mt", False,
            )
            self.native.release(0, stream)


def install(directory: Path) -> None:
    """Replace the disk embedding constructor on the pinned vLLM engine."""
    from vllm.config import get_current_vllm_config
    from vllm.models.qwen4_exp.nvidia import ngram_embedding

    base = ngram_embedding.Qwen4ExpPLEDiskOffloadEmbedding
    if getattr(base, "_cmp170hx_gds_path", None) == directory:
        return
    if getattr(base, "_cmp170hx_gds_path", None) is not None:
        raise RuntimeError("Changing the PLE GDS artifact requires a restart")

    class GdsDiskEmbedding(base):
        _cmp170hx_gds_path = directory
        _gds_reader: GdsRows | None = None

        def bind_file_shards(self) -> None:
            if self._bound or self._dummy:
                super().bind_file_shards()
                return
            if get_current_vllm_config().parallel_config.tensor_parallel_size != 1:
                raise ValueError("PLE GDS requires tensor_parallel_size=1")
            shards = self._shards["weight"]
            if (
                self._nvfp4 or self.org_vocab_size != ROWS
                or self._shard_size != ROWS_PER_SHARD
                or set(shards) != set(range(NUM_SHARDS))
                or any(s.shape != (ROWS_PER_SHARD, ROW_BYTES) for s in shards.values())
            ):
                raise ValueError("Unsupported PLE GDS FP8 checkpoint shard layout")
            if (
                self.weight_scale.numel() != 1
                or self.weight_scale.item() != 0.00019931793212890625
            ):
                raise ValueError("PLE GDS and checkpoint scales differ")
            self._gds_reader = GdsRows(directory, self._planes["weight"][0])
            self._shards.clear()
            self._bound = True
            logger.info("PLE disk offload: cuFile sync_mt from %s", directory)

        def stage_rows(self, num_tokens: int) -> None:
            if self._gds_reader is None:
                if not self._dummy:
                    raise RuntimeError("PLE GDS staging called before binding")
                super().stage_rows(num_tokens)
                return
            self._gds_reader.read(self._host_ids[:num_tokens].flatten())

    ngram_embedding.Qwen4ExpPLEDiskOffloadEmbedding = GdsDiskEmbedding
