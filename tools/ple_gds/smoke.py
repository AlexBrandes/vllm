# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare FP8 PLE staging and graph replay with the original checkpoint."""

import argparse
import os
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from artifact import source_spec
from vllm_cmp170hx.ple.gds_artifact import ROW_BYTES, ROWS, ROWS_PER_SHARD, SCALE


def smoke(directory: Path, checkpoint: Path) -> None:
    from vllm.models.qwen4_exp.nvidia import ngram_embedding as embedding_module
    from vllm_cmp170hx import plugin

    plugin.register_fork_env_vars()
    with patch.dict(os.environ, {"VLLM_PLE_GDS_ARTIFACT": str(directory)}):
        plugin.register_ple_gds()
    disk_embedding = embedding_module.Qwen4ExpPLEDiskOffloadEmbedding
    source_shards = source_spec(checkpoint)["components"][0]["shards"]
    # An offset view exercises the GPU gather's byte-addressed output.
    dtype = torch.float8_e4m3fn
    owner = torch.empty((8194, 1, ROW_BYTES), dtype=dtype, device="cuda")
    output = owner[1:]
    assert output.data_ptr() % 4096 != 0
    shape = torch.empty((ROWS_PER_SHARD, ROW_BYTES), dtype=torch.uint8, device="meta")
    embedding = SimpleNamespace(
        _bound=False, _dummy=False, _nvfp4=False, _shard_size=ROWS_PER_SHARD,
        _shards={"weight": dict.fromkeys(range(len(source_shards)), shape)},
        _planes={"weight": (output, None, None)},
        _host_ids=torch.empty((8193, 1), dtype=torch.int64, pin_memory=True),
        org_vocab_size=ROWS,
        weight_scale=torch.frombuffer(bytearray(SCALE), dtype=torch.bfloat16).float(),
    )
    embedding.copy_ids_to_host = disk_embedding.copy_ids_to_host.__get__(embedding)
    embedding.stage_rows = disk_embedding.stage_rows.__get__(embedding)
    ngram = SimpleNamespace(compute_ngram_ids=lambda ids, *_: ids[:, None])
    modules = [(ngram, embedding)]
    # bind_file_shards calls the same accessor used by the engine during loading.
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        disk_embedding.bind_file_shards(embedding)
    assert embedding._bound and not embedding._shards

    fd = os.open(checkpoint, os.O_RDONLY)
    try:
        def expected(ids):
            offsets = [
                source_shards[shard]["offset"] + local * ROW_BYTES
                for shard, local in (divmod(row, ROWS_PER_SHARD) for row in ids)
            ]
            return b"".join(os.pread(fd, ROW_BYTES, offset) for offset in offsets)

        def stage(values):
            ids = torch.tensor(values, device=output.device)
            embedding_module.stage_checkpoint_rows(modules, ids, None, None)

        stage([25, 127])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            staged = disk_embedding.staged_rows(embedding, 2)
            captured = staged.view(torch.uint8).clone()
        for ids in ([25, 127], [51, ROWS - 1], [25, 127]):
            stage(ids)
            graph.replay()
            assert captured.cpu().numpy().tobytes() == expected(ids)
        print("PASS CUDA Graph replay against checkpoint bytes", flush=True)

        rng = random.Random(12)
        boundary = [25, 25, 51, 127, 128, 255, 256, ROWS_PER_SHARD - 1,
                    ROWS_PER_SHARD, ROWS - 1]
        for count in (32, 256, 2048, 8193):
            ids = boundary + [rng.randrange(ROWS) for _ in range(count - len(boundary))]
            rng.shuffle(ids)
            stage(ids)
            got = output[:count].view(torch.uint8).cpu().numpy().tobytes()
            assert got == expected(ids)
            print(f"PASS {count} staged rows", flush=True)
    finally:
        os.close(fd)
        embedding._gds_reader.native.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    smoke(args.artifact, args.checkpoint)
