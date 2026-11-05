# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The threaded page-cache gather of disk-offload PLE rows
(vllm_cmp170hx/ple/disk_reader.py). The hook in Qwen4ExpPLEDiskOffloadEmbedding
._read_rows that calls it is tested on the vLLM branch
(tests/models/qwen4_exp/test_ple.py,
``test_disk_offload_threaded_gather_matches_serial_reads``)."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.envs as envs
from vllm_cmp170hx import plugin
from vllm_cmp170hx.envs import FORK_ENV_VARS
from vllm_cmp170hx.ple import disk_reader

ROW_BYTES = 24
SHARD_SIZE = 1000
NUM_ROWS = 3500  # shards 0-2 full, shard 3 holds 500 rows


@pytest.fixture
def table(tmp_path):
    """Four shards in two files at unaligned offsets, as a safetensors file
    places them after its header; shard 3 is first in its file."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, (NUM_ROWS, ROW_BYTES), dtype=np.uint8)
    layout = {"a.bin": [(0, 123), (1, None), (2, None)], "b.bin": [(3, 77)]}
    fds, bases, files = [0] * 4, [0] * 4, []
    for file_name, shards in layout.items():
        path = tmp_path / file_name
        blob = bytearray()
        for shard, pad in shards:
            blob += bytes(pad or 5)
            bases[shard] = len(blob)
            blob += data[shard * SHARD_SIZE : (shard + 1) * SHARD_SIZE].tobytes()
        path.write_bytes(bytes(blob) + bytes(11))
        fd = os.open(path, os.O_RDONLY)
        files.append(fd)
        for shard, _ in shards:
            fds[shard] = fd
    sources = (torch.tensor(fds), torch.tensor(bases))
    embedding = SimpleNamespace(
        _sources={"weight": sources}, _shard_size=SHARD_SIZE, org_vocab_size=NUM_ROWS
    )
    yield embedding, data
    for fd in files:
        os.close(fd)


def _read(embedding, ids):
    ids = torch.tensor(sorted(set(ids)), dtype=torch.long)
    shard = ids // SHARD_SIZE
    rows = torch.full((ids.numel(), ROW_BYTES), 7, dtype=torch.uint8)
    done = disk_reader.read_rows(
        embedding, "weight", shard, ids - shard * SHARD_SIZE, rows
    )
    return done, ids, rows


def test_env_var_is_registered_with_vllm():
    plugin.register_fork_env_vars()
    assert (
        envs.environment_variables["VLLM_PLE_DISK_READ_THREADS"]
        is (FORK_ENV_VARS["VLLM_PLE_DISK_READ_THREADS"])
    )


def test_on_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_PLE_DISK_READ_THREADS", raising=False)
    plugin.register_fork_env_vars()
    assert envs.VLLM_PLE_DISK_READ_THREADS == 32


@pytest.mark.parametrize("threads", ["1", "4", "32"])
def test_rows_match_the_files(monkeypatch, table, threads):
    monkeypatch.setenv("VLLM_PLE_DISK_READ_THREADS", threads)
    embedding, data = table
    rng = np.random.default_rng(1)
    ids = rng.integers(0, NUM_ROWS, 2000).tolist() + [0, 999, 1000, 2999, 3000, 3499]
    done, ids, rows = _read(embedding, ids)
    assert done
    np.testing.assert_array_equal(rows.numpy(), data[ids.numpy()])


def test_one_row_and_no_rows(monkeypatch, table):
    monkeypatch.setenv("VLLM_PLE_DISK_READ_THREADS", "8")
    embedding, data = table
    done, _, rows = _read(embedding, [3499])
    assert done
    np.testing.assert_array_equal(rows.numpy(), data[[3499]])
    done, _, rows = _read(embedding, [])
    assert done and rows.numel() == 0


def test_off_leaves_the_read_to_upstream(monkeypatch, table):
    monkeypatch.setenv("VLLM_PLE_DISK_READ_THREADS", "0")
    embedding, _ = table
    done, _, rows = _read(embedding, [1, 2, 3])
    assert not done
    assert (rows == 7).all()


def test_negative_threads_rejected(monkeypatch, table):
    monkeypatch.setenv("VLLM_PLE_DISK_READ_THREADS", "-1")
    with pytest.raises(ValueError, match="VLLM_PLE_DISK_READ_THREADS"):
        _read(table[0], [1])


def test_truncated_file_fails_closed(monkeypatch, table, tmp_path):
    """A row past the end of its file raises instead of faulting the map."""
    monkeypatch.setenv("VLLM_PLE_DISK_READ_THREADS", "4")
    embedding, data = table
    done, ids, rows = _read(embedding, [10, 3499])
    assert done
    np.testing.assert_array_equal(rows.numpy(), data[ids.numpy()])
    os.truncate(tmp_path / "b.bin", 77 + 500 * ROW_BYTES - 4)
    if not os.path.exists("/proc/self/fd"):
        monkeypatch.setattr(disk_reader.os, "readlink", lambda path: path)
    with pytest.raises(ValueError, match="truncated at offset"):
        _read(embedding, [10, 3499])
    done, ids, rows = _read(embedding, [10, 3498])
    np.testing.assert_array_equal(rows.numpy(), data[ids.numpy()])
