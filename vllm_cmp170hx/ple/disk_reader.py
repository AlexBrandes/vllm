# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A threaded page-cache gather for Qwen4Exp's disk-offload PLE rows
(VLLM_PLE_DISK_READ_THREADS).

With ``--engram-config '{"disk_offload": true}'``, each step reads the PLE rows
of that step's distinct n-gram ids from the checkpoint shard files
(``Qwen4ExpPLEDiskOffloadEmbedding._read_rows``) with one ``preadv`` per row on
the step's critical path. A 1,680-token chunk of real text has thousands of
distinct ids, and the serial reads stalled pipeline stage 0 for ~37 ms per
chunk against ~10 ms on repetitive text: 7-13% of public-v1 prefill on the
CMP 170HX.

Here the shard files are mapped read-only once (``MADV_RANDOM``: a fault reads
only its own page) and the sorted distinct rows are copied out by a thread
pool, in tasks that never cross a shard. ``numpy.take`` releases the GIL for
the copy, so the page faults of different tasks overlap. This is the gather of
the production engine's mmap PLE reader. With the variable 0, the hook returns
False and upstream's serial reads run.

A read past the end of a mapped file faults (SIGBUS) instead of returning
short, so each step first checks every shard it reads against its file's
current size and fails closed as upstream does on a short read.

The vLLM branch's hook in ``_read_rows`` calls :func:`read_rows`. Rows are
located exactly as upstream locates them: ``bind_file_shards`` records, per
plane, each shard's file descriptor and byte offset (``_sources``), and shard
``i`` holds ids ``[i * _shard_size, (i + 1) * _shard_size)`` of
``org_vocab_size``.
"""

import mmap
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm_cmp170hx import plugin

logger = init_logger(__name__)

# Fewer rows than this per task cost more in dispatch than they gain.
MIN_ROWS_PER_TASK = 64

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _threads() -> int:
    # The worker can load the model in a process where the general plugin did
    # not run (unit tests); the variable must still resolve.
    plugin.register_fork_env_vars()
    threads = envs.VLLM_PLE_DISK_READ_THREADS
    if threads < 0:
        raise ValueError(f"VLLM_PLE_DISK_READ_THREADS must be >= 0, got {threads}")
    return threads


def _get_pool(threads: int) -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(threads, thread_name_prefix="cmp170hx-ple")
            logger.info(
                "PLE disk offload: threaded page-cache gather, %d threads", threads
            )
        return _pool


def _shard_views(
    embedding, name: str, row_bytes: int
) -> dict[int, tuple[np.ndarray, int, int]]:
    """``{shard: (uint8 [rows, row_bytes] view, fd, byte offset)}`` over
    read-only maps of the shard files, built once per plane."""
    cache = embedding.__dict__.setdefault("_cmp170hx_shard_views", {})
    views = cache.get(name)
    if views is not None:
        return views
    maps = embedding.__dict__.setdefault("_cmp170hx_maps", {})
    fds, bases = embedding._sources[name]
    shard_size, num_rows = embedding._shard_size, embedding.org_vocab_size
    views = {}
    for shard, (fd, base) in enumerate(zip(fds.tolist(), bases.tolist())):
        if fd not in maps:
            maps[fd] = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            maps[fd].madvise(mmap.MADV_RANDOM)
        rows = min(shard_size, num_rows - shard * shard_size)
        view = np.frombuffer(
            maps[fd], dtype=np.uint8, count=rows * row_bytes, offset=base
        ).reshape(rows, row_bytes)
        views[shard] = (view, fd, base)
    cache[name] = views
    return views


def read_rows(
    embedding,
    name: str,
    shard: torch.Tensor,
    local: torch.Tensor,
    rows: torch.Tensor,
) -> bool:
    """Read one plane's rows for sorted distinct ids into ``rows``.

    ``shard`` and ``local`` locate each id (sorted by id, so by shard);
    ``rows`` is the host ``uint8 [ids, row_bytes]`` destination. Returns False,
    reading nothing, when the threaded gather is off.
    """
    threads = _threads()
    if not threads:
        return False
    num = local.shape[0]
    if not num:
        return True
    row_bytes = rows.shape[1]
    views = _shard_views(embedding, name, row_bytes)
    shard_np, local_np, out = shard.numpy(), local.numpy(), rows.numpy()
    step = max(MIN_ROWS_PER_TASK, -(-num // threads))
    bounds = (np.flatnonzero(np.diff(shard_np)) + 1).tolist()
    tasks = []
    for start, end in zip([0, *bounds], [*bounds, num]):
        view, fd, base = views[int(shard_np[start])]
        # Ids are sorted, so the run's last row ends furthest into the file.
        last = base + int(local_np[end - 1]) * row_bytes
        if os.fstat(fd).st_size < last + row_bytes:
            raise ValueError(
                f"PLE shard file {os.readlink(f'/proc/self/fd/{fd}')} is "
                f"truncated at offset {last}"
            )
        for lo in range(start, end, step):
            tasks.append((view, lo, min(lo + step, end)))

    def copy(task) -> None:
        view, lo, hi = task
        # The caller checked every id against the table size.
        np.take(view, local_np[lo:hi], axis=0, out=out[lo:hi], mode="clip")

    if len(tasks) == 1:
        copy(tasks[0])
    else:
        for _ in _get_pool(threads).map(copy, tasks):
            pass
    return True
