# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank-ordered torch profiler trace export (``VLLM_SM80_SERIAL_TRACE_EXPORT``).

Stopping a torch profiler session turns the collected activity into a trace in
host memory, writes it out and keeps the results until the next session
starts. Every worker of an engine stops at the same moment, so the host holds
every rank's trace at once.

With the switch set, the workers of one engine stop their sessions one after
another, in the order of their position in the engine (pipeline stage, prefill
context rank, tensor-parallel rank). A worker waits until the worker before it
has left a done-marker in the trace directory; the marker is written only after
that worker has written its trace, dropped the stopped profiler with its
results and returned the freed heap pages to the system (glibc
``malloc_trim``). So the host holds one rank's trace at a time. During the stop
torch's temporary uncompressed JSON (a gzipped trace is written uncompressed
first) goes into the trace directory instead of the temporary directory, which
may be in memory (tmpfs).

Markers are named by the engine (the workers' parent process), the session
number and the position; the next worker removes the marker it waited for, the
last worker writes none. A worker that waits longer than ``WAIT_S`` (below the
engine's 300 s execute-model timeout, since a session stopped after its
iteration limit stops inside a model step) stops anyway and warns. A process
that is no model-parallel worker (the front-end profiler) is not ordered; its
results are still released. Off by default: nothing is patched.
"""

import ctypes
import ctypes.util
import gc
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

FLAG = "VLLM_SM80_SERIAL_TRACE_EXPORT"
WAIT_S = 240.0
POLL_S = 0.05


def enabled() -> bool:
    import vllm.envs as envs

    return bool(getattr(envs, FLAG))


def install() -> bool:
    """Wrap ``TorchProfilerWrapper._stop`` when the switch is set."""
    if not enabled():
        return False
    from vllm.profiler.wrapper import TorchProfilerWrapper

    if getattr(TorchProfilerWrapper, "_sm80_serial_trace_export", False):
        return True
    stop = TorchProfilerWrapper._stop

    def _stop(self) -> None:
        from vllm.config.profiler import _is_uri_path

        session = getattr(self, "_sm80_export_session", 0)
        self._sm80_export_session = session + 1
        directory = self._torch_profiler_trace_dir
        local = bool(directory) and not _is_uri_path(directory)
        where = engine_position() if local else None
        if where is not None and where[0] > 0:
            wait_for(marker(directory, session, where[0] - 1))
        try:
            with temp_dir(directory if local else None):
                stop(self)
        finally:
            release(self)
            if where is not None and where[0] < where[1] - 1:
                mark(marker(directory, session, where[0]))

    TorchProfilerWrapper._stop = _stop
    TorchProfilerWrapper._sm80_serial_trace_export = True
    logger.warning_once(
        "Serial trace export on: the workers of an engine stop their torch "
        "profiler sessions one at a time (%s=1)",
        FLAG,
    )
    return True


def engine_position() -> tuple[int, int] | None:
    """(position, workers) of this worker in its engine: pipeline stage, then
    prefill context rank, then tensor-parallel rank. None outside a
    model-parallel worker."""
    from vllm.distributed import parallel_state as ps

    if not ps.model_parallel_is_initialized():
        return None
    tp, pp = ps.get_tp_group(), ps.get_pp_group()
    pcp_rank, pcp_size = 0, 1
    if ps._PCP is not None:
        pcp_rank, pcp_size = ps._PCP.rank_in_group, ps._PCP.world_size
    position = (pp.rank_in_group * pcp_size + pcp_rank) * tp.world_size
    return position + tp.rank_in_group, pp.world_size * pcp_size * tp.world_size


def marker(directory: str, session: int, position: int) -> Path:
    """The done-marker of ``position``'s ``session`` in this engine."""
    name = f".trace-export.{os.getppid()}.{session}.{position}.done"
    return Path(directory) / name


def wait_for(path: Path) -> bool:
    """Wait for the previous worker's marker and remove it; False (with a
    warning) when it does not appear within WAIT_S."""
    deadline = time.monotonic() + WAIT_S
    while not path.exists():
        if time.monotonic() >= deadline:
            logger.warning(
                "Serial trace export: no %s after %.0f s; exporting anyway",
                path,
                WAIT_S,
            )
            return False
        time.sleep(POLL_S)
    path.unlink(missing_ok=True)
    return True


def mark(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


@contextmanager
def temp_dir(directory: str | None) -> Iterator[None]:
    """Temporary files of the block go into ``directory`` (None: unchanged)."""
    if directory is None:
        yield
        return
    Path(directory).mkdir(parents=True, exist_ok=True)
    saved = tempfile.tempdir
    tempfile.tempdir = str(directory)
    try:
        yield
    finally:
        tempfile.tempdir = saved


def release(wrapper: Any) -> None:
    """Drop the stopped profiler (its results) and return freed heap pages to
    the system. The next session's start creates a new profiler."""
    wrapper.__dict__.pop("profiler", None)
    gc.collect()
    name = ctypes.util.find_library("c")
    if name is None:
        return
    libc = ctypes.CDLL(name)
    if hasattr(libc, "malloc_trim"):
        libc.malloc_trim(0)
