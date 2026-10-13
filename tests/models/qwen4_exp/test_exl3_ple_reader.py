# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The EXL3 PLE row reader: positional reads, retries, truncation, read
faults, failed-token masks, page-cache residency, slot reuse, the
prefetch timing summary and the SSD read overlapped with the layer ahead."""

import errno
import json
import logging
import mmap
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from itertools import accumulate

import numpy as np
import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.common import exl3_ple
from vllm.models.qwen4_exp.common.exl3_ple import (
    Exl3PLEEmbedding,
    Exl3PLERowReader,
    Exl3PLESlot,
    Exl3PLETiming,
    copy_masking_failed,
    poison_rows,
    pread_exact_into,
)


_PLE_PREFIX = "model.layers.0.ple.ple_embedding.ngram_embedding"


def _packed_ple_file(
    tmp_path, offset=32, *, bits=6, dim=160, sizes=(5, 4), rows=10, offsets=None,
    metadata=None, written_rows=None,
):
    """Small real PLE file: two heads, a sparse gap before the trellis and one
    padded row. Row r holds the byte r + 1 repeated."""
    words = 1 + dim * bits // 16
    heads = len(sizes)
    offsets = list(accumulate((0, *sizes[:-1]))) if offsets is None else offsets
    rows_data = [bytes([row + 1]) * (2 * words) for row in range(rows)]
    parts = [
        ("head_bias", "F16", [heads, dim], bytes(range(256)) * (heads * dim)),
        ("head_offsets", "I64", [heads], struct.pack(f"<{heads}q", *offsets)),
        ("head_vocab_sizes", "I64", [heads], struct.pack(f"<{heads}q", *sizes)),
        ("layer_multipliers", "I64", [3], struct.pack("<3q", 3, 5, 7)),
        ("gap", "I16", [offset // 2], None),
        ("trellis", "I16", [rows, words], b"".join(rows_data)),
    ]
    header = {
        "__metadata__": {
            "format": "exl3_ngram_trellis", "version": "1", "K": str(bits),
            "codebook": "mul1", "row_dim": str(dim), "rows": str(rows),
            **(metadata or {}),
        }
    }
    end = 0
    for name, dtype, shape, _ in parts:
        size = (2 if dtype in ("F16", "I16") else 8) * shape[0] * (
            shape[1] if len(shape) > 1 else 1
        )
        header[f"{_PLE_PREFIX}.{name}"] = {
            "dtype": dtype, "shape": shape, "data_offsets": [end, end + size],
        }
        end += size
    raw = json.dumps(header).encode()
    path = tmp_path / "ngram_embedding.safetensors"
    with path.open("wb") as file:
        file.write(struct.pack("<Q", len(raw)))
        file.write(raw)
        for name, _, _, data in parts:
            info = header[f"{_PLE_PREFIX}.{name}"]["data_offsets"]
            size = info[1] - info[0]
            if name == "trellis" and written_rows is not None:
                data = data[: written_rows * 2 * words]
            if data is None:
                file.seek(size, os.SEEK_CUR)
            else:
                file.write(data[:size])
        file.truncate(8 + len(raw) + end)
    base = 8 + len(raw) + header[f"{_PLE_PREFIX}.trellis"]["data_offsets"][0]
    return path, f"{_PLE_PREFIX}.trellis", rows_data, base


_PLE_LAYOUTS = [
    pytest.param(6, 160, id="k6x160"),
    pytest.param(3, 32, id="k3x32"),
    pytest.param(5, 16, id="k5x16"),
]


@pytest.mark.parametrize("bits,dim", _PLE_LAYOUTS)
@pytest.mark.parametrize("offset", [32, (1 << 32) + 16])
def test_exl3_ple_positional_rows_preserve_order_and_masks(
    tmp_path, offset, bits, dim
):
    path, key, rows, base = _packed_ple_file(tmp_path, offset, bits=bits, dim=dim)
    reader = Exl3PLERowReader(str(path), key, 4, 10)
    try:
        ids = [8, 4, 5, 4, 9, -1, 3, 10]
        row_bytes = reader.row_bytes
        target = bytearray(len(ids) * row_bytes)
        reader.read_rows(ids, memoryview(target))
        assert reader.base == base
        assert row_bytes == 2 + dim * bits // 8 == len(rows[0])
        assert reader.layout.bits == bits
        assert target == b"".join(
            rows[i] if i in (4, 5, 8) else bytes(row_bytes) for i in ids
        )
        assert os.lseek(reader.fd, 0, os.SEEK_CUR) == 0
    finally:
        reader.close()


@pytest.mark.parametrize("bits,dim", _PLE_LAYOUTS)
def test_exl3_ple_read_retries_interruptions_and_partial_reads(
    tmp_path, monkeypatch, bits, dim
):
    path, key, rows, _ = _packed_ple_file(tmp_path, bits=bits, dim=dim)
    reader = Exl3PLERowReader(str(path), key, 0, 10)
    real_preadv = os.preadv
    attempts = 0

    def interrupted_read(fd, buffers, offset):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise InterruptedError()
        return real_preadv(fd, [buffers[0][:7]], offset)

    monkeypatch.setattr(os, "preadv", interrupted_read)
    try:
        target = bytearray(2 * reader.row_bytes)
        reader.read_rows([8, 1], memoryview(target))
        assert target == rows[8] + rows[1]
    finally:
        reader.close()


def test_exl3_ple_truncation_after_open_fails_only_cut_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(exl3_ple.time, "sleep", lambda _: None)
    path, key, rows, base = _packed_ple_file(tmp_path)
    reader = Exl3PLERowReader(str(path), key, 0, 10, retries=2)
    try:
        os.truncate(path, base + 8 * reader.row_bytes + 10)
        target = bytearray(2 * reader.row_bytes)
        failed = reader.read_rows([8, 1], memoryview(target))
        assert [(index, type(error)) for index, error in failed] == [(0, EOFError)]
        assert "Truncated PLE" in str(failed[0][1])
        assert target[reader.row_bytes :] == rows[1]
    finally:
        reader.close()
    with pytest.raises(ValueError, match="extent"):
        Exl3PLERowReader(str(path), key, 0, 10)


def test_exl3_ple_rejects_invalid_ranges_and_destinations(tmp_path):
    path, key, _, _ = _packed_ple_file(tmp_path)
    with pytest.raises(ValueError, match="owner range"):
        Exl3PLERowReader(str(path), key, 0, 11)
    with pytest.raises(ValueError, match="trellis"):
        Exl3PLERowReader(str(path), key.removesuffix(".trellis"), 0, 10)
    reader = Exl3PLERowReader(str(path), key, 0, 10)
    try:
        with pytest.raises(ValueError, match="destination"):
            reader.read_rows([1, 2], memoryview(bytearray(reader.row_bytes)))
        with pytest.raises(ValueError, match="64-bit"):
            pread_exact_into(reader.fd, memoryview(bytearray(1)), (1 << 63) - 1)
    finally:
        reader.close()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"offsets": [0, 6]}, "head range"),
        ({"sizes": (5, 0)}, "head range"),
        ({"rows": 140}, "padding"),
        ({"metadata": {"codebook": "mcg"}}, "format"),
        ({"metadata": {"rows": "11"}}, "metadata"),
        ({"metadata": {"K": "5"}}, "do not hold"),
        ({"bits": 3, "dim": 24}, "do not hold"),
    ],
)
def test_exl3_ple_rejects_invalid_layouts(tmp_path, kwargs, match):
    path, key, _, _ = _packed_ple_file(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=match):
        Exl3PLERowReader(str(path), key, 0, 10)


def _faulty_preadv(monkeypatch, fault, rows_to_fail, base, row_bytes, times=None):
    """Make reads of the given rows fail `times` times (always if None)."""
    real_preadv = os.preadv
    attempts = {}

    def preadv(fd, buffers, offset):
        row, within = divmod(offset - base, row_bytes)
        if row in rows_to_fail:
            if within == 0:
                attempts[row] = attempts.get(row, 0) + 1
            if times is None or attempts[row] <= times:
                if fault != "short":
                    raise OSError(fault, os.strerror(fault))
                # Part of the row, then end of file.
                if within == 0:
                    return real_preadv(fd, [buffers[0][:5]], offset)
                return 0
        return real_preadv(fd, buffers, offset)

    monkeypatch.setattr(os, "preadv", preadv)
    return attempts


_READ_FAULTS = [
    pytest.param(errno.EINTR, id="eintr"),
    pytest.param(errno.EIO, id="eio"),
    pytest.param("short", id="short"),
]


@pytest.mark.parametrize("bits,dim", _PLE_LAYOUTS)
@pytest.mark.parametrize("fault", _READ_FAULTS)
def test_exl3_ple_transient_read_error_is_retried(
    tmp_path, monkeypatch, fault, bits, dim
):
    sleeps = []
    monkeypatch.setattr(exl3_ple.time, "sleep", sleeps.append)
    path, key, rows, base = _packed_ple_file(tmp_path, bits=bits, dim=dim)
    reader = Exl3PLERowReader(str(path), key, 0, 10, retries=3)
    try:
        attempts = _faulty_preadv(
            monkeypatch, fault, {4}, base, reader.row_bytes, times=1
        )
        ids = [4, 2, 4, 8]
        target = bytearray(len(ids) * reader.row_bytes)
        assert reader.read_rows(ids, memoryview(target)) == []
        assert target == b"".join(rows[i] for i in ids)
        assert sleeps == [0.001]
        assert attempts[4] >= 2
    finally:
        reader.close()


@pytest.mark.parametrize("fault", _READ_FAULTS)
def test_exl3_ple_persistent_read_error_fails_only_its_rows(
    tmp_path, monkeypatch, fault
):
    sleeps = []
    monkeypatch.setattr(exl3_ple.time, "sleep", sleeps.append)
    path, key, rows, base = _packed_ple_file(tmp_path)
    reader = Exl3PLERowReader(str(path), key, 0, 10, retries=3)
    try:
        _faulty_preadv(monkeypatch, fault, {5}, base, reader.row_bytes)
        ids = [5, 2, 9, 5, 7]
        target = bytearray(len(ids) * reader.row_bytes)
        failed = reader.read_rows(ids, memoryview(target))
        assert [index for index, _ in failed] == [0, 3]
        error = EOFError if fault == "short" else OSError
        assert all(isinstance(e, error) for _, e in failed)
        # Bounded: retries + 1 attempts per failed row, doubling backoff.
        assert sleeps == [0.001, 0.002, 0.004] * 2
        size = reader.row_bytes
        for index, row in ((1, rows[2]), (2, bytes(size)), (4, rows[7])):
            assert target[index * size : (index + 1) * size] == row
        # The reader is not stuck: later reads of other rows succeed...
        target = bytearray(2 * reader.row_bytes)
        assert reader.read_rows([1, 8], memoryview(target)) == []
        assert target == rows[1] + rows[8]
        # ...and the failed row reads again once the fault clears.
        monkeypatch.undo()
        target = bytearray(reader.row_bytes)
        assert reader.read_rows([5], memoryview(target)) == []
        assert target == rows[5]
    finally:
        reader.close()


def test_exl3_ple_poisoned_rows_become_zero_failed_tokens():
    """A poisoned row decodes to NaN (NaN scale times any codebook value);
    the NaN survives the rank sum and only its token is zeroed and flagged."""
    packed = np.zeros((6, 61), dtype=np.int16)
    packed[:, 0] = np.float16(0.5).view(np.int16)
    poison_rows(packed, [3])
    scales = packed[:, 0].view(np.float16).astype(np.float32)
    assert np.isnan(scales[3]) and not np.isnan(np.delete(scales, 3)).any()
    # Rows are [tokens=3, heads=2, dim]; row 3 is token 1, head 1.
    owner = torch.from_numpy(scales).repeat_interleave(4).view(3, 2, 4)
    other = torch.ones(3, 2, 4)
    summed = (owner * 2 + other).to(torch.bfloat16)
    output = torch.full((3, 8), 7.0, dtype=torch.bfloat16)
    failed = copy_masking_failed(summed, output)
    assert failed.tolist() == [False, True, False]
    assert torch.equal(output[1], torch.zeros(8, dtype=torch.bfloat16))
    assert torch.equal(output[[0, 2]], summed[[0, 2]].flatten(-2))


def test_exl3_ple_advises_random_reads_and_cold_drop(tmp_path, monkeypatch):
    path, key, _, base = _packed_ple_file(tmp_path)
    calls = []
    real_fadvise = os.posix_fadvise

    def record(fd, offset, length, advice):
        calls.append((offset, length, advice))
        real_fadvise(fd, offset, length, advice)

    monkeypatch.setattr(os, "posix_fadvise", record)
    reader = Exl3PLERowReader(str(path), key, 0, 10)
    try:
        assert calls == [(0, 0, os.POSIX_FADV_RANDOM)]
        reader.drop_page_cache()
        assert calls[1] == (base, 10 * reader.row_bytes, os.POSIX_FADV_DONTNEED)
    finally:
        reader.close()


def test_exl3_ple_page_cache_residency_counts_table_pages(tmp_path):
    """Only written table pages are resident; holes and other tensors are not
    counted, and sampling itself faults nothing in."""
    path, key, _, base = _packed_ple_file(
        tmp_path, offset=(1 << 20), sizes=(150, 40), rows=200, written_rows=10
    )
    page = mmap.PAGESIZE
    reader = Exl3PLERowReader(str(path), key, 0, 200)
    try:
        end = base + 200 * reader.row_bytes
        total = (end - 1) // page - base // page + 1
        written = (base + 10 * reader.row_bytes - 1) // page - base // page + 1
        assert written < total
        assert reader.page_cache_residency() == (written, total)
        assert reader.page_cache_residency() == (written, total)
        os.fsync(reader.fd)
        reader.drop_page_cache()
        resident, _ = reader.page_cache_residency()
        assert resident in (0, written)  # tmpfs keeps pages; disks drop them
    finally:
        reader.close()


def test_exl3_ple_slot_waits_for_copy_before_reuse():
    """A released lease still waits for the asynchronous output copy."""
    class Completion:
        def __init__(self):
            self.entered = threading.Event()
            self.done = threading.Event()

        def synchronize(self):
            self.entered.set()
            if not self.done.wait(5):
                raise TimeoutError("test copy did not complete")

    completion = Completion()
    completion.done.set()
    slot = Exl3PLESlot(8, completion)
    with pytest.raises(ValueError, match="capacity"):
        slot.acquire(9)
    slot.acquire(8)
    with pytest.raises(RuntimeError, match="reading"):
        slot.acquire(1)
    with pytest.raises(RuntimeError, match="no matching"):
        slot.release()
    slot.ready()
    with pytest.raises(RuntimeError, match="ready"):
        slot.acquire(1)
    completion.done.clear()
    completion.entered.clear()
    slot.release()
    acquired = threading.Event()

    def reuse():
        slot.acquire(1)
        acquired.set()

    worker = threading.Thread(target=reuse)
    worker.start()
    try:
        assert completion.entered.wait(2)
        assert not acquired.is_set()
        with pytest.raises(RuntimeError, match="waiting"):
            slot.acquire(1)
    finally:
        completion.done.set()
        worker.join(5)
    assert acquired.is_set()
    slot.ready()
    slot.release()


def test_exl3_ple_timing_switch_defaults_off(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_EXL3_PLE_TIMING", raising=False)
    assert envs.VLLM_EXL3_PLE_TIMING is False
    monkeypatch.setenv("VLLM_EXL3_PLE_TIMING", "1")
    assert envs.VLLM_EXL3_PLE_TIMING is True


def test_exl3_ple_timing_logs_one_line_per_window():
    messages = []

    class Capture(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    log = logging.getLogger(exl3_ple.__name__)
    handler, level = Capture(logging.INFO), log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        timing = Exl3PLETiming("owner=[0,10) workers=8")
        timing.WINDOW = 4
        timing.flush()  # nothing yet: no line
        for i in range(3):
            timing.add(64, 10_000, 800_000 + 100_000 * i)
        assert messages == []
        timing.add(64, 50_000, 1_200_000)  # fourth: the window is full
        assert len(messages) == 1 and timing.samples == []
        line = messages[0]
        # reads 800/900/1000/1200 us over 4 x 64 rows; waits 10/10/10/50 us
        for part in ("EXL3 PLE timing owner=[0,10) workers=8", "4 prefetches",
                     "rows mean 64.0 max 64", "ids wait us p50 10 p90 38 max 50;",
                     "row reads us p50 950 ", "max 1200 mean 975",
                     "us per row 15.23"):
            assert part in line, (part, line)
        timing.add(512, 1_000, 2_400_000)
        timing.flush()  # a partial window at close
        assert len(messages) == 2 and "1 prefetches" in messages[1]
    finally:
        log.removeHandler(handler)
        log.setLevel(level)


def test_exl3_ple_overlap_switch_defaults_off(monkeypatch):
    import vllm.envs as envs

    monkeypatch.delenv("VLLM_EXL3_PLE_OVERLAP", raising=False)
    assert envs.VLLM_EXL3_PLE_OVERLAP is False
    monkeypatch.setenv("VLLM_EXL3_PLE_OVERLAP", "1")
    assert envs.VLLM_EXL3_PLE_OVERLAP is True


class _Event:
    """A CUDA event stand-in that writes what the host asks of it to a trace;
    CPU tensors finish their work at once."""

    def __init__(self, name, trace):
        self.name, self.trace = name, trace

    def record(self, stream=None):
        self.trace.append(f"record {self.name}")

    def synchronize(self):
        self.trace.append(f"sync {self.name}")


class _Stream(_Event):
    def wait_stream(self, other):
        self.trace.append(f"{self.name} waits {other.name}")

    def wait_event(self, event):
        self.trace.append(f"{self.name} waits {event.name}")


_HEADS, _TOKENS = 2, 8
_REAL_PREADV = os.preadv


def _ssd_embedding(tmp_path, monkeypatch, trace, *, overlap, owner_lo=3):
    """An SSD PLE embedding over a small real table, on the host: streams and
    events trace the host's calls, the decoder records the rows it gets."""
    path, key, _, _ = _packed_ple_file(tmp_path)
    reader = Exl3PLERowReader(str(path), key, owner_lo, 10, retries=0)
    read_rows = reader.read_rows

    def traced_read_rows(*args):
        trace.append("read")
        return read_rows(*args)

    reader.read_rows = traced_read_rows
    current = _Stream("current", trace)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "stream", contextmanager(lambda stream: (yield)))
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    decoded = []

    def decode(packed, ids, bias, offsets, sizes, lo, hi, bits, out):
        trace.append("decode")
        decoded.append((packed.clone(), ids.clone()))
        # Any function of the packed bytes; a poisoned scale gives NaN.
        scale = packed[:, :1].view(torch.float16).float()
        out.copy_((scale * packed[:, 1:].float().sum(-1, keepdim=True)).expand_as(out))

    from vllm.sm80.kernels.exl3 import ops

    monkeypatch.setattr(ops, "ple_decode_out", decode)
    layout = reader.layout
    capacity = _TOKENS * _HEADS
    module = object.__new__(Exl3PLEEmbedding)
    nn.Module.__init__(module)
    module.placement = "ssd"
    module.overlap_reads = overlap
    module._deferred = None
    module.tp_size = module.etp_data_parallel_size = 1
    module.embedding_dim = layout.dim
    module._heads = _HEADS
    module._output_dim = _HEADS * layout.dim
    module._capacity = capacity
    module._reader = reader
    module._workers = 2
    module._pool = ThreadPoolExecutor(max_workers=2)
    module._timing = None
    module._failed_tokens = None
    module.head_bias = torch.zeros(_HEADS, layout.dim, dtype=torch.float16)
    module.head_offsets = torch.tensor(layout.head_offsets)
    module.head_vocab_sizes = torch.tensor(layout.head_vocab_sizes)
    module._stream = _Stream("side", trace)
    module._ids_ready = _Event("ids", trace)
    module._ready = _Event("ready", trace)
    module._released = _Event("released", trace)
    # Zeroed, so a failed row's unread bytes match between modules.
    module._ids = torch.zeros(capacity, dtype=torch.int64)
    module._packed = torch.zeros(capacity, layout.words, dtype=torch.int16)
    module._decoded = torch.zeros(capacity, layout.dim, dtype=torch.bfloat16)
    module._host_ids = torch.zeros(capacity, dtype=torch.int64)
    module._host_rows = torch.zeros(capacity, layout.words, dtype=torch.int16)
    module._host_ids_np = module._host_ids.numpy()
    module._host_rows_np = module._host_rows.numpy()
    module._host_bytes = memoryview(module._host_rows_np).cast("B")
    module._slot = exl3_ple.Exl3PLESlot(capacity, module._released)
    return module, decoded


def _step(module, ids, trace):
    """One forward's PLE part: prefetch, the layer ahead, finalize."""
    module.start_prefetch(None, torch.tensor(ids, dtype=torch.int64))
    trace.append("layer ahead")
    output = module.forward(torch.empty(len(ids), 1))
    return output, module.take_failed_tokens()


def _reads_once(trace):
    """Pool workers each trace their chunk; keep one entry per read."""
    return [e for i, e in enumerate(trace) if e != "read" or trace[i - 1] != "read"]


@pytest.mark.parametrize("overlap", [False, True])
def test_exl3_ple_overlap_keeps_order_and_moves_only_the_layer(
    tmp_path, monkeypatch, overlap
):
    """The deferred step makes the same host calls in the same order; only
    the layer ahead is queued before the ids wait instead of after the decode.
    """
    trace = []
    module, _ = _ssd_embedding(tmp_path, monkeypatch, trace, overlap=overlap)
    try:
        for _ in range(2):  # the second step reuses the slot
            trace.clear()
            module.start_prefetch(None, torch.tensor([[3, 6], [4, 9]]))
            assert module._slot.state == ("reading" if overlap else "ready")
            trace.append("layer ahead")
            module.forward(torch.empty(2, 1))
            assert module._slot.state == "idle"
            assert module.take_failed_tokens().tolist() == [False, False]
            calls = [
                "sync released",  # the slot waits for the previous copy
                "side waits current",
                "record ids",
                "sync ids",
                "read",
                "decode",
                "record ready",
                "current waits ready",
                "record released",
            ]
            calls.insert(3 if overlap else 7, "layer ahead")
            assert _reads_once(trace) == calls
    finally:
        module.close()


def test_exl3_ple_overlap_gives_the_consumer_the_same_bytes(tmp_path, monkeypatch):
    """Rows, ids, outputs and failure masks match bit for bit across token
    counts, unowned and padding rows, a failed row, and arms switched between
    steps as the in-boot A/B does."""
    monkeypatch.setattr(exl3_ple.time, "sleep", lambda seconds: None)
    steps = [
        [[3, 5]],
        [[4, 8], [0, 9], [3, 6]],  # row 0 is unowned, row 9 padding
        [[r % 5, 5 + r % 4] for r in range(_TOKENS)],  # rows 4 and 6 fail below
        [[2, 7]],
    ]
    results = {}
    for arms in ("off", "on", "alternating"):
        trace = []
        (tmp_path / arms).mkdir()
        module, decoded = _ssd_embedding(
            tmp_path / arms, monkeypatch, trace, overlap=arms == "on"
        )
        outputs = []
        try:
            for index, ids in enumerate(steps):
                if arms == "alternating":
                    module.overlap_reads = bool(index % 2)
                if index == 2:
                    _faulty_preadv(
                        monkeypatch,
                        errno.EIO,
                        {4, 6},
                        module._reader.base,
                        module._reader.row_bytes,
                    )
                output, failed = _step(module, ids, trace)
                if index == 2:
                    monkeypatch.setattr(os, "preadv", _REAL_PREADV)
                outputs.append((output.view(torch.int16).clone(), failed.clone()))
        finally:
            module.close()
        results[arms] = (decoded, outputs)
    reference_decoded, reference_outputs = results["off"]
    assert reference_outputs[2][1].tolist() == [
        r % 5 == 4 or 5 + r % 4 == 6 for r in range(_TOKENS)
    ]
    assert not any(failed.any() for _, failed in reference_outputs[:2])
    for arms in ("on", "alternating"):
        decoded, outputs = results[arms]
        assert len(decoded) == len(outputs) == len(steps)
        for (packed, ids), (ref_packed, ref_ids) in zip(decoded, reference_decoded):
            assert torch.equal(packed, ref_packed) and torch.equal(ids, ref_ids)
        for (output, failed), (ref_output, ref_failed) in zip(
            outputs, reference_outputs
        ):
            assert torch.equal(output, ref_output)
            assert torch.equal(failed, ref_failed)


@pytest.mark.parametrize("overlap", [False, True])
def test_exl3_ple_overlap_read_error_fails_the_slot(tmp_path, monkeypatch, overlap):
    """A read that raises fails the slot after every chunk is joined; with the
    overlap it surfaces at finalize instead of at the prefetch."""
    trace = []
    module, decoded = _ssd_embedding(tmp_path, monkeypatch, trace, overlap=overlap)
    joined = []

    def broken(ids, target, begin, end):
        joined.append(begin)
        if begin == 0:
            raise RuntimeError("storage gone")
        return []

    module._reader.read_rows = broken
    ids = torch.tensor([[3, 5], [4, 6], [3, 7], [4, 8]])
    try:
        if overlap:
            module.start_prefetch(None, ids)
            assert joined == [] and module._slot.state == "reading"
            with pytest.raises(RuntimeError, match="storage gone"):
                module.forward(torch.empty(4, 1))
        else:
            with pytest.raises(RuntimeError, match="storage gone"):
                module.start_prefetch(None, ids)
        assert sorted(joined) == [0, 4]  # both chunks ran before the failure
        assert module._slot.state == "failed" and decoded == []
        assert module._deferred is None
        with pytest.raises(RuntimeError, match="slot is failed"):
            module.start_prefetch(None, ids)
        with pytest.raises(RuntimeError, match="no matching"):
            module.forward(torch.empty(4, 1))
    finally:
        module.close()


@pytest.mark.parametrize("overlap", [False, True])
def test_exl3_ple_overlap_cancelled_step_reads_nothing(tmp_path, monkeypatch, overlap):
    """A forward that ends between prefetch and finalize leaves the slot held:
    the next prefetch is refused as before. With the overlap no row was read
    and nothing is queued, so close has nothing to wait for."""
    trace = []
    module, decoded = _ssd_embedding(tmp_path, monkeypatch, trace, overlap=overlap)
    try:
        with pytest.raises(RuntimeError, match="no matching"):
            module.forward(torch.empty(1, 1))  # finalize before any prefetch
        assert module._slot.state == "idle"
        module.start_prefetch(None, torch.tensor([[3, 5]]))
        state = "reading" if overlap else "ready"
        with pytest.raises(RuntimeError, match=f"slot is {state}"):
            module.start_prefetch(None, torch.tensor([[4, 6]]))
        assert ("read" in trace) is not overlap
        assert len(decoded) == (0 if overlap else 1)
        assert module._slot.state == state
    finally:
        module.close()
    assert module._pool is None and module._reader.fd == -1
