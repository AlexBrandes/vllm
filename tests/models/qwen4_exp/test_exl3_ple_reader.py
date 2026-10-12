# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The EXL3 PLE row reader: positional reads, retries, truncation, read
faults, failed-token masks, page-cache residency, slot reuse and the
prefetch timing summary."""

import errno
import json
import logging
import mmap
import os
import struct
import threading
from itertools import accumulate

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.common import exl3_ple
from vllm.models.qwen4_exp.common.exl3_ple import (
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
