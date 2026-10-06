# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded compressed PLE storage and selected-row decoding."""

import bisect
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.exl3_utils import (
    PLE_AUXILIARIES,
    TensorHeader,
    ple_layout,
    read_headers,
    read_metadata,
)
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from .ngram_embedding import Qwen4ExpPLEEmbedding, Qwen4ExpPLEEmbeddingMethod

logger = init_logger(__name__)


def pread_exact_into(fd: int, target: memoryview, offset: int) -> None:
    """Fill a byte view without changing the descriptor's seek position."""
    if offset < 0 or offset + len(target) > (1 << 63) - 1:
        raise ValueError("PLE file offset exceeds signed 64-bit range")
    done = 0
    while done < len(target):
        try:
            count = os.preadv(fd, [target[done:]], offset + done)
        except InterruptedError:
            continue
        if count == 0:
            raise EOFError(f"Truncated PLE data at byte {offset + done}")
        done += count


class Exl3PLERowReader:
    """Read owned rows of a validated compressed PLE table into caller storage."""

    def __init__(self, path: str, key: str, owner_lo: int, owner_hi: int):
        self.fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            if not key.endswith(".trellis"):
                raise ValueError("PLE source key must name the packed trellis")
            source = Path(path)
            self.layout = ple_layout(
                read_headers(source),
                read_metadata(source),
                key.removesuffix(".trellis"),
            )
            if not 0 <= owner_lo <= owner_hi <= self.layout.rows:
                raise ValueError("PLE owner range exceeds stored padded rows")
            self.owner_lo, self.owner_hi = owner_lo, owner_hi
            self.base = self.layout.trellis.offset
            self.row_bytes = self.layout.row_bytes
        except BaseException:
            self.close()
            raise

    def read_tensor(self, header: TensorHeader) -> bytearray:
        data = bytearray(header.nbytes)
        pread_exact_into(self.fd, memoryview(data), header.offset)
        return data

    def owns(self, row: int) -> bool:
        if not self.owner_lo <= row < self.owner_hi:
            return False
        offsets = self.layout.head_offsets
        head = bisect.bisect_right(offsets, row) - 1
        return head >= 0 and row < offsets[head] + self.layout.head_vocab_sizes[head]

    def read_rows(
        self, ids, target: memoryview, begin: int = 0, end: int | None = None
    ) -> None:
        """Preserve order/repetitions; mask only nonowned or padding rows."""
        end = len(ids) if end is None else end
        if not 0 <= begin <= end <= len(ids) or len(target) < end * self.row_bytes:
            raise ValueError("PLE row destination is too small")
        zero = bytes(self.row_bytes)
        for index in range(begin, end):
            row = int(ids[index])
            dest = target[index * self.row_bytes : (index + 1) * self.row_bytes]
            if self.owns(row):
                pread_exact_into(self.fd, dest, self.base + row * self.row_bytes)
            else:
                dest[:] = zero

    def close(self) -> None:
        fd = getattr(self, "fd", -1)
        if fd >= 0:
            os.close(fd)
            self.fd = -1

    def __del__(self):
        self.close()


class Exl3PLESlot:
    """The PLE step state: one bounded staging lease whose reuse waits for the
    previous output copy."""

    def __init__(self, capacity, completion):
        self.capacity = capacity
        self.completion = completion
        self.state = "idle"

    def acquire(self, count):
        if self.state != "idle":
            raise RuntimeError(f"PLE prefetch slot is {self.state}")
        if not 0 <= count <= self.capacity:
            raise ValueError("PLE request exceeds configured staging capacity")
        self.state = "waiting"
        try:
            self.completion.synchronize()
        except BaseException:
            self.state = "failed"
            raise
        self.state = "reading"

    def ready(self):
        if self.state != "reading":
            raise RuntimeError("PLE reads must complete before decode readiness")
        self.state = "ready"

    def release(self):
        if self.state != "ready":
            raise RuntimeError("PLE finalize has no matching prefetched rows")
        self.state = "idle"

    def fail(self):
        self.state = "failed"


class Exl3PLEEmbeddingMethod(Qwen4ExpPLEEmbeddingMethod):
    """Keep the table compressed independently of BF16 activations."""

    def __init__(self, quant_config=None):
        from vllm import envs

        self.placement = envs.VLLM_EXL3_PLE_PLACEMENT
        if self.placement not in ("ssd", "ram", "device"):
            raise ValueError("EXL3 PLE placement must be ssd, ram or device")

    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        if params_dtype != torch.bfloat16:
            raise ValueError("EXL3 PLE requires BF16 output activations")
        # A zero-sized storage marker satisfies the embedding module interface.
        # The loader joins the file extent instead of loading a dense parameter.
        layer.register_parameter(
            "weight",
            nn.Parameter(torch.empty(0, dtype=torch.int16), requires_grad=False),
        )

    def process_weights_after_loading(self, layer):
        layer.check_ready()

    def embedding(self, layer, input_):
        raise RuntimeError("EXL3 PLE lookup requires start_prefetch and finalize")

    def dequantize(self, layer, embeddings, output_dtype):
        if output_dtype != torch.bfloat16 or embeddings.dtype != output_dtype:
            raise ValueError("EXL3 PLE decoder output must be BF16")
        return embeddings


class Exl3PLEEmbedding(Qwen4ExpPLEEmbedding):
    """One bounded slot, reused only after its graph-output copy completes."""

    supports_prefetch = True

    def __init__(self, *args, num_ngram_heads=1, max_total_tokens=0, **kwargs):
        super().__init__(
            *args,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            **kwargs,
        )
        if max_total_tokens <= 0:
            raise ValueError("EXL3 PLE needs a positive staging token capacity")
        self.placement = self.embedding_method.placement
        self._heads = num_ngram_heads
        self._capacity = max_total_tokens * self.etp_data_parallel_size * self._heads
        self._output_dim = self._heads * self.embedding_dim
        self._reader = None
        self._pool = None
        self._slot: Exl3PLESlot | None = None

    def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
        raise RuntimeError("EXL3 PLE must never allocate a dense table")

    def check_ready(self):
        if self._slot is None:
            raise RuntimeError("EXL3 PLE file/auxiliary loader join is missing")

    def setup_file(
        self, path, key, head_offsets, head_vocab_sizes, layer_multipliers
    ) -> set[str]:
        """Load auxiliaries exactly once; return consumed source keys.

        The manifest must exclude these keys and the trellis from ordinary
        tensor iteration. Hash buffers belong to the parent n-gram module.
        """
        if self._reader is not None:
            raise RuntimeError("EXL3 PLE file was already set up")
        device = self.weight.device
        if device.type != "cuda" or any(
            t.device != device
            for t in (head_offsets, head_vocab_sizes, layer_multipliers)
        ):
            raise ValueError("PLE hash auxiliaries must be CUDA-resident")
        reader = Exl3PLERowReader(
            path,
            key,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
        )
        self._reader = reader
        try:
            layout = reader.layout
            if (
                layout.rows != self.org_vocab_size
                or layout.dim != self.embedding_dim
                or layout.heads != self._heads
            ):
                raise ValueError("PLE checkpoint rows/dim/heads differ from model")
            tensors = {}
            for name, header in zip(PLE_AUXILIARIES, layout.auxiliaries):
                dtype = torch.float16 if header.dtype == "F16" else torch.int64
                tensors[name] = torch.frombuffer(
                    reader.read_tensor(header), dtype=dtype
                ).reshape(header.shape)
            for name, target in (
                ("head_offsets", head_offsets),
                ("head_vocab_sizes", head_vocab_sizes),
                ("layer_multipliers", layer_multipliers),
            ):
                if tensors[name].shape != target.shape:
                    raise ValueError(f"Invalid PLE auxiliary {name}")
            consumed = {key} | {header.name for header in layout.auxiliaries}
            head_offsets.copy_(tensors["head_offsets"])
            head_vocab_sizes.copy_(tensors["head_vocab_sizes"])
            layer_multipliers.copy_(tensors["layer_multipliers"])
            self.register_buffer("head_bias", tensors["head_bias"].to(device))
            self.head_offsets = head_offsets
            self.head_vocab_sizes = head_vocab_sizes
            self._stream = torch.cuda.Stream(device=device)
            self._ids_ready = torch.cuda.Event()
            self._ready = torch.cuda.Event()
            self._released = torch.cuda.Event()
            self._released.record(torch.cuda.current_stream())
            self._ids = torch.empty(self._capacity, dtype=torch.int64, device=device)
            self._packed = torch.empty(
                self._capacity, layout.words, dtype=torch.int16, device=device
            )
            self._decoded = torch.empty(
                self._capacity, layout.dim, dtype=torch.bfloat16, device=device
            )
            if self.placement == "ssd":
                self._host_ids = torch.empty(
                    self._capacity, dtype=torch.int64, device="cpu", pin_memory=True
                )
                self._host_rows = torch.empty(
                    self._capacity,
                    layout.words,
                    dtype=torch.int16,
                    device="cpu",
                    pin_memory=True,
                )
                self._host_ids_np = self._host_ids.numpy()
                self._host_bytes = memoryview(self._host_rows.numpy()).cast("B")
                self._workers = min(8, os.cpu_count() or 1)
                self._pool = ThreadPoolExecutor(
                    max_workers=self._workers, thread_name_prefix="ple-read"
                )
            else:
                self._local_ids = torch.empty_like(self._ids)
                count = reader.owner_hi - reader.owner_lo
                if count <= 0:
                    raise ValueError("PLE comparison shard must contain rows")
                if self.placement == "ram":
                    if not is_uva_available():
                        raise RuntimeError("RAM PLE comparison requires CUDA UVA")
                    self._table = torch.empty(
                        count,
                        layout.words,
                        dtype=torch.int16,
                        device="cpu",
                        pin_memory=True,
                    )
                    view = memoryview(self._table.numpy()).cast("B")
                    pread_exact_into(
                        reader.fd,
                        view,
                        reader.base + reader.owner_lo * reader.row_bytes,
                    )
                    self._table_view = get_accelerator_view_from_cpu_tensor(self._table)
                else:
                    self._table = torch.empty(
                        count, layout.words, dtype=torch.int16, device=device
                    )
                    block = min(count, self._capacity)
                    staging = torch.empty(
                        block,
                        layout.words,
                        dtype=torch.int16,
                        device="cpu",
                        pin_memory=True,
                    )
                    view = memoryview(staging.numpy()).cast("B")
                    for begin in range(0, count, block):
                        length = min(block, count - begin)
                        pread_exact_into(
                            reader.fd,
                            view[: length * reader.row_bytes],
                            reader.base + (reader.owner_lo + begin) * reader.row_bytes,
                        )
                        self._table[begin : begin + length].copy_(staging[:length])
                    self._table_view = self._table
            self._slot = Exl3PLESlot(self._capacity, self._released)
            logger.info(
                "EXL3 PLE: placement=%s storage=int16 output=bfloat16 "
                "owner=[%d,%d) row_bytes=%d capacity=%d",
                self.placement,
                reader.owner_lo,
                reader.owner_hi,
                reader.row_bytes,
                self._capacity,
            )
            return consumed
        except BaseException:
            reader.close()
            raise

    def close(self):
        if self._slot is not None:
            self._stream.synchronize()
            self._released.synchronize()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._reader is not None:
            self._reader.close()
        self._slot = None

    @eager_break_during_capture
    def start_prefetch(self, hidden_states, ngram_ids):
        self.check_ready()
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("EXL3 PLE requires eager or split CUDA graphs")
        slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
        gathered = self._gather_dp_ids(ngram_ids, slot_size)
        count = gathered.numel()
        self._slot.acquire(count)
        try:
            self._local_tokens = ngram_ids.shape[0]
            self._slot_offset = slot_offset
            self._gathered_tokens = gathered.shape[0]
            stream = self._stream
            stream.wait_stream(torch.cuda.current_stream())
            gathered.record_stream(stream)
            with torch.cuda.stream(stream):
                self._ids[:count].copy_(gathered.reshape(-1))
                if self.placement == "ssd":
                    self._host_ids[:count].copy_(self._ids[:count], non_blocking=True)
                    self._ids_ready.record(stream)
            if self.placement == "ssd":
                self._ids_ready.synchronize()
                ids = self._host_ids_np[:count]
                step = (count + self._workers - 1) // self._workers
                futures = []
                if step:
                    for begin in range(0, count, step):
                        futures.append(
                            self._pool.submit(
                                self._reader.read_rows,
                                ids,
                                self._host_bytes,
                                begin,
                                min(begin + step, count),
                            )
                        )
                # Join every read even if one fails, before the slot can close.
                errors = []
                for future in futures:
                    try:
                        future.result()
                    except BaseException as error:
                        errors.append(error)
                if errors:
                    raise errors[0]
            with torch.cuda.stream(stream):
                packed = self._packed[:count]
                if self.placement == "ssd":
                    packed.copy_(self._host_rows[:count], non_blocking=True)
                else:
                    local = self._local_ids[:count]
                    torch.sub(self._ids[:count], self._reader.owner_lo, out=local)
                    # Invalid indices use row zero; the decoder masks before bias.
                    local.clamp_(0, self._reader.owner_hi - self._reader.owner_lo - 1)
                    torch.index_select(self._table_view, 0, local, out=packed)
                from vllm.model_executor.layers.quantization.exl3_ops import (
                    ple_decode_out,
                )

                ple_decode_out(
                    packed,
                    self._ids[:count],
                    self.head_bias,
                    self.head_offsets,
                    self.head_vocab_sizes,
                    self._reader.owner_lo,
                    self._reader.owner_hi,
                    self._reader.layout.bits,
                    self._decoded[:count],
                )
                self._ready.record(stream)
            self._slot.ready()
        except BaseException:
            self._slot.fail()
            raise

    @eager_break_during_capture
    def _finalize_prefetch(self, output):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("EXL3 PLE finalize requires an eager graph break")
        if self._slot.state != "ready" or output.shape[0] != self._local_tokens:
            raise RuntimeError("PLE finalize has no matching prefetched rows")
        current = torch.cuda.current_stream()
        current.wait_event(self._ready)
        rows = self._decoded[: self._gathered_tokens * self._heads].view(
            self._gathered_tokens, self._heads, self.embedding_dim
        )
        if self.tp_size > 1:
            rows = self.parallel_group.all_reduce(rows)
        rows = self._select_embeddings(rows, self._local_tokens, self._slot_offset)
        output.copy_(rows.flatten(-2))
        self._released.record(current)
        self._slot.release()

    def forward(self, hidden_states):
        output = self._decoded.new_empty((hidden_states.shape[0], self._output_dim))
        self._finalize_prefetch(output)
        return output
