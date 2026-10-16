"""vllm_ple_mmap_qwen4 — serve Qwen4Exp N-gram (PLE) table from NVMe via mmap.

Why: the 51B-parameter n-gram table is 44 GiB in FP8 and vLLM keeps it resident
(GPU, or pinned host RAM with VLLM_PLE_CPU_OFFLOAD). On a DGX Spark / GX10 the
host and the GPU share one 121 GiB pool, so neither fits next to the 78 GiB main
model. But a token only ever touches 16 rows x 160 bytes of that table, so the
table can live on disk and be served through the page cache — exactly what
llama.cpp does with its GGUF mmap.

How: with VLLM_PLE_MMAP=1 this module patches ``Qwen3_8FlashNextNGramEmbedding``:
  * ``__init__`` swaps the 44/95 GiB ``VocabParallelEmbedding`` for a tiny
    placeholder whose ``forward(ids)`` gathers rows from ``np.memmap`` views of the
    checkpoint's ``model-plefp8-*.safetensors`` shards (zero-copy, page-cache backed);
  * ``load_weights`` drops the 128 shard tensors on the floor, keeps the global FP8
    ``weight_scale`` (as ``_offload_weight_scale``, which the untouched
    ``Qwen3_8FlashNextPLELayer._dequantize_embeddings`` already consumes) and opens
    the memmaps.
  * With ``VLLM_PLE_CPU_OFFLOAD=1``, vLLM's PLE worker starts the mmap gather
    before the GPU reaches the PLE layer, writes into a small pinned buffer,
    and copies the result on a dedicated CUDA stream.  This keeps the table on
    NVMe while overlapping most of the per-step lookup with GPU work.
  * Without CPU offload, ``forward_impl`` (hashing + lookup) is wrapped in a custom op
    ``vllm::ple_mmap_lookup`` so that (a) torch.compile treats it as opaque — the
    stock version trips an Inductor int64 indexing assert on sm_121 — and (b) it can
    be listed in ``-cc.splitting_ops`` and run OUTSIDE piecewise CUDA graphs: the
    gather is CPU work + a pageable H2D copy, which cannot live inside a capture.
    Use ``-cc.cudagraph_mode=PIECEWISE`` (not FULL*) with the splitting op list in
    serve-flashnext-vllm.sh, or ``--enforce-eager``.
Nothing else in vLLM changes: the n-gram hashing, the short-conv, the dequant path
are the stock ones.

Knobs (env):
  VLLM_PLE_MMAP=1            enable
  VLLM_PLE_MMAP_WORKERS=32   gather threads (page faults overlap across threads)
  VLLM_PLE_MMAP_CHUNK=2048   rows per gather task
  VLLM_PLE_MMAP_PREWARM=0    1 = stream the whole table once at load to fill the
                             page cache with whatever memory is free (harmless,
                             evictable; ~10 s at 4.7 GB/s)
  VLLM_PLE_CPU_OFFLOAD=1      use vLLM's async PLE worker + pinned H2D path

Install: the Dockerfile copies this file next to vllm and appends
``_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)`` to the end of
``vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py``. See the repo README.
"""

from __future__ import annotations

import glob
import json
import logging
import mmap as _mmap
import os
import re
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger("vllm.ple_mmap")

ENV_ENABLE = "VLLM_PLE_MMAP"
ENV_RANDOM = "VLLM_PLE_MMAP_RANDOM"
_FP8_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    # 本机 W4A16-AutoRound 的 PLE 表是 BF16 (非 fp8 分片): 原样透传, 不做反量化
    "BF16": torch.bfloat16,
    "F16": torch.float16,
}
_NON_FP8_DTYPES = {"BF16", "F16"}


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "0").lower() in ("1", "true", "yes")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _apply_madv_random(handler: object) -> None:
    """把 PLE 表的 mmap 标成随机访问, 关掉内核预读.

    (2026-09-16 加) PLE 表 (102 GB 级) 的访问是稀疏随机行 gather —— 每次只碰
    几个 4K 页, 顺序预读读进来的邻居页几乎永远用不上, 纯浪费 NVMe 带宽和
    page cache 空间。而权重加载 (169 GB) 是纯顺序读, 恰恰需要大预读。

    内核的 queue/read_ahead_kb 是**整盘队列级**参数, 两个负载共用同一块
    nvme0n1, 无法按文件/分区区分 (nvme0n1p2 与 p3 共享一个 request_queue)。
    所以这里用 per-VMA 的 MADV_RANDOM, 精确只关这张表, 整盘预读保持
    2048 KB 让加载跑满带宽。

    设 VLLM_PLE_MMAP_RANDOM=0 可关掉本行为。
    """
    if os.environ.get(ENV_RANDOM, "1").lower() in ("0", "false", "no"):
        return
    if handler is None:
        return
    try:
        handler.madvise(_mmap.MADV_RANDOM)
    except Exception:
        logger.warning(
            "PLE mmap: madvise(MADV_RANDOM) 失败, 退回内核预读", exc_info=True
        )


# --------------------------------------------------------------------------- #
# safetensors header parsing (no dependency on the safetensors package: we need
# raw file offsets, which its Python API does not expose)
# --------------------------------------------------------------------------- #
def parse_safetensors_header(path: str) -> tuple[dict, int]:
    """Return (header_dict, data_start_offset) of a safetensors file."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


class MmapPleTable:
    """Row gather over a table split into ``split_ngram_parts`` shard files.

    ``shards``: {shard_index: (path, absolute_byte_offset, rows)}. Shard ``i``
    holds global rows ``[i*shard_size, i*shard_size + rows)`` (vLLM's
    ``copy_ple_embedding_shard_`` layout).
    """

    def __init__(
        self,
        shards: dict[int, tuple[str, int, int]],
        shard_size: int,
        row_bytes: int,
        torch_dtype: torch.dtype,
        workers: int = 32,
        chunk: int = 2048,
    ) -> None:
        if not shards:
            raise ValueError("no PLE shards")
        self.shard_size = int(shard_size)
        self.row_bytes = int(row_bytes)
        self.torch_dtype = torch_dtype
        self.chunk = max(1, int(chunk))
        self.paths: list[str | None] = [None] * (max(shards) + 1)
        self.mm: list[np.memmap | None] = [None] * (max(shards) + 1)
        self.rows_total = 0
        for idx, (path, offset, rows) in shards.items():
            self.paths[idx] = path
            self.mm[idx] = np.memmap(
                path, dtype=np.uint8, mode="r", offset=offset, shape=(rows, row_bytes)
            )
            _apply_madv_random(getattr(self.mm[idx], "_mmap", None))
            self.rows_total += rows
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """ids: int64 [N] global row ids -> uint8 [N, row_bytes] (a fresh array)."""
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.empty((0, self.row_bytes), dtype=np.uint8)
        # Dedupe + sort: repeated n-grams are common, and sorted rows improve
        # locality inside a shard.
        uniq, inverse = np.unique(ids, return_inverse=True)
        if uniq[0] < 0 or uniq[-1] >= self.shard_size * len(self.mm):
            raise IndexError(
                f"PLE row id out of range: [{uniq[0]}, {uniq[-1]}] "
                f"for {self.rows_total} rows"
            )
        shard = uniq // self.shard_size
        local = uniq - shard * self.shard_size
        out = np.empty((uniq.size, self.row_bytes), dtype=np.uint8)

        bounds = np.flatnonzero(np.diff(shard)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [uniq.size]))
        tasks: list[tuple[int, int, int]] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            si = int(shard[s])
            for c in range(s, e, self.chunk):
                tasks.append((si, c, min(c + self.chunk, e)))

        def run(task: tuple[int, int, int]) -> None:
            si, a, b = task
            mm = self.mm[si]
            if mm is None:
                raise IndexError(f"PLE shard {si} missing")
            # Fancy indexing on a memmap: page faults do the I/O; NumPy releases
            # the GIL for the copy, so tasks overlap across threads.
            out[a:b] = mm[local[a:b]]

        if len(tasks) == 1:
            run(tasks[0])
        else:
            for _ in self.pool.map(run, tasks):
                pass
        return out[inverse]

    def prewarm(self) -> None:
        """Stream every shard once so the page cache holds as much as it can."""
        block = 64 << 20
        for path, mm in zip(self.paths, self.mm):
            if path is None or mm is None:
                continue
            start = mm.offset
            end = start + mm.shape[0] * mm.shape[1]
            with open(path, "rb", buffering=0) as f:
                pos = start
                while pos < end:
                    n = f.readinto(bytearray(min(block, end - pos)))  # noqa: F841
                    if not n:
                        break
                    pos += n


# --------------------------------------------------------------------------- #
# Placeholder that stands in for VocabParallelEmbedding
# --------------------------------------------------------------------------- #
class _MmapNgramEmbedding(nn.Module):
    """Duck-types the bits of VocabParallelEmbedding the PLE code reads.

    No ``weight`` attribute on purpose: ``Qwen3_8FlashNextPLELayer`` then falls
    back to ``ple_embedding._offload_weight_scale`` for the FP8 scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.org_vocab_size = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.table: MmapPleTable | None = None
        self._zeros_dtype = torch.bfloat16

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        table = self.table
        if table is None:
            # Weights never loaded (e.g. --load-format dummy): keep the plumbing
            # alive with zeros so kernel tests can run without the 44 GiB table.
            return torch.zeros(
                (*ids.shape, self.embedding_dim),
                dtype=self._zeros_dtype,
                device=ids.device,
            )
        ids_np = ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)
        rows = table.gather(ids_np)  # uint8 [N, row_bytes], fresh & writable
        out = torch.from_numpy(rows).view(table.torch_dtype)
        out = out.to(ids.device, non_blocking=True)
        return out.reshape(*ids.shape, self.embedding_dim)


# --------------------------------------------------------------------------- #
# Patch
# --------------------------------------------------------------------------- #
def _find_shards(
    model_path: str, layer_idx: int
) -> tuple[dict[int, tuple[str, int, int]], str | None, tuple[str, int, int] | None]:
    """Locate ``layers.<idx>.ple.ple_embedding.ngram_embedding.shard_N.weight``.

    Returns (shards, dtype_str, scale_entry) where scale_entry is
    (path, abs_offset, nbytes) of ``ngram_embedding.weight_scale`` or None.
    """
    shard_re = re.compile(
        rf"layers\.{layer_idx}\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
    )
    scale_re = re.compile(
        rf"layers\.{layer_idx}\.ple\.ple_embedding\.ngram_embedding\.weight_scale$"
    )
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        files = sorted(
            {
                os.path.join(model_path, fn)
                for name, fn in weight_map.items()
                if shard_re.search(name) or scale_re.search(name)
            }
        )
    else:
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    shards: dict[int, tuple[str, int, int]] = {}
    dtype_str: str | None = None
    scale_entry: tuple[str, int, int] | None = None
    for path in files:
        header, data_start = parse_safetensors_header(path)
        for name, meta in header.items():
            m = shard_re.search(name)
            if m:
                start, end = meta["data_offsets"]
                rows, cols = meta["shape"]
                if dtype_str is None:
                    dtype_str = meta["dtype"]
                elif meta["dtype"] != dtype_str:
                    raise ValueError("PLE shards have mixed dtypes")
                if end - start != rows * cols * _itemsize(dtype_str):
                    raise ValueError(f"PLE shard {name}: size/shape mismatch")
                shards[int(m.group(1))] = (path, data_start + start, rows)
                shard_cols = cols
            elif scale_re.search(name):
                start, end = meta["data_offsets"]
                scale_entry = (path, data_start + start, end - start, meta["dtype"])  # type: ignore[assignment]
    if shards:
        # Return cols through dtype_str consumer; keep it simple: stash on dict.
        shards["__cols__"] = shard_cols  # type: ignore[index]
    return shards, dtype_str, scale_entry


def _itemsize(dtype_str: str) -> int:
    return {
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "U8": 1,
        "I8": 1,
        "BF16": 2,
        "F16": 2,
        "F32": 4,
    }[dtype_str]


def _read_scale(entry: tuple) -> torch.Tensor:
    path, offset, nbytes, dtype_str = entry
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(nbytes)
    if dtype_str == "F32":
        return torch.tensor(struct.unpack("<f", raw[:4])[0], dtype=torch.float32)
    if dtype_str == "BF16":
        u16 = struct.unpack("<H", raw[:2])[0]
        return torch.tensor(u16 << 16, dtype=torch.int32).view(torch.float32).squeeze()
    if dtype_str == "F16":
        return torch.frombuffer(bytearray(raw[:2]), dtype=torch.float16).clone().squeeze()
    raise ValueError(f"unsupported weight_scale dtype {dtype_str}")



def apply(cls: type) -> None:
    """Install the NVMe mmap backend for stable vLLM's Qwen4Exp PLE table."""
    if not enabled():
        return

    import importlib

    module = importlib.import_module(cls.__module__)
    base_cls = module.Qwen4ExpPLEEmbedding
    device_cls = module.Qwen4ExpPLEDeviceEmbedding
    eager_break = module.eager_break_during_capture

    class Qwen4ExpPLEMmapEmbedding(base_cls):
        supports_prefetch = False

        def __init__(self, *args, prefix: str, **kwargs) -> None:
            self._mmap_prefix = prefix
            self.table: MmapPleTable | None = None
            super().__init__(*args, prefix=prefix, **kwargs)

        def allocate_embedding_weight(
            self,
            num_embeddings: int,
            embedding_dim: int,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            del num_embeddings
            return torch.empty((1, embedding_dim), dtype=dtype)

        def start_prefetch(
            self,
            hidden_states: torch.Tensor,
            ngram_ids: torch.Tensor,
        ) -> None:
            del hidden_states, ngram_ids

        @eager_break
        def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
            if self.etp_data_parallel_size != 1 or self.tp_size != 1:
                raise RuntimeError(
                    "PLE mmap requires ETP/TP size 1; use pipeline parallelism only"
                )
            table = self.table
            if table is None:
                raise RuntimeError("PLE mmap table was not initialized from checkpoint shards")
            ids_np = ngram_ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)
            # vLLM poisons dummy-profile token buffers with 0x7fff bytes. The
            # resident embedding path never observes those rows, but an eager
            # mmap lookup does. Map only that explicit profiler sentinel to row
            # zero and zero its output; preserve hard failures for every other
            # out-of-range ID.
            profiler_sentinel = np.int64(0x7FFF7FFF7FFF7FFF)
            sentinel_mask = ids_np == profiler_sentinel
            if sentinel_mask.any():
                ids_np = ids_np.copy()
                ids_np[sentinel_mask] = 0
            rows = table.gather(ids_np)
            if sentinel_mask.any():
                rows[sentinel_mask] = 0
            output = torch.from_numpy(rows).view(table.torch_dtype)
            output = output.to(ngram_ids.device, non_blocking=True)
            return output.reshape(*ngram_ids.shape, self.embedding_dim)

        def open_table(self, model_path: str) -> None:
            layer_match = re.search(r"layers\.(\d+)\.ple\.ple_embedding", self._mmap_prefix)
            if layer_match is None:
                raise RuntimeError(
                    f"PLE mmap: cannot determine layer from prefix {self._mmap_prefix!r}"
                )
            layer_idx = int(layer_match.group(1))
            shards, dtype_str, scale_entry = _find_shards(model_path, layer_idx)
            cols = shards.pop("__cols__", None)
            if not shards or dtype_str is None or cols is None:
                raise RuntimeError(f"PLE mmap: no checkpoint shards found for layer {layer_idx}")
            if cols != self.embedding_dim:
                raise RuntimeError(
                    f"PLE mmap: shard width {cols} != embedding_dim {self.embedding_dim}"
                )
            if dtype_str not in _FP8_DTYPES:
                raise RuntimeError(f"PLE mmap: unsupported shard dtype {dtype_str}")
            if dtype_str not in _NON_FP8_DTYPES and scale_entry is None:
                raise RuntimeError("PLE mmap: FP8 shards are missing weight_scale")
            shard_size = (self.org_vocab_size + len(shards) - 1) // len(shards)
            self.table = MmapPleTable(
                shards,
                shard_size,
                cols * _itemsize(dtype_str),
                _FP8_DTYPES[dtype_str],
                workers=_env_int("VLLM_PLE_MMAP_WORKERS", 32),
                chunk=_env_int("VLLM_PLE_MMAP_CHUNK", 2048),
            )
            if os.environ.get("VLLM_PLE_MMAP_PREWARM", "0").lower() in (
                "1",
                "true",
                "yes",
            ):
                self.table.prewarm()
            logger.info(
                "PLE mmap: layer=%d dtype=%s rows=%d shards=%d",
                layer_idx,
                dtype_str,
                self.table.rows_total,
                len(shards),
            )

    original_init = cls.__init__
    original_load_weights = cls.load_weights

    def patched_init(self, *args, **kwargs):
        current_device = module.Qwen4ExpPLEDeviceEmbedding
        current_host = module.Qwen4ExpPLEPinnedHostEmbedding
        module.Qwen4ExpPLEDeviceEmbedding = Qwen4ExpPLEMmapEmbedding
        module.Qwen4ExpPLEPinnedHostEmbedding = Qwen4ExpPLEMmapEmbedding
        try:
            original_init(self, *args, **kwargs)
        finally:
            module.Qwen4ExpPLEDeviceEmbedding = current_device
            module.Qwen4ExpPLEPinnedHostEmbedding = current_host

    def patched_load_weights(self, weights):
        regular = []
        for name, tensor in weights:
            if name.startswith("ngram_embedding.shard_") and name.endswith(".weight"):
                continue
            regular.append((name, tensor))
        loaded = original_load_weights(self, regular)
        from vllm.config import get_current_vllm_config

        model_path = get_current_vllm_config().model_config.model
        self.ngram_embedding.open_table(model_path)
        loaded.add("ngram_embedding.weight")
        return loaded

    cls.__init__ = patched_init
    cls.load_weights = patched_load_weights
    logger.info("PLE mmap: installed Qwen4Exp NVMe backend")
