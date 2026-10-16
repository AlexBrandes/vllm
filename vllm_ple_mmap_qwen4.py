"""vllm_ple_mmap_qwen4 — serve Qwen4Exp N-gram (PLE) table from NVMe via mmap.

Why: the n-gram table is 102 GiB in BF16 on this box (44 GiB FP8 elsewhere)
and vLLM keeps it resident (GPU, or pinned host RAM with VLLM_PLE_CPU_OFFLOAD).
Neither fits next to the main model. But a token only ever touches
``ngram_heads`` rows x ``head_dim * itemsize`` bytes of that table, so the
table can live on disk and be served through the page cache — exactly what
llama.cpp does with its GGUF mmap.

How: with VLLM_PLE_MMAP=1 this module patches ``Qwen4ExpNGramEmbedding``:
  * the embedding is a tiny placeholder whose weight is never allocated;
  * ``load_weights`` drops the shard tensors, keeps the global FP8 scale (for
    FP8 checkpoints) and opens ``np.memmap`` views of the checkpoint's PLE
    shards (zero-copy, page-cache backed, MADV_RANDOM).

Two lookup paths:

1. SYNC (production, VLLM_PLE_HOST_IDS unset/0) — the embedding forward runs
   inside an ``@eager_break`` segment: blocking D2H of the GPU n-gram ids,
   32-thread NVMe page-cache gather, H2D into a persistent output buffer.
   Requires breakable PIECEWISE graphs; the D2H is illegal during FULL
   cudagraph capture (this is what killed the 2026-10-14 FULL_AND_PIECEWISE +
   VLLM_PLE_CPU_OFFLOAD arm at capture).

2. HOST IDS (FN7 lab, VLLM_PLE_HOST_IDS=1) — the n-gram row ids are computed
   on the HOST from the worker's own CPU state before the forward runs, the
   rows are gathered from the mmap table into a pinned buffer by a sidecar
   thread, and one non-blocking H2D fills a persistent GPU rows buffer. The
   PLE-layer forward then ONLY returns a view of that persistent buffer: no
   D2H, no host sync, no allocation and no NVMe touch inside the forward, so
   it is legal inside FULL cudagraph capture (with breakable graphs the model
   forward is captured whole; the escape hatch VLLM_PLE_HOST_IDS_EAGER_BREAK=1
   keeps the decorated eager-segment form for bisection).
   The model-state overlay (model_state.py in this patch dir) drives the
   sidecar from ``Qwen4ExpModelState.prepare_inputs``;
   ``prepare_dummy_inputs`` only zeroes rows, so CUDA-graph capture and memory
   profiling never touch NVMe. The host-side batch reconstruction is an exact
   mirror of the worker GPU kernels (``_prepare_prefill_inputs_kernel``,
   ``_combine_sampled_and_draft_tokens_kernel``, ``_prepare_ngram_context``)
   and the n-gram hash is a numpy port of ``_ple_ngram_ids_kernel`` (int64
   wrap, floor-mod) — see the CPU equivalence tests next to this file. MTP
   draft steps never use PLE (the drafter's layer index is
   ``num_hidden_layers + k``, not a PLE layer, and it passes
   ``ngram_context=None``), and only PP stage 0 hosts the single PLE layer
   (ple_layer_ids=[2] -> layer_idx 1), so exactly one lookup per engine step
   moves to the host.

Knobs (env):
  VLLM_PLE_MMAP=1            enable the mmap backend
  VLLM_PLE_MMAP_RANDOM=1     MADV_RANDOM on the table (production state)
  VLLM_PLE_MMAP_WORKERS=32   gather threads (page faults overlap across threads)
  VLLM_PLE_MMAP_CHUNK=2048   rows per gather task
  VLLM_PLE_MMAP_PREWARM=0    1 = stream the whole table once at load
  VLLM_PLE_HOST_IDS=1        FN7 host-side id path (needs the model_state overlay)
  VLLM_PLE_HOST_IDS_TIMEOUT_MS=5000
                             sidecar publish wait before a slow-publish warning;
                             the step keeps waiting for the worker (the gather is
                             NVMe-bound, so re-reading the same rows inline only
                             doubled the I/O while the worker was still running)
  VLLM_PLE_HOST_IDS_HANG_S=300
                             total publish wait before the step fails closed
                             (a hung worker, not a slow disk)
  VLLM_PLE_HOST_IDS_ROWS     rows-buffer capacity in tokens; default
                             max_num_batched_tokens + 64
  VLLM_PLE_HOST_IDS_EAGER_BREAK=1  keep the host-ids forward as an eager break
  LAB_PLE_OUT_ROWS=262144    legacy sync path persistent output rows

Install: the image's ngram_embedding.py ends with
``_ple_mmap_apply(Qwen4ExpNGramEmbedding)`` (image layer), so mounting this
file over /usr/local/lib/python3.12/dist-packages/vllm_ple_mmap_qwen4.py is
enough; the FN7 variants additionally mount the model_state.py overlay.
"""

from __future__ import annotations

import glob
import json
import logging
import mmap as _mmap
import os
import re
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger("vllm.ple_mmap")

ENV_ENABLE = "VLLM_PLE_MMAP"
ENV_RANDOM = "VLLM_PLE_MMAP_RANDOM"
ENV_HOST_IDS = "VLLM_PLE_HOST_IDS"
ENV_HOST_IDS_TIMEOUT_MS = "VLLM_PLE_HOST_IDS_TIMEOUT_MS"
ENV_HOST_IDS_HANG_S = "VLLM_PLE_HOST_IDS_HANG_S"
ENV_HOST_IDS_ROWS = "VLLM_PLE_HOST_IDS_ROWS"
ENV_HOST_IDS_EAGER_BREAK = "VLLM_PLE_HOST_IDS_EAGER_BREAK"
_FP8_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    # 本机 W4A16-AutoRound 的 PLE 表是 BF16 (非 fp8 分片): 原样透传, 不做反量化
    "BF16": torch.bfloat16,
    "F16": torch.float16,
}
_NON_FP8_DTYPES = {"BF16", "F16"}


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes")


def enabled() -> bool:
    return _env_bool(ENV_ENABLE)


def host_ids_enabled() -> bool:
    return _env_bool(ENV_HOST_IDS)


def _max_num_batched_tokens() -> int:
    from vllm.config import get_current_vllm_config

    return int(get_current_vllm_config().scheduler_config.max_num_batched_tokens)


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

    def _gather_uniq(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ids int64[N] -> (unique-sorted ids, per-input index into them)."""
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
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
        return out, inverse

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """ids: int64 [N] global row ids -> uint8 [N, row_bytes] (a fresh array)."""
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.empty((0, self.row_bytes), dtype=np.uint8)
        out, inverse = self._gather_uniq(ids)
        return out[inverse]

    def gather_into(self, ids: np.ndarray, out: np.ndarray) -> None:
        """Same as ``gather`` but writes [N, row_bytes] into ``out`` (writable,
        e.g. a pinned-buffer numpy view). No allocation when N == 0."""
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        if out.shape[1:] != (self.row_bytes,) or out.shape[0] < ids.size:
            raise ValueError(f"gather_into out shape {out.shape} for {ids.size} ids")
        if ids.size == 0:
            return
        uniq_rows, inverse = self._gather_uniq(ids)
        out[: ids.size] = uniq_rows[inverse]

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
# FN7: host-side n-gram ids (numpy port of _ple_ngram_ids_kernel) and the
# host-side batch reconstruction (mirror of the worker input-prep kernels).
# Pure numpy; unit-tested on CPU against the in-tree reference
# ``Qwen4ExpNGramEmbedding.compute_ngram_ids`` CPU branch.
# --------------------------------------------------------------------------- #
def host_ngram_ids(
    tokens: np.ndarray,
    qsl: np.ndarray,
    ctx: np.ndarray,
    multipliers: np.ndarray,
    sizes: np.ndarray,
    offsets: np.ndarray,
    eos_token_id: int,
    heads_per_ngram: int,
) -> np.ndarray:
    """Per-token n-gram row ids, int64 [T, ngram_heads].

    Mirrors ``vllm/models/qwen4_exp/nvidia/ops/ple.py::_ple_ngram_ids_kernel``:
    per head column, ``mixed`` starts at ``token * multipliers[0]``; shift
    ``s`` in ``1..ctx_len`` XORs ``pred_token(s) * multipliers[s]`` into heads
    with ``ngram_order > s``; predecessors come from the batch tokens while
    inside the request's chunk, else from ``ctx`` (the worker's per-request
    n-gram context); once an EOS is crossed walking backwards, older terms are
    EOS. Final ids are ``floor(mod(mixed, size)) + offset`` (int64 products
    wrap like the kernel; the floor fixup matches the kernel's
    ``remainders < 0`` branch and ``torch.remainder``).
    """
    tokens = np.ascontiguousarray(tokens, dtype=np.int64).reshape(-1)
    qsl = np.ascontiguousarray(qsl, dtype=np.int64).reshape(-1)
    ctx = np.ascontiguousarray(ctx, dtype=np.int64)
    multipliers = np.ascontiguousarray(multipliers, dtype=np.int64).reshape(-1)
    sizes = np.ascontiguousarray(sizes, dtype=np.int64).reshape(-1)
    offsets = np.ascontiguousarray(offsets, dtype=np.int64).reshape(-1)
    if ctx.ndim != 2:
        raise ValueError(f"ctx must be 2D, got {ctx.shape}")
    num_reqs = qsl.shape[0] - 1
    ctx_len = ctx.shape[1] if ctx.shape[0] else ctx.shape[1]
    heads = sizes.shape[0]
    if heads != ctx_len * heads_per_ngram:
        raise ValueError(
            f"heads {heads} != ctx_len {ctx_len} * heads_per_ngram "
            f"{heads_per_ngram}: buffer/layout mismatch"
        )
    num_tokens = tokens.shape[0]
    if num_tokens == 0:
        return np.zeros((0, heads), dtype=np.int64)

    pos = np.arange(num_tokens, dtype=np.int64)
    req = np.searchsorted(qsl, pos, side="right") - 1
    np.clip(req, 0, max(num_reqs - 1, 0), out=req)
    chunk_pos = pos - qsl[req]

    ngram_order = (np.arange(heads, dtype=np.int64) // heads_per_ngram) + 2
    with np.errstate(over="ignore"):
        mixed = tokens[:, None] * multipliers[0]  # [T, H] broadcast, int64 wrap
        crossed = np.zeros(num_tokens, dtype=bool)
        eos = np.int64(eos_token_id)
        for shift in range(1, ctx_len + 1):
            in_step = chunk_pos >= shift
            cand = np.empty(num_tokens, dtype=np.int64)
            cand[in_step] = tokens[pos[in_step] - shift]
            tail = ~in_step
            if tail.any():
                ctx_col = ctx_len - shift + chunk_pos[tail]
                cand[tail] = ctx[req[tail], ctx_col]
            cand = np.where(crossed, eos, cand)
            crossed = crossed | (cand == eos)
            term = cand[:, None] * multipliers[shift]
            mixed = mixed ^ np.where((ngram_order > shift)[None, :], term, 0)
        rem = np.remainder(mixed, sizes[None, :])
        return rem + offsets[None, :]


class HostIdsParams:
    """CPU snapshot of the hash buffers + geometry of one Qwen4ExpNGramEmbedding."""

    def __init__(self, ngram_module: nn.Module, bonus_tokens: int) -> None:
        if int(bonus_tokens) != 1:
            raise RuntimeError(
                f"PLE host-ids supports num_new_sampled_tokens_per_step=1, got "
                f"{bonus_tokens}"
            )
        self.ngram_size = int(ngram_module.ngram_size)
        self.heads_per_ngram = int(ngram_module.heads_per_ngram)
        self.ctx_len = self.ngram_size - 1
        self.ngram_heads = self.ctx_len * self.heads_per_ngram
        self.eos_token_id = int(ngram_module.eos_token_id)
        self.multipliers = self._cpu_i64(ngram_module.layer_multipliers)
        self.sizes = self._cpu_i64(ngram_module.ngram_heads_vocab_sizes)
        self.offsets = self._cpu_i64(ngram_module.ngram_heads_offsets)
        self.bonus_tokens = int(bonus_tokens)

    @staticmethod
    def _cpu_i64(buf: torch.Tensor) -> np.ndarray:
        return buf.detach().to("cpu", dtype=torch.int64).numpy().reshape(-1).copy()


def _is_real_batch(input_batch: object, req_states: object) -> bool:
    """False for dummy/profiling batches (make_dummy fakes req_ids)."""
    num_reqs = int(input_batch.num_reqs)  # type: ignore[attr-defined]
    if num_reqs == 0:
        return False
    req_ids = getattr(input_batch, "req_ids", None)
    if not req_ids or len(req_ids) != num_reqs:
        return False
    return req_ids[0] in req_states.req_id_to_index  # type: ignore[attr-defined]


def reconstruct_host_batch(
    params: HostIdsParams,
    input_batch: object,
    req_states: object,
    nc_np: np.ndarray,
    draft_np: np.ndarray,
    last_sampled_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Rebuild (tokens[T], qsl[nr+1], ctx[nr, ctx_len], T, padded_T) exactly as
    the worker's GPU input-prep kernels present the batch to the model.

    Mirrors:
      * ``_prepare_prefill_inputs_kernel``: prefill rows read
        ``all_token_ids[slot, num_computed + j]``;
      * ``_combine_sampled_and_draft_tokens_kernel``: for decode-ish rows
        (``seq_len > prefill_len``) the bonus-token slot
        ``seq_len - num_logits`` gets ``last_sampled_tokens`` when its position
        is ``>= prefill_len`` (else the prompt-tail ``all_token_ids`` value
        stays), and the last ``num_draft`` slots get ``draft_tokens``;
      * ``Qwen4ExpModelState._prepare_ngram_context``: ctx =
        ``all_token_ids[slot, nc - ctx_len : nc]`` (EOS-padded at the start of
        the request) — i.e. on a prefix-cache hit the context comes from the
        request's stored token vector, the same source the GPU path reads.

    Raises on any layout the mirror does not cover (fail closed: under FULL
    cudagraph variants there is no capture-safe GPU fallback).
    """
    num_reqs = int(input_batch.num_reqs)  # type: ignore[attr-defined]
    all_tbl = getattr(req_states.all_token_ids, "_uva_buf", None)  # type: ignore[attr-defined]
    if all_tbl is None:
        raise RuntimeError(
            "PLE host-ids requires all_token_ids to be UVA-backed "
            "(StagedWriteTensor uva_instead_of_gpu); host view unavailable"
        )
    all_np = all_tbl.np  # int32 [max_num_reqs, max_model_len], pinned
    max_model_len = int(all_np.shape[1])

    slots = np.asarray(input_batch.idx_mapping_np[:num_reqs], dtype=np.int64)  # type: ignore[attr-defined]
    qsl = np.asarray(input_batch.query_start_loc_np[: num_reqs + 1], dtype=np.int64)  # type: ignore[attr-defined]
    cu_logits = np.asarray(input_batch.cu_num_logits_np[: num_reqs + 1], dtype=np.int64)  # type: ignore[attr-defined]
    prefill_len = np.asarray(req_states.prefill_len.np, dtype=np.int64)[slots]  # type: ignore[attr-defined]
    nc = np.asarray(nc_np, dtype=np.int64)[slots]
    if np.any(slots >= nc_np.shape[0]):
        raise RuntimeError("PLE host-ids: request slot out of mirror range")

    real_tokens = int(qsl[num_reqs])
    padded_tokens = int(input_batch.num_tokens_after_padding)  # type: ignore[attr-defined]
    qlens = np.diff(qsl)
    num_logits = np.diff(cu_logits)
    bonus = params.bonus_tokens
    num_draft = num_logits - bonus
    if num_draft.shape[0] != num_reqs or np.any(num_draft < 0):
        raise RuntimeError("PLE host-ids: inconsistent num_logits layout")

    tokens = np.zeros(max(real_tokens, 0), dtype=np.int64)
    ctx = np.full((num_reqs, params.ctx_len), params.eos_token_id, dtype=np.int64)
    for r in range(num_reqs):
        s = int(slots[r])
        nc_r = int(nc[r])
        qlen = int(qlens[r])
        nd = int(num_draft[r])
        pl = int(prefill_len[r])
        if qlen == 0:
            continue
        seq_len = nc_r + qlen
        if nc_r < 0 or seq_len > max_model_len:
            raise RuntimeError(
                f"PLE host-ids: request {r} window [{nc_r},{seq_len}) exceeds "
                f"max_model_len {max_model_len}"
            )
        # n-gram context: tokens just before the chunk.
        lo = max(0, nc_r - params.ctx_len)
        nvalid = nc_r - lo
        if nvalid > 0:
            ctx[r, params.ctx_len - nvalid :] = all_np[s, lo:nc_r].astype(np.int64)
        start = int(qsl[r])
        if seq_len <= pl:
            # Pure prefill row: kernel reads every token from all_token_ids.
            tokens[start : start + qlen] = all_np[s, nc_r : nc_r + qlen].astype(
                np.int64
            )
            continue
        # Decode-ish row.
        nlg = nd + bonus
        j0 = qlen - nlg  # local index of the bonus-token slot
        if j0 < 0:
            raise RuntimeError(
                f"PLE host-ids: request {r} qlen {qlen} < num_logits {nlg}"
            )
        if j0 > 0 and nc_r + j0 > pl:
            raise RuntimeError(
                f"PLE host-ids: request {r} leading rows past prefill_len with "
                "qlen > num_logits (stale input_ids slots; unsupported layout)"
            )
        if j0 > 0:
            tokens[start : start + j0] = all_np[s, nc_r : nc_r + j0].astype(np.int64)
        pos0 = nc_r + j0
        if pos0 >= pl:
            tokens[start + j0] = int(last_sampled_np[s])
        else:
            tokens[start + j0] = int(all_np[s, pos0])
        if nd > 0:
            if nd > draft_np.shape[1]:
                raise RuntimeError(
                    f"PLE host-ids: request {r} has {nd} drafts, mirror holds "
                    f"{draft_np.shape[1]}"
                )
            tokens[start + qlen - nd : start + qlen] = draft_np[s, :nd]
    return tokens, qsl, ctx, real_tokens, padded_tokens


# --------------------------------------------------------------------------- #
# FN7: host-ids sidecar service
# --------------------------------------------------------------------------- #
class PleHostIdsService:
    """Persistent GPU rows buffer + pinned transport + one gather worker.

    Per real step (driven from ``Qwen4ExpModelState.prepare_inputs``):
      main thread: record an event ``E`` on the compute stream (covers the
        previous step's sampler/post_update GPU writes into the UVA token
        table, num_computed_tokens, draft_tokens), queue the job, then wait
        for the worker's publish and enqueue one non-blocking pinned->GPU
        copy into ``rows_gpu`` plus a zero fill of the padded row region.
      worker thread: ``E.synchronize()`` (GPU idle by then in the sync loop),
        D2H the tiny request-state mirrors, reconstruct the batch on host,
        compute ids (``host_ngram_ids``) and gather rows into the pinned rows
        buffer, publish.

    Safety:
      * The single pinned rows buffer is overwritten by the worker only after
        ``E.synchronize()``, and ``E`` of step k+1 is recorded on the compute
        stream after step k's H2D copy of the same buffer was enqueued -> the
        worker never races the in-flight H2D.
      * A slow publish (NVMe contention) is waited out, never re-gathered
        inline; a worker that has not published after
        ``VLLM_PLE_HOST_IDS_HANG_S`` fails the step closed.
      * Capture/profiling (dummy) steps never launch the worker; their rows
        are zero-filled.
    """

    def __init__(
        self,
        table: MmapPleTable,
        device: torch.device,
        ngram_heads: int,
        head_dim: int,
        cap_tokens: int,
        timeout_s: float,
    ) -> None:
        self.table = table
        self.device = device
        self.heads = int(ngram_heads)
        self.head_dim = int(head_dim)
        self.out_dim = self.heads * self.head_dim
        self.cap_tokens = int(cap_tokens)
        self.timeout_s = float(timeout_s)
        self.rows_gpu = torch.zeros(
            (self.cap_tokens, self.out_dim),
            dtype=table.torch_dtype,
            device=device,
        )
        rows_bytes = self.cap_tokens * self.heads * table.row_bytes
        self._pinned_rows = torch.zeros(rows_bytes, dtype=torch.uint8, pin_memory=True)
        self._pinned_rows_np = self._pinned_rows.numpy().reshape(-1, table.row_bytes)
        # Request-state mirrors, sized on first real step.
        self._mirror_cap = 0
        self._pinned_nc: torch.Tensor | None = None
        self._pinned_draft: torch.Tensor | None = None
        self._pinned_last: torch.Tensor | None = None
        self._nc_np: np.ndarray | None = None
        self._draft_np: np.ndarray | None = None
        self._last_np: np.ndarray | None = None

        self._params: HostIdsParams | None = None
        self._params_src: nn.Module | None = None
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._job: dict | None = None
        self._publish_seq = 0
        self._job_seq = 0
        self._worker_error: BaseException | None = None
        self._stop = threading.Event()
        self._slow_count = 0
        self.hang_s = max(float(os.environ.get(ENV_HOST_IDS_HANG_S, "300")),
                          self.timeout_s)
        self._worker = threading.Thread(
            target=self._worker_loop, name="ple-host-ids-gather", daemon=True
        )
        self._worker.start()
        logger.info(
            "PLE host-ids: service ready (heads=%d head_dim=%d cap_tokens=%d "
            "rows_gpu=%.1f MiB pinned=%.1f MiB)",
            self.heads,
            self.head_dim,
            self.cap_tokens,
            self.rows_gpu.numel() * self.rows_gpu.element_size() / 2**20,
            rows_bytes / 2**20,
        )

    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        with self._cv:
            self._stop.set()
            self._cv.notify_all()

    def view(self, num_tokens: int) -> torch.Tensor:
        if num_tokens > self.cap_tokens:
            raise RuntimeError(
                f"PLE host-ids rows buffer too small: {num_tokens} > "
                f"{self.cap_tokens} (raise {ENV_HOST_IDS_ROWS})"
            )
        return self.rows_gpu[:num_tokens]

    def zero_rows(self, num_tokens: int) -> None:
        n = min(int(num_tokens), self.cap_tokens)
        if n > 0:
            self.rows_gpu[:n].zero_()

    def _snapshot_params(self, ngram_module: nn.Module, bonus: int) -> HostIdsParams:
        if self._params_src is not ngram_module or self._params is None:
            self._params = HostIdsParams(ngram_module, bonus)
            self._params_src = ngram_module
            logger.info(
                "PLE host-ids: params ngram_size=%d heads_per_ngram=%d "
                "heads=%d eos=%d",
                self._params.ngram_size,
                self._params.heads_per_ngram,
                self._params.ngram_heads,
                self._params.eos_token_id,
            )
        return self._params

    def _ensure_mirrors(self, req_states: object) -> None:
        cap = int(req_states.max_num_reqs)  # type: ignore[attr-defined]
        k = max(1, int(req_states.num_speculative_steps))  # type: ignore[attr-defined]
        if self._mirror_cap >= cap and _shape_ok(self._pinned_draft, k):
            return
        self._mirror_cap = cap
        self._pinned_nc = torch.zeros(cap, dtype=torch.int32, pin_memory=True)
        self._pinned_draft = torch.zeros((cap, k), dtype=torch.int64, pin_memory=True)
        self._pinned_last = torch.zeros(cap, dtype=torch.int64, pin_memory=True)
        self._nc_np = self._pinned_nc.numpy()
        self._draft_np = self._pinned_draft.numpy()
        self._last_np = self._pinned_last.numpy()

    # ------------------------------------------------------------------ #
    def prepare(
        self,
        ngram_module: nn.Module,
        input_batch: object,
        req_states: object,
        bonus_tokens: int,
    ) -> None:
        padded = int(input_batch.num_tokens_after_padding)  # type: ignore[attr-defined]
        if padded > self.cap_tokens:
            raise RuntimeError(
                f"PLE host-ids: padded tokens {padded} > rows buffer "
                f"{self.cap_tokens}; raise {ENV_HOST_IDS_ROWS}"
            )
        params = self._snapshot_params(ngram_module, bonus_tokens)
        if not _is_real_batch(input_batch, req_states):
            # Dummy/DP/profiling step: no NVMe, deterministic zero rows.
            self.zero_rows(padded)
            return
        if getattr(input_batch, "max_query_len", None) is not None:
            raise RuntimeError(
                "PLE host-ids: adaptive_verification present; host mirror of "
                "the GPU draft reallocation is not implemented (fail closed)"
            )
        self._ensure_mirrors(req_states)

        with self._cv:
            if self._job is not None and self._publish_seq != self._job_seq:
                raise RuntimeError(
                    "PLE host-ids: gather worker still busy from a previous "
                    "step (stalled NVMe?); failing closed"
                )
            if self._worker_error is not None:
                err = self._worker_error
                self._worker_error = None
                raise RuntimeError(f"PLE host-ids: previous gather failed: {err}") from err
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self._job_seq += 1
            self._job = {
                "seq": self._job_seq,
                "event": event,
                "batch": input_batch,
                "req_states": req_states,
                "params": params,
            }
            self._cv.notify_all()

        t0 = time.monotonic()
        if not self._wait_publish(self.timeout_s):
            self._slow_count += 1
            logger.warning(
                "PLE host-ids: worker publish slower than %.1fs (NVMe gather "
                "contention?); still waiting (occurrence %d)",
                self.timeout_s,
                self._slow_count,
            )
            if not self._wait_publish(self.hang_s - self.timeout_s):
                raise RuntimeError(
                    f"PLE host-ids: gather worker has not published after "
                    f"{self.hang_s:.0f}s (hung); failing closed"
                )
            logger.warning(
                "PLE host-ids: slow publish completed after %.1fs",
                time.monotonic() - t0,
            )
        real_tokens = int(input_batch.num_tokens)  # type: ignore[attr-defined]
        if real_tokens > self.cap_tokens:
            raise RuntimeError(
                f"PLE host-ids: {real_tokens} real tokens > buffer "
                f"{self.cap_tokens}"
            )
        rows = real_tokens * self.heads
        src_np = self._pinned_rows_np

        if rows > 0:
            src_t = torch.frombuffer(src_np[:rows].reshape(-1), dtype=torch.uint8)
            dst = self.rows_gpu[:real_tokens].reshape(-1).view(torch.uint8)
            dst.copy_(src_t, non_blocking=True)
        if padded > real_tokens:
            self.rows_gpu[real_tokens:padded].zero_()

    def _wait_publish(self, timeout: float) -> bool:
        """Wait (seq-matched) for the worker to publish this job.

        Returns False on timeout; raises if the worker reported a failure.
        """
        seq = self._job_seq
        t0 = time.monotonic()
        with self._cv:
            while self._publish_seq != seq:
                remaining = timeout - (time.monotonic() - t0)
                if remaining <= 0:
                    return False
                self._cv.wait(min(0.005, remaining))
            self._job = None
            err = self._worker_error
            self._worker_error = None
            if err is not None:
                raise RuntimeError(f"PLE host-ids: gather failed: {err}") from err
            return True

    def _mirror_compute(self, input_batch, req_states, params) -> np.ndarray:
        """D2H the tiny request-state mirrors, reconstruct, compute row ids.

        Runs on the gather worker after its event synchronize (previous-step
        GPU writes into the UVA token table / num_computed_tokens /
        draft_tokens / last_sampled_tokens are then complete) and inline on
        the main thread after ``event.synchronize()`` in the fallback.
        """
        assert self._pinned_nc is not None
        self._pinned_nc.copy_(req_states.num_computed_tokens.gpu)
        self._pinned_draft.copy_(req_states.draft_tokens)
        self._pinned_last.copy_(req_states.last_sampled_tokens.reshape(-1))
        tokens, qsl, ctx, _T, _P = reconstruct_host_batch(
            params, input_batch, req_states, self._nc_np, self._draft_np, self._last_np
        )
        return host_ngram_ids(
            tokens,
            qsl,
            ctx,
            params.multipliers,
            params.sizes,
            params.offsets,
            params.eos_token_id,
            params.heads_per_ngram,
        )

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                while self._job is None and not self._stop.is_set():
                    self._cv.wait(0.1)
                job = self._job
                if job is None:
                    continue
                self._job = None
            try:
                # Threads do not inherit inference mode; the mirror buffers were
                # allocated inside it, so in-place copies must run inside it too.
                with torch.inference_mode():
                    job["event"].synchronize()
                    ids = self._mirror_compute(
                        job["batch"], job["req_states"], job["params"]
                    )
                    rows = int(ids.shape[0]) * self.heads
                    self.table.gather_into(ids.reshape(-1), self._pinned_rows_np[:rows])
                with self._cv:
                    self._publish_seq = job["seq"]
                    self._cv.notify_all()
            except BaseException as exc:  # noqa: BLE001 — surfaced on main thread
                logger.exception("PLE host-ids: worker gather failed")
                with self._cv:
                    if self._publish_seq != job["seq"]:
                        self._worker_error = exc
                        self._publish_seq = job["seq"]
                    self._cv.notify_all()


def _shape_ok(t: torch.Tensor | None, k: int) -> bool:
    return t is not None and t.shape[1] >= k


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
    shard_cols = 0
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
    """Install the NVMe mmap backend for stable vllm's Qwen4Exp PLE table."""
    if not enabled():
        return

    import importlib

    module = importlib.import_module(cls.__module__)
    base_cls = module.Qwen4ExpPLEEmbedding
    eager_break = module.eager_break_during_capture
    use_host_ids = host_ids_enabled()

    class Qwen4ExpPLEMmapEmbedding(base_cls):
        supports_prefetch = False

        def __init__(self, *args, prefix: str, **kwargs) -> None:
            self._mmap_prefix = prefix
            self.table: MmapPleTable | None = None
            self._host_service: PleHostIdsService | None = None
            # Capture geometry for the host-ids service before the base
            # __init__ drops these kwargs.
            self._ple_ngram_heads = int(kwargs.get("num_ngram_heads", 1) or 1)
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
            # Legacy sync path (also the host-ids-disabled fallback):
            # blocking D2H of ids -> NVMe page-cache gather -> H2D into a
            # persistent output buffer (breakable-graph eager segments must
            # write into a persistent buffer; a fresh return tensor leaves
            # downstream graph segments reading the capture-time address).
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
            n = output.shape[0]
            buf = getattr(self, "_lab_ple_out", None)
            if buf is None:
                cap = int(os.environ.get("LAB_PLE_OUT_ROWS", "262144"))
                buf = torch.empty((max(cap, n), self.embedding_dim), dtype=table.torch_dtype,
                                  device=ngram_ids.device)
                self._lab_ple_out = buf
            if n > buf.shape[0]:
                raise RuntimeError(f"PLE lab buffer too small: {n} > {buf.shape[0]}")
            buf[:n].copy_(output)
            return buf[:n].reshape(*ngram_ids.shape, self.embedding_dim)

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
            if use_host_ids:
                if self.etp_data_parallel_size != 1 or self.tp_size != 1:
                    raise RuntimeError(
                        "PLE host-ids requires ETP/TP size 1; use pipeline parallelism only"
                    )
                # NOTE: production resolves async_scheduling=True for
                # method="mtp" (EagleModelTypes). That is SAFE for this path:
                # every state read (num_computed_tokens.gpu / draft_tokens /
                # last_sampled_tokens / the UVA token table) happens after an
                # event synchronized on the compute stream, i.e. the worker
                # sees exactly the bytes the GPU input-prep kernels of THIS
                # step see. Nothing here uses the optimistic host mirrors
                # (num_computed_tokens_np).
                cap_tokens = _env_int(
                    ENV_HOST_IDS_ROWS, _max_num_batched_tokens() + 64
                )
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
                self._host_service = PleHostIdsService(
                    self.table,
                    device,
                    self._ple_ngram_heads,
                    self.embedding_dim,
                    cap_tokens,
                    _env_int(ENV_HOST_IDS_TIMEOUT_MS, 5000) / 1000.0,
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

    def patched_forward(self, hidden_states, input_ids, query_start_loc, ngram_context):
        embedding = self.ngram_embedding
        service = getattr(embedding, "_host_service", None)
        if service is not None:
            # FN7 host-ids path: rows were staged before the forward. Pure
            # persistent-buffer view: capture-safe inside FULL graphs, no
            # D2H/H2D/alloc/NVMe work.
            n = input_ids.reshape(-1).shape[0]
            if n != hidden_states.shape[0]:
                raise RuntimeError(
                    f"PLE host-ids token mismatch {n} vs {hidden_states.shape[0]}"
                )
            return service.view(n)
        # Production path (host-ids disabled): identical to the image code.
        if embedding.supports_prefetch:
            return embedding(hidden_states)
        ngram_ids = self.compute_ngram_ids(input_ids, query_start_loc, ngram_context)
        return self.ngram_embedding(ngram_ids).flatten(-2)

    cls.__init__ = patched_init
    cls.load_weights = patched_load_weights
    # Production parity: with host-ids OFF this module must behave EXACTLY
    # like the production runtime patch, which never touches cls.forward
    # (the eager break lives on the embedding class forward + the
    # vllm::ple_mmap_lookup split op). Only the host-ids path replaces it.
    if use_host_ids:
        cls.forward = (
            eager_break(patched_forward)
            if _env_bool(ENV_HOST_IDS_EAGER_BREAK)
            else patched_forward
        )
    logger.info(
        "PLE mmap: installed Qwen4Exp NVMe backend (host-ids=%s)",
        "on" if use_host_ids else "off",
    )


__all__ = [
    "MmapPleTable",
    "HostIdsParams",
    "PleHostIdsService",
    "apply",
    "enabled",
    "host_ids_enabled",
    "host_ngram_ids",
    "reconstruct_host_batch",
    "parse_safetensors_header",
    "_find_shards",
]
