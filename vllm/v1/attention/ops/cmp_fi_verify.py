# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer paged prefill as the speculative-decode verify attention (Qwen3.8-27B;
VLLM_CMP_VERIFY_FLASHINFER=1).

On the CMP 170HX the split-KV Triton verify kernel (spec_decode_attn.py) reads
the KV cache at 0.45-0.78 TB/s; FlashInfer's BatchPrefillWithPagedKVCache
kernel (fa2 backend, KV split across CTAs, GQA rows packed) reads it at
0.86-1.24 TB/s for the same 8-token verify blocks.

Constraints of the engine it is fitted to:
  * KV layout: FLASH_ATTN's per-layer cache (blocks, kv_heads, block, 2*D),
    passed as the strided K/V views the backend already splits.
  * Async scheduling: the host knows only an upper bound of each request's KV
    length. plan() is given upper-bound page counts (host scheduling only);
    the device buffers the kernel reads carry the exact length through
    last_page_len = seq_len - (pages - 1) * page_size (may be <= 0).
  * CUDA graphs: one use_cuda_graph wrapper per padded request count, all on
    persistent buffers and one workspace; build() re-plans the wrapper of the
    batch about to run, so a captured graph replays with the new plan.
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

ENABLED = os.environ.get("VLLM_CMP_VERIFY_FLASHINFER", "0") == "1"
# Plan once per step for all KV cache groups (they differ only in their block
# tables): the first group to see a batch plans, the others copy its host plan
# state and fill only their own page lists. All wrappers share one float and
# one int workspace (one plan is live at a time).
SHARE_PLAN = os.environ.get("VLLM_CMP_VERIFY_FI_SHARE_PLAN", "0") == "1"
_PER_WRAPPER = frozenset(
    {
        "_qo_indptr_buf",
        "_paged_kv_indptr_buf",
        "_paged_kv_indices_buf",
        "_paged_kv_last_page_len_buf",
        "_custom_mask_buf",
        "_mask_indptr_buf",
        "_float_workspace_buffer",
        "_int_workspace_buffer",
        "_pin_memory_int_workspace_buffer",
        "_kv_lens_buffer",
    }
)


class _Shared:
    def __init__(self, ws: torch.Tensor):
        self.ws = ws
        self.int_ws = None
        self.pin_ws = None
        self.last: dict[int, tuple] = {}  # nreq -> (plan key, planned wrapper)


MAX_Q_PER_REQ = 16


@triton.jit
def _fill_pages_kernel(
    idx_ptr,
    last_ptr,
    ind_ptr,
    bt_ptr,
    seq_ptr,
    bt_stride,
    page_size,
    nreq,
    BLOCK: tl.constexpr,
):
    # One program: the device page list from the EXACT lengths. The host plan
    # used upper-bound page counts; when the upper bound has crossed into a
    # page the exact length has not reached, FlashInfer misreads a
    # last_page_len <= 0 (verbatim-copy and vision failures at prompts just
    # below a 448-token boundary). Here every request gets ceil(seq / page)
    # pages (>= 1) and a last_page_len in [1, page] (0 only for empty padded
    # requests); indptr is rewritten to match. Total pages <= the planned total.
    start = 0
    for r in range(0, nreq):
        seq = tl.load(seq_ptr + r)
        n = tl.maximum((seq + page_size - 1) // page_size, 1)
        tl.store(ind_ptr + r, start)
        for off in range(0, n, BLOCK):
            p = off + tl.arange(0, BLOCK)
            m = p < n
            tl.store(
                idx_ptr + start + p,
                tl.load(bt_ptr + r * bt_stride + p, mask=m, other=0),
                mask=m,
            )
        tl.store(last_ptr + r, seq - (n - 1) * page_size)
        start += n
    tl.store(ind_ptr + nreq, start)


class CmpFiVerify:
    def __init__(
        self,
        max_reqs: int,
        max_pages: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        sm_scale: float,
        dtype: torch.dtype,
        device: torch.device,
        workspace_mb: int = 64,
        shared: _Shared | None = None,
        ub_slack: int = 7,
    ):
        self.max_reqs, self.max_pages = max_reqs, max_pages
        # host upper bound - exact KV length <= ub_slack (rejected drafts of the
        # previous step: num_speculative_tokens)
        self.ub_slack = ub_slack
        self.shared = shared
        self.hq, self.hkv, self.d, self.ps = (
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
        )
        self.scale, self.dtype = sm_scale, dtype
        self.ws = (
            shared.ws
            if shared is not None
            else torch.empty(workspace_mb << 20, dtype=torch.uint8, device=device)
        )
        self.qo = torch.zeros(max_reqs + 1, dtype=torch.int32, device=device)
        self.ind = torch.zeros(max_reqs + 1, dtype=torch.int32, device=device)
        self.idx = torch.zeros(max_pages, dtype=torch.int32, device=device)
        self.last = torch.zeros(max_reqs, dtype=torch.int32, device=device)
        self.wrappers: dict[int, object] = {}

    def _wrapper(self, nreq: int):
        w = self.wrappers.get(nreq)
        if w is None:
            import flashinfer

            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.ws,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=self.qo[: nreq + 1],
                paged_kv_indptr_buf=self.ind[: nreq + 1],
                paged_kv_indices_buf=self.idx,
                paged_kv_last_page_len_buf=self.last[:nreq],
                backend="fa2",
            )
            sh = self.shared
            if sh is not None:
                if sh.int_ws is None:
                    sh.int_ws, sh.pin_ws = (
                        w._int_workspace_buffer,
                        w._pin_memory_int_workspace_buffer,
                    )
                else:
                    w._int_workspace_buffer = sh.int_ws
                    w._pin_memory_int_workspace_buffer = sh.pin_ws
            self.wrappers[nreq] = w
        return w

    def _do_plan(self, w, qo, ind, total, last_ub, fixed_split_size=-1):
        w.plan(
            qo,
            ind,
            self.idx[:total],
            last_ub,
            self.hq,
            self.hkv,
            self.d,
            self.ps,
            causal=True,
            sm_scale=self.scale,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
            fixed_split_size=fixed_split_size,
        )

    @staticmethod
    def _chunk_size(w) -> int:
        # PrefillPlanInfo: [.., 9] kv_chunk_size_ptr_offset, [14] split_kv; the
        # planner staged the int workspace in pinned host memory before the
        # device copy, so the chosen chunk size is readable without a sync.
        info = w._plan_info
        if not info[14]:
            return 0
        off = int(info[9])
        b = w._pin_memory_int_workspace_buffer[off : off + 4]
        return int.from_bytes(bytes(b.tolist()), "little", signed=True)

    def _ambiguous(self, cs: int, ub: list[int]) -> bool:
        # the kernel counts a request's KV chunks from the exact length, the
        # merge from the planned (upper-bound) length: they must agree for
        # every exact length in [ub - slack, ub]
        lo = self.ub_slack
        for u in ub:
            if u <= 0:
                continue
            if -(-u // cs) != -(-max(u - lo, 1) // cs):
                return True
        return False

    def _plan_consistent(self, w, nreq, qo, ind, total, last_ub, seq_ub_cpu):
        self._do_plan(w, qo, ind, total, last_ub)
        cs = self._chunk_size(w)
        if cs <= 0:
            return
        ub = [int(x) for x in seq_ub_cpu[:nreq].tolist()]
        if not self._ambiguous(cs, ub):
            return
        self.replans = getattr(self, "replans", 0) + 1
        # Only LARGER chunks: a CUDA-graph wrapper's tile grid and int
        # workspace are sized by its capture-time plan, and a smaller chunk
        # size means more KV tiles than that (an illegal-address crash under
        # concurrent load). A chunk >= max(ub) is one chunk per request,
        # which is never ambiguous, so the search always ends.
        # ...and keep KV splitting on (the longest request keeps >= 2 chunks):
        # a plan whose split flag flips lays out the int workspace differently
        # from the one a CUDA graph captured.
        top = (max(ub) - 1) // 2
        cand = cs + 16
        while cand <= top:
            if not self._ambiguous(cand, ub):
                self._do_plan(w, qo, ind, total, last_ub, fixed_split_size=cand)
                return
            cand += 16
        raise RuntimeError("cmp_fi_verify: no unambiguous split KV chunk size")

    def plan(
        self,
        nreq: int,
        qo_cpu: torch.Tensor,
        seq_ub_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
    ):
        """qo_cpu: query_start_loc [nreq + 1] (CPU); seq_ub_cpu: upper bounds of
        the KV lengths [nreq] (CPU); seq_lens: exact KV lengths [nreq] (GPU);
        block_table: [>= nreq, max_blocks] (GPU). Returns the wrapper to run,
        or None when the batch does not fit the persistent buffers."""
        if nreq > self.max_reqs:
            return None
        pages = ((seq_ub_cpu[:nreq].to(torch.int64) + self.ps - 1) // self.ps).clamp_(
            min=1
        )
        if pages.max().item() > block_table.shape[1]:
            return None
        ind = torch.zeros(nreq + 1, dtype=torch.int32)
        torch.cumsum(pages, 0, out=ind[1:])
        total = int(ind[-1])
        if total > self.max_pages:
            return None
        last_ub = (
            (seq_ub_cpu[:nreq].to(torch.int64) - (pages - 1) * self.ps)
            .clamp_(min=1)
            .to(torch.int32)
        )
        w = self._wrapper(nreq)
        qo = qo_cpu[: nreq + 1].to(torch.int32)
        sh = self.shared
        key = (
            (tuple(qo.tolist()), tuple(seq_ub_cpu[:nreq].tolist()))
            if sh is not None
            else None
        )
        m = sh.last.get(nreq) if sh is not None else None
        if m is not None and m[0] == key and m[1] is not w:
            # same batch already planned by another group this step
            src = m[1]
            for k, v in src.__dict__.items():
                if k not in _PER_WRAPPER:
                    w.__dict__[k] = v
            w._qo_indptr_buf.copy_(src._qo_indptr_buf, non_blocking=True)
            w._paged_kv_indptr_buf.copy_(src._paged_kv_indptr_buf, non_blocking=True)
        else:
            self._plan_consistent(w, nreq, qo, ind, total, last_ub, seq_ub_cpu)
            if sh is not None:
                sh.last[nreq] = (key, w)
        # device side: compact page list from the block table, exact lengths
        _fill_pages_kernel[(1,)](
            self.idx,
            self.last,
            self.ind,
            block_table,
            seq_lens,
            block_table.stride(0),
            self.ps,
            nreq,
            BLOCK=512,
        )
        return w


def exact_seq_lens_cpu(seq_lens: torch.Tensor) -> torch.Tensor:
    """seq_lens.cpu() (a GPU sync) where the async scheduler allows it."""
    try:
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed  # type: ignore
    except Exception:  # older layout
        try:
            from vllm.v1.worker.utils import gpu_sync_allowed  # type: ignore
        except Exception:
            gpu_sync_allowed = None
    if gpu_sync_allowed is None:
        return seq_lens.cpu()
    with gpu_sync_allowed():
        return seq_lens.cpu()


_STATE: dict = {}


def get(
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    sm_scale,
    dtype,
    device,
    max_reqs,
    max_model_len,
    group_key=(),
    ub_slack: int = 7,
) -> CmpFiVerify:
    # One state per KV cache group (group_key: its layer names): each group has
    # its own block table, and a plan lives in its wrapper's buffers and
    # workspace, so groups must not share them.
    key = (
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        float(sm_scale),
        dtype,
        str(device),
        tuple(group_key),
    )
    st = _STATE.get(key)
    if st is None:
        max_pages = max_reqs * (-(-max_model_len // page_size) + 2)
        shared = None
        if SHARE_PLAN:
            skey = key[:-1]
            shared = _SHARED.get(skey)
            if shared is None:
                shared = _SHARED[skey] = _Shared(
                    torch.empty(64 << 20, dtype=torch.uint8, device=device)
                )
        st = CmpFiVerify(
            max_reqs,
            max_pages,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            sm_scale,
            dtype,
            device,
            shared=shared,
            ub_slack=ub_slack,
        )
        _STATE[key] = st
    return st


_SHARED: dict = {}
