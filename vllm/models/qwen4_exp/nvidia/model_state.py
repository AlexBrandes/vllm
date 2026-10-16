# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState


class _Qwen4ExpModelStateBase(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        # PLE layer 2 is assigned to the first stage by the production PP layout.
        self.uses_ngram_embedding = (
            bool(config.ple_layer_ids) and get_pp_group().is_first_rank
        )
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        # PLE runs inside captured regions, so these buffers keep a fixed shape
        # and address as the active request count changes between replays.
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_reqs_padded = input_batch.num_reqs_after_padding
        context = self.ngram_context[:num_reqs_padded]
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc[: num_reqs_padded + 1]
        query_start_loc.copy_(input_batch.query_start_loc[: num_reqs_padded + 1])
        # A replayed CUDA graph reads the request layout through the view it was
        # captured with, which can be longer than this batch's layout. Entries
        # past num_reqs_padded would otherwise keep offsets from an earlier batch;
        # the n-gram id kernel binary-searches this array, so a stale
        # (non-monotonic) tail maps real tokens to phantom requests and corrupts
        # their PLE ids. Pad the tail with the batch end: extra requests are empty.
        tail = self.ple_query_start_loc[num_reqs_padded + 1 :]
        if tail.numel():
            tail.copy_(query_start_loc[-1:].expand(tail.numel()))
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=self._prepare_ngram_context(input_batch, req_states),
        )
        return model_inputs

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc = self.ple_query_start_loc[: num_reqs + 1]
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1:])

        ngram_context = self.ngram_context[:num_reqs]
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        return model_inputs


__all__ = ["Qwen4ExpModelState"]


# ============================================================================
# FN7 overlay (lab): host-side PLE n-gram ids.
#
# Everything ABOVE is the image's Qwen4ExpModelState verbatim (class renamed
# to _Qwen4ExpModelStateBase; image sha256 4a26632e…). This subclass only
# adds the drive call for the PLE host-ids gather service implemented in
# vllm_ple_mmap_qwen4 (mounted module). It activates ONLY when
# VLLM_PLE_HOST_IDS=1 and the mmap embedding loaded with a host service;
# with the env unset this class behaves exactly like the base.
# ============================================================================
import importlib as _importlib  # noqa: E402
import logging as _logging  # noqa: E402

_ple_log = _logging.getLogger("vllm.fn7.model_state")
_ple_mod = None
try:
    _ple_mod = _importlib.import_module("vllm_ple_mmap_qwen4")
except Exception as _e:  # module absent: host-ids cannot be enabled
    _ple_log.debug("PLE host-ids: module unavailable (%s)", _e)


class Qwen4ExpModelState(_Qwen4ExpModelStateBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self._ple_host_svc = None
        self._ple_host_mod = None
        if not getattr(self, "uses_ngram_embedding", False):
            return  # PP stage without the PLE layer
        ngram_mods = [
            m
            for m in self.model.modules()
            if getattr(m, "ngram_embedding", None) is not None
            and hasattr(m, "compute_ngram_ids")
        ]
        if not ngram_mods:
            return
        if _ple_mod is None or not _ple_mod.host_ids_enabled():
            return  # production sync path stays active
        for mod in ngram_mods:
            svc = getattr(mod.ngram_embedding, "_host_service", None)
            if svc is not None:
                self._ple_host_svc = svc
                self._ple_host_mod = mod
                break
        if self._ple_host_svc is None:
            raise RuntimeError(
                "VLLM_PLE_HOST_IDS=1 but the PLE mmap embedding has no host "
                "service (check the vllm_ple_mmap_qwen4 mount and "
                "VLLM_PLE_MMAP=1)"
            )
        bonus = int(self.num_new_sampled_tokens_per_step)
        if bonus != 1:
            raise RuntimeError(
                f"PLE host-ids: num_new_sampled_tokens_per_step={bonus}, "
                "mirror supports 1 (MTP) only"
            )
        _ple_log.info(
            "PLE host-ids: bound service to %s", type(self._ple_host_mod).__name__
        )

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if self._ple_host_svc is not None:
            # Stage rows_gpu for THIS step before any graph replay/forward.
            # Dummy/profiling batches are detected inside prepare() and only
            # zero the rows; real batches must never touch NVMe here (worker
            # thread does that after its event).
            self._ple_host_svc.prepare(
                self._ple_host_mod,
                input_batch,
                req_states,
                self.num_new_sampled_tokens_per_step,
            )
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if self._ple_host_svc is not None:
            # CUDA-graph capture: forward reads rows_gpu[:num_tokens]; keep
            # zeros there (no NVMe, deterministic capture).
            self._ple_host_svc.zero_rows(num_tokens)
        return model_inputs
