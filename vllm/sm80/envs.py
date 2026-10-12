# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment variables of the CMP 170HX (sm_80) fork.

``vllm/envs.py`` calls :func:`register_env_vars` at the end of its import, so
every name below is an ordinary vLLM environment variable in every process:
``envs.X``, ``envs.is_set``, ``envs.compile_factors`` and the unknown-variable
check in ``Platform.validate_environ`` all see it. The table lives here so the
upstream file carries two lines instead of the declarations.
"""

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type and default of each variable in FORK_ENV_VARS. Readers use
    # ``vllm.envs.X``, which resolves through vllm.envs.__getattr__.
    VLLM_AMPERE_MARLIN_BUILD: bool = False
    VLLM_CUSTOM_ALLREDUCE_PCIE_P2P: bool = False
    VLLM_CUSTOM_ALLREDUCE_ALGO: str = ""
    VLLM_CUSTOM_ALLREDUCE_FLAGS: bool = False
    VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES: int = 262144
    VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S: float = 60.0
    VLLM_GLM5_PREFILL_KERNELS: bool = False
    VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE: bool = False
    VLLM_GLM5_PREFILL_MIN_TOKENS: int = 512
    VLLM_INDEXER_DECODE_RAW_K: bool = False
    VLLM_INDEXER_DSA_REF: bool = False
    VLLM_GLM5_SPARSE_MLA_MIN_CTX_MULT: float = 2.0
    VLLM_GLM5_SMLA_PREFILL_PRED_LOAD: bool = False
    VLLM_AMPERE_KDA_PREFILL_WIDE: bool = False
    VLLM_STATE_INDEX_CHECK: bool = False
    VLLM_AMPERE_KDA_PREFILL_NARROW: bool = False
    VLLM_AMPERE_DECODE_KERNELS: bool = False
    VLLM_GLM5_DECODE_MOE_ROUTING: bool = True
    VLLM_GLM5_DECODE_MHC_V2: bool = False
    VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS: int = 32
    VLLM_MOE_MASK_PADDING_MAX_TOKENS: int = 8
    VLLM_MOE_DECODE_ROUTE_V2: bool = False
    VLLM_GLM5_DECODE_MOE_ROUTE_V2_MAX_TOKENS: int = 32
    VLLM_GLM5_DECODE_KDA_WIDE: bool = True
    VLLM_AMPERE_KDA_FOLD: bool = False
    VLLM_AMPERE_KDA_RECOVER_OFF: bool = False
    VLLM_FLA_PIN_AUTOTUNE: bool = False
    VLLM_AMPERE_DECODE_IDX_GLUE: bool = False
    VLLM_AMPERE_KDA_SMALL_STACK: bool = False
    VLLM_GLM5_DECODE_IDX_GLUE_PARTS: str = "weights,glue,fwht,cache,moesum"
    VLLM_MOE_DETERMINISTIC_ALIGN: int = 0
    VLLM_GLM5_TOPK_SORTED: bool = False
    VLLM_GLM5_TOPK_TIEFIX: bool = False
    VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS: int = 0
    VLLM_MOE_MASK_PADDING: bool = False
    VLLM_GLM5_MOE_ROUTE_V2_MASK: bool = True
    VLLM_THIN_GEMM: bool = False
    VLLM_GLM5_DRAFTER_ROPE_FIT: bool = False
    VLLM_TOOL_CHOICE_NONE_MASK: bool = False
    VLLM_DFLASH_BOUNDARY_CACHE: bool = False
    VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE: str = ""
    VLLM_GLM5_DRAFTER_SELECTOR_SHARD: bool = False
    VLLM_MEM_ATTRIBUTION: bool = False
    VLLM_KV_MAMBA_INFLIGHT_STATES: bool = False
    VLLM_THIN_GEMM_MAX_TOKENS: int = 32
    VLLM_THIN_GEMM_CONCAT: bool = False
    VLLM_SHARED_EXPERT_REORDER: bool = False
    VLLM_PP_SPREAD_DECODES: bool = False
    VLLM_PP_PACKED_HOP: bool = False
    VLLM_PP_HOP_NO_METADATA: bool = False
    VLLM_PP_SPLIT_DRAFT_EVENT: bool = False
    VLLM_PP_DRAFT_TAIL_STAGE: int = -1
    VLLM_GLM5_PP_FOLD_DRAFT_FC: bool = False
    VLLM_AMPERE_MARLIN_PREFILL_WIDE: bool = False
    VLLM_GLM5_PP_MARLIN_PREFILL_MIN_TOKENS: int = 384
    VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED: bool = True
    VLLM_AMPERE_MARLIN_PREFILL_NARROW: bool = False
    VLLM_GLM5_TP4_MARLIN_PREFILL_MIN_TOKENS: int = 384
    VLLM_AMPERE_MARLIN_DECODE_CUDA: bool = False
    VLLM_AMPERE_MARLIN_DECODE_V2: bool = False
    VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS: bool = False
    VLLM_GLM5_MARLIN_DECODE_PP_MULTI: bool = False
    VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS: str = "9-32"
    VLLM_HOST_ALLREDUCE: bool = False
    VLLM_HOST_ALLREDUCE_MAX_SIZE: int = 512 * 1024
    VLLM_EXL3: bool = False
    VLLM_EXL3_PLE_PLACEMENT: str = "ssd"
    VLLM_QWEN_QSA_TOPK_SORTED: bool = True
    VLLM_EXL3_PLE_COLD: bool = False
    VLLM_EXL3_PLE_READ_RETRIES: int = 3
    VLLM_QWEN4_ROUTER_FP32: bool = False
    VLLM_QWEN4_MTP_PER_STREAM_NORM: bool = False

# Same logger name as vllm/envs.py, so the retired-name warnings keep their
# logger.
logger = logging.getLogger("vllm.envs")


def _env_with_choices(*args: Any, **kwargs: Any) -> Callable[[], str | None]:
    """``vllm.envs.env_with_choices``, imported when a getter runs: this module
    is imported while ``vllm.envs`` is still loading."""
    from vllm.envs import env_with_choices

    return env_with_choices(*args, **kwargs)


def _glm5_route_v2_max_tokens() -> int:
    """VLLM_GLM5_DECODE_MOE_ROUTE_V2_MAX_TOKENS, rejected above 64: the fused
    router (vllm/sm80/kernels/moe/route.py) has no path for more rows."""
    value = int(os.getenv("VLLM_GLM5_DECODE_MOE_ROUTE_V2_MAX_TOKENS", "32"))
    if value > 64:
        raise ValueError(
            f"VLLM_GLM5_DECODE_MOE_ROUTE_V2_MAX_TOKENS={value} exceeds 64, the "
            "most rows the sm_80 fused MoE router handles"
        )
    return value


def _custom_allreduce_algo() -> str:
    """Mirror of the ``std::getenv`` parse in ``csrc/custom_all_reduce.cuh``.

    The C++ side compares the raw bytes against four spellings and throws on
    anything else, so this neither strips nor lowercases. Unset is the empty
    string, which is what the kernel's own size heuristic means.
    """
    raw = os.getenv("VLLM_CUSTOM_ALLREDUCE_ALGO")
    if raw is None:
        return ""
    if raw not in ("1stage", "oneshot", "2stage", "twoshot"):
        raise ValueError(
            f"Invalid VLLM_CUSTOM_ALLREDUCE_ALGO: {raw}. "
            "Valid values: 1stage, oneshot, 2stage, twoshot"
        )
    return raw


FORK_ENV_VARS: dict[str, Callable[[], Any]] = {
    # If set to 1, build the optional sm_80 Marlin decode and prefill library
    # (_ampere_marlin_C); see docs/features/quantization/README.md.
    "VLLM_AMPERE_MARLIN_BUILD": lambda: (
        os.getenv("VLLM_AMPERE_MARLIN_BUILD", "0") == "1"
    ),
    # Let the custom all-reduce treat *PCIe* peer-to-peer as "fully
    # connected". Upstream only counts NVLink, so on a PCIe-only node with
    # more than two GPUs CustomAllreduce disables itself before it ever runs
    # the P2P probe. With this set, every pair must report NVML
    # P2P_CAPS_INDEX_READ and _WRITE OK (or peer access both ways under
    # torch) -- and the gpu_p2p_access_check probe, which is the only check
    # that moves real bytes, becomes mandatory regardless of
    # VLLM_SKIP_P2P_CHECK. It is also the switch that hands the fast path
    # from the host-staged all-reduce to the device-memory one: see
    # vllm/sm80/distributed/host_shm_all_reduce.py.
    # Off by default; a driver that advertises peer access it cannot route
    # turns a wrong answer here into a hang, so opt in only after measuring.
    "VLLM_CUSTOM_ALLREDUCE_PCIE_P2P": lambda: bool(
        int(os.getenv("VLLM_CUSTOM_ALLREDUCE_PCIE_P2P", "0"))
    ),
    # Which CustomAllreduce kernel runs: unset (the default) leaves the
    # in-kernel size heuristic alone, "1stage"/"oneshot" and
    # "2stage"/"twoshot" force one. Nothing in Python consumes it -- it is
    # read with std::getenv inside csrc/custom_all_reduce.cuh -- and
    # upstream declares it nowhere, so an operator who sets it (serve.sh
    # does, to 2stage) gets one 'Unknown vLLM environment variable
    # detected' per worker at start. It is declared here for that reason
    # alone. The accessor mirrors the C++ parse exactly: an exact byte
    # compare against the four spellings, no case folding and no
    # whitespace stripping, raising on anything else the way the kernel
    # throws at its first all-reduce.
    "VLLM_CUSTOM_ALLREDUCE_ALGO": lambda: _custom_allreduce_algo(),
    # Flags-in-data two-shot all-reduce for bf16 messages up to
    # VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES at TP2 or TP4 over PCIe peer-to-peer
    # (vllm/sm80/distributed/custom_all_reduce_flags.py). Bitwise identical to
    # the kernel it replaces: the two-stage kernel at TP4, upstream's one-stage
    # (or forced two-stage) kernel at TP2. Off unless set.
    "VLLM_CUSTOM_ALLREDUCE_FLAGS": lambda: bool(
        int(os.getenv("VLLM_CUSTOM_ALLREDUCE_FLAGS", "0"))
    ),
    "VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES": lambda: int(
        os.getenv("VLLM_CUSTOM_ALLREDUCE_FLAGS_MAX_BYTES", "262144")
    ),
    # Longest a flags-in-data all-reduce waits for a peer before it records
    # the overrun and traps (the process then exits with the record logged).
    "VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S": lambda: float(
        os.getenv("VLLM_CUSTOM_ALLREDUCE_FLAGS_WAIT_S", "60")
    ),
    # Opt in to the sm_80 prefill kernels (the mHC pre-norm GEMM in
    # vllm/sm80/kernels/mhc/ and the sparse-MLA prefill in
    # vllm/sm80/kernels/attention/;
    # GA100-class parts only; measured on CMP 170HX, 70 SMs). Default OFF: with
    # this unset no such kernel is imported and every call site keeps the
    # upstream code path exactly.
    "VLLM_GLM5_PREFILL_KERNELS": lambda: bool(
        int(os.getenv("VLLM_GLM5_PREFILL_KERNELS", "0"))
    ),
    # Sparse-MLA prefill with all 64 heads on one card (pipeline parallel,
    # TP=1): a Gluon kernel that gathers each cache row once for all heads and
    # compacts the valid index slots before the loop
    # (vllm/sm80/kernels/attention/sparse_prefill_mla_pp.py). Taken only on sm_80 with
    # 64 heads, bf16, the 512-wide NoPE layout, 2176 index slots and 1..2312
    # query rows; everything else keeps the Triton kernel. Needs
    # VLLM_GLM5_PREFILL_KERNELS=1 as well. Off by default.
    "VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE": lambda: bool(
        int(os.getenv("VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE", "0"))
    ),
    # Token threshold. Below it the upstream kernels win (they are tuned for
    # decode shapes) and every captured CUDA graph keeps the upstream kernel.
    "VLLM_GLM5_PREFILL_MIN_TOKENS": lambda: int(
        os.getenv("VLLM_GLM5_PREFILL_MIN_TOKENS", "512")
    ),
    # DSA indexer decode logits (vllm/sm80/kernels/attention/triton_mqa_logits.py,
    # Triton path) on sm_80: dequantize the e4m3 K tiles by bit placement
    # (raw bf16 = value * 2^-120, 2^60 folded into Q and the head weights)
    # instead of through fp32. Bitwise identical logits on sm_80, where the
    # tensor cores were verified to keep the bf16 subnormal inputs this
    # produces; other devices keep the fp32 dequant. Off by default (fp32
    # dequant); the GLM launcher sets 1.
    "VLLM_INDEXER_DECODE_RAW_K": lambda: bool(
        int(os.getenv("VLLM_INDEXER_DECODE_RAW_K", "0"))
    ),
    # Debug: 1 swaps the DSA indexer's Triton fp8 MQA-logits kernels
    # (vllm/sm80/kernels/attention/triton_mqa_logits.py) for their pure-torch
    # references.
    "VLLM_INDEXER_DSA_REF": lambda: os.getenv("VLLM_INDEXER_DSA_REF", "0") == "1",
    # Sparse MLA only: require seq_kv >= this multiple of index_topk. The tuned
    # kernel wins by manufacturing L2 reuse across queries; when the context is
    # barely larger than top-k almost every row is selected by almost every
    # query, so it is L2-resident already and the extra index arithmetic costs
    # ~7 % (measured 0.93x at ctx 2304 / topk 2048, 1.39x at ctx 8192).
    "VLLM_GLM5_SPARSE_MLA_MIN_CTX_MULT": lambda: float(
        os.getenv("VLLM_GLM5_SPARSE_MLA_MIN_CTX_MULT", "2.0")
    ),
    # Predicate the row gather of the sm_80 sparse-MLA prefill kernel
    # (vllm/sm80/kernels/attention/sparse_prefill_mla.py) on the index mask, so that
    # invalid (-1) top-k slots fetch nothing instead of all reading cache row
    # 0. The output is bitwise unchanged. Taken only with all 64 heads on one
    # card (pipeline parallel, TP=1) or 16 heads per rank (tensor parallel 4)
    # at the 512-wide NoPE layout, 2176 index slots and the schedule that head
    # count gets; every other shape keeps the plain gather.
    # Needs VLLM_GLM5_PREFILL_KERNELS=1 as well. Off by default.
    "VLLM_GLM5_SMLA_PREFILL_PRED_LOAD": lambda: bool(
        int(os.getenv("VLLM_GLM5_SMLA_PREFILL_PRED_LOAD", "0"))
    ),
    # sm_80 KDA chunked prefill (vllm/models/glm5next/nvidia/ops/kda_prefill.py)
    # in place of the upstream chunk_kda_with_fused_gate, when all 64 heads live on one
    # card (pipeline parallel, TP=1): six launches instead of nine, fixed
    # launch configurations, fp32-exact intermediates where the recurrence
    # needs them. Taken only on sm_80 with the Triton prefill backend, head
    # dim 128, bf16 activations, fp32 state, the safe gate at -5, and chunks
    # of up to 2312 tokens and 16 sequences; everything else keeps the
    # upstream path. Off by default.
    "VLLM_AMPERE_KDA_PREFILL_WIDE": lambda: bool(
        int(os.getenv("VLLM_AMPERE_KDA_PREFILL_WIDE", "0"))
    ),
    # Debug: range-check every KDA/Mamba state index (GDN metadata, align-mode
    # pre-/post-copy block columns and block ids) on the host before the
    # kernels use it; raises instead of addressing outside the state pool.
    # Synchronizes every step. Off by default.
    "VLLM_STATE_INDEX_CHECK": lambda: bool(
        int(os.getenv("VLLM_STATE_INDEX_CHECK", "0"))
    ),
    # The same sm_80 KDA chunked prefill with tensor parallel 4 (16 heads per
    # card), for prefill chunks of up to 3460 tokens (the TP token budget) and
    # 16 sequences; the other conditions as VLLM_AMPERE_KDA_PREFILL_WIDE. Its own
    # switch so the two layouts are judged separately. Off by default.
    "VLLM_AMPERE_KDA_PREFILL_NARROW": lambda: bool(
        int(os.getenv("VLLM_AMPERE_KDA_PREFILL_NARROW", "0"))
    ),
    # Opt in to the sm_80 decode kernels (mHC and MoE routing in
    # vllm/sm80/kernels/, the KDA decode step in vllm/models/glm5next/nvidia/ops/;
    # GA100-class parts only; measured on CMP 170HX, 70 SMs). Default OFF: with
    # this unset no such kernel is imported or called and every call site keeps
    # the upstream code path exactly. The per-family switches below are only
    # consulted when this is on, so a single family can be disabled on the
    # server without a rebuild -- which is how a regression gets bisected. The
    # fused KDA decode step has no family switch: this one turns it off.
    #
    "VLLM_AMPERE_DECODE_KERNELS": lambda: bool(
        int(os.getenv("VLLM_AMPERE_DECODE_KERNELS", "0"))
    ),
    "VLLM_GLM5_DECODE_MOE_ROUTING": lambda: bool(
        int(os.getenv("VLLM_GLM5_DECODE_MOE_ROUTING", "1"))
    ),
    # Opt in to the sm_80 MLA-indexer decode glue folds in
    # vllm/models/glm5next/nvidia/ops/idx_glue.py (GA100-class parts only).
    # Default OFF: with this unset every call site keeps its existing code path
    # exactly.
    # VLLM_GLM5_DECODE_IDX_GLUE_PARTS selects the folds (comma list, only read
    # when the switch is on): weights (fp32 head weights from the wk+weights
    # thin GEMM), glue (indexer casts/fill/copy folded into kernels), fwht
    # (weight scale folded into the query Hadamard+quant, more CTAs), cache
    # (MLA latent cache write merged with the top-k index remap), moesum
    # (routed moe_sum fused with the shared-expert add).
    "VLLM_AMPERE_DECODE_IDX_GLUE": lambda: bool(
        int(os.getenv("VLLM_AMPERE_DECODE_IDX_GLUE", "0"))
    ),
    "VLLM_GLM5_DECODE_IDX_GLUE_PARTS": lambda: os.getenv(
        "VLLM_GLM5_DECODE_IDX_GLUE_PARTS", "weights,glue,fwht,cache,moesum"
    ),
    # mHC v3 (vllm/sm80/kernels/mhc/decode.py): one implementation for
    # 1 <= M <= VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS that replaces TileLang in
    # that range. Off by default; needs VLLM_AMPERE_DECODE_KERNELS=1 as well.
    "VLLM_GLM5_DECODE_MHC_V2": lambda: bool(
        int(os.getenv("VLLM_GLM5_DECODE_MHC_V2", "0"))
    ),
    "VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS": lambda: int(
        os.getenv("VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS", "32")
    ),
    # Row bound shared by the sm_80 fused MoE router (use_ampere_moe_routing)
    # and the MoE padding mask. The router never dispatches above it: 8 is the
    # top of its measured win region (0.93x at M=32), not a shared cap; mHC's
    # is VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS. The fused KDA decode step has no
    # token bound of its own; its shape gate is use_ampere_kda_decode_v2 in
    # vllm/models/glm5next/nvidia/ops/kda_decode.py, where the per-M table also
    # lives. With VLLM_MOE_MASK_PADDING=1, batches of up to this many rows are
    # left unmasked whether or not VLLM_AMPERE_DECODE_KERNELS is on (the fused
    # router hands its own block alignment over for them); set 0 to mask every
    # batch.
    "VLLM_MOE_MASK_PADDING_MAX_TOKENS": lambda: int(
        os.getenv("VLLM_MOE_MASK_PADDING_MAX_TOKENS", "8")
    ),
    # Opt in to vllm/sm80/kernels/moe/route.py: the MoE router GEMV, the
    # sigmoid/bias top-k and the Marlin block alignment as one op (one launch
    # for 2 <= M <= 64), replacing GateLinear's cuBLAS split-K GEMM + reduce and
    # the routing kernels. Needs VLLM_AMPERE_DECODE_KERNELS=1 as well. Default
    # OFF: with it unset nothing in vllm/sm80/kernels/moe/route.py is imported
    # or called.
    "VLLM_MOE_DECODE_ROUTE_V2": lambda: bool(
        int(os.getenv("VLLM_MOE_DECODE_ROUTE_V2", "0"))
    ),
    # Upper token bound for VLLM_MOE_DECODE_ROUTE_V2: the largest decode
    # batch it was validated and timed at (M = 1..32, 33 validated). Values
    # above 64, the most rows the fused router covers, are rejected.
    "VLLM_GLM5_DECODE_MOE_ROUTE_V2_MAX_TOKENS": lambda: _glm5_route_v2_max_tokens(),
    # Kill switch for the fused KDA decode step at 64 heads per card (pipeline
    # parallel) beyond one sequence of up to 5 tokens: with 1 (default) the
    # 64-head layout takes vllm/models/glm5next/nvidia/ops/kda_decode.py at every
    # decode shape its gate covers; 0 restores the earlier split (one sequence of up to
    # 5 tokens on v2 with the full speculative tile, everything else on the
    # unfused fla chain). No effect at 16 heads.
    "VLLM_GLM5_DECODE_KDA_WIDE": lambda: bool(
        int(os.getenv("VLLM_GLM5_DECODE_KDA_WIDE", "1"))
    ),
    # RecoverSSM commit fold (TP4, vllm/sm80/kernels/kda/recover.py): the
    # replay of a step's accepted tokens over the KDA state runs at the start
    # of the next verify instead of in a separate commit after sampling; steps
    # whose state is read before then (block boundaries) commit as before.
    # Bitwise the same state. Off by default; no effect where RecoverSSM is off.
    "VLLM_AMPERE_KDA_FOLD": lambda: bool(int(os.getenv("VLLM_AMPERE_KDA_FOLD", "0"))),
    # Diagnostic, never for serving: RecoverSSM off (TP4,
    # vllm/sm80/kernels/kda/recover.py) where its gate holds. The KDA layers
    # then keep one state per draft position, as wherever that gate is closed
    # (draft-position state pages, the per-position verify; no records, commit
    # or commit fold): the same-tree reference without state recovery. Off by
    # default.
    "VLLM_AMPERE_KDA_RECOVER_OFF": lambda: bool(
        int(os.getenv("VLLM_AMPERE_KDA_RECOVER_OFF", "0"))
    ),
    # Pin every @triton.autotune'd kernel of the vendored flash-linear-attention
    # ops and the GLM-5 KDA chunked-prefill kernels to ONE config per autotune
    # key on sm_80 (table in
    # vllm/sm80/fla/autotune_pins.py), so a boot
    # never benchmarks and the chosen config, hence the reduction order, is a
    # pure function of the shape. Read when those modules are imported; other
    # devices keep normal autotuning. Off by default.
    "VLLM_FLA_PIN_AUTOTUNE": lambda: bool(
        int(os.getenv("VLLM_FLA_PIN_AUTOTUNE", "0"))
    ),
    # RecoverSSM verify of the fused KDA decode (TP4, kda_decode.py; the commit
    # fold's included): nseq 7-8 take one 16-row warp per CTA (twice the
    # CTAs) instead of one 32-row warp that spills 2,184 B of stack per thread
    # (2,624 B with the fold; at most 872 B at 16 rows). The driver reserves
    # the largest stack launched (at least its default 1,024 B) for every
    # thread an SM can hold and the verify's first launch falls in the
    # CUDA-graph memory profile, so the smaller stack leaves that memory to
    # the KV cache. The same warp layouts, hence bitwise the same outputs and
    # states. Off by default; no effect where RecoverSSM is off.
    "VLLM_AMPERE_KDA_SMALL_STACK": lambda: bool(
        int(os.getenv("VLLM_AMPERE_KDA_SMALL_STACK", "0"))
    ),
    # Rank tokens inside each expert segment of moe_align_block_size by
    # ascending flat routed-row index instead of by atomic arrival order.
    # num_experts is 288 for GLM-5.3-Flash, so the CUDA launcher always
    # takes its two-kernel path, whose count_and_sort kernel ranks with a
    # global atomicAdd; the fused Marlin MoE then accumulates each expert
    # in whatever order that produced. 0 = upstream CUDA op (default),
    # 1 = deterministic Triton kernel; any other value is rejected.
    "VLLM_MOE_DETERMINISTIC_ALIGN": lambda: int(
        _env_with_choices("VLLM_MOE_DETERMINISTIC_ALIGN", "0", ["0", "1"])()
    ),
    # Put each row of the sparse indexer's selected top-k (token or pool ids)
    # in ascending order before attention reads it. The stock top-k kernels
    # return the right set in an arrival-dependent order, and the sparse MLA
    # kernel accumulates its online softmax in index order, so the same
    # prompt otherwise gets a different attention output from run to run
    # (last-bit differences that the rest of the model amplifies). Only the
    # order changes, never the set. The sort runs inside the tie-fix launch,
    # so this needs VLLM_GLM5_TOPK_TIEFIX=1; set alone, the first indexer
    # forward raises. Default OFF: unset, the indexer is byte-identical to
    # upstream.
    "VLLM_GLM5_TOPK_SORTED": lambda: bool(
        int(os.getenv("VLLM_GLM5_TOPK_SORTED", "0"))
    ),
    # Keep the fast indexer top-k kernels but make the selected SET a function
    # of the scores alone: where entries tied with the k-th score straddle the
    # selection boundary, the tied slots go to the lowest column indices.
    # The stock kernels resolve such ties by atomic arrival order, and which
    # kernel runs depends on the row count and length, so the set otherwise
    # depends on batch shape. One extra read of the logits per row;
    # uncontested rows are left untouched.
    # Default OFF: unset, the indexer is byte-identical to upstream.
    "VLLM_GLM5_TOPK_TIEFIX": lambda: bool(
        int(os.getenv("VLLM_GLM5_TOPK_TIEFIX", "0"))
    ),
    # With VLLM_GLM5_TOPK_TIEFIX and VLLM_GLM5_TOPK_SORTED both set: for
    # batches of at most this many rows over wide logits (decode at small
    # concurrency over a long context), run the tie fix + sort as two launches
    # over (row, chunk) instead of one program per row, so a few long rows
    # use many SMs. Same bytes. 0 (default) = off: one program per row.
    "VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS": lambda: int(
        os.getenv("VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS", "0")
    ),
    # Route the padding rows of a padded batch (CUDA-graph sizes) to no
    # expert (topk_ids = -1) before the fused MoE. Their hidden states come
    # from stale input buffers, and where they land in the expert blocks
    # changes how Marlin splits K for the real rows next to them, so the same
    # request otherwise gets last-bit different MoE outputs depending on what
    # ran before. Uses the forward context's is_padding mask; batches of up to
    # VLLM_MOE_MASK_PADDING_MAX_TOKENS rows are left unmasked. Default OFF.
    "VLLM_MOE_MASK_PADDING": lambda: bool(
        int(os.getenv("VLLM_MOE_MASK_PADDING", "0"))
    ),
    # With VLLM_MOE_MASK_PADDING=1 and the sm_80 fused router v2
    # (VLLM_MOE_DECODE_ROUTE_V2=1): route the padding rows to no expert
    # inside the router's own launch (read from the device-side is_padding
    # mask), so its block alignment is already that of the masked ids and is
    # kept, instead of being dropped and recomputed on every padded batch of
    # 9..32 rows. No effect unless both of those are on. Default ON; set 0
    # for the recompute path.
    "VLLM_GLM5_MOE_ROUTE_V2_MASK": lambda: bool(
        int(os.getenv("VLLM_GLM5_MOE_ROUTE_V2_MASK", "1"))
    ),
    # Opt in to the sm_80 thin-M BF16 GEMM in vllm/sm80/kernels/thin_gemm/ for the
    # layers GLM-5.3-Flash's W4A16 recipe leaves unquantized (GA100-class parts
    # only; measured on CMP 170HX, 70 SMs). Default OFF: with this unset the
    # kernel module is never imported, triton is never touched, and
    # dispatch_unquantized_gemm returns the upstream callable itself, so the
    # unflagged path is byte-identical rather than merely equivalent.
    "VLLM_THIN_GEMM": lambda: bool(int(os.getenv("VLLM_THIN_GEMM", "0"))),
    # Size the DFlash drafter's RoPE cos/sin cache to the reachable positions
    # (max_model_len plus a small query margin) instead of the drafter config's
    # max_position_embeddings. Rows are position-indexed, so every reachable
    # row is unchanged; the unreachable tail is simply not allocated.
    "VLLM_GLM5_DRAFTER_ROPE_FIT": lambda: bool(
        int(os.getenv("VLLM_GLM5_DRAFTER_ROPE_FIT", "0"))
    ),
    # GLM-4.7-format parser (glm47): with tool_choice="none", forbid the
    # tool-call start token in decoding (draft and target). Off by default:
    # requests are adjusted as upstream adjusts them; the GLM launcher sets 1.
    "VLLM_TOOL_CHOICE_NONE_MASK": lambda: bool(
        int(os.getenv("VLLM_TOOL_CHOICE_NONE_MASK", "0"))
    ),
    "VLLM_DFLASH_BOUNDARY_CACHE": lambda: bool(
        int(os.getenv("VLLM_DFLASH_BOUNDARY_CACHE", "0"))
    ),
    # Diagnostics (step-cost calibration): a file holding a depth; while it
    # holds one inside the adaptive_k load width, every step verifies that
    # depth.
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE": lambda: os.getenv(
        "VLLM_GLM5_DFLASH_ADAPTIVE_K_FORCE_FILE", ""
    ),
    # Shard the DFlash2 candidate selector's two (vocab, rank) codebooks by
    # vocab rows across the tensor-parallel ranks instead of replicating them.
    # Each rank looks up the rows it owns, zero elsewhere, and one all-reduce
    # of the two gathered row sets rebuilds them exactly (x + 0 = x).
    "VLLM_GLM5_DRAFTER_SELECTOR_SHARD": lambda: bool(
        int(os.getenv("VLLM_GLM5_DRAFTER_SELECTOR_SHARD", "0"))
    ),
    # Log where device memory goes at start-up: resident bytes per module
    # group after loading, the peak of each profile_run stage, the KV sizing
    # terms, and what the attention metadata builders allocate after the KV
    # cache is sized. Logging only; the KV cache size is unchanged.
    "VLLM_MEM_ATTRIBUTION": lambda: bool(
        int(os.getenv("VLLM_MEM_ATTRIBUTION", "0"))
    ),
    # Align-mode Mamba KV reservation under async scheduling or pipeline
    # parallelism: also reserve the state blocks a request holds for the
    # prefill chunks it has in flight (min(max_concurrent_batches,
    # cdiv(max_in_flight_tokens, block)) - 1 per group beyond the two of
    # synchronous scheduling), charged once per running request. Changes the
    # reported KV capacity and the single-request fit check; the block pool
    # and the admission cap are unchanged. Default OFF.
    "VLLM_KV_MAMBA_INFLIGHT_STATES": lambda: bool(
        int(os.getenv("VLLM_KV_MAMBA_INFLIGHT_STATES", "0"))
    ),
    # Re-order the multi-stream shared-expert overlap in the MoE runner.
    # Upstream enqueues the shared experts on the aux stream *before* the gate
    # and the routed dispatch, then joins after the routed kernels. Because the
    # aux branch is submitted first it is already resident when the main stream
    # finally reaches the routed Marlin kernels, so on CUDA under breakable
    # cudagraphs almost none of it actually overlaps (a decode trace measured
    # ~4%). With this on, the runner instead marks the aux stream's start point
    # up front, enqueues the routed experts, and only then runs the shared
    # experts on the aux stream, joining before the output is read.
    # Enqueue order only: the same kernels read the same tensors and the join
    # still precedes every read of the shared-expert output, so the numerics
    # are unchanged. It engages only where the multi-stream order was already
    # chosen, i.e. at or below VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
    # tokens (decode), never for the 1152-token prefill chunks.
    # Default OFF until it has its own A/B on the cards.
    "VLLM_SHARED_EXPERT_REORDER": lambda: bool(
        int(os.getenv("VLLM_SHARED_EXPERT_REORDER", "0"))
    ),
    # Pipeline parallelism: cap each scheduling step at ceil(decoding
    # requests / pp_size) decode requests so the pp_size in-flight
    # micro-batches share the decodes instead of one taking them all and the
    # others idling (a decode request is ineligible until its results return
    # pp_size steps later). No effect with pipeline_parallel_size 1 or with
    # at most one decoding request. Default OFF until measured on the cards.
    "VLLM_PP_SPREAD_DECODES": lambda: bool(
        int(os.getenv("VLLM_PP_SPREAD_DECODES", "0"))
    ),
    # Pipeline parallelism (V2 runner, TP=1): send each stage hand-off as one
    # flat device transfer instead of one per tensor (the GLM-5.3-Flash
    # boundary carries the residual streams plus drafter aux states). The
    # per-step metadata still travels and is checked. Default OFF.
    "VLLM_PP_PACKED_HOP": lambda: bool(int(os.getenv("VLLM_PP_PACKED_HOP", "0"))),
    # With VLLM_PP_PACKED_HOP: after a first-hop handshake, send no per-step
    # metadata; the receiver's persistent buffer gives the layout and the
    # scheduler output gives the row count (only the scheduled rows travel).
    # Default OFF.
    "VLLM_PP_HOP_NO_METADATA": lambda: bool(
        int(os.getenv("VLLM_PP_HOP_NO_METADATA", "0"))
    ),
    # Pipeline parallelism with speculative decoding: the first stages wait
    # only for the sampled tokens and counts before preparing the next step,
    # and for the draft tokens (sent after the drafter ran) only right before
    # the forward. Default OFF.
    "VLLM_PP_SPLIT_DRAFT_EVENT": lambda: bool(
        int(os.getenv("VLLM_PP_SPLIT_DRAFT_EVENT", "0"))
    ),
    # Pipeline parallelism with a DFlash2 drafter: run the drafter's tail
    # (candidate lm_head pass, top-k, selector, walk) on this earlier stage
    # instead of the last one, which then only runs the drafter's layers.
    # Needs no KV cache on that stage, one lm_head + selector copy. Draft
    # tokens are bit-identical. -1 (default) = off.
    "VLLM_PP_DRAFT_TAIL_STAGE": lambda: int(
        os.getenv("VLLM_PP_DRAFT_TAIL_STAGE", "-1")
    ),
    # GLM-5.3-Flash + DFlash under PP: each stage multiplies its own aux
    # hidden states by their slice of the drafter's input projection (fc) and
    # forwards one fp32 partial sum instead of relaying the aux states; the
    # last stage uses the sum as the fc output. Same arithmetic up to the fp32
    # summation order. Default OFF.
    "VLLM_GLM5_PP_FOLD_DRAFT_FC": lambda: bool(
        int(os.getenv("VLLM_GLM5_PP_FOLD_DRAFT_FC", "0"))
    ),
    # Whole-expert Marlin W4A16 MoE at prefill (pipeline parallel, TP=1: all
    # 288 experts, N=2048, on one sm_80 card): align the routed rows into one
    # block list per size 64/48/32/16 (cheapest cover per expert) instead of
    # padding every expert to one block size, and run each Marlin GEMM once
    # per list (vllm/sm80/kernels/marlin_moe/prefill.py). Same compiled
    # Marlin kernels; the split cover changes stream-K boundaries and reduction
    # order, so output is not bitwise equal to fused_marlin_moe. Taken only
    # for M >= VLLM_GLM5_PP_MARLIN_PREFILL_MIN_TOKENS on exactly that
    # configuration; everything else keeps fused_marlin_moe (TP4, N=512, has
    # its own switch, VLLM_AMPERE_MARLIN_PREFILL_NARROW).
    # Off by default.
    "VLLM_AMPERE_MARLIN_PREFILL_WIDE": lambda: bool(
        int(os.getenv("VLLM_AMPERE_MARLIN_PREFILL_WIDE", "0"))
    ),
    "VLLM_GLM5_PP_MARLIN_PREFILL_MIN_TOKENS": lambda: int(
        os.getenv("VLLM_GLM5_PP_MARLIN_PREFILL_MIN_TOKENS", "384")
    ),
    # With VLLM_AMPERE_MARLIN_PREFILL_WIDE: run the split-list Marlin GEMMs of
    # the whole-expert (N=2048) prefill through the optional library's
    # prefill_tile_gemm with the tile table of
    # vllm/sm80/kernels/marlin_moe/prefill.py ((64,512) tiles for the
    # 64/48/32-row lists in both GEMMs) and the optimistic fast dequant (tiles
    # with a scale above 2^-5 recomputed by the regular kernel). Without the
    # library or its op, the released kernels run. The TP4 (N=512) lists
    # always run on the released kernels. On by default; 0 is the kill switch
    # (released split path unchanged).
    "VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED": lambda: bool(
        int(os.getenv("VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED", "1"))
    ),
    # The same split-block Marlin MoE prefill under tensor parallel 4 (each
    # card holds all 288 experts sharded to N=512): taken only for M >=
    # VLLM_GLM5_TP4_MARLIN_PREFILL_MIN_TOKENS with N=512 and otherwise the
    # conditions of VLLM_AMPERE_MARLIN_PREFILL_WIDE. Its own switch so the two
    # layouts are judged separately. Off by default.
    "VLLM_AMPERE_MARLIN_PREFILL_NARROW": lambda: bool(
        int(os.getenv("VLLM_AMPERE_MARLIN_PREFILL_NARROW", "0"))
    ),
    "VLLM_GLM5_TP4_MARLIN_PREFILL_MIN_TOKENS": lambda: int(
        os.getenv("VLLM_GLM5_TP4_MARLIN_PREFILL_MIN_TOKENS", "384")
    ),
    # Optional prebuilt sm_80 Marlin decode kernels. Enabling the flag
    # requires vllm._ampere_marlin_C; startup fails if it is missing or
    # incompatible.
    "VLLM_AMPERE_MARLIN_DECODE_CUDA": lambda: bool(
        int(os.getenv("VLLM_AMPERE_MARLIN_DECODE_CUDA", "0"))
    ),
    # With VLLM_AMPERE_MARLIN_DECODE_CUDA=1: run the tensor-parallel (N=512)
    # decode GEMMs through the library's decode_gemm_v2 (one-FMA exact
    # dequant, 4-row chunks at 2-7 rows, zero padding fragments at 32 rows).
    # Same per-output summation order as decode_gemm_orig, so the output is
    # bitwise equal to it while every weight scale lies in [2^-110, 2^10) or
    # is zero (checked once per layer) and no nonzero weight-activation
    # product is below 2^-110 (vllm/sm80/kernels/marlin_moe/decode.py).
    # Without the op, at other widths, or out of that scale range,
    # decode_gemm_orig runs. Off by default.
    "VLLM_AMPERE_MARLIN_DECODE_V2": lambda: bool(
        int(os.getenv("VLLM_AMPERE_MARLIN_DECODE_V2", "0"))
    ),
    # Whole-expert (pipeline-parallel, N=2048) compiled decode also at 5-7
    # rows (draft depths 4-6 of one request), not only at 4 and 8. Needs
    # VLLM_AMPERE_MARLIN_DECODE_CUDA=1.
    "VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS": lambda: bool(
        int(os.getenv("VLLM_GLM5_MARLIN_DECODE_PP_MID_ROWS", "0"))
    ),
    # Whole-expert (N=2048) compiled decode for multi-request batches, at the
    # row ranges of _ROWS ("lo-hi[,lo-hi...]" within 9..64; default 9-32, the
    # most rows one PP4 step verifies with DEPTHS 7,5 and 8 sequences). The
    # N=2048 scratch grows to the largest row count named there.
    "VLLM_GLM5_MARLIN_DECODE_PP_MULTI": lambda: bool(
        int(os.getenv("VLLM_GLM5_MARLIN_DECODE_PP_MULTI", "0"))
    ),
    "VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS": lambda: os.getenv(
        "VLLM_GLM5_MARLIN_DECODE_PP_MULTI_ROWS", "9-32"
    ).strip(),
    # Host-staged all-reduce for PCIe-only multi-GPU nodes with no peer access
    # (the CMP 170HX case). Off by default. When on it stands aside only if
    # VLLM_CUSTOM_ALLREDUCE_PCIE_P2P is also set *and* peer access is
    # really there, so the two fast paths are mutually exclusive by the
    # operator's choice rather than by what the driver happens to report. See
    # vllm/sm80/distributed/host_shm_all_reduce.py.
    "VLLM_HOST_ALLREDUCE": lambda: bool(
        int(os.getenv("VLLM_HOST_ALLREDUCE", "0"))
    ),
    # Largest message the host path serves; anything above goes to NCCL. 512 KiB
    # is where the measured curve stops paying: at 8-256 KiB the host path is
    # 1.06-2.04x NCCL, while prefill-sized messages (>= 2 MiB) already run near
    # PCIe wire speed on the ring and must not be taken away from it.
    "VLLM_HOST_ALLREDUCE_MAX_SIZE": lambda: int(
        os.getenv("VLLM_HOST_ALLREDUCE_MAX_SIZE", str(512 * 1024))
    ),
    # Never dispatch above this M (= num_seqs * (1 + num_spec)). 32 is measured,
    # and it is a cudagraph capture size, so the bound lands exactly on one.
    # Count-weighted over the whole shape table: M=32 is 1.157x with 0 of 16
    # shapes losing, M=40 is 0.799x with 10 of 16 losing. Raising it one capture
    # size costs ~43 % of the step. Re-run the token-bound measurements before
    # touching this.
    "VLLM_THIN_GEMM_MAX_TOKENS": lambda: int(
        os.getenv("VLLM_THIN_GEMM_MAX_TOKENS", "32")
    ),
    # Run each MLA layer's two thin-GEMM concat groups (q_b_proj + the
    # indexer's wq_b on q_c; fused_qkv_a_proj + the indexer's kpool gate on
    # hidden_states) as one launch per group, paired inside the thin-GEMM
    # dispatch (vllm/sm80/kernels/thin_gemm/concat.py). Every member keeps
    # its solo schedule, so each output is bitwise the separate call. Needs
    # VLLM_THIN_GEMM, sm_80 and compilation mode NONE. Default OFF: unset,
    # the pairing module is never imported.
    "VLLM_THIN_GEMM_CONCAT": lambda: bool(
        int(os.getenv("VLLM_THIN_GEMM_CONCAT", "0"))
    ),
    # Qwen EXL3 checkpoints (vllm/sm80/kernels/exl3/quant.py).
    "VLLM_EXL3": lambda: bool(int(os.getenv("VLLM_EXL3", "0"))),
    "VLLM_EXL3_PLE_PLACEMENT": lambda: _env_with_choices(
        "VLLM_EXL3_PLE_PLACEMENT", "ssd", ["ssd", "ram", "device"]
    )(),
    # Order Qwen QSA's native-selected blocks before expansion/attention.
    # Set 0 to retain native traversal order without changing selection.
    "VLLM_QWEN_QSA_TOPK_SORTED": lambda: bool(
        int(os.getenv("VLLM_QWEN_QSA_TOPK_SORTED", "1"))
    ),
    # Drop the compressed PLE table from the page cache at setup so a run
    # starts cold (reads come from storage). Residency is logged either way.
    "VLLM_EXL3_PLE_COLD": lambda: bool(int(os.getenv("VLLM_EXL3_PLE_COLD", "0"))),
    # Retries (1 ms backoff, doubling) for a failed or truncated PLE row read
    # before the requests using that row are finished with an error.
    "VLLM_EXL3_PLE_READ_RETRIES": lambda: int(
        os.getenv("VLLM_EXL3_PLE_READ_RETRIES", "3")
    ),
    # Preserve BF16 router operands but accumulate and return logits in FP32.
    "VLLM_QWEN4_ROUTER_FP32": lambda: bool(
        int(os.getenv("VLLM_QWEN4_ROUTER_FP32", "0"))
    ),
    # Normalize each MTP hidden stream instead of the concatenated stack.
    "VLLM_QWEN4_MTP_PER_STREAM_NORM": lambda: bool(
        int(os.getenv("VLLM_QWEN4_MTP_PER_STREAM_NORM", "0"))
    ),
}


# Environment variables retired in this release, with what replaces them. A
# retired name never supplies a value (no aliases): setting one only logs a
# warning, once per process, when this module is imported. Remove this table,
# warn_retired_env_vars() and its call below in the next release.
RETIRED_ENV_VARS: dict[str, str] = {
    # Renamed: shared code is named after the feature, not the model.
    "VLLM_GLM5_DECODE_MOE_MAX_TOKENS": (
        "renamed to VLLM_MOE_MASK_PADDING_MAX_TOKENS, which is now also the MoE "
        "padding mask cutoff whether or not VLLM_AMPERE_DECODE_KERNELS is on"
    ),
    "VLLM_GLM5_FLA_PIN_AUTOTUNE": "renamed to VLLM_FLA_PIN_AUTOTUNE",
    "VLLM_GLM5_HOST_ALLREDUCE": "renamed to VLLM_HOST_ALLREDUCE",
    "VLLM_GLM5_HOST_ALLREDUCE_MAX_SIZE": "renamed to VLLM_HOST_ALLREDUCE_MAX_SIZE",
    "VLLM_GLM5_MEM_ATTRIBUTION": "renamed to VLLM_MEM_ATTRIBUTION",
    "VLLM_GLM5_MOE_MASK_PADDING": (
        "renamed to VLLM_MOE_MASK_PADDING. Batches of up to "
        "VLLM_MOE_MASK_PADDING_MAX_TOKENS rows (default 8) are now left unmasked "
        "even with VLLM_AMPERE_DECODE_KERNELS off; set that cutoff to 0 to mask "
        "every batch"
    ),
    "VLLM_GLM5_SHARED_EXPERT_REORDER": "renamed to VLLM_SHARED_EXPERT_REORDER",
    "VLLM_GLM5_STATE_INDEX_CHECK": "renamed to VLLM_STATE_INDEX_CHECK",
    "VLLM_GLM5_THIN_GEMM": "renamed to VLLM_THIN_GEMM",
    "VLLM_GLM5_THIN_GEMM_MAX_TOKENS": "renamed to VLLM_THIN_GEMM_MAX_TOKENS",
    "VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE": (
        "renamed to VLLM_CUSTOM_ALLREDUCE_PCIE_P2P"
    ),
    "VLLM_BUILD_AMPERE_MARLIN": "renamed to VLLM_AMPERE_MARLIN_BUILD",
    "VLLM_DSA_INDEXER_REF": "renamed to VLLM_INDEXER_DSA_REF",
    "VLLM_GLM5_DECODE_IDX_GLUE": "renamed to VLLM_AMPERE_DECODE_IDX_GLUE",
    "VLLM_GLM5_DECODE_KERNELS": "renamed to VLLM_AMPERE_DECODE_KERNELS",
    "VLLM_GLM5_DECODE_MOE_ROUTE_V2": "renamed to VLLM_MOE_DECODE_ROUTE_V2",
    "VLLM_GLM5_DETERMINISTIC_MOE_ALIGN": "renamed to VLLM_MOE_DETERMINISTIC_ALIGN",
    "VLLM_GLM5_DFLASH_BOUNDARY_CACHE": "renamed to VLLM_DFLASH_BOUNDARY_CACHE",
    "VLLM_GLM5_INDEXER_DECODE_RAW_K": "renamed to VLLM_INDEXER_DECODE_RAW_K",
    "VLLM_GLM5_MARLIN_DECODE_CUDA": "renamed to VLLM_AMPERE_MARLIN_DECODE_CUDA",
    "VLLM_GLM5_PP_KDA_PREFILL": "renamed to VLLM_AMPERE_KDA_PREFILL_WIDE",
    "VLLM_GLM5_PP_MARLIN_PREFILL": "renamed to VLLM_AMPERE_MARLIN_PREFILL_WIDE",
    "VLLM_GLM5_PP_MARLIN_PREFILL_COMPILED": (
        "renamed to VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED"
    ),
    "VLLM_GLM5_PP_SPARSE_MLA_PREFILL": (
        "renamed to VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE"
    ),
    "VLLM_GLM5_TOOL_CHOICE_NONE_MASK": "renamed to VLLM_TOOL_CHOICE_NONE_MASK",
    "VLLM_GLM5_TP4_KDA_PREFILL": "renamed to VLLM_AMPERE_KDA_PREFILL_NARROW",
    "VLLM_GLM5_TP4_MARLIN_PREFILL": (
        "renamed to VLLM_AMPERE_MARLIN_PREFILL_NARROW"
    ),
    # Replaced by a vLLM option or a standard location.
    "VLLM_GLM5_DFLASH_ADAPTIVE_K": (
        "replaced by --speculative-config adaptive_k (by_load)"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_DEPTHS": (
        "replaced by --speculative-config adaptive_k.by_load"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_DRAFT_WIDTH": (
        "replaced by --speculative-config adaptive_k.draft_by_load"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_ACCEPT": (
        "replaced by --speculative-config adaptive_k.accept"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_COSTS": (
        "replaced by --speculative-config adaptive_k.accept.costs"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_COSTS_MULTI": (
        "replaced by --speculative-config adaptive_k.accept.costs_multi"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_HYST": (
        "replaced by --speculative-config adaptive_k.accept.hysteresis"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_PRIOR": (
        "replaced by --speculative-config adaptive_k.accept.prior"
    ),
    "VLLM_GLM5_DFLASH_ADAPTIVE_K_LOG": (
        "replaced by --speculative-config adaptive_k.log_interval"
    ),
    "VLLM_GLM5_LOCAL_LOGITS": "replaced by --enable-batch-sharded-sampling",
    "VLLM_GLM5_HOST_ALLREDUCE_BUILD_DIR": (
        "replaced by TORCH_EXTENSIONS_DIR (default "
        "~/.cache/vllm/host_shm_all_reduce)"
    ),
    "VLLM_CUSTOM_ALLREDUCE_FLAGS_BUILD_DIR": (
        "replaced by $VLLM_CACHE_ROOT/custom_all_reduce_flags"
    ),
    "VLLM_GLM5_AUX_HIDDEN_TENSOR": (
        "replaced by the model config field aux_hidden_state_tensor; for the "
        "branch tensor pass --hf-overrides "
        '{"text_config": {"aux_hidden_state_tensor": "branch"}}'
    ),
    # Removed with the code they selected.
    "VLLM_GLM5_DECODE_MHC": (
        "removed with the v1 sm_80 mHC decode kernel; the v3 kernel, opted in "
        "with VLLM_GLM5_DECODE_MHC_V2, covers the same shapes"
    ),
    "VLLM_GLM5_DECODE_MHC_MAX_TOKENS": (
        "removed with the v1 sm_80 mHC decode kernel; the v3 bound is "
        "VLLM_GLM5_DECODE_MHC_V2_MAX_TOKENS"
    ),
    "VLLM_GLM5_DECODE_MHC_V3": (
        "removed; v3 is the only sm_80 mHC decode kernel, opted in with "
        "VLLM_GLM5_DECODE_MHC_V2"
    ),
    "VLLM_GLM5_DECODE_MHC_V2_FN_BF16": (
        "removed; sm_80 mHC decode always reads the fp32 weights"
    ),
    "VLLM_GLM5_THIN_GEMM_V74": (
        "removed; the 74-SM schedule is the only thin GEMM"
    ),
    "VLLM_GLM5_PREFILL_PACK_BF16X2": (
        "removed; the paired-rounding pack is the only mHC pre-norm pack"
    ),
    "VLLM_GLM5_MARLIN_DECODE_VARIANT": (
        "removed; compiled Marlin decode always uses the split-K order "
        "(the former default, orig)"
    ),
    "VLLM_GLM5_CUSTOM_ALLREDUCE_MAX_SIZE": (
        "removed; the custom all-reduce uses the 8 MiB default cap"
    ),
    "THIN_GEMM_MAX_M": (
        "removed; it was read only by uncalled code. The thin GEMM bound is "
        "VLLM_THIN_GEMM_MAX_TOKENS"
    ),
    "VLLM_KV_SWA_INFLIGHT_SCRATCH": (
        "removed; the KV capacity report charges sliding-window in-flight "
        "blocks per concurrency slot, as upstream"
    ),
    "VLLM_GLM5_INDEXER_DECODE_ROWS": (
        "removed; the indexer decode block tables are sized by "
        "max_num_batched_tokens, as upstream"
    ),
    "VLLM_GLM5_INDEXER_GATHER_CLAMP": (
        "removed; the indexer K-gather workspace and prefill chunk limit are "
        "sized as upstream (40 * max_model_len, in compressed rows)"
    ),
    "VLLM_GLM5_TP4_MARLIN_PREFILL_COMPILED": (
        "removed; the TP4 split-list Marlin MoE prefill runs on the released "
        "Marlin kernels"
    ),
    # The TP prefill comm/compute overlap and its chunk-size observer.
    "VLLM_GLM5_PREFILL_OVERLAP": (
        "removed; the TP all-reduces of a prefill run unsplit on the main "
        "stream, as upstream"
    ),
    "VLLM_GLM5_PREFILL_OVERLAP_SPLITS": "removed with VLLM_GLM5_PREFILL_OVERLAP",
    "VLLM_GLM5_PREFILL_OVERLAP_MIN_TOKENS": "removed with VLLM_GLM5_PREFILL_OVERLAP",
    "VLLM_GLM5_PREFILL_OVERLAP_DEBUG": "removed with VLLM_GLM5_PREFILL_OVERLAP",
    "VLLM_GLM5_PREFILL_OVERLAP_UNDER_CA": "removed with VLLM_GLM5_PREFILL_OVERLAP",
    "VLLM_GLM5_PREFILL_OVERLAP_BACKEND": "removed with VLLM_GLM5_PREFILL_OVERLAP",
    "VLLM_GLM5_PREFILL_OBSERVE": (
        "removed with VLLM_GLM5_PREFILL_OVERLAP (the prefill chunk-size "
        "observer lived in the overlap module)"
    ),
    "VLLM_PP_DRAFT_TAIL_VERIFY": (
        "removed; the last stage no longer reruns the draft tail to compare it "
        "with the drafts VLLM_PP_DRAFT_TAIL_STAGE sends"
    ),
    "VLLM_GLM5_HOST_ALLREDUCE_TEST_FAIL_RANK": (
        "removed; the host-staged all-reduce no longer has a forced "
        "registration failure for testing"
    ),
    "VLLM_HOST_ALLREDUCE_TEST_FAIL_RANK": (
        "removed; the host-staged all-reduce no longer has a forced "
        "registration failure for testing"
    ),
}
_warned_retired_env_vars: set[str] = set()


def warn_retired_env_vars() -> None:
    """Warn once per process for every retired name that is set."""
    for name, replacement in RETIRED_ENV_VARS.items():
        if name in os.environ and name not in _warned_retired_env_vars:
            _warned_retired_env_vars.add(name)
            logger.warning("%s is retired and ignored: %s.", name, replacement)


def register_env_vars(environment_variables: dict[str, Callable[[], Any]]) -> None:
    """Add the fork's variables to vLLM's table, then warn about retired names.

    Raises if upstream vLLM declares one of the fork's names itself, so a
    rebase that brings such a name cannot silently swap the reader.
    """
    for name, getter in FORK_ENV_VARS.items():
        existing = environment_variables.get(name)
        if existing is not None and existing is not getter:
            raise ValueError(
                f"{name} is declared by vLLM and by vllm/sm80/envs.py; "
                "keep one declaration"
            )
    environment_variables.update(FORK_ENV_VARS)
    warn_retired_env_vars()
