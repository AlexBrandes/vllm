# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified by Morrowmake for CMP 170HX support; see repository history.
"""Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.warmup.attention_warmup import (
    mixed_batch_attention_warmup,
)
from vllm.model_executor.warmup.b12x_warmup import b12x_warmup
from vllm.model_executor.warmup.cutedsl_warmup import cutedsl_warmup
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.model_executor.warmup.flashinfer_autotune_cache import (
    resolve_flashinfer_autotune_v2_root,
    resolve_rank_flashinfer_autotune_file,
    use_flashinfer_autotune_v2,
)
from vllm.model_executor.warmup.flashinfer_sparse_mla_warmup import (
    autotune_hisparse_flashinfer_attention,
    deepseek_v4_sparse_mla_attention_warmup,
    flashinfer_sparse_mla_decode_autotune_warmup,
)
from vllm.model_executor.warmup.kimi_k3_triton_warmup import (
    kimi_k3_triton_warmup,
)
from vllm.model_executor.warmup.mamba_triton_warmup import mamba_triton_warmup
from vllm.model_executor.warmup.qwen4_exp_qsa_warmup import (
    qwen4_exp_qsa_triton_warmup,
)
from vllm.model_executor.warmup.qwen_triton_warmup import qwen_triton_warmup
from vllm.model_executor.warmup.qwen_vl_triton_warmup import qwen_vl_triton_warmup
from vllm.model_executor.warmup.replayssm_warmup import (
    replayssm_autotune_warmup,
)
from vllm.model_executor.warmup.watermark_sample_warmup import (
    watermark_sample_warmup,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

_LL_BF16_WARMUP_M_RANGE = range(1, 17)


def _ll_bf16_router_shapes_from_model(
    model: torch.nn.Module,
) -> tuple[tuple[int, int], ...]:
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    shapes: set[tuple[int, int]] = set()
    for module in model.modules():
        if not isinstance(module, GateLinear):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if weight.dim() != 2 or weight.dtype != torch.bfloat16:
            continue
        n, k = weight.shape
        if k % 8 == 0:
            shapes.add((int(k), int(n)))
    return tuple(sorted(shapes))


def _warmup_ll_bf16_router_gemm(model: torch.nn.Module) -> None:
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        is_available as is_ll_bf16_gemm_available,
    )
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
        ll_bf16_gemm_kernel,
    )

    if not is_ll_bf16_gemm_available():
        return

    shapes = _ll_bf16_router_shapes_from_model(model)
    if not shapes:
        logger.debug_once(
            "Skipping ll_bf16 router GEMM warmup: no bf16 GateLinear shapes found."
        )
        return

    logger.info_once("Warming up ll_bf16 router GEMM kernels for shapes: %s.", shapes)
    ll_bf16_gemm_kernel.warmup(
        shapes=shapes,
        m_values=_LL_BF16_WARMUP_M_RANGE,
    )


def _warmup_bf16x3_router_gemm(
    model: torch.nn.Module,
    max_num_tokens: int,
) -> None:
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl import (  # noqa: E501
        warmup_bf16x3_router_gemm,
    )
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    gate = next(
        (
            module
            for module in model.modules()
            if isinstance(module, GateLinear) and module.allow_bf16x3_router_gemm
        ),
        None,
    )
    if gate is None:
        logger.debug_once(
            "Skipping BF16x3 router GEMM warmup: no eligible GateLinear found."
        )
        return

    min_num_tokens = gate.FP32_MAX_TOKENS + 1 if gate.allow_fp32_router_gemm else 1
    logger.info_once(
        "Warming up BF16x3 router GEMM for K=%d, M=%d.",
        gate.input_size,
        gate.output_size,
    )
    configs = warmup_bf16x3_router_gemm(
        gate.input_size,
        gate.output_size,
        min_num_tokens,
        max_num_tokens,
    )
    logger.info_once("Warmed up BF16x3 router GEMM configs: %s.", configs)


def _warmup_gemm_rs_ar() -> None:
    # Model construction (Kimi-K3, DeepSeek-V4.1) imports this module only
    # when GEMM-RS/AR is enabled and initializes its singleton before
    # kernel_warmup runs. Avoid importing it here so other models do not
    # compile the RS/AR variants.
    module = sys.modules.get("vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar")
    if module is None:
        return
    compiled = module.warmup_gemm_rs_ar()
    if compiled:
        logger.info_once("Warmed up %d GEMM-RS/AR variants.", compiled)


def _autotune_kimi_k3_kda_qkvg(model: torch.nn.Module) -> None:
    module = sys.modules.get("vllm.models.kimi_k3.nvidia.low_latency_gemm")
    if module is not None:
        module.autotune_kda_qkvg(model)


def kernel_warmup(worker: "Worker", *, process_local_only: bool = False):
    from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
        minimax_m3_msa_warmup,
    )
    from vllm.model_executor.warmup.step5_dsa_warmup import step5_dsa_warmup

    if not worker.use_v2_model_runner:
        # The KV-block zeroing kernel is driven by the scheduler's
        # `new_block_ids_to_zero`, so no dummy run ever reaches it.
        zeroer = getattr(worker.model_runner, "_kv_block_zeroer", None)
        if zeroer is not None:
            zeroer.warmup(worker.model_runner.kv_cache_config.num_blocks)

    enable_jit_warmup = worker.vllm_config.kernel_config.enable_jit_warmup
    if enable_jit_warmup:
        logger.info("JIT kernel warmup starting.")
        jit_warmup_start = time.perf_counter()
        try:
            registry = (
                worker.model_runner.jit_warmup_registry  # type: ignore[attr-defined]
            )
            registry.warmup()
        except Exception:
            logger.exception(
                "JIT kernel warmup failed after %.2fs.",
                time.perf_counter() - jit_warmup_start,
            )
            raise
        logger.info(
            "JIT kernel warmup finished in %.2fs.",
            time.perf_counter() - jit_warmup_start,
        )

    qwen_triton_warmup(worker.model_runner, worker.vllm_config.model_config)
    qwen_vl_triton_warmup(worker.model_runner)
    mamba_triton_warmup(worker.model_runner)

    compilation_config = worker.vllm_config.compilation_config
    cudagraph_capture_sizes = list(compilation_config.cudagraph_capture_sizes or [])

    # The compiled Marlin decode (its own switch, on TP and PP workers alike):
    # its persistent scratch must exist before profiling/capture use the ops.
    if envs.VLLM_AMPERE_MARLIN_DECODE_CUDA:
        from vllm.sm80.kernels.marlin_moe.decode import (
            open_late_scratch,
            warmup_from_worker,
        )

        # kernel_warmup runs after the KV cache is sized: the whole-expert
        # decode scratch may be allocated from here on.
        open_late_scratch()
        warmup_from_worker(worker)

    # The sm_80 decode kernels, in this order: mHC (vllm/sm80/kernels/mhc/),
    # MoE routing (vllm/sm80/kernels/moe/), the GLM KDA decode step
    # (vllm/models/glm5next/nvidia/ops/kda_decode.py) and the fused MoE router.
    # This MUST run before capture_model(): routing._scratch() and
    # kda_decode._counter() are allocate-once module-level buffers, and an
    # allocation made during a capture lands in that graph's private pool, so
    # a second capture would reuse freed memory. Triton also compiles on first
    # launch, which cannot happen inside a capture at all. No-op unless
    # VLLM_AMPERE_DECODE_KERNELS; the compiled Marlin decode above is warmed
    # independently of it. The GLM KDA layers import kda_decode when the
    # switch is set; shared code does not import model packages, so its
    # warmup is looked up among the loaded modules.
    if envs.VLLM_AMPERE_DECODE_KERNELS:
        from vllm.sm80.kernels.mhc.warmup import (
            warmup_from_worker as warmup_mhc_decode,
        )
        from vllm.sm80.kernels.moe.warmup import (
            warmup_route_from_worker,
            warmup_routing_from_worker,
        )

        kda_decode = sys.modules.get("vllm.models.glm5next.nvidia.ops.kda_decode")
        warmup_mhc_decode(worker, cudagraph_capture_sizes)
        warmup_routing_from_worker(worker, cudagraph_capture_sizes)
        if kda_decode is not None:
            kda_decode.warmup_from_worker(worker, cudagraph_capture_sizes)
        warmup_route_from_worker(worker, cudagraph_capture_sizes)
    # The sm_80 KDA chunked prefill at 64 or 16 heads
    # (vllm/models/glm5next/nvidia/ops/kda_prefill.py): compile its kernels
    # before the first prefill. Prefill runs eagerly, so this is about
    # first-request latency, not capture.
    # Only with VLLM_AMPERE_KDA_PREFILL_WIDE=1 or VLLM_AMPERE_KDA_PREFILL_NARROW=1, and
    # a no-op unless a KDA layer took the path. The KDA layers import the
    # module when either switch is set; shared code does not import model
    # packages, so the warmup is looked up among the loaded modules.
    kda_prefill = sys.modules.get("vllm.models.glm5next.nvidia.ops.kda_prefill")
    if kda_prefill is not None and (
        envs.VLLM_AMPERE_KDA_PREFILL_WIDE or envs.VLLM_AMPERE_KDA_PREFILL_NARROW
    ):
        kda_prefill.warmup_from_worker(worker)
    # The sm_80 thin-M BF16 GEMM (vllm/sm80/kernels/thin_gemm/). This MUST run before
    # capture_model(): thin_gemm keeps its FP32 split-K partials and arrival
    # counters in an allocate-once module-level cache, and an allocation made
    # during a capture lands in that graph's private pool, so a second capture
    # would reuse freed memory. Triton also compiles on first launch, which
    # cannot happen inside a capture at all. No-op unless VLLM_THIN_GEMM.
    from vllm.sm80.kernels.thin_gemm.warmup import warmup_ampere_thin_gemm

    warmup_ampere_thin_gemm(worker, cudagraph_capture_sizes)
    # The 64-head sparse-MLA prefill kernel (sparse_prefill_mla_pp.py). Prefill
    # runs eagerly, but compiling here keeps the Gluon JIT off the first
    # request. No-op unless VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE and
    # VLLM_GLM5_PREFILL_KERNELS are set and this rank runs 64 heads on sm_80.
    if envs.VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE:
        from vllm.sm80.kernels.attention.sparse_prefill_mla_pp import (
            warmup_from_worker,
        )

        if warmup_from_worker(worker):
            logger.info("Compiled the PP sparse-MLA prefill kernel.")

    # The deterministic MoE block alignment (vllm/sm80/kernels/moe/align.py).
    # Same reason: its per-(device, E) scratch is allocate-once and Triton
    # compiles on first launch, neither of which may happen inside a capture.
    # No-op unless VLLM_MOE_DETERMINISTIC_ALIGN selects the kernel path.
    from vllm.sm80.kernels.moe.align import (
        warmup_from_worker as warmup_moe_align,
    )

    warmed_experts = warmup_moe_align(worker)
    if warmed_experts:
        logger.info(
            "Warmed up the deterministic MoE alignment kernels for %d experts.",
            warmed_experts,
        )

    # The split-block Marlin MoE prefill (vllm/sm80/kernels/marlin_moe/
    # prefill.py). Same reason: its block-list buffers are
    # allocate-once per (device, E) and its Triton kernels compile on first
    # launch. Only with VLLM_AMPERE_MARLIN_PREFILL_WIDE or
    # VLLM_AMPERE_MARLIN_PREFILL_NARROW.
    if envs.VLLM_AMPERE_MARLIN_PREFILL_WIDE or envs.VLLM_AMPERE_MARLIN_PREFILL_NARROW:
        from vllm.sm80.kernels.marlin_moe.prefill import (
            warmup_from_worker as warmup_pp_marlin_prefill,
        )

        if warmup_pp_marlin_prefill(worker):
            logger.info("Warmed up the split-block Marlin MoE prefill lists.")

    # Run next so input-prep kernels JIT against pristine runner state.
    if enable_jit_warmup:
        kimi_k3_triton_warmup(worker)
        watermark_sample_warmup(worker)
        qwen4_exp_qsa_triton_warmup(worker)

    if enable_jit_warmup and current_platform.is_device_capability_family(100):
        _warmup_bf16x3_router_gemm(
            worker.get_model(),
            worker.scheduler_config.max_num_batched_tokens,
        )

    if current_platform.has_device_capability(90):
        _warmup_ll_bf16_router_gemm(worker.get_model())

    _warmup_gemm_rs_ar()

    if worker.vllm_config.kernel_config.enable_cutedsl_warmup:
        # TODO(roberto): Remove after registered CuTeDSL warmups are migrated
        # to the shared JIT warmup infrastructure.
        # https://github.com/vllm-project/vllm/pull/47451
        cutedsl_warmup()

    if process_local_only:
        return

    if current_platform.is_rocm():
        from vllm.model_executor.warmup.rocm_segmented_attn_autotune_warmup import (
            rocm_segmented_attn_autotune_warmup,
        )

        rocm_segmented_attn_autotune_warmup(worker)

    flashinfer_sparse_mla_decode_autotune_warmup(worker)
    deepseek_v4_sparse_mla_attention_warmup(worker)
    step5_dsa_warmup(worker)

    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        is_deep_gemm_supported() and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        deep_gemm_warmup(model, max_tokens)

    b12x_warmup(worker, cudagraph_capture_sizes)

    minimax_m3_msa_warmup(worker)

    # Allocate the exact decode-sized workspace, autotune cache misses, and
    # resolve every CUDA Graph bucket before capture begins.
    # Lazy import: flashinfer_pcie_ipc_all_reduce imports flashinfer.comm,
    # which initializes CUDA at import time and must not run at module scope.
    if envs.VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:
        from vllm.distributed.device_communicators import (
            flashinfer_pcie_ipc_all_reduce,
        )

        flashinfer_pcie_ipc_all_reduce.warmup_flashinfer_pcie_ipc_allreduce(worker)

    enable_flashinfer_autotune = (
        worker.vllm_config.kernel_config.enable_flashinfer_autotune
    )
    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if enable_flashinfer_autotune is False:
        logger.info_once("Skipping FlashInfer autotune because it is disabled.")
    elif has_flashinfer() and current_platform.has_device_capability(90):
        flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        # NOTE: This should be `any` instead of `all` but other hybrid attention
        # backends don't support this dummy run. Once we remove
        # `build_for_cudagraph_capture`, we can change it to `any`.
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info_once("Warming up FlashInfer attention.")
        mixed_batch_attention_warmup(worker)


def _flashinfer_autotune_skip_ops(runner: "GPUModelRunner") -> set[str] | None:
    if envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS is not None:
        return set(envs.VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS) or None

    from vllm.model_executor.kernels.linear import (
        FlashInferCuteDslNvFp4LinearKernel,
    )

    for module in runner.get_model().modules():
        for holder_name in ("quant_method", "scheme"):
            kernel = getattr(getattr(module, holder_name, None), "kernel", None)
            # CuTe-DSL mm_fp4 tuning JIT-compiles every tactic and its
            # fallback is already the heuristic; all mm_fp4 backends share
            # the "fp4_gemm" op name, so skip only when cute-dsl is selected.
            if isinstance(kernel, FlashInferCuteDslNvFp4LinearKernel):
                return {"fp4_gemm"}
    return None


def _flashinfer_deferred_moe_token_counts(
    runner: "GPUModelRunner",
) -> tuple[int, ...]:
    """Return bounded token counts that exercise deferred MoE finalization."""
    from vllm.model_executor.layers.fused_moe import MoERunner

    max_tokens = runner.scheduler_config.max_num_batched_tokens
    token_counts: list[int] = []
    for module in runner.get_model().modules():
        if not isinstance(module, MoERunner):
            continue

        moe_config = module.moe_config
        max_deferred_tokens = moe_config.defer_moe_finalize_max_num_tokens
        if moe_config.use_deferred_moe_finalize and max_deferred_tokens > 0:
            token_counts.append(min(max_tokens, max_deferred_tokens))

    return tuple(dict.fromkeys(token_counts))


def _flashinfer_autotune_token_counts(runner: "GPUModelRunner") -> tuple[int, ...]:
    max_tokens = runner.scheduler_config.max_num_batched_tokens
    # Tune the widest bucket set first so bounded passes reuse its configs.
    token_counts = [max_tokens]
    token_counts.extend(_flashinfer_deferred_moe_token_counts(runner))
    return tuple(dict.fromkeys(token_counts))


def _run_flashinfer_autotune_dummy_runs(
    runner: "GPUModelRunner", *, skip_attn: bool = False
) -> None:
    import vllm.utils.flashinfer as fi_utils

    dummy_run_kwargs = {"skip_attn": True} if skip_attn else {}
    for num_tokens in _flashinfer_autotune_token_counts(runner):
        # The drafter may use more tokens than this pass in the same dummy run.
        # Include its max M when building the autotune buckets.
        max_tuning_tokens = num_tokens
        speculator = getattr(runner, "speculator", None)
        if speculator is not None:
            num_query_per_req = speculator.num_query_per_req
            num_draft_reqs = min(
                num_tokens,
                runner.max_num_reqs,
                runner.max_num_tokens // num_query_per_req,
            )
            max_tuning_tokens = max(
                max_tuning_tokens, num_draft_reqs * num_query_per_req
            )

        tuning_buckets = fi_utils.flashinfer_get_hybrid_num_tokens_buckets(
            max_tuning_tokens
        )
        logger.info(
            "Running FlashInfer autotune with %d tokens and token buckets %s.",
            num_tokens,
            tuning_buckets,
        )
        # Round M up to match serving-time bucket selection for non-bucket M.
        # See https://github.com/flashinfer-ai/flashinfer/issues/5450
        with fi_utils.autotune(tuning_buckets=tuning_buckets, round_up=True):
            runner._dummy_run(
                num_tokens=num_tokens,
                skip_eplb=True,
                is_profile=True,
                randomize_inputs=True,
                **dummy_run_kwargs,
            )


def _autotune_cache_fingerprint(path: Path) -> tuple[str, int] | None:
    """Identify a saved autotune file by its FlashInfer metadata and size."""
    try:
        configs = json.loads(path.read_text())
        metadata = configs.pop("_metadata", None)
    except (OSError, ValueError, AttributeError):
        return None
    return json.dumps(metadata, sort_keys=True), len(configs)


def _all_ranks_have_matching_cache(path: Path, group) -> bool:
    """True iff every rank in ``group`` has its own, mutually consistent file."""
    fingerprint = _autotune_cache_fingerprint(path)
    if group.world_size == 1:
        return fingerprint is not None
    gathered: list[tuple[str, int] | None] = [None] * group.world_size
    torch.distributed.all_gather_object(gathered, fingerprint, group=group.cpu_group)
    return fingerprint is not None and all(f == fingerprint for f in gathered)


def _load_shared_autotune_cache(cache_path: Path, tuner, tune_group) -> bool:
    """Load the per-rank autotune cache into the tuner, group-wide.

    Returns True only when every rank of the tune group had a matching file
    and loaded it: the dummy-run sweep contains collectives, so skipping it
    is safe only if the decision is unanimous.
    """
    if not _all_ranks_have_matching_cache(cache_path, tune_group):
        return False
    loaded = tuner.load_configs(str(cache_path))
    if tune_group.world_size > 1:
        loaded_by_rank: list[bool | None] = [None] * tune_group.world_size
        torch.distributed.all_gather_object(
            loaded_by_rank, loaded, group=tune_group.cpu_group
        )
        loaded = all(loaded_by_rank)
    if not loaded:
        tuner.clear_cache()
    return loaded


def flashinfer_autotune(runner: "GPUModelRunner", *, skip_attn: bool = False) -> None:
    """Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.

    Args:
        runner: The model runner to tune.
        skip_attn: Additionally skip attention in the tuning dummy runs.

    With PP > 1, stages run different layers and may profile different ops,
    so each stage's TP group tunes separately with its own cache file;
    otherwise the world group tunes together. Per-tactic timings are
    averaged over the tuning group so all its ranks select the same tactic.

    Legacy results are persisted per rank: FlashInfer keys MoE entries by tp/ep rank
    (``MoERunner.get_cache_key_extras``), so one rank's file only hits on that
    rank. A rank with a cache hit skips the per-tactic reduce the others block
    in, so ranks keep loaded configs only if every rank in the tuning group
    has a matching file and successfully loads it.

    """
    from flashinfer.autotuner import AutoTuner, set_autotune_process_group

    import vllm.utils.flashinfer as fi_utils
    from vllm.distributed.parallel_state import (
        get_pp_group,
        get_tp_group,
        get_world_group,
    )

    world = get_world_group()
    pp_size = get_pp_group().world_size
    tune_group = get_tp_group() if pp_size > 1 else world
    is_leader = tune_group.rank_in_group == 0
    tuner = AutoTuner.get()

    autotune_kwargs: dict = {}
    skip_ops = _flashinfer_autotune_skip_ops(runner)
    if skip_ops:
        logger.info_once(
            "Skipping FlashInfer autotuning for ops %s",
            tuple(sorted(skip_ops)),
        )
        autotune_kwargs["skip_ops"] = skip_ops

    use_autotune_v2 = use_flashinfer_autotune_v2(runner)
    cache_hit = False
    if use_autotune_v2:
        from flashinfer import autotune_v2, autotune_v2_reload

        cache_root = resolve_flashinfer_autotune_v2_root()
        if is_leader:
            logger.info_once(
                "Using FlashInfer managed autotune cache (root=%s). Pass "
                "--disable-flashinfer-autotune to skip autotuning.",
                cache_root if cache_root is not None else "flashinfer default",
            )
        autotune_context = autotune_v2(
            mode="tune", cache_root=cache_root, **autotune_kwargs
        )
    else:
        cache_path = resolve_rank_flashinfer_autotune_file(runner)
        if is_leader:
            logger.info_once(
                "Using FlashInfer autotune cache file: %s. Pass "
                "--disable-flashinfer-autotune to skip autotuning.",
                cache_path,
            )

        cache_hit = _load_shared_autotune_cache(cache_path, tuner, tune_group)
        autotune_context = fi_utils.autotune(tune_mode=True, **autotune_kwargs)

    group = tune_group.cpu_group if tune_group.world_size > 1 else None
    set_autotune_process_group(group)
    try:
        with torch.inference_mode(), autotune_context:
            hisparse_enabled = (
                runner.vllm_config.attention_config.hisparse_config is not None
            )
            skip_attn = skip_attn or hisparse_enabled
            if hisparse_enabled:
                # HiSparse hot-buffer attention is bounded by decode batch
                # size, not the prefill-sized batch used for the full model.
                autotune_hisparse_flashinfer_attention(runner)
            if not cache_hit:
                _run_flashinfer_autotune_dummy_runs(runner, skip_attn=skip_attn)
            replayssm_autotune_warmup(runner)
            _autotune_kimi_k3_kda_qkvg(runner.get_model())
    finally:
        set_autotune_process_group(None)

    if world.world_size > 1:
        world.barrier()
    if use_autotune_v2:
        if world.world_size > 1:
            # All ranks serve the shared store's final state after publishing.
            autotune_v2_reload()
            world.barrier()
    elif tuner._dirty:
        tuner.save_configs(str(cache_path))
