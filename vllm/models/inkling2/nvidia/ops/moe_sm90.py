# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed experts on Hopper: Humming expert GEMMs (bf16
activations, NVFP4 / MXFP8 weights) in place of the TRT-LLM-gen ones.

Rows stay in the flat (token * topk + slot) order: the gate/up GEMM
gathers each row's token, the down GEMM reads and writes row by row, and
`ops.moe.combine` sums them with the identity `src2dst` (-1: not local).
"""

import json
import os
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size as _moe_align_block_size_compiled,
)
from vllm.platforms import current_platform


def moe_align_block_size(ids, block_size, num_experts, ignore_invalid_experts=False):
    # 170HX (sm_80), 21 Nov 2026: the image's compiled moe_align_block_size has sm_100 kernels only and silently
    # returns garbage before Hopper; use the Triton/torch version from the sm_80 shims there (same contract for this call).
    if current_platform.has_device_capability(90):
        return _moe_align_block_size_compiled(
            ids, block_size, num_experts, ignore_invalid_experts=ignore_invalid_experts
        )
    assert ignore_invalid_experts
    from vllm.models.inkling2.nvidia.ops.sm80.inkling_sm80_moe_align import (
        moe_align_block_size_fast,
    )

    return moe_align_block_size_fast(ids, block_size, num_experts)


from vllm.triton_utils import tl, triton

from .activation import _situ
from .linear import k_group_partials
from .moe import _load_bias, _local_experts, _select_experts

try:
    from vllm.models.inkling2.nvidia.ops.sm80.inkling_sm80_cuda import (
        HAS_CUDA_EXT as _SM80_HAS_CUDA,
    )
    from vllm.models.inkling2.nvidia.ops.sm80.inkling_sm80_cuda import (
        route_align_1k_cuda as _sm80_route_align_1k,
    )
    from vllm.models.inkling2.nvidia.ops.sm80.inkling_sm80_cuda import (
        select_align_cuda as _sm80_select_align,
    )
except Exception:
    _SM80_HAS_CUDA = False
    _sm80_route_align_1k = None
    _sm80_select_align = None

if TYPE_CHECKING:
    from vllm.utils.humming import HummingInputSchema

# Row block size when no Humming tuning range covers the row count.
_DEFAULT_BLOCK_M = 64


@triton.jit
def _select_kernel(
    Partials,
    Bias,
    ExpertBounds,
    IsPadding,
    Ids,
    Weights,
    Src2Dst,
    T,
    route_scale,
    E: tl.constexpr,
    E_POW2: tl.constexpr,
    GROUPS: tl.constexpr,
    TOPK: tl.constexpr,
    SORT: tl.constexpr,
):
    """Local expert ids (-1: not local), routing weights and identity
    `src2dst` of the flat rows of one token."""
    token = tl.program_id(0) + tl.arange(0, 1)
    bias = _load_bias(Bias, E, E_POW2)
    ids, weights = _select_experts(
        Partials, bias, token, T, route_scale, E, E_POW2, GROUPS, TOPK, SORT
    )
    padding = None
    if IsPadding is not None:
        padding = tl.load(IsPadding + token)
    local = _local_experts(
        ids, tl.load(ExpertBounds), tl.load(ExpertBounds + 1), padding
    )
    rows = token[:, None] * TOPK + tl.arange(0, TOPK)[None, :]
    tl.store(Ids + rows, local)
    tl.store(Weights + rows, weights)
    tl.store(Src2Dst + rows, tl.where(local >= 0, rows, -1))


def select_experts(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    expert_bounds: torch.Tensor,
    is_padding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The router and top-k of `ops.moe.route`, without its layout.

    Returns the local expert id of each token's slots `[T, topk]` (-1: not
    one of the local experts `expert_bounds`, or a padding token), their
    routing weights `[T * topk]` and the identity `src2dst` `[T * topk]`
    of the local rows (-1 elsewhere).
    """
    tokens = x.shape[0]
    experts = gate_weight.shape[0]
    partials = k_group_partials(x, gate_weight)
    ids = torch.empty((tokens, topk), dtype=torch.int32, device=x.device)
    weights = torch.empty(tokens * topk, dtype=torch.float32, device=x.device)
    src2dst = torch.empty(tokens * topk, dtype=torch.int32, device=x.device)
    _select_kernel[(tokens,)](
        partials,
        bias,
        expert_bounds,
        is_padding,
        ids,
        weights,
        src2dst,
        tokens,
        route_scale,
        E=experts,
        E_POW2=triton.next_power_of_2(experts),
        GROUPS=partials.shape[0],
        TOPK=topk,
        SORT=tokens <= 128,
        num_warps=4 if tokens <= 256 else 1,
    )
    return ids, weights, src2dst


def select_and_align_experts_sm80(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    num_local_experts: int,
    ep_size: int,
    is_padding: torch.Tensor | None = None,
):
    """sm_80 fused router + top-8 + block_m=8 align for single-GPU decode batches (T <= 32).

    Returns ((ids, weights, src2dst), (sorted_ids, expert_ids, num_padded)) or None to fall back.
    """
    if (
        not _SM80_HAS_CUDA
        or not current_platform.is_device_capability_family(80)
        or os.environ.get("INKLING_SM80_NO_CUDA_ROUTE") == "1"
        or ep_size != 1
        or num_local_experts != 128
        or topk != 8
        or x.dtype != torch.bfloat16
        or gate_weight.dtype != torch.bfloat16
        or bias.dtype != torch.float32
        or x.shape[1] != 2304
        or gate_weight.shape != (128, 2304)
        or not x.is_contiguous()
        or not gate_weight.is_contiguous()
        or not bias.is_contiguous()
    ):
        return None
    tokens = x.shape[0]
    if is_padding is not None and (
        is_padding.dtype != torch.bool or not is_padding.is_contiguous()
    ):
        is_padding = is_padding.to(torch.bool).contiguous()
    if 0 < tokens <= 8:
        ids, weights, src2dst, sorted_ids, expert_ids, num_padded = (
            _sm80_route_align_1k(x, gate_weight, bias, is_padding, route_scale)
        )
        return (ids, weights, src2dst), {8: (sorted_ids, expert_ids, num_padded)}
    if 8 < tokens <= 64:
        partials = k_group_partials(x, gate_weight)
        bm = 16 if tokens > 28 else 8
        ids, weights, src2dst, s_ids, e_ids, n_pad = _sm80_select_align(
            partials, bias, is_padding, route_scale, block_size=bm
        )
        return (ids, weights, src2dst), {bm: (s_ids, e_ids, n_pad)}
    return None


@triton.jit
def _swiglu_kernel(
    GateUp,
    Weights,
    Src2Dst,
    Out,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
):
    """bf16 softcapped SwiGLU of a local row times its routing weight."""
    row = tl.program_id(0)
    if tl.load(Src2Dst + row) < 0:
        return
    start = tl.program_id(1) * BLOCK
    gamma = tl.load(Weights + row)
    values = _situ(GateUp, row, start, gamma, N, BLOCK, GATE_CAP, UP_CAP)
    tl.store(Out + row * N + start + tl.arange(0, BLOCK), values.to(tl.bfloat16))


def swiglu(
    gate_up: torch.Tensor,
    weights: torch.Tensor,
    src2dst: torch.Tensor,
    gate_cap: float,
    up_cap: float,
) -> torch.Tensor:
    """`activation.swiglu_quantize`'s SiTU of the interleaved (gate, up)
    columns times the routing weights, in bf16, for the local rows (the
    others are left unset)."""
    rows, n = gate_up.shape[0], gate_up.shape[1] // 2
    block = 256 if n % 256 == 0 else 128
    assert n % block == 0
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=gate_up.device)
    _swiglu_kernel[(rows, n // block)](
        gate_up,
        weights,
        src2dst,
        out,
        N=n,
        BLOCK=block,
        GATE_CAP=gate_cap,
        UP_CAP=up_cap,
        num_warps=2,
    )
    return out


class HummingExperts:
    """Humming tensors and kernel configs of the two expert projections.

    Args:
        weights: Per projection (`w13`, `w2`), the row-major checkpoint
            weights `[E, N, K / pack]`, block scales `[E, N, K / group]`
            and, for NVFP4, the per-expert global scales `[E]` (else None).
        shapes: `(N, K)` of each projection.
        nvfp4: NVFP4 (else MXFP8) weights.
        input_schema: Quantized input contract, or None for BF16 activations.

    """

    def __init__(
        self,
        weights: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]],
        shapes: tuple[tuple[int, int], tuple[int, int]],
        nvfp4: bool,
        input_schema: "HummingInputSchema | None" = None,
    ) -> None:
        from vllm.utils.humming import (
            GemmType,
            HummingInputSchema,
            ModeloptMxfp8WeightSchema,
            ModeloptNvfp4WeightSchema,
            get_heuristics_config,
            prepare_and_transform_humming_tensors,
        )

        self.configs = {}
        self.tensors: dict[str, dict[str, torch.Tensor]] = {}
        self.tunings: dict[str, list] = {}
        self.tuning_strs: dict[str, str] = {}
        for name, (n, k) in zip(("w13", "w2"), shapes):
            weight, scale, global_scale = weights[name]
            num_experts = weight.shape[0]
            tensors = {"weight": weight, "weight_scale": scale}
            if nvfp4:
                source = ModeloptNvfp4WeightSchema()
                tensors["weight_scale_2"] = global_scale
            else:
                source = ModeloptMxfp8WeightSchema()
            schema, tensors = source.convert_humming(
                tensors=tensors,
                shape_n_stacks=[n],
                shape_k_stacks=[k],
                param_dtype=torch.bfloat16,
                num_experts=num_experts,
            )
            config, tensors = prepare_and_transform_humming_tensors(
                tensors=tensors,
                shape_n=n,
                shape_k=k,
                pad_n_to_multiple=256,
                pad_k_to_multiple=128,
                input_schema=input_schema or HummingInputSchema(),
                weight_schema=schema,
                num_experts=num_experts,
                torch_dtype=torch.bfloat16,
                device=weight.device,
            )
            assert config.shape_n == n and config.shape_k == k, (n, k)
            tuning = get_heuristics_config(
                layer_config=config,
                use_f16_accum=False,
                use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
                gemm_type=GemmType.INDEXED,
            )
            # Stream-K adds the K slices into the output with bf16 atomics,
            # whose order (and so the result) changes from run to run.
            for _, _, kernel_config in tuning:
                kernel_config["use_stream_k"] = False
            # sm_80: 8-bucket real-skew Cold-L2 CG decode (1u/2u/4u/8u) + 180W steady-state prefill (2k/4k/8k/12k):
            # Explicitly sets num_sms = device SM count (74 on 170HX) so small-bucket heuristics never halve active SMs
            if not current_platform.has_device_capability(90):
                base_kc = dict(tuning[-1][2])
                base_kc["use_stream_k"] = False
                base_kc["num_sms"] = int(
                    torch.cuda.get_device_properties(
                        weight.device
                    ).multi_processor_count
                )
                legacy = json.loads(os.environ.get("INKLING_SM80_HUMMING") or "null")
                if name == "w13":
                    # 1u (T=5, rows=40): real skewed routing favors [8,128,128]s3c4 NVFP4 & [8,64,128]s3c8 MXFP8
                    default_s = (
                        {
                            "block_shape": [8, 128, 128],
                            "warp_shape": [8, 64, 64],
                            "num_stages": 3,
                            "num_ctas_per_sm": 4,
                        }
                        if nvfp4
                        else {
                            "block_shape": [8, 64, 128],
                            "warp_shape": [8, 32, 64],
                            "num_stages": 3,
                            "num_ctas_per_sm": 8,
                        }
                    )
                    # 2u (T=10, rows=80)
                    default_b2 = (
                        {
                            "block_shape": [8, 128, 128],
                            "warp_shape": [8, 64, 64],
                            "num_stages": 3,
                            "num_ctas_per_sm": 4,
                        }
                        if nvfp4
                        else {
                            "block_shape": [8, 64, 128],
                            "warp_shape": [8, 32, 64],
                            "num_stages": 3,
                            "num_ctas_per_sm": 4,
                        }
                    )
                    # 4u (T=20, rows=160): [8,128,128]s3c4 beats [8,64,128] by +8..20 tok/s across all 4u workloads
                    default_b4 = {
                        "block_shape": [8, 128, 128],
                        "warp_shape": [8, 64, 64],
                        "num_stages": 3,
                        "num_ctas_per_sm": 4,
                    }
                    # 8u (T=40, rows=320): BM=16 beats BM=8 on skewed routing (175 vs 195us NVFP4; 245 vs 305us MXFP8)
                    # Alignment for BM=16 is emitted for free inside sm80_select_align_mcta_kernel
                    default_b8 = {
                        "block_shape": [16, 128, 128],
                        "warp_shape": [16, 64, 64],
                        "num_stages": 3,
                        "num_ctas_per_sm": 4,
                    }
                    default_m = (
                        {
                            "block_shape": [32, 128, 64],
                            "warp_shape": [32, 64, 32],
                            "num_stages": 4,
                            "num_ctas_per_sm": 2,
                        }
                        if nvfp4
                        else {
                            "block_shape": [32, 256, 64],
                            "warp_shape": [32, 64, 32],
                            "num_stages": 4,
                            "num_ctas_per_sm": 1,
                        }
                    )
                    # T=2048 (rows=16384, 180W steady)
                    default_p2k = {
                        "block_shape": [64, 128, 64],
                        "warp_shape": [64, 64, 32],
                        "num_stages": 3,
                        "num_ctas_per_sm": 2,
                    }
                    # T=4096 & T=8192 (rows=24576..73728, 180W steady): BK=64 beats BK=32 on real skewed routing
                    default_p8k = (
                        {
                            "block_shape": [128, 256, 64],
                            "warp_shape": [64, 64, 64],
                            "num_stages": 4,
                            "num_ctas_per_sm": 1,
                        }
                        if nvfp4
                        else {
                            "block_shape": [128, 256, 64],
                            "warp_shape": [64, 64, 64],
                            "num_stages": 2,
                            "num_ctas_per_sm": 1,
                        }
                    )
                    # T=12288 (rows > 73728, 180W steady)
                    default_p12k = (
                        {
                            "block_shape": [128, 256, 64],
                            "warp_shape": [64, 64, 64],
                            "num_stages": 4,
                            "num_ctas_per_sm": 1,
                        }
                        if nvfp4
                        else {
                            "block_shape": [128, 256, 64],
                            "warp_shape": [64, 64, 64],
                            "num_stages": 2,
                            "num_ctas_per_sm": 1,
                        }
                    )
                    small_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W13_S") or "null"
                        )
                        or legacy
                        or default_s
                    )
                    b2_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W13_B2") or "null"
                        )
                        or legacy
                        or default_b2
                    )
                    b4_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W13_B4") or "null"
                        )
                        or legacy
                        or default_b4
                    )
                    b8_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W13_B") or "null"
                        )
                        or legacy
                        or default_b8
                    )
                    med_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W13_M") or "null"
                        )
                        or default_m
                    )
                    pre2k_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W13_P2K") or "null"
                        )
                        or default_p2k
                    )
                    pre8k_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W13_P") or "null"
                        )
                        or default_p8k
                    )
                    pre12k_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W13_P12K") or "null"
                        )
                        or default_p12k
                    )
                else:
                    # 1u (T=5, rows=40)
                    default_s = (
                        {
                            "block_shape": [8, 64, 128],
                            "warp_shape": [8, 32, 64],
                            "num_stages": 4,
                            "num_ctas_per_sm": 4,
                        }
                        if nvfp4
                        else {
                            "block_shape": [8, 64, 128],
                            "warp_shape": [8, 32, 64],
                            "num_stages": 3,
                            "num_ctas_per_sm": 4,
                        }
                    )
                    # 2u (T=10, rows=80)
                    default_b2 = (
                        {
                            "block_shape": [8, 128, 64],
                            "warp_shape": [8, 64, 32],
                            "num_stages": 4,
                            "num_ctas_per_sm": 4,
                        }
                        if nvfp4
                        else {
                            "block_shape": [8, 64, 128],
                            "warp_shape": [8, 32, 64],
                            "num_stages": 3,
                            "num_ctas_per_sm": 4,
                        }
                    )
                    # 4u (T=20, rows=160)
                    default_b4 = (
                        {
                            "block_shape": [8, 128, 64],
                            "warp_shape": [8, 64, 32],
                            "num_stages": 4,
                            "num_ctas_per_sm": 4,
                        }
                        if nvfp4
                        else {
                            "block_shape": [8, 64, 128],
                            "warp_shape": [8, 32, 64],
                            "num_stages": 3,
                            "num_ctas_per_sm": 4,
                        }
                    )
                    # 8u (T=40, rows=320): BM=16 beats BM=8 (NVFP4 96 vs 103us; MXFP8 122 vs 146us)
                    default_b8 = (
                        {
                            "block_shape": [16, 128, 64],
                            "warp_shape": [16, 64, 32],
                            "num_stages": 4,
                            "num_ctas_per_sm": 4,
                        }
                        if nvfp4
                        else {
                            "block_shape": [16, 64, 128],
                            "warp_shape": [16, 32, 64],
                            "num_stages": 3,
                            "num_ctas_per_sm": 4,
                        }
                    )
                    default_m = {
                        "block_shape": [32, 128, 64],
                        "warp_shape": [32, 64, 32],
                        "num_stages": 4,
                        "num_ctas_per_sm": 2,
                    }
                    default_p2k = {
                        "block_shape": [64, 128, 64],
                        "warp_shape": [64, 64, 32],
                        "num_stages": 4,
                        "num_ctas_per_sm": 2,
                    }
                    default_p8k = {
                        "block_shape": [128, 256, 64],
                        "warp_shape": [64, 64, 64],
                        "num_stages": 2,
                        "num_ctas_per_sm": 1,
                    }
                    default_p12k = (
                        {
                            "block_shape": [128, 256, 64],
                            "warp_shape": [64, 64, 64],
                            "num_stages": 3,
                            "num_ctas_per_sm": 1,
                        }
                        if nvfp4
                        else {
                            "block_shape": [128, 256, 64],
                            "warp_shape": [64, 64, 64],
                            "num_stages": 2,
                            "num_ctas_per_sm": 1,
                        }
                    )
                    small_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W2_S") or "null"
                        )
                        or legacy
                        or default_s
                    )
                    b2_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W2_B2") or "null"
                        )
                        or legacy
                        or default_b2
                    )
                    b4_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W2_B4") or "null"
                        )
                        or legacy
                        or default_b4
                    )
                    b8_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W2_B") or "null"
                        )
                        or legacy
                        or default_b8
                    )
                    med_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W2_M") or "null"
                        )
                        or default_m
                    )
                    pre2k_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W2_P2K") or "null"
                        )
                        or default_p2k
                    )
                    pre8k_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W2_P") or "null"
                        )
                        or default_p8k
                    )
                    pre12k_ov = (
                        json.loads(
                            os.environ.get("INKLING_SM80_HUMMING_W2_P12K") or "null"
                        )
                        or default_p12k
                    )
                tuning = [
                    (0, 64, {**base_kc, **small_ov}),
                    (64, 120, {**base_kc, **b2_ov}),
                    (120, 224, {**base_kc, **b4_ov}),
                    (224, 1840, {**base_kc, **b8_ov}),
                    (1840, 4096, {**base_kc, **med_ov}),
                    (4096, 24576, {**base_kc, **pre2k_ov}),
                    (24576, 73728, {**base_kc, **pre8k_ov}),
                    (73728, 1073741824, {**base_kc, **pre12k_ov}),
                ]
            self.configs[name] = config
            self.tensors[name] = tensors
            self.tunings[name] = tuning
            self.tuning_strs[name] = json.dumps(tuning)
        self.num_experts = num_experts
        self.compute_config = json.dumps(
            {
                "use_batch_invariant": envs.VLLM_BATCH_INVARIANT,
                "use_f16_accum": False,
                "gemm_type": GemmType.INDEXED.value,
            }
        )
        self.locks = torch.zeros(2048, dtype=torch.int32, device=weight.device)

    def _block_m(self, name: str, rows: int) -> int:
        for lower, upper, config in self.tunings[name]:
            if lower < rows <= upper:
                return config["block_shape"][0]
        return _DEFAULT_BLOCK_M

    def _gemm(
        self,
        name: str,
        inputs: torch.Tensor,
        rows: int,
        alignment: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        top_k: int,
        valid_rows: int,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.humming import humming_forward

        config = self.configs[name]
        tensors = self.tensors[name]
        out = torch.empty(
            (rows, config.shape_n), dtype=torch.bfloat16, device=inputs.device
        )
        sorted_ids, expert_ids, num_tokens_padded = alignment
        return humming_forward(
            config,
            inputs=inputs,
            weight=tensors["weight"],
            weight_scale=tensors.get("weight_scale"),
            weight_scale_2=tensors.get("weight_scale_2"),
            bias=bias,
            outputs=out,
            locks=self.locks,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_padded=num_tokens_padded,
            top_k=top_k,
            valid_shape_m=valid_rows,
            compute_config=self.compute_config,
            tuning_config=self.tuning_strs[name],
        )

    def __call__(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        src2dst: torch.Tensor,
        num_global_experts: int,
        gate_cap: float,
        up_cap: float,
        alignment_8: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """bf16 `[T * topk, H]` down-projection rows of the local slots of
        `ids` (`select_experts`); the other rows are left unset."""
        tokens, topk = ids.shape
        rows = tokens * topk
        # Expected local rows, which pick the kernel configs.
        valid_rows = -(-rows * self.num_experts // num_global_experts)
        block_m = {name: self._block_m(name, valid_rows) for name in ("w13", "w2")}
        alignments: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        if alignment_8 is not None:
            if isinstance(alignment_8, dict):
                for sz, al in alignment_8.items():
                    if sz in block_m.values():
                        alignments[sz] = al
            elif 8 in block_m.values():
                alignments[8] = alignment_8
        for size in set(block_m.values()):
            if size not in alignments:
                alignments[size] = moe_align_block_size(
                    ids, size, self.num_experts, ignore_invalid_experts=True
                )[:3]
        use_fused_situ = (
            not current_platform.has_device_capability(90)
            and os.environ.get("INKLING_SM80_NO_FUSED_SITU") != "1"
            and abs(gate_cap - 4.0) < 1e-6
            and abs(up_cap - 25.0) < 1e-6
            and weights.dtype == torch.float32
            and weights.is_contiguous()
            and self.configs["w13"].shape_n % 2 == 0
        )
        if use_fused_situ:
            raw = self._gemm(
                "w13",
                x,
                rows,
                alignments[block_m["w13"]],
                topk,
                valid_rows,
                bias=weights,
            )
            half_n = self.configs["w13"].shape_n // 2
            activated = raw.view(-1)[: rows * half_n].view(rows, half_n)
            del raw
        else:
            gate_up = self._gemm(
                "w13", x, rows, alignments[block_m["w13"]], topk, valid_rows
            )
            activated = swiglu(gate_up, weights, src2dst, gate_cap, up_cap)
            del gate_up
        return self._gemm(
            "w2", activated, rows, alignments[block_m["w2"]], 1, valid_rows
        )
