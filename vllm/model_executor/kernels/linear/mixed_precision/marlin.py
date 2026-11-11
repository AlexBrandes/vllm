# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fusion import cmp_int8_act_fusion as _cmp_act
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_SUPPORTED_GROUP_SIZES,
    apply_gptq_marlin_linear,
    marlin_act_int8_process_scales,
    marlin_make_empty,
    marlin_pad_dim,
    marlin_pad_qweight,
    marlin_pad_scales,
    marlin_padded_nk,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_zero_points,
    query_marlin_supported_quant_types,
    unpack_cols,
)
from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig


class MarlinLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        # Marlin uses inline PTX, so it can only be compatible with Nvidia
        if not current_platform.is_cuda():
            return False, "Marlin only supported on CUDA"

        quant_types = query_marlin_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by"
                f"  Marlin, supported types are: {quant_types}",
            )

        if c.group_size not in MARLIN_SUPPORTED_GROUP_SIZES:
            return (
                False,
                f"Group size ({c.group_size}) not supported by "
                "Marlin, supported group sizes are: "
                f"{MARLIN_SUPPORTED_GROUP_SIZES}",
            )

        # A group straddling TP ranks cannot be fixed by padding.
        if (
            c.group_size != -1
            and c.group_size < c.full_weight_shape[0]
            and c.partition_weight_shape[0] % c.group_size != 0
        ):
            return False, (
                f"in_features per partition {c.partition_weight_shape[0]} is "
                f"not divisible by group_size = {c.group_size}."
            )

        # Tile misalignment is fixed by zero-padding at weight prep.
        return True, None

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = getattr(layer, self.w_q_name).device
        c = self.config
        is_a_8bit = c.act_type is not None and c.act_type.itemsize == 1

        if is_a_8bit:
            assert c.weight_type == scalar_types.uint4b8, (
                "W8A8 is not supported by marlin kernel."
            )

        if c.act_type == torch.float8_e4m3fn:
            ops.marlin_int4_fp8_preprocess(getattr(layer, self.w_q_name), inplace=True)
            # in place: a new tensor would move the scales on a direct reload
            getattr(layer, self.w_s_name).data.mul_(512)

        # Qwen3.8-27B (port of HyperQwen's
        # marlin-int8-negative-scales.patch): the int8-activation kernel reads
        # the int16-requantized group scales as *unsigned* int16, so groups
        # with a negative scale (about half of an AutoRound symmetric export)
        # come out as garbage. Fold the sign into the int4 codes: s -> -s,
        # uint4b8 code v -> 16 - v (q -> -q), clamped to 15, so q == -8 in a
        # flipped group becomes -7 (one LSB).
        if (
            c.act_type == torch.int8
            and c.group_size != -1
            and not getattr(c, "has_g_idx", False)
            and not c.zero_points
        ):
            _wq = getattr(layer, self.w_q_name)
            _ws = getattr(layer, self.w_s_name)
            permute_param_layout_(_wq, input_dim=0, output_dim=1, packed_dim=0)
            permute_param_layout_(_ws, input_dim=0, output_dim=1)
            _neg = _ws.data < 0
            if bool(_neg.any()):
                _q = _wq.data
                _rows_per_group = c.group_size // 8
                _neg_rows = _neg.repeat_interleave(_rows_per_group, dim=0)[
                    : _q.shape[0]
                ]
                _shifts = torch.arange(0, 32, 4, device=_q.device, dtype=torch.int32)
                _nib = (_q.unsqueeze(-1) >> _shifts) & 0xF
                _flip = torch.clamp(16 - _nib, max=15)
                _nib = torch.where(_neg_rows.unsqueeze(-1), _flip, _nib)
                _out = torch.zeros_like(_q)
                for _i in range(8):
                    _out |= _nib[..., _i] << (4 * _i)
                _wq.data = _out
                _ws.data = torch.where(_neg, -_ws.data, _ws.data)
                del _nib, _flip, _out

        size_k, size_n = c.partition_weight_shape
        padded_n, padded_k = marlin_padded_nk(size_n, size_k, c.group_size)

        # Default name since marlin requires empty parameter for zp,
        # TODO: remove this requirement from marlin (allow optional tensors)
        if self.w_zp_name is None:
            self.w_zp_name = "w_zp"

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = ops.gptq_marlin_repack(
                marlin_pad_qweight(
                    x.data.contiguous(), size_n, size_k, padded_n, padded_k
                ),
                size_k=padded_k,
                size_n=padded_n,
                num_bits=c.weight_type.size_bits,
                is_a_8bit=is_a_8bit,
            )
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = marlin_permute_scales(
                marlin_pad_scales(
                    x.data.contiguous(),
                    size_n,
                    size_k,
                    padded_n,
                    padded_k,
                    c.group_size,
                ),
                size_k=padded_k,
                size_n=padded_n,
                group_size=c.group_size,
                is_a_8bit=is_a_8bit,
            )

            if c.group_size == -1:
                num_groups = 1
            else:
                num_groups = c.partition_weight_shape[0] // c.group_size

            if c.act_type == torch.int8 and num_groups > 1:
                x.data, input_global_scale = marlin_act_int8_process_scales(x.data)
                replace_parameter(layer, "input_global_scale", input_global_scale)
            else:
                layer.input_global_scale = None
            return x

        if c.zero_points:
            grouped_k = size_k // c.group_size if c.group_size != -1 else 1
            padded_grouped_k = padded_k // c.group_size if c.group_size != -1 else 1
            self._transform_param(
                layer,
                self.w_zp_name,
                lambda x: marlin_zero_points(
                    marlin_pad_scales(
                        unpack_cols(
                            x.t(),
                            c.weight_type.size_bits,
                            grouped_k,
                            size_n,
                        ),
                        size_n,
                        size_k,
                        padded_n,
                        padded_k,
                        c.group_size,
                    ),
                    size_k=padded_grouped_k,
                    size_n=padded_n,
                    num_bits=c.weight_type.size_bits,
                    is_a_8bit=is_a_8bit,
                ),
            )
        else:
            # Marlin takes an empty zp; a loaded unused one keeps its data.
            self._retire_unused_zp(layer)
            self.w_zp_name = "w_zp"
            replace_parameter(layer, self.w_zp_name, marlin_make_empty(device))
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)

        if hasattr(layer, "bias") and layer.bias is not None:
            replace_parameter(
                layer,
                "bias",
                marlin_permute_bias(marlin_pad_dim(layer.bias, size_n, padded_n)),
            )

        # Qwen3.8-27B: take a pre-quantized int8 per-token
        # activation from fused producers (cmp_int8_act_fusion.py) when the
        # layer runs W4A8 with no padding (VLLM_CMP_INT8_ACT_FUSION=1).
        if self.input_quant_key() is not None and padded_k == size_k:
            layer._input_quant_key = self.input_quant_key()

    def input_quant_key(self):
        c = self.config
        if (
            _cmp_act.ENABLED
            and c.act_type == torch.int8
            and c.weight_type == scalar_types.uint4b8
            and not c.zero_points
        ):
            return _cmp_act.KEY
        return None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp = self._get_weight_params(layer)
        qa = (
            as_quantized_activation(x, self.input_quant_key())
            if isinstance(x, QuantizedActivation)
            else None
        )
        if qa is not None:
            out_shape = qa.orig_shape[:-1] + (c.partition_weight_shape[1],)
            return apply_gptq_marlin_linear(
                input=qa.data,
                weight=w_q,
                weight_scale=w_s,
                weight_zp=w_zp,  # type: ignore
                workspace=None,
                wtype=c.weight_type,
                input_size_per_partition=c.partition_weight_shape[0],
                output_size_per_partition=c.partition_weight_shape[1],
                # the int8 producers (cmp_int8_act_fusion) fold this layer's
                # input_global_scale into qa.scale already
                input_global_scale=None,
                bias=bias,
                input_dtype=c.act_type,
                input_scales=qa.scale,
                orig_dtype=qa.orig_dtype,
            ).reshape(out_shape)
        return apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            workspace=None,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=c.partition_weight_shape[1],
            input_global_scale=getattr(layer, "input_global_scale", None),
            bias=bias,
            input_dtype=c.act_type,
        )
