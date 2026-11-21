# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import TYPE_CHECKING, ClassVar

from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class Inkling2DSparkSpeculatorConfig(DSparkSpeculatorConfig):
    speculator_class = "vllm.models.inkling2.speculator.Inkling2DSparkSpeculator"
    # HYPERINKLING_LOOKUP scans the token history every draft step: keep it on the GPU
    # (max_num_seqs x max_model_len int32, 6 MB at 6 x 262k).
    prefer_dense_token_history: ClassVar[bool] = (
        os.environ.get("HYPERINKLING_LOOKUP") == "1"
    )

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        super().adapt_draft_model_config(spec)
        from vllm.models.inkling2.nvidia.ops import use_sm12x_path

        if use_sm12x_path():
            # The target's clipped, unit-scale FP8 cache does not describe the
            # draft's unnormalized values. Keep the draft cache in BF16.
            if spec.kv_cache_dtype is None:
                spec.kv_cache_dtype = "auto"
            elif spec.kv_cache_dtype not in ("auto", "bfloat16"):
                raise ValueError("Inkling2 DSpark on SM12 requires BF16 draft KV")
