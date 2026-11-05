#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# GLM-5.3-Flash EXL3 2.60 bpw, PP2 / TP1, DFlash2 k3, on two CMP 170HX cards.
# /model: turboderp/GLM-5.3-Flash-exl3 (sc_2.60bpw); /draft: incoai/GLM-5.3-Flash-DFlash2.
set -euo pipefail

export VLLM_CACHE_ROOT=/cache/vllm
export TRITON_CACHE_DIR=/cache/triton
export CUDA_CACHE_PATH=/cache/cuda

# Pipeline split: 23 target layers on the first card, 22 and the drafter on the second.
export VLLM_PP_LAYER_PARTITION=23,22
# Scheduling: spread decodes over the in-flight micro-batches, and never mix
# prefill and decode tokens in one step (the tensor-core paths are per phase).
export VLLM_PP_SPREAD_DECODES=1
export VLLM_PP_HOMOGENEOUS_STEPS=1
# sm_80 KDA chunked prefill (64 heads per card).
export VLLM_SM80_KDA_PREFILL=1
# EXL3 tensor-core prefill: routed experts and the dense 6-bit families.
export VLLM_EXL3_MOE_PREFILL_TC=1
export VLLM_EXL3_DENSE_PREFILL_TC=1
export VLLM_EXL3_DENSE_PREFILL_ONE_PRODUCT=float
# EXL3 small-row tensor-core decode: routed experts and every dense family.
export VLLM_EXL3_MOE_DECODE_TC=1
export VLLM_EXL3_MOE_DECODE_TC_MODE=full
export VLLM_EXL3_DENSE_DECODE_TC=1
export VLLM_EXL3_DENSE_DECODE_TC_FAMILIES=all
export VLLM_EXL3_DENSE_DECODE_SPLITK=1
export VLLM_EXL3_DECODE_VARIANT=off

exec vllm serve /model \
  --host 0.0.0.0 --port 8000 \
  --served-model-name glm-5.3-flash-exl3-2.60bpw \
  --quantization exl3 --load-format exl3 --dtype bfloat16 \
  --pipeline-parallel-size 2 --tensor-parallel-size 1 --mm-encoder-tp-mode data \
  --max-model-len 225280 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --prefill-chunk-with-decodes 512 --block-size 4608 \
  --gpu-memory-utilization 0.97 \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm47 \
  --enable-prompt-tokens-details \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"method": "dflash", "model": "/draft", "num_speculative_tokens": 3, "quantization": null, "draft_load_config": {"load_format": "safetensors"}}' \
  --hf-overrides '{"text_config": {"aux_hidden_state_tensor": "stream_mean"}}' \
  "$@"
