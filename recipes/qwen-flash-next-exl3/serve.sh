#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

export VLLM_CACHE_ROOT=/cache/vllm
export TRITON_CACHE_DIR=/cache/triton
export CUDA_CACHE_PATH=/cache/cuda
export VLLM_PP_LAYER_PARTITION=26,22
export VLLM_PP_HOP_NO_METADATA=1
export VLLM_PP_PACKED_HOP=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_CUSTOM_ALLREDUCE_FLAGS=1
export VLLM_CUSTOM_ALLREDUCE_PCIE_P2P=0
export VLLM_EXL3_PLE_PLACEMENT=ssd
export VLLM_EXL3_PLE_NATIVE_READ=1
export VLLM_EXL3_PLE_OVERLAP=1
export VLLM_EXL3_DENSE_MMA=1
export VLLM_EXL3_DENSE_PREFILL_TC=1
export VLLM_EXL3_DENSE_SPLITK=1
export VLLM_EXL3_MOE_PREFILL_TC=1
export VLLM_EXL3_MOE_V2=1
export VLLM_FLA_AUTOTUNE_SKIP_SPILLING=1
export VLLM_QWEN4_QSA_TOPK_CANONICAL=1
export VLLM_THIN_GEMM=1
export VLLM_THIN_GEMM_GLUON=1

exec /opt/venv/bin/vllm serve /model \
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen3.8-flash-next-exl3-pp2 \
  --distributed-executor-backend mp --trust-remote-code \
  --quantization exl3 --load-format exl3 --mm-encoder-tp-mode data \
  --tensor-parallel-size 1 --pipeline-parallel-size 2 --enable-expert-parallel \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --attention-config '{"indexer_kv_dtype":"bf16"}' \
  --engram-config '{"cpu_offload":false}' \
  --max-model-len 262144 --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 3460 --max-num-seqs 8 \
  --long-prefill-token-threshold 0 --enable-prompt-tokens-details \
  --enable-prefix-caching --prefix-match-unit 4 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"disable_eagle_block_drop":true}' \
  "$@"
