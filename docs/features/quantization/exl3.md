# EXL3

EXL3 loads the checkpoint's packed trellis matrices and original FP16 scales without requantizing them. Linear and whole-expert MoE operators accept and return BF16 tensors, with FP32 transforms and accumulation. There is no FP16 activation adapter, persistent decoded-weight cache, or dependency on the standalone ExLlamaV3 runtime.

Each GPU block decodes a 16-row slab of trellis tiles once into shared memory and applies it to a block of up to 128 rows; routed experts are grouped by local expert on the device every step, so each active expert is decoded once per step and block of rows. Every output element is still one FP32 fused multiply-add chain over the input channels in increasing order, so a row's result does not depend on how many rows share the launch or which block shape was chosen. The extension keeps the original scalar kernels as test-only reference operators (`linear_out_reference`, `moe_out_reference`) that the fast operators must match bit for bit.

EXL3 runs on the extension `vllm._exl3_C`, built for SM 8.0 only; selecting EXL3 on any other GPU is an error. The vLLM build does not compile it: build it ahead of time with `python csrc/exl3/build_standalone.py --out vllm` from the source tree, with the PyTorch that serves (CUDA-enabled PyTorch, a matching CUDA toolkit, a C++20 compiler and Ninja; no GPU needed). Serving never compiles this extension. Its operators are registered through the non-stable `TORCH_LIBRARY` API, so loading refuses a library whose recorded PyTorch version, CUDA major version or C++ ABI differs from the running PyTorch. A successful build, CPU helper check or Meta dispatch is not GPU numerical, graph, quality or capacity validation. Validate those paths before deployment.

## Enablement and checkpoint loading

EXL3 is registered by the fork's vLLM plugin (`vllm/sm80/plugin.py`, entry point `sm80` of the `vllm-sm80-plugin` distribution in `plugins/sm80/`): the quantization method `exl3`, which a checkpoint whose `quant_method` is `exl3` selects, and the load format `exl3`. Without the plugin such a checkpoint is an unknown quantization method at startup, never an unquantized model.

`VLLM_EXL3` defaults to `0`. Set it to `1` to select EXL3; a selected EXL3 checkpoint with the switch disabled is an error, not a fallback to another quantization method. A missing or incompatible compiled extension is also an error. EXL3 needs `--load-format exl3`, and models with a vision tower also need `--mm-encoder-tp-mode data`; building the model without them is an error. Off the CUDA platform, model configuration refuses EXL3 checkpoints.

```bash
VLLM_EXL3=1 vllm serve ./exl3-checkpoint \
  --quantization exl3 --load-format exl3 --dtype bfloat16
```

The checkpoint must already be a local safetensors directory. Loading inventories actual file headers, verifies packed geometry/codebook markers and accounts for consumed, file-backed and explicitly unused tensors; the counts are logged per model. The inventory is released after the last model that uses it is loaded: the target, or the MTP draft when a speculative draft is configured (the draft is built with the target's configuration). An index cannot hide auxiliary files or override their headers. Integer rates from 1 through 8 bits are supported; 1.5-, 2.5- and 3.5-bit formats require the `mul1` codebook.

Tensor-parallel cuts must preserve complete 128-channel transform blocks. Routed experts use whole-expert expert parallelism rather than splitting each expert's transform. Pipeline parallelism other than one and LoRA are rejected. Ordinary embeddings and active non-PLE weights remain on the GPU; general CPU weight offload is not supported by this path.

Dense layers share one fixed 32 MiB FP32 scratch per GPU, plus one for the shared-expert stream, and process rows in chunks that fit it, so neither batch size nor graph capture grows it. The routed-expert scratch is sized for the rows the selected dispatch delivers: with tensor and expert parallelism but no data parallelism, each rank's own tokens.

## Qwen3.8-Flash-Next

The model integration preserves packed constituents of fused projections, replicates shared experts when their widths cannot be legally split, retains full indexer keys, and uses data-parallel vision encoding. Sparse attention computes Q/gate, K, V and the indexer Q/K in one fused projection; Q/gate splits across ranks, K/V follow the key-value heads, and every rank holds the whole indexer matrix. Native MTP shares the target embedding and head when the checkpoint does not provide dedicated ones. The loader requires the compressed ngram table and MTP hyper-connection mixer patch in addition to the main shards.

For the two-card configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_EXL3=1 \
VLLM_EXL3_PLE_PLACEMENT=ssd \
VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
vllm serve ./qwen-exl3-checkpoint \
  --quantization exl3 --load-format exl3 --mm-encoder-tp-mode data \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --enable-expert-parallel \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --attention-config '{"indexer_kv_dtype":"bf16"}' \
  --engram-config '{"cpu_offload":false}' \
  --max-model-len 262144 --gpu-memory-utilization 0.95 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The MTP count is configurable; this example is not a measured optimum. Keep vision enabled. Validate eager execution before graph replay and test the native context independently of a shorter diagnostic boot. `tests/models/qwen4_exp/test_exl3_construction.py` builds the target and MTP draft at two-rank expert-parallel shapes on the CPU from the checkpoint's configuration and tensor headers when `VLLM_TEST_EXL3_QWEN_CHECKPOINT` names the checkpoint directory.

`VLLM_QWEN4_ROUTER_FP32=1` selects the existing `GateLinear` BF16-by-BF16 GEMM with FP32 accumulation and FP32 router logits, without changing the stored router weights. It defaults to `0`; the model logs the selected precision at construction. This switch applies to both the target and native MTP router.

`VLLM_QWEN4_MTP_PER_STREAM_NORM=1` computes a separate RMS for each native MTP hidden stream. It preserves the checkpoint's affine weight for each channel and applies the Gemma `(1 + weight)` scaling in FP32 before returning BF16. The default `0` retains one RMS over the entire concatenated hidden stack. The draft model logs the selected normalization. These are precision/normalization comparison paths, not validated default changes.

## Compressed PLE placement

`VLLM_EXL3_PLE_PLACEMENT` accepts `ssd` (default), `ram` or `device`. Placement is explicit and logged; RAM/device modes are comparisons, not automatic fallbacks.

- `ssd`: retain the file descriptor and read only requested owned rows with buffered positional reads into bounded reusable pinned staging. A failed or truncated row read is retried up to `VLLM_EXL3_PLE_READ_RETRIES` times (default 3, backoff 1 ms doubling); see below for a read that still fails.
- `ram`: explicitly pin the compressed rank shard and expose its accelerator view.
- `device`: explicitly load the compressed rank shard to the GPU.

All modes use the same GPU row decoder. Its layout comes from the PLE file: `K` from the codec metadata, the channel count from the head bias, and the row size (one FP16 scale word plus a ring of `channels * K` bits) from the trellis. Ownership and head padding are masked before bias; only decoded BF16 rows participate in the embedding reduction. Repeated IDs preserve request order. Original auxiliary tensors stay GPU-resident.

SSD reads occur at eager Python breaks between CUDA graph segments: ID completion, CPU reads, H2D readiness, decode and output-copy completion have explicit ordering. This is not a captured CUDA host callback. Staging cannot be reused until the previous output consumer has completed. Warm stable workspaces before capture; growth after capture locking is rejected.

A row read that still fails after its retries fails only the requests that use the row; the engine keeps serving the rest. The reader stages the row with an FP16 NaN scale, so the decoder emits a NaN row. NaN survives the tensor-parallel sum of row shards, so every rank sees which tokens failed, whichever rank owns the row. Those tokens' PLE embeddings are zeroed before the forward continues, so NaN cannot reach other requests. The model runner reports the requests that own the tokens. The scheduler finishes them with `FINISHED_ERROR` (finish reason `error`) and drops from the prefix cache the KV blocks they computed from that step on. It also fails any running request that already reused those blocks, so no request is served from a wrong embedding. The worker logs the row and the OS error; the scheduler logs the request IDs. The reader stays usable, and later requests read the row again. Errors while reading the table or its auxiliaries during startup still fail startup. `ram` and `device` read the table only at startup.

The PLE file is opened with `posix_fadvise(POSIX_FADV_RANDOM)`, so the kernel does not read ahead around requested rows. Buffered reads still hit the page cache, so warm runs mostly read RAM. Set `VLLM_EXL3_PLE_COLD=1` to call `POSIX_FADV_DONTNEED` on the table at setup (no root needed; pages that another process has mapped stay resident). Page-cache residency of the table (via `mincore`) is logged at setup, after the cold drop and at process exit. `Exl3PLEEmbedding.page_cache_residency()` returns `(resident, total)` pages for sampling between runs.

## Third-party material

Codec adaptations preserve the upstream MIT grant and modification credit in [`vllm/third_party/exl3`](../../../vllm/third_party/exl3). Both the `LICENSE` and `NOTICE` are included in source and binary distributions.
