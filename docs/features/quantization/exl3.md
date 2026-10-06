# EXL3

EXL3 loads the checkpoint's packed trellis matrices and original FP16 scales without requantizing them. Linear and whole-expert MoE operators accept and return BF16 tensors, with FP32 transforms and accumulation. There is no FP16 activation adapter, persistent decoded-weight cache, or dependency on the standalone ExLlamaV3 runtime.

The CUDA build includes `vllm._exl3_C`, built for SM 8.0 only; selecting EXL3 on any other GPU is an error. It is compiled ahead of time; serving never compiles this extension. A successful build, CPU helper check or Meta dispatch is not GPU numerical, graph, quality or capacity validation. Validate those paths before deployment.

## Enablement and checkpoint loading

`VLLM_EXL3` defaults to `0`. Set it to `1` to select EXL3; a selected EXL3 checkpoint with the switch disabled is an error, not a fallback to another quantization method. A missing or incompatible compiled extension is also an error.

```bash
VLLM_EXL3=1 vllm serve ./exl3-checkpoint \
  --quantization exl3 --dtype bfloat16
```

The checkpoint must already be a local safetensors directory. Loading inventories actual file headers, verifies packed geometry/codebook markers and accounts for consumed, file-backed and explicitly unused tensors; the counts are logged and the inventory is released once the model (and, separately, the MTP draft) is loaded. An index cannot hide auxiliary files or override their headers. Integer rates from 1 through 8 bits are supported; 1.5-, 2.5- and 3.5-bit formats require the `mul1` codebook.

Tensor-parallel cuts must preserve complete 128-channel transform blocks. Routed experts use whole-expert expert parallelism rather than splitting each expert's transform. Pipeline parallelism other than one and LoRA are rejected. Ordinary embeddings and active non-PLE weights remain on the GPU; general CPU weight offload is not supported by this path.

Dense layers share one fixed 32 MiB FP32 scratch per GPU, plus one for the shared-expert stream, and process rows in chunks that fit it, so neither batch size nor graph capture grows it. The routed-expert scratch is sized for the rows the selected dispatch delivers: with tensor and expert parallelism but no data parallelism, each rank's own tokens.

## Qwen3.8-Flash-Next

The model integration preserves packed constituents of fused projections, replicates shared experts when their widths cannot be legally split, retains full indexer keys, and uses data-parallel vision encoding. Native MTP shares the target embedding and head when the checkpoint does not provide dedicated ones. The loader requires the compressed ngram table and MTP hyper-connection mixer patch in addition to the main shards.

For the two-card configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_EXL3=1 \
VLLM_EXL3_PLE_PLACEMENT=ssd \
VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
vllm serve ./qwen-exl3-checkpoint \
  --quantization exl3 \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --enable-expert-parallel \
  --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --attention-config '{"indexer_kv_dtype":"bf16"}' \
  --engram-config '{"cpu_offload":false}' \
  --max-model-len 262144 --gpu-memory-utilization 0.95 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

The MTP count is configurable; this example is not a measured optimum. Keep vision enabled. Validate eager execution before graph replay and test the native context independently of a shorter diagnostic boot.

## Compressed PLE placement

`VLLM_EXL3_PLE_PLACEMENT` accepts `ssd` (default), `ram` or `device`. Placement is explicit and logged; RAM/device modes are comparisons, not automatic fallbacks.

- `ssd`: retain the file descriptor and read only requested owned rows with buffered positional reads into bounded reusable pinned staging. Read errors and truncated valid rows fail the request.
- `ram`: explicitly pin the compressed rank shard and expose its accelerator view.
- `device`: explicitly load the compressed rank shard to the GPU.

All modes use the same GPU row decoder. Its layout comes from the PLE file: `K` from the codec metadata, the channel count from the head bias, and the row size (one FP16 scale word plus a ring of `channels * K` bits) from the trellis. Ownership and head padding are masked before bias; only decoded BF16 rows participate in the embedding reduction. Repeated IDs preserve request order. Original auxiliary tensors stay GPU-resident.

SSD reads occur at eager Python breaks between CUDA graph segments: ID completion, CPU reads, H2D readiness, decode and output-copy completion have explicit ordering. This is not a captured CUDA host callback. Staging cannot be reused until the previous output consumer has completed. Warm stable workspaces before capture; growth after capture locking is rejected.

The PLE file is opened with `posix_fadvise(POSIX_FADV_RANDOM)`, so the kernel does not read ahead around requested rows. Buffered reads still hit the page cache, so warm runs mostly read RAM. Set `VLLM_EXL3_PLE_COLD=1` to call `POSIX_FADV_DONTNEED` on the table at setup (no root needed; pages that another process has mapped stay resident). Page-cache residency of the table (via `mincore`) is logged at setup, after the cold drop and at process exit. `Exl3PLEEmbedding.page_cache_residency()` returns `(resident, total)` pages for sampling between runs.

## Third-party material

Codec adaptations preserve the upstream MIT grant and modification credit in [`vllm/third_party/exl3`](../../../vllm/third_party/exl3). Both the `LICENSE` and `NOTICE` are included in source and binary distributions.
