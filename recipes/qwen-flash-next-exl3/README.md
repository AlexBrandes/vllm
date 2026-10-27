# Qwen3.8-Flash-Next EXL3 on two CMP 170HX cards

EXL3 5.05 bpw, **PP2 / TP1**, native **MTP3**, and **FULL_AND_PIECEWISE**
CUDA graphs. The compressed N-gram (PLE) table stays on NVMe; the engine stages
its rows before full decode graph replay. Vision stays enabled on stage zero.

This recipe uses our published engine at
[`344e5c2`](https://github.com/AlexBrandes/vllm/commit/344e5c252fbdda1670676f3a2ccfb1a3914b0274),
including the PP2 dense/MoE tensor-core prefill and decode improvements.
It does not use the older W4A16 mmap reader or replace that production recipe.

## Requirements

- Linux x86-64 and two CMP 170HX cards exposing 64 GiB each.
- The measured EXL3 configuration has **74 SMs/card, 170 W/card, and PCIe Gen2 ×4
  links**, with no GPU peer-to-peer transport. Dense tensor-core fast paths are
  gated on the supported hardware; a 70-SM configuration is not the measured case.
- NVIDIA driver 580 or newer, Docker, and the NVIDIA Container Toolkit.
- NVMe storage: about 120 GiB for the checkpoint, plus the engine image and
  compilation caches. Leave build space as well. The measured host has 93 GB RAM.
- A working Hugging Face `hf` CLI to download the checkpoint.

The launcher does **not** modify drivers, power limits, PCIe settings, host page
cache, or other containers. Set up and qualify your cards separately. Do not
launch while another service owns the same GPUs.

## 1. Build the image

From a checkout of this fork's `cmp170hx` branch:

```bash
git clone --branch cmp170hx https://github.com/AlexBrandes/vllm.git
cd vllm
docker build -t qwen-flash-next-exl3:344e5c2 recipes/qwen-flash-next-exl3
```

The [Dockerfile](Dockerfile) pins Morrowmake's runtime image by digest and our
engine source by commit and archive checksum. It retains the compatible upstream
binary extensions and rebuilds the EXL3 extension against that image's PyTorch.
The build needs network access and CPU/RAM, **not a GPU**. It uses two compiler
jobs. No local source overlays, model weights, credentials, or Git stores are
copied into the image.

This is a source-built recipe, not a claim that a new prebuilt image is available
on GHCR. The original measured image and source snapshot are recorded in
[results.json](results.json); a local rebuild has its own image identity.

## 2. Download the checkpoint

Run from the same checkout. These directories are outside the source tree:

```bash
export MODEL_DIR="$HOME/models/Qwen3.8-Flash-Next-exl3"
export CACHE_DIR="$HOME/.cache/qwen-flash-next-exl3-pp2"
mkdir -p "$MODEL_DIR" "$CACHE_DIR"
hf download turboderp/Qwen3.8-Flash-Next-exl3 \
  --revision 7cef615f7fd3681295b68848018876dfabc336c7 \
  --local-dir "$MODEL_DIR"
```

Download the whole pinned checkpoint, including its compressed PLE and MTP
auxiliary tensors, tokenizer, and processor files. Do not mix files from other
bit-rate branches. Model weights retain the checkpoint's license.

## 3. Start

Keep `MODEL_DIR` and `CACHE_DIR` set to absolute paths from step 2:

```bash
docker run --rm --name qwen-flash-next-exl3-pp2 \
  --gpus all --shm-size 1g \
  --user "$(id -u):$(id -g)" --workdir /cache \
  -e HOME=/cache -e CUDA_VISIBLE_DEVICES=0,1 \
  -p 127.0.0.1:8000:8000 \
  -v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR:/cache" \
  qwen-flash-next-exl3:344e5c2
```

This runs in the foreground. First startup compiles kernels and captures graphs;
keep the cache directory between starts. The API is exposed **only on localhost**,
without an API key. Do not expose it to an untrusted network without authentication.
On a larger host, select the intended two visible GPUs instead of `0,1`.

From another terminal, after the server reports readiness:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-exl3-pp2","messages":[{"role":"user","content":"Explain pipeline parallelism in two sentences."}],"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

Stop only this container with:

```bash
docker stop -t 120 qwen-flash-next-exl3-pp2
```

`--rm` removes the stopped container, not the checkpoint or compilation cache.

## Serving configuration

[serve.sh](serve.sh) contains the full environment and arguments, matching the
October 25 public benchmark configuration apart from paths and bind address.

| Setting | Value |
| --- | --- |
| Engine | `344e5c252fbdda1670676f3a2ccfb1a3914b0274` |
| Checkpoint | `turboderp/Qwen3.8-Flash-Next-exl3`, 5.05 bpw, revision `7cef615f7fd3681295b68848018876dfabc336c7` |
| Pipeline / tensor parallelism | 2 / 1; layer partition 26/22 |
| Context | 262,144 tokens |
| KV / indexer cache | BF16 / BF16 |
| Speculation | MTP, 3 draft tokens, `disable_eagle_block_drop=true` |
| Graphs | FULL_AND_PIECEWISE; no forced eager execution |
| Prefix matching | 4 tokens; prefix caching enabled |
| Batch limit / sequences | 3,460 tokens / 8 sequences |
| GPU memory utilization | 0.95 |
| PLE | Compressed SSD reader, native reads and overlap enabled |
| Vision | Enabled; data-parallel encoder on the first pipeline stage |

The dense MMA, dense prefill, dense split-K, MoE prefill, and MoE decode switches
are enabled. The PP2 prefill path reserves larger scratch buffers to avoid
unnecessary chunking. Unsupported shapes retain their existing paths.
See [the engine documentation](../../docs/features/quantization/exl3.md#experimental-pp2)
for dispatch gates, state ownership, PLE failure handling, and graph behavior.

## Measurements

[results.json](results.json) contains all C1/C2/C4/C8 decode cells and the
4K–128K cold-prefill ladder, with the three individual values and their medians.
The run IDs, source fingerprint, original image ID, and native extension hash
are included. Rates come from benchlib's SparkDash-compatible `public-v1` runs:
greedy, thinking off, 400 output tokens, engine-direct. C2/C4/C8 rates are
aggregate throughput.

The full 262K serving limit is a configuration value, not a claim that the speed
ladder measured prefill at 262K. Quality, vision, and graph validation are separate
from these throughput tables. Rebuilding the image does not itself rerun those
gates, and this recipe does not alter our production deployment.

## Credits

The engine and this recipe are Apache-2.0. The pinned base image retains its
own component licenses and Morrowmake's notices. EXL3 derives from turboderp's
work; the [bundled EXL3 notices](../../plugins/sm80/licenses/exl3) remain intact.
See [the fork credits](../../README.md#credits) for original CMPUnlocker,
Morrowmake, PixelML, upstream vLLM, and the community PP/PLE work.
