# GLM-5.3-Flash EXL3 on two CMP 170HX cards

turboderp's EXL3 2.60 bpw quantization of GLM-5.3-Flash, **PP2 / TP1**, with
the DFlash2 drafter at **3 draft tokens** and **FULL_DECODE_ONLY** CUDA graphs.
The model is multimodal: image and video inputs are served, audio is not.
Context is 225,280 tokens with up to four sequences.

This recipe runs on the fork's current engine: upstream vLLM
[`a0d7356b98`](https://github.com/vllm-project/vllm/commit/a0d7356b98e929c17051bac44f5a086e4117974d),
the vLLM branch [`vllm/a0d7356b98`](https://github.com/AlexBrandes/vllm/tree/vllm/a0d7356b98)
at [`2a4e137909`](https://github.com/AlexBrandes/vllm/commit/2a4e1379093a84fd0852c197bb76c09a34a0473f), and the
[`vllm-cmp170hx` plugin](../../README.md) at
[`947ef7c7e1`](https://github.com/AlexBrandes/vllm/commit/947ef7c7e1a96135521dce27cf953178816da42e).

## Requirements

- Linux x86-64 and two CMP 170HX cards exposing 64 GiB each.
- The measured configuration has **74 SMs/card, 170 W/card and PCIe Gen2 ×4
  links**, with no GPU peer-to-peer transport. The tensor-core prefill and
  decode paths are gated on the measured shapes; other shapes keep the scalar
  EXL3 kernels.
- NVIDIA driver 580 or newer (CUDA 13.0), Docker and the NVIDIA Container Toolkit.
- About 110 GB of NVMe storage for the checkpoint, 2.3 GB for the drafter, plus
  the engine image (8.3 GB) and compilation caches.
- A working Hugging Face `hf` CLI to download the checkpoint and drafter.

The launcher does **not** modify drivers, power limits, PCIe settings, host page
cache, or other containers. Set up and qualify your cards separately. Do not
launch while another service owns the same GPUs.

## 1. Build the image

From a checkout of this fork's `plugin` branch:

```bash
git clone --branch plugin https://github.com/AlexBrandes/vllm.git
cd vllm
docker build -t glm53-flash-exl3:947ef7c recipes/glm53-flash-exl3
```

The [Dockerfile](Dockerfile) starts from NVIDIA's CUDA 13.0 development image,
pinned by digest, and assembles the engine from its three pins: upstream vLLM's
prebuilt wheel for the base commit, the vLLM branch's Python files over it, and
the plugin, whose EXL3 extension it compiles against the installed PyTorch.
Each download is checked against its SHA-256. The build needs network access
and CPU/RAM, **not a GPU**; it uses two compiler jobs and took 17 minutes on
the measured host. No model weights, credentials or local files are copied into
the image.

## 2. Download the checkpoint and the drafter

These directories are outside the source tree:

```bash
export MODEL_DIR="$HOME/models/GLM-5.3-Flash-exl3-sc_2.60bpw"
export DRAFT_DIR="$HOME/models/GLM-5.3-Flash-DFlash2"
export CACHE_DIR="$HOME/.cache/glm53-flash-exl3-pp2"
mkdir -p "$MODEL_DIR" "$DRAFT_DIR" "$CACHE_DIR"
hf download turboderp/GLM-5.3-Flash-exl3 \
  --revision 24781209f2862a939cc21e01d256c943bd1a58a3 --local-dir "$MODEL_DIR"
hf download incoai/GLM-5.3-Flash-DFlash2 \
  --revision bf582e4eacc1810f76656d1811693ff6c6737d2a --local-dir "$DRAFT_DIR"
```

The checkpoint revision is the `sc_2.60bpw` branch of turboderp's repository:
2.5-bit routed experts, 6-bit dense and attention matrices, LM head and vision
tower. The checkpoint keeps GLM-5.3-Flash's MIT license. The drafter is
licensed **CC BY-NC-ND 4.0**: non-commercial use only, and it may not be
redistributed; read its model card before use.

## 3. Start

Keep `MODEL_DIR`, `DRAFT_DIR` and `CACHE_DIR` set to the absolute paths from step 2:

```bash
docker run --rm --name glm53-flash-exl3-pp2 \
  --gpus all --shm-size 16g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/cache -e HF_HUB_OFFLINE=1 -e CUDA_VISIBLE_DEVICES=0,1 \
  -p 127.0.0.1:8000:8000 \
  -v "$MODEL_DIR:/model:ro" -v "$DRAFT_DIR:/draft:ro" -v "$CACHE_DIR:/cache" \
  -v "$PWD/recipes/glm53-flash-exl3/chat_template.jinja:/model/chat_template.jinja:ro" \
  glm53-flash-exl3:947ef7c
```

The last mount replaces the checkpoint's chat template with
[this recipe's](chat_template.jinja): it honours
`chat_template_kwargs.enable_thinking=false` and renders image and video
placeholders, which the checkpoint's template drops. It must replace the
tokenizer's own template (not `--chat-template`), because vLLM's GLM reasoning
parser reads that one.

This runs in the foreground. First startup compiles kernels and captures graphs;
keep the cache directory between starts. The API is exposed **only on localhost**,
without an API key. Do not expose it to an untrusted network without authentication.
On a larger host, select the intended two visible GPUs instead of `0,1`.

From another terminal, after the server reports readiness:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash-exl3-2.60bpw","messages":[{"role":"user","content":"Explain pipeline parallelism in two sentences."}],"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

Stop only this container with `docker stop -t 120 glm53-flash-exl3-pp2`.

## Serving configuration

[serve.sh](serve.sh) contains the full environment and arguments, matching the
measured configuration apart from paths and bind address.

| Setting | Value |
| --- | --- |
| Engine | upstream `a0d7356b98`, vLLM branch `2a4e137909`, plugin `947ef7c7e1` |
| Checkpoint | `turboderp/GLM-5.3-Flash-exl3` `sc_2.60bpw`, revision `24781209f2862a939cc21e01d256c943bd1a58a3` |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2`, revision `bf582e4eacc1810f76656d1811693ff6c6737d2a`; BF16, `stream_mean` auxiliary state |
| Pipeline / tensor parallelism | 2 / 1; layer partition 23/22 |
| Context / sequences | 225,280 tokens / 4 sequences |
| Batch / prefill beside decodes | 2,048 tokens per step / 512 prefill tokens while decoding |
| Speculation | DFlash, 3 draft tokens |
| Graphs | FULL_DECODE_ONLY |
| KV block size | 4,608 tokens |
| GPU memory utilization | 0.97 |
| Tool calls / reasoning | `glm47` parsers, auto tool choice, strict tool calling (vLLM default) |

The EXL3 tensor-core paths are on for prefill (routed experts and dense 6-bit
families) and for small-row decode (routed experts and every dense family, with
split-K). Under pipeline parallelism the scheduler spreads decodes over the
in-flight micro-batches and never mixes prefill and decode tokens in one step,
because those kernels are selected per phase.

## Measurements

Measured on **2× CMP 170HX, 170 W/card, PCIe Gen2 ×4/card**, on an image built
with `docker build --no-cache` from this Dockerfile, with benchlib's
SparkDash-compatible `public-v1` (SparkDash `3e6d4ef` protocol: greedy,
thinking off, 400 output tokens, engine-direct). Values are medians of three
complete runs. Decode in tokens/s; C2–C8 are aggregate throughput across that
many simultaneous requests (at most four run at once, so C8 queues).

| Content | C1 | C2 total | C4 total | C8 total |
| --- | ---: | ---: | ---: | ---: |
| code | 76.5 | 109.9 | 151.4 | 137.0 |
| json | 70.9 | 113.2 | 164.6 | 156.5 |
| prose | 57.8 | 92.4 | 139.6 | 123.0 |
| structured | 83.0 | 122.5 | 191.9 | 161.1 |

Cold prefill, tokens/s:

| 4K | 8K | 16K | 32K | 64K | 128K |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,741 | 1,852 | 1,842 | 1,836 | 1,823 | 1,803 |

[results.json](results.json) has every cell of the three runs, single-stream
decode after 4K–128K prompts (benchlib's addition), the run IDs, the source
fingerprint, the measured image ID and the EXL3 extension hash. A rebuild
compiles its own extension and has its own image identity.

### Quality

Same image and session, greedy, thinking off, sent to the engine directly.

| Check | Result |
| --- | --- |
| HumanEval, pass@1 (164 problems) | **142/164** (86.6%) |
| GSM8K (1,319 test questions) | **1,228/1,319** (93.1%) |
| Needle retrieval at 4K, 16K, 32K and 64K | 16/16 |
| 225,280-token window | 225,279 prompt tokens + 1 output token accepted; 225,280 refused |
| Image and video input | 7/7 (three images from 64×64 to 4000×3000, two images in one prompt, two short videos) |

- **HumanEval** uses EvalPlus's instruct prompt as one chat message (1,024
  tokens) and OpenAI's `check_correctness` program (3 s limit). The Codex
  paper's raw-prompt completion mode scores 134/164.
- **GSM8K** runs vLLM's `tests/evals/gsm8k/gsm8k_eval.py` in its chat mode:
  5-shot, 1,024 tokens, the last integer in the reply. The script's default
  completion mode scores 1,190/1,319.
- With thinking off, replies about video can open with a short description of
  the frames before the answer.

These are checks of this quantized checkpoint served by this engine, not a
model evaluation; compare with the model card's own numbers with that in mind.

## Credits

The engine and this recipe are Apache-2.0. EXL3 derives from turboderp's
ExLlamaV3 (MIT); its notices ship with the plugin in `licenses/exl3/`. The sm_80
GLM-5.3 code is derived from Morrowmake's vllm-sm80. See
[the fork credits](../../README.md#credits) for CMPUnlocker, Morrowmake,
PixelML, upstream vLLM, the checkpoint and drafter authors, and SparkDash.
