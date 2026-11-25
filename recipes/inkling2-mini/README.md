# Inkling 2 Mini on one CMP 170HX card

Thinking Machines' Inkling 2 Mini from its released checkpoint (NVFP4 routed experts on Humming
kernels), with INT8 attention, LM head and embeddings made at load, an FP8 KV cache, the model's
own DSpark module as the drafter, a 524,288-token context and text and image input, on one card.
The runnable recipe lives in
[AlexBrandes/cmp170hx-inkling2-mini](https://github.com/AlexBrandes/cmp170hx-inkling2-mini).
This page pins that release rather than duplicating its scripts.

## Requirements

- Linux x86-64, one CMP 170HX card, 40 GB or 64 GB, suitable cooling.
- NVIDIA driver 580 or newer and Docker with the NVIDIA Container Toolkit.
- About 20 GB of free storage for the checkpoint, plus the engine image and compilation caches.
- About 12 GB of free host RAM.
- The recipe was measured at 170 W on a 40 GB card on a **PCIe Gen2 ×4 link**.

## Run

```bash
git clone https://github.com/AlexBrandes/cmp170hx-inkling2-mini.git
cd cmp170hx-inkling2-mini
git checkout --detach 783a52b5cefff909f33c5c80ea6be42c6a61ce21
./download.sh
./start.sh                # GPU=1 ./start.sh picks another card
```

`download.sh` fetches the pinned checkpoint from Thinking Machines' repo and checks every file.
`start.sh` uses Docker and leaves the card's power limit alone. The first start compiles kernels
for several minutes; later starts reuse `./cache`. Do not start this alongside another service
using the same GPU.

The API is local-only at `http://127.0.0.1:8000`, model `inkling-2-mini`. Wait for readiness
before sending a request:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Explain speculative decoding in two sentences."}],"max_tokens":128,"temperature":0,"reasoning_effort":"none"}'
```

Stop only the recipe's container with `./stop.sh` from that checkout.
For settings, reasoning effort and tool calls, use the
[pinned recipe README](https://github.com/AlexBrandes/cmp170hx-inkling2-mini/blob/783a52b5cefff909f33c5c80ea6be42c6a61ce21/README.md).

## Release identity

| Component | Pin |
| --- | --- |
| Recipe | `783a52b5cefff909f33c5c80ea6be42c6a61ce21` |
| Upstream vLLM | wheel for `a0d7356b98e929c17051bac44f5a086e4117974d` |
| vLLM branch | `inkling2/vllm` at `5a859096f3c6de44193a4609bc5486ddfddd5eac` |
| Plugin | `inkling2/plugin` at `9c9be42b7785aec33e703147b2f9eaa38a92a87c` |
| Image | `ghcr.io/alexbrandes/cmp170hx-inkling2-mini@sha256:b799892a60a7485bcecdc0459d9394fcbb324587b3ed72ab5651b9d4612ce504` |
| Checkpoint | `thinkingmachines/Inkling-2-Mini` at `c9d028302cd0f035fdf8e7e91f089a4ce94d2195`, as released |
| Layout | one card, TP1 |
| Context / concurrency | 524,288 tokens / up to 16 sequences |
| Speculation / graphs | DSpark (INT4 at load), 4 draft tokens for one or two requests, 3 for more / FULL_DECODE_ONLY |

The [public results](https://github.com/AlexBrandes/cmp170hx-inkling2-mini/blob/783a52b5cefff909f33c5c80ea6be42c6a61ce21/results/2026-11-24.json)
and [benchmark tables](https://github.com/AlexBrandes/cmp170hx-inkling2-mini/tree/783a52b5cefff909f33c5c80ea6be42c6a61ce21#benchmarks)
cover three public-v1 runs, image input, needle retrieval, the 524,288-token window, and GSM8K,
HumanEval and GPQA Diamond.

## Credits and licenses

The engine is Apache-2.0; the standalone recipe is MIT, and the checkpoint is Apache-2.0. The
sm_80 prefill attention, conv/mix, controller, routing and Humming expert kernels are ported from
[inkling-2-mini-sm80](https://github.com/snailbrainx/inkling-2-mini-sm80) by snailbrainx, and
the Ampere path comes from [HyperInkling](https://github.com/Pento95/HyperInkling) by Pento95.
The recipe's [NOTICE](https://github.com/AlexBrandes/cmp170hx-inkling2-mini/blob/783a52b5cefff909f33c5c80ea6be42c6a61ce21/NOTICE)
credits these, upstream vLLM's Inkling 2 code, Morrowmake's vllm-sm80, Thinking Machines and the
benchmark authors. See the [fork credits](../../README.md#credits) for CMPUnlocker, Morrowmake,
PixelML, and the wider CMP community.
