# Qwen3.8-Flash-Next W4A16 on two CMP 170HX cards

Intel's W4A16 AutoRound checkpoint, pipeline parallelism across two cards, text,
image and video input, and the N-gram (PLE) table read from NVMe through the page
cache. The runnable recipe lives in
[AlexBrandes/cmp170hx-qwen38-flash-next](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next).
This page pins that release rather than duplicating its scripts.

## Requirements

- Linux x86-64, two CMP 170HX cards exposing 64 GiB each, suitable cooling.
- NVIDIA driver 580 or newer and Docker with the NVIDIA Container Toolkit.
- About 180 GB of free NVMe storage for the checkpoint, plus the image/cache.
- Enough host RAM for the PLE page cache; the measured host has 93 GB RAM.
- The recipe was measured at 170 W/card on **PCIe Gen2 ×4 links**, without GPU P2P.

## Run

```bash
git clone https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next.git
cd cmp170hx-qwen38-flash-next
git checkout --detach 501279ab072692f26f5638d134a9d40482f1788b
./download.sh
./start.sh
```

`download.sh` fetches the pinned model. `start.sh` uses Docker and, by default,
invokes `host-tune.sh` through sudo: it sets the cards to 170 W, changes NVMe
readahead, drops the host page cache, and compacts memory. **Those are host-wide
changes.** Use `HOST_TUNE=0 ./start.sh` to leave the host settings alone; the
published performance may not apply. Do not start this alongside another service
using the same GPUs.

The API is local-only at `http://127.0.0.1:8000`, model
`qwen3.8-flash-next-w4a16`. Wait for readiness before sending a request:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-w4a16","messages":[{"role":"user","content":"Explain pipeline parallelism in two sentences."}],"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

Stop only the recipe's container with `./stop.sh` from that checkout.
For settings, reasoning, tool calls, and troubleshooting, use the
[pinned recipe README](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next/blob/501279ab072692f26f5638d134a9d40482f1788b/README.md).

## Release identity

| Component | Pin |
| --- | --- |
| Recipe | `501279ab072692f26f5638d134a9d40482f1788b` |
| Upstream vLLM | wheel for `a0d7356b98e929c17051bac44f5a086e4117974d` |
| vLLM branch | `vllm/a0d7356b98` at `ea2ef399a614ea72b65b15804000342f629a850f` |
| Plugin | `plugin` at `bbba5c27cf9e367e3652b876d53ea7e1c80f16e5` |
| Image | `ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:a87f558b57e60b4f3c796d7be11d12199ce77f393df6ed1c40181ce932717757` |
| Checkpoint | `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` at `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897` |
| Layout | PP2 / TP1, layer partition 26/22 |
| Context / concurrency | 262,144 tokens / up to 8 sequences |
| Speculation / graphs | MTP5 with an NVFP4 draft head / FULL_AND_PIECEWISE |
| PLE table | vLLM `disk_offload`, rows gathered by the plugin's threaded page-cache reader |

The [public results](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next/blob/501279ab072692f26f5638d134a9d40482f1788b/results/2026-11-05.json)
and [benchmark tables](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next/tree/501279ab072692f26f5638d134a9d40482f1788b#benchmarks)
cover three public-v1 runs of this image on November 5, 2026, plus correctness,
image and video input, needle, context window, HumanEval and GSM8K.

## Credits and licenses

The engine is Apache-2.0; the standalone recipe is MIT, and model weights have
their own license. Its [NOTICE](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next/blob/501279ab072692f26f5638d134a9d40482f1788b/NOTICE)
credits the PLE reader's community origins (polyuij42-del and nguyenthimy2022kg-alt),
Morrowmake's vllm-sm80 layout, Qwen, Intel AutoRound, and the benchmark authors.
See the [fork credits](../../README.md#credits) for CMPUnlocker, Morrowmake,
PixelML, and the wider CMP community.
