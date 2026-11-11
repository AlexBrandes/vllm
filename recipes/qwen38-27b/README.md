# Qwen3.8-27B on one CMP 170HX card

dbirks' W4A16 quantization of Qwen3.8-27B on a single card, 64 GB or 40 GB, with
the DFlash2 drafter at seven draft tokens, text, image and video input, and the
plugin's INT8 prefill and W4A8 Marlin paths. The runnable recipe lives in
[AlexBrandes/cmp170hx-qwen38-27b](https://github.com/AlexBrandes/cmp170hx-qwen38-27b).
This page pins that release rather than duplicating its scripts.

## Requirements

- Linux x86-64, one CMP 170HX card (64 GB or 40 GB), suitable cooling.
- NVIDIA driver 580 or newer and Docker with the NVIDIA Container Toolkit.
- About 27 GB of free storage while `download.sh` runs (about 16 GB afterwards:
  checkpoint and drafter), plus the image/cache, and about 26 GB of RAM while it
  builds the INT8 embedding table.
- The recipe was measured at 170 W on a 40 GB card over a **PCIe Gen2 ×4 link**.

## Run

```bash
git clone https://github.com/AlexBrandes/cmp170hx-qwen38-27b.git
cd cmp170hx-qwen38-27b
git checkout --detach 6587a76a6fdccd1c355536cedf57564e6f0c800b
./download.sh
./start.sh
```

`download.sh` fetches the pinned checkpoint, its "fast" layout pieces and the
drafter, assembles the checkpoint and checks every file by sha256. `start.sh`
uses Docker and, by default, invokes `host-tune.sh` through sudo to set the card
to 170 W; use `HOST_TUNE=0 ./start.sh` to leave the power limit alone. It sets the
context window from the card's memory: 262,144 tokens on a 64 GB card, 163,840 on
a 40 GB card. `GPU=1` picks another card. Do not start this alongside another
service using the same GPU.

The API is local-only at `http://127.0.0.1:8000`, model `qwen3.8-27b`. Wait for
readiness before sending a request:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Explain speculative decoding in two sentences."}],"max_tokens":128,"temperature":0}'
```

Thinking is off by default. Stop only the recipe's container with `./stop.sh`
from that checkout. For settings, reasoning and tool calls, use the
[pinned recipe README](https://github.com/AlexBrandes/cmp170hx-qwen38-27b/blob/6587a76a6fdccd1c355536cedf57564e6f0c800b/README.md).

## Release identity

| Component | Pin |
| --- | --- |
| Recipe | `6587a76a6fdccd1c355536cedf57564e6f0c800b` |
| Upstream vLLM | wheel for `a0d7356b98e929c17051bac44f5a086e4117974d` |
| vLLM branch | `qwen38-27b/vllm` at `eb0f19d117e35e32631e41be1cced3d98368d776` (over `vllm/a0d7356b98`) |
| Plugin | `qwen38-27b/plugin` at `b03ecee908aaf9abb26b0c97472faa871fd814d1` |
| Image | `ghcr.io/alexbrandes/cmp170hx-qwen38-27b@sha256:1c9a3796337e3f8ab2ad5d359465a8b99dbe5df32342a18ca5ea715748195920` |
| Checkpoint | `dbirks/Qwen3.8-27B-W4A16-AutoRound` at `1f05c441c4e64ae0549de44fa9ea5a6d43610314`, with the LM head and MTP module of `syvai/qwen3.8-27b-3090-fast-variant` at `124c14e7e8c7d2f5402933b9af368e772a9fcf0c` |
| Drafter | `syvai/Qwen3.8-27B-DFlash2-W4A16` at `4d30ec736ffc6b8688dc2ae2b502d9b48bdec279` |
| Layout | one card |
| Context / concurrency | 262,144 tokens on 64 GB, 163,840 on 40 GB / up to 8 sequences |
| Speculation / graphs | DFlash, 7 draft tokens / FULL_AND_PIECEWISE |

The [public results](https://github.com/AlexBrandes/cmp170hx-qwen38-27b/blob/6587a76a6fdccd1c355536cedf57564e6f0c800b/results/2026-11-11.json)
and [benchmark tables](https://github.com/AlexBrandes/cmp170hx-qwen38-27b/tree/6587a76a6fdccd1c355536cedf57564e6f0c800b#benchmarks)
cover three public-v1 runs on a 40 GB card, plus correctness, image and video
input, needle, the context window on both card sizes, HumanEval, GSM8K and GPQA
Diamond.

## Credits and licenses

The engine is Apache-2.0; the standalone recipe is MIT except two HyperQwen tools
(Apache-2.0), and the checkpoint and drafter have their own licenses. Its
[NOTICE](https://github.com/AlexBrandes/cmp170hx-qwen38-27b/blob/6587a76a6fdccd1c355536cedf57564e6f0c800b/NOTICE)
credits syv-ai's HyperQwen, Morrowmake's vllm-sm80, the Qwen team, the checkpoint
and drafter authors and the benchmark authors. See the
[fork credits](../../README.md#credits) for CMPUnlocker, Morrowmake, PixelML, and
the wider CMP community.
