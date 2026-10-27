# Qwen3.8-Flash-Next W4A16 on two CMP 170HX cards

This is our earlier production recipe: Intel's W4A16 AutoRound checkpoint,
pipeline parallelism across two cards, and the BF16 N-gram (PLE) table read from
NVMe. The runnable recipe lives in
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
git checkout --detach c0db9613d146eda17338b84f2440ead5505f53c5
./download.sh
./start.sh
```

`download.sh` fetches the pinned model and derives the W4A16 MTP draft head.
`start.sh` uses Docker and, by default, invokes `host-tune.sh` through sudo:
it sets the cards to 170 W, changes NVMe readahead, drops the host page cache,
and compacts memory. **Those are host-wide changes.** Use `HOST_TUNE=0 ./start.sh`
to leave the host settings alone; the published performance may not apply.
Do not start this alongside another service using the same GPUs.

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
[pinned recipe README](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next/blob/c0db9613d146eda17338b84f2440ead5505f53c5/README.md).

## Release identity

| Component | Pin |
| --- | --- |
| Recipe | `c0db9613d146eda17338b84f2440ead5505f53c5` |
| Engine branch | `AlexBrandes/vllm:cmp170hx-qwen38-flash-next` |
| Engine commit | `02acf67ca10bb0c6efddd5454facb3bfe5dba7d9` (vLLM v0.30.0 + six commits) |
| Image | `ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:d6fc872d943b3e9e5cc9eb583b51d72af200e039fbbb7401700bb38d665f0957` |
| Checkpoint | `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` at `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897` |
| Layout | PP2 / TP1, layer partition 26/22 |
| Context / concurrency | 262,144 tokens / up to 8 sequences |
| Speculation / graphs | MTP5 / FULL_AND_PIECEWISE with breakable PLE segments |

The [public results](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next/blob/c0db9613d146eda17338b84f2440ead5505f53c5/results/2026-10-17.json)
and [benchmark tables](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next/tree/c0db9613d146eda17338b84f2440ead5505f53c5#benchmarks)
cover three runs of this release on October 17, 2026. They are not measurements
of the newer EXL3 engine or of every subsequent production configuration change.

The engine changes include PP support, correct recurrent-state copies, a W4A16
draft head, graph-safe PLE lookup, bounded prefill scheduling, and fused QSA prepare.
They remain preserved on their original branch while the newer engine line is
consolidated. Do not substitute the moving `cmp170hx` branch for this release pin.

## Credits and licenses

The engine is Apache-2.0; the standalone recipe is MIT, and model weights have
their own license. Its [NOTICE](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next/blob/c0db9613d146eda17338b84f2440ead5505f53c5/NOTICE)
credits the original PP/PLE work by polyuij42-del and nguyenthimy2022kg-alt,
upstream vLLM's QSA fusion, Qwen, Intel AutoRound, and the benchmark authors.
See the [fork credits](../../README.md#credits) for CMPUnlocker, Morrowmake,
PixelML, and the wider CMP community.
