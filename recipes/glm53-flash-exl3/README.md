# GLM-5.3-Flash EXL3 on two CMP 170HX cards

GLM-5.3-Flash as a 2.6 bpw EXL3 checkpoint whose routed experts use the Q4N-X
codebook, with INT8 dense weights, pipeline parallelism across two cards, the
DFlash2 drafter at three draft tokens, and text, image and video input. The
runnable recipe lives in
[AlexBrandes/cmp170hx-glm53-flash-exl3](https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3).
This page pins that release rather than duplicating its scripts.

## Requirements

- Linux x86-64, two CMP 170HX cards exposing 64 GB or more each, suitable cooling.
- NVIDIA driver 580 or newer and Docker with the NVIDIA Container Toolkit.
- About 120 GB of free NVMe storage while `download.sh` runs (118 GB afterwards: checkpoint
  with its INT8 dense weights, and the 4-bit drafter), plus the image/cache.
- The recipe was measured at 170 W/card on **PCIe Gen2 ×4 links**, without GPU P2P.

## Run

```bash
git clone https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3.git
cd cmp170hx-glm53-flash-exl3
git checkout --detach ecdcaa550e61195f00e04ddacb29cfe82d7759a3
./download.sh
./start.sh
```

`download.sh` fetches the pinned checkpoint and drafter and converts the drafter to
4 bits locally. The drafter is licensed CC BY-NC-ND 4.0: non-commercial use only.
`start.sh` uses Docker and leaves the cards' power limits alone. Do not start this
alongside another service using the same GPUs.

The API is local-only at `http://127.0.0.1:8000`, model
`glm-5.3-flash-exl3-2.60bpw`. Wait for readiness before sending a request:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3-flash-exl3-2.60bpw","messages":[{"role":"user","content":"Explain pipeline parallelism in two sentences."}],"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

Stop only the recipe's container with `./stop.sh` from that checkout.
For settings, reasoning, tool calls, and the chat template, use the
[pinned recipe README](https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3/blob/ecdcaa550e61195f00e04ddacb29cfe82d7759a3/README.md).

## Release identity

| Component | Pin |
| --- | --- |
| Recipe | `ecdcaa550e61195f00e04ddacb29cfe82d7759a3` |
| Upstream vLLM | wheel for `a0d7356b98e929c17051bac44f5a086e4117974d` |
| vLLM branch | `vllm/a0d7356b98` at `ea2ef399a614ea72b65b15804000342f629a850f` |
| Plugin | `plugin` at `ff413b2df19718342e02d9763f161d69afce7b4a` |
| Image | `ghcr.io/alexbrandes/cmp170hx-glm53-flash-exl3@sha256:125662d873c809229ec85ebf18ce76f1f7b9a8a86470edd447880c116af32f8f` |
| Checkpoint | `codebrandes/GLM-5.3-Flash-Q4NX-2.6bpw` at `799c91c775f3cbd98de292329c0efee2ddc5be10` (includes the INT8 dense weights) |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2` at `bf582e4eacc1810f76656d1811693ff6c6737d2a`, converted to 4 bits by `download.sh` |
| Layout | PP2 / TP1, layer partition 23/22 |
| Context / concurrency | 225,280 tokens / up to 4 sequences |
| Speculation / graphs | DFlash, 3 draft tokens / FULL_DECODE_ONLY |

The [public results](https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3/blob/ecdcaa550e61195f00e04ddacb29cfe82d7759a3/results/q4nx-2.6bpw.json)
and [benchmark tables](https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3/tree/ecdcaa550e61195f00e04ddacb29cfe82d7759a3#benchmarks)
cover one public-v1 run, KL divergence and perplexity against the BF16 and FP8
releases, GSM8K, IFEval, and image and video input, each beside the previous
release (turboderp's EXL3 checkpoint).

## Credits and licenses

The engine is Apache-2.0; the standalone recipe is MIT, and the checkpoint and
drafter have their own licenses. Its [NOTICE](https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3/blob/ecdcaa550e61195f00e04ddacb29cfe82d7759a3/NOTICE)
credits turboderp's ExLlamaV3, Morrowmake's vllm-sm80, Z.AI, incoai and the
benchmark authors. See the [fork credits](../../README.md#credits) for CMPUnlocker,
Morrowmake, PixelML, and the wider CMP community.
