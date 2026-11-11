# GLM-5.3-Flash EXL3 on two CMP 170HX cards

turboderp's EXL3 2.60 bpw quantization of GLM-5.3-Flash, pipeline parallelism
across two cards, the DFlash2 drafter at three draft tokens, text, image and
video input, the plugin's tensor-core EXL3 kernels for the routed experts, and INT8
dense matrices built from the original weights. The
runnable recipe lives in
[AlexBrandes/cmp170hx-glm53-flash-exl3](https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3).
This page pins that release rather than duplicating its scripts.

## Requirements

- Linux x86-64, two CMP 170HX cards exposing 64 GiB each, suitable cooling.
- NVIDIA driver 580 or newer and Docker with the NVIDIA Container Toolkit.
- About 126 GB of free NVMe storage while `download.sh` runs (113 GB afterwards: checkpoint,
  drafter and INT8 dense weights), plus the image/cache.
- The recipe was measured at 170 W/card on **PCIe Gen2 ×4 links**, without GPU P2P.

## Run

```bash
git clone https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3.git
cd cmp170hx-glm53-flash-exl3
git checkout --detach 0294590212713617fdfa5d5be46385f30c657f6a
./download.sh
./start.sh
```

`download.sh` fetches the pinned checkpoint and drafter. The drafter is licensed
CC BY-NC-ND 4.0: non-commercial use only. `start.sh` uses Docker and, by default,
invokes `host-tune.sh` through sudo to set both cards to 170 W; use
`HOST_TUNE=0 ./start.sh` to leave power limits alone. Do not start this alongside
another service using the same GPUs.

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
[pinned recipe README](https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3/blob/0294590212713617fdfa5d5be46385f30c657f6a/README.md).

## Release identity

| Component | Pin |
| --- | --- |
| Recipe | `0294590212713617fdfa5d5be46385f30c657f6a` |
| Upstream vLLM | wheel for `a0d7356b98e929c17051bac44f5a086e4117974d` |
| vLLM branch | `vllm/a0d7356b98` at `ea2ef399a614ea72b65b15804000342f629a850f` |
| Plugin | `plugin` at `0c79bc78fdc30708f1995d5759828d2a08e39f04` |
| Image | `ghcr.io/alexbrandes/cmp170hx-glm53-flash-exl3@sha256:daa037541ed91b304c5f4168b6dad4820362ef25c1e6ba3193bf5aef562ed076` |
| Checkpoint | `turboderp/GLM-5.3-Flash-exl3` `sc_2.60bpw` at `24781209f2862a939cc21e01d256c943bd1a58a3` |
| Dense weights | INT8 copy built by `download.sh` from `zai-org/GLM-5.3-Flash` at `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2` at `bf582e4eacc1810f76656d1811693ff6c6737d2a` |
| Layout | PP2 / TP1, layer partition 23/22 |
| Context / concurrency | 225,280 tokens / up to 4 sequences |
| Speculation / graphs | DFlash, 3 draft tokens / FULL_DECODE_ONLY |

The [public results](https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3/blob/0294590212713617fdfa5d5be46385f30c657f6a/results/2026-11-11.json)
and [benchmark tables](https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3/tree/0294590212713617fdfa5d5be46385f30c657f6a#benchmarks)
cover three public-v1 runs, plus correctness, image and video
input, needle, context window, HumanEval, GSM8K and GPQA Diamond.

## Credits and licenses

The engine is Apache-2.0; the standalone recipe is MIT, and the checkpoint and
drafter have their own licenses. Its [NOTICE](https://github.com/AlexBrandes/cmp170hx-glm53-flash-exl3/blob/0294590212713617fdfa5d5be46385f30c657f6a/NOTICE)
credits turboderp's ExLlamaV3, Morrowmake's vllm-sm80, Z.AI, incoai and the
benchmark authors. See the [fork credits](../../README.md#credits) for CMPUnlocker,
Morrowmake, PixelML, and the wider CMP community.
