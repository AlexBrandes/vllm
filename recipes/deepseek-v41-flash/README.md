# DeepSeek-V4.1-Flash EXL3 on three CMP 170HX cards

DeepSeek-V4.1-Flash-1017 as a 2.2 bpw EXL3 checkpoint whose routed experts use the A4NX
codebook, with INT8 dense weights, pipeline parallelism across three cards, the model's own MTP
layers as the DSpark drafter, and text and image input. The runnable recipe lives in
[AlexBrandes/cmp170hx-deepseek-v41-flash](https://github.com/AlexBrandes/cmp170hx-deepseek-v41-flash).
This page pins that release rather than duplicating its scripts.

## Requirements

- Linux x86-64, three CMP 170HX cards: two 64 GB and one 40 GB (listed second), suitable cooling.
  Three 64 GB cards or two 80 GB cards should also work (not tested); the recipe has the settings
  and KV cache estimates.
- NVIDIA driver 580 or newer and Docker with the NVIDIA Container Toolkit.
- About 370 GB of free NVMe storage: the checkpoint with its INT8 dense weights (167 GB) and
  DeepSeek's Engram tables (203 GB), which the engine reads from disk while serving.
- The recipe was measured at 170 W/card on **PCIe Gen2 ×4 links**, without GPU P2P.

## Run

```bash
git clone https://github.com/AlexBrandes/cmp170hx-deepseek-v41-flash.git
cd cmp170hx-deepseek-v41-flash
git checkout --detach 828688311b9c448c92f854fcc6d9dc9c7490d95c
./download.sh
GPUS=0,2,1 ./start.sh     # the 40 GB card second
```

`download.sh` fetches the pinned checkpoint and the Engram tables from DeepSeek's repo.
`start.sh` uses Docker and leaves the cards' power limits alone. Do not start this alongside
another service using the same GPUs.

The API is local-only at `http://127.0.0.1:8000`, model `deepseek-v4.1-flash-a4nx-2.2bpw`. Wait
for readiness before sending a request:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash-a4nx-2.2bpw","messages":[{"role":"user","content":"Explain pipeline parallelism in two sentences."}],"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

Stop only the recipe's container with `./stop.sh` from that checkout.
For settings, reasoning, tool calls, and other card sets, use the
[pinned recipe README](https://github.com/AlexBrandes/cmp170hx-deepseek-v41-flash/blob/828688311b9c448c92f854fcc6d9dc9c7490d95c/README.md).

## Release identity

| Component | Pin |
| --- | --- |
| Recipe | `828688311b9c448c92f854fcc6d9dc9c7490d95c` |
| Upstream vLLM | wheel for `a0d7356b98e929c17051bac44f5a086e4117974d` |
| vLLM branch | `dsv41-flash/vllm` at `a18d750b19fd5120554ecf85a9bb1be0058df695` |
| Plugin | `dsv41-flash/plugin` at `0342f80c51fb2837456d231860028880122ee5b6` |
| Image | `ghcr.io/alexbrandes/cmp170hx-dsv41-flash-a4nx@sha256:990be3cf6f219eccdabb3e826253f46a1e06967e327b65e2b00e260813b212eb` |
| Checkpoint | `codebrandes/DeepSeek-V4.1-Flash-1017-A4NX-2.2bpw` at `2aaf191ccb6078319ed41c1bc85ed1b2e6f2c093` (includes the INT8 dense weights) |
| Engram tables | `deepseek-ai/DeepSeek-V4.1-Flash-1017` at `1bae0e1088993b9b82de08f59de5be5a34e42170`, shards 47–48 |
| Layout | PP3 / TP1, layer partition 15/10/15 |
| Context / concurrency | 262,144 tokens / up to 4 sequences |
| Speculation / graphs | DSpark (MTP layers), 5 draft tokens / FULL_DECODE_ONLY |

The [public results](https://github.com/AlexBrandes/cmp170hx-deepseek-v41-flash/blob/828688311b9c448c92f854fcc6d9dc9c7490d95c/results/a4nx-2.2bpw.json)
and [benchmark tables](https://github.com/AlexBrandes/cmp170hx-deepseek-v41-flash/tree/828688311b9c448c92f854fcc6d9dc9c7490d95c#benchmarks)
cover three public-v1 runs, image input, and GSM8K, HumanEval+, IFEval, MMLU, RULER and KL
divergence against the original FP8 release.

## Credits and licenses

The engine is Apache-2.0; the standalone recipe is MIT, and the checkpoint has its own license.
Its [NOTICE](https://github.com/AlexBrandes/cmp170hx-deepseek-v41-flash/blob/828688311b9c448c92f854fcc6d9dc9c7490d95c/NOTICE)
credits turboderp's ExLlamaV3, Morrowmake's vllm-sm80, DeepSeek and the benchmark authors. See
the [fork credits](../../README.md#credits) for CMPUnlocker, Morrowmake, PixelML, and the wider
CMP community.
