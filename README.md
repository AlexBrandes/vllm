# vLLM on the CMP 170HX

Our home for running large models on NVIDIA CMP 170HX cards: engine improvements,
reproducible recipes, and measurements from the hardware we actually use.

The focus is getting useful performance out of **PCIe Gen2 ×4 links**, not assuming
an NVLink or ×16 setup. Our Qwen recipes run on **two 64 GiB cards at 170 W each**,
using pipeline parallelism and keeping the large N-gram embedding (PLE) table on
NVMe. No GPU peer-to-peer transport is required by these recipes.

## Choose a recipe

| Model | Format | Layout | Start here |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next | W4A16 AutoRound | PP2 / TP1 | [W4A16 production recipe](recipes/qwen-flash-next-w4a16/README.md) — pinned container, download and launch scripts |
| Qwen3.8-Flash-Next | EXL3 5.05 bpw | PP2 / TP1 | [EXL3 recipe](recipes/qwen-flash-next-exl3/README.md) — source-built container and the validated serving configuration |

Both configure a **262,144-token context**, native MTP speculative decoding,
BF16 KV cache, and an OpenAI-compatible API. Follow the selected recipe's hardware,
storage, and installation requirements; the two formats use different engines
and PLE readers.

Our **GLM-5.3-Flash recipe is being finished**. It will be added here once its
configuration and validation are ready; it is not a released recipe in this fork yet.

## Measured performance

These are our measurements on **2× CMP 170HX, 170 W/card, PCIe Gen2 ×4/card**.
Decode uses the SparkDash `3e6d4ef` protocol: greedy sampling, thinking off,
400 output tokens. Values are medians of three complete runs. C8 means total
throughput across eight simultaneous requests, not the speed of each request.

| Recipe | MTP drafts | Code decode, C1 | Code decode, C8 total | Cold prefill, 32K |
| --- | ---: | ---: | ---: | ---: |
| W4A16 · October 17 | 5 | 226.2 tok/s | 840.4 tok/s | 7,104 tok/s |
| EXL3 · October 25 | 3 | 163.1 tok/s | 499.8 tok/s | 4,000 tok/s |

The [W4A16 results](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next/tree/c0db9613d146eda17338b84f2440ead5505f53c5#benchmarks)
and [EXL3 results](recipes/qwen-flash-next-exl3/results.json) identify their engine,
configuration, and measurement runs. These are **whole-recipe comparisons**, not
a controlled comparison of quantization alone: engine versions, MTP depth, and
runtime settings differ. Performance also depends on prompt content, concurrency,
page-cache residency, cooling, and storage contention.

## What we maintain

- **Qwen pipeline parallelism:** stage ownership, native MTP, vision on the first
  stage, and the state handling needed for requests to cross pipeline stages.
- **EXL3 performance:** PP2 tensor-core dense and MoE paths for prefill and decode,
  cheaper dense-prefill arithmetic, and full decode graphs with PLE rows staged
  outside graph replay.
- **The earlier W4A16 production engine:** graph-safe PLE reads, recurrent-state
  fixes, a quantized MTP draft head, bounded prefill beside running decodes, and
  fused QSA preparation. Its source and published image remain pinned and available.
- **Recipes with evidence:** exact revisions, launch settings, hardware details,
  and benchmark results rather than a moving branch presented as a tested release.

### Branches

| Branch | Purpose |
| --- | --- |
| **`cmp170hx`** | Default branch and home for our engine development and recipe index |
| `main` | Clean upstream vLLM mirror; no local engine patches |
| `work/<topic>` | Development branches targeting `cmp170hx` |
| `morrowmake/<topic>` | Selected changes ported for PRs against Morrowmake's `ampere` branch |
| `cmp170hx-qwen38-flash-next` | Preserved v0.30.0-based W4A16 production engine, `02acf67ca` |
| `qwen4exp-pp-w4a16-exl3` | Existing source branch for [Morrowmake PR #7](https://github.com/Morrowmake/vllm-cmp170hx/pull/7) |

`cmp170hx` starts from our published EXL3/PP engine at `344e5c2`, based on
Morrowmake's `ampere` at `f02c7a70`. The earlier W4A16 production line is **not yet
fully consolidated into this engine**. Its recipe deliberately keeps its own
validated pin. Updating this branch does not change either recipe's engine pin
or update the existing upstream PR.

New engine work belongs here first. Changes offered to Morrowmake are ported onto
his target revision separately, so our deployment work does not expand his PRs.
The `morrowmake/qwen4exp-pp-w4a16-exl3` branch preserves the current contribution;
its refresh onto his latest `ampere` is still separate integration work.

## Credits

This work exists because other people made these cards useful and shared what
worked. In particular:

- **[CMPUnlocker — the original project by amoghmunikote](https://github.com/amoghmunikote/cmpunlocker)**
  for the driver-unlock work that makes these cards practical for compute. Hardware
  and driver setup belong to that project; this fork does not install or modify drivers.
- **[Morrowmake](https://github.com/Morrowmake/vllm-cmp170hx)** for the CMP-focused
  vLLM engine, SM80 kernels, EXL3 integration, recipes, and optimization ideas.
  The newer engine here is derived from that work, not an independent rewrite.
- **[PixelML / club-170hx](https://github.com/PixelML/club-170hx)** for hardware and
  deployment guides, community measurements, and ideas for serving across these
  cards and their PCIe constraints.
- **[vLLM and its contributors](https://github.com/vllm-project/vllm)** for the
  serving engine, and **[turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3)**
  for EXL3 and its reference implementation.
- **[polyuij42-del](https://github.com/polyuij42-del/qwen38-flash-next-w4a16-cmp170hx)**
  and **[nguyenthimy2022kg-alt](https://github.com/nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX)**
  for the community PP and SSD-backed PLE work behind our earlier W4A16 engine.
- **Qwen, Intel AutoRound, and the checkpoint authors**, and
  **[MiaAI-Lab / SparkDash](https://github.com/MiaAI-Lab/sparkDash)** for the
  public benchmark protocol.

See [third-party sources and idea credits](docs/THIRD_PARTY.md),
[NOTICE](NOTICE), and the recipe-specific notices for attribution and licensing.
Credits do not imply endorsement.

## Development and licensing

The engine is **Apache-2.0**; inherited code, packaged dependencies, recipe
repositories, and model weights retain their own licenses. We do not distribute
model weights in this repository.

For engine details, see [EXL3](docs/features/quantization/exl3.md),
[pipeline parallelism](docs/serving/parallelism_scaling.md), and the
[inherited fork change map](FORK.md). The original vLLM presentation is preserved
in [the upstream README](docs/UPSTREAM_README.md).
