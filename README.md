# vLLM on the CMP 170HX

Our home for running large models on NVIDIA CMP 170HX cards: vLLM changes for
this hardware, reproducible recipes, and measurements from the cards we actually
use.

The focus is getting useful performance out of **64 GiB cards at 170 W joined by
PCIe Gen2 ×4 links**, not assuming an NVLink or ×16 setup. Our recipes split
models across cards and do not require GPU peer-to-peer transport.

## How this repository is organized

This fork tracks upstream vLLM. Everything that can be built as a vLLM plugin is
built as one; the few changes a plugin cannot make are kept as a short list of
small commits on top of upstream.

- **The `vllm-cmp170hx` plugin** (package `vllm_cmp170hx`) holds the fork's code:
  model and kernel changes, configuration and fixes. vLLM loads it at startup
  through its plugin entry point, and the engine refuses to start without it.
- **A vLLM branch** for each upstream version we run on, named
  `vllm/<upstream commit>`: upstream vLLM at that commit plus our commits, one per
  feature. Most are hooks of a few lines that call into the plugin; a few are
  upstream bug fixes we carry until upstream has them.

An engine is three pins: an upstream commit, a `vllm/` branch commit and a plugin
commit. Moving to a newer upstream means re-applying our commits on a new
`vllm/` branch and checking the plugin against it.
[`FORK.md`](FORK.md) lists every plugin file and every change to upstream vLLM;
the [development guide](docs/development.md) has the rules and the steps.

| Branch | Holds |
| --- | --- |
| **`plugin`** (default) | This page, the recipes, and the plugin |
| `vllm/<upstream commit>` | Upstream vLLM at that commit plus our commits; the current one is `vllm/165213f911` |
| `main` | A clean mirror of upstream vLLM |

Each recipe pins the exact engine it was validated with.

## Choose a recipe

| Model | Format | Layout | Start here |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next | W4A16 AutoRound | PP2 / TP1 | [W4A16 production recipe](recipes/qwen-flash-next-w4a16/README.md) — pinned container, download and launch scripts |
| Qwen3.8-Flash-Next | EXL3 5.05 bpw | PP2 / TP1 | [EXL3 recipe](recipes/qwen-flash-next-exl3/README.md) — source-built container and the validated serving configuration |

Each recipe lists its hardware, storage and software requirements, its engine
pins, and its launch settings.

## Measured performance

These are our measurements on **2× CMP 170HX, 170 W/card, PCIe Gen2 ×4/card**.
Decode uses the SparkDash `3e6d4ef` protocol: greedy sampling, thinking off,
400 output tokens. Values are medians of three complete runs. C8 means total
throughput across eight simultaneous requests, not the speed of each request.

| Recipe | Draft tokens | Code decode, C1 | Code decode, C8 total | Cold prefill, 32K |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-Flash-Next W4A16 · October 17 | 5 | 226.2 tok/s | 840.4 tok/s | 7,104 tok/s |
| Qwen3.8-Flash-Next EXL3 · October 25 | 3 | 163.1 tok/s | 499.8 tok/s | 4,000 tok/s |

The [W4A16 results](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next/tree/c0db9613d146eda17338b84f2440ead5505f53c5#benchmarks)
and [EXL3 results](recipes/qwen-flash-next-exl3/results.json) identify their engine,
configuration, and measurement runs. Each row measures a **whole recipe**: engine
version, quantization, speculative depth and runtime settings differ between
rows. Performance also depends on prompt content, concurrency, cooling, and
storage.

## What we maintain

- **Recipes:** complete configurations for the models we run, with exact
  revisions, launch settings, hardware details and benchmark results, rather
  than a moving branch presented as a tested release.
- **vLLM changes for the CMP 170HX:** splitting models across cards over slow
  PCIe links, kernels and quantization paths that suit these cards, model
  support, and fixes for problems found on this hardware.
- **Measurements:** reproducible benchmarks of each recipe on our cards,
  published with the engine, configuration and runs they came from.
- **Contributions back:** changes that are useful beyond this fork go upstream
  to vLLM or to the related CMP 170HX projects below.

## Credits

This work exists because other people made these cards useful and shared what
worked. In particular:

- **[CMPUnlocker — the original project by amoghmunikote](https://github.com/amoghmunikote/cmpunlocker)**
  for the driver-unlock work that makes these cards practical for compute. Hardware
  and driver setup belong to that project; this fork does not install or modify drivers.
- **[Morrowmake](https://github.com/Morrowmake/vllm-cmp170hx)** for the CMP-focused
  vLLM engine, SM80 kernels, EXL3 integration, recipes, and optimization ideas. Our
  engine is derived from that work and follows its plugin layout. Our work overlaps
  heavily with theirs, so we aim to contribute to that project wherever it is useful.
- **[PixelML / club-170hx](https://github.com/PixelML/club-170hx)** for hardware and
  deployment guides, community measurements, and ideas for serving across these
  cards and their PCIe constraints.
- **[vLLM and its contributors](https://github.com/vllm-project/vllm)** for the
  serving engine, and **[turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3)**
  for EXL3 and its reference implementation.
- **The model and checkpoint authors**, and
  **[MiaAI-Lab / SparkDash](https://github.com/MiaAI-Lab/sparkDash)** for the
  public benchmark protocol.

See [NOTICE](NOTICE) and each recipe's notices for attribution and licensing.
Credits do not imply endorsement.

## Licensing

The plugin and our vLLM changes are **Apache-2.0**, as upstream vLLM; inherited
code, packaged dependencies, recipe repositories, and model weights retain their
own licenses. We do not distribute model weights in this repository. The original
vLLM presentation is in
[the upstream README](https://github.com/vllm-project/vllm/blob/main/README.md).
