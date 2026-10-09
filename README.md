<!-- Modified by Morrowmake: CMP 170HX presentation and source attribution. -->

<h1 align="center">vLLM for the NVIDIA CMP 170HX</h1>

<p align="center">
  <img alt="target" src="https://img.shields.io/badge/target-Ampere%20sm__80-76b900?style=flat">
  &nbsp;
  <img alt="base" src="https://img.shields.io/badge/upstream-vLLM%200.30.1%20dev-4b32c3?style=flat">
  &nbsp;
  <img alt="licence" src="https://img.shields.io/badge/licence-Apache--2.0-blue?style=flat">
</p>

This is a fork of [vLLM](https://github.com/vllm-project/vllm) for the
**NVIDIA CMP 170HX** (Ampere, sm_80), built to serve models fast on one card
or several: tensor-parallel and pipeline-parallel layouts across cards, a
full-precision KV cache and repeatable single-request outputs.
**GLM-5.3-Flash** is the first supported model; more will follow.

The fork has two layers:

- **General work** for sm_80 and PCIe-connected cards: kernels and attention
  backends, all-reduce through host memory or PCIe, KV accounting,
  determinism, pipeline-parallel scheduling, and an optional compiled Marlin
  extension.
- **Model-specific work**, gated by model family: it lives in that family's
  model code or behind its own switches, and its kernels check the exact
  shapes they were built for, handing anything else back to upstream's path.

Every performance feature sits behind an environment variable and is **off in
the code by default**. Each supported model has a recipe repository whose
launcher switches on the tested set. `0` turns a feature off again, with no rebuild.
Variables carry the prefix of the model they were first built for
(`VLLM_GLM5_*`), including some general ones.

## Using it: take a recipe

Each model is served through its own recipe, which pins an exact fork commit,
ships the engine as a container image, downloads the weights and starts an
OpenAI-compatible server. Questions about running a model belong in its
recipe's issues.

| Model | Recipe |
|---|---|
| GLM-5.3-Flash | [Morrowmake/glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe) |

## Supported models

### GLM-5.3-Flash

320B MoE, W4A16, 262,144-token context, DFlash2 speculative decoding.
Upstream's sparse-attention path needs Hopper; this fork adds the sm_80 path
that makes the model run at all, then makes it fast.

```bash
git clone https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe.git
cd glm53-flash-cmp170hx-recipe && ./start.sh     # LAYOUT=pp4 for pipeline-parallel 4
```

From the recipe's [release 1.5.0 results](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe#results)
(DFlash2 at k=3, 180 W per card; methods and caveats are there):

| | TP4 (default) | PP4 (`LAYOUT=pp4`) |
|---|---:|---:|
| Streaming decode, 1 user, structured | **267.3 tok/s** | 141.8 tok/s |
| Decode, 8 users, aggregate, structured | **758.9 tok/s** | 603.0 tok/s |
| Cold prefill | 2,657 tok/s | **6,575 tok/s** |
| KV pool at 262,144 context | 1,176,646 tokens | **2,334,498 tokens** |

Quality with DFlash2 (release 1.4.3, fixed batches): HumanEval 162/164 under
TP4 and 163/164 under PP4; GSM8K 1,281/1,319 and 1,284/1,319.

GLM-specific switches (the recipe's
[kill switch table](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe#kill-switches)
says which layout each serves):

- Fused decode kernels for mHC mixing, MoE routing and alignment, and the KDA
  decode step (gate projections, convolution, delta rule and gated norm in one
  kernel, one KDA state per request under TP4) — `VLLM_AMPERE_DECODE_KERNELS`;
  its TP4 verify (the commit fold's included) with at most 16 rows per warp,
  less stack and more KV cache, bitwise the same
  (not yet validated, off unless set) — `VLLM_AMPERE_KDA_SMALL_STACK`;
  the KDA step for PP4 decode batches beyond one short request (where it
  measures faster) — `VLLM_GLM5_DECODE_KDA_WIDE`; second
  generation `VLLM_MOE_DECODE_ROUTE_V2`, `VLLM_GLM5_DECODE_MHC_V2`,
  `VLLM_AMPERE_DECODE_IDX_GLUE`; under TP4, the KDA state commit after
  sampling folded into the next decode step, same state bit for bit
  (not yet validated, off unless set) — `VLLM_AMPERE_KDA_FOLD`; sparse
  attention decode at up to 64 query rows, 16 or 64 heads, that walks only
  each index row's valid span (not yet validated, off unless set) —
  `VLLM_AMPERE_SPARSE_MLA_DECODE_V2`.
  A diagnostic, never for serving: under TP4, one KDA state per draft
  position instead of one per request (the reference without state
  recovery, as wherever that path's limits are not met) —
  `VLLM_AMPERE_KDA_RECOVER_OFF`.
- Prefill kernels (mHC projection, sparse attention) — `VLLM_GLM5_PREFILL_KERNELS`,
  `VLLM_GLM5_SMLA_PREFILL_PRED_LOAD`; at TP4 shapes `VLLM_AMPERE_KDA_PREFILL_NARROW`,
  `VLLM_AMPERE_MARLIN_PREFILL_NARROW`; at PP4 shapes `VLLM_AMPERE_KDA_PREFILL_WIDE`,
  `VLLM_AMPERE_SPARSE_MLA_PREFILL_WIDE`, `VLLM_AMPERE_MARLIN_PREFILL_WIDE`.
- Batch-sharded logits and sampling (TP4) — `--enable-batch-sharded-sampling`.
- Drafter: selector tables split across cards — `VLLM_GLM5_DRAFTER_SELECTOR_SHARD`;
  position table sized to the context — `VLLM_GLM5_DRAFTER_ROPE_FIT`; folded
  input projection under PP4 — `VLLM_GLM5_PP_FOLD_DRAFT_FC`.
- Always on: a rejected draft can no longer overwrite the tail of the
  sparse-attention key pool.

## General features

**sm_80 kernels and backends.** A sparse-MLA attention backend, an sm_80 path
for the attention indexer and its FP8 stores, and sm_80 key-pool compression
(always on). A retuned sparse-attention decode schedule. Thin-batch BF16
GEMMs tuned per shape, up to 32 rows — `VLLM_THIN_GEMM`. MoE shared
experts that overlap the routed experts — `VLLM_SHARED_EXPERT_REORDER`.

**All-reduce over PCIe.** For cards without GPU peer access, small all-reduces
take one round trip through shared host memory — `VLLM_HOST_ALLREDUCE`.
Where peer access is granted, a PCIe custom all-reduce runs in device memory
instead — `VLLM_CUSTOM_ALLREDUCE_PCIE_P2P`, kernel chosen by
`VLLM_CUSTOM_ALLREDUCE_ALGO`; it falls back to the host path on its own.

**KV accounting.** The KV reserve covers what prefill chunks in flight really
hold, so the reported pool is one the server can fill —
`VLLM_KV_MAMBA_INFLIGHT_STATES`.
`VLLM_MEM_ATTRIBUTION` logs where device memory goes at start-up.

**Determinism.** A request on its own returns the same tokens and
log-probabilities on every run and every install: fixed-order MoE block
alignment (`VLLM_MOE_DETERMINISTIC_ALIGN`), CUDA-graph padding kept out of
the MoE (`VLLM_MOE_MASK_PADDING`), indexer top-k consistent on ties and in
a fixed order (`VLLM_GLM5_TOPK_TIEFIX`, `VLLM_GLM5_TOPK_SORTED`,
`VLLM_GLM5_TOPK_TIEFIX_SPLIT_ROWS`), pinned linear-attention prefill
configurations (`VLLM_FLA_PIN_AUTOTUNE`).

**Pipeline-parallel and scheduling.** Decodes spread over every in-flight
micro-batch — `VLLM_PP_SPREAD_DECODES`; one packed, metadata-free hand-off
between stages — `VLLM_PP_PACKED_HOP`, `VLLM_PP_HOP_NO_METADATA`; drafter
synchronisation — `VLLM_PP_SPLIT_DRAFT_EVENT`; the drafter's final step on a
chosen stage — `VLLM_PP_DRAFT_TAIL_STAGE` (`-1` off). Fair prefill, where long
prompts yield to running decodes — `--prefill-chunk-with-decodes N`. A
load-adaptive draft count, off unless set — `adaptive_k` in
`--speculative-config` ([docs](docs/features/speculative_decoding/adaptive_k.md)).

**Optional compiled Marlin.** One sm_80 library, `vllm._ampere_marlin_C`, for
both layouts: `VLLM_AMPERE_MARLIN_DECODE_CUDA` (eligible decode batches). Requesting
it without a compatible library fails at startup rather than falling back
silently. The decode kernels split the first projection along K: faster, and
the fp32 summation order differs from the released Marlin kernels. Under PP4
the split-block prefill (`VLLM_AMPERE_MARLIN_PREFILL_WIDE`) also runs on the
library's (64,512) tiles with a one-HFMA2 dequant —
`VLLM_AMPERE_MARLIN_PREFILL_WIDE_COMPILED`, on by default, the released kernels
without the library; TP4 prefill uses the released kernels.

**Correctness fixes (always on).** 64-bit KV row offsets in the
sparse-attention kernels and a vocabulary clamp in the sampler kernels.

Every custom kernel on a default path is checked on real captured inputs
against a 64-bit reference, side by side with the code it replaces, and must
be at least as accurate.

**Removed opt-in paths.** Draft-confidence skipping and its fitting logs, depth-two
verification, the full canonical top-k serving override and the separate tie-repair
fallback, cross-layer prefill overlap, replicated input embeddings, and the legacy
sparse-attention decode schedule are no longer supported. Their released defaults
were disabled; the shipped adaptive acceptance tracking and depth-three-through-seven
policy, fused top-k tie fix and sorting, sharded embeddings, and current
sparse-attention schedule remain unchanged. The within-layer prefill overlap is
removed; TP4 prefill all-reduces run unsplit, as upstream. The canonical top-k reference
is retained only in tests.

## Upstream base

The fork sits on upstream `main` at
[`e55d076f89`](https://github.com/vllm-project/vllm/commit/e55d076f89fd01a0538a3e496d8ff20bf7980100)
(2026-09-25), the vLLM **0.30.1** development line (`git describe`:
`v0.30.1rc0-181-ge55d076f8`). `main` on this fork stays a clean mirror of
upstream; general fixes are offered back upstream as pull requests.

## Branches and releases

The work is one branch, today `ampere-glm53`. When the second model lands it
will be renamed to a model-neutral name; GitHub redirects the old name.

A recipe pins an exact fork commit, never the branch. The branch moves to a
newer upstream from time to time; every commit a recipe release pins is kept
under a tag `<recipe>-<version>` (for GLM-5.3-Flash, `glm53-recipe-<version>`),
so older releases keep installing. Release notes live in each recipe's
changelog, e.g. [GLM-5.3-Flash](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/blob/main/CHANGELOG.md).

## Building from source

A recipe's native install (`RUNTIME=native ./install.sh`) is the reference.
In short, on Python 3.12 with torch 2.13.0 (CUDA 13.0):

```bash
git clone -b ampere-glm53 https://github.com/Morrowmake/vllm-cmp170hx.git && cd vllm-cmp170hx
VLLM_USE_PRECOMPILED=1 uv pip install -e .    # upstream's precompiled extensions carry sm_80
uv pip install -r requirements/cuda.txt
# optional compiled Marlin, built on its own without rebuilding the engine:
VLLM_AMPERE_MARLIN_BUILD=1 python csrc/libtorch_stable/moe/ampere_marlin/build_standalone.py --out vllm
```

The optional library needs a CUDA toolkit (12.8 or newer), a C++20 compiler and
Ninja; no GPU is needed to build it. `VLLM_AMPERE_MARLIN_BUILD=1` cannot be
combined with `VLLM_USE_PRECOMPILED=1` in one install. Each recipe names the
upstream wheel that supplies the precompiled extensions for its pin.

## Licence and credit

Apache-2.0, as upstream ([LICENSE](LICENSE)). This fork is a layer on the work
of the [vLLM project](https://github.com/vllm-project/vllm) and its
contributors; thank you. Upstream's own README is kept at
[docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

Some implementations are adapted from other projects; these are not all
independently authored patches. [Third-party sources and idea credits](docs/THIRD_PARTY.md)
identify those adaptations, their licences and the ideas that informed our
changes. Original component copyright and licence notices remain applicable.
