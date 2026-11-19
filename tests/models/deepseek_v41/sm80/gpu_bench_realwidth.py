# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-width, few-layer DeepSeek-V4.1 engine run on one GPU: the served
kernels at the served shapes, without the converted checkpoint.

Builds a header-only (sparse, all-zero data) EXL3 checkpoint with the real
config cut to ``--layers`` decoder layers (hidden 5120, 384 routed experts as
Q4N-X at 2.0 / 2.5 bpw, the real attention, indexer, compressor, mHC, Engram
on layer 1, the three DSpark stages with 128 Q4N-X experts at 2.5 or 2.0 bpw) and a
header-only dense W8 sidecar copied from the real one's header, then serves
it through vLLM with the lab unit's kernel switches and times prefill and
decode. All weights are zero, so the tokens are meaningless; the point is
that the Q4N-X V4.1 families, the W8 sidecar paths, Engram, mHC and DSpark
load and run at the served geometry, and what a layer costs (Interpretation
only; serving numbers come from benchlib).

    python tests/models/deepseek_v41/sm80/gpu_bench_realwidth.py --out DIR \
        --tokenizer-dir SRC --w8-header W8HDR [--layers 4] [--dspark]
"""

import argparse
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

LAB_ENV = {
    "VLLM_EXL3_MOE_PREFILL_TC": "1",
    "VLLM_EXL3_MOE_DECODE_TC": "1",
    "VLLM_EXL3_MOE_DECODE_TC_MODE": "full",
    "VLLM_EXL3_Q4N_PREFILL": "i8",
    "VLLM_EXL3_Q4N_I8_GEMM": "t2c4",
    "VLLM_EXL3_Q4N_DECODE": "i8",
    "VLLM_EXL3_Q4N_DECODE_GEMM": "t2c4",
    "VLLM_EXL3_MOE_DECODE_TC_MAX_ROWS": "64",
    "VLLM_EXL3_MOE_COMBINE_WIDE": "1",
    "VLLM_EXL3_ROUTE_CHECK": "0",
    "VLLM_EXL3_DENSE_PREFILL_TC": "1",
    "VLLM_EXL3_DENSE_DECODE_TC": "1",
    "VLLM_EXL3_DENSE_DECODE_TC_FAMILIES": "all",
    "VLLM_EXL3_DENSE_DECODE_SPLITK": "1",
    "VLLM_EXL3_DECODE_VARIANT": "off",
    "VLLM_EXL3_DENSE_W8_DECODE": "1",
    "VLLM_DEEPSEEK_V41_DSPARK_EMBED_INT8": "1",
    "VLLM_DEEPSEEK_V41_TRIM_ROPE": "1",
    "VLLM_DEEPSEEK_V41_FUSED_MHC": "1",
    "VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH": "1",
    "VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def build(
    out: Path, tokenizer_dir: Path, layers: int, routed_25: set[int], realistic: bool = False,
    stage: str = "default", draft_bits_x2: int = 5,
) -> None:
    from tests.models.deepseek_v41.sm80 import exl3_fixture as fixture
    from tests.models.deepseek_v41.sm80.test_engram_storage import _write_table

    config = json.loads((fixture.FIXTURES / "config.json").read_text())
    text = config["text_config"]
    ratios = text["compress_ratios"]
    text["compress_ratios"] = ratios[:layers] + ratios[-text["num_nextn_predict_layers"]:]
    text["num_hidden_layers"] = layers
    # The last layer becomes a ratio-1 KV / index / candidate source (as layer
    # 20 is in the full model), so both compressor ratios run and the cut
    # model stays valid (upstream's decoder replay needs ratio <= 1 at the
    # last KV source).
    text["compress_ratios"][layers - 1] = 1
    text["kv_source_layer_ids"] = sorted(
        {s for s in text["kv_source_layer_ids"] if s < layers} | {layers - 1}
    )
    text["index_source_layer_ids"] = sorted(
        {s for s in text["index_source_layer_ids"] if s < layers} | {layers - 1}
    )
    text["candidate_source_layer_id"] = layers - 1
    keep = [i for i, l in enumerate(text["engram_layer_ids"]) if l < layers]
    text["engram_layer_ids"] = [text["engram_layer_ids"][i] for i in keep]
    text["engram_num_embeddings"] = [text["engram_num_embeddings"][i] for i in keep]
    text["dspark_target_layer_ids"] = list(range(layers - 3, layers))
    if stage == "last":
        # A last-pipeline-stage-like cut: every layer ratio 1 reading one
        # ratio-1 KV / candidate source (as layers 25-39 read layer 20, here
        # the first layer stands in for the shadow), a non-KV index source
        # every 4 layers (24, 28, ...), no Engram.
        text["compress_ratios"] = [1] * layers + [0] * text["num_nextn_predict_layers"]
        text["kv_source_layer_ids"] = [0]
        text["index_source_layer_ids"] = list(range(0, layers, 4))
        text["candidate_source_layer_id"] = 0
        text["engram_layer_ids"] = []
        text["engram_num_embeddings"] = []
    tables = out / "engram-tables"
    config["engram_table_dir"] = str(tables)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(config))
    tensors = fixture.checkpoint_tensors(config)
    # Our converter: Engram wkv BF16; routed experts Q4N-X (marker q4nx).
    for name in [n for n in tensors if ".engram.wkv." in n]:
        del tensors[name]
    hc = text["hc_mult"]
    n_hash = (text["engram_max_ngram_size"] - 1) * text["engram_n_heads"]
    for layer in text["engram_layer_ids"]:
        tensors[f"layers.{layer}.engram.wkv.weight"] = (
            "BF16", (text["hidden_size"] * (hc + 1), n_hash * text["engram_head_dim"])
        )
    for name in [n for n in tensors if ".ffn.experts." in n and n.endswith(".mul1")]:
        del tensors[name]
        tensors[name[: -len(".mul1")] + ".q4nx"] = ("I32", ())
    for name, (dtype, shape) in list(tensors.items()):
        if ".ffn.experts." in name and name.endswith(".trellis"):
            draft = name.startswith("mtp.")
            layer = int(name.split(".")[1])
            bits_x2 = draft_bits_x2 if draft else (5 if layer in routed_25 else 4)
            tensors[name] = (dtype, (*shape[:-1], 8 * bits_x2))
    header, offset = {}, 0
    for name, (dtype, shape) in sorted(tensors.items()):
        size = fixture.tensor_bytes(dtype, tuple(shape))
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    markers = {"mul1": fixture.MUL1, "q4nx": fixture.MUL1}
    with open(out / "model-00001-of-00001.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        f.truncate(8 + len(raw) + offset)
        for name, entry in header.items():
            marker = name.rsplit(".", 1)[-1]
            if marker in markers:
                os.pwrite(f.fileno(), struct.pack("<I", markers[marker]),
                          8 + len(raw) + entry["data_offsets"][0])
        if realistic:
            _realistic_values(f.fileno(), 8 + len(raw), header)
    tables.mkdir()
    for layer, rows in zip(text["engram_layer_ids"], text["engram_num_embeddings"]):
        _write_table(tables, layer, rows, [0], f"engram-{layer}.safetensors")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if (tokenizer_dir / name).exists():
            shutil.copyfile(tokenizer_dir / name, out / name)


def _realistic_values(fd: int, base: int, header: dict) -> None:
    """Values that make the routing realistic in an otherwise all-zero
    checkpoint: a random embedding, unit norm weights and random router
    gates (logit spread ~1.4), so tokens spread over the experts as served
    tokens do (an all-zero model routes every token to the same experts)."""
    import torch

    generator = torch.Generator().manual_seed(0)
    for name, entry in header.items():
        lo, hi = entry["data_offsets"]
        shape = entry["shape"]
        if name == "embed.weight":
            rows_per_chunk = 4096
            for start in range(0, shape[0], rows_per_chunk):
                rows = min(rows_per_chunk, shape[0] - start)
                chunk = torch.randn(rows, shape[1], generator=generator).to(torch.bfloat16)
                os.pwrite(fd, chunk.view(torch.int16).numpy().tobytes(),
                          base + lo + start * shape[1] * 2)
        elif "norm" in name and name.endswith(".weight") and entry["dtype"] == "BF16":
            ones = torch.ones(shape, dtype=torch.bfloat16)
            os.pwrite(fd, ones.view(torch.int16).numpy().tobytes(), base + lo)
        elif name.endswith(".ffn.gate.weight"):
            # BF16-valued (as the official gates are), stored F16 as our converter does.
            gate = (torch.randn(shape, generator=generator) * 0.02).to(torch.bfloat16).to(torch.float16)
            os.pwrite(fd, gate.view(torch.int16).numpy().tobytes(), base + lo)


def sparse_sidecar(header_file: Path, out: Path, size: int) -> Path:
    path = out / "w8" / "dense.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header_file.read_bytes())
        f.truncate(size)
    return path


def profile(
    out: Path, w8: Path | None, max_len: int, profile_dir: Path, mode: str = "decode"
) -> dict:
    """Eager decode under the torch profiler: CUDA time per kernel name over
    64 decode steps at batch 1 (kernel durations; launch gaps excluded)."""
    import glob
    import gzip

    os.environ.update(LAB_ENV)
    if w8 is not None:
        os.environ["VLLM_EXL3_DENSE_W8"] = str(w8)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(out), quantization="exl3", load_format="exl3", dtype="bfloat16",
        max_model_len=max_len, max_num_seqs=4, max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85, mm_encoder_tp_mode="data",
        tokenizer_mode="deepseek_v41", enable_prefix_caching=False, seed=0,
        enforce_eager=True,
        profiler_config={"profiler": "torch", "torch_profiler_dir": str(profile_dir)},
    )
    if mode == "prefill":
        # Two 2048-token prompts, one token each: prefill chunks only.
        prompt = [
            {"prompt_token_ids": [(13 * i + b) % 100000 + 3 for i in range(2048)]}
            for b in range(2)
        ]
        params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    else:
        prompt = [{"prompt_token_ids": [(13 * i) % 100000 + 3 for i in range(512)]}]
        params = SamplingParams(max_tokens=65, temperature=0.0, ignore_eos=True)
    llm.generate(prompt, params)
    llm.start_profile()
    llm.generate(prompt, params)
    llm.stop_profile()
    del llm
    totals: dict[str, float] = {}
    for path in glob.glob(str(profile_dir / "**" / "*.json*"), recursive=True):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            trace = json.load(f)
        for event in trace.get("traceEvents", []):
            if event.get("cat") == "kernel":
                name = event.get("name", "?")[:90]
                totals[name] = totals.get(name, 0.0) + float(event.get("dur", 0.0))
    total = sum(totals.values())
    top = sorted(totals.items(), key=lambda kv: -kv[1])[:25]
    return {"kernel_us_total": round(total), "top": [[n, round(us)] for n, us in top]}


def measure(out: Path, w8: Path | None, dspark: bool, max_len: int, util: float,
            spec_tokens: int, images: int | None = None, batched_tokens: int = 2048) -> dict:
    """Boot as a serving rank would (FULL_DECODE_ONLY, 4 sequences, 2048-token
    steps, the given max-model-len and utilization, optional DSpark) and run
    one short request; the memory accounting is in vLLM's own log lines
    (model loading, memory profiling, KV cache, CUDA graphs)."""
    os.environ.update(LAB_ENV)
    if w8 is not None:
        os.environ["VLLM_EXL3_DENSE_W8"] = str(w8)
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=str(out), quantization="exl3", load_format="exl3", dtype="bfloat16",
        max_model_len=max_len, max_num_seqs=4, max_num_batched_tokens=batched_tokens,
        gpu_memory_utilization=util, mm_encoder_tp_mode="data",
        tokenizer_mode="deepseek_v41", enable_prefix_caching=False, seed=0,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        swa_bounded_replay=False,
    )
    if images is not None:
        kwargs["limit_mm_per_prompt"] = {"image": images}
    if dspark:
        spec = {"method": "dspark"}
        if spec_tokens:
            spec["num_speculative_tokens"] = spec_tokens
        kwargs["speculative_config"] = spec
    llm = LLM(**kwargs)
    prompts = [{"prompt_token_ids": [(13 * i) % 100000 + 3 for i in range(2048)]}] * 4
    llm.generate(prompts, SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True))
    import torch

    free, total = torch.cuda.mem_get_info()
    del llm
    return {"measure": True, "dspark": dspark, "max_len": max_len, "util": util,
            "client_view_free_gib": round(free / 2**30, 2), "total_gib": round(total / 2**30, 2)}


def run(out: Path, w8: Path | None, dspark: bool, max_len: int) -> dict:
    os.environ.update(LAB_ENV)
    if w8 is not None:
        os.environ["VLLM_EXL3_DENSE_W8"] = str(w8)
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=str(out), quantization="exl3", load_format="exl3", dtype="bfloat16",
        max_model_len=max_len, max_num_seqs=4, max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85, mm_encoder_tp_mode="data",
        tokenizer_mode="deepseek_v41", enable_prefix_caching=False, seed=0,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    )
    if dspark:
        kwargs["speculative_config"] = {"method": "dspark"}
    t0 = time.time()
    llm = LLM(**kwargs)
    boot = time.time() - t0
    result = {"boot_s": round(boot, 1), "dspark": dspark}
    for batch, prompt_len, out_len in ((1, 2048, 1), (1, 2048, 129), (4, 2048, 129)):
        prompts = [
            {"prompt_token_ids": [(13 * i + 7 * b) % 100000 + 3 for i in range(prompt_len)]}
            for b in range(batch)
        ]
        params = SamplingParams(max_tokens=out_len, temperature=0.0, ignore_eos=True)
        llm.generate(prompts, params)  # warm
        times = []
        for _ in range(int(os.environ.get("DSV41_BENCH_REPEATS", "3"))):
            t0 = time.time()
            llm.generate(prompts, params)
            times.append(time.time() - t0)
        times.sort()
        result[f"b{batch}_p{prompt_len}_o{out_len}_s"] = round(times[len(times) // 2], 4)
        result[f"b{batch}_p{prompt_len}_o{out_len}_spread"] = round(times[-1] - times[0], 4)
    p1 = result["b1_p2048_o1_s"]
    result["decode_ms_per_step_b1"] = round(
        (result["b1_p2048_o129_s"] - p1) / 128 * 1000, 2
    )
    result["decode_ms_per_step_b4"] = round(
        (result["b4_p2048_o129_s"] - result["b1_p2048_o1_s"] * 4) / 128 * 1000, 2
    )
    # DSV41_BENCH_LONG=N[,N...]: one N-token prompt, prefill alone and with 64
    # decode steps (median of DSV41_BENCH_REPEATS), at a max-model-len >= N + 65.
    for n in [int(x) for x in os.environ.get("DSV41_BENCH_LONG", "").split(",") if x]:
        prompt = [{"prompt_token_ids": [(13 * i + 5) % 100000 + 3 for i in range(n)]}]
        for out_len in (1, 65):
            params = SamplingParams(max_tokens=out_len, temperature=0.0, ignore_eos=True)
            llm.generate(prompt, params)  # warm
            times = []
            for _ in range(int(os.environ.get("DSV41_BENCH_REPEATS", "3"))):
                t0 = time.time()
                llm.generate(prompt, params)
                times.append(time.time() - t0)
            times.sort()
            result[f"long{n}_o{out_len}_s"] = round(times[len(times) // 2], 4)
            result[f"long{n}_o{out_len}_spread"] = round(times[-1] - times[0], 4)
        result[f"long{n}_decode_ms_per_step"] = round(
            (result[f"long{n}_o65_s"] - result[f"long{n}_o1_s"]) / 64 * 1000, 2
        )
    del llm
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--w8-header", type=Path)
    parser.add_argument("--w8-size", type=int, default=8098376784)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--routed-25", default="0")
    parser.add_argument("--dspark", action="store_true")
    parser.add_argument("--draft-bits-x2", type=int, default=5,
                        help="the DSpark drafter's routed experts: 5 (2.5 bpw) or 4 (2.0)")
    parser.add_argument("--realistic", action="store_true")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--stage", default="default", choices=("default", "last"))
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--util", type=float, default=0.97)
    parser.add_argument("--spec-tokens", type=int, default=0)
    parser.add_argument("--images", type=int)
    parser.add_argument("--profile-mode", default="decode", choices=("decode", "prefill"))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--max-len", type=int, default=16384)
    parser.add_argument("--batched-tokens", type=int, default=2048,
                        help="--measure: max-num-batched-tokens")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    routed_25 = {int(x) for x in args.routed_25.split(",") if x}
    if not args.skip_build:
        build(args.out, args.tokenizer_dir, args.layers, routed_25, args.realistic, args.stage,
              args.draft_bits_x2)
    w8 = None
    if args.w8_header is not None:
        w8 = sparse_sidecar(args.w8_header, args.out, args.w8_size)
    if args.build_only:
        return
    if args.measure:
        print(json.dumps(measure(args.out, w8, args.dspark, args.max_len, args.util,
                                 args.spec_tokens, args.images, args.batched_tokens)),
              flush=True)
        return
    if args.profile_dir is not None:
        print(
            json.dumps(
                profile(args.out, w8, args.max_len, args.profile_dir, args.profile_mode)
            ),
            flush=True,
        )
        return
    print(json.dumps(run(args.out, w8, args.dspark, args.max_len)), flush=True)


if __name__ == "__main__":
    main()
