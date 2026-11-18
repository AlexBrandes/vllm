# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The first converted DeepSeek-V4.1 layers through the engine on one GPU.

Builds a checkpoint directory from the converter's finished per-module files
(``qtensors/layers.<i>.safetensors``, the Engram projection, the embedding;
linked, not copied), cut to those layers: the real config with
``num_hidden_layers`` = the number of layers, the last one turned into the
ratio-1 KV / index / candidate source when it is not one already (ratio 2 is
kept when it is the last KV source; the decoder replay then stays off), the
real Engram tables (``engram_table_dir`` = the official shards), unit final
norm and a header-only EXL3 head the dense W8 sidecar serves (its real INT8
head). Then serves it: greedy tokens for a few prompts with and without the
W8 sidecar (dense from EXL3 instead), eager vs FULL_DECODE_ONLY graphs, the
first-token top logprobs of both arms, and median timings. A cut model makes
no sensible text; the point is real tensors (names, Q4N-X trellis, scales)
through the served paths.

    python tests/models/deepseek_v41/sm80/gpu_real_layers.py --qtensors DIR \
        --layers 3 --src OFFICIAL_DIR --out OUT --w8 SIDECAR [--mode w8|exl3] [--graphs]
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path


def build(qtensors: Path, layers: int, src: Path, out: Path) -> None:
    from tests.models.deepseek_v41.sm80 import exl3_fixture as fixture

    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.safetensors"):
        f.unlink()
    config = json.loads((src / "config.json").read_text())
    config["quantization_config"] = json.loads(
        (fixture.FIXTURES / "config.json").read_text()
    )["quantization_config"]
    text = config["text_config"]
    ratios = text["compress_ratios"]
    text["compress_ratios"] = ratios[:layers] + ratios[-text["num_nextn_predict_layers"]:]
    text["num_hidden_layers"] = layers
    kv = sorted(s for s in text["kv_source_layer_ids"] if s < layers)
    index = sorted(s for s in text["index_source_layer_ids"] if s < layers)
    text["kv_source_layer_ids"] = kv
    text["index_source_layer_ids"] = index
    text["candidate_source_layer_id"] = max(index) if index else -1
    keep = [i for i, l in enumerate(text["engram_layer_ids"]) if l < layers]
    text["engram_layer_ids"] = [text["engram_layer_ids"][i] for i in keep]
    text["engram_num_embeddings"] = [text["engram_num_embeddings"][i] for i in keep]
    text["dspark_target_layer_ids"] = list(range(max(0, layers - 3), layers))
    config["engram_table_dir"] = str(src)
    (out / "config.json").write_text(json.dumps(config))
    names = ["embed"] + [f"layers.{i}" for i in range(layers)]
    names += [f"layers.{l}.engram" for l in text["engram_layer_ids"]]
    for name in names:
        (out / f"{name}.safetensors").symlink_to(qtensors / f"{name}.safetensors")
    # Final norm (ones) and a header-only EXL3 head (zero trellis, mul1)
    # for the W8 sidecar's real head to replace.
    hidden, vocab = text["hidden_size"], text["vocab_size"]
    tensors = {
        "norm.weight": ("BF16", [hidden]),
        "head.trellis": ("I16", [hidden // 16, vocab // 16, 8 * 12]),
        "head.suh": ("F16", [hidden]),
        "head.svh": ("F16", [vocab]),
        "head.mul1": ("I32", []),
        # The image sentinels come with the vision tower (converted last).
        "image_start": ("BF16", [hidden]),
        "image_end": ("BF16", [hidden]),
        "image_newline": ("BF16", [hidden]),
    }
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = fixture.tensor_bytes(dtype, tuple(shape))
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(out / "zz-head-norm.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        f.truncate(8 + len(raw) + offset)
        lo = header["norm.weight"]["data_offsets"][0]
        os.pwrite(f.fileno(), struct.pack("<H", 0x3F80) * hidden, 8 + len(raw) + lo)
        lo = header["head.mul1"]["data_offsets"][0]
        os.pwrite(f.fileno(), struct.pack("<I", fixture.MUL1), 8 + len(raw) + lo)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        target = out / name
        if not target.exists():
            target.symlink_to(src / name)


def run(out: Path, w8: Path | None, graphs: bool) -> dict:
    from tests.models.deepseek_v41.sm80.gpu_bench_realwidth import LAB_ENV

    os.environ.update(LAB_ENV)
    if w8 is not None:
        os.environ["VLLM_EXL3_DENSE_W8"] = str(w8)
    else:
        os.environ.pop("VLLM_EXL3_DENSE_W8", None)
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=str(out), quantization="exl3", load_format="exl3", dtype="bfloat16",
        max_model_len=8192, max_num_seqs=4, max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85, mm_encoder_tp_mode="data",
        tokenizer_mode="deepseek_v41", enable_prefix_caching=False, seed=0,
        # The vision tower is converted last; this cut has none (text only).
        limit_mm_per_prompt={"image": 0},
    )
    if graphs:
        kwargs["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY"}
    else:
        kwargs["enforce_eager"] = True
    t0 = time.time()
    llm = LLM(**kwargs)
    boot = time.time() - t0
    tok = llm.get_tokenizer()
    texts = [
        "The capital of France is",
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
        " ".join(["The quick brown fox jumps over the lazy dog."] * 90),
    ]
    prompts = [{"prompt_token_ids": tok.encode(t)} for t in texts]
    params = SamplingParams(max_tokens=16, temperature=0.0, logprobs=5, ignore_eos=True)
    outs = llm.generate(prompts, params)
    again = llm.generate(prompts, params)
    timing = SamplingParams(max_tokens=129, temperature=0.0, ignore_eos=True)
    long = [{"prompt_token_ids": [(13 * i) % 100000 + 3 for i in range(2048)]}]
    llm.generate(long, timing)
    times = []
    for _ in range(3):
        t1 = time.time()
        llm.generate(long, timing)
        times.append(time.time() - t1)
    result = {
        "w8": w8 is not None,
        "graphs": graphs,
        "boot_s": round(boot, 1),
        "prompt_tokens": [len(p["prompt_token_ids"]) for p in prompts],
        "tokens": [list(o.outputs[0].token_ids) for o in outs],
        "repeat_equal": [
            list(a.outputs[0].token_ids) == list(b.outputs[0].token_ids)
            for a, b in zip(outs, again)
        ],
        "first_top5": [
            sorted(((t, round(lp.logprob, 4)) for t, lp in o.outputs[0].logprobs[0].items()),
                   key=lambda x: -x[1])
            for o in outs
        ],
        "finite": all(
            lp.logprob == lp.logprob for o in outs for step in o.outputs[0].logprobs
            for lp in step.values()
        ),
        "p2048_o129_median_s": round(sorted(times)[1], 3),
    }
    del llm
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qtensors", type=Path, required=True)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--w8", type=Path)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    if not args.skip_build:
        build(args.qtensors, args.layers, args.src, args.out)
    if args.build_only:
        return
    print(json.dumps(run(args.out, args.w8, args.graphs)), flush=True)


if __name__ == "__main__":
    main()
