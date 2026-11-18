# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end engine smoke of the sm_80 DeepSeek-V4.1 path on one GPU, with
the tiny random EXL3 checkpoint of ``exl3_fixture`` (real tensor names and
kinds, 8 layers, both Engram layers, compressed/indexed layers, DSpark
stages) and sparse all-zero Engram tables in the real base-checkpoint
layout. It boots vLLM (V2 runner, this plugin), generates greedily from
prompts long enough to reach the compressed and indexed attention, and
checks that eager and CUDA-graph decoding agree token for token.

Run from the plugin tree in a GPU container (one card):

    python tests/models/deepseek_v41/sm80/gpu_smoke_tiny.py \
        --out /work/out/tiny --tokenizer-dir /src [--graphs] [--dspark]

The tokenizer files come from the official checkpoint (``--tokenizer-dir``);
prompts are token ids below the tiny vocabulary. Prints one JSON line.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def build(out: Path, tokenizer_dir: Path) -> None:
    from tests.models.deepseek_v41.sm80 import exl3_fixture as fixture
    from tests.models.deepseek_v41.sm80.test_engram_storage import _write_table

    config = fixture.tiny_config()
    # The real vocabulary (embedding 129280 x 512): chat prompts and image
    # sentinels from the real tokenizer stay in range.
    config["text_config"]["vocab_size"] = 129280
    tables = out / "engram-tables"
    config["engram_table_dir"] = str(tables)
    if out.exists():
        shutil.rmtree(out)
    fixture.write_checkpoint(out, config)
    tables.mkdir(parents=True)
    text = config["text_config"]
    for layer, rows in zip(text["engram_layer_ids"], text["engram_num_embeddings"]):
        _write_table(tables, layer, rows, [0, 1, 2], f"engram-{layer}.safetensors")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if (tokenizer_dir / name).exists():
            shutil.copy(tokenizer_dir / name, out / name)
    for extra in ("encoding", "inference"):
        if (tokenizer_dir / extra).is_dir():
            shutil.copytree(tokenizer_dir / extra, out / extra)


def _image_request():
    import base64
    import io

    from PIL import Image

    image = Image.new("RGB", (336, 224))
    for x in range(336):
        for y in range(224):
            image.putpixel((x, y), ((x * 7) % 256, (y * 5) % 256, (x + y) % 256))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "Describe the picture."},
            ],
        }
    ]


def run(
    out: Path, graphs: bool, dspark: bool, prompt_lens: list[int], image: bool = False
) -> dict:
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=str(out),
        quantization="exl3",
        load_format="exl3",
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=1024,
        gpu_memory_utilization=0.30,
        mm_encoder_tp_mode="data",
        tokenizer_mode="deepseek_v41",
        enable_prefix_caching=False,
        seed=0,
    )
    if graphs:
        kwargs["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY"}
    else:
        kwargs["enforce_eager"] = True
    if dspark:
        kwargs["speculative_config"] = {"method": "dspark", "num_speculative_tokens": 4}
    t0 = time.time()
    llm = LLM(**kwargs)
    boot = time.time() - t0
    prompts = [
        {"prompt_token_ids": [(7 * i + 13 * n) % 100000 + 3 for i in range(n)]}
        for n in prompt_lens
    ]
    params = SamplingParams(max_tokens=24, temperature=0.0, logprobs=3, ignore_eos=True)
    t0 = time.time()
    outs = llm.generate(prompts, params)
    gen = time.time() - t0
    image_tokens = None
    if image:
        chat = llm.chat(_image_request(), params)
        image_tokens = list(chat[0].outputs[0].token_ids)
    result = {
        "image_tokens": image_tokens,
        "graphs": graphs,
        "dspark": dspark,
        "boot_s": round(boot, 1),
        "generate_s": round(gen, 2),
        "tokens": [list(o.outputs[0].token_ids) for o in outs],
        "first_logprobs": [
            sorted(
                ((t, round(lp.logprob, 4)) for t, lp in o.outputs[0].logprobs[0].items()),
                key=lambda x: -x[1],
            )
            for o in outs
        ],
    }
    del llm
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--dspark", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--image", action="store_true")
    parser.add_argument("--prompt-lens", default="12,300,900")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    if not args.skip_build:
        build(args.out, args.tokenizer_dir)
    if args.build_only:
        return
    lens = [int(x) for x in args.prompt_lens.split(",")]
    print(
        json.dumps(run(args.out, args.graphs, args.dspark, lens, image=args.image)),
        flush=True,
    )


if __name__ == "__main__":
    main()
