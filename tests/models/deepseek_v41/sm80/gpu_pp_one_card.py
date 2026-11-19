# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline-parallel DeepSeek-V4.1 on ONE GPU: every PP rank is a separate
worker process on the same card, so the shadow sources, the extra
IntermediateTensors keys, their relay through middle stages and CUDA graphs on
later stages run for real (NCCL send/recv between processes) before the
three-card boot.

Two test-only patches, installed through a ``vllm.general_plugins`` entry
point this script writes into a scratch directory (``PYTHONPATH``), active
only with ``DSV41_PP_ONE_CARD=1``:

* every worker's local GPU index is 0 (the one visible card);
* each rank gets its own ``NCCL_HOSTID`` before NCCL starts, so NCCL treats
  the ranks as separate hosts (its duplicate-GPU check is per host) and
  connects them over loopback sockets.

Arms (tiny random EXL3 checkpoint of ``gpu_smoke_tiny.py``: 8 layers, KV
groups 2-4 and 5-7, index sources 2/5/6, candidate 5, Engram 1 and 4, DSpark
targets 5-7), each in a fresh process, FULL_DECODE_ONLY graphs:

* PP1 (reference);
* PP2 ``4,4`` (stage 1 shadows source 2);
* PP3 ``4,3,1`` (stage 1 shadows 2; stage 2 shadows 5 and 6, as the real
  stage 2 shadows 20 and 24);
* PP3 ``3,1,4`` (stage 2 needs source 2 from stage 0: relayed through stage 1);
* optionally the same with DSpark;
* ``--lab``: every arm with the lab unit's scheduling and mHC switches
  (homogeneous PP steps, spread decodes, 512-token prefill chunks beside
  decodes, fused mHC with finish v2 and the uniform kernel A).

An arm ``P@V`` runs partition ``P`` with the environment of variant ``V``
(``--variant V=KEY=VALUE``, repeatable), e.g. ``8,8@pack`` with
``--variant pack=VLLM_DEEPSEEK_V41_KV_PACK=1`` compares packed KV records to
the BF16 rows of arm ``8``. ``--prefix-caching`` adds prompts that share a
2048-token prefix with an earlier one (prefix-cache hits; each output records
its ``num_cached_tokens``).

The first arm is the reference (any checkpoint works with ``--out``, e.g. a
real-layer cut of ``gpu_real_layers.py --build-only`` with ``--w8`` and
``--bench-env``). Each prompt is generated alone (identical batch composition in every arm),
greedy, with top-3 logprobs; every PP arm must equal PP1 token for token and
logprob for logprob. Prints one JSON line.

    python tests/models/deepseek_v41/sm80/gpu_pp_one_card.py --out /work/out/tiny \
        [--dspark] [--lab] [--arms 8,4:4,4:3:1,3:1:4]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

LAB_ENV = {
    "VLLM_PP_HOMOGENEOUS_STEPS": "1",
    "VLLM_PP_SPREAD_DECODES": "1",
    "VLLM_DEEPSEEK_V41_FUSED_MHC": "1",
    "VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH": "1",
    "VLLM_DEEPSEEK_V41_FUSED_MHC_FINISH_V2": "1",
    "VLLM_SM80_MHC_UNIFORM_KERNEL_A": "1",
}

SHIM = '''
import os


def register():
    if os.environ.get("DSV41_PP_ONE_CARD") != "1":
        return
    from vllm.config import parallel
    from vllm.v1.worker import gpu_worker

    if getattr(gpu_worker, "_pp_one_card", False):
        return
    gpu_worker._pp_one_card = True
    parallel.ParallelConfig.get_local_gpu_index = (
        lambda self, local_rank, dp_local_rank=None: 0
    )
    inner = gpu_worker.init_worker_distributed_environment

    def init_dist(vllm_config, rank, *args, **kwargs):
        os.environ["NCCL_HOSTID"] = f"dsv41-pp-one-card-{rank}"
        return inner(vllm_config, rank, *args, **kwargs)

    gpu_worker.init_worker_distributed_environment = init_dist

    rows = os.environ.get("DSV41_GATE_GEMV_MAX_ROWS")
    if rows:
        # Diagnosis only: the FP64 gate GEMV (row-independent) for prefill-sized
        # calls too (VLLM_DEEPSEEK_V41_GATE_GEMV=1 must be set).
        from vllm_cmp170hx.models.deepseek_v41.ops import sm80_gate_gemv

        sm80_gate_gemv.MAX_ROWS = int(rows)

    dump = os.environ.get("DSV41_PP_DRAFT_DUMP")
    if dump:
        import json

        from vllm.v1.worker.gpu import model_runner

        make = model_runner.init_speculator

        def init_speculator(*args, **kwargs):
            speculator = make(*args, **kwargs)
            propose = speculator.propose

            def recorded(*args, **kw):
                drafts = propose(*args, **kw)
                input_batch = args[0] if args else kw["input_batch"]
                aux = args[4] if len(args) > 4 else kw.get("aux_hidden_states")
                if not kw.get("dummy_run") and not kw.get("is_profile"):
                    n, r = input_batch.num_tokens, input_batch.num_reqs
                    line = {
                        "draft": drafts[:r].tolist(),
                        "aux": [float(a[:n].double().sum()) for a in aux or []],
                    }
                    with open(dump, "a") as f:
                        f.write(json.dumps(line) + "\\n")
                return drafts

            speculator.propose = recorded
            return speculator

        model_runner.init_speculator = init_speculator
'''


def write_shim(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "dsv41_pp_one_card.py").write_text(SHIM)
    info = directory / "dsv41_pp_one_card-0.0.dist-info"
    info.mkdir(exist_ok=True)
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: dsv41-pp-one-card\nVersion: 0.0\n"
    )
    (info / "entry_points.txt").write_text(
        "[vllm.general_plugins]\ndsv41_pp_one_card = dsv41_pp_one_card:register\n"
    )


def run_arm(
    out: Path, partition: str, dspark: bool, prompt_lens: list[int], lab: bool,
    swa_replay: bool = True, batched_tokens: int = 1024, image: bool = False,
    prefix_caching: bool = False,
) -> dict:
    stages = len(partition.split(","))
    if stages > 1:
        os.environ["VLLM_PP_LAYER_PARTITION"] = partition
    else:
        os.environ.pop("VLLM_PP_LAYER_PARTITION", None)
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=str(out), quantization="exl3", load_format="exl3", dtype="bfloat16",
        max_model_len=4096, max_num_seqs=4, max_num_batched_tokens=batched_tokens,
        gpu_memory_utilization=0.2, kv_cache_memory_bytes=1 << 30,
        mm_encoder_tp_mode="data", tokenizer_mode="deepseek_v41",
        enable_prefix_caching=prefix_caching, seed=0,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        pipeline_parallel_size=stages, distributed_executor_backend="mp",
    )
    if dspark:
        kwargs["speculative_config"] = {"method": "dspark", "num_speculative_tokens": 4}
    if not swa_replay:
        kwargs["swa_bounded_replay"] = False
    if lab:
        kwargs["additional_config"] = {"sm80": {"prefill_chunk_with_decodes": 512}}
    llm = LLM(**kwargs)
    params = SamplingParams(max_tokens=24, temperature=0.0, logprobs=3, ignore_eos=True)
    prompts = [
        {"prompt_token_ids": [(7 * i + 13 * n) % 100000 + 3 for i in range(n)]}
        for n in prompt_lens
    ]
    if prefix_caching:
        # A 2300-token prompt, then two that share its first 2048 tokens (hits
        # once it is cached), then the first again (a hit on all of it).
        base = [(11 * i + 5) % 100000 + 3 for i in range(2300)]
        prompts += [
            {"prompt_token_ids": base},
            {"prompt_token_ids": base[:2048] + [(17 * i + 9) % 100000 + 3 for i in range(200)]},
            {"prompt_token_ids": base[:2048] + [(19 * i + 2) % 100000 + 3 for i in range(37)]},
            {"prompt_token_ids": base},
        ]
    alone = [llm.generate([p], params)[0] for p in prompts]
    batched = llm.generate(prompts, params)
    image_out = None
    if image:
        # Vision on the first stage (the tower is a meta stub on later ones).
        from tests.models.deepseek_v41.sm80.gpu_smoke_tiny import _image_request

        chat = llm.chat(_image_request(), params)[0].outputs[0]
        image_out = {
            "tokens": list(chat.token_ids),
            "logprobs": [sorted((t, lp.logprob) for t, lp in step.items())
                         for step in chat.logprobs],
        }
    result = {
        "partition": partition,
        "dspark": dspark,
        "lab": lab,
        "tokens": [list(o.outputs[0].token_ids) for o in alone],
        "logprobs": [
            [sorted((t, lp.logprob) for t, lp in step.items()) for step in o.outputs[0].logprobs]
            for o in alone
        ],
        "batched_tokens": [list(o.outputs[0].token_ids) for o in batched],
        "num_cached_tokens": [o.num_cached_tokens for o in alone],
        "image": image_out,
        "batched_first_logprobs": [
            sorted((t, lp.logprob) for t, lp in o.outputs[0].logprobs[0].items())
            for o in batched
        ],
    }
    del llm
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default="8,4:4,4:3:1,3:1:4")
    parser.add_argument("--dspark", action="store_true")
    parser.add_argument("--lab", action="store_true")
    parser.add_argument("--image", action="store_true", help="also one image chat per arm")
    parser.add_argument("--batched-tokens", type=int, default=1024,
                        help="max-num-batched-tokens (smaller: prompts prefill in chunks)")
    parser.add_argument("--no-swa-replay", action="store_true",
                        help="every arm without upstream's decoder SWA replay (a cut "
                        "whose last KV source has ratio 2 cannot replay at PP1)")
    parser.add_argument("--env", action="append", default=[],
                        help="KEY=VALUE set for every arm, after --lab/--bench-env")
    parser.add_argument("--w8", help="dense W8 sidecar (VLLM_EXL3_DENSE_W8)")
    parser.add_argument("--bench-env", action="store_true",
                        help="the lab unit's EXL3 kernel switches (gpu_bench_realwidth.LAB_ENV)")
    parser.add_argument("--prompt-lens", default="12,300,900")
    parser.add_argument("--variant", action="append", default=[],
                        help="NAME=KEY=VALUE: arm P@NAME runs with KEY=VALUE")
    parser.add_argument("--prefix-caching", action="store_true")
    parser.add_argument("--arm")
    parser.add_argument("--result")
    parser.add_argument("--log-dir", help="keep each arm's engine log here")
    args = parser.parse_args()
    lens = [int(x) for x in args.prompt_lens.split(",")]
    root = Path(__file__).resolve().parents[4]
    if args.arm:
        sys.path.insert(0, str(root))
        res = run_arm(args.out, args.arm.split("@")[0].replace(":", ","), args.dspark,
                      lens, args.lab, not args.no_swa_replay, args.batched_tokens,
                      args.image, args.prefix_caching)
        Path(args.result).write_text(json.dumps(res))
        return
    shim = Path(tempfile.mkdtemp(prefix="dsv41-pp-one-card-"))
    write_shim(shim)
    env = dict(os.environ)
    env["DSV41_PP_ONE_CARD"] = "1"
    if args.lab:
        env.update(LAB_ENV)
    if args.bench_env:
        sys.path.insert(0, str(root))
        from tests.models.deepseek_v41.sm80.gpu_bench_realwidth import LAB_ENV as BENCH_ENV

        env.update(BENCH_ENV)
    if args.w8:
        env["VLLM_EXL3_DENSE_W8"] = args.w8
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(shim), str(root), env.get("PYTHONPATH", "")])
    )
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_SOCKET_IFNAME", "lo")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_SHM_DISABLE", "1")
    variants: dict[str, dict[str, str]] = {}
    for item in args.variant:
        name, _, kv = item.partition("=")
        key, _, value = kv.partition("=")
        variants.setdefault(name, {})[key] = value
    results, errors = {}, {}
    for arm in args.arms.split(","):
        tag = arm.replace(":", "-").replace("@", "-")
        result = shim / f"{tag}.json"
        cmd = [sys.executable, __file__, "--out", str(args.out), "--arm", arm,
               "--result", str(result), "--prompt-lens", args.prompt_lens]
        cmd += ["--dspark"] * args.dspark + ["--lab"] * args.lab
        cmd += ["--no-swa-replay"] * args.no_swa_replay
        cmd += ["--batched-tokens", str(args.batched_tokens)] + ["--image"] * args.image
        cmd += ["--prefix-caching"] * args.prefix_caching
        drafts = shim / f"{tag}.drafts.jsonl"
        arm_env = dict(env, **variants.get(arm.partition("@")[2], {}))
        if args.dspark:
            arm_env["DSV41_PP_DRAFT_DUMP"] = str(drafts)
        proc = subprocess.run(cmd, env=arm_env, capture_output=True, text=True, timeout=1500)
        if drafts.exists():
            results.setdefault("_drafts", {})[arm] = drafts.read_text().splitlines()
        if args.log_dir:
            log = Path(args.log_dir) / f"pp-one-card-{tag}.log"
            log.write_text(proc.stdout + proc.stderr)
            for keep in (result, drafts):
                if keep.exists():
                    (Path(args.log_dir) / keep.name).write_bytes(keep.read_bytes())
        if proc.returncode or not result.exists():
            errors[arm] = (proc.stderr or proc.stdout)[-3000:]
            continue
        results[arm] = json.loads(result.read_text())
    drafts = results.pop("_drafts", {})
    summary = {"dspark": args.dspark, "lab": args.lab, "errors": errors}
    first = args.arms.split(",")[0]
    reference = results.get(first)
    for arm, res in results.items():
        entry = {"tokens": res["tokens"]} if arm == first else {}
        if reference is not None and arm != first:
            entry["tokens_equal"] = res["tokens"] == reference["tokens"]
            entry["logprobs_equal"] = res["logprobs"] == reference["logprobs"]
            diffs = [
                abs(a[1] - b[1])
                for ra, rb in zip(res["logprobs"], reference["logprobs"])
                for sa, sb in zip(ra, rb)
                for a, b in zip(sa, sb)
                if a[0] == b[0]
            ]
            entry["max_logprob_diff"] = max(diffs) if diffs else None
            # The first generated position (prefill only, before any token
            # divergence feeds back) and where each prompt's tokens diverge.
            entry["first_step_max_diff"] = max(
                (abs(a[1] - b[1])
                 for ra, rb in zip(res["logprobs"], reference["logprobs"])
                 for a, b in zip(ra[0], rb[0]) if a[0] == b[0]),
                default=None,
            )
            entry["first_divergence"] = [
                next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
                for ta, tb in zip(res["tokens"], reference["tokens"])
            ]
            entry["batched_tokens_equal"] = res["batched_tokens"] == reference["batched_tokens"]
            entry["batched_first_logprobs_equal"] = (
                res["batched_first_logprobs"] == reference["batched_first_logprobs"]
            )
            entry["num_cached_tokens"] = res.get("num_cached_tokens")
        entry["batched_equals_alone"] = res["batched_tokens"] == res["tokens"]
        if res.get("image") is not None and reference is not None and arm != first:
            entry["image_equal"] = res["image"] == reference.get("image")
        # This arm's own batch sensitivity: first-step logprob change between
        # a prompt alone and the same prompt batched with the others.
        entry["batched_vs_alone_first_step_max_diff"] = max(
            (abs(a[1] - b[1])
             for ra, rb in zip(res["batched_first_logprobs"], res["logprobs"])
             for a, b in zip(ra, rb[0]) if a[0] == b[0]),
            default=None,
        )
        if arm in drafts:
            # Every proposal (draft tokens and the target-layer streams the
            # drafter read) must match PP1's, step for step.
            entry["propose_calls"] = len(drafts[arm])
            if arm != first and first in drafts:
                entry["drafts_equal"] = drafts[arm] == drafts[first]
            accepted = 0
            for line in drafts[arm]:
                accepted += sum(1 for row in json.loads(line)["draft"] for t in row if t)
            entry["draft_tokens_nonzero"] = accepted
        summary[arm] = entry
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
