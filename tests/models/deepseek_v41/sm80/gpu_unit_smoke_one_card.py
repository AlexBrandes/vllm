# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The lab unit's own ``vllm serve`` command line (recipes/dsv41-flash-pp3-lab)
on ONE GPU: its flags and environment, the tiny checkpoint of
``gpu_smoke_tiny.py`` instead of /model, PP3 4,3,1 with every rank on the
one card (``gpu_pp_one_card.py``'s shim), a 1 GiB KV pool; then /v1/models,
a chat, a tool-choice chat, a completion with logprobs and a repeated greedy
completion. Prints one JSON line; the server log goes to ``--log``.

    python tests/models/deepseek_v41/sm80/gpu_unit_smoke_one_card.py \
        --model /work/out/tiny --log /work/logs/unit-smoke.log
"""
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request

import argparse  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
sys.path.insert(0, ROOT)
from tests.models.deepseek_v41.sm80.gpu_pp_one_card import write_shim  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--log", required=True)
ap.add_argument("--port", type=int, default=18599)
ap.add_argument("--unit", default=os.path.join(ROOT, "recipes/dsv41-flash-pp3-lab/cmp-dsv41-pp3-lab.service"))
args = ap.parse_args()
UNIT, MODEL, PORT = args.unit, args.model, args.port

line = next(l for l in open(UNIT).read().splitlines() if l.startswith("ExecStart="))
words = shlex.split(line[len("ExecStart="):])
image_at = next(i for i, w in enumerate(words) if w.startswith("lab-dsv41-pp3:"))
docker_part, serve = words[:image_at], words[image_at + 1:]
env = dict(os.environ)
skip = {"PATH", "CUDA_VISIBLE_DEVICES", "VLLM_EXL3_DENSE_W8", "VLLM_ENGRAM_ROW_CACHE_GIB"}
for i, w in enumerate(docker_part):
    if w == "-e":
        k, _, v = docker_part[i + 1].partition("=")
        if k not in skip:
            env[k] = v
env["VLLM_PP_LAYER_PARTITION"] = "4,3,1"
env["VLLM_ENGRAM_ROW_CACHE_GIB"] = "1"
shim = "/tmp/dsv41-unit-smoke-shim"
from pathlib import Path  # noqa: E402

write_shim(Path(shim))
env["PYTHONPATH"] = shim + ":" + ROOT + ":" + env.get("PYTHONPATH", "")
env.update(DSV41_PP_ONE_CARD="1", NCCL_IB_DISABLE="1", NCCL_SOCKET_IFNAME="lo",
           NCCL_P2P_DISABLE="1", NCCL_SHM_DISABLE="1")


def replace(flag, value):
    i = serve.index(flag)
    serve[i + 1] = value


serve[serve.index("/model")] = MODEL
replace("--port", str(PORT))
replace("--max-model-len", "4096")
replace("--gpu-memory-utilization", "0.25")
replace("--hf-overrides", json.dumps({"engram_table_dir": MODEL + "/engram-tables"}))
serve += ["--kv-cache-memory-bytes", str(1 << 30), "--distributed-executor-backend", "mp"]
print("serve:", " ".join(serve), flush=True)
print("env:", {k: env[k] for k in sorted(env) if k.startswith(("VLLM_", "PYTORCH"))}, flush=True)
log = open(args.log, "w")
proc = subprocess.Popen(serve, env=env, stdout=log, stderr=subprocess.STDOUT)
base = f"http://127.0.0.1:{PORT}"
result = {}


def call(path, body=None, timeout=120):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())


try:
    t0 = time.time()
    while time.time() - t0 < 900:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited {proc.returncode}")
        try:
            status, models = call("/v1/models", timeout=5)
            break
        except Exception:  # noqa: BLE001
            time.sleep(5)
    else:
        raise RuntimeError("server not ready in 900 s")
    result["ready_s"] = round(time.time() - t0, 1)
    result["models"] = [m["id"] for m in models["data"]]
    name = result["models"][0]
    status, out = call("/v1/chat/completions", {
        "model": name, "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 16, "temperature": 0})
    result["chat"] = {"status": status, "finish": out["choices"][0]["finish_reason"],
                      "usage": out.get("usage"), "content_len": len(out["choices"][0]["message"].get("content") or "")}
    status, out = call("/v1/chat/completions", {
        "model": name, "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "weather",
                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
        "tool_choice": "auto", "max_tokens": 16, "temperature": 0})
    result["tools"] = {"status": status, "finish": out["choices"][0]["finish_reason"]}
    status, out = call("/v1/completions", {"model": name, "prompt": "The capital of France is",
                                           "max_tokens": 8, "temperature": 0, "logprobs": 2})
    result["completion"] = {"status": status, "tokens": len(out["choices"][0]["logprobs"]["tokens"])}
    # The same request twice: identical text.
    a = call("/v1/completions", {"model": name, "prompt": "1 2 3 4", "max_tokens": 16, "temperature": 0})[1]
    b = call("/v1/completions", {"model": name, "prompt": "1 2 3 4", "max_tokens": 16, "temperature": 0})[1]
    result["repeat_equal"] = a["choices"][0]["text"] == b["choices"][0]["text"]
except Exception as exc:  # noqa: BLE001
    result["error"] = f"{type(exc).__name__}: {exc}"[:500]
finally:
    proc.terminate()
    try:
        proc.wait(60)
    except subprocess.TimeoutExpired:
        proc.kill()
print(json.dumps(result), flush=True)
