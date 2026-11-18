#!/bin/bash
# Run the lab unit's container by hand on the CMP host, inside a lease that holds cards 1, 2 and 4
# (the coordinator's two-step host/fleet protocol; never without it). Reads ExecStart from
# cmp-dsv41-pp3-lab.service next to this script, so the unit file stays the single source of flags.
#   IMAGE=lab-dsv41-pp3:<sha> [PARTITION=15,10,15] [DSPARK=0] [EMBED_INT8=0] [BATCHED=1024]
#   [MAX_MODEL_LEN=98304] [UTIL=0.97] [GATE_GEMV_ROWS=64] [EXTRA_ENV="-e K=V ..."] ./run-lab.sh
# DSPARK=0 drops --speculative-config; EMBED_INT8=0 drops VLLM_DEEPSEEK_V41_EMBED_INT8 (the fallback is BATCHED=1024).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${IMAGE:?set IMAGE=lab-dsv41-pp3:<plugin sha>}"
CMD=$(python3 - "$HERE/cmp-dsv41-pp3-lab.service" <<'PY'
import sys
lines = open(sys.argv[1]).read().splitlines()
line = next(l for l in lines if l.startswith("ExecStart="))
print(line[len("ExecStart="):])
PY
)
CMD=${CMD/lab-dsv41-pp3:PLUGIN_SHA/$IMAGE}
if [ -n "${PARTITION:-}" ]; then CMD=${CMD/VLLM_PP_LAYER_PARTITION=15,10,15/VLLM_PP_LAYER_PARTITION=$PARTITION}; fi
if [ -n "${EXTRA_ENV:-}" ]; then CMD=${CMD/--shm-size 16g/--shm-size 16g $EXTRA_ENV}; fi
if [ "${DSPARK:-1}" = 0 ]; then CMD=${CMD/ --speculative-config \'\{\"method\": \"dspark\"\}\'/}; fi
if [ "${EMBED_INT8:-1}" = 0 ]; then CMD=${CMD/ -e VLLM_DEEPSEEK_V41_EMBED_INT8=1/}; fi
if [ -n "${BATCHED:-}" ]; then
  CMD=${CMD/--max-num-batched-tokens 2048/--max-num-batched-tokens $BATCHED}
  CMD=${CMD/--prefill-chunk-with-decodes 2048/--prefill-chunk-with-decodes $BATCHED}
fi
if [ -n "${MAX_MODEL_LEN:-}" ]; then CMD=${CMD/--max-model-len 131072/--max-model-len $MAX_MODEL_LEN}; fi
if [ -n "${GATE_GEMV_ROWS:-}" ]; then
  CMD=${CMD/VLLM_DEEPSEEK_V41_GATE_GEMV_MAX_ROWS=2048/VLLM_DEEPSEEK_V41_GATE_GEMV_MAX_ROWS=$GATE_GEMV_ROWS}
fi
if [ -n "${UTIL:-}" ]; then CMD=${CMD/--gpu-memory-utilization 0.97/--gpu-memory-utilization $UTIL}; fi
docker rm -f dsv41-pp3-lab >/dev/null 2>&1 || true
echo "$CMD" | sed 's/ -e / \\\n  -e /g' >&2
eval "$CMD"
