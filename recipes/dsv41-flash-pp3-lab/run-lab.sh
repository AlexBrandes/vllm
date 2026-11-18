#!/bin/bash
# Run the lab unit's container by hand on the CMP host, inside a lease that holds cards 1, 2 and 4
# (the coordinator's two-step host/fleet protocol; never without it). Reads ExecStart from
# cmp-dsv41-pp3-lab.service next to this script, so the unit file stays the single source of flags.
#   IMAGE=lab-dsv41-pp3:<sha> [PARTITION=15,10,15] [DSPARK=1] [EXTRA_ENV="-e K=V ..."] ./run-lab.sh
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
if [ "${DSPARK:-0}" = 1 ]; then CMD="$CMD --speculative-config '{\"method\": \"dspark\"}'"; fi
docker rm -f dsv41-pp3-lab >/dev/null 2>&1 || true
echo "$CMD" | sed 's/ -e / \\\n  -e /g' >&2
eval "$CMD"
