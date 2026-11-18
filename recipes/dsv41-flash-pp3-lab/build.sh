#!/bin/bash
# Build the DeepSeek-V4.1 lab image on the CMP host from local commits of dsv41/plugin and
# dsv41/vllm. Run on the Mac:  recipes/dsv41-flash-pp3-lab/build.sh [plugin-ref] [vllm-ref]
# The extension is compiled in a GPU-less container of the base image (no lease needed;
# build_standalone.py --out takes a directory);
# the image tag is lab-dsv41-pp3:<plugin short sha>.
set -euo pipefail
PLUGIN_REF=${1:-dsv41/plugin}
VLLM_REF=${2:-dsv41/vllm}
BASE=${BASE:-lab-glm53-q4nx-kf:6734e9fe03}
HOST=${HOST:-alex@192.168.86.146}
KEY=${KEY:-$HOME/.ssh/cmp170hx-build-20260906}
PUB=$HOME/llm-gateway/publish
P=$(git -C $PUB/dsv41-plugin rev-parse --short=10 "$PLUGIN_REF")
V=$(git -C $PUB/dsv41-vllm rev-parse --short=10 "$VLLM_REF")
CTX=/home/alex/dsv41-engine/image-$P
ssh -i $KEY $HOST "rm -rf $CTX && mkdir -p $CTX/plugin $CTX/vllm-overlay $CTX/src"
git -C $PUB/dsv41-plugin archive "$PLUGIN_REF" vllm_cmp170hx csrc | ssh -i $KEY $HOST "tar -x -C $CTX/src"
FILES=$(git -C $PUB/dsv41-vllm diff --name-only glm53/vllm-a0d7356b98 "$VLLM_REF" -- vllm)
git -C $PUB/dsv41-vllm archive "$VLLM_REF" $FILES | ssh -i $KEY $HOST "tar -x -C $CTX/vllm-overlay"
scp -q -i $KEY "$(dirname "$0")/Dockerfile" $HOST:$CTX/Dockerfile
# PREBUILT_SO=<host path>: reuse an _exl3_C built from the same csrc (skips the compile).
ssh -i $KEY $HOST "set -e; cd $CTX; cp -a src/vllm_cmp170hx plugin/; \
  if [ -n '${PREBUILT_SO:-}' ]; then cp '${PREBUILT_SO:-}' plugin/vllm_cmp170hx/_exl3_C.abi3.so; else \
  docker run --rm --entrypoint '' -v $CTX/src:/src -w /src/csrc/exl3 -e MAX_JOBS=8 $BASE \
    /opt/venv/bin/python build_standalone.py --out /src/ext --build-dir /src/ext-build; \
  cp src/ext/_exl3_C.abi3.so plugin/vllm_cmp170hx/_exl3_C.abi3.so; fi; \
  docker build -q --build-arg BASE=$BASE --build-arg PLUGIN_COMMIT=$P --build-arg VLLM_COMMIT=$V \
    -t lab-dsv41-pp3:$P ."
echo "built lab-dsv41-pp3:$P (plugin $P, vLLM $V)"
