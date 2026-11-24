#!/usr/bin/env bash
# Host-side: build sparse_mla.so (op dsv41_mla::sparse_mla) inside a lab image (its nvcc and torch), no GPU.
#   build.sh <image> <dir containing sparse_mla_sm80.cu>   -> <dir>/sparse_mla.so
set -euo pipefail
IMG=$1; DIR=$(cd "$2" && pwd)
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -e NVIDIA_VISIBLE_DEVICES=void \
  -v "$DIR":/src --entrypoint= "$IMG" bash -c '
set -e
PY=$(command -v python3); SP=$($PY -c "import site;print(site.getsitepackages()[0])"); TI=$SP/torch/include
ABI=$($PY -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
nvcc -O3 -std=c++20 -gencode=arch=compute_80,code=sm_80 --expt-relaxed-constexpr -DNDEBUG -Xcompiler -fPIC -shared \
  -isystem $TI -isystem $TI/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=$ABI -L$SP/torch/lib \
  -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -Xptxas=-v /src/sparse_mla_sm80.cu -o /src/sparse_mla.so 2>&1 | grep -E "error|registers|spill" || true
sha256sum /src/sparse_mla.so'
