#!/usr/bin/env bash
# Build the K3 Marlin route dispatcher (marlin_host.so) for the vllm-cmp170hx engine INSIDE the
# engine image (its torch headers, its vLLM _C library), no GPU needed:
#   docker run --rm --entrypoint= -v SRC:/src:ro -v OUT:/out IMAGE bash /src/build.sh /out
set -euo pipefail
SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT=${1:?build dir}
SP=/opt/venv/lib/python3.12/site-packages
mkdir -p "$OUT"
cp -R "$SRC"/generate_host.py "$SRC"/routes.json "$SRC"/marlin.upstream.cu "$SRC"/kernel.h \
      "$SRC"/core "$SRC"/libtorch_stable "$OUT"/
cd "$OUT"
python3 generate_host.py
# STAGES=1 (experimental, not used by the recipes): link stage-variant kernels (marlin-stages/build.sh output,
# STAGES_LIB=/path/libcmp_marlin_stages.so), loaded from next to marlin_host.so.
EXTRA_LIBS=()
if [ "${STAGES:-}" = 1 ]; then
  cp "${STAGES_LIB:?STAGES_LIB}" "$OUT/libcmp_marlin_stages.so"
  EXTRA_LIBS=(-L"$OUT" -lcmp_marlin_stages -Wl,-rpath,'$ORIGIN')
fi
CUDA_VISIBLE_DEVICES= g++ -std=c++17 -O2 -shared -fPIC -DUSE_CUDA -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I"$OUT" -I"$OUT/libtorch_stable" -I"$SP/torch/include" -I"$SP/torch/include/torch/csrc/api/include" \
  -I/usr/local/cuda/include \
  marlin_host.cpp -o marlin_host.so \
  "$SP/vllm/_C_stable_libtorch.abi3.so" "$SP/torch/lib/libtorch.so" "$SP/torch/lib/libtorch_cpu.so" \
  "$SP/torch/lib/libtorch_cuda.so" "$SP/torch/lib/libc10.so" "$SP/torch/lib/libc10_cuda.so" \
  "${EXTRA_LIBS[@]}" -L/usr/local/cuda/lib64 -lcudart -Wl,-z,defs \
  -Wl,-rpath,"$SP/torch/lib"
python3 - <<PY
import hashlib, json
import os
files = (["libcmp_marlin_stages.so"] if os.path.exists("libcmp_marlin_stages.so") else []) + ["generate_host.py", "routes.json", "marlin_host_abi.h", "marlin.upstream.cu", "kernel.h",
         "core/scalar_type.hpp", "libtorch_stable/torch_utils.h", "marlin_host.cpp", "marlin_host.so"]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"schemaVersion": 1, "stages": os.path.exists("libcmp_marlin_stages.so"),
           "files": {f: sha(f) for f in files},
           "deviceLibrary": "$SP/vllm/_C_stable_libtorch.abi3.so",
           "deviceLibrarySha256": sha("$SP/vllm/_C_stable_libtorch.abi3.so")},
          open("source-manifest.json", "w"), indent=1)
print("marlin_host.so", sha("marlin_host.so"))
PY
