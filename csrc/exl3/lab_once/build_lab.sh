#!/usr/bin/env bash
# Host-side: build exl3_i8_lab.so and cmp_i8mm.so inside the GLM production image
# (its nvcc and torch). CUTLASS: the first complete copy (cute + sm80 visitor
# epilogue) in the image, else $DIR/cutlass (copied from the 27B engine image by
# run_leg.sh). Usage: build_lab.sh <image> <dir>
set -euo pipefail
IMG=$1; DIR=$(cd "$2" && pwd)
docker run --rm --name lab-glm53-i8-build --network none --cpus 4 --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e NVIDIA_VISIBLE_DEVICES=void -v "$DIR":/lab --entrypoint= "$IMG" bash -c '
set -e
PY=$(command -v python3); SP=$($PY -c "import site;print(site.getsitepackages()[0])")
TI=$SP/torch/include
CUT=""
for h in $(find / -xdev -path "*include/cutlass/epilogue/threadblock/fusion/visitors.hpp" 2>/dev/null); do
  d=${h%/include/cutlass/epilogue/threadblock/fusion/visitors.hpp}
  if [ -f $d/include/cute/tensor.hpp ] && [ -f $d/include/cutlass/gemm/kernel/default_gemm_universal_with_visitor.h ]; then CUT=$d; break; fi
done
[ -z "$CUT" ] && [ -f /lab/cutlass/include/cute/tensor.hpp ] && CUT=/lab/cutlass
echo "nvcc: $(nvcc --version | tail -1)"; echo "torch: $($PY -c "import torch;print(torch.__version__)")"; echo "cutlass: ${CUT:-none}"
ABI=$($PY -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
COMMON="-O3 -std=c++20 -gencode=arch=compute_80,code=sm_80 --expt-relaxed-constexpr -DNDEBUG -Xcompiler -fPIC -shared -isystem $TI -isystem $TI/torch/csrc/api/include -D_GLIBCXX_USE_CXX11_ABI=$ABI -L$SP/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch"
cd /lab
if [ ! -s /lab/exl3_i8_lab.so ] || [ /lab/exl3_i8_lab.cu -nt /lab/exl3_i8_lab.so ]; then
  nvcc $COMMON -I/lab /lab/exl3_i8_lab.cu -o /lab/exl3_i8_lab.so
fi
if [ -n "$CUT" ] && { [ ! -s /lab/cmp_i8mm.so ] || [ /lab/cmp_i8mm.cu -nt /lab/cmp_i8mm.so ]; }; then
  nvcc $COMMON -I$CUT/include -I$CUT/tools/util/include /lab/cmp_i8mm.cu -o /lab/cmp_i8mm.so
fi
sha256sum /lab/*.so'
