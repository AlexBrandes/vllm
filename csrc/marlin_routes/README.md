# K3 Marlin routes

`marlin_host.so`: a host-only dispatcher for vLLM's Marlin GEMM that reuses the
wheel's installed device kernels (`_C_stable_libtorch.abi3.so`) and picks the
tile, thread count and CTAs per SM for int8-activation (W4A8) calls from a
table measured on the CMP 170HX (`routes.json`). Every other call keeps the
stock selector.

- `generate_host.py` writes `marlin_host.cpp` from upstream's `marlin.cu` at the
  engine's wheel commit (`marlin.upstream.cu`, Apache-2.0, checked by hash) and
  `routes.json`.
- `build.sh <out dir>` generates and compiles it inside the engine image (g++,
  no GPU) and writes `source-manifest.json` with the sha256 of every input.

The engine loads it when `VLLM_CMP_MARLIN_ROUTES` names the library and
`VLLM_CMP_MARLIN_ROUTES_SHA256` matches its hash (fails closed otherwise).
