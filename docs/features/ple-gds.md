# Qwen4Exp FP8 PLE on GPUDirect Storage

Set `VLLM_PLE_GDS_ARTIFACT` to a read-only generation of the compact FP8 PLE
artifact to stage rows directly from NVMe into the CUDA Graph input buffer.
The default disk-offload path uses the checkpoint files and the page-cache
reader. The GDS path uses 16 synchronous cuFile workers per embedding and
supports the tested Qwen4Exp Flash Next FP8 PLE table with TP=1 and PP=2. It
requires a compatible NVIDIA driver, CUDA toolkit, cuFile, NVMe and file system
with direct I/O and PCIe P2PDMA support.

## Build the extension

Run this with the engine's Python and the CUDA toolkit inside its build
environment. `CUFILE_ROOT` contains `include/cufile.h` and `lib/libcufile.so.0`.
Use `--cufile-lib` if your library is in a different directory.

```bash
python csrc/ple_gds/build_standalone.py --cufile "$CUFILE_ROOT"
python -m pip install --no-deps .
```

The optional build checks the SHA-256 of the pinned Apache-2.0 native reader
from Qwen-Flash-SM80-170HX. Use `--source /path/to/ple_gds_persistent.cpp` for
an offline build with the same source revision. The plugin wheel packages the
local extension when it is built before installation.

## Prepare and verify the table

`CHECKPOINT` is the single safetensors file with the original FP8 PLE shards
and BF16 scale. Use the compact converter from Qwen-Flash-SM80-170HX at
`4d229f1438486fa759727f128fc28f3609423dbb`.

```bash
PYTHONPATH=. python tools/ple_gds/artifact.py "$CHECKPOINT" > ple-source.json
python "$QWEN_FLASH_SM80/scripts/convert-ple-to-gds-layout.py" \
  --source-spec ple-source.json --output "$PLE_GDS_ROOT" --compact
ARTIFACT="$PLE_GDS_ROOT/generations/$(cat "$PLE_GDS_ROOT/CURRENT")"
PYTHONPATH=. python tools/ple_gds/artifact.py "$CHECKPOINT" --verify "$ARTIFACT"
```

The verifier reads the full artifact and checkpoint rows with `O_DIRECT`,
compares the row bytes in shard-ID order and checks the BF16 scale. The current
backend supports a single FP8 table identity. At startup it checks the
read-only layout, size, manifest digest and scale. Full payload verification
is performed offline, so keep the verified generation immutable and mount it
read-only in the serving container.

## Serve and smoke-test

Use the engine branch `vllm/a0d7356b98`, enable disk offload with
`--engram-config '{"disk_offload":true}'` and point the worker environment at
the immutable generation. Pass the original FP8 PLE checkpoint through the
existing checkpoint loader so it supplies the original shard layout and scale.
For a container, expose the target GPUs and make the host udev information
available to cuFile (typically a read-only `/run/udev` bind mount).

```bash
export VLLM_PLE_GDS_ARTIFACT="$ARTIFACT"
export CUFILE_ENV_PATH_JSON="$PWD/tools/ple_gds/cufile.json"
export CUFILE_USE_PCIP2PDMA=1
PYTHONPATH=. python tools/ple_gds/smoke.py "$ARTIFACT" "$CHECKPOINT"
```

The smoke compares individual staged rows with the original checkpoint bytes.
It covers cross-page rows, repeated and shuffled IDs, a request spanning three
reader chunks, unaligned output storage and repeated CUDA Graph replay using
the same output buffer. Use the cuFile logs to verify P2PDMA on your storage
and GPU topology. Unset `VLLM_PLE_GDS_ARTIFACT` and restart to use the standard
reader.
