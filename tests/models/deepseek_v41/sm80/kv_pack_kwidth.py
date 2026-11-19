# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile-only check (no GPU needed): the sm_80 attention readers compiled
for packed records have the same MMA operand layouts (``kWidth``) and MMA
count as for BF16 rows, i.e. the same K accumulation order. Triton picks a
dot operand's kWidth from the narrowest type it traces the operand back to,
so a decoded uint8 tile would otherwise get kWidth 4 instead of 2.

    python tests/models/deepseek_v41/sm80/kv_pack_kwidth.py
"""

import json
import re

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

import vllm.triton_utils as _tu

# Without a GPU vLLM hands out placeholder Triton modules; compiling needs
# the real ones (installed before the kernels' modules import them).
_tu.triton, _tu.tl, _tu.HAS_TRITON = triton, tl, True

from vllm_cmp170hx.models.deepseek_v41.ops import sm80_index_scores as si
from vllm_cmp170hx.models.deepseek_v41.ops import sm80_sparse_attn as sa


def compile_sm80(jit_fn, types: dict, consts: dict, num_warps: int):
    fn = jit_fn.fn if hasattr(jit_fn, "fn") else jit_fn
    names = jit_fn.arg_names
    signature, constexprs = {}, {}
    for i, name in enumerate(names):
        if name in consts:
            signature[name] = "constexpr"
            constexprs[(i,)] = consts[name]
        else:
            signature[name] = types.get(name, "i64")
    src = ASTSource(fn=jit_fn, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=GPUTarget("cuda", 80, 32),
                          options={"num_warps": num_warps})


def layout(kernel) -> dict:
    ttgir = kernel.asm["ttgir"]
    return {
        "kwidth": sorted(set(re.findall(r"kWidth = (\d+)", ttgir))),
        "mma_sync": kernel.asm["ptx"].count("mma.sync"),
    }


def shared_bytes(kernel) -> int:
    return int(kernel.metadata.shared)


# sm_80 (A100 / CMP 170HX) opt-in shared memory per block.
SM80_SHARED = 166912


def main() -> None:
    out, smem = {}, {}
    for swa, comp in ((False, False), (True, False), (True, True)):
        types = {"q_ptr": "*bf16", "swa_ptr": "*u8" if swa else "*bf16",
                 "swa_idx_ptr": "*i32", "comp_ptr": "*u8" if comp else "*bf16",
                 "comp_bt_ptr": "*i32", "comp_idx_ptr": "*i32", "req_ptr": "*i32",
                 "sink_ptr": "*fp32", "out_ptr": "*bf16", "scale": "fp32",
                 "n_heads": "i32", "n_window": "i32", "n_comp": "i32"}
        consts = dict(SWA_E=64, COMP_E=128, HAS_COMP=True, D=512, BLOCK_H=16, BLOCK_N=32,
                      SWA_PACKED=swa, COMP_PACKED=comp)
        k = compile_sm80(sa._sparse_attn_kernel, types, consts, 8)
        out[f"sparse_attn swa_packed={swa} comp_packed={comp}"] = layout(k)
        smem[f"sparse_attn swa_packed={swa} comp_packed={comp}"] = shared_bytes(k)
    for packed in (False, True):
        types = {"q_ptr": "*bf16", "w_ptr": "*bf16", "nvis_ptr": "*i32", "req_ptr": "*i32",
                 "k_ptr": "*u8" if packed else "*bf16", "bt_ptr": "*i32",
                 "out_ptr": "*fp32"}
        consts = dict(ENTRIES=512, H=32, D=128, HPR=32, BLOCK_N=64, PACKED=packed)
        k = compile_sm80(si._index_scores_kernel, types, consts, 4)
        out[f"index_scores packed={packed}"] = layout(k)
        smem[f"index_scores packed={packed}"] = shared_bytes(k)
    print(json.dumps(out, indent=1))
    print(json.dumps({"shared_bytes": smem}))
    ref_a = out["sparse_attn swa_packed=False comp_packed=False"]
    ref_i = out["index_scores packed=False"]
    ok = all(v == ref_a for k, v in out.items() if k.startswith("sparse")) and all(
        v == ref_i for k, v in out.items() if k.startswith("index")) and all(
        b <= SM80_SHARED for b in smem.values())
    print("SAME_MMA_LAYOUT" if ok else "MMA_LAYOUT_DIFFERS")


if __name__ == "__main__":
    main()
