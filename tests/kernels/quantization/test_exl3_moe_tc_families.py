# SPDX-License-Identifier: Apache-2.0
"""Host-side gates of the tensor-core MoE families (ops.MOE_TC_FAMILIES): GLM keeps its exact former gate, and
DeepSeek-V4.1's Q4N-X routed experts (target top-6 of 384, DSpark drafter top-3 of 128) are admitted only at keys
132 / 133. CPU only (no extension call); also runnable as a plain script."""

import torch

from vllm_cmp170hx.kernels.exl3 import ops as exl3_ops

SM80 = (8, 0)


def _meta(hidden, intermediate, bits, codebook, experts):
    return [[[hidden, intermediate, bits, codebook], [hidden, intermediate, bits, codebook],
             [intermediate, hidden, bits, codebook]]] * experts


def _both(*args):
    return (exl3_ops.moe_prefill_closed_reason(*args), exl3_ops.moe_decode_tc_closed_reason(*args))


def test_glm_gate_unchanged():
    q4n = exl3_ops.MOE_GLM_Q4N_EXPERT_BITS  # lab (4, 5, 6, 7); release (5, 6)
    formats = [(5, 2, 5), (7, 2, 7)] + [(b, cb, b + 64 * (cb - 2)) for cb in (3, 4) for b in q4n]
    for bits, codebook, key in formats:
        assert _both(SM80, torch.bfloat16, 4096, 2048, 8, 288, 288, _meta(4096, 2048, bits, codebook, 288),
                     key) == (None, None), key
    reasons = _both(SM80, torch.bfloat16, 4096, 2048, 8, 288, 288, _meta(4096, 2048, 6, 2, 288), 6)
    assert all(r.startswith("GLM ") and "5-bit or 3.5-bit" in r for r in reasons), reasons
    assert all(r is not None for r in _both(SM80, torch.bfloat16, 4096, 2048, 6, 288, 288,
                                            _meta(4096, 2048, 5, 4, 288), 133))


def test_dsv41_gate():
    for top_k, experts in ((6, 384), (3, 128)):
        for bits in (4, 5):
            assert _both(SM80, torch.bfloat16, 5120, 2304, top_k, experts, experts,
                         _meta(5120, 2304, bits, 4, experts), 128 + bits) == (None, None), (top_k, bits)
    closed = [
        (SM80, torch.bfloat16, 5120, 2304, 6, 384, 384, _meta(5120, 2304, 6, 4, 384), 134),  # 3.0 bpw: no family
        (SM80, torch.bfloat16, 5120, 2304, 6, 384, 384, _meta(5120, 2304, 5, 2, 384), 5),  # mul1: no family
        (SM80, torch.bfloat16, 5120, 2304, 8, 384, 384, _meta(5120, 2304, 5, 4, 384), 133),  # top-8
        (SM80, torch.bfloat16, 5120, 2304, 6, 192, 384, _meta(5120, 2304, 5, 4, 192), 133),  # EP split
        (SM80, torch.bfloat16, 5120, 2304, 3, 384, 384, _meta(5120, 2304, 5, 4, 384), 133),  # top-3 of 384
        (SM80, torch.bfloat16, 5120, 2304, 6, 384, 384, _meta(5120, 2304, 4, 4, 384), 133),  # meta / key mismatch
        ((8, 6), torch.bfloat16, 5120, 2304, 6, 384, 384, _meta(5120, 2304, 5, 4, 384), 133),
        (SM80, torch.float16, 5120, 2304, 6, 384, 384, _meta(5120, 2304, 5, 4, 384), 133),
    ]
    for args in closed:
        assert all(r is not None for r in _both(*args)), args[2:7]


def test_unserved_shapes_stay_closed():
    reason = exl3_ops.moe_decode_tc_closed_reason(SM80, torch.bfloat16, 5120, 2048, 6, 384, 384,
                                                  _meta(5120, 2048, 5, 4, 384), 133)
    assert reason is not None and "(5120, 2048)" in reason
    assert exl3_ops.moe_tc_family(4096, 2048).name == "GLM"
    assert exl3_ops.moe_tc_family(5120, 2304).name == "DeepSeek-V4.1"
    assert exl3_ops.moe_tc_family(2560, 640) is None


if __name__ == "__main__":
    test_glm_gate_unchanged()
    test_dsv41_gate()
    test_unserved_shapes_stay_closed()
    print("moe tc family gates: ok")
