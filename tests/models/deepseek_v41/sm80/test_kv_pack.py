# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact packed KV records (``VLLM_DEEPSEEK_V41_KV_PACK``) against BF16 rows.

1. Records: for BF16 inputs spread over many magnitudes (zeros, -0.0, tiny
   and huge blocks, values on rounding ties and at the clamps) a record
   decodes to exactly the BF16 value the qdq of the BF16 path stores, for
   MXFP8 (window), NVFP4 (compressed) and MXFP4 (index keys).
2. Store: the Triton store writes the torch definition's bytes; slot -1
   writes nothing (CUDA).
3. Layers: the three layer stacks of ``test_attn_layers`` (chunked prefill,
   decode steps of 1-3 tokens, CUDA-graph padding rows, recycled block ids)
   run once with BF16 caches and once with packed caches in padded pages:
   every ``wo_a`` input and every top-k selection is bit for bit equal and
   every decoded cache row equals the BF16 row -- on CPU (torch fallbacks)
   and on CUDA (Triton kernels).
4. Kernels: index scores and sparse attention over packed caches equal the
   same kernels over BF16 caches holding the decoded values (random caches,
   block ids past 2**31 bytes).
5. Compiled code: the readers' MMA operand layouts (``kWidth``) and MMA
   counts are the BF16 variant's (``kv_pack_kwidth.py``, compile only), so
   the K accumulation order is the same by construction.
6. Planner: the packed specs unify to one 36,864 B page with the block
   sizes the KV-size audit expects (window 64, compressed 256/128, index
   keys 1024/512 tokens), the compressor ring padded.

    pytest -q tests/models/deepseek_v41/sm80/test_kv_pack.py
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vllm_cmp170hx.models.deepseek_v41.ops import sm80_attn_core as core
from vllm_cmp170hx.models.deepseek_v41.ops import sm80_kv_pack as kvpack
from vllm_cmp170hx.models.deepseek_v41.ops.sm80_act_quant import (
    act_quant_qdq_fp8,
    fp4_qdq,
)

from .dsv41_ref import LayerSpec, RefArgs, make_weights
from .test_attn_layers import make_inputs, make_schedule, run_sm80

BF16 = torch.bfloat16
CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
RECORDS = [(kvpack.MXFP8, 512), (kvpack.NVFP4, 512), (kvpack.MXFP4, 128)]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _reference_qdq(x: torch.Tensor, rec: kvpack.KVRecord) -> torch.Tensor:
    if rec.fp4:
        return fp4_qdq(x, rec.block, rec.scale_kind)
    return act_quant_qdq_fp8(x, rec.block)


def hard_rows(n: int, dim: int, seed: int) -> torch.Tensor:
    """BF16 rows whose 16-value blocks span 2**-60 .. 2**60, with zeros,
    -0.0, constant blocks, ties of the E4M3/E2M1 grids and clamp values."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    mag = torch.randint(-60, 61, (n, dim // 16, 1), generator=g).float()
    x = (x.view(n, dim // 16, 16) * torch.exp2(mag)).view(n, dim)
    x[0] = 0.0
    x[1] = -0.0
    x[2, : dim // 2] = 1.0
    x[2, dim // 2 :] = -448.0
    # Grid ties relative to the block max: 448 * (k + 0.5) / 2**j etc.
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, -0.25, -5.0,
                         4.5, 9.0, 13.0, 17.0, 240.0, 448.0])
    x[3] = ties.repeat(dim // 16)
    x[4] = -ties.repeat(dim // 16) * 2.0**-20
    x[5, :] = 1e-38
    x[6, ::2] = -0.0
    return x.to(BF16)


@pytest.mark.parametrize("rec,dim", RECORDS, ids=lambda v: getattr(v, "name", v))
def test_record_decodes_to_qdq_bitwise(rec, dim) -> None:
    x = hard_rows(64, dim, 1)
    packed = kvpack.pack(x, rec)
    assert packed.dtype == torch.uint8 and packed.shape == (64, rec.stride(dim))
    assert torch.equal(_bits(kvpack.unpack(packed, rec, dim).to(BF16)),
                       _bits(_reference_qdq(x, rec)))


def test_record_sizes() -> None:
    assert kvpack.MXFP8.record_bytes(512) == kvpack.MXFP8.stride(512) == 528
    assert kvpack.NVFP4.record_bytes(512) == kvpack.NVFP4.stride(512) == 288
    assert kvpack.MXFP4.record_bytes(128) == 68 and kvpack.MXFP4.stride(128) == 72
    page = kvpack.PACKED_PAGE_BYTES
    assert page == 128 * 288 == 512 * 72 >= 64 * 528


def test_codes_follow_the_upstream_layout() -> None:
    """NVFP4: even element in the low nibble, then the e4m3 scales; MXFP8:
    the 512 codes, then UE8M0 bytes (exponent + 127)."""
    x = torch.zeros(1, 512)
    x[0, 0], x[0, 1] = 6.0, -0.5  # block 0 of 16: scale e4m3(1.0) = 0x38
    rec = kvpack.pack(x.to(BF16), kvpack.NVFP4)[0]
    assert rec[0].item() == (0x7 | (0x9 << 4))  # 6.0 -> 0b0111, -0.5 -> 0b1001
    assert rec[256].item() == 0x38
    y = torch.zeros(1, 512)
    y[0, 0] = 448.0  # scale 2**0 -> UE8M0 127, code 0x7e
    rec = kvpack.pack(y.to(BF16), kvpack.MXFP8)[0]
    assert rec[0].item() == 0x7E and rec[512].item() == 127


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("rec,dim", RECORDS, ids=lambda v: getattr(v, "name", v))
def test_triton_decode_equals_the_definition(rec, dim) -> None:
    """``load_records_tl`` (the readers' decode) gives the torch decode's
    BF16 values for every row, masked rows +0.0."""
    from vllm.triton_utils import tl, triton

    @triton.jit
    def decode(ptr, out, stride, D: tl.constexpr, FP4: tl.constexpr, QB: tl.constexpr,
               E4: tl.constexpr, CB: tl.constexpr):
        r = tl.program_id(0).to(tl.int64) * 32 + tl.arange(0, 32)
        v = kvpack.load_records_tl(ptr, r * stride, r % 7 != 3, D, FP4, QB, E4, CB)
        tl.store(out + r[:, None] * D + tl.arange(0, D)[None, :], v)

    x = hard_rows(2048, dim, 7)
    recs = kvpack.pack(x, rec)
    want = kvpack.unpack(recs, rec, dim).to(BF16)
    want[torch.arange(2048) % 7 == 3] = 0.0
    out = torch.empty(2048, dim, dtype=BF16, device="cuda")
    decode[(64,)](recs.cuda(), out, recs.shape[1], dim, rec.fp4, rec.block,
                  rec.e4m3_scale, rec.code_bytes(dim))
    assert torch.equal(_bits(out.cpu()), _bits(want))


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("rec,dim", RECORDS, ids=lambda v: getattr(v, "name", v))
def test_triton_store_writes_the_definition(rec, dim) -> None:
    x = hard_rows(70, dim, 2).cuda()
    entries = 16
    cache = torch.full((12, entries, rec.stride(dim) + 8), 0xAB, dtype=torch.uint8,
                       device="cuda")[:, :, : rec.stride(dim)]
    slots = torch.randperm(12 * entries, device="cuda")[:70]
    slots[5] = -1
    kvpack.store(cache, slots, x, rec)
    want = kvpack.pack(x.cpu(), rec)
    got = cache.cpu()
    rb = rec.record_bytes(dim)
    for t in range(70):
        s = int(slots[t])
        if s < 0:
            continue
        assert torch.equal(got[s // entries, s % entries, :rb], want[t, :rb]), t
    # Nothing else was written.
    written = {int(s) for s in slots.tolist() if s >= 0}
    for s in range(12 * entries):
        if s not in written:
            assert bool((got[s // entries, s % entries] == 0xAB).all()), s


LAYER_STACKS = [
    ("ratio0", [LayerSpec(0)]),
    ("ratio2_source", [LayerSpec(2, kv_source=True, index_source=True)]),
    (
        "ratio1_candidate_stack",
        [
            LayerSpec(1, kv_source=True, index_source=True, candidate_source=True),
            LayerSpec(1),
            LayerSpec(1, index_source=True, uses_candidates=True),
        ],
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,specs", LAYER_STACKS, ids=[n for n, _ in LAYER_STACKS])
def test_packed_layers_equal_bf16_layers(name, specs, device) -> None:
    torch.manual_seed(0)
    args = RefArgs()
    weights = [make_weights(args, s, 100 + i) for i, s in enumerate(specs)]
    xs = make_inputs(args, len(specs), 5)
    schedule = make_schedule(11)
    base, outs, topks = run_sm80(args, specs, weights, xs, schedule, device=device)
    packed, pouts, ptopks = run_sm80(
        args, specs, weights, xs, schedule, device=device, packed=True, page_pad=160
    )
    assert packed.swa_caches[0].dtype == torch.uint8
    assert outs.keys() == pouts.keys() and topks == ptopks
    for key, want in outs.items():
        assert torch.equal(_bits(pouts[key]), _bits(want)), f"wo_a input {key}"
    for rid in base.reqs:
        total = base.reqs[rid].length
        for li, spec in enumerate(specs):
            for p in range(max(0, total - args.window_size), total):
                assert torch.equal(_bits(packed.window_row(li, rid, p)),
                                   _bits(base.window_row(li, rid, p))), (li, rid, p)
            if spec.kv_source:
                for j in range(total // spec.ratio):
                    assert torch.equal(_bits(packed.comp_row(rid, j)),
                                       _bits(base.comp_row(rid, j))), (rid, j)
                    if spec.index_source:
                        assert torch.equal(
                            _bits(packed.comp_row(rid, j, idxk=True)),
                            _bits(base.comp_row(rid, j, idxk=True)),
                        ), (rid, j)


def _random_packed(blocks: int, entries: int, rec, dim: int, g, device, pad: int = 0):
    """A packed cache of random records and the BF16 cache of their values."""
    x = torch.randn(blocks * entries, dim, generator=g).to(BF16)
    recs = kvpack.pack(x, rec).view(blocks, entries, -1)
    stride = rec.stride(dim)
    flat = torch.zeros(blocks * (entries * stride + pad), dtype=torch.uint8)
    cache = flat.as_strided((blocks, entries, stride), (entries * stride + pad, stride, 1))
    cache.copy_(recs)
    values = kvpack.unpack(recs, rec, dim).to(BF16)
    return cache.to(device), values.to(device)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_index_scores_packed_equal_bf16() -> None:
    from vllm_cmp170hx.models.deepseek_v41.ops.sm80_index_scores import (
        index_scores_triton,
    )

    g = torch.Generator().manual_seed(3)
    n, blocks, entries, h, d = 37, 60, 64, 32, 128
    packed, values = _random_packed(blocks, entries, kvpack.MXFP4, d, g, "cuda", pad=40)
    q = kvpack.qdq(torch.randn(n, h, d, generator=g).to(BF16), kvpack.MXFP4).cuda()
    w = torch.randn(n, h, generator=g).to(BF16).cuda()
    n_vis = torch.randint(0, 400, (n,), generator=g).cuda()
    req = torch.randint(0, 3, (n,), generator=g).cuda()
    bt = torch.stack([torch.randperm(blocks, generator=g)[:8] for _ in range(3)])
    bt = bt.to(torch.int32).cuda()
    for hpr in (8, 32):
        a = index_scores_triton(q, w, n_vis, req, values, bt, 450, hpr)
        b = index_scores_triton(q, w, n_vis, req, packed, bt, 450, hpr)
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
        c = core.index_scores_torch(q.cpu(), w.cpu(), n_vis.cpu(), req.cpu(),
                                    packed.cpu(), bt.cpu(), 450, hpr)
        assert torch.equal(c, b.cpu())


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("with_comp", [False, True])
def test_sparse_attention_packed_equal_bf16(with_comp: bool) -> None:
    from vllm_cmp170hx.models.deepseek_v41.ops.sm80_sparse_attn import (
        sparse_attention_triton,
    )

    g = torch.Generator().manual_seed(4)
    n, h, d, window = 29, 64, 512, 128
    swa_p, swa_v = _random_packed(40, 64, kvpack.MXFP8, d, g, "cuda", pad=3072)
    comp_p, comp_v = _random_packed(30, 128, kvpack.NVFP4, d, g, "cuda")
    q = torch.randn(n, h, d, generator=g).to(BF16).cuda()
    idx = torch.randint(-1, 40 * 64, (n, window), generator=g).to(torch.int32).cuda()
    req = torch.randint(0, 2, (n,), generator=g).cuda()
    sink = torch.randn(h, generator=g).cuda()
    if with_comp:
        bt = torch.stack([torch.randperm(30, generator=g)[:6] for _ in range(2)])
        bt = bt.to(torch.int32).cuda()
        local = torch.randint(-1, 6 * 128, (n, 96), generator=g).to(torch.int32).cuda()
    else:
        bt = local = None
    a = sparse_attention_triton(q, swa_v, idx, comp_v if with_comp else None, bt,
                                local, req, sink, d**-0.5)
    b = sparse_attention_triton(q, swa_p, idx, comp_p if with_comp else None, bt,
                                local, req, sink, d**-0.5)
    assert torch.equal(_bits(a), _bits(b))


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_packed_kernels_int64_offsets() -> None:
    """Block ids whose byte offsets pass 2**31 (as a large KV pool does)."""
    from vllm_cmp170hx.models.deepseek_v41.ops.sm80_index_scores import (
        index_scores_triton,
    )

    g = torch.Generator().manual_seed(5)
    entries, d, page = 512, 128, kvpack.PACKED_PAGE_BYTES
    blocks = (2**31) // page + 4
    if torch.cuda.mem_get_info()[0] < blocks * page + (1 << 30):
        pytest.skip("not enough free device memory for a 2 GiB cache")
    flat = torch.zeros(blocks * page, dtype=torch.uint8, device="cuda")
    cache = flat.as_strided((blocks, entries, 72), (page, 72, 1))
    hi = blocks - 2
    x = torch.randn(entries, d, generator=g).to(BF16)
    cache[hi] = kvpack.pack(x, kvpack.MXFP4).cuda()
    small = torch.zeros(2, entries, 72, dtype=torch.uint8, device="cuda")
    small[1] = cache[hi]
    q = kvpack.qdq(torch.randn(4, 32, d, generator=g).to(BF16), kvpack.MXFP4).cuda()
    w = torch.randn(4, 32, generator=g).to(BF16).cuda()
    n_vis = torch.full((4,), entries, device="cuda")
    req = torch.zeros(4, dtype=torch.int64, device="cuda")
    a = index_scores_triton(q, w, n_vis, req, cache,
                            torch.tensor([[hi]], dtype=torch.int32, device="cuda"),
                            entries, 8)
    b = index_scores_triton(q, w, n_vis, req, small,
                            torch.tensor([[1]], dtype=torch.int32, device="cuda"),
                            entries, 8)
    assert torch.equal(a, b)
    del flat, cache


def test_packed_specs_unify_to_one_page() -> None:
    from vllm.v1.core.kv_cache_planning import unify_kv_cache_spec_page_size
    from vllm.v1.kv_cache_interface import CircularBufferSpec

    from vllm_cmp170hx.models.deepseek_v41.sm80_attention import (
        packed_state_spec,
        packed_swa_spec,
    )

    base = 32  # cache_config.block_size of the lab boots (tokens)
    specs = {
        "swa": packed_swa_spec(64, 512, 128, "auto", False),
        "c2": packed_state_spec(base, 512, kvpack.NVFP4, 2),
        "i2": packed_state_spec(base, 128, kvpack.MXFP4, 2),
        "c1": packed_state_spec(base, 512, kvpack.NVFP4, 1),
        "i1": packed_state_spec(base, 128, kvpack.MXFP4, 1),
        "ring": CircularBufferSpec(block_size=8, num_kv_heads=1, head_size=1024,
                                   head_size_v=0, dtype=torch.float32),
    }
    unified = unify_kv_cache_spec_page_size(specs)
    assert {s.page_size_bytes for s in unified.values()} == {kvpack.PACKED_PAGE_BYTES}
    blocks = {k: s.block_size for k, s in unified.items()}
    assert blocks == {"swa": 64, "c2": 256, "i2": 1024, "c1": 128, "i1": 512, "ring": 8}
    # A ring page twice as large (more speculative rows): the window block
    # doubles and the ring is the padded page.
    specs["swa"] = packed_swa_spec(128, 512, 128, "auto", False)
    specs["ring"] = CircularBufferSpec(block_size=16, num_kv_heads=1, head_size=1024,
                                       head_size_v=0, dtype=torch.float32)
    unified = unify_kv_cache_spec_page_size(specs)
    assert {s.page_size_bytes for s in unified.values()} == {2 * kvpack.PACKED_PAGE_BYTES}
    assert unified["ring"].block_size == 16 and unified["swa"].block_size == 128


def test_packed_readers_compile_to_the_bf16_mma_layout() -> None:
    script = Path(__file__).with_name("kv_pack_kwidth.py")
    root = Path(__file__).resolve().parents[4]
    proc = subprocess.run([sys.executable, str(script)], cwd=root, capture_output=True,
                          text=True, timeout=600)
    assert "SAME_MMA_LAYOUT" in proc.stdout, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    from tests.models.deepseek_v41.sm80 import exl3_fixture as fixture

    directory = tmp_path_factory.mktemp("tiny_kvpack")
    fixture.write_checkpoint(directory, fixture.tiny_config())
    return directory


def _construction_specs(monkeypatch, checkpoint, pack: bool) -> dict:
    from tests.models.deepseek_v41.sm80 import construct
    from vllm_cmp170hx import plugin

    plugin._register_exl3()
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_DEEPSEEK_V41_KV_PACK", "1" if pack else "0")
    construct.set_groups(monkeypatch, 1, 0)
    construct.stub_engram_tables(monkeypatch, 1)
    target_config, draft_config = construct.vllm_configs(str(checkpoint), 1)
    construct.build_model(target_config)
    construct.build_model(draft_config)
    specs = {}
    for config in (target_config, draft_config):
        for name, layer in config.compilation_config.static_forward_context.items():
            get = getattr(layer, "get_kv_cache_spec", None)
            spec = get(config) if get is not None else None
            if spec is not None:
                specs[name] = spec
    return specs


def test_constructed_model_charges_packed_pages(monkeypatch, meta_platform, tiny_checkpoint):
    """The tiny target + DSpark drafter built with VLLM_DEEPSEEK_V41_KV_PACK:
    every window cache is MXFP8 in 64-row blocks, every compressed cache
    NVFP4, every index-key cache MXFP4 (uint8), and the planner unifies all
    of them (rings included) to one 36,864 B page; without the switch the
    specs are the BF16 ones (32 KiB pages)."""
    from vllm.v1.core.kv_cache_planning import unify_kv_cache_spec_page_size

    packed = _construction_specs(monkeypatch, tiny_checkpoint, True)
    kinds = {"swa": 0, "comp": 0, "idxk": 0, "ring": 0}
    for name, spec in packed.items():
        if name.endswith(".swa_cache"):
            kinds["swa"] += 1
            assert spec.dtype == torch.uint8 and spec.block_size == 64
            assert spec.state_content_bytes == 528, name
        elif name.endswith(".indexer.k_cache"):
            kinds["idxk"] += 1
            assert spec.dtype == torch.uint8 and spec.state_content_bytes == 72, name
        elif name.endswith(".compressor.state_cache"):
            kinds["ring"] += 1
        else:
            kinds["comp"] += 1
            assert spec.dtype == torch.uint8 and spec.state_content_bytes == 288, name
    assert all(kinds.values()), kinds
    unified = unify_kv_cache_spec_page_size(packed)
    assert {s.page_size_bytes for s in unified.values()} == {kvpack.PACKED_PAGE_BYTES}
    plain = _construction_specs(monkeypatch, tiny_checkpoint, False)
    assert plain.keys() == packed.keys()
    assert all(s.dtype != torch.uint8 for s in plain.values())


from tests.models.test_native_checkpoint_config_meta import (  # noqa: E402
    meta_platform,  # noqa: F401
)
