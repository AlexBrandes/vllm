# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's KV caches as exact packed records on sm_80.

The reference rounds every cached KV value through FP8 or FP4 before it is
stored (``sm80_attn_core``: window rows FP8 (32, UE8M0), compressed rows FP4
(16, E4M3 scale), index keys FP4 (32, E8M0)), so a cache can hold the codes
and scales instead of BF16 rows of their products. Decoding a record gives
the same fp32 numbers the BF16 row holds (``code * scale`` is exact in fp32
and in BF16), so every reader that turns the decoded values into its BF16
MMA operand computes bit for bit what it computes from the BF16 rows.

Records (``KVRecord``), each state's codes followed by its own scales as in
the fork's V4.1 FlashMLA records (``deepseek_v41/common/ops/cache_utils.py``):

* ``MXFP8`` (sliding window): ``dim`` e4m3 codes, then ``dim / 32`` UE8M0
  bytes (``exponent + 127``). 528 B for 512 dims, upstream's V4.1 record.
* ``NVFP4`` (compressed KV): ``dim / 2`` bytes of e2m1 pairs, the even
  element in the low nibble, then ``dim / 16`` e4m3 scale bytes. 288 B for
  512 dims, upstream's ``nvfp4_ds_mla`` compressed record.
* ``MXFP4`` (index keys): ``dim / 2`` bytes of e2m1 pairs, then ``dim / 32``
  UE8M0 bytes: 68 B for 128 dims (upstream's MXFP4 indexer encoding),
  stored at a 72 B stride so a 512-state page is 36,864 B like the others.

A cache of records is a uint8 tensor ``[blocks, states_per_block, stride]``
with any block stride (the KV planner may pad a page past its records).

Encoding takes the pre-quantization BF16 values (not the qdq'd ones) and
quantizes them with the same scale and grid arithmetic as
``sm80_act_quant``'s qdq (its ``*_quant`` factors), so the codes and scales
are the reference's.

Torch functions are the definition (any device); on CUDA the store is one
Triton launch (``pack_store``), the readers decode in their own kernels with
:func:`load_records_tl`.
"""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm_cmp170hx.models.deepseek_v41.ops.sm80_act_quant import (
    fp4_quant,
    fp4_quant_tile,
    fp8_quant,
    fp8_quant_tile,
)

# One page size for every packed group: 64 window rows (64 x 528 = 33,792 B,
# padded), 128 compressed states (128 x 288) and 512 index keys (512 x 72).
PACKED_PAGE_BYTES = 36864
# Block size (tokens) of a packed sliding-window cache.
PACKED_SWA_BLOCK = 64


@dataclass(frozen=True)
class KVRecord:
    name: str
    fp4: bool
    block: int  # values per scale
    e4m3_scale: bool  # scale byte: e4m3 value (NVFP4) or UE8M0 exponent (MX)
    align: int = 8  # record stride granule

    def code_bytes(self, dim: int) -> int:
        return dim // 2 if self.fp4 else dim

    def scale_bytes(self, dim: int) -> int:
        return dim // self.block

    def record_bytes(self, dim: int) -> int:
        return self.code_bytes(dim) + self.scale_bytes(dim)

    def stride(self, dim: int) -> int:
        """Bytes per stored state (the record, padded to ``align``)."""
        return -(-self.record_bytes(dim) // self.align) * self.align

    @property
    def scale_kind(self) -> str:
        return "e4m3" if self.e4m3_scale else "e8m0"


MXFP8 = KVRecord("mxfp8", fp4=False, block=32, e4m3_scale=False)
NVFP4 = KVRecord("nvfp4", fp4=True, block=16, e4m3_scale=True)
MXFP4 = KVRecord("mxfp4", fp4=True, block=32, e4m3_scale=False)


def is_packed(cache: torch.Tensor | None) -> bool:
    return cache is not None and cache.dtype == torch.uint8


def check_cache(cache: torch.Tensor, rec: KVRecord, dim: int) -> None:
    """Readers load records as 32-bit words: strides and base 4-byte aligned."""
    if (
        cache.dim() != 3
        or cache.shape[2] < rec.record_bytes(dim)
        or cache.stride(2) != 1
        or cache.stride(1) % 4
        or cache.stride(0) % 4
        or cache.data_ptr() % 4
    ):
        raise ValueError(
            f"{rec.name} cache of {dim} dims needs [blocks, states, >= "
            f"{rec.record_bytes(dim)}] contiguous records at 4-byte aligned "
            f"strides, got {tuple(cache.shape)} strides {cache.stride()}"
        )


# ---------------------------------------------------------------------------
# torch definition
# ---------------------------------------------------------------------------


def _sign_bit(v: torch.Tensor) -> torch.Tensor:
    return (v.view(torch.int32) >> 31) & 1


def encode_e4m3(q: torch.Tensor) -> torch.Tensor:
    """fp32 values on the E4M3 grid (|q| <= 448) -> int32 codes, the sign of
    -0.0 kept."""
    a = q.abs()
    ab = a.view(torch.int32)
    normal = (((ab >> 23) - 120) << 3) | ((ab >> 20) & 7)
    sub = (a * 512.0).to(torch.int32)
    return torch.where(a >= 2.0**-6, normal, sub) | (_sign_bit(q) << 7)


def encode_e2m1(q: torch.Tensor) -> torch.Tensor:
    """fp32 values on the E2M1 grid -> int32 codes (0..15)."""
    a = q.abs()
    ab = a.view(torch.int32)
    normal = (((ab >> 23) - 126) << 1) | ((ab >> 22) & 1)
    sub = (a * 2.0).to(torch.int32)
    return torch.where(a >= 1.0, normal, sub) | (_sign_bit(q) << 3)


def _with_sign(v: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    return (v.view(torch.int32) | (sign << 31)).view(torch.float32)


def decode_e4m3(c: torch.Tensor) -> torch.Tensor:
    c = c.to(torch.int32)
    e, m = (c >> 3) & 15, c & 7
    normal = (((e + 120) << 23) | (m << 20)).view(torch.float32)
    sub = m.to(torch.float32) * 2.0**-9
    return _with_sign(torch.where(e > 0, normal, sub), (c >> 7) & 1)


def decode_e2m1(c: torch.Tensor) -> torch.Tensor:
    c = c.to(torch.int32)
    e, m = (c >> 1) & 3, c & 1
    normal = (((e + 126) << 23) | (m << 22)).view(torch.float32)
    sub = m.to(torch.float32) * 0.5
    return _with_sign(torch.where(e > 0, normal, sub), (c >> 3) & 1)


def _scale_byte(s: torch.Tensor, rec: KVRecord) -> torch.Tensor:
    if rec.e4m3_scale:
        return encode_e4m3(s)
    return (s.view(torch.int32) >> 23) & 0xFF


def _scale_value(b: torch.Tensor, rec: KVRecord) -> torch.Tensor:
    if rec.e4m3_scale:
        return decode_e4m3(b)
    return (b.to(torch.int32) << 23).view(torch.float32)


def pack(x: torch.Tensor, rec: KVRecord) -> torch.Tensor:
    """``[..., dim]`` pre-quantization values -> ``[..., stride]`` uint8
    records (padding bytes zero)."""
    dim = x.shape[-1]
    if rec.fp4:
        q, s = fp4_quant(x, rec.block, rec.scale_kind)
        codes = encode_e2m1(q).flatten(-2)
        codes = codes[..., 0::2] | (codes[..., 1::2] << 4)
    else:
        q, s = fp8_quant(x, rec.block)
        codes = encode_e4m3(q).flatten(-2)
    scales = _scale_byte(s.squeeze(-1).contiguous(), rec)
    pad = rec.stride(dim) - rec.record_bytes(dim)
    parts = [codes, scales]
    if pad:
        parts.append(torch.zeros_like(scales[..., :1]).expand(*scales.shape[:-1], pad))
    return torch.cat(parts, dim=-1).to(torch.uint8)


def unpack(records: torch.Tensor, rec: KVRecord, dim: int) -> torch.Tensor:
    """``[..., >= record_bytes]`` uint8 records -> ``[..., dim]`` fp32 values
    (exact in BF16)."""
    b = records.to(torch.int32)
    nc = rec.code_bytes(dim)
    if rec.fp4:
        codes = torch.stack((b[..., :nc] & 15, b[..., :nc] >> 4), dim=-1).flatten(-2)
        v = decode_e2m1(codes)
    else:
        v = decode_e4m3(b[..., :nc])
    s = _scale_value(b[..., nc : nc + rec.scale_bytes(dim)], rec)
    return (v.unflatten(-1, (-1, rec.block)) * s.unsqueeze(-1)).flatten(-2)


def qdq(x: torch.Tensor, rec: KVRecord) -> torch.Tensor:
    """The BF16 values a record of ``x`` decodes to (``unpack(pack(x))``)."""
    return unpack(pack(x, rec), rec, x.shape[-1]).to(torch.bfloat16)


def gather(cache: torch.Tensor, blk: torch.Tensor, off: torch.Tensor, rec: KVRecord,
           dim: int) -> torch.Tensor:
    """BF16 rows ``cache[blk, off]`` of a packed cache, decoded."""
    return unpack(cache[blk, off], rec, dim).to(torch.bfloat16)


def store_torch(cache: torch.Tensor, slots: torch.Tensor, x: torch.Tensor,
                rec: KVRecord) -> None:
    """:func:`store` by tensor indexing (rows with ``slot < 0`` write
    nothing; reads the valid count on the host)."""
    keep = slots >= 0
    s = slots[keep].to(torch.int64)
    e = cache.shape[1]
    rb = rec.record_bytes(x.shape[-1])
    cache[s // e, s % e, :rb] = pack(x[: slots.shape[0]][keep], rec)[..., :rb]


# ---------------------------------------------------------------------------
# Triton
# ---------------------------------------------------------------------------


@triton.jit
def _sign_bit_tl(v):
    return (v.to(tl.int32, bitcast=True) >> 31) & 1


@triton.jit
def encode_e4m3_tl(q):
    a = tl.abs(q)
    ab = a.to(tl.int32, bitcast=True)
    normal = (((ab >> 23) - 120) << 3) | ((ab >> 20) & 7)
    sub = (a * 512.0).to(tl.int32)
    return tl.where(a >= 0.015625, normal, sub) | (_sign_bit_tl(q) << 7)


@triton.jit
def encode_e2m1_tl(q):
    a = tl.abs(q)
    ab = a.to(tl.int32, bitcast=True)
    normal = (((ab >> 23) - 126) << 1) | ((ab >> 22) & 1)
    sub = (a * 2.0).to(tl.int32)
    return tl.where(a >= 1.0, normal, sub) | (_sign_bit_tl(q) << 3)


@triton.jit
def _with_sign_tl(v, sign):
    return (v.to(tl.int32, bitcast=True) | (sign << 31)).to(tl.float32, bitcast=True)


@triton.jit
def decode_e4m3_tl(c):
    """int32 e4m3 codes -> fp32."""
    e = (c >> 3) & 15
    m = c & 7
    normal = (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True)
    sub = m.to(tl.float32) * 0.001953125
    return _with_sign_tl(tl.where(e > 0, normal, sub), (c >> 7) & 1)


@triton.jit
def decode_e2m1_tl(c):
    """int32 e2m1 codes (0..15) -> fp32."""
    e = (c >> 1) & 3
    m = c & 1
    normal = (((e + 126) << 23) | (m << 22)).to(tl.float32, bitcast=True)
    sub = m.to(tl.float32) * 0.5
    return _with_sign_tl(tl.where(e > 0, normal, sub), (c >> 3) & 1)


@triton.jit
def _lane_tl(cw, s, shift: tl.constexpr, FP4: tl.constexpr):
    """BF16 value of the code at bit ``shift`` of each word times its scale."""
    if FP4:
        v = decode_e2m1_tl((cw >> shift) & 15)
    else:
        v = decode_e4m3_tl((cw >> shift) & 0xFF)
    return (v * s).to(tl.bfloat16)


@triton.jit
def load_records_tl(ptr, row_off, ok, D: tl.constexpr, FP4: tl.constexpr,
                    QBLOCK: tl.constexpr, E4M3_SCALE: tl.constexpr,
                    CODE_BYTES: tl.constexpr):
    """BF16 ``[N, D]`` values of the records at byte offsets ``row_off`` [N]
    (int64, multiples of 4) from uint8 ``ptr``, as an MMA operand; rows not
    ``ok`` are +0.0 (as a masked BF16 load with ``other=0``).

    The codes are read as 32-bit words, each word's 4 (FP8) or 8 (FP4)
    values decoded and scaled per lane to BF16, and the lanes interleaved
    with BF16 joins. That shape keeps the MMA operand layout the BF16 rows
    get: Triton derives a dot operand's ``kWidth`` from the narrowest load
    in its backward slice, halved when the slice holds a join of twice that
    width (``computeOrigBitWidth``), so 8-bit loads or 32-bit joins would
    give kWidth 4 instead of BF16's 2, which permutes the K order of the
    MMA accumulation (results would differ in the last bits). The scale
    bytes are selected by shifts, without joins."""
    words = (ptr + row_off[:, None]).to(tl.pointer_type(tl.int32))
    n: tl.constexpr = row_off.shape[0]
    if FP4:
        W: tl.constexpr = D // 8
        R: tl.constexpr = QBLOCK // 8  # words per scale
    else:
        W: tl.constexpr = D // 4
        R: tl.constexpr = QBLOCK // 4
    NSW: tl.constexpr = D // QBLOCK // 4  # scale words
    cw = tl.load(words + tl.arange(0, W)[None, :], mask=ok[:, None], other=0)
    sw = tl.load(words + CODE_BYTES // 4 + tl.arange(0, NSW)[None, :], mask=ok[:, None],
                 other=0)
    sw = tl.reshape(tl.broadcast_to(sw[:, :, None], (n, NSW, 4 * R)), (n, W))
    sb = (sw >> (((tl.arange(0, W) // R) % 4) * 8)[None, :]) & 0xFF
    if E4M3_SCALE:
        s = decode_e4m3_tl(sb)
    else:
        s = (sb << 23).to(tl.float32, bitcast=True)
    if FP4:
        # Value 8 w + j is nibble j of word w (byte j // 2, low nibble first).
        v = tl.join(
            tl.join(tl.join(_lane_tl(cw, s, 0, True), _lane_tl(cw, s, 16, True)),
                    tl.join(_lane_tl(cw, s, 8, True), _lane_tl(cw, s, 24, True))),
            tl.join(tl.join(_lane_tl(cw, s, 4, True), _lane_tl(cw, s, 20, True)),
                    tl.join(_lane_tl(cw, s, 12, True), _lane_tl(cw, s, 28, True))),
        )
    else:
        # Value 4 w + k is byte k of word w.
        v = tl.join(tl.join(_lane_tl(cw, s, 0, False), _lane_tl(cw, s, 16, False)),
                    tl.join(_lane_tl(cw, s, 8, False), _lane_tl(cw, s, 24, False)))
    return tl.reshape(v, (n, D))


@triton.jit
def _pack_store_kernel(
    x_ptr,
    slot_ptr,
    cache_ptr,
    stride_xt,
    stride_cb,
    stride_ce,
    E: tl.constexpr,
    D: tl.constexpr,
    QBLOCK: tl.constexpr,
    FP4: tl.constexpr,
    E4M3_SCALE: tl.constexpr,
    CODE_BYTES: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    slot = tl.load(slot_ptr + t).to(tl.int64)
    if slot < 0:
        return
    base = cache_ptr + (slot // E) * stride_cb + (slot % E) * stride_ce
    NB: tl.constexpr = D // QBLOCK
    rows = tl.arange(0, NB)
    cols = tl.arange(0, QBLOCK)
    x = tl.load(x_ptr + t * stride_xt + rows[:, None] * QBLOCK + cols[None, :]).to(tl.float32)
    if FP4:
        q, s = fp4_quant_tile(x, E4M3_SCALE)
        codes = encode_e2m1_tl(q)
        lo, hi = tl.split(tl.reshape(codes, (NB, QBLOCK // 2, 2)))
        half = tl.arange(0, QBLOCK // 2)
        tl.store(base + rows[:, None] * (QBLOCK // 2) + half[None, :],
                 (lo | (hi << 4)).to(tl.uint8))
    else:
        q, s = fp8_quant_tile(x)
        tl.store(base + rows[:, None] * QBLOCK + cols[None, :],
                 encode_e4m3_tl(q).to(tl.uint8))
    if E4M3_SCALE:
        sb = encode_e4m3_tl(s)
    else:
        sb = (s.to(tl.int32, bitcast=True) >> 23) & 0xFF
    tl.store(base + CODE_BYTES + rows, sb.to(tl.uint8))


def store(cache: torch.Tensor, slots: torch.Tensor, x: torch.Tensor, rec: KVRecord) -> None:
    """Quantize rows ``x`` [>= T, dim] (pre-quantization values, BF16) and
    write their records to ``cache`` at ``slots`` [T] (``slot < 0``: nothing
    written). On CUDA one launch, no host read."""
    dim = x.shape[-1]
    check_cache(cache, rec, dim)
    if not x.is_cuda:
        store_torch(cache, slots, x, rec)
        return
    n = slots.shape[0]
    if n == 0:
        return
    if x.stride(-1) != 1:
        x = x.contiguous()
    _pack_store_kernel[(n,)](
        x,
        slots,
        cache,
        x.stride(0),
        cache.stride(0),
        cache.stride(1),
        E=cache.shape[1],
        D=dim,
        QBLOCK=rec.block,
        FP4=rec.fp4,
        E4M3_SCALE=rec.e4m3_scale,
        CODE_BYTES=rec.code_bytes(dim),
        num_warps=4,
    )
