// EXL3 routed-expert MoE on the tensor cores for prefill-sized calls (more
// than 64 rows) of two served families of one set of kernel templates:
// Qwen (hidden 2560, intermediate 640, every expert matrix bits_x2 10 with
// the mul1 codebook, codebook 2) and GLM (hidden 4096, intermediate 2048,
// every expert matrix bits_x2 5 with the mul1 codebook, SwiGLU clamped to
// the family's fixed limit 10.0). The caller
// (vllm_cmp170hx/kernels/exl3/quant.py) takes this operator only for those
// configurations, in prefill steps, for calls of more than 64 rows; every
// other call stays on moe_out (moe.cu) or moe_out_v2 (moe_v2.cu), which this
// file does not change.
//
// The small-row schedule (moe_out_decode_tc, csrc/exl3/moe_prefill.h) runs
// the GLM GEMMs on a decode-sized route: 8-row route tiles and GEMM blocks of
// a single 8-row MMA group, so a block's operand stage is one eighth the size
// and many more blocks share an SM. Decode combines expert contributions with
// the original FP32 routing-weight FMAs; prefill retains its FP64 combine.
//
// Arithmetic: the input and activation stages (including moe_out's SwiGLU
// clipping for the family that has a limit) and the combine's per-route
// transform are moe_out's FP32 operations. Decode also preserves its FP32
// route sum; prefill sums in FP64 and rounds once to FP32, then to BF16. The
// expert GEMMs run m16n8k16 FP16 tensor-core MMAs on exactly decoded weights (A
// = W^T: the trellis lane layout is the A fragment layout) and a hi/lo FP16
// split of the FP32 activations (B; scaled exactly by one power of two per row
// and projection; lo is the FP16-rounded residual, so hi + lo keeps ~22 of the
// 24 significand bits). Per k16 step the low products (C = 0) seed the high
// MMA, and that k16 partial is summed in FP32 in increasing k. Everything is
// fixed-order and a row never depends on other rows: results are
// deterministic and independent of the routing order inside an expert.

#include "moe_prefill.h"
#include "common.h"
#include "codec.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <cstring>
#include <limits>
#include <type_traits>

namespace exl3 {
namespace moe_prefill {

// Rows per route tile (expert GEMM block) live in the served families
// (Family, below the GEMM shapes): 64 for the prefill families, 8 for the
// decode families, where a block holds exactly one 8-row MMA group and the
// operand stage of every K chunk is sized for 8 rows instead of 64.

// Integers the route kernel writes into the workspace tail each call:
//   [0] local assignment count, [1] route tile count, [2..3] unused,
//   sorted[assignments]: local assignment ids grouped by local expert,
//   tiles[assignments] (16-byte aligned): {expert, first sorted position, rows,
//   0}, position[assignments]: sorted position of each assignment (-1: not
//   local).
struct Layout {
  static constexpr int64_t kHeader = 4;
  __host__ __device__ static constexpr int64_t sorted_offset() {
    return kHeader;
  }
  __host__ __device__ static constexpr int64_t tiles_offset(
      int64_t assignments) {
    return kHeader + (assignments + 3) / 4 * 4;
  }
  __host__ __device__ static constexpr int64_t position_offset(
      int64_t assignments) {
    return tiles_offset(assignments) + 4 * assignments;
  }
  __host__ __device__ static constexpr int64_t ints(int64_t assignments) {
    return position_offset(assignments) + assignments;
  }
};

namespace {

constexpr int kRouteThreads = 1024;
constexpr int kOperandWarps = 8;

// Expert GEMM block: kWarps warps, each owning Shape::kTPW adjacent n16 output
// tiles, over one route tile of up to kTileRows rows; K streamed in kKC chunks,
// kStages deep.
constexpr int kWarps = 8;
constexpr int kThreads = 32 * kWarps;
constexpr int kKC = 64;
constexpr int kSteps = kKC / 16;
constexpr int kShippedStages = 3;

struct Routing {
  const void* ids;
  const void* expert_map;
  const int64_t* pointers;
  const int64_t* metadata;
  int64_t global_experts;
  int64_t local_experts;
  int64_t topk;
  bool ids64;
  bool map64;
};

__device__ __forceinline__ int64_t integer_at(const void* ptr, bool wide,
                                              int64_t i) {
  return wide ? static_cast<const int64_t*>(ptr)[i]
              : static_cast<const int32_t*>(ptr)[i];
}
__device__ __forceinline__ int64_t local_expert(const Routing& routing,
                                                int64_t assignment) {
  const int64_t global = integer_at(routing.ids, routing.ids64, assignment);
  if (global < 0 || global >= routing.global_experts) return -1;
  const int64_t local = integer_at(routing.expert_map, routing.map64, global);
  return local >= 0 && local < routing.local_experts ? local : -1;
}
__device__ __forceinline__ const half* expert_part(const Routing& routing,
                                                   int64_t expert,
                                                   int projection, int part) {
  return reinterpret_cast<const half*>(
      routing.pointers[expert * 9 + projection * 3 + part]);
}

// Exclusive block-wide scan of two counters; also returns the totals.
__device__ __forceinline__ void block_scan2(int a, int b, int& a_before,
                                            int& b_before, int& a_total,
                                            int& b_total) {
  __shared__ int warp_a[32];
  __shared__ int warp_b[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  int x = a;
  int y = b;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int xu = __shfl_up_sync(0xffffffffu, x, d);
    const int yu = __shfl_up_sync(0xffffffffu, y, d);
    if (lane >= d) {
      x += xu;
      y += yu;
    }
  }
  if (lane == 31) {
    warp_a[warp] = x;
    warp_b[warp] = y;
  }
  __syncthreads();
  if (warp == 0) {
    int wx = lane < warps ? warp_a[lane] : 0;
    int wy = lane < warps ? warp_b[lane] : 0;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int xu = __shfl_up_sync(0xffffffffu, wx, d);
      const int yu = __shfl_up_sync(0xffffffffu, wy, d);
      if (lane >= d) {
        wx += xu;
        wy += yu;
      }
    }
    warp_a[lane] = wx;
    warp_b[lane] = wy;
  }
  __syncthreads();
  a_before = x - a + (warp > 0 ? warp_a[warp - 1] : 0);
  b_before = y - b + (warp > 0 ? warp_b[warp - 1] : 0);
  a_total = warp_a[warps - 1];
  b_total = warp_b[warps - 1];
}

// Groups the local assignments by local expert into kTileRows-row tiles and
// records each assignment's sorted position (one block). Order inside an
// expert is arbitrary: a row's result never depends on its position. Traps
// on an expert outside the family's format (the caller's host checks cover the
// shapes; formats and trellis alignment live in device metadata).
template <class F>
__global__ void __launch_bounds__(kRouteThreads)
    route(Routing routing, int64_t assignments, int* table) {
  extern __shared__ int route_shared[];
  constexpr int kHidden = F::kHidden;
  constexpr int kIntermediate = F::kIntermediate;
  constexpr int kBitsX2 = F::kBitsX2;
  constexpr int kCodebook = F::kCodebook;
  constexpr int kTileRows = F::kTileRows;
  const int experts = static_cast<int>(routing.local_experts);
  int* cursor = route_shared;
  int* row_begin = cursor + experts;
  int* tile_begin = row_begin + experts;  // experts + 1 entries
  for (int e = threadIdx.x; e < experts; e += blockDim.x) {
    cursor[e] = 0;
    for (int q = 0; q < 3; ++q) {
      const int64_t* meta = routing.metadata + int64_t(e) * 12 + q * 4;
      const int64_t in = q < 2 ? kHidden : kIntermediate;
      const int64_t out = q < 2 ? kIntermediate : kHidden;
      const uintptr_t trellis =
          static_cast<uintptr_t>(routing.pointers[int64_t(e) * 9 + q * 3]);
      if (meta[0] != in || meta[1] != out || meta[2] != kBitsX2 ||
          meta[3] != kCodebook || (trellis & 15) != 0) {
        __trap();
      }
    }
  }
  __syncthreads();
  for (int64_t a = threadIdx.x; a < assignments; a += blockDim.x) {
    const int64_t e = local_expert(routing, a);
    if (e >= 0) atomicAdd(&cursor[e], 1);
  }
  __syncthreads();
  const int per = (experts + blockDim.x - 1) / blockDim.x;
  const int e0 = min(experts, static_cast<int>(threadIdx.x) * per);
  const int e1 = min(experts, e0 + per);
  int rows = 0;
  int tiles = 0;
  for (int e = e0; e < e1; ++e) {
    rows += cursor[e];
    tiles += (cursor[e] + kTileRows - 1) / kTileRows;
  }
  int row_at, tile_at, row_total, tile_total;
  block_scan2(rows, tiles, row_at, tile_at, row_total, tile_total);
  for (int e = e0; e < e1; ++e) {
    const int count = cursor[e];
    row_begin[e] = row_at;
    tile_begin[e] = tile_at;
    cursor[e] = row_at;
    row_at += count;
    tile_at += (count + kTileRows - 1) / kTileRows;
  }
  if (threadIdx.x == 0) {
    tile_begin[experts] = tile_total;
    table[0] = row_total;
    table[1] = tile_total;
  }
  __syncthreads();
  int4* tile_table =
      reinterpret_cast<int4*>(table + Layout::tiles_offset(assignments));
  for (int t = threadIdx.x; t < tile_total; t += blockDim.x) {
    // Last expert whose first tile is at or before t (empty experts share
    // their successor's start, so the search lands on the owner).
    int lo = 0;
    int hi = experts;
    while (hi - lo > 1) {
      const int mid = (lo + hi) >> 1;
      if (tile_begin[mid] <= t)
        lo = mid;
      else
        hi = mid;
    }
    const int first = row_begin[lo] + (t - tile_begin[lo]) * kTileRows;
    const int end = lo + 1 < experts ? row_begin[lo + 1] : row_total;
    tile_table[t] = make_int4(lo, first, min(kTileRows, end - first), 0);
  }
  int* sorted = table + Layout::sorted_offset();
  int* position = table + Layout::position_offset(assignments);
  for (int64_t a = threadIdx.x; a < assignments; a += blockDim.x) {
    const int64_t e = local_expert(routing, a);
    int at = -1;
    if (e >= 0) {
      at = atomicAdd(&cursor[e], 1);
      sorted[at] = static_cast<int>(a);
    }
    position[a] = at;
  }
}

// --------------------------------------------------------------------------
// Tensor-core operands.

// Exponent e of the power of two that puts max |v| 2^e in [2^14, 2^15), with
// 2^e and 2^-e normal FP32 (peak 0: e = 0).
__device__ __forceinline__ int split_exponent(float peak) {
  const int biased = static_cast<int>(__float_as_uint(peak) >> 23);
  return peak == 0.0f ? 0 : min(141 - biased, 126);
}
__device__ __forceinline__ float pow2(int e) {
  return __uint_as_float(uint32_t(127 + e) << 23);
}

// H128 over one 128-value block held by the 8 lanes of an aligned lane group
// (l = lane % 8): v[8 h + j] is element 64 h + 8 l + j. Exactly the per-element
// operations and stage order of hadamard128_warp (distances 1..64 in
// increasing order, lower element v + other, upper other - v, then the
// 1/sqrt(128) scale), so the results are bitwise equal; distances 1, 2, 4 and
// 64 run in registers, 8, 16, 32 by shuffles inside the group (3 shuffle
// stages instead of 5). Every lane of the warp must call it.
__device__ __forceinline__ void h128_register_stage(float (&v)[16],
                                                    int distance) {
  float next[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float other = v[i ^ distance];
    next[i] = (i & distance) ? __fsub_rn(other, v[i]) : __fadd_rn(v[i], other);
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = next[i];
}
__device__ __forceinline__ void hadamard128_group(float (&v)[16], int lane) {
  h128_register_stage(v, 1);
  h128_register_stage(v, 2);
  h128_register_stage(v, 4);
#pragma unroll
  for (int d = 1; d < 8; d <<= 1) {  // element distance 8 d
    const bool upper = (lane & d) != 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float other = __shfl_xor_sync(0xffffffffu, v[i], d);
      v[i] = upper ? __fsub_rn(other, v[i]) : __fadd_rn(v[i], other);
    }
  }
  h128_register_stage(v, 8);  // element distance 64
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = __fmul_rn(v[i], kH128Scale);
}

// Eight consecutive values as FP32 (exact widening); one 16-byte load when
// the row base is 16-byte aligned (`vec`), element loads otherwise.
__device__ __forceinline__ void load8(const __nv_bfloat16* p, bool vec,
                                      float* f) {
  if (vec) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      f[2 * q] = __uint_as_float(w[q] << 16);
      f[2 * q + 1] = __uint_as_float(w[q] & 0xffff0000u);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = __bfloat162float(p[j]);
  }
}
__device__ __forceinline__ void load8(const half* p, bool vec, float* f) {
  if (vec) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float2 pair =
          __half22float2(*reinterpret_cast<const __half2*>(&w[q]));
      f[2 * q] = pair.x;
      f[2 * q + 1] = pair.y;
    }
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = __half2float(p[j]);
  }
}
// FP32 workspace rows: always 16-byte aligned (checked workspace base, row and
// column offsets multiples of 8 floats).
__device__ __forceinline__ void load8(const float* p, float* f) {
  const float4 a = *reinterpret_cast<const float4*>(p);
  const float4 b = *reinterpret_cast<const float4*>(p + 4);
  f[0] = a.x;
  f[1] = a.y;
  f[2] = a.z;
  f[3] = a.w;
  f[4] = b.x;
  f[5] = b.y;
  f[6] = b.z;
  f[7] = b.w;
}
__device__ __forceinline__ bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

template <int kVals>
__device__ __forceinline__ float lane_peak(const float (&v)[kVals]) {
  float peak = 0.0f;
#pragma unroll
  for (int i = 0; i < kVals; ++i) peak = fmaxf(peak, fabsf(v[i]));
  return peak;
}

__device__ __forceinline__ uint32_t pack_halves(__half a, __half b) {
  return uint32_t(__half_as_ushort(a)) | (uint32_t(__half_as_ushort(b)) << 16);
}
// Eight consecutive operand values y = v 2^e as the FP16 high halves at `hi`
// and the FP16-rounded residuals at `lo` (both 16-byte aligned).
__device__ __forceinline__ void store8_split(const float* v, float scale,
                                             __half* hi, __half* lo) {
  uint32_t h[4];
  uint32_t l[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float y0 = __fmul_rn(v[2 * q], scale);
    const float y1 = __fmul_rn(v[2 * q + 1], scale);
    const __half h0 = __float2half_rn(y0);
    const __half h1 = __float2half_rn(y1);
    h[q] = pack_halves(h0, h1);
    l[q] = pack_halves(__float2half_rn(__fsub_rn(y0, __half2float(h0))),
                       __float2half_rn(__fsub_rn(y1, __half2float(h1))));
  }
  *reinterpret_cast<uint4*>(hi) = make_uint4(h[0], h[1], h[2], h[3]);
  *reinterpret_cast<uint4*>(lo) = make_uint4(l[0], l[1], l[2], l[3]);
}

// One warp per (sorted position, gate/up projection): x * suh and H128 (the
// shipped FP32 operations), then the split operand row (K halves high, then
// K halves low) and its 2^-e. Lane group g of iteration it holds 128-block
// 4 it + g.
template <class F>
__global__ void __launch_bounds__(32 * kOperandWarps)
    gate_up_operand(const __nv_bfloat16* x, Routing routing, const int* table,
                    __half* operand, float* unscale) {
  constexpr int kHidden = F::kHidden;
  constexpr int kIters = kHidden / 512;
  const int lane = threadIdx.x & 31;
  const int lane_col = 128 * (lane >> 3) + 8 * (lane & 7);
  const int* sorted = table + Layout::sorted_offset();
  const int64_t tasks = int64_t(table[0]) * 2;
  for (int64_t task = int64_t(blockIdx.x) * kOperandWarps + (threadIdx.x >> 5);
       task < tasks; task += int64_t(gridDim.x) * kOperandWarps) {
    const int64_t pos = task >> 1;
    const int projection = static_cast<int>(task & 1);
    const int64_t assignment = sorted[pos];
    const half* suh =
        expert_part(routing, local_expert(routing, assignment), projection, 1);
    const __nv_bfloat16* row = x + (assignment / routing.topk) * kHidden;
    const bool vec = aligned16(row) && aligned16(suh);
    float v[kIters][16];
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int col = 512 * it + 64 * h + lane_col;
        float xf[8];
        float sf[8];
        load8(row + col, vec, xf);
        load8(suh + col, vec, sf);
#pragma unroll
        for (int j = 0; j < 8; ++j) v[it][8 * h + j] = __fmul_rn(xf[j], sf[j]);
      }
      hadamard128_group(v[it], lane);
    }
    float peak = lane_peak(reinterpret_cast<const float (&)[kIters * 16]>(v));
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1)
      peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, d));
    const int e = split_exponent(peak);
    const float scale = pow2(e);
    __half* out = operand + task * 2 * kHidden;
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int col = 512 * it + 64 * h + lane_col;
        store8_split(&v[it][8 * h], scale, out + col, out + kHidden + col);
      }
    }
    if (lane == 0) unscale[pos * 3 + projection] = pow2(-e);
  }
}

// The mixed role modes' single tensor-core projection: one warp per sorted
// position, exactly gate_up_operand's FP32 operations for that projection
// (x * suh, H128, hi/lo split, 2^-e). The operand row for the position sits at
// operand + pos * 2 * kHidden halves (a one-projection GateUp row: kRowHalves =
// 2 * kHidden), so the single-projection GEMM reads it unchanged.
template <class F>
__global__ void __launch_bounds__(32 * kOperandWarps)
    gate_up_operand_proj(const __nv_bfloat16* x, Routing routing,
                         const int* table, __half* operand, float* unscale,
                         int projection) {
  constexpr int kHidden = F::kHidden;
  constexpr int kIters = kHidden / 512;
  const int lane = threadIdx.x & 31;
  const int lane_col = 128 * (lane >> 3) + 8 * (lane & 7);
  const int* sorted = table + Layout::sorted_offset();
  const int64_t tasks = table[0];
  for (int64_t pos = int64_t(blockIdx.x) * kOperandWarps + (threadIdx.x >> 5);
       pos < tasks; pos += int64_t(gridDim.x) * kOperandWarps) {
    const int64_t assignment = sorted[pos];
    const half* suh =
        expert_part(routing, local_expert(routing, assignment), projection, 1);
    const __nv_bfloat16* row = x + (assignment / routing.topk) * kHidden;
    const bool vec = aligned16(row) && aligned16(suh);
    float v[kIters][16];
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int col = 512 * it + 64 * h + lane_col;
        float xf[8];
        float sf[8];
        load8(row + col, vec, xf);
        load8(suh + col, vec, sf);
#pragma unroll
        for (int j = 0; j < 8; ++j) v[it][8 * h + j] = __fmul_rn(xf[j], sf[j]);
      }
      hadamard128_group(v[it], lane);
    }
    float peak = lane_peak(reinterpret_cast<const float (&)[kIters * 16]>(v));
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1)
      peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, d));
    const int e = split_exponent(peak);
    const float scale = pow2(e);
    __half* out = operand + pos * 2 * kHidden;
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int col = 512 * it + 64 * h + lane_col;
        store8_split(&v[it][8 * h], scale, out + col, out + kHidden + col);
      }
    }
    if (lane == 0) unscale[pos * 3 + projection] = pow2(-e);
  }
}

// Hybrid: scalar gate/up and activation have already produced H128-transformed
// down inputs in assignment order. Only reorder and split; do not transform
// twice.
template <class F>
__global__ void __launch_bounds__(32 * kOperandWarps)
    stage_down_operand(const float* activated, const int* table,
                       __half* operand, float* unscale) {
  constexpr int kIntermediate = F::kIntermediate;
  static_assert(kIntermediate % 512 == 0,
                "hybrid serves the GLM down geometry");
  constexpr int kIters = kIntermediate / 512;
  const int lane = threadIdx.x & 31;
  const int lane_col = 128 * (lane >> 3) + 8 * (lane & 7);
  const int* sorted = table + Layout::sorted_offset();
  for (int64_t pos = int64_t(blockIdx.x) * kOperandWarps + (threadIdx.x >> 5);
       pos < table[0]; pos += int64_t(gridDim.x) * kOperandWarps) {
    const float* row = activated + int64_t(sorted[pos]) * kIntermediate;
    float v[kIters][16];
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
#pragma unroll
      for (int h = 0; h < 2; ++h)
        load8(row + 512 * it + 64 * h + lane_col, &v[it][8 * h]);
    }
    float peak = lane_peak(reinterpret_cast<const float (&)[kIters * 16]>(v));
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1)
      peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, d));
    const int e = split_exponent(peak);
    const float scale = pow2(e);
    __half* out = operand + pos * 2 * kIntermediate;
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int col = 512 * it + 64 * h + lane_col;
        store8_split(&v[it][8 * h], scale, out + col,
                     out + kIntermediate + col);
      }
    }
    if (lane == 0) unscale[pos * 3 + 2] = pow2(-e);
  }
}

// Down transform of one value: gate and up output transforms already applied
// (H128 done), then svh, the family's SwiGLU clipping where it has a limit
// (moe_out's exact fminf/fmaxf operations), SiLU(gate) * up and down suh
// (the shipped FP32 operations).
template <class F>
__device__ __forceinline__ float down_input(float g, float u, float gate_svh,
                                            float up_svh, float down_suh) {
  float gv = __fmul_rn(g, gate_svh);
  float uv = __fmul_rn(u, up_svh);
  if constexpr (F::kSwigluLimit > 0.0f) {
    gv = fminf(gv, F::kSwigluLimit);
    uv = fminf(fmaxf(uv, -F::kSwigluLimit), F::kSwigluLimit);
  }
  const float silu = gv / __fadd_rn(1.0f, expf(-gv));
  return __fmul_rn(__fmul_rn(silu, uv), down_suh);
}

// One warp per sorted position: gate and up output transforms (H128, svh),
// the SwiGLU stage, down suh, down H128 (the shipped FP32 operations), then
// the split down operand row and its 2^-e. For five 128-blocks: blocks 0..3
// in the lane-group layout (group g holds block g), block 4 in the warp
// layout of hadamard128_warp (lane holds k = 512 + lane + 32 i). A wider
// intermediate keeps every block in the hadamard128_warp layout (the same
// H128 bitwise, the group layout's shuffles at warp distance).
template <class F>
__global__ void __launch_bounds__(32 * kOperandWarps)
    down_operand(Routing routing, const int* table, const float* gate_up,
                 __half* operand, float* unscale) {
  constexpr int kIntermediate = F::kIntermediate;
  constexpr int kTail = 512;
  const int lane = threadIdx.x & 31;
  const int lane_col = 128 * (lane >> 3) + 8 * (lane & 7);
  const int* sorted = table + Layout::sorted_offset();
  const int64_t tasks = table[0];
  for (int64_t pos = int64_t(blockIdx.x) * kOperandWarps + (threadIdx.x >> 5);
       pos < tasks; pos += int64_t(gridDim.x) * kOperandWarps) {
    const int64_t expert = local_expert(routing, sorted[pos]);
    const half* gate_svh = expert_part(routing, expert, 0, 2);
    const half* up_svh = expert_part(routing, expert, 1, 2);
    const half* down_suh = expert_part(routing, expert, 2, 1);
    const float* gate = gate_up + pos * 2 * kIntermediate;
    const float* up = gate + kIntermediate;

    if constexpr (kIntermediate == 5 * 128) {
      const bool vec =
          aligned16(gate_svh) && aligned16(up_svh) && aligned16(down_suh);
      float v[16];
      {
        float g[16];
        float u[16];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          load8(gate + lane_col + 64 * h, &g[8 * h]);
          load8(up + lane_col + 64 * h, &u[8 * h]);
        }
        hadamard128_group(g, lane);
        hadamard128_group(u, lane);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int col = lane_col + 64 * h;
          float gs[8];
          float us[8];
          float ds[8];
          load8(gate_svh + col, vec, gs);
          load8(up_svh + col, vec, us);
          load8(down_suh + col, vec, ds);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            v[8 * h + j] =
                down_input<F>(g[8 * h + j], u[8 * h + j], gs[j], us[j], ds[j]);
          }
        }
        hadamard128_group(v, lane);
      }
      float w[4];
      {
        float g[4];
        float u[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          g[i] = gate[kTail + lane + 32 * i];
          u[i] = up[kTail + lane + 32 * i];
        }
        hadamard128_warp(g, lane);
        hadamard128_warp(u, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int col = kTail + lane + 32 * i;
          w[i] = down_input<F>(g[i], u[i], __half2float(gate_svh[col]),
                               __half2float(up_svh[col]),
                               __half2float(down_suh[col]));
        }
        hadamard128_warp(w, lane);
      }

      float peak = fmaxf(lane_peak(v), lane_peak(w));
#pragma unroll
      for (int d = 16; d >= 1; d >>= 1)
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, d));
      const int e = split_exponent(peak);
      const float scale = pow2(e);
      __half* out = operand + pos * 2 * kIntermediate;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int col = lane_col + 64 * h;
        store8_split(&v[8 * h], scale, out + col, out + kIntermediate + col);
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int k = kTail + lane + 32 * i;
        const float y = __fmul_rn(w[i], scale);
        const __half h = __float2half_rn(y);
        out[k] = h;
        out[kIntermediate + k] = __float2half_rn(__fsub_rn(y, __half2float(h)));
      }
      if (lane == 0) unscale[pos * 3 + 2] = pow2(-e);
    } else {
      static_assert(kIntermediate % 128 == 0, "one H128 per 128-block");
      // Wide intermediate: every 128-block in the hadamard128_warp layout
      // (lane holds k = 128 b + lane + 32 i).
      constexpr int kBlocks = kIntermediate / 128;
      float v[kBlocks][4];
#pragma unroll
      for (int b = 0; b < kBlocks; ++b) {
        float g[4];
        float u[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int col = 128 * b + lane + 32 * i;
          g[i] = gate[col];
          u[i] = up[col];
        }
        hadamard128_warp(g, lane);
        hadamard128_warp(u, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int col = 128 * b + lane + 32 * i;
          v[b][i] = down_input<F>(g[i], u[i], __half2float(gate_svh[col]),
                                  __half2float(up_svh[col]),
                                  __half2float(down_suh[col]));
        }
        hadamard128_warp(v[b], lane);
      }
      float peak = 0.0f;
#pragma unroll
      for (int b = 0; b < kBlocks; ++b) peak = fmaxf(peak, lane_peak(v[b]));
#pragma unroll
      for (int d = 16; d >= 1; d >>= 1)
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, d));
      const int e = split_exponent(peak);
      const float scale = pow2(e);
      __half* out = operand + pos * 2 * kIntermediate;
#pragma unroll
      for (int b = 0; b < kBlocks; ++b) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int k = 128 * b + lane + 32 * i;
          const float y = __fmul_rn(v[b][i], scale);
          const __half h = __float2half_rn(y);
          out[k] = h;
          out[kIntermediate + k] =
              __float2half_rn(__fsub_rn(y, __half2float(h)));
        }
      }
      if (lane == 0) unscale[pos * 3 + 2] = pow2(-e);
    }
  }
}

// --------------------------------------------------------------------------
// Decode-family operand staging: one block per task instead of one warp.
//
// The 8-row decode families stage at most kMoeDecodeTcMaxRows * topk
// positions, so the warp-per-task kernels above launch a handful of blocks
// and each warp walks a whole row serially (down_operand: 16 128-blocks).
// These kernels give each task a block and each 128-block (or 512-column
// slice) its own warp. Every element runs the same FP32 operations as the
// warp kernels; only the row peak is reduced across warps, and the peak is
// a maximum of non-negative, non-NaN values (lane_peak starts at +0 and
// fmaxf drops NaN), which is order-independent -- so the split exponent, the
// operand halves and 2^-e are bitwise those of the warp kernels.

// Row peak of a block: warp butterfly, then the warps' peaks through shared
// memory. Every thread of the block must call it once.
template <int kBlockWarps>
__device__ __forceinline__ float block_peak(float peak, int lane, int warp) {
  __shared__ float warp_peaks[kBlockWarps];
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1)
    peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, d));
  if (lane == 0) warp_peaks[warp] = peak;
  __syncthreads();
  float row = warp_peaks[0];
#pragma unroll
  for (int w = 1; w < kBlockWarps; ++w) row = fmaxf(row, warp_peaks[w]);
  return row;
}

// gate_up_operand with one block per (sorted position, projection) task:
// warp w holds iteration w's 512 columns (lane group g: 128-block 4 w + g).
template <class F>
__global__ void __launch_bounds__(32 * (F::kHidden / 512))
    gate_up_operand_block(const __nv_bfloat16* x, Routing routing,
                          const int* table, __half* operand, float* unscale) {
  constexpr int kHidden = F::kHidden;
  constexpr int kBlockWarps = kHidden / 512;
  const int lane = threadIdx.x & 31;
  const int it = threadIdx.x >> 5;
  const int lane_col = 128 * (lane >> 3) + 8 * (lane & 7);
  const int* sorted = table + Layout::sorted_offset();
  const int64_t task = blockIdx.x;
  if (task >= int64_t(table[0]) * 2) return;  // block-uniform
  const int64_t pos = task >> 1;
  const int projection = static_cast<int>(task & 1);
  const int64_t assignment = sorted[pos];
  const half* suh =
      expert_part(routing, local_expert(routing, assignment), projection, 1);
  const __nv_bfloat16* row = x + (assignment / routing.topk) * kHidden;
  const bool vec = aligned16(row) && aligned16(suh);
  float v[16];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int col = 512 * it + 64 * h + lane_col;
    float xf[8];
    float sf[8];
    load8(row + col, vec, xf);
    load8(suh + col, vec, sf);
#pragma unroll
    for (int j = 0; j < 8; ++j) v[8 * h + j] = __fmul_rn(xf[j], sf[j]);
  }
  hadamard128_group(v, lane);
  const float peak = block_peak<kBlockWarps>(lane_peak(v), lane, it);
  const int e = split_exponent(peak);
  const float scale = pow2(e);
  __half* out = operand + task * 2 * kHidden;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int col = 512 * it + 64 * h + lane_col;
    store8_split(&v[8 * h], scale, out + col, out + kHidden + col);
  }
  if (threadIdx.x == 0) unscale[pos * 3 + projection] = pow2(-e);
}

// down_operand's wide branch with one block per sorted position: warp b holds
// 128-block b in the hadamard128_warp layout (lane holds k = 128 b + lane +
// 32 i), exactly the wide branch's per-block operations.
template <class F>
__global__ void __launch_bounds__(32 * (F::kIntermediate / 128))
    down_operand_block(Routing routing, const int* table, const float* gate_up,
                       __half* operand, float* unscale) {
  constexpr int kIntermediate = F::kIntermediate;
  constexpr int kBlockWarps = kIntermediate / 128;
  static_assert(kIntermediate != 5 * 128,
                "the five-block layout stays on down_operand");
  const int lane = threadIdx.x & 31;
  const int b = threadIdx.x >> 5;
  const int* sorted = table + Layout::sorted_offset();
  const int64_t pos = blockIdx.x;
  if (pos >= table[0]) return;  // block-uniform
  const int64_t expert = local_expert(routing, sorted[pos]);
  const half* gate_svh = expert_part(routing, expert, 0, 2);
  const half* up_svh = expert_part(routing, expert, 1, 2);
  const half* down_suh = expert_part(routing, expert, 2, 1);
  const float* gate = gate_up + pos * 2 * kIntermediate;
  const float* up = gate + kIntermediate;
  float g[4];
  float u[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = 128 * b + lane + 32 * i;
    g[i] = gate[col];
    u[i] = up[col];
  }
  hadamard128_warp(g, lane);
  hadamard128_warp(u, lane);
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int col = 128 * b + lane + 32 * i;
    v[i] = down_input<F>(g[i], u[i], __half2float(gate_svh[col]),
                         __half2float(up_svh[col]),
                         __half2float(down_suh[col]));
  }
  hadamard128_warp(v, lane);
  const float peak = block_peak<kBlockWarps>(lane_peak(v), lane, b);
  const int e = split_exponent(peak);
  const float scale = pow2(e);
  __half* out = operand + pos * 2 * kIntermediate;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int k = 128 * b + lane + 32 * i;
    const float y = __fmul_rn(v[i], scale);
    const __half h = __float2half_rn(y);
    out[k] = h;
    out[kIntermediate + k] = __float2half_rn(__fsub_rn(y, __half2float(h)));
  }
  if (threadIdx.x == 0) unscale[pos * 3 + 2] = pow2(-e);
}

// --------------------------------------------------------------------------
// Expert GEMM.

__device__ __forceinline__ uint32_t smem_address(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// 16-byte cp.async to a shared-memory address; src_size 0 zero-fills.
__device__ __forceinline__ void copy16_to(uint32_t dst, const void* src,
                                          int src_size) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst),
               "l"(src), "r"(src_size));
}
__device__ __forceinline__ void copy_commit() {
  asm volatile("cp.async.commit_group;");
}
template <int kPending>
__device__ __forceinline__ void copy_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(kPending));
}
// d = A B + c (FP16 in, FP32 accumulate).
__device__ __forceinline__ void mma(const uint32_t (&a)[4], uint32_t b0,
                                    uint32_t b1, const float (&c)[4],
                                    float (&d)[4]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]),
        "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Two mul1 codebook values as an FP16 pair (low half = s0): exactly
// codebook_pair<2> before its widening to FP32.
__device__ __forceinline__ uint32_t mul1_half2(uint32_t s0, uint32_t s1) {
  const uint32_t sum0 = __dp4a(s0 * 0x83DCD12Du, 0x01010101u, 0x6400u);
  const uint32_t sum1 = __dp4a(s1 * 0x83DCD12Du, 0x01010101u, 0x6400u);
  const uint32_t packed = __byte_perm(sum0, sum1, 0x5410);
  const uint32_t scale = 0x1eee1eeeu;
  const uint32_t bias = 0xc931c931u;
  const __half2 value = __hfma2(*reinterpret_cast<const __half2*>(&packed),
                                *reinterpret_cast<const __half2*>(&scale),
                                *reinterpret_cast<const __half2*>(&bias));
  return *reinterpret_cast<const uint32_t*>(&value);
}

// A lane's eight decoded values as the m16n8k16 A fragment of W^T (fragment
// row = output column, fragment column = k): value pairs (0,1), (4,5), (2,3),
// (6,7) of decode_lane's order.
template <int BitsX2>
__device__ __forceinline__ void decode_fragment(
    const uint32_t (&words)[TileRing<BitsX2>::kLoadWords], int shift,
    uint32_t (&a)[4]) {
  using Ring = TileRing<BitsX2>;
  uint32_t u[Ring::kWindowWords];
#pragma unroll
  for (int i = 0; i < Ring::kWindowWords; ++i) {
    const uint32_t next = i + 1 < Ring::kLoadWords ? words[i + 1] : 0u;
    u[i] = __funnelshift_l(next, words[i], shift);
  }
  uint32_t s[8];
  s[0] = window_state<Ring::rel(0), Ring::kWindowWords>(u);
  s[1] = window_state<Ring::rel(1), Ring::kWindowWords>(u);
  s[2] = window_state<Ring::rel(2), Ring::kWindowWords>(u);
  s[3] = window_state<Ring::rel(3), Ring::kWindowWords>(u);
  s[4] = window_state<Ring::rel(4), Ring::kWindowWords>(u);
  s[5] = window_state<Ring::rel(5), Ring::kWindowWords>(u);
  s[6] = window_state<Ring::rel(6), Ring::kWindowWords>(u);
  s[7] = window_state<Ring::rel(7), Ring::kWindowWords>(u);
  a[0] = mul1_half2(s[0], s[1]);
  a[1] = mul1_half2(s[4], s[5]);
  a[2] = mul1_half2(s[2], s[3]);
  a[3] = mul1_half2(s[6], s[7]);
}

struct Gemm {
  const int* table;
  int64_t assignments;  // routing table capacity
  const int64_t* pointers;
  const __half* operand;  // [pos][projection][hi, lo][k]
  const float* unscale;   // [pos][3]
  float*
      out;  // [pos][projection][n], or [assignment][n] for a mixed projection
  const int*
      row_map;  // sorted position -> assignment; single gate/up shapes only
};

// Compile-time GEMM shape: K, N, projections per launch (2: gate and up; 1:
// down), the first projection's index in the expert's 3 matrices and the n16
// output tiles per warp (each B fragment feeds TPW decoded A fragments), and
// the expert weight format bits_x2 and the rows per route tile.
//
// One stage of the K pipeline: the hi/lo operand rows of the route tile (row
// = 64 halves in eight 16-byte chunks, chunk c stored at c ^ (row & 7)) at
// offset 0, then the block's kBlockTiles adjacent trellis tiles of each k16
// step at kStageW.
template <int K, int N, int Projections, int First, int TPW, int BitsX2,
          int TileRows>
struct Shape {
  static constexpr int kTileRows = TileRows;
  static constexpr int kK = K;
  static constexpr int kN = N;
  static constexpr int kProjections = Projections;
  static constexpr int kFirst = First;
  static constexpr int kTPW = TPW;
  static constexpr int kBitsX2 = BitsX2;
  static constexpr int kTileBytes = 16 * BitsX2;  // one 16x16 trellis tile
  static constexpr int kBlockTiles = kWarps * TPW;
  static constexpr int kChunks = K / kKC;
  static constexpr int kColBlocks = N / (16 * kBlockTiles);
  static constexpr int kTilesN = N / 16;
  static constexpr int64_t kRowHalves = int64_t(Projections) * 2 * K;
  static constexpr int64_t kWChunk = int64_t(kSteps) * kTilesN * kTileBytes;
  static constexpr uint32_t kStageW = 2 * kTileRows * kKC * 2;
  static constexpr uint32_t kStageBytes =
      kStageW + kSteps * kBlockTiles * kTileBytes;
  static constexpr int kStages = kShippedStages;
  static constexpr int kSmemBytes = kStages * kStageBytes;
  static_assert(K % kKC == 0 && N % (16 * kBlockTiles) == 0 &&
                    kChunks >= kStages - 1,
                "served shape");
  static_assert(kStageBytes % 16 == 0, "16-byte aligned stages");
};
// A shape with a deeper cp.async ring (lab: VLLM_EXL3_MOE_DECODE_TC_STAGES).
// Only how far ahead the trellis and operand chunks are fetched changes; every
// MMA and FP32 add runs in the same order, so the results are bitwise S's.
template <class S, int Stages>
struct Staged : S {
  static constexpr int kStages = Stages;
  static constexpr int kSmemBytes = Stages * S::kStageBytes;
  static_assert(Stages >= 2 && S::kChunks >= Stages - 1, "pipeline depth");
};
// One served family: the MoE shape, the expert weight format (always the
// mul1 codebook) and the integer SwiGLU limit the family ships with (0: none;
// the GLM family's fixed model limit 10, used as the exact FP32 constant, so
// the template argument stays integral) and the rows per route tile. The
// family selects every kernel template instantiation; the tensor-core
// algorithm and the arithmetic are shared.
template <int Hidden, int Intermediate, int BitsX2, int SwigluLimit,
          int TileRows>
struct Family {
  static constexpr int kTileRows = TileRows;
  static constexpr int kHidden = Hidden;
  static constexpr int kIntermediate = Intermediate;
  static constexpr int kBitsX2 = BitsX2;
  static constexpr int kCodebook = 2;
  static constexpr float kSwigluLimit = static_cast<float>(SwigluLimit);
  using GateUpShape = Shape<Hidden, Intermediate, 2, 0, 1, BitsX2, TileRows>;
  // The mixed role modes compute one gate/up projection on the tensor cores in
  // its own launch (Projections 1); the MMA arithmetic per projection is the
  // GateUpShape's, so a tensor-core gate/up here is bit-for-bit the full
  // mode's.
  using TcGateShape = Shape<Hidden, Intermediate, 1, 0, 1, BitsX2, TileRows>;
  using TcUpShape = Shape<Hidden, Intermediate, 1, 1, 1, BitsX2, TileRows>;
  using DownShape = Shape<Intermediate, Hidden, 1, 2, 2, BitsX2, TileRows>;
  static_assert(Hidden % 512 == 0, "operand and combine spans");
  static_assert(Intermediate % 128 == 0, "one H128 per 128-block");
  static_assert(Intermediate <= Hidden, "the down rows fit the operand region");
};
using QwenFamily = Family<2560, 640, 10, 0, 64>;
using GlmFamily = Family<4096, 2048, 5, 10, 64>;
using Glm35Family = Family<4096, 2048, 7, 10, 64>;
// The GLM decode families: the same shape, format and SwiGLU limit as the
// GLM prefill families, on the small-row schedule of 8-row route tiles.
using GlmDecodeFamily = Family<4096, 2048, 5, 10, 8>;
using Glm35DecodeFamily = Family<4096, 2048, 7, 10, 8>;

template <class S>
__device__ __forceinline__ uint32_t next_stage(uint32_t offset) {
  return offset + S::kStageBytes == S::kStages * S::kStageBytes
             ? 0u
             : offset + S::kStageBytes;
}
__device__ __forceinline__ void ldmatrix4_at(uint32_t address,
                                             uint32_t (&r)[4]) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(address));
}
__device__ __forceinline__ uint32_t lds32(uint32_t address) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(address));
  return v;
}

// The K loop and epilogue of one block item for a route tile of kG 8-row
// groups (rows in 8 kG - 7 .. 8 kG): straight-line code, so the group MMAs of
// a k16 step overlap (a runtime group bound made every group a separately
// serialised branch). The ldmatrix of group g + 1 is issued before the MMAs of
// group g; each B fragment serves the warp's S::kTPW n16 tiles. Shared
// addresses are per-lane constants plus the stage offset; `load` issues the
// next chunk's copies into its own rotating stage.
template <int kG, class S, int kPlanes, class Load>
__device__ __forceinline__ void run_tile(const Gemm& p, uint32_t smem,
                                         Load& load, int rows, int first,
                                         int pi, int cb) {
  constexpr int TPW = S::kTPW;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  using Ring = TileRing<S::kBitsX2>;
  float sum[TPW][kG][4];
#pragma unroll
  for (int j = 0; j < TPW; ++j) {
#pragma unroll
    for (int g = 0; g < kG; ++g) {
#pragma unroll
      for (int i = 0; i < 4; ++i) sum[j][g][i] = 0.0f;
    }
  }
  const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  // ldmatrix source of this lane: matrix lane / 8 = (hi k0..7, hi k8..15,
  // lo k0..7, lo k8..15) of rows 8 g + lane % 8; for k16 step s the 16-byte
  // vector (2 s + lane / 8 % 2) ^ (lane % 8) = 2 s ^ x.
  const int x = ((lane >> 3) & 1) ^ (lane & 7);
  const uint32_t b_lane =
      smem + (lane >> 4) * (S::kTileRows * kKC * 2) + (lane & 7) * (kKC * 2);
  // Trellis words of this lane's window in the warp's first tile of step 0.
  uint32_t w_lane[Ring::kLoadWords];
  int shift;
  {
    const LaneWindow<S::kBitsX2> window(lane);
    shift = window.shift;
#pragma unroll
    for (int i = 0; i < Ring::kLoadWords; ++i) {
      w_lane[i] =
          smem + S::kStageW + TPW * warp * S::kTileBytes + 4 * window.word[i];
    }
  }

#pragma unroll
  for (int s = 0; s < S::kStages - 1; ++s) {
    load();
    copy_commit();
  }
  uint32_t read = 0;
  for (int chunk = 0; chunk < S::kChunks; ++chunk) {
    copy_wait<S::kStages - 2>();
    __syncthreads();
    if (chunk + S::kStages - 1 < S::kChunks) load();
    copy_commit();
#pragma unroll
    for (int step = 0; step < kSteps; ++step) {
      uint32_t a[TPW][4];
#pragma unroll
      for (int j = 0; j < TPW; ++j) {
        uint32_t words[Ring::kLoadWords];
#pragma unroll
        for (int i = 0; i < Ring::kLoadWords; ++i) {
          words[i] =
              lds32(w_lane[i] + read + step * (S::kBlockTiles * S::kTileBytes) +
                    j * S::kTileBytes);
        }
        decode_fragment<S::kBitsX2>(words, shift, a[j]);
      }
      const uint32_t bsrc = b_lane + read + ((x ^ (2 * step)) * 16);
      uint32_t b[2][4];
      ldmatrix4_at(bsrc, b[0]);
#pragma unroll
      for (int g = 0; g < kG; ++g) {
        if (g + 1 < kG)
          ldmatrix4_at(bsrc + (g + 1) * 8 * kKC * 2, b[(g + 1) & 1]);
        const uint32_t (&bg)[4] = b[g & 1];
#pragma unroll
        for (int j = 0; j < TPW; ++j) {
          // The k16 low products (C = 0, lo magnitude) seed the high MMA, so
          // one truncating accumulate at hi magnitude per k16; the k16
          // partial is then summed in FP32 (RN) in increasing k.
          // kPlanes (VLLM_EXL3_MOE_PREFILL_ONE_PLANE, prefill families only):
          // 1 -- the FP16 hi plane alone (activation rounded once to FP16), one
          // MMA per k16; 2 -- both planes as independent MMAs added in FP32 RN
          // (no lo -> hi dependency); 0 -- the shipped lo-seeded pair.
          float d[4];
          if constexpr (kPlanes == 1) {
            mma(a[j], bg[0], bg[1], zero, d);
          } else if constexpr (kPlanes == 2) {
            float dl[4];
            mma(a[j], bg[2], bg[3], zero, dl);
            mma(a[j], bg[0], bg[1], zero, d);
#pragma unroll
            for (int i = 0; i < 4; ++i) d[i] = __fadd_rn(d[i], dl[i]);
          } else {
            mma(a[j], bg[2], bg[3], zero, d);
            mma(a[j], bg[0], bg[1], d, d);
          }
#pragma unroll
          for (int i = 0; i < 4; ++i)
            sum[j][g][i] = __fadd_rn(sum[j][g][i], d[i]);
        }
      }
    }
    read = next_stage<S>(read);
  }

  // Lane holds output columns 16 tile + lane / 4 (+ 8) of rows 8 g + 2 (lane %
  // 4) (+ 1).
  const int t = lane & 3;
#pragma unroll
  for (int j = 0; j < TPW; ++j) {
    const int col = (cb * S::kBlockTiles + TPW * warp + j) * 16 + (lane >> 2);
#pragma unroll
    for (int g = 0; g < kG; ++g) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int r = 8 * g + 2 * t + c;
        if (r < rows) {
          const int64_t pos = first + r;
          const float scale = p.unscale[pos * 3 + S::kFirst + pi];
          int64_t row = pos;
          if constexpr (S::kProjections == 1 && S::kFirst < 2)
            row = p.row_map[pos];
          float* out = p.out + (row * S::kProjections + pi) * S::kN + col;
          out[0] = __fmul_rn(sum[j][g][c], scale);
          out[8] = __fmul_rn(sum[j][g][2 + c], scale);
        }
      }
    }
  }
}

// Block items: (route tile, projection, column block of 16 kBlockTiles), column
// blocks fastest so concurrently running blocks share the tile's operand rows
// in L2.
template <class S, int kPlanes = 0>
__global__ void __launch_bounds__(kThreads, 2) gemm(Gemm p) {
  extern __shared__ __align__(16) unsigned char gemm_shared[];
  const uint32_t smem = smem_address(gemm_shared);
  const int64_t items = int64_t(p.table[1]) * S::kProjections * S::kColBlocks;
  const int64_t item = blockIdx.x;
  if (item >= items) return;
  const int64_t tile_index = item / (S::kProjections * S::kColBlocks);
  const int rest = static_cast<int>(item % (S::kProjections * S::kColBlocks));
  const int pi = rest / S::kColBlocks;
  const int cb = rest % S::kColBlocks;
  const int4 tile = reinterpret_cast<const int4*>(
      p.table + Layout::tiles_offset(p.assignments))[tile_index];
  const int rows = tile.z;
  const unsigned char* trellis = reinterpret_cast<const unsigned char*>(
      p.pointers[int64_t(tile.x) * 9 + (S::kFirst + pi) * 3]);
  const __half* operand =
      p.operand + int64_t(tile.y) * S::kRowHalves + int64_t(pi) * 2 * S::kK;

  // Per-thread cp.async work of a stage: the stage vectors v = tid + kThreads j
  // (16 bytes each). The first 16 kTileRows are operand rows: vector v is
  // chunk v % 8 of row (v / 8) % kTileRows of part v / (8 kTileRows) (the row
  // halves, then their lo residuals), stored at shared offset
  // part * (kTileRows kKC 2) + row (kKC 2) with chunk c at offset
  // (c ^ (row & 7)) 16-byte units; the rest are the block's trellis tiles.
  // Sources advance by one chunk per load. Operand rows past the tile's 8-row
  // groups are never issued; rows past the tile inside them are zero-filled
  // from row 0.
  const int tid = threadIdx.x;
  static_assert(
      kThreads == 256 && (S::kTileRows == 64 || S::kTileRows == 8) && kKC == 64,
      "operand copy layout");
  constexpr int kStepVectors = S::kBlockTiles * S::kTileBytes / 16;
  constexpr int kWVectors = kSteps * kStepVectors;
  constexpr int kOperandVectors = S::kTileRows * 16;
  constexpr int kStageVectors = kOperandVectors + kWVectors;
  constexpr int kCopies = (kStageVectors + kThreads - 1) / kThreads;
  constexpr int kOperandAdvance = kKC * 2;  // bytes per thread and chunk
  const int padded_rows = (rows + 7) & ~7;
  const char* src[kCopies];
  uint32_t dst[kCopies];
  int size[kCopies];  // -1: a slot this thread never copies
  int advance[kCopies];
#pragma unroll
  for (int j = 0; j < kCopies; ++j) {
    const int v = tid + j * kThreads;
    // A dead slot addresses the last live vector; its copy is never issued.
    const int at = v < kStageVectors ? v : kStageVectors - 1;
    if (at < kOperandVectors) {
      const int part = at / (8 * S::kTileRows);
      const int r = (at / 8) % S::kTileRows;
      const int c = at & 7;
      src[j] = reinterpret_cast<const char*>(
          operand + int64_t(r < rows ? r : 0) * S::kRowHalves + part * S::kK +
          c * 8);
      dst[j] = smem + part * (S::kTileRows * (kKC * 2)) + r * (kKC * 2) +
               ((c ^ (r & 7)) * 16);
      size[j] =
          v < kOperandVectors && r < padded_rows ? (r < rows ? 16 : 0) : -1;
      advance[j] = kOperandAdvance;
    } else {
      const int w = at - kOperandVectors;
      src[j] = reinterpret_cast<const char*>(trellis) +
               (int64_t(w / kStepVectors) * S::kTilesN +
                int64_t(cb) * S::kBlockTiles) *
                   S::kTileBytes +
               (w % kStepVectors) * 16;
      dst[j] = smem + S::kStageW + w * 16;
      size[j] = v < kStageVectors ? 16 : -1;
      advance[j] = S::kWChunk;
    }
  }
  uint32_t stage = 0;

  auto load = [&]() {
#pragma unroll
    for (int j = 0; j < kCopies; ++j) {
      if (size[j] >= 0) {
        copy16_to(dst[j] + stage, src[j], size[j]);
        src[j] += advance[j];
      }
    }
    stage = next_stage<S>(stage);
  };

  if constexpr (S::kTileRows == 8) {
    // A decode route tile holds at most one 8-row group (route above).
    run_tile<1, S, kPlanes>(p, smem, load, rows, tile.y, pi, cb);
  } else {
    switch ((rows + 7) >> 3) {
#define EXL3_GROUPS(G)                                   \
  case G:                                                \
    run_tile<G, S, kPlanes>(p, smem, load, rows, tile.y, pi, cb); \
    break;
      EXL3_GROUPS(1)
      EXL3_GROUPS(2) EXL3_GROUPS(3) EXL3_GROUPS(4) EXL3_GROUPS(5) EXL3_GROUPS(6)
          EXL3_GROUPS(7) EXL3_GROUPS(8)
#undef EXL3_GROUPS
              default : break;
    }
  }
}

// Eight consecutive sums: one 16-byte store per 8 BF16 when the output base
// is 16-byte aligned, element stores otherwise.
__device__ __forceinline__ void store8_sum(__nv_bfloat16* out, bool vec,
                                           const float* sum) {
  if (vec) {
    uint32_t w[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      w[q] =
          uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(sum[2 * q]))) |
          (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(sum[2 * q + 1])))
           << 16);
    }
    *reinterpret_cast<uint4*>(out) = make_uint4(w[0], w[1], w[2], w[3]);
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = __float2bfloat16_rn(sum[j]);
  }
}
// One warp per (token, 512-column span; lane group g holds 128-block g of the
// span): each local route's down output transform (H128, svh) in the shipped
// FP32 operations. Decode preserves the original FP32 routing-weight FMA per
// route before BF16 rounding; prefill retains its FP64 sum and FP32 boundary.
template <typename Out, class F>
__global__ void combine(const float* down, const float* weights,
                        Routing routing, const int* table, int64_t assignments,
                        int64_t rows, Out* out) {
  constexpr int kHidden = F::kHidden;
  constexpr int kSpans = kHidden / 512;
  using Accumulator = std::conditional_t<F::kTileRows == 8, float, double>;
  const int lane = threadIdx.x & 31;
  const int lane_col = 128 * (lane >> 3) + 8 * (lane & 7);
  const int* position = table + Layout::position_offset(assignments);
  const bool out_vec = aligned16(out);
  // One task per warp (no grid-stride loop), so the shuffles are in provably
  // converged code (a task loop makes ptxas call a WARPSYNC stub per shuffle).
  {
    const int64_t task =
        int64_t(blockIdx.x) * kOperandWarps + (threadIdx.x >> 5);
    if (task >= rows * kSpans) return;
    const int64_t row = task / kSpans;
    const int base = static_cast<int>(task % kSpans) * 512 + lane_col;
    Accumulator sum[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) sum[i] = 0;
    for (int64_t route = 0; route < routing.topk; ++route) {
      const int64_t assignment = row * routing.topk + route;
      const int pos = position[assignment];
      if (pos < 0) continue;
      const half* svh =
          expert_part(routing, local_expert(routing, assignment), 2, 2);
      const bool vec = aligned16(svh);
      const Accumulator weight = weights[assignment];
      const float* src = down + int64_t(pos) * kHidden + base;
      float v[16];
      load8(src, v);
      load8(src + 64, v + 8);
      hadamard128_group(v, lane);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float s[8];
        load8(svh + base + 64 * h, vec, s);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float value = __fmul_rn(v[8 * h + j], s[j]);
          if constexpr (F::kTileRows == 8) {
            sum[8 * h + j] = __fmaf_rn(value, weight, sum[8 * h + j]);
          } else {
            sum[8 * h + j] =
                fma(static_cast<double>(value), weight, sum[8 * h + j]);
          }
        }
      }
    }
    float rounded[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if constexpr (F::kTileRows == 8) {
        rounded[i] = sum[i];
      } else {
        rounded[i] = __double2float_rn(sum[i]);
      }
    }
    Out* dst = out + row * kHidden + base;
    store8_sum(dst, out_vec, rounded);
    store8_sum(dst + 64, out_vec, rounded + 8);
  }
}

// combine for the 8-row decode families: one warp per (token, 128-block) in
// the hadamard128_warp layout (lane holds column 128 b + lane + 32 i; the
// same H128 bitwise as the lane-group layout), with every route's loads
// issued before the first FMA. The routes are still accumulated one after
// another in route order with the same FP32 multiply and FMA, so each output
// is bitwise combine's. Routes of other experts' ranks (position < 0) run the
// H128 on zeros to keep the shuffles converged and add nothing.
template <typename Out, class F, int kTopk>
__global__ void __launch_bounds__(32 * kOperandWarps)
    combine_block(const float* down, const float* weights, Routing routing,
                  const int* table, int64_t assignments, int64_t rows,
                  Out* out) {
  static_assert(F::kTileRows == 8, "the FP32 decode combine");
  constexpr int kHidden = F::kHidden;
  constexpr int kBlocks = kHidden / 128;
  const int lane = threadIdx.x & 31;
  const int* position = table + Layout::position_offset(assignments);
  const int64_t task = int64_t(blockIdx.x) * kOperandWarps + (threadIdx.x >> 5);
  if (task >= rows * kBlocks) return;  // warp-uniform
  const int64_t row = task / kBlocks;
  const int base = static_cast<int>(task % kBlocks) * 128 + lane;
  float v[kTopk][4];
  float s[kTopk][4];
  float weight[kTopk];
  bool live[kTopk];
#pragma unroll
  for (int route = 0; route < kTopk; ++route) {
    const int64_t assignment = row * kTopk + route;
    const int pos = position[assignment];
    live[route] = pos >= 0;
    weight[route] = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) v[route][i] = s[route][i] = 0.0f;
    if (live[route]) {
      const half* svh =
          expert_part(routing, local_expert(routing, assignment), 2, 2);
      const float* src = down + int64_t(pos) * kHidden + base;
      weight[route] = weights[assignment];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[route][i] = src[32 * i];
        s[route][i] = __half2float(svh[base + 32 * i]);
      }
    }
  }
  float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int route = 0; route < kTopk; ++route) {
    hadamard128_warp(v[route], lane);
    if (live[route]) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float value = __fmul_rn(v[route][i], s[route][i]);
        sum[i] = __fmaf_rn(value, weight[route], sum[i]);
      }
    }
  }
  Out* dst = out + row * kHidden + base;
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[32 * i] = __float2bfloat16_rn(sum[i]);
}

unsigned int warp_grid(int64_t tasks) {
  return static_cast<unsigned int>(
      std::min<int64_t>((tasks + kOperandWarps - 1) / kOperandWarps, 65535));
}

template <class S, int kPlanes = 0>
void launch_gemm(const Gemm& p, int64_t max_items, cudaStream_t stream) {
  static const bool configured = [] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gemm<S, kPlanes>, cudaFuncAttributeMaxDynamicSharedMemorySize, S::kSmemBytes));
    return true;
  }();
  (void)configured;
  TORCH_CHECK(max_items <= std::numeric_limits<int>::max(),
              "EXL3 MoE prefill grid too large");
  gemm<S, kPlanes><<<static_cast<unsigned int>(max_items), kThreads, S::kSmemBytes,
            stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

bool served_shape(int64_t hidden, int64_t intermediate) {
  return (hidden == QwenFamily::kHidden &&
          intermediate == QwenFamily::kIntermediate) ||
         (hidden == GlmFamily::kHidden &&
          intermediate == GlmFamily::kIntermediate);
}

// The shapes moe_out_decode_tc serves (the GLM families; the bits_x2 check
// lives in run_decode).
bool served_decode_shape(int64_t hidden, int64_t intermediate) {
  return hidden == GlmFamily::kHidden &&
         intermediate == GlmFamily::kIntermediate;
}

int64_t workspace_elements(int64_t assignments, int64_t hidden,
                           int64_t intermediate) {
  const int64_t operand =
      assignments * 2 * hidden;  // FP16 gate/up rows (later down)
  const int64_t gate_up = assignments * 2 * intermediate;
  const int64_t unscale = (assignments * 3 + 3) / 4 * 4;
  return operand + gate_up + unscale + (Layout::ints(assignments) + 3) / 4 * 4;
}

// VLLM_EXL3_MOE_PREFILL_ONE_PLANE=1 (default off): prefill tile families run
// their expert GEMMs on the activation's FP16 hi plane alone (one MMA per k16,
// see run_tile). Changes arithmetic -- the activation is rounded once to FP16
// instead of keeping ~22 significand bits -- so it is a lab variant under the
// TC admission policy. 8-row decode families are never affected.
// "1": one plane; "pair": both planes as independent MMAs. Returns kPlanes.
template <class F>
int prefill_one_plane() {
  if constexpr (F::kTileRows <= 8) {
    return 0;
  } else {
    static const int mode = [] {
      const char* v = std::getenv("VLLM_EXL3_MOE_PREFILL_ONE_PLANE");
      if (v == nullptr) return 0;
      const std::string s(v);
      if (s == "1") return 1;
      if (s == "pair") return 2;
      TORCH_CHECK(s.empty() || s == "0",
                  "VLLM_EXL3_MOE_PREFILL_ONE_PLANE must be 0, 1 or pair");
      return 0;
    }();
    return mode;
  }
}

// The block-per-task operand staging and the per-128-block combine (bitwise
// the warp kernels' results) serve the 8-row decode families, whose launches
// are a few warps wide. VLLM_EXL3_MOE_DECODE_GLUE=warp keeps the warp kernels
// (A/B and fallback switch; read once).
constexpr int64_t kDecodeGlueTopk = 8;
template <class F>
constexpr bool decode_glue_blocks() {
  return F::kTileRows == 8 && F::kIntermediate != 5 * 128 &&
         F::kHidden / 512 <= 32 && F::kIntermediate / 128 <= 32;
}
inline bool decode_glue_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("VLLM_EXL3_MOE_DECODE_GLUE");
    if (v == nullptr) return true;
    const std::string s(v);
    TORCH_CHECK(s.empty() || s == "block" || s == "warp",
                "VLLM_EXL3_MOE_DECODE_GLUE must be block or warp");
    return s != "warp";
  }();
  return enabled;
}

// VLLM_EXL3_MOE_DECODE_TC_STAGES (lab, read once): cp.async ring depth of the
// 8-row decode GEMMs, 3 (default, the shipped depth), 4, 6 or 8. Bitwise
// identical results at every depth (Staged).
inline int decode_gemm_stages() {
  static const int stages = [] {
    const char* v = std::getenv("VLLM_EXL3_MOE_DECODE_TC_STAGES");
    if (v == nullptr || *v == 0) return 3;
    const std::string s(v);
    TORCH_CHECK(s == "3" || s == "4" || s == "6" || s == "8",
                "VLLM_EXL3_MOE_DECODE_TC_STAGES must be 3, 4, 6 or 8");
    return std::stoi(s);
  }();
  return stages;
}
template <class S>
void launch_decode_gemm(const Gemm& p, int64_t items, cudaStream_t stream) {
  switch (decode_gemm_stages()) {
    case 4: launch_gemm<Staged<S, 4>>(p, items, stream); break;
    case 6: launch_gemm<Staged<S, 6>>(p, items, stream); break;
    case 8: launch_gemm<Staged<S, 8>>(p, items, stream); break;
    default: launch_gemm<S>(p, items, stream); break;
  }
}

template <class F, bool kDownOnly = false>
void run_family(const __nv_bfloat16* x, const void* ids, bool ids64,
                const float* weights, const void* expert_map, bool map64,
                int64_t global_experts, const int64_t* pointers,
                const int64_t* metadata, int64_t local_experts, int64_t rows,
                int64_t topk, __nv_bfloat16* out, float* workspace,
                cudaStream_t stream, const float* activated_down = nullptr) {
  constexpr int64_t kHidden = F::kHidden;
  constexpr int64_t kIntermediate = F::kIntermediate;
  constexpr int64_t kTileRows = F::kTileRows;
  const int64_t assignments = rows * topk;
  Routing routing{ids,           expert_map, pointers, metadata, global_experts,
                  local_experts, topk,       ids64,    map64};
  // Workspace: operand region (gate/up split rows; then the down split rows
  // and the down results), gate/up results, 2^-e per (position, projection),
  // routing table.
  __half* operand = reinterpret_cast<__half*>(workspace);
  float* down_result = workspace + assignments * kIntermediate;
  float* gate_up = workspace + assignments * 2 * kHidden;
  float* unscale = gate_up + assignments * 2 * kIntermediate;
  int* table = reinterpret_cast<int*>(unscale + (assignments * 3 + 3) / 4 * 4);
  route<F><<<1, kRouteThreads, (3 * local_experts + 1) * sizeof(int), stream>>>(
      routing, assignments, table);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t max_tiles = std::min<int64_t>(
      assignments, assignments / kTileRows +
                       std::min<int64_t>(local_experts, assignments) + 1);
  using GateUp = typename F::GateUpShape;
  using Down = typename F::DownShape;
  Gemm p{};
  p.table = table;
  p.assignments = assignments;
  p.pointers = pointers;
  p.operand = operand;
  p.unscale = unscale;
  p.out = gate_up;
  if constexpr (kDownOnly) {
    stage_down_operand<F>
        <<<warp_grid(assignments), 32 * kOperandWarps, 0, stream>>>(
            activated_down, table, operand, unscale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    if constexpr (decode_glue_blocks<F>()) {
      if (decode_glue_enabled()) {
        gate_up_operand_block<F>
            <<<static_cast<unsigned int>(2 * assignments), 32 * (kHidden / 512),
               0, stream>>>(x, routing, table, operand, unscale);
      } else {
        gate_up_operand<F>
            <<<warp_grid(2 * assignments), 32 * kOperandWarps, 0, stream>>>(
                x, routing, table, operand, unscale);
      }
    } else {
      gate_up_operand<F>
          <<<warp_grid(2 * assignments), 32 * kOperandWarps, 0, stream>>>(
              x, routing, table, operand, unscale);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    {
      const int64_t items = max_tiles * GateUp::kProjections * GateUp::kColBlocks;
      if constexpr (kTileRows == 8) {
        launch_decode_gemm<GateUp>(p, items, stream);
      } else {
        switch (prefill_one_plane<F>()) {
          case 1: launch_gemm<GateUp, 1>(p, items, stream); break;
          case 2: launch_gemm<GateUp, 2>(p, items, stream); break;
          default: launch_gemm<GateUp>(p, items, stream); break;
        }
      }
    }
    if constexpr (decode_glue_blocks<F>()) {
      if (decode_glue_enabled()) {
        down_operand_block<F>
            <<<static_cast<unsigned int>(assignments), 32 * (kIntermediate / 128),
               0, stream>>>(routing, table, gate_up, operand, unscale);
      } else {
        down_operand<F>
            <<<warp_grid(assignments), 32 * kOperandWarps, 0, stream>>>(
                routing, table, gate_up, operand, unscale);
      }
    } else {
      down_operand<F><<<warp_grid(assignments), 32 * kOperandWarps, 0, stream>>>(
          routing, table, gate_up, operand, unscale);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  p.out = down_result;
  {
    const int64_t items = max_tiles * Down::kProjections * Down::kColBlocks;
    if constexpr (kTileRows == 8) {
      launch_decode_gemm<Down>(p, items, stream);
    } else {
      switch (prefill_one_plane<F>()) {
        case 1: launch_gemm<Down, 1>(p, items, stream); break;
        case 2: launch_gemm<Down, 2>(p, items, stream); break;
        default: launch_gemm<Down>(p, items, stream); break;
      }
    }
  }
  if constexpr (decode_glue_blocks<F>()) {
    if (topk == kDecodeGlueTopk && decode_glue_enabled()) {
      const int64_t tasks = rows * (kHidden / 128);
      combine_block<__nv_bfloat16, F, kDecodeGlueTopk>
          <<<static_cast<unsigned int>((tasks + kOperandWarps - 1) /
                                       kOperandWarps),
             32 * kOperandWarps, 0, stream>>>(down_result, weights, routing,
                                              table, assignments, rows, out);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
  }
  // combine: one warp per task, no grid cap.
  const int64_t combine_blocks =
      (rows * (kHidden / 512) + kOperandWarps - 1) / kOperandWarps;
  TORCH_CHECK(combine_blocks <= std::numeric_limits<int>::max(),
              "EXL3 MoE prefill grid too large");
  const unsigned int grid = static_cast<unsigned int>(combine_blocks);
  combine<__nv_bfloat16, F><<<grid, 32 * kOperandWarps, 0, stream>>>(
      down_result, weights, routing, table, assignments, rows, out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run(const __nv_bfloat16* x, const void* ids, bool ids64,
         const float* weights, const void* expert_map, bool map64,
         int64_t global_experts, int64_t hidden, int64_t intermediate,
         int64_t bits_x2, const int64_t* pointers, const int64_t* metadata,
         int64_t local_experts, int64_t rows, int64_t topk, __nv_bfloat16* out,
         float* workspace, cudaStream_t stream) {
  if (hidden == GlmFamily::kHidden &&
      intermediate == GlmFamily::kIntermediate) {
    if (bits_x2 == GlmFamily::kBitsX2) {
      run_family<GlmFamily>(x, ids, ids64, weights, expert_map, map64,
                            global_experts, pointers, metadata, local_experts,
                            rows, topk, out, workspace, stream);
      return;
    }
    TORCH_CHECK(
        bits_x2 == Glm35Family::kBitsX2,
        "EXL3 MoE prefill runs the GLM 5-bit or 3.5-bit family, not bits_x2 ",
        bits_x2);
    run_family<Glm35Family>(x, ids, ids64, weights, expert_map, map64,
                            global_experts, pointers, metadata, local_experts,
                            rows, topk, out, workspace, stream);
    return;
  }
  TORCH_CHECK(hidden == QwenFamily::kHidden &&
                  intermediate == QwenFamily::kIntermediate &&
                  bits_x2 == QwenFamily::kBitsX2,
              "EXL3 MoE prefill run of an unserved shape or format");
  run_family<QwenFamily>(x, ids, ids64, weights, expert_map, map64,
                         global_experts, pointers, metadata, local_experts,
                         rows, topk, out, workspace, stream);
}

// The decode schedule: the GLM families with 8-row route tiles. The
// arithmetic, the kernels and the workspace layout are run_family's; only the
// Family (hence the GEMM shapes' rows per tile and the route tile size)
// differs from the prefill entry.
template <bool kDownOnly = false>
void run_decode(const __nv_bfloat16* x, const void* ids, bool ids64,
                const float* weights, const void* expert_map, bool map64,
                int64_t global_experts, int64_t hidden, int64_t intermediate,
                int64_t bits_x2, const int64_t* pointers,
                const int64_t* metadata, int64_t local_experts, int64_t rows,
                int64_t topk, __nv_bfloat16* out, float* workspace,
                cudaStream_t stream, const float* activated_down = nullptr) {
  TORCH_CHECK(
      hidden == GlmFamily::kHidden && intermediate == GlmFamily::kIntermediate,
      "EXL3 MoE decode tensor-core run of an unserved shape");
  TORCH_CHECK(rows <= kMoeDecodeTcMaxRows,
              "EXL3 MoE decode tensor-core schedule serves at most ",
              kMoeDecodeTcMaxRows, " rows, not ", rows);
  if (bits_x2 == GlmFamily::kBitsX2) {
    run_family<GlmDecodeFamily, kDownOnly>(
        x, ids, ids64, weights, expert_map, map64, global_experts, pointers,
        metadata, local_experts, rows, topk, out, workspace, stream,
        activated_down);
    return;
  }
  TORCH_CHECK(
      bits_x2 == Glm35Family::kBitsX2,
      "EXL3 MoE decode runs the GLM 5-bit or 3.5-bit family, not bits_x2 ",
      bits_x2);
  run_family<Glm35DecodeFamily, kDownOnly>(
      x, ids, ids64, weights, expert_map, map64, global_experts, pointers,
      metadata, local_experts, rows, topk, out, workspace, stream,
      activated_down);
}

// One gate/up projection on the decode tensor-core schedule for the mixed role
// modes. It runs route<F>, stages the operand for `projection` with the full
// path's operand arithmetic, and launches the single-projection GEMM with the
// table's sorted list as the output row map, so the raw projection log lands in
// assignment order. It writes no down rows and no combine: the caller finishes
// with the scalar activation and moe_decode_tc_down_finish. All buffers live in
// caller-reserved regions of the decode workspace (operand, unscale, table).
template <class F>
void run_projection(const __nv_bfloat16* x, const void* ids, bool ids64,
                    const void* expert_map, bool map64, int64_t global_experts,
                    const int64_t* pointers, const int64_t* metadata,
                    int64_t local_experts, int64_t rows, int64_t topk,
                    int projection, float* operand_region, float* unscale,
                    int* table, float* out_assignment, cudaStream_t stream) {
  constexpr int64_t kTileRows = F::kTileRows;
  const int64_t assignments = rows * topk;
  Routing routing{ids,           expert_map, pointers, metadata, global_experts,
                  local_experts, topk,       ids64,    map64};
  __half* operand = reinterpret_cast<__half*>(operand_region);
  route<F><<<1, kRouteThreads, (3 * local_experts + 1) * sizeof(int), stream>>>(
      routing, assignments, table);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  gate_up_operand_proj<F>
      <<<warp_grid(assignments), 32 * kOperandWarps, 0, stream>>>(
          x, routing, table, operand, unscale, projection);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t max_tiles = std::min<int64_t>(
      assignments, assignments / kTileRows +
                       std::min<int64_t>(local_experts, assignments) + 1);
  Gemm p{};
  p.table = table;
  p.assignments = assignments;
  p.pointers = pointers;
  p.operand = operand;
  p.unscale = unscale;
  p.out = out_assignment;
  p.row_map = table + Layout::sorted_offset();
  if (projection == 0) {
    using S = typename F::TcGateShape;
    launch_gemm<S>(p, max_tiles * S::kProjections * S::kColBlocks, stream);
  } else {
    using S = typename F::TcUpShape;
    launch_gemm<S>(p, max_tiles * S::kProjections * S::kColBlocks, stream);
  }
}
}  // namespace moe_prefill

int64_t moe_workspace_size_prefill(int64_t max_rows, int64_t hidden_size,
                                   int64_t intermediate_size, int64_t topk,
                                   int64_t local_experts) {
  TORCH_CHECK(max_rows >= 0 && topk >= 0 && local_experts >= 0,
              "EXL3 rows/topk/local_experts must be nonnegative");
  TORCH_CHECK(moe_prefill::served_shape(hidden_size, intermediate_size),
              "EXL3 MoE prefill covers hidden 2560 and intermediate 640, or "
              "hidden 4096 and ",
              "intermediate 2048");
  if (local_experts == 0) return 0;
  const int64_t assignments = checked_mul(max_rows, topk);
  TORCH_CHECK(assignments <= std::numeric_limits<int>::max(),
              "EXL3 MoE has too many routed assignments");
  return workspace_elements(moe_prefill::workspace_elements(
      assignments, hidden_size, intermediate_size));
}

void moe_out_prefill(const torch::Tensor& x, const torch::Tensor& topk_ids,
                     const torch::Tensor& topk_weights,
                     const torch::Tensor& packed_ptrs,
                     const torch::Tensor& packed_meta,
                     const torch::Tensor& expert_map, int64_t intermediate_size,
                     int64_t bits_x2, torch::Tensor& out,
                     torch::Tensor& workspace) {
  TORCH_CHECK(x.is_cuda(), "EXL3 x must be CUDA");
  check_tensor(x, x, at::kBFloat16, "x");
  check_integer(topk_ids, x, "topk_ids");
  check_integer(expert_map, x, "expert_map");
  check_tensor(topk_weights, x, at::kFloat, "topk_weights");
  check_tensor(packed_ptrs, x, at::kLong, "packed_ptrs");
  check_tensor(packed_meta, x, at::kLong, "packed_meta");
  check_tensor(out, x, at::kBFloat16, "out");
  check_tensor(workspace, x, at::kFloat, "workspace");
  TORCH_CHECK(x.dim() == 2 && out.dim() == 2 && topk_ids.dim() == 2 &&
                  topk_weights.dim() == 2 && expert_map.dim() == 1 &&
                  packed_ptrs.dim() == 3 && packed_meta.dim() == 3,
              "EXL3 MoE tensor ranks mismatch");
  const int64_t rows = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t topk = topk_ids.size(1);
  const int64_t experts = packed_ptrs.size(0);
  TORCH_CHECK(topk_ids.size(0) == rows &&
                  topk_weights.sizes() == topk_ids.sizes() &&
                  out.sizes() == x.sizes(),
              "EXL3 MoE row/topk/output geometry mismatch");
  TORCH_CHECK(packed_ptrs.size(1) == 3 && packed_ptrs.size(2) == 3 &&
                  packed_meta.size(0) == experts && packed_meta.size(1) == 3 &&
                  packed_meta.size(2) == 4,
              "EXL3 MoE pointer/metadata geometry mismatch");
  TORCH_CHECK(experts > 0 && experts <= kMoePrefillMaxExperts,
              "EXL3 MoE prefill needs 1..", kMoePrefillMaxExperts,
              " local experts");
  TORCH_CHECK(rows == 0 || rows >= kMoePrefillMinRows,
              "EXL3 MoE prefill serves calls of at least ", kMoePrefillMinRows,
              " rows");
  const int64_t required = moe_workspace_size_prefill(
      rows, hidden, intermediate_size, topk, experts);
  TORCH_CHECK(workspace.numel() >= required, "EXL3 MoE workspace is too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0,
              "EXL3 MoE workspace must be 16-byte aligned");
  check_mutations(
      out, workspace,
      {&x, &topk_ids, &topk_weights, &packed_ptrs, &packed_meta, &expert_map});
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  moe_prefill::run(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      topk_ids.data_ptr(), topk_ids.scalar_type() == at::kLong,
      topk_weights.data_ptr<float>(), expert_map.data_ptr(),
      expert_map.scalar_type() == at::kLong, expert_map.numel(), hidden,
      intermediate_size, bits_x2, packed_ptrs.data_ptr<int64_t>(),
      packed_meta.data_ptr<int64_t>(), experts, rows, topk,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      workspace.data_ptr<float>(), stream);
}

int64_t moe_workspace_size_decode_tc(int64_t max_rows, int64_t hidden_size,
                                     int64_t intermediate_size, int64_t topk,
                                     int64_t local_experts) {
  TORCH_CHECK(max_rows >= 0 && topk >= 0 && local_experts >= 0,
              "EXL3 rows/topk/local_experts must be nonnegative");
  TORCH_CHECK(moe_prefill::served_decode_shape(hidden_size, intermediate_size),
              "EXL3 MoE decode covers hidden 4096 and intermediate 2048");
  if (local_experts == 0) return 0;
  const int64_t assignments = checked_mul(max_rows, topk);
  TORCH_CHECK(assignments <= std::numeric_limits<int>::max(),
              "EXL3 MoE has too many routed assignments");
  return workspace_elements(moe_prefill::workspace_elements(
      assignments, hidden_size, intermediate_size));
}

// The decode gate/up role mode, read once per process (the master decode
// tensor-core opt-in routes here; the mode selects which gate/up projections
// the tensor-core schedule owns). `full` is the default. Any other value is a
// hard error, and the obsolete single-flag hybrid is refused outright so a
// stale environment cannot silently select the unqualified full tensor-core
// schedule.
enum class DecodeTcMode { kFull, kDown, kGateDown, kUpDown };

DecodeTcMode decode_tc_mode() {
  static const DecodeTcMode mode = [] {
    const char* legacy = std::getenv("VLLM_EXL3_MOE_DECODE_TC_DOWN_ONLY");
    TORCH_CHECK(
        !legacy || !*legacy || std::strcmp(legacy, "0") == 0,
        "VLLM_EXL3_MOE_DECODE_TC_DOWN_ONLY is obsolete and refused; set "
        "VLLM_EXL3_MOE_DECODE_TC_MODE=down for the scalar-gate/up, TC-down "
        "hybrid (full, gate_down or up_down otherwise)");
    const char* value = std::getenv("VLLM_EXL3_MOE_DECODE_TC_MODE");
    if (!value || !*value || std::strcmp(value, "full") == 0)
      return DecodeTcMode::kFull;
    if (std::strcmp(value, "down") == 0) return DecodeTcMode::kDown;
    if (std::strcmp(value, "gate_down") == 0) return DecodeTcMode::kGateDown;
    if (std::strcmp(value, "up_down") == 0) return DecodeTcMode::kUpDown;
    TORCH_CHECK(false, "Unsupported VLLM_EXL3_MOE_DECODE_TC_MODE '", value,
                "'; expected full, down, gate_down or up_down");
    return DecodeTcMode::kFull;  // unreachable
  }();
  return mode;
}

void moe_out_decode_tc(const torch::Tensor& x, const torch::Tensor& topk_ids,
                       const torch::Tensor& topk_weights,
                       const torch::Tensor& packed_ptrs,
                       const torch::Tensor& packed_meta,
                       const torch::Tensor& expert_map,
                       int64_t intermediate_size, int64_t bits_x2,
                       torch::Tensor& out, torch::Tensor& workspace) {
  const DecodeTcMode mode = decode_tc_mode();
  switch (mode) {
    case DecodeTcMode::kFull:
      break;  // the all-tensor-core schedule below
    case DecodeTcMode::kDown:
    case DecodeTcMode::kGateDown:
    case DecodeTcMode::kUpDown: {
      TORCH_CHECK(bits_x2 == moe_prefill::GlmFamily::kBitsX2 ||
                      bits_x2 == moe_prefill::Glm35Family::kBitsX2,
                  "EXL3 MoE hybrid decode requires a GLM format");
      const double limit = bits_x2 == moe_prefill::GlmFamily::kBitsX2
                               ? moe_prefill::GlmFamily::kSwigluLimit
                               : moe_prefill::Glm35Family::kSwigluLimit;
      if (mode == DecodeTcMode::kDown) {
        moe_out_decode_tc_down(x, topk_ids, topk_weights, packed_ptrs,
                               packed_meta, expert_map, intermediate_size,
                               bits_x2, out, workspace, limit);
      } else {
        moe_out_decode_tc_mixed(x, topk_ids, topk_weights, packed_ptrs,
                                packed_meta, expert_map, intermediate_size,
                                bits_x2, out, workspace, limit,
                                mode == DecodeTcMode::kGateDown);
      }
      return;
    }
  }
  TORCH_CHECK(x.is_cuda(), "EXL3 x must be CUDA");
  check_tensor(x, x, at::kBFloat16, "x");
  check_integer(topk_ids, x, "topk_ids");
  check_integer(expert_map, x, "expert_map");
  check_tensor(topk_weights, x, at::kFloat, "topk_weights");
  check_tensor(packed_ptrs, x, at::kLong, "packed_ptrs");
  check_tensor(packed_meta, x, at::kLong, "packed_meta");
  check_tensor(out, x, at::kBFloat16, "out");
  check_tensor(workspace, x, at::kFloat, "workspace");
  TORCH_CHECK(x.dim() == 2 && out.dim() == 2 && topk_ids.dim() == 2 &&
                  topk_weights.dim() == 2 && expert_map.dim() == 1 &&
                  packed_ptrs.dim() == 3 && packed_meta.dim() == 3,
              "EXL3 MoE tensor ranks mismatch");
  const int64_t rows = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t topk = topk_ids.size(1);
  const int64_t experts = packed_ptrs.size(0);
  TORCH_CHECK(topk_ids.size(0) == rows &&
                  topk_weights.sizes() == topk_ids.sizes() &&
                  out.sizes() == x.sizes(),
              "EXL3 MoE row/topk/output geometry mismatch");
  TORCH_CHECK(packed_ptrs.size(1) == 3 && packed_ptrs.size(2) == 3 &&
                  packed_meta.size(0) == experts && packed_meta.size(1) == 3 &&
                  packed_meta.size(2) == 4,
              "EXL3 MoE pointer/metadata geometry mismatch");
  TORCH_CHECK(experts > 0 && experts <= kMoeDecodeTcMaxExperts,
              "EXL3 MoE decode needs 1..", kMoeDecodeTcMaxExperts,
              " local experts");
  TORCH_CHECK(rows == 0 || rows <= kMoeDecodeTcMaxRows,
              "EXL3 MoE decode tensor-core serves calls of at most ",
              kMoeDecodeTcMaxRows, " rows");
  const int64_t required = moe_workspace_size_decode_tc(
      rows, hidden, intermediate_size, topk, experts);
  TORCH_CHECK(workspace.numel() >= required, "EXL3 MoE workspace is too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0,
              "EXL3 MoE workspace must be 16-byte aligned");
  check_mutations(
      out, workspace,
      {&x, &topk_ids, &topk_weights, &packed_ptrs, &packed_meta, &expert_map});
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  moe_prefill::run_decode(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      topk_ids.data_ptr(), topk_ids.scalar_type() == at::kLong,
      topk_weights.data_ptr<float>(), expert_map.data_ptr(),
      expert_map.scalar_type() == at::kLong, expert_map.numel(), hidden,
      intermediate_size, bits_x2, packed_ptrs.data_ptr<int64_t>(),
      packed_meta.data_ptr<int64_t>(), experts, rows, topk,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      workspace.data_ptr<float>(), stream);
}

void moe_decode_tc_down_finish(const torch::Tensor& topk_ids,
                               const torch::Tensor& topk_weights,
                               const torch::Tensor& packed_ptrs,
                               const torch::Tensor& packed_meta,
                               const torch::Tensor& expert_map,
                               int64_t intermediate_size, int64_t bits_x2,
                               torch::Tensor& out, torch::Tensor& workspace) {
  const int64_t rows = out.size(0);
  const int64_t hidden = out.size(1);
  const int64_t topk = topk_ids.size(1);
  float* scratch = workspace.data_ptr<float>();
  // Scalar routing/activation are finished before this stream-ordered stage.
  // TC routing uses its own tail; staging reads [A*H, A*(H+I)) and writes
  // [0, A*I). Only the following GEMM reuses the activated-input region.
  moe_prefill::run_decode<true>(
      nullptr, topk_ids.data_ptr(), topk_ids.scalar_type() == at::kLong,
      topk_weights.data_ptr<float>(), expert_map.data_ptr(),
      expert_map.scalar_type() == at::kLong, expert_map.numel(), hidden,
      intermediate_size, bits_x2, packed_ptrs.data_ptr<int64_t>(),
      packed_meta.data_ptr<int64_t>(), packed_ptrs.size(0), rows, topk,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()), scratch,
      at::cuda::getCurrentCUDAStream(out.get_device()).stream(),
      scratch + rows * topk * hidden);
}

// Internal one-projection entry for the mixed role modes (see moe_prefill.h).
// The tensor-core operand stage lives at the workspace base (the mixed layout's
// operand region); out_assignment, table and unscale are the caller's reserved
// regions. The caller has already validated every tensor.
void moe_decode_tc_projection(
    const torch::Tensor& x, const torch::Tensor& topk_ids,
    const torch::Tensor& expert_map, const torch::Tensor& packed_ptrs,
    const torch::Tensor& packed_meta, int64_t intermediate_size,
    int64_t bits_x2, int projection, torch::Tensor& workspace,
    float* out_assignment, int* table, float* unscale) {
  TORCH_CHECK(projection == 0 || projection == 1,
              "EXL3 MoE decode TC projection is gate (0) or up (1)");
  const int64_t rows = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t topk = topk_ids.size(1);
  const int64_t experts = packed_ptrs.size(0);
  TORCH_CHECK(hidden == moe_prefill::GlmFamily::kHidden &&
                  intermediate_size == moe_prefill::GlmFamily::kIntermediate,
              "EXL3 MoE decode TC projection serves the GLM decode shape only");
  TORCH_CHECK(rows <= kMoeDecodeTcMaxRows,
              "EXL3 MoE decode TC projection serves at most ",
              kMoeDecodeTcMaxRows, " rows");
  TORCH_CHECK(bits_x2 == moe_prefill::GlmFamily::kBitsX2 ||
                  bits_x2 == moe_prefill::Glm35Family::kBitsX2,
              "EXL3 MoE decode TC projection runs a GLM format, not bits_x2 ",
              bits_x2);
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const __nv_bfloat16* xptr =
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  float* operand_region = workspace.data_ptr<float>();
  const bool ids64 = topk_ids.scalar_type() == at::kLong;
  const bool map64 = expert_map.scalar_type() == at::kLong;
  if (bits_x2 == moe_prefill::GlmFamily::kBitsX2) {
    moe_prefill::run_projection<moe_prefill::GlmDecodeFamily>(
        xptr, topk_ids.data_ptr(), ids64, expert_map.data_ptr(), map64,
        expert_map.numel(), packed_ptrs.data_ptr<int64_t>(),
        packed_meta.data_ptr<int64_t>(), experts, rows, topk, projection,
        operand_region, unscale, table, out_assignment, stream);
    return;
  }
  moe_prefill::run_projection<moe_prefill::Glm35DecodeFamily>(
      xptr, topk_ids.data_ptr(), ids64, expert_map.data_ptr(), map64,
      expert_map.numel(), packed_ptrs.data_ptr<int64_t>(),
      packed_meta.data_ptr<int64_t>(), experts, rows, topk, projection,
      operand_region, unscale, table, out_assignment, stream);
}

}  // namespace exl3
