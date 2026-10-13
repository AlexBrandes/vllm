// EXL3 routed-expert MoE on the tensor cores for decode-sized calls of one
// served shape: hidden 2560, intermediate 640, every expert matrix bits_x2 10
// with the mul1 codebook. The caller (vllm/sm80/kernels/exl3/quant.py) takes
// this operator only for that configuration and at most 64 rows; every other
// call stays on moe_out (moe.cu), which this file does not change.
//
// Arithmetic differs from moe_out: the expert GEMMs run m16n8k16 FP16 MMAs on
// exactly decoded weights and a hi/lo FP16 split of the scaled activations,
// and the input, activation and combine stages run in FP64. Results are
// deterministic and a row never depends on other rows.

#include "ops.h"
#include "common.h"
#include "codec_moe_v2.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>

namespace exl3 {
namespace moe_v2 {
namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kRouteThreads = 1024;
// Route kernel shared memory: three int arrays per local expert.
constexpr int64_t kMaxRouteExperts = 4096;

// Integers the route kernel writes into the workspace each call:
//   [0] local assignment count, [1] row-tile count, [2..3] unused,
//   sorted[assignments]: local assignment ids grouped by local expert,
//   tiles[assignments] (16-byte aligned): {expert, first sorted index, rows, 0},
//   experts[assignments]: the local expert of each sorted entry.
struct RouteLayout {
  static constexpr int64_t kHeader = 4;
  __host__ __device__ static constexpr int64_t sorted_offset() { return kHeader; }
  __host__ __device__ static constexpr int64_t tiles_offset(int64_t assignments) {
    return kHeader + (assignments + 3) / 4 * 4;
  }
  __host__ __device__ static constexpr int64_t experts_offset(int64_t assignments) {
    return tiles_offset(assignments) + 4 * assignments;
  }
  __host__ __device__ static constexpr int64_t ints(int64_t assignments) {
    return experts_offset(assignments) + assignments;
  }
};

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

__device__ __forceinline__ int64_t integer_at(const void* ptr, bool wide, int64_t i) {
  return wide ? static_cast<const int64_t*>(ptr)[i]
              : static_cast<const int32_t*>(ptr)[i];
}
__device__ __forceinline__ int64_t local_expert(const Routing& routing, int64_t assignment) {
  const int64_t global = integer_at(routing.ids, routing.ids64, assignment);
  if (global < 0 || global >= routing.global_experts) return -1;
  const int64_t local = integer_at(routing.expert_map, routing.map64, global);
  return local >= 0 && local < routing.local_experts ? local : -1;
}
// [trellis, suh, svh] of one projection of one local expert.
__device__ __forceinline__ const int64_t* expert_pointers(const Routing& routing,
                                                          int64_t expert, int projection) {
  return routing.pointers + expert * 9 + projection * 3;
}
__device__ __forceinline__ const half* expert_suh(const Routing& routing, int64_t expert,
                                                  int projection) {
  return reinterpret_cast<const half*>(expert_pointers(routing, expert, projection)[1]);
}
__device__ __forceinline__ const half* expert_svh(const Routing& routing, int64_t expert,
                                                  int projection) {
  return reinterpret_cast<const half*>(expert_pointers(routing, expert, projection)[2]);
}

unsigned int warp_blocks(int64_t tasks) {
  return launch_blocks((tasks + kWarpsPerBlock - 1) / kWarpsPerBlock);
}

// Exclusive block-wide scan of two counters; also returns the totals.
__device__ __forceinline__ void block_scan2(int a, int b, int& a_before, int& b_before,
                                            int& a_total, int& b_total) {
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

// Groups the local assignments by local expert into tiles of up to
// block_rows rows (one block, every call) and records each sorted entry's
// expert. Order inside an expert is arbitrary: each assignment's result
// depends only on its own row, never on its position or neighbours.
__global__ void __launch_bounds__(kRouteThreads)
route_assignments(Routing routing, int64_t assignments, int block_rows, int* table) {
  extern __shared__ int route_shared[];
  const int experts = static_cast<int>(routing.local_experts);
  int* cursor = route_shared;              // counts, then fill cursors
  int* row_begin = cursor + experts;
  int* tile_begin = row_begin + experts;   // experts + 1 entries
  for (int e = threadIdx.x; e < experts; e += blockDim.x) cursor[e] = 0;
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
    tiles += (cursor[e] + block_rows - 1) / block_rows;
  }
  int row_at, tile_at, row_total, tile_total;
  block_scan2(rows, tiles, row_at, tile_at, row_total, tile_total);
  for (int e = e0; e < e1; ++e) {
    const int count = cursor[e];
    row_begin[e] = row_at;
    tile_begin[e] = tile_at;
    cursor[e] = row_at;
    row_at += count;
    tile_at += (count + block_rows - 1) / block_rows;
  }
  if (threadIdx.x == 0) {
    tile_begin[experts] = tile_total;
    table[0] = row_total;
    table[1] = tile_total;
  }
  __syncthreads();
  int4* tile_table = reinterpret_cast<int4*>(table + RouteLayout::tiles_offset(assignments));
  for (int t = threadIdx.x; t < tile_total; t += blockDim.x) {
    // Last expert whose first tile is at or before t (empty experts share
    // their successor's start, so the search lands on the owner).
    int lo = 0;
    int hi = experts;
    while (hi - lo > 1) {
      const int mid = (lo + hi) >> 1;
      if (tile_begin[mid] <= t) lo = mid; else hi = mid;
    }
    const int first = row_begin[lo] + (t - tile_begin[lo]) * block_rows;
    const int end = lo + 1 < experts ? row_begin[lo + 1] : row_total;
    tile_table[t] = make_int4(lo, first, min(block_rows, end - first), 0);
  }
  int* sorted = table + RouteLayout::sorted_offset();
  int* sorted_experts = table + RouteLayout::experts_offset(assignments);
  for (int64_t a = threadIdx.x; a < assignments; a += blockDim.x) {
    const int64_t e = local_expert(routing, a);
    if (e >= 0) {
      const int at = atomicAdd(&cursor[e], 1);
      sorted[at] = static_cast<int>(a);
      sorted_experts[at] = static_cast<int>(e);
    }
  }
}

__device__ __forceinline__ double silu(double g) { return g / __dadd_rn(1.0, exp(-g)); }

// Lane order of a K-long row (MMA B operand): k step k / 16, then lane t's k
// values 2t, 2t + 1, 2t + 8, 2t + 9 (t = (k % 8) / 2).
__device__ __forceinline__ int64_t lane_position(int64_t k) {
  const int kk = static_cast<int>(k & 15);
  return (k & ~int64_t(15)) + ((kk & 7) >> 1) * 4 + ((kk >> 3) << 1) + (kk & 1);
}

// The tensor-core B operand of one row and K slice held by a warp (lane holds
// the slice's k = lane + 32 i): e per (row, slice) puts max |v| 2^e in
// [2^14, 2^15) (2^e and 2^-e stay normal FP32), hi = f16(v 2^e) (exact
// scaling), lo = f16(v 2^e - hi), stored at lane_position(k) of the slice's
// high and low FP16 rows; 2^-e goes to *unscale.
template <int kVals>
__device__ __forceinline__ void store_lane_halves(const float (&v)[kVals], int lane, __half* hi,
                                                  __half* lo, float* unscale) {
  float peak = 0.0f;
#pragma unroll
  for (int i = 0; i < kVals; ++i) peak = fmaxf(peak, fabsf(v[i]));
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, d));
  const int biased = static_cast<int>(__float_as_uint(peak) >> 23);
  const int e = peak == 0.0f ? 0 : min(141 - biased, 126);
  const float scale = __uint_as_float(uint32_t(127 + e) << 23);
#pragma unroll
  for (int i = 0; i < kVals; ++i) {
    const int64_t at = lane_position(lane + 32 * i);
    const float y = __fmul_rn(v[i], scale);
    const __half h = __float2half_rn(y);
    hi[at] = h;
    lo[at] = __float2half_rn(__fsub_rn(y, __half2float(h)));
  }
  if (lane == 0) *unscale = __uint_as_float(uint32_t(127 - e) << 23);
}

// One warp per (local assignment, 128-column block): gate and up output
// transforms (H128, svh), SiLU(gate) * up, down suh, then the down H128, all
// in FP64, rounded once to FP32 and written as the down GEMM's B operand
// (per assignment [hi | lo] FP16 rows of ipad halves, one 2^-e per
// (assignment, 128-column slice)). Gate and up arrive as kSlices K-slice
// partials (slice_stride apart), summed in increasing slice order; every
// load is issued before the first sum (one memory round trip).
template <int kSlices>
__global__ void activate_down_input(Routing routing, const int* table, int64_t assignments,
                                    int64_t intermediate, int64_t ipad, const float* gate,
                                    const float* up, int64_t slice_stride, __half* halves,
                                    float* unscale) {
  const int lane = threadIdx.x & 31;
  const int* sorted = table + RouteLayout::sorted_offset();
  const int* sorted_experts = table + RouteLayout::experts_offset(assignments);
  const int64_t blocks = ipad / 128;
  const int64_t tasks = int64_t(table[0]) * blocks;
  for (int64_t task = int64_t(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
       task < tasks; task += int64_t(gridDim.x) * kWarpsPerBlock) {
    const int64_t assignment = sorted[task / blocks];
    const int64_t expert = sorted_experts[task / blocks];
    const int64_t base = (task % blocks) * 128 + lane;
    const half* gate_svh = expert_svh(routing, expert, 0);
    const half* up_svh = expert_svh(routing, expert, 1);
    const half* down_suh = expert_suh(routing, expert, 2);
    float gs[kSlices][4];
    float us[kSlices][4];
#pragma unroll
    for (int s = 0; s < kSlices; ++s) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        gs[s][i] = gate[s * slice_stride + assignment * ipad + base + 32 * i];
        us[s][i] = up[s * slice_stride + assignment * ipad + base + 32 * i];
      }
    }
    double g[4];
    double u[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      g[i] = gs[0][i];
      u[i] = us[0][i];
#pragma unroll
      for (int s = 1; s < kSlices; ++s) {
        g[i] = __dadd_rn(g[i], double(gs[s][i]));
        u[i] = __dadd_rn(u[i], double(us[s][i]));
      }
    }
    hadamard128_warp_f64(g, lane);
    hadamard128_warp_f64(u, lane);
    double v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t col = base + 32 * i;
      const double gv = __dmul_rn(g[i], double(__half2float(gate_svh[col])));
      const double uv = __dmul_rn(u[i], double(__half2float(up_svh[col])));
      v[i] = 0.0;
      if (col < intermediate) {
        v[i] = __dmul_rn(__dmul_rn(silu(gv), uv), double(__half2float(down_suh[col])));
      }
    }
    hadamard128_warp_f64(v, lane);
    float f[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] = __double2float_rn(v[i]);
    const int64_t block = task % blocks;
    __half* hi = halves + assignment * 2 * ipad + block * 128;
    store_lane_halves<4>(f, lane, hi, hi + ipad, unscale + assignment * blocks + block);
  }
}

// One warp per (token, 128-column block): each local route's down output
// (the sum of kSlices K-slice partials, slice_stride apart, in increasing
// slice order) transform (H128, svh), then exactly one routing-weight FMA per
// route in route order, all in FP64, then FP32 rounding and BF16 rounding of
// that FP32 value. Two local routes are loaded at a time so their memory
// round trips overlap.
template <int kSlices>
__global__ void combine_routes(const float* down, const float* weights, Routing routing,
                               int64_t rows, int64_t hidden, int64_t hpad,
                               int64_t slice_stride, __nv_bfloat16* out) {
  const int lane = threadIdx.x & 31;
  const int64_t blocks = hpad / 128;
  for (int64_t task = int64_t(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
       task < rows * blocks; task += int64_t(gridDim.x) * kWarpsPerBlock) {
    const int64_t row = task / blocks;
    const int64_t base = (task % blocks) * 128 + lane;
    double sum[4] = {0.0, 0.0, 0.0, 0.0};
    for (int64_t first = 0; first < routing.topk; first += 32) {
      // Lane r looks up route first + r; the warp then walks the local routes
      // in order, two at a time.
      const int64_t mine = row * routing.topk + first + lane;
      const int64_t my_expert = first + lane < routing.topk ? local_expert(routing, mine) : -1;
      const half* my_svh = my_expert >= 0 ? expert_svh(routing, my_expert, 2) : nullptr;
      const float my_weight = my_expert >= 0 ? weights[mine] : 0.0f;
      unsigned local = __ballot_sync(0xffffffffu, my_expert >= 0);
      auto fetch = [&](int r, float (&v)[kSlices][4], float (&scale)[4]) {
        const int64_t at = (row * routing.topk + first + r) * hpad + base;
        const half* svh = reinterpret_cast<const half*>(
            __shfl_sync(0xffffffffu, reinterpret_cast<uintptr_t>(my_svh), r));
#pragma unroll
        for (int s = 0; s < kSlices; ++s) {
#pragma unroll
          for (int i = 0; i < 4; ++i) v[s][i] = down[s * slice_stride + at + 32 * i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) scale[i] = __half2float(svh[base + 32 * i]);
      };
      auto add_route = [&](int r, const float (&v)[kSlices][4], const float (&scale)[4]) {
        const double weight = __shfl_sync(0xffffffffu, my_weight, r);
        double x[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          x[i] = v[0][i];
#pragma unroll
          for (int s = 1; s < kSlices; ++s) x[i] = __dadd_rn(x[i], double(v[s][i]));
        }
        hadamard128_warp_f64(x, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          sum[i] = __fma_rn(__dmul_rn(x[i], double(scale[i])), weight, sum[i]);
        }
      };
      while (local != 0) {
        const int r0 = __ffs(local) - 1;
        local &= local - 1;
        float v0[kSlices][4], s0[4];
        fetch(r0, v0, s0);
        if (local != 0) {
          const int r1 = __ffs(local) - 1;
          local &= local - 1;
          float v1[kSlices][4], s1[4];
          fetch(r1, v1, s1);
          add_route(r0, v0, s0);
          add_route(r1, v1, s1);
        } else {
          add_route(r0, v0, s0);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t col = base + 32 * i;
      if (col < hidden) out[row * hidden + col] = __float2bfloat16_rn(__double2float_rn(sum[i]));
    }
  }
}

// ---------------------------------------------------------------------------
// Lane-tile expert GEMM (bits_x2 10, codebook 2).
//
// An item is one warp's kNT output n16 tiles x one K slice of one route tile;
// no shared memory and no block barriers. The warp decodes its own 16x16
// trellis tiles straight into m16n8k16 A fragments (A = W^T: 16 output
// columns x 16 k; decode_lane_fragment_mul1) and multiplies them on the tensor
// cores (FP16 in, FP32 out, C = 0 per k16 step) with B = 16 k x 8 columns: per
// group of four route rows, columns 0..3 hold the rows' high FP16 parts and
// 4..7 the low parts of their activations scaled by 2^e per (row, K slice)
// (store_lane_halves; produced once per call by lane_gate_up_input and
// activate_down_input, read straight from global memory). The k16 results are
// summed in FP32 in increasing k per K slice, high and low apart; the slice
// partial (hi + lo) 2^-e goes to the workspace and the consumer kernel sums
// the slices in increasing slice order (FP64). Everything is fixed-order: a
// row's result never depends on other rows.
constexpr int kLaneRows = 8;                    // route rows per tile
constexpr int64_t kLaneMaxAssignments = 1024;   // routed rows the workspace layout covers
constexpr int kLaneThreads = 256;
constexpr int kLaneAhead = 3;                   // k steps prefetched to L2

struct LaneProblem {
  const int* table;
  const int64_t* pointers;   // [local][3 projections][trellis, suh, svh]
  const int64_t* metadata;   // [local][3 projections][padded_in, padded_out, bits_x2, codebook]
  int64_t assignments;       // routing table capacity
  int k_tiles;               // K / 16
  int n_tiles;               // N / 16
  int projections;           // 2: gate (out0) and up (out1); 1: down (out0)
  int first_projection;
  float* out0;               // [slice][assignment][N] partials
  float* out1;
  int64_t slice_stride;
  // B operand: per (assignment, projection) row [hi | lo] FP16 rows of K
  // halves in lane order, and 2^-e per (assignment, projection, K slice).
  const __half* halves;
  const float* unscale;
};

__device__ __forceinline__ void mma_f16_f32(const uint32_t (&a)[4], uint2 b, float (&d)[4]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y), "f"(0.0f));
}

// One warp: kNT output n16 tiles x kKT k16 steps of one route tile with
// kGroups n8 row groups (row_of: lane r holds tile row r's assignment), k
// outer: the kNT tiles of a k step are adjacent in the trellis (immediate
// offsets from three per-lane word pointers); the next k step's tiles and B
// fragments are in flight while this one decodes.
template <int kGroups, int kNT, int kKT, bool kAligned>
__device__ __forceinline__ void lane_tiles(const LaneProblem& p, int pi, int slice, int row_of,
                                           int rows, const uint16_t* trellis, int64_t tiles_n,
                                           int n0, int lane, const LaneWindow<10>& window,
                                           float* out) {
  using Ring = TileRing<10>;
  constexpr int kTileWords = Ring::kTileHalfwords / 2;
  const int kt0 = slice * kKT;
  const int slices = p.k_tiles / kKT;
  const int64_t row_halves = 2 * int64_t(p.k_tiles) * 16;
  const int t = lane & 3;
  const int gq = lane >> 2;
  // B source of this lane: column gq of each group = hi (gq < 4) or lo of
  // row 4g + gq % 4, k values 2t, 2t + 1, 2t + 8, 2t + 9 of each k step.
  const uint2* bsrc[kGroups];
  bool bok[kGroups];
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
    const int r = 4 * g + (gq & 3);
    bok[g] = r < rows;
    const int64_t assignment = __shfl_sync(0xffffffffu, row_of, bok[g] ? r : 0);
    bsrc[g] = reinterpret_cast<const uint2*>(
                  p.halves + (assignment * p.projections + pi) * row_halves +
                  (gq >> 2) * (row_halves / 2) + kt0 * 16) + t;
  }
  // 2^-e of the rows this lane stores: 4g + 2t + c (lanes t < 2).
  float unscale[kGroups][2];
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int r = 4 * g + 2 * (t & 1) + c;
      const int64_t assignment = __shfl_sync(0xffffffffu, row_of, r < rows ? r : 0);
      unscale[g][c] = __ldg(p.unscale + (assignment * p.projections + pi) * slices + slice);
    }
  }
  const uint16_t* step = trellis + (int64_t(kt0) * tiles_n + n0) * Ring::kTileHalfwords;
  const int64_t stride = tiles_n * Ring::kTileHalfwords;
  const uint32_t* wp[Ring::kLoadWords];
#pragma unroll
  for (int i = 0; i < Ring::kLoadWords; ++i) {
    wp[i] = reinterpret_cast<const uint32_t*>(step) + window.word[i];
  }
  uint32_t words[kNT][Ring::kLoadWords];
  auto fetch = [&](int j) {
    if constexpr (kAligned) {
#pragma unroll
      for (int i = 0; i < Ring::kLoadWords; ++i) words[j][i] = __ldg(wp[i] + j * kTileWords);
    } else {
      load_lane_words<10>(step + j * Ring::kTileHalfwords, window, false, words[j]);
    }
  };
#pragma unroll
  for (int j = 0; j < kNT; ++j) fetch(j);
  uint2 bnext[kGroups];
#pragma unroll
  for (int g = 0; g < kGroups; ++g) bnext[g] = bok[g] ? __ldg(bsrc[g]) : make_uint2(0u, 0u);
  float acc[kNT][kGroups][4];
#pragma unroll
  for (int j = 0; j < kNT; ++j) {
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[j][g][i] = 0.0f;
    }
  }
  // L2 prefetch kLaneAhead k steps ahead: lane = 128-byte line of the step.
  constexpr int kStepBytes = kNT * Ring::kTileHalfwords * 2;
  const char* ahead = reinterpret_cast<const char*>(step + kLaneAhead * stride) +
                      min(lane * 128, kStepBytes - 1);
#pragma unroll 1
  for (int kt = 0; kt < kKT; ++kt) {
    if (lane * 128 < kStepBytes + 127 && kt + kLaneAhead < kKT) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(ahead));
    }
    ahead += stride * 2;
    const bool more = kt + 1 < kKT;
    uint2 b[kGroups];
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
      b[g] = bnext[g];
      if (more && bok[g]) bnext[g] = __ldg(bsrc[g] + 4 * (kt + 1));
    }
    if constexpr (kAligned) {
#pragma unroll
      for (int i = 0; i < Ring::kLoadWords; ++i) wp[i] += stride / 2;
    } else {
      step += stride;
    }
#pragma unroll
    for (int j = 0; j < kNT; ++j) {
      uint32_t a[4];
      decode_lane_fragment_mul1<10>(words[j], window.shift, a);
      if (more) fetch(j);
#pragma unroll
      for (int g = 0; g < kGroups; ++g) {
        float d[4];
        mma_f16_f32(a, b[g], d);
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][g][i] = __fadd_rn(acc[j][g][i], d[i]);
      }
    }
  }
  // Lanes t < 2 hold the high sums of rows 4g + 2t + {0, 1}; lane t + 2 the
  // low sums of the same rows and columns.
  const int64_t n_stride = int64_t(p.n_tiles) * 16;
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int r = 4 * g + 2 * (t & 1) + c;
      const int64_t assignment = __shfl_sync(0xffffffffu, row_of, r < rows ? r : 0);
#pragma unroll
      for (int j = 0; j < kNT; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int i = 2 * h + c;
          const float low = __shfl_xor_sync(0xffffffffu, acc[j][g][i], 2);
          if (t < 2 && r < rows) {
            out[assignment * n_stride + (n0 + j) * 16 + gq + 8 * h] =
                __fmul_rn(__fadd_rn(acc[j][g][i], low), unscale[g][c]);
          }
        }
      }
    }
  }
}

// Warp items: (route tile, projection, K slice, column group of kNT n16
// tiles), column groups fastest so a block's warps share B rows in L1.
template <int kNT, int kKT>
__global__ void __launch_bounds__(kLaneThreads, 2)
moe_lane_kernel(LaneProblem p) {
  const int lane = threadIdx.x & 31;
  const LaneWindow<10> window(lane);
  const int slices = p.k_tiles / kKT;
  const int groups_n = p.n_tiles / kNT;
  const int64_t per_tile = int64_t(p.projections) * slices * groups_n;
  const int64_t items = int64_t(p.table[1]) * per_tile;
  const int* sorted = p.table + RouteLayout::sorted_offset();
  const int4* tiles = reinterpret_cast<const int4*>(p.table + RouteLayout::tiles_offset(p.assignments));
  const int64_t warps = int64_t(gridDim.x) * (kLaneThreads / 32);
  for (int64_t item = int64_t(blockIdx.x) * (kLaneThreads / 32) + (threadIdx.x >> 5);
       item < items; item += warps) {
    const int64_t tile_index = item / per_tile;
    int rest = static_cast<int>(item % per_tile);
    const int column_group = rest % groups_n;
    rest /= groups_n;
    const int slice = rest % slices;
    const int pi = rest / slices;
    const int projection = p.first_projection + pi;
    const int4 tile = tiles[tile_index];
    const int rows = tile.z;
    const int64_t* ptrs = p.pointers + int64_t(tile.x) * 9 + projection * 3;
    const int64_t* meta = p.metadata + int64_t(tile.x) * 12 + projection * 4;
    if (meta[2] != 10 || meta[3] != 2) __trap();
    const uint16_t* trellis = reinterpret_cast<const uint16_t*>(ptrs[0]);
    const int64_t tiles_n = meta[1] / 16;
    const int row_of = lane < rows ? sorted[tile.y + lane] : 0;
    const int n0 = column_group * kNT;
    const bool aligned = (reinterpret_cast<uintptr_t>(trellis) & 3) == 0;
    float* out = (pi == 0 ? p.out0 : p.out1) + slice * p.slice_stride;
#define EXL3_LANE_GROUPS(G)                                                                   \
    if (aligned) {                                                                            \
      lane_tiles<G, kNT, kKT, true>(p, pi, slice, row_of, rows, trellis, tiles_n, n0, lane,   \
                                    window, out);                                             \
    } else {                                                                                  \
      lane_tiles<G, kNT, kKT, false>(p, pi, slice, row_of, rows, trellis, tiles_n, n0, lane,  \
                                     window, out);                                            \
    }
    if (rows <= 4) {
      EXL3_LANE_GROUPS(1);
    } else {
      EXL3_LANE_GROUPS(2);
    }
#undef EXL3_LANE_GROUPS
  }
}

// One warp per (sorted entry, projection, K slice) of gate/up: x * suh and
// H128 in FP64 per 128 block, rounded once to FP32, then the tensor-core B
// operand of the slice (store_lane_halves).
template <int kSliceK>
__global__ void lane_gate_up_input(const __nv_bfloat16* x, Routing routing, const int* table,
                                   int64_t assignments, int64_t hidden, int64_t hpad,
                                   __half* halves, float* unscale) {
  constexpr int kVals = kSliceK / 32;
  const int lane = threadIdx.x & 31;
  const int* sorted = table + RouteLayout::sorted_offset();
  const int* sorted_experts = table + RouteLayout::experts_offset(assignments);
  const int64_t slices = hpad / kSliceK;
  const int64_t tasks = int64_t(table[0]) * 2 * slices;
  for (int64_t task = int64_t(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
       task < tasks; task += int64_t(gridDim.x) * kWarpsPerBlock) {
    const int64_t pos = task / (2 * slices);
    const int projection = static_cast<int>((task / slices) % 2);
    const int64_t slice = task % slices;
    const int64_t assignment = sorted[pos];
    const half* suh = expert_suh(routing, sorted_experts[pos], projection);
    const __nv_bfloat16* row = x + (assignment / routing.topk) * hidden;
    float v[kVals];
#pragma unroll
    for (int b = 0; b < kSliceK / 128; ++b) {
      double block[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t col = slice * kSliceK + b * 128 + lane + 32 * i;
        block[i] = col < hidden ? __dmul_rn(__bfloat162float(row[col]), __half2float(suh[col]))
                                : 0.0;
      }
      hadamard128_warp_f64(block, lane);
#pragma unroll
      for (int i = 0; i < 4; ++i) v[4 * b + i] = __double2float_rn(block[i]);
    }
    __half* hi = halves + (assignment * 2 + projection) * 2 * hpad + slice * kSliceK;
    store_lane_halves<kVals>(v, lane, hi, hi + hpad,
                             unscale + (assignment * 2 + projection) * slices + slice);
  }
}

// The served shape: hidden 2560, intermediate 640 (padded sizes). Items
// cover 5 n16 tiles; K slices of two H128 blocks (gate/up) and one (down).
constexpr int kLaneColumnTiles = 5;
constexpr int kGateUpSliceTiles = 16;
constexpr int kDownSliceTiles = 8;
constexpr int64_t kLaneHidden = 2560;
constexpr int64_t kLaneIntermediate = 640;
constexpr int kGateUpSlices = kLaneHidden / (16 * kGateUpSliceTiles);
constexpr int kDownSlices = kLaneIntermediate / (16 * kDownSliceTiles);
static_assert(kLaneHidden % (16 * kGateUpSliceTiles) == 0 &&
                  kLaneIntermediate % (16 * kDownSliceTiles) == 0 &&
                  kLaneHidden % (16 * kLaneColumnTiles) == 0 &&
                  kLaneIntermediate % (16 * kLaneColumnTiles) == 0 &&
                  16 * kDownSliceTiles == 128 && (16 * kGateUpSliceTiles) % 128 == 0,
              "column groups and K slices must divide the served shape; down slices are the "
              "activation stage's 128 blocks");

void check_shape(int64_t hpad, int64_t ipad, int64_t assignments) {
  TORCH_CHECK(hpad == kLaneHidden && ipad == kLaneIntermediate,
              "EXL3 MoE v2 supports padded hidden 2560 and intermediate 640 only");
  TORCH_CHECK(assignments <= kLaneMaxAssignments,
              "EXL3 MoE v2 supports at most 1024 routed assignments per call");
}

template <int kKT>
void launch_lane(const LaneProblem& p, int64_t max_tiles, cudaStream_t stream) {
  auto* kernel = &moe_lane_kernel<kLaneColumnTiles, kKT>;
  static const int64_t resident = [kernel] {
    int device = 0;
    int sms = 0;
    int per_sm = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel,
                                                                 kLaneThreads, 0));
    return int64_t(std::max(sms, 1)) * std::max(per_sm, 1);
  }();
  const int64_t per_tile = int64_t(p.projections) * (p.k_tiles / kKT) *
                           (p.n_tiles / kLaneColumnTiles);
  constexpr int64_t kWarps = kLaneThreads / 32;
  const int64_t grid = std::min((max_tiles * per_tile + kWarps - 1) / kWarps, resident);
  if (grid <= 0) return;
  kernel<<<static_cast<unsigned int>(grid), kLaneThreads, 0, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// FP32 workspace elements before the routing table: gate, up and down slice
// partials; the gate/up and down B operands (FP16 hi and lo rows) and their
// 2^-e per (row, slice), rounded up to whole 16-byte groups.
int64_t lane_workspace(int64_t assignments, int64_t hpad, int64_t ipad) {
  const int64_t gate_up = checked_mul(2 * kGateUpSlices, ipad);
  const int64_t down = checked_mul(kDownSlices, hpad);
  const int64_t operands = checked_add(checked_mul(2, hpad), ipad);   // FP16 hi + lo
  const int64_t unscale = (assignments * (2 * kGateUpSlices + kDownSlices) + 3) / 4 * 4;
  return checked_add(
      checked_mul(assignments, checked_add(checked_add(gate_up, down), operands)), unscale);
}

}  // namespace
}  // namespace moe_v2

int64_t moe_workspace_size_v2(int64_t max_rows, int64_t hidden_size,
                              int64_t intermediate_size, int64_t topk,
                              int64_t local_experts) {
  using namespace moe_v2;
  TORCH_CHECK(max_rows >= 0 && topk >= 0 && local_experts >= 0,
              "EXL3 rows/topk/local_experts must be nonnegative");
  const int64_t hpad = padded_size(hidden_size);
  const int64_t ipad = padded_size(intermediate_size);
  if (local_experts == 0) return 0;
  const int64_t assignments = checked_mul(max_rows, topk);
  check_shape(hpad, ipad, assignments);
  return workspace_elements(
      checked_add(lane_workspace(assignments, hpad, ipad), RouteLayout::ints(assignments)));
}

void moe_out_v2(const torch::Tensor& x, const torch::Tensor& topk_ids,
                const torch::Tensor& topk_weights,
                const torch::Tensor& packed_ptrs,
                const torch::Tensor& packed_meta,
                const torch::Tensor& expert_map, int64_t intermediate_size,
                torch::Tensor& out, torch::Tensor& workspace) {
  using namespace moe_v2;
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
  const int64_t hpad = padded_size(hidden);
  const int64_t ipad = padded_size(intermediate_size);
  TORCH_CHECK(topk_ids.size(0) == rows && topk_weights.sizes() == topk_ids.sizes() &&
                  out.sizes() == x.sizes(), "EXL3 MoE row/topk/output geometry mismatch");
  TORCH_CHECK(packed_ptrs.size(1) == 3 && packed_ptrs.size(2) == 3 &&
                  packed_meta.size(0) == experts && packed_meta.size(1) == 3 &&
                  packed_meta.size(2) == 4, "EXL3 MoE pointer/metadata geometry mismatch");
  TORCH_CHECK(experts <= kMaxRouteExperts, "EXL3 MoE has too many local experts");
  const int64_t assignments = checked_mul(rows, topk);
  check_shape(hpad, ipad, assignments);
  const int64_t required =
      moe_workspace_size_v2(rows, hidden, intermediate_size, topk, experts);
  TORCH_CHECK(workspace.numel() >= required, "EXL3 MoE workspace is too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0,
              "EXL3 MoE workspace must be 16-byte aligned");
  check_mutations(out, workspace,
                  {&x, &topk_ids, &topk_weights, &packed_ptrs, &packed_meta, &expert_map});
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  Routing routing{topk_ids.data_ptr(), expert_map.data_ptr(),
                  packed_ptrs.data_ptr<int64_t>(), packed_meta.data_ptr<int64_t>(),
                  expert_map.numel(), experts, topk,
                  topk_ids.scalar_type() == at::kLong, expert_map.scalar_type() == at::kLong};
  float* input = workspace.data_ptr<float>();
  const float* down_out = input;
  int64_t down_stride = 0;
  if (experts > 0 && assignments > 0) {
    const int64_t gate_up_stride = assignments * ipad;
    down_stride = assignments * hpad;
    float* gate = input;
    float* up = gate + kGateUpSlices * gate_up_stride;
    float* down = up + kGateUpSlices * gate_up_stride;
    // B operands: [assignment][gate, up][hi, lo][hpad] and [assignment][hi, lo][ipad]
    // FP16, then 2^-e per (assignment, projection, slice).
    __half* gate_up_halves = reinterpret_cast<__half*>(down + kDownSlices * down_stride);
    __half* down_halves = gate_up_halves + assignments * 2 * 2 * hpad;
    float* gate_up_unscale = reinterpret_cast<float*>(down_halves + assignments * 2 * ipad);
    float* down_unscale = gate_up_unscale + assignments * 2 * kGateUpSlices;
    // The int4 route tiles need a 16-byte aligned table.
    int* table = reinterpret_cast<int*>(
        gate_up_unscale + (assignments * (2 * kGateUpSlices + kDownSlices) + 3) / 4 * 4);
    down_out = down;
    route_assignments<<<1, kRouteThreads, (3 * experts + 1) * sizeof(int), stream>>>(
        routing, assignments, kLaneRows, table);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    lane_gate_up_input<16 * kGateUpSliceTiles>
        <<<warp_blocks(assignments * 2 * kGateUpSlices), 32 * kWarpsPerBlock, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), routing, table,
            assignments, hidden, hpad, gate_up_halves, gate_up_unscale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // Row tiles: sum over experts of ceil(rows_e / kLaneRows).
    const int64_t max_tiles = std::min<int64_t>(
        assignments, assignments / kLaneRows + std::min<int64_t>(experts, assignments) + 1);
    LaneProblem p{};
    p.table = table;
    p.pointers = routing.pointers;
    p.metadata = routing.metadata;
    p.assignments = assignments;
    p.k_tiles = static_cast<int>(hpad / 16);
    p.n_tiles = static_cast<int>(ipad / 16);
    p.projections = 2;
    p.first_projection = 0;
    p.out0 = gate;
    p.out1 = up;
    p.slice_stride = gate_up_stride;
    p.halves = gate_up_halves;
    p.unscale = gate_up_unscale;
    launch_lane<kGateUpSliceTiles>(p, max_tiles, stream);
    activate_down_input<kGateUpSlices>
        <<<warp_blocks(assignments * (ipad / 128)), 32 * kWarpsPerBlock, 0, stream>>>(
            routing, table, assignments, intermediate_size, ipad, gate, up, gate_up_stride,
            down_halves, down_unscale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    p.k_tiles = static_cast<int>(ipad / 16);
    p.n_tiles = static_cast<int>(hpad / 16);
    p.projections = 1;
    p.first_projection = 2;
    p.out0 = down;
    p.out1 = nullptr;
    p.slice_stride = down_stride;
    p.halves = down_halves;
    p.unscale = down_unscale;
    launch_lane<kDownSliceTiles>(p, max_tiles, stream);
  } else {
    // No local expert or no route: every row is exactly zero.
    routing.topk = 0;
  }
  combine_routes<kDownSlices>
      <<<warp_blocks(rows * (hpad / 128)), 32 * kWarpsPerBlock, 0, stream>>>(
          down_out, topk_weights.data_ptr<float>(), routing, rows, hidden, hpad, down_stride,
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace exl3
