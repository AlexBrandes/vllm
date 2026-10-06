#include "gemm.cuh"
#include "ops.h"

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

namespace exl3 {
namespace {

// Every format the manifest admits: even rates 1..8 bits for both codebooks,
// plus the half-integer rates 1.5/2.5/3.5 bits for mul1.
#define EXL3_FORMATS(X) \
  X(2, 1) X(4, 1) X(6, 1) X(8, 1) X(10, 1) X(12, 1) X(14, 1) X(16, 1) \
  X(2, 2) X(3, 2) X(4, 2) X(5, 2) X(6, 2) X(7, 2) X(8, 2) X(10, 2) \
  X(12, 2) X(14, 2) X(16, 2)

template <int Threads, int BM, int BN, int BK, int TM, int TN, int MinBlocks>
struct GemmConfig {
  static constexpr int kThreads = Threads;
  static constexpr int kBM = BM;
  static constexpr int kBN = BN;
  static constexpr int kBK = BK;
  static constexpr int kTM = TM;
  static constexpr int kTN = TN;
  static constexpr int kMinBlocks = MinBlocks;
  // Activations are stored k-major; rows are read as float4 when TM >= 4.
  static constexpr int kAStride = TM >= 4 ? BM + 4 : BM + 1;
  // BN + 4 is 4 (mod 16): the eight stores of a decoding warp hit 32 banks.
  static constexpr int kWStride = BN + 4;
  static constexpr int kColThreads = BN / TN;
  static constexpr int kFmaThreads = (BM / TM) * kColThreads;
  static constexpr int kTileRows = BK / 16;
  static constexpr int kTileCols = BN / 16;
  static constexpr int kTasksPerThread = kTileRows * kTileCols * 32 / Threads;
  static constexpr int kAVectors = BM * BK / 4;
  static constexpr int kAPerThread = (kAVectors + Threads - 1) / Threads;
  static_assert(Threads % 32 == 0, "whole warps");
  static_assert(kTileRows * kTileCols * 32 % Threads == 0, "even decode split");
  static_assert(kFmaThreads <= Threads, "micro-tiles exceed the block");
  static_assert(BK % 16 == 0 && 128 % BK == 0 && BN % 16 == 0 && 128 % BN == 0,
                "K/N tiles must divide the H128 blocks");
  static_assert((TM == 1 || TM == 4 || TM == 8) && (TN == 1 || TN == 4 || TN == 8),
                "micro-tile widths");
};

using TinyConfig = GemmConfig<128, 4, 32, 64, 1, 1, 4>;
using SmallConfig = GemmConfig<128, 16, 32, 32, 4, 1, 4>;
using MediumConfig = GemmConfig<128, 64, 64, 16, 8, 4, 3>;
using LargeConfig = GemmConfig<256, 128, 128, 16, 8, 8, 2>;

template <class Cfg>
struct GemmSmem {
  float a[2][Cfg::kBK][Cfg::kAStride];
  float w[2][Cfg::kBK][Cfg::kWStride];
  int row[Cfg::kBM];
};

// Offset of micro-tile element i for thread t. Width 8 is split into two
// groups of four, half a block apart, so a warp's float4 reads are contiguous.
template <int TX, int BX>
__device__ __forceinline__ int micro_offset(int t, int i) {
  if constexpr (TX == 8) {
    return (i < 4 ? 0 : BX / 2) + t * 4 + (i & 3);
  } else if constexpr (TX == 4) {
    return t * 4 + i;
  } else {
    return t;
  }
}

template <int TX, int BX>
__device__ __forceinline__ void load_micro(const float* row, int t, float (&v)[TX]) {
  if constexpr (TX == 1) {
    v[0] = row[t];
  } else {
#pragma unroll
    for (int g = 0; g < TX / 4; ++g) {
      const float4 q = *reinterpret_cast<const float4*>(row + micro_offset<TX, BX>(t, 4 * g));
      v[4 * g] = q.x;
      v[4 * g + 1] = q.y;
      v[4 * g + 2] = q.z;
      v[4 * g + 3] = q.w;
    }
  }
}

template <int kBitsX2, int kCodebook, class Cfg>
__device__ __forceinline__ void run_tile(const GemmProblem& p, GemmSmem<Cfg>& s,
                                         const uint16_t* trellis, int64_t tiles_n,
                                         int64_t col0, int rows) {
  using Ring = TileRing<kBitsX2>;
  constexpr int BK = Cfg::kBK;
  constexpr int T = Cfg::kThreads;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const LaneWindow<kBitsX2> window(lane);
  const bool aligned = (reinterpret_cast<uintptr_t>(trellis) & 3) == 0;
  const int chunks = p.k / BK;
  const int64_t tile_col0 = col0 / 16;

  uint32_t words[Cfg::kTasksPerThread][Ring::kLoadWords];
  float4 staged[Cfg::kAPerThread];

  auto fetch = [&](int chunk) {
#pragma unroll
    for (int t = 0; t < Cfg::kTasksPerThread; ++t) {
      const int tile = (tid + t * T) >> 5;
      const int tile_row = tile / Cfg::kTileCols;
      const int tile_col = tile % Cfg::kTileCols;
      const int64_t index = (int64_t(chunk) * Cfg::kTileRows + tile_row) * tiles_n +
                            tile_col0 + tile_col;
      load_lane_words<kBitsX2>(trellis + index * Ring::kTileHalfwords, window, aligned,
                               words[t]);
    }
#pragma unroll
    for (int v = 0; v < Cfg::kAPerThread; ++v) {
      const int index = tid + v * T;
      staged[v] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (index < Cfg::kAVectors) {
        const int r = index / (BK / 4);
        const int k4 = index % (BK / 4);
        const int row = s.row[r];
        if (row >= 0) {
          staged[v] = *reinterpret_cast<const float4*>(
              p.a + int64_t(row) * p.a_stride + int64_t(chunk) * BK + 4 * k4);
        }
      }
    }
  };

  auto stash = [&](int buffer) {
#pragma unroll
    for (int t = 0; t < Cfg::kTasksPerThread; ++t) {
      const int tile = (tid + t * T) >> 5;
      const int tile_row = tile / Cfg::kTileCols;
      const int tile_col = tile % Cfg::kTileCols;
      float value[8];
      decode_lane<kBitsX2, kCodebook>(words[t], window.shift, value);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        s.w[buffer][tile_row * 16 + lane_tile_row(lane, j)]
           [tile_col * 16 + lane_tile_col(lane, j)] = value[j];
      }
    }
#pragma unroll
    for (int v = 0; v < Cfg::kAPerThread; ++v) {
      const int index = tid + v * T;
      if (index < Cfg::kAVectors) {
        const int r = index / (BK / 4);
        const int k4 = index % (BK / 4);
        s.a[buffer][4 * k4][r] = staged[v].x;
        s.a[buffer][4 * k4 + 1][r] = staged[v].y;
        s.a[buffer][4 * k4 + 2][r] = staged[v].z;
        s.a[buffer][4 * k4 + 3][r] = staged[v].w;
      }
    }
  };

  const int tn = tid % Cfg::kColThreads;
  const int tm = tid / Cfg::kColThreads;
  const bool computes = tid < Cfg::kFmaThreads &&
                        micro_offset<Cfg::kTM, Cfg::kBM>(tm, 0) < rows;
  float acc[Cfg::kTM][Cfg::kTN];
#pragma unroll
  for (int i = 0; i < Cfg::kTM; ++i) {
#pragma unroll
    for (int j = 0; j < Cfg::kTN; ++j) acc[i][j] = 0.0f;
  }

  fetch(0);
  stash(0);
  __syncthreads();
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int buffer = chunk & 1;
    const bool more = chunk + 1 < chunks;
    if (more) fetch(chunk + 1);
    if (computes) {
#pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
        float av[Cfg::kTM];
        float wv[Cfg::kTN];
        load_micro<Cfg::kTM, Cfg::kBM>(&s.a[buffer][kk][0], tm, av);
        load_micro<Cfg::kTN, Cfg::kBN>(&s.w[buffer][kk][0], tn, wv);
#pragma unroll
        for (int i = 0; i < Cfg::kTM; ++i) {
#pragma unroll
          for (int j = 0; j < Cfg::kTN; ++j) {
            acc[i][j] = __fmaf_rn(av[i], wv[j], acc[i][j]);
          }
        }
      }
    }
    if (more) stash(buffer ^ 1);
    __syncthreads();
  }

  if (computes) {
#pragma unroll
    for (int i = 0; i < Cfg::kTM; ++i) {
      const int r = micro_offset<Cfg::kTM, Cfg::kBM>(tm, i);
      if (r < rows) {
        float* out = p.c + int64_t(s.row[r]) * p.c_stride + col0;
        if constexpr (Cfg::kTN == 1) {
          out[tn] = acc[i][0];
        } else {
#pragma unroll
          for (int g = 0; g < Cfg::kTN / 4; ++g) {
            *reinterpret_cast<float4*>(out + micro_offset<Cfg::kTN, Cfg::kBN>(tn, 4 * g)) =
                make_float4(acc[i][4 * g], acc[i][4 * g + 1], acc[i][4 * g + 2],
                            acc[i][4 * g + 3]);
          }
        }
      }
    }
  }
}

template <class Cfg>
__global__ void __launch_bounds__(Cfg::kThreads, Cfg::kMinBlocks)
gemm_kernel(GemmProblem p) {
  __shared__ __align__(16) GemmSmem<Cfg> s;
  const int64_t col_tiles = p.n / Cfg::kBN;
  const bool moe = p.route != nullptr;
  const int64_t row_tiles = moe ? p.route[1] : (p.rows + Cfg::kBM - 1) / Cfg::kBM;
  const int64_t items = row_tiles * col_tiles;
  const int* sorted = moe ? p.route + RouteLayout::sorted_offset() : nullptr;
  for (int64_t item = blockIdx.x; item < items; item += gridDim.x) {
    const int64_t row_tile = item / col_tiles;
    const int64_t col0 = (item % col_tiles) * Cfg::kBN;
    const uint16_t* trellis = p.trellis;
    int64_t tiles_n = p.tiles_n;
    int format = p.format;
    int rows;
    if (moe) {
      // Tiles follow the sorted list, so a block never mixes experts.
      const int4 tile = reinterpret_cast<const int4*>(
          p.route + RouteLayout::tiles_offset(p.rows))[row_tile];
      const int64_t* ptrs = p.pointers + int64_t(tile.x) * 9 + p.projection * 3;
      const int64_t* meta = p.metadata + int64_t(tile.x) * 12 + p.projection * 4;
      trellis = reinterpret_cast<const uint16_t*>(ptrs[0]);
      tiles_n = meta[1] / 16;
      format = format_key(static_cast<int>(meta[2]), static_cast<int>(meta[3]));
      rows = tile.z;
      for (int i = threadIdx.x; i < Cfg::kBM; i += Cfg::kThreads) {
        s.row[i] = i < rows ? sorted[tile.y + i] : -1;
      }
    } else {
      const int64_t row0 = row_tile * Cfg::kBM;
      rows = static_cast<int>(std::min<int64_t>(Cfg::kBM, p.rows - row0));
      for (int i = threadIdx.x; i < Cfg::kBM; i += Cfg::kThreads) {
        s.row[i] = i < rows ? static_cast<int>(row0 + i) : -1;
      }
    }
    __syncthreads();
    switch (format) {
#define EXL3_CASE(B, C) \
      case format_key(B, C): run_tile<B, C, Cfg>(p, s, trellis, tiles_n, col0, rows); break;
      EXL3_FORMATS(EXL3_CASE)
#undef EXL3_CASE
      default:
        // The manifest admits only the formats above.
        __trap();
    }
    __syncthreads();
  }
}

template <class Cfg>
void launch(const GemmProblem& problem, int64_t max_items, cudaStream_t stream) {
  if (max_items <= 0) return;
  const int64_t grid = std::min<int64_t>(max_items, std::numeric_limits<int>::max());
  gemm_kernel<Cfg><<<static_cast<unsigned int>(grid), Cfg::kThreads, 0, stream>>>(problem);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

int gemm_block_rows(GemmShape shape) {
  switch (shape) {
    case GemmShape::kTiny: return TinyConfig::kBM;
    case GemmShape::kSmall: return SmallConfig::kBM;
    case GemmShape::kMedium: return MediumConfig::kBM;
    case GemmShape::kLarge: return LargeConfig::kBM;
  }
  TORCH_CHECK(false, "unknown EXL3 GEMM shape");
}

int gemm_block_cols(GemmShape shape) {
  switch (shape) {
    case GemmShape::kTiny: return TinyConfig::kBN;
    case GemmShape::kSmall: return SmallConfig::kBN;
    case GemmShape::kMedium: return MediumConfig::kBN;
    case GemmShape::kLarge: return LargeConfig::kBN;
  }
  TORCH_CHECK(false, "unknown EXL3 GEMM shape");
}

namespace {
// Test-only override of the shape heuristic, so a harness can run every
// block shape at every row count. -1 selects the heuristic.
std::atomic<int> shape_override{-1};
}  // namespace

void set_gemm_shape_override(int64_t shape) {
  TORCH_CHECK(shape >= -1 && shape <= static_cast<int>(GemmShape::kLarge),
              "EXL3 GEMM shape override must be -1 (heuristic) or 0..3");
  shape_override.store(static_cast<int>(shape));
}

// Issue-cycle model of one block per element of K, for a block holding
// `rows` valid rows. Decoding costs ~65 thread instructions per encoder-lane
// task (8 weights); staging an activation float4 ~5; a warp FFMA occupies
// the FP32 pipe for 2 cycles, plus the shared loads of the micro-tile.
template <class Cfg>
void block_cycles_per_k(int64_t rows, double& issue, double& duration) {
  constexpr int kWarps = Cfg::kThreads / 32;
  constexpr int kRowStep = Cfg::kTM == 1 ? 1 : 4;
  constexpr double kLoads = (Cfg::kTM == 8 ? 2 : 1) + (Cfg::kTN == 8 ? 2 : 1);
  constexpr double kFmaCost = (2.0 * Cfg::kTM * Cfg::kTN + kLoads) / (32.0 * Cfg::kTM * Cfg::kTN);
  const int64_t row_threads = std::min<int64_t>(Cfg::kBM / Cfg::kTM,
                                                (rows + kRowStep - 1) / kRowStep);
  const double fma_warps = std::ceil(double(row_threads * Cfg::kColThreads) / 32.0);
  const double common = Cfg::kBN * 65.0 / 256.0 + Cfg::kBM * 5.0 / 128.0;
  const double fma = double(row_threads * Cfg::kTM) * Cfg::kBN * kFmaCost;
  issue = common + fma;
  // A lone block spreads its warps over the four schedulers; the FMA chain
  // itself needs at least 4 cycles per k.
  duration = std::max(4.0, common / std::min(kWarps, 4) + fma / std::min(fma_warps, 4.0));
}

// Estimated cycles per element of K for a dense launch with this config:
// the slower of one block's own duration and all blocks' issue spread over
// the SMs.
template <class Cfg>
double dense_cycles_per_k(int64_t rows, int64_t n, int sms) {
  const int64_t col_tiles = n / Cfg::kBN;
  const int64_t full = rows / Cfg::kBM;
  const int64_t rest = rows % Cfg::kBM;
  double issue = 0.0;
  double duration = 0.0;
  double block_issue, block_duration;
  if (full > 0) {
    block_cycles_per_k<Cfg>(Cfg::kBM, block_issue, block_duration);
    issue += double(full * col_tiles) * block_issue;
    duration = std::max(duration, block_duration);
  }
  if (rest > 0) {
    block_cycles_per_k<Cfg>(rest, block_issue, block_duration);
    issue += double(col_tiles) * block_issue;
    duration = std::max(duration, block_duration);
  }
  return std::max(duration, issue / (4.0 * std::max(sms, 1)));
}

GemmShape dense_gemm_shape(int64_t rows, int64_t n, int sms) {
  const int forced = shape_override.load();
  if (forced >= 0) return static_cast<GemmShape>(forced);
  const double cost[] = {dense_cycles_per_k<TinyConfig>(rows, n, sms),
                         dense_cycles_per_k<SmallConfig>(rows, n, sms),
                         dense_cycles_per_k<MediumConfig>(rows, n, sms),
                         dense_cycles_per_k<LargeConfig>(rows, n, sms)};
  int best = 0;
  for (int shape = 1; shape < 4; ++shape) {
    if (cost[shape] < cost[best]) best = shape;
  }
  return static_cast<GemmShape>(best);
}

GemmShape moe_gemm_shape(int64_t assignments, int64_t global_experts) {
  const int forced = shape_override.load();
  if (forced >= 0) return static_cast<GemmShape>(forced);
  // Expected rows per routed expert; actual loads vary per expert and per
  // step, which only changes how full the row tiles are.
  const int64_t expected = global_experts > 0 ? assignments / global_experts : assignments;
  if (expected <= 1) return GemmShape::kTiny;
  if (expected <= 8) return GemmShape::kSmall;
  return GemmShape::kMedium;
}

void launch_gemm(const GemmProblem& problem, GemmShape shape, int64_t max_items,
                 cudaStream_t stream) {
  // MoE formats live in device metadata validated at load; a dense format is
  // known here, so reject it before the device dispatch would trap.
  bool known = problem.route != nullptr;
#define EXL3_KNOWN(B, C) known = known || problem.format == format_key(B, C);
  EXL3_FORMATS(EXL3_KNOWN)
#undef EXL3_KNOWN
  TORCH_CHECK(known, "Unsupported EXL3 rate/codebook combination");
  switch (shape) {
    case GemmShape::kTiny: return launch<TinyConfig>(problem, max_items, stream);
    case GemmShape::kSmall: return launch<SmallConfig>(problem, max_items, stream);
    case GemmShape::kMedium: return launch<MediumConfig>(problem, max_items, stream);
    case GemmShape::kLarge: return launch<LargeConfig>(problem, max_items, stream);
  }
  TORCH_CHECK(false, "unknown EXL3 GEMM shape");
}

}  // namespace exl3
