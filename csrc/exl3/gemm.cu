#include "gemm.cuh"
#include "ops.h"

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace exl3 {
namespace {

// Every format the manifest admits: even rates 1..8 bits for both codebooks,
// plus the half-integer rates 1.5/2.5/3.5 bits for mul1.
#define EXL3_FORMATS(X) \
  X(2, 1) X(4, 1) X(6, 1) X(8, 1) X(10, 1) X(12, 1) X(14, 1) X(16, 1) \
  X(2, 2) X(3, 2) X(4, 2) X(5, 2) X(6, 2) X(7, 2) X(8, 2) X(10, 2) \
  X(12, 2) X(14, 2) X(16, 2)

template <int Threads, int BM, int BN, int BK, int TM, int TN, int MinBlocks,
          bool FuseInput, int Depth = 1, bool ShflLoad = false>
struct GemmConfig {
  using Weight = float;
  static constexpr bool kHalfWeights = false;
  static constexpr int kAsyncStages = 0;
  static constexpr int kThreads = Threads;
  static constexpr int kBM = BM;
  static constexpr int kBN = BN;
  static constexpr int kBK = BK;
  static constexpr int kTM = TM;
  static constexpr int kTN = TN;
  static constexpr int kMinBlocks = MinBlocks;
  // Can form MoE gate/up activations in the block from BF16 rows.
  static constexpr bool kFuseInput = FuseInput;
  // Two decoded-weight buffers overlap the trellis decode with the FMAs but
  // cost the shared memory that caps occupancy at four blocks per SM. A block
  // count target above four trades that overlap for resident warps.
  static constexpr int kBuffers = MinBlocks > 4 ? 1 : 2;
  // Trellis prefetch depth in chunks. The decode is global-load-latency bound
  // (89% of sampled stalls are LDG), so one extra chunk in flight covers
  // roughly twice the memory latency without another shared buffer.
  static constexpr int kDepth = Depth;
  // Fetch each lane's ring window from the warp's own loaded words instead of
  // loading it per lane.
  static constexpr bool kShflLoad = ShflLoad;
  static constexpr int kWarps = Threads / 32;
  static constexpr int kRowsPerWarp = (BM + kWarps - 1) / kWarps;
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

using TinyConfig = GemmConfig<128, 4, 32, 64, 1, 1, 4, true>;
// Decode-shaped variants of the tiny block, reachable only through the
// harness override until a measurement selects one: the default asks for four
// blocks per SM (128 registers/thread, 25% occupancy, which is what the decode
// profile shows). These trade the register budget for twice the resident
// blocks, or halve the K stage to shrink both staging and registers.
using TinyWideConfig = GemmConfig<128, 4, 32, 64, 1, 1, 8, true>;
using TinyB32Config = GemmConfig<128, 4, 32, 32, 1, 1, 8, true>;
// The same trade at the 16-row block, for the moderate row counts where the
// heuristic already picks it.
using SmallSlimConfig = GemmConfig<128, 16, 32, 32, 4, 1, 8, true>;
// Deeper per-thread reuse of each shared-memory load: four output columns per
// thread with all 128 threads computing means the tile must hold 16 (or 8)
// rows, which wastes FMA work at four rows but cuts the shared loads per FMA
// from two to 1.25 -- the profile's L1/shared 57% is exactly those loads.
using Row16Ilp4Config = GemmConfig<128, 16, 32, 32, 1, 4, 4, true>;
using Row8Ilp4Config = GemmConfig<128, 8, 64, 32, 1, 4, 4, true>;
// Half the chunk barriers at the cost of one resident block.
using TinyBk128Config = GemmConfig<128, 4, 32, 128, 1, 1, 4, true>;
// Row-blocking inside the thread instead of column-blocking: one shared weight
// load feeds four FMAs (one per row) and one float4 activation load feeds four
// more, so the shared traffic per FMA drops fourfold. 128 output columns keep
// all 128 threads computing at four rows.
using Row4TmaConfig = GemmConfig<128, 4, 128, 32, 4, 1, 4, true>;
using Row8TmaConfig = GemmConfig<128, 8, 64, 32, 4, 1, 4, true>;
// The same row-blocking with 64-thread blocks and 32-wide columns, so a narrow
// matrix still tiles into more blocks than the card has multiprocessors.
using Row8Tma64Config = GemmConfig<64, 8, 32, 32, 4, 1, 4, true>;
// A speculative step routes one row per touched expert, so a four-row tile
// already spends three of its four FMA rows on padding and a sixteen-row tile
// fifteen of sixteen. These blocks fit the row count instead.
using Row2Config = GemmConfig<128, 2, 32, 64, 1, 1, 4, true>;
using Row1Config = GemmConfig<128, 1, 32, 64, 1, 1, 4, true>;
// The decode's stall profile is 89% global load: keep two trellis chunks in
// flight instead of one.
using TinyDeepConfig = GemmConfig<128, 4, 32, 64, 1, 1, 4, true, 2>;
using TinyShflConfig = GemmConfig<128, 4, 32, 64, 1, 1, 4, true, 1, true>;
// The bit-exactness contract makes every output element one sequential FMA
// chain, so a thread holding one element cannot hide FMA latency. These keep
// the tiny block's four rows (no padding waste) and widen the per-thread
// micro-tile to four output columns instead.
using TinyIlp128Config = GemmConfig<128, 4, 128, 32, 1, 4, 4, true>;
using TinyIlp64Config = GemmConfig<128, 4, 64, 32, 1, 4, 4, true>;
using SmallConfig = GemmConfig<128, 16, 32, 32, 4, 1, 4, true>;
using MediumConfig = GemmConfig<128, 64, 64, 16, 8, 4, 3, false>;
using LargeConfig = GemmConfig<256, 128, 128, 16, 8, 8, 2, false>;

struct GlmHalfConfig : TinyConfig {
  using Weight = half;
  static constexpr bool kHalfWeights = true;
};

struct GlmAsyncConfig : TinyConfig {
  static constexpr int kAsyncStages = 4;
};

template <int Stages>
struct PackedStages {
  // Eight 16x16 tiles per tiny K/N slab; 28 words holds the 3.5-bit ring.
  __align__(16) uint32_t packed[Stages][8][28];
};
template <>
struct PackedStages<0> {};

template <class Cfg>
struct GemmSmem : PackedStages<Cfg::kAsyncStages> {
  float a[Cfg::kBuffers][Cfg::kBK][Cfg::kAStride];
  typename Cfg::Weight w[Cfg::kBuffers][Cfg::kBK][Cfg::kWStride];
  // One H128 block of formed activations per row (fused MoE gate/up only).
  float block[Cfg::kFuseInput ? Cfg::kBM : 1][Cfg::kFuseInput ? 128 : 1];
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

template <int TX, int BX>
__device__ __forceinline__ void load_micro(const half* row, int t, float (&v)[TX]) {
#pragma unroll
  for (int i = 0; i < TX; ++i) v[i] = __half2float(row[micro_offset<TX, BX>(t, i)]);
}

template <int kBitsX2, int kCodebook, class Cfg, int FixedK = 0, int FormsInput = -1>
__device__ __forceinline__ void run_tile(const GemmProblem& p, GemmSmem<Cfg>& s,
                                         const uint16_t* trellis, int64_t tiles_n,
                                         int64_t col0, int rows, float* c, const half* suh) {
  using Ring = TileRing<kBitsX2>;
  constexpr int BK = Cfg::kBK;
  constexpr int T = Cfg::kThreads;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const LaneWindow<kBitsX2> window(lane);
  const bool aligned = (reinterpret_cast<uintptr_t>(trellis) & 3) == 0;
  const int chunks = (FixedK ? FixedK : p.k) / BK;
  const int64_t tile_col0 = col0 / 16;

  uint32_t words[Cfg::kDepth][Cfg::kTasksPerThread][Ring::kLoadWords];
  // Activation staging lives beside the trellis words: a deeper prefetch must
  // not overwrite the activations of a chunk that has not been published yet.
  float4 staged[Cfg::kDepth][Cfg::kAPerThread];
  // Fused MoE gate/up: x * suh for this warp's rows of the current H128 block.
  const bool forms_input = FormsInput < 0 ? Cfg::kFuseInput && p.x != nullptr : FormsInput != 0;
  const int warp = tid >> 5;
  float formed[Cfg::kDepth][Cfg::kFuseInput ? Cfg::kRowsPerWarp : 1][4];

  auto copy_packed = [&](int chunk) {
    if constexpr (Cfg::kAsyncStages != 0) {
      constexpr int Vectors = Ring::kWords / 4;
      if (chunk < chunks) {
        for (int v = tid; v < Cfg::kTileRows * Cfg::kTileCols * Vectors; v += T) {
          const int tile = v / Vectors;
          const int word = 4 * (v % Vectors);
          const int64_t index = (int64_t(chunk) * Cfg::kTileRows + tile / Cfg::kTileCols) *
                                    tiles_n + tile_col0 + tile % Cfg::kTileCols;
          const void* src = trellis + index * Ring::kTileHalfwords + 2 * word;
          void* dst = &s.packed[chunk % Cfg::kAsyncStages][tile][word];
          const auto shared = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
          asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                       :: "r"(shared), "l"(src) : "memory");
        }
      }
      // Empty groups at the tail preserve the four-stage wait invariant.
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
  };

  auto fetch = [&](int slot, int chunk) {
#pragma unroll
    for (int t = 0; t < Cfg::kTasksPerThread; ++t) {
      const int tile = (tid + t * T) >> 5;
      const int tile_row = tile / Cfg::kTileCols;
      const int tile_col = tile % Cfg::kTileCols;
      const int64_t index = (int64_t(chunk) * Cfg::kTileRows + tile_row) * tiles_n +
                            tile_col0 + tile_col;
      const uint16_t* ring = trellis + index * Ring::kTileHalfwords;
      if constexpr (Cfg::kAsyncStages != 0) {
#pragma unroll
        for (int i = 0; i < Ring::kLoadWords; ++i) {
          words[slot][t][i] = s.packed[chunk % Cfg::kAsyncStages][tile][window.word[i]];
        }
      } else if constexpr (Cfg::kShflLoad) {
        load_lane_words_shfl<kBitsX2>(ring, window, aligned, words[slot][t]);
      } else {
        load_lane_words<kBitsX2>(ring, window, aligned, words[slot][t]);
      }
    }
    if (forms_input) {
      if constexpr (Cfg::kFuseInput) {
        // Exactly linear_input's products, one H128 block at a time.
        if ((chunk & (128 / BK - 1)) == 0) {
#pragma unroll
          for (int j = 0; j < Cfg::kRowsPerWarp; ++j) {
            const int r = warp * Cfg::kRowsPerWarp + j;
            const int row = r < Cfg::kBM ? s.row[r] : -1;
            const __nv_bfloat16* xrow = p.x + (row >= 0 ? row / p.topk : 0) * p.hidden;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int64_t col = int64_t(chunk) * BK + lane + 32 * i;
              formed[slot][j][i] = row >= 0 && col < p.hidden
                                     ? __fmul_rn(__bfloat162float(xrow[col]),
                                                 __half2float(suh[col]))
                                     : 0.0f;
            }
          }
        }
      }
      return;
    }
#pragma unroll
    for (int v = 0; v < Cfg::kAPerThread; ++v) {
      const int index = tid + v * T;
      staged[slot][v] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (index < Cfg::kAVectors) {
        const int r = index / (BK / 4);
        const int k4 = index % (BK / 4);
        const int row = s.row[r];
        if (row >= 0) {
          staged[slot][v] = *reinterpret_cast<const float4*>(
              p.a + int64_t(row) * p.a_stride + int64_t(chunk) * BK + 4 * k4);
        }
      }
    }
  };

  auto stash = [&](int slot, int buffer, int chunk) {
#pragma unroll
    for (int t = 0; t < Cfg::kTasksPerThread; ++t) {
      const int tile = (tid + t * T) >> 5;
      const int tile_row = tile / Cfg::kTileCols;
      const int tile_col = tile % Cfg::kTileCols;
      float value[8];
      decode_lane<kBitsX2, kCodebook>(words[slot][t], window.shift, value);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        auto& weight = s.w[buffer][tile_row * 16 + lane_tile_row(lane, j)]
                          [tile_col * 16 + lane_tile_col(lane, j)];
        if constexpr (Cfg::kHalfWeights) {
          // Codebook arithmetic already rounded to FP16 before widening.
          // This changes storage only, never the FP32 accumulation chain.
          weight = __float2half_rn(value[j]);
        } else {
          weight = value[j];
        }
      }
    }
    if (forms_input) {
      if constexpr (Cfg::kFuseInput) {
        if ((chunk & (128 / BK - 1)) == 0) {
          // The previous block's last slice was copied before the barrier
          // that ended the previous chunk.
#pragma unroll
          for (int j = 0; j < Cfg::kRowsPerWarp; ++j) {
            const int r = warp * Cfg::kRowsPerWarp + j;
            hadamard128_warp(formed[slot][j], lane);
            if (r < Cfg::kBM) {
#pragma unroll
              for (int i = 0; i < 4; ++i) s.block[r][lane + 32 * i] = formed[slot][j][i];
            }
          }
          __syncthreads();
        }
        const int offset = (chunk & (128 / BK - 1)) * BK;
        for (int index = tid; index < Cfg::kBM * BK; index += T) {
          const int r = index / BK;
          const int kk = index % BK;
          s.a[buffer][kk][r] = s.block[r][offset + kk];
        }
      }
      return;
    }
#pragma unroll
    for (int v = 0; v < Cfg::kAPerThread; ++v) {
      const int index = tid + v * T;
      if (index < Cfg::kAVectors) {
        const int r = index / (BK / 4);
        const int k4 = index % (BK / 4);
        s.a[buffer][4 * k4][r] = staged[slot][v].x;
        s.a[buffer][4 * k4 + 1][r] = staged[slot][v].y;
        s.a[buffer][4 * k4 + 2][r] = staged[slot][v].z;
        s.a[buffer][4 * k4 + 3][r] = staged[slot][v].w;
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

  if constexpr (Cfg::kAsyncStages != 0) {
    static_assert(Cfg::kAsyncStages == 4);
#pragma unroll
    for (int chunk = 0; chunk < 4; ++chunk) copy_packed(chunk);
    asm volatile("cp.async.wait_group 3;" ::: "memory");
    __syncthreads();
    fetch(0, 0);
    stash(0, 0, 0);
    __syncthreads();
    for (int chunk = 0; chunk < chunks; ++chunk) {
      const int buffer = chunk & 1;
      const bool more = chunk + 1 < chunks;
      // The current raw slab was consumed by stash before this reuse.
      copy_packed(chunk + 4);
      if (more) {
        asm volatile("cp.async.wait_group 3;" ::: "memory");
        __syncthreads();
        fetch(0, chunk + 1);
      }
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
      if (more) stash(0, buffer ^ 1, chunk + 1);
      __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
  } else if constexpr (Cfg::kBuffers == 2 && Cfg::kDepth == 1) {
    fetch(0, 0);
    stash(0, 0, 0);
    __syncthreads();
    for (int chunk = 0; chunk < chunks; ++chunk) {
      const int buffer = chunk & 1;
      const bool more = chunk + 1 < chunks;
      if (more) fetch(0, chunk + 1);
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
      if (more) stash(0, buffer ^ 1, chunk + 1);
      __syncthreads();
    }
  } else if constexpr (Cfg::kBuffers == 2) {
    // Two chunks of trellis in flight: the load for chunk + 2 issues while
    // chunk's FMAs run and chunk + 1 is decoded into the other shared buffer.
    fetch(0, 0);
    if (chunks > 1) fetch(1, 1);
    stash(0, 0, 0);
    __syncthreads();
    for (int chunk = 0; chunk < chunks; ++chunk) {
      const int buffer = chunk & 1;
      const bool more = chunk + 1 < chunks;
      if (chunk + 2 < chunks) fetch(chunk & 1, chunk + 2);
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
      if (more) stash(buffer ^ 1, buffer ^ 1, chunk + 1);
      __syncthreads();
    }
  } else {
    // One buffer: decode, publish, consume, repeat. The barrier pair keeps the
    // decoded slab intact while it is read; the k order is unchanged, so the
    // per-output chains (and the results) are identical.
    for (int chunk = 0; chunk < chunks; ++chunk) {
      fetch(0, chunk);
      stash(0, 0, chunk);
      __syncthreads();
      if (computes) {
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
          float av[Cfg::kTM];
          float wv[Cfg::kTN];
          load_micro<Cfg::kTM, Cfg::kBM>(&s.a[0][kk][0], tm, av);
          load_micro<Cfg::kTN, Cfg::kBN>(&s.w[0][kk][0], tn, wv);
#pragma unroll
          for (int i = 0; i < Cfg::kTM; ++i) {
#pragma unroll
            for (int j = 0; j < Cfg::kTN; ++j) {
              acc[i][j] = __fmaf_rn(av[i], wv[j], acc[i][j]);
            }
          }
        }
      }
      __syncthreads();
    }
  }

  if (computes) {
#pragma unroll
    for (int i = 0; i < Cfg::kTM; ++i) {
      const int r = micro_offset<Cfg::kTM, Cfg::kBM>(tm, i);
      if (r < rows) {
        float* out = c + int64_t(s.row[r]) * p.c_stride + col0;
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

template <class Cfg, bool SkipGlm = false, bool AlignedOnly = false>
__global__ void __launch_bounds__(Cfg::kThreads, Cfg::kMinBlocks)
gemm_kernel(GemmProblem p) {
  __shared__ __align__(16) GemmSmem<Cfg> s;
  const int64_t col_tiles = p.n / Cfg::kBN;
  const bool moe = p.route != nullptr;
  const int64_t row_tiles = moe ? p.route[1] : (p.rows + Cfg::kBM - 1) / Cfg::kBM;
  // Fused MoE gate/up: items of the gate projection, then of the up one.
  const int64_t per_projection = row_tiles * col_tiles;
  const int64_t items = (p.x != nullptr ? 2 : 1) * per_projection;
  const int* sorted = moe ? p.route + RouteLayout::sorted_offset() : nullptr;
  for (int64_t item = blockIdx.x; item < items; item += gridDim.x) {
    const bool second = item >= per_projection;
    const int64_t local_item = second ? item - per_projection : item;
    const int64_t row_tile = local_item / col_tiles;
    const int64_t col0 = (local_item % col_tiles) * Cfg::kBN;
    const int projection = second ? 1 : p.projection;
    float* c = second ? p.c_up : p.c;
    const uint16_t* trellis = p.trellis;
    const half* suh = nullptr;
    int64_t tiles_n = p.tiles_n;
    int format = p.format;
    int rows;
    if (moe) {
      // Tiles follow the sorted list, so a block never mixes experts.
      const int4 tile = reinterpret_cast<const int4*>(
          p.route + RouteLayout::tiles_offset(p.rows))[row_tile];
      const int64_t* ptrs = p.pointers + int64_t(tile.x) * 9 + projection * 3;
      const int64_t* meta = p.metadata + int64_t(tile.x) * 12 + projection * 4;
      trellis = reinterpret_cast<const uint16_t*>(ptrs[0]);
      suh = reinterpret_cast<const half*>(ptrs[1]);
      tiles_n = meta[1] / 16;
      format = format_key(static_cast<int>(meta[2]), static_cast<int>(meta[3]));
      if constexpr (SkipGlm) {
        if (meta[0] == p.k && meta[1] == p.n &&
            (format == format_key(5, 2) || format == format_key(7, 2)) &&
            (!AlignedOnly || (reinterpret_cast<uintptr_t>(trellis) & 15) == 0)) continue;
      }
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
      case format_key(B, C): run_tile<B, C, Cfg>(p, s, trellis, tiles_n, col0, rows, c, suh); \
        break;
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

// Separate GLM entry points keep the nineteen-format dense/routed dispatcher
// out of the hot kernel. Unsupported formats still run in the original kernel.
template <class Cfg, bool Fused>
__global__ void __launch_bounds__(Cfg::kThreads, Cfg::kMinBlocks)
glm_decode_kernel(GemmProblem p) {
  __shared__ __align__(16) GemmSmem<Cfg> s;
  constexpr int K = Fused ? 4096 : 2048;
  constexpr int N = Fused ? 2048 : 4096;
  constexpr int ColTiles = N / Cfg::kBN;
  const int64_t per_projection = int64_t(p.route[1]) * ColTiles;
  const int64_t items = (Fused ? 2 : 1) * per_projection;
  const int* sorted = p.route + RouteLayout::sorted_offset();
  for (int64_t item = blockIdx.x; item < items; item += gridDim.x) {
    const bool second = Fused && item >= per_projection;
    const int64_t local_item = second ? item - per_projection : item;
    const int64_t row_tile = local_item / ColTiles;
    const int64_t col0 = (local_item % ColTiles) * Cfg::kBN;
    const int projection = Fused ? (second ? 1 : 0) : 2;
    const int4 tile = reinterpret_cast<const int4*>(
        p.route + RouteLayout::tiles_offset(p.rows))[row_tile];
    const int64_t* meta = p.metadata + int64_t(tile.x) * 12 + projection * 4;
    if (meta[0] != K || meta[1] != N) continue;
    const int format = format_key(static_cast<int>(meta[2]), static_cast<int>(meta[3]));
    if (format != format_key(5, 2) && format != format_key(7, 2)) continue;
    const int64_t* ptrs = p.pointers + int64_t(tile.x) * 9 + projection * 3;
    const auto* trellis = reinterpret_cast<const uint16_t*>(ptrs[0]);
    if constexpr (Cfg::kAsyncStages != 0) {
      if ((reinterpret_cast<uintptr_t>(trellis) & 15) != 0) continue;
    }
    const auto* suh = reinterpret_cast<const half*>(ptrs[1]);
    for (int i = threadIdx.x; i < Cfg::kBM; i += Cfg::kThreads) {
      s.row[i] = i < tile.z ? sorted[tile.y + i] : -1;
    }
    __syncthreads();
    float* c = second ? p.c_up : p.c;
    if (format == format_key(5, 2)) {
      run_tile<5, 2, Cfg, K, Fused>(p, s, trellis, N / 16, col0, tile.z, c, suh);
    } else {
      run_tile<7, 2, Cfg, K, Fused>(p, s, trellis, N / 16, col0, tile.z, c, suh);
    }
    __syncthreads();
  }
}

template <class Cfg>
void launch_glm_decode(const GemmProblem& p, int64_t max_items, cudaStream_t stream) {
  const bool fused = p.x != nullptr && p.k == 4096 && p.n == 2048;
  const bool down = p.x == nullptr && p.projection == 2 && p.k == 2048 && p.n == 4096;
  if (!p.route || p.rows > 40 || (!fused && !down)) {
    return launch<TinyConfig>(p, max_items, stream);
  }
  if (max_items <= 0) return;
  const auto grid = static_cast<unsigned int>(
      std::min<int64_t>(max_items, std::numeric_limits<int>::max()));
  if (fused) {
    glm_decode_kernel<Cfg, true><<<grid, Cfg::kThreads, 0, stream>>>(p);
  } else {
    glm_decode_kernel<Cfg, false><<<grid, Cfg::kThreads, 0, stream>>>(p);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  gemm_kernel<TinyConfig, true, Cfg::kAsyncStages != 0>
      <<<grid, TinyConfig::kThreads, 0, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

bool gemm_fuses_input(GemmShape shape) {
  switch (shape) {
    case GemmShape::kTiny: return TinyConfig::kFuseInput;
    case GemmShape::kGlmSpecialized: return TinyConfig::kFuseInput;
    case GemmShape::kGlmHalf: return TinyConfig::kFuseInput;
    case GemmShape::kGlmAsync: return TinyConfig::kFuseInput;
    case GemmShape::kSmall: return SmallConfig::kFuseInput;
    case GemmShape::kMedium: return MediumConfig::kFuseInput;
    case GemmShape::kLarge: return LargeConfig::kFuseInput;
    case GemmShape::kTinyWide: return TinyWideConfig::kFuseInput;
    case GemmShape::kTinyB32: return TinyB32Config::kFuseInput;
    case GemmShape::kTinyIlp128: return TinyIlp128Config::kFuseInput;
    case GemmShape::kTinyIlp64: return TinyIlp64Config::kFuseInput;
    case GemmShape::kSmallSlim: return SmallSlimConfig::kFuseInput;
    case GemmShape::kRow16Ilp4: return Row16Ilp4Config::kFuseInput;
    case GemmShape::kRow8Ilp4: return Row8Ilp4Config::kFuseInput;
    case GemmShape::kTinyBk128: return TinyBk128Config::kFuseInput;
    case GemmShape::kRow4Tma: return Row4TmaConfig::kFuseInput;
    case GemmShape::kRow8Tma: return Row8TmaConfig::kFuseInput;
    case GemmShape::kRow8Tma64: return Row8Tma64Config::kFuseInput;
    case GemmShape::kRow2: return Row2Config::kFuseInput;
    case GemmShape::kRow1: return Row1Config::kFuseInput;
    case GemmShape::kTinyDeep: return TinyDeepConfig::kFuseInput;
    case GemmShape::kTinyShfl: return TinyShflConfig::kFuseInput;
  }
  TORCH_CHECK(false, "unknown EXL3 GEMM shape");
}

int gemm_block_rows(GemmShape shape) {
  switch (shape) {
    case GemmShape::kTiny: return TinyConfig::kBM;
    case GemmShape::kGlmSpecialized: return TinyConfig::kBM;
    case GemmShape::kGlmHalf: return TinyConfig::kBM;
    case GemmShape::kGlmAsync: return TinyConfig::kBM;
    case GemmShape::kSmall: return SmallConfig::kBM;
    case GemmShape::kMedium: return MediumConfig::kBM;
    case GemmShape::kLarge: return LargeConfig::kBM;
    case GemmShape::kTinyWide: return TinyWideConfig::kBM;
    case GemmShape::kTinyB32: return TinyB32Config::kBM;
    case GemmShape::kTinyIlp128: return TinyIlp128Config::kBM;
    case GemmShape::kTinyIlp64: return TinyIlp64Config::kBM;
    case GemmShape::kSmallSlim: return SmallSlimConfig::kBM;
    case GemmShape::kRow16Ilp4: return Row16Ilp4Config::kBM;
    case GemmShape::kRow8Ilp4: return Row8Ilp4Config::kBM;
    case GemmShape::kTinyBk128: return TinyBk128Config::kBM;
    case GemmShape::kRow4Tma: return Row4TmaConfig::kBM;
    case GemmShape::kRow8Tma: return Row8TmaConfig::kBM;
    case GemmShape::kRow8Tma64: return Row8Tma64Config::kBM;
    case GemmShape::kRow2: return Row2Config::kBM;
    case GemmShape::kRow1: return Row1Config::kBM;
    case GemmShape::kTinyDeep: return TinyDeepConfig::kBM;
    case GemmShape::kTinyShfl: return TinyShflConfig::kBM;
  }
  TORCH_CHECK(false, "unknown EXL3 GEMM shape");
}

int gemm_block_cols(GemmShape shape) {
  switch (shape) {
    case GemmShape::kTiny: return TinyConfig::kBN;
    case GemmShape::kGlmSpecialized: return TinyConfig::kBN;
    case GemmShape::kGlmHalf: return TinyConfig::kBN;
    case GemmShape::kGlmAsync: return TinyConfig::kBN;
    case GemmShape::kSmall: return SmallConfig::kBN;
    case GemmShape::kMedium: return MediumConfig::kBN;
    case GemmShape::kLarge: return LargeConfig::kBN;
    case GemmShape::kTinyWide: return TinyWideConfig::kBN;
    case GemmShape::kTinyB32: return TinyB32Config::kBN;
    case GemmShape::kTinyIlp128: return TinyIlp128Config::kBN;
    case GemmShape::kTinyIlp64: return TinyIlp64Config::kBN;
    case GemmShape::kSmallSlim: return SmallSlimConfig::kBN;
    case GemmShape::kRow16Ilp4: return Row16Ilp4Config::kBN;
    case GemmShape::kRow8Ilp4: return Row8Ilp4Config::kBN;
    case GemmShape::kTinyBk128: return TinyBk128Config::kBN;
    case GemmShape::kRow4Tma: return Row4TmaConfig::kBN;
    case GemmShape::kRow8Tma: return Row8TmaConfig::kBN;
    case GemmShape::kRow8Tma64: return Row8Tma64Config::kBN;
    case GemmShape::kRow2: return Row2Config::kBN;
    case GemmShape::kRow1: return Row1Config::kBN;
    case GemmShape::kTinyDeep: return TinyDeepConfig::kBN;
    case GemmShape::kTinyShfl: return TinyShflConfig::kBN;
  }
  TORCH_CHECK(false, "unknown EXL3 GEMM shape");
}

namespace {
// Test-only override of the shape heuristic, so a harness can run every
// block shape at every row count. -1 selects the heuristic.
std::atomic<int> shape_override{-1};
}  // namespace

void set_gemm_shape_override(int64_t shape) {
  TORCH_CHECK(shape >= -1 && shape <= static_cast<int>(GemmShape::kGlmAsync),
              "EXL3 GEMM shape override must be -1 (heuristic) or 0..21");
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
// the slower of one block's own duration and the issue of the busiest SM,
// which receives ceil(blocks / SMs) blocks.
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
  const int64_t blocks = (full + (rest > 0 ? 1 : 0)) * col_tiles;
  const int64_t per_sm = (blocks + std::max(sms, 1) - 1) / std::max(sms, 1);
  return std::max(duration, double(per_sm) * (issue / double(blocks)) / 4.0);
}

GemmShape dense_gemm_shape(int64_t rows, int64_t n, int sms) {
  const int forced = shape_override.load();
  if (forced >= 0) return static_cast<GemmShape>(forced);
  // A wide output matrix at verification width reuses one shared weight load
  // across four row FMAs: measured 29% faster on 4096x24576 and 20% on
  // 1536x16384 at four rows. Narrow matrices stay on the four-row block,
  // where 64-wide columns would leave fewer blocks than multiprocessors.
  if (rows >= 4 && n >= 16384) return GemmShape::kRow8Tma;
  // The two decode-shaped variants are override-only: the heuristic keeps
  // selecting among the four measured shapes until a harness shows a variant
  // wins, at which point it joins this model deliberately.
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
  // From the assignments per global expert; actual loads vary per expert and
  // per step, which only changes how full the row tiles are. Thresholds
  // measured on 2560x640 experts with real routing (CMP 170HX).
  const int64_t experts = std::max<int64_t>(global_experts, 1);
  if (2 * assignments <= experts) {
    static const GemmShape configured = [] {
      const char* value = std::getenv("VLLM_EXL3_DECODE_VARIANT");
      if (!value || !*value || std::strcmp(value, "off") == 0) return GemmShape::kTiny;
      if (std::strcmp(value, "half") == 0) return GemmShape::kGlmHalf;
      if (std::strcmp(value, "async") == 0) return GemmShape::kGlmAsync;
      TORCH_CHECK(std::strcmp(value, "specialized") == 0,
                  "Unsupported VLLM_EXL3_DECODE_VARIANT");
      return GemmShape::kGlmSpecialized;
    }();
    return assignments <= 40 ? configured : GemmShape::kTiny;
  }
  if (assignments <= 16 * experts) return GemmShape::kSmall;
  return GemmShape::kMedium;
}

bool moe_fuses_gate_up(GemmShape shape, int64_t assignments) {
  // Forming the activations in every column block pays only while a step
  // has very few routed rows (measured: up to 40 assignments).
  return gemm_fuses_input(shape) && assignments <= 40;
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
  TORCH_CHECK(problem.x == nullptr || (problem.route != nullptr && gemm_fuses_input(shape)),
              "EXL3 fused gate/up needs a MoE problem and a tiny or small shape");
  switch (shape) {
    case GemmShape::kTiny: return launch<TinyConfig>(problem, max_items, stream);
    case GemmShape::kGlmSpecialized: return launch_glm_decode<TinyConfig>(problem, max_items, stream);
    case GemmShape::kGlmHalf: return launch_glm_decode<GlmHalfConfig>(problem, max_items, stream);
    case GemmShape::kGlmAsync: return launch_glm_decode<GlmAsyncConfig>(problem, max_items, stream);
    case GemmShape::kSmall: return launch<SmallConfig>(problem, max_items, stream);
    case GemmShape::kMedium: return launch<MediumConfig>(problem, max_items, stream);
    case GemmShape::kLarge: return launch<LargeConfig>(problem, max_items, stream);
    case GemmShape::kTinyWide: return launch<TinyWideConfig>(problem, max_items, stream);
    case GemmShape::kTinyB32: return launch<TinyB32Config>(problem, max_items, stream);
    case GemmShape::kTinyIlp128: return launch<TinyIlp128Config>(problem, max_items, stream);
    case GemmShape::kTinyIlp64: return launch<TinyIlp64Config>(problem, max_items, stream);
    case GemmShape::kSmallSlim: return launch<SmallSlimConfig>(problem, max_items, stream);
    case GemmShape::kRow16Ilp4: return launch<Row16Ilp4Config>(problem, max_items, stream);
    case GemmShape::kRow8Ilp4: return launch<Row8Ilp4Config>(problem, max_items, stream);
    case GemmShape::kTinyBk128: return launch<TinyBk128Config>(problem, max_items, stream);
    case GemmShape::kRow4Tma: return launch<Row4TmaConfig>(problem, max_items, stream);
    case GemmShape::kRow8Tma: return launch<Row8TmaConfig>(problem, max_items, stream);
    case GemmShape::kRow8Tma64: return launch<Row8Tma64Config>(problem, max_items, stream);
    case GemmShape::kRow2: return launch<Row2Config>(problem, max_items, stream);
    case GemmShape::kRow1: return launch<Row1Config>(problem, max_items, stream);
    case GemmShape::kTinyDeep: return launch<TinyDeepConfig>(problem, max_items, stream);
    case GemmShape::kTinyShfl: return launch<TinyShflConfig>(problem, max_items, stream);
  }
  TORCH_CHECK(false, "unknown EXL3 GEMM shape");
}

}  // namespace exl3
