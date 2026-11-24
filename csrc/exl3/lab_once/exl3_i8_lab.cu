// exl3_i8_lab: decode-once-per-chunk building blocks for GLM-5.3-Flash EXL3
// dense prefill (lab only; loaded with torch.ops.load_library).
//
// EXL3 linear (csrc/exl3/reference.cu):
//   out[r, n] = svh[n] * H128_out( sum_k xin[r, k] * W~[k, n] ),
//   xin = H128_in(x * suh)  (per 128-wide block).
// Fold the output side into the weights once per chunk:
//   W'[k, n] = svh[n] * sum_{n' in block(n)} W~[k, n'] H[n', n]
// so out = xin @ W'. INT8 path: W' per output channel (scale s_w[n] = colmax/127,
// computed once per matrix at load), xin per token (s_x[r] = rowmax/127), then
// an int8 x int8 -> bf16 scaled GEMM (cmp_i8mm, same epilogue as vLLM's
// cutlass_scaled_mm). FP16 path: W' and xin in FP16, then cuBLAS.
//
// Ops (namespace exl3_i8_lab):
//   colmax(trellis, svh, bits_x2, codebook) -> float[N]   max_k |W'[k, n]|
//   rebuild_int8(trellis, svh, inv_sw, bits_x2, codebook, w8[N, K] int8)
//   rebuild_fp16(trellis, svh, bits_x2, codebook, w16[N, K] fp16)
//   act_int8(x bf16[M, logical_in], suh, xq int8[M, padded_in], sx float[M])
//   act_fp16(x bf16[M, logical_in], suh, xr fp16[M, padded_in])
//   act_int8_dec(...)  act_int8 with a 512/1024-thread block per row (decode-sized calls; same results)
//   w8_decode_mm(out bf16[M, N], xq, w8[N, K], sx, sw)  1..16 rows; equals cutlass_scaled_mm bit for bit
//   w8_decode_grouped_mm(out bf16[M, G N], xq[G M, K], w8p, swp int64[G] device pointers, sx[G M], N, K)
//                       one launch for G block-diagonal groups (wo_a); each group = w8_decode_mm bit for bit
#include <torch/library.h>
#include <array>
#include <string>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "codec_prefill.cuh"

namespace exl3_i8_lab {
namespace {

using exl3::hadamard128_warp;
using exl3::lane_tile_col;
using exl3::lane_tile_row;
using exl3::LaneWindow;
using exl3::TileRing;

constexpr int kThreads = 256;
constexpr int kKS = 32;          // k rows per block (two 16-row tiles)
constexpr int kBufStride = 132;  // floats; 4 * row + col keeps decode stores conflict-free
constexpr int kQStride = 9;      // uint32 words per n row of the int8 stage (8 + 1 pad)
constexpr int kHStride = 34;     // halves per n row of the fp16 stage

enum Mode { kColMax = 0, kInt8 = 1, kFp16 = 2 };

template <int kBitsX2, int kCodebook, int kMode>
__global__ void __launch_bounds__(kThreads) rebuild_kernel(
    const uint16_t* __restrict__ trellis, const half* __restrict__ svh,
    const float* __restrict__ inv_sw, int64_t K, int64_t N, int64_t n_begin,
    void* __restrict__ out, float* __restrict__ colmax) {
  using Ring = TileRing<kBitsX2>;
  __shared__ float buf[kKS * kBufStride];
  __shared__ uint32_t qw[128 * kQStride];
  __shared__ __align__(4) half hb[128 * kHStride];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t k0 = int64_t(blockIdx.x) * kKS;
  const int64_t n0 = n_begin + int64_t(blockIdx.y) * 128;  // global output column
  const int64_t o0 = int64_t(blockIdx.y) * 128;            // row in this call's output
  const int64_t tiles_n = N / 16;
  const LaneWindow<kBitsX2> window(lane);
  // Warp w decodes n-tile w of both k-tiles.
#pragma unroll
  for (int s = 0; s < kKS / 16; ++s) {
    const int64_t kt = k0 / 16 + s;
    const int64_t nt = n0 / 16 + warp;
    const uint16_t* tile = trellis + (kt * tiles_n + nt) * Ring::kTileHalfwords;
    uint32_t words[Ring::kLoadWords];
    exl3::load_lane_words<kBitsX2>(tile, window, true, words);
    uint32_t pair[4];
    exl3::dense_prefill_tc::decode_lane_half2<kBitsX2, kCodebook>(words, window.shift, pair);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const __half2 h = *reinterpret_cast<const __half2*>(&pair[j >> 1]);
      const float v = (j & 1) ? __high2float(h) : __low2float(h);
      buf[(16 * s + lane_tile_row(lane, j)) * kBufStride + 16 * warp + lane_tile_col(lane, j)] = v;
    }
  }
  __syncthreads();
  float sv[4], inv[4], cmax[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int64_t n = n0 + lane + 32 * i;
    sv[i] = __half2float(svh[n]);
    inv[i] = kMode == kInt8 ? inv_sw[n] : 0.0f;
    cmax[i] = 0.0f;
  }
  uint32_t packed[4] = {0u, 0u, 0u, 0u};
  // Warp w owns k rows 4w .. 4w + 3 (consecutive, so int8 values pack into a word).
#pragma unroll
  for (int rr = 0; rr < 4; ++rr) {
    const int r = 4 * warp + rr;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = buf[r * kBufStride + lane + 32 * i];
    hadamard128_warp(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float w = __fmul_rn(v[i], sv[i]);
      if constexpr (kMode == kColMax) {
        cmax[i] = fmaxf(cmax[i], fabsf(w));
      } else if constexpr (kMode == kInt8) {
        int q = __float2int_rn(__fmul_rn(w, inv[i]));
        q = min(max(q, -127), 127);
        packed[i] |= (uint32_t(q) & 0xffu) << (8 * rr);
      } else {
        hb[(lane + 32 * i) * kHStride + r] = __float2half_rn(w);
      }
    }
  }
  if constexpr (kMode == kColMax) {
#pragma unroll
    for (int i = 0; i < 4; ++i)  // non-negative floats order like their int bits
      atomicMax(reinterpret_cast<int*>(colmax + n0 + lane + 32 * i), __float_as_int(cmax[i]));
    return;
  } else if constexpr (kMode == kInt8) {
#pragma unroll
    for (int i = 0; i < 4; ++i) qw[(lane + 32 * i) * kQStride + warp] = packed[i];
    __syncthreads();
    int8_t* w8 = reinterpret_cast<int8_t*>(out);
    // 128 rows x 8 words; consecutive threads write consecutive words of a row.
#pragma unroll
    for (int e = threadIdx.x; e < 128 * 8; e += kThreads) {
      const int row = e >> 3, word = e & 7;
      *reinterpret_cast<uint32_t*>(w8 + (o0 + row) * K + k0 + 4 * word) =
          qw[row * kQStride + word];
    }
  } else {
    __syncthreads();
    half* w16 = reinterpret_cast<half*>(out);
    // 128 rows x 16 words (32 halves).
#pragma unroll
    for (int e = threadIdx.x; e < 128 * 16; e += kThreads) {
      const int row = e >> 4, word = e & 15;
      *reinterpret_cast<uint32_t*>(w16 + (o0 + row) * K + k0 + 2 * word) =
          *reinterpret_cast<const uint32_t*>(hb + row * kHStride + 2 * word);
    }
  }
}

template <bool kInt8, int kT = kThreads>
__global__ void __launch_bounds__(kT) act_kernel(
    const __nv_bfloat16* __restrict__ x, const half* __restrict__ suh, int64_t logical_in,
    int64_t padded_in, int8_t* __restrict__ xq, float* __restrict__ sx, half* __restrict__ xr) {
  extern __shared__ float row_buf[];
  __shared__ float wmax[kT / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t r = blockIdx.x;
  float m = 0.0f;
  for (int64_t b = warp; b < padded_in / 128; b += kT / 32) {
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t col = b * 128 + lane + 32 * i;
      v[i] = col < logical_in ? __fmul_rn(__bfloat162float(x[r * logical_in + col]),
                                          __half2float(suh[col]))
                              : 0.0f;
    }
    hadamard128_warp(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t col = b * 128 + lane + 32 * i;
      if constexpr (kInt8) {
        row_buf[col] = v[i];
        m = fmaxf(m, fabsf(v[i]));
      } else {
        xr[r * padded_in + col] = __float2half_rn(v[i]);
      }
    }
  }
  if constexpr (!kInt8) return;
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
  if (lane == 0) wmax[warp] = m;
  __syncthreads();
  m = 0.0f;
#pragma unroll
  for (int w = 0; w < kT / 32; ++w) m = fmaxf(m, wmax[w]);
  const float inv = m > 0.0f ? 127.0f / m : 0.0f;
  for (int64_t col = threadIdx.x; col < padded_in; col += kT) {
    int q = __float2int_rn(__fmul_rn(row_buf[col], inv));
    q = min(max(q, -127), 127);
    xq[r * padded_in + col] = static_cast<int8_t>(q);
  }
  if (threadIdx.x == 0) sx[r] = m > 0.0f ? m / 127.0f : 1.0f;
}


// act_kernel<true> with the row held in registers (VLLM_EXL3_ACT_REG=1): warp w holds 128-blocks w, w + kT / 32,
// ... (at most kNB), each in hadamard128_warp's lane layout, so the per-element operations (FP32 x * suh, the
// H128 stages, the scaled round-to-nearest and clamp) and the order-independent row max are act_kernel's
// exactly: bitwise the same xq and sx, without the shared-memory row round trip and its second pass.
// kGrouped (act_int8_dec_grouped): x is [T, G, K] and output row r = g T + t (group-major) reads x[t, g, :], so the
// grouped projection needs no transposed copy; the per-row arithmetic is unchanged.
template <int kT, int kNB, bool kGrouped = false>
__global__ void __launch_bounds__(kT) act_reg_kernel(
    const __nv_bfloat16* __restrict__ x, const half* __restrict__ suh, int64_t logical_in,
    int64_t padded_in, int8_t* __restrict__ xq, float* __restrict__ sx, int64_t tokens = 0,
    int64_t groups = 0) {
  __shared__ float wmax[kT / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t r = blockIdx.x;
  const int64_t blocks = padded_in / 128;
  float v[kNB][4];
  float m = 0.0f;
#pragma unroll
  for (int nb = 0; nb < kNB; ++nb) {
    const int64_t b = warp + int64_t(nb) * (kT / 32);
    if (b < blocks) {  // warp-uniform
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t col = b * 128 + lane + 32 * i;
        int64_t src = r * logical_in + col;
        if constexpr (kGrouped) src = ((r % tokens) * groups + r / tokens) * logical_in + col;
        v[nb][i] = col < logical_in ? __fmul_rn(__bfloat162float(x[src]), __half2float(suh[col]))
                                    : 0.0f;
      }
      hadamard128_warp(v[nb], lane);
#pragma unroll
      for (int i = 0; i < 4; ++i) m = fmaxf(m, fabsf(v[nb][i]));
    }
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
  if (lane == 0) wmax[warp] = m;
  __syncthreads();
  m = 0.0f;
#pragma unroll
  for (int w = 0; w < kT / 32; ++w) m = fmaxf(m, wmax[w]);
  const float inv = m > 0.0f ? 127.0f / m : 0.0f;
#pragma unroll
  for (int nb = 0; nb < kNB; ++nb) {
    const int64_t b = warp + int64_t(nb) * (kT / 32);
    if (b < blocks) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int q = __float2int_rn(__fmul_rn(v[nb][i], inv));
        q = min(max(q, -127), 127);
        xq[r * padded_in + b * 128 + lane + 32 * i] = static_cast<int8_t>(q);
      }
    }
  }
  if (threadIdx.x == 0) sx[r] = m > 0.0f ? m / 127.0f : 1.0f;
}

inline bool act_reg_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("VLLM_EXL3_ACT_REG");
    return v != nullptr && v[0] == '1';
  }();
  return on;
}

// Decode-sized W8A8 GEMM (VLLM_EXL3_DENSE_W8_DECODE): out[m, n] = bf16(sx[m] * (sw[n] * float(sum_k xq[m, k] *
// w8[n, k]))), m <= 16, w8 row-major [N, K] (the sidecar as stored), N % 16 == 0, K % 128 == 0. Bitwise equal to
// vLLM cutlass_scaled_mm (exact int32 dot product, ScaledEpilogue order scale_a * (scale_b * acc) in FP32).
// One CTA per 16 output rows, kDecWarps warps interleave 128-byte K chunks (chunk c = warp + kDecWarps * j) with
// kDecUnroll chunks in flight, so a CTA streams 16 rows x 1 KiB contiguous runs; lane (g, t) loads bytes
// [16t, 16t + 16) and [64 + 16t, 64 + 16t + 16) of each chunk of rows g and g + 8 (whole 32-byte sectors per load)
// straight into s8 m16n8k32 A fragments (MMA step s takes words 2s / 2s + 1), and the token's same bytes as B.
// Card 4, CUDA-graph back to back, 4 rows: 1.06-1.13x a same-size streaming read (cutlass_scaled_mm 1.21-1.56x).
constexpr int kDecWarps = 4;
constexpr int kDecUnroll = 2;

__device__ __forceinline__ void mma_s8s8(const uint4& a, uint32_t b0, uint32_t b1, int (&d)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

// One CTA's 16 output rows of out[m, :] (row stride out_ld) = xq[m, k] . w8[n, k]^T, shared by the single and
// the grouped launch so both run the same arithmetic.
// kW / kU (default kDecWarps / kDecUnroll): warps per CTA and chunks in flight per warp. Every partial is an exact
// s32 sum and the cross-warp reduction is an integer sum, so any (kW, kU) is bitwise the default's result.
template <int kTiles, int kW = kDecWarps, int kU = kDecUnroll>  // kTiles 1: m <= 8, 2: m <= 16
__device__ __forceinline__ void w8_decode_tile(const int8_t* __restrict__ xq, const int8_t* __restrict__ w8,
                                               const float* __restrict__ sx, const float* __restrict__ sw,
                                               __nv_bfloat16* __restrict__ out, int64_t out_ld, int m,
                                               int k, int row0) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int chunks = k >> 7;
  const uint4* wa = reinterpret_cast<const uint4*>(w8 + int64_t(row0 + g) * k) + t;
  const uint4* wb = reinterpret_cast<const uint4*>(w8 + int64_t(row0 + g + 8) * k) + t;
  const uint4* xa[kTiles];
  bool live[kTiles];
#pragma unroll
  for (int i = 0; i < kTiles; ++i) {
    const int tok = 8 * i + g;
    live[i] = tok < m;
    xa[i] = reinterpret_cast<const uint4*>(xq + int64_t(live[i] ? tok : 0) * k) + t;
  }
  int acc[kTiles][4];
#pragma unroll
  for (int i = 0; i < kTiles; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0;
  for (int c0 = warp; c0 < chunks; c0 += kW * kU) {
    uint4 ra[kU][2], rb[kU][2];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int c = c0 + u * kW;
      if (c < chunks) {
        ra[u][0] = ldg_stream(wa + 8 * c); ra[u][1] = ldg_stream(wa + 8 * c + 4);
        rb[u][0] = ldg_stream(wb + 8 * c); rb[u][1] = ldg_stream(wb + 8 * c + 4);
      }
    }
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int c = c0 + u * kW;
      if (c < chunks) {
        const uint32_t A[8] = {ra[u][0].x, ra[u][0].y, ra[u][0].z, ra[u][0].w,
                               ra[u][1].x, ra[u][1].y, ra[u][1].z, ra[u][1].w};
        const uint32_t B[8] = {rb[u][0].x, rb[u][0].y, rb[u][0].z, rb[u][0].w,
                               rb[u][1].x, rb[u][1].y, rb[u][1].z, rb[u][1].w};
#pragma unroll
        for (int i = 0; i < kTiles; ++i) {
          uint32_t X[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
          if (live[i]) {
            const uint4 x0 = __ldg(xa[i] + 8 * c), x1 = __ldg(xa[i] + 8 * c + 4);
            X[0] = x0.x; X[1] = x0.y; X[2] = x0.z; X[3] = x0.w;
            X[4] = x1.x; X[5] = x1.y; X[6] = x1.z; X[7] = x1.w;
          }
#pragma unroll
          for (int s = 0; s < 4; ++s)
            mma_s8s8(make_uint4(A[2 * s], B[2 * s], A[2 * s + 1], B[2 * s + 1]), X[2 * s],
                     X[2 * s + 1], acc[i]);
        }
      }
    }
  }
  __shared__ int red[kW][kTiles][4][32];
#pragma unroll
  for (int i = 0; i < kTiles; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) red[warp][i][j][lane] = acc[i][j];
  __syncthreads();
  if (warp != 0) return;
#pragma unroll
  for (int i = 0; i < kTiles; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int sum = 0;
#pragma unroll
      for (int w = 0; w < kW; ++w) sum += red[w][i][j][lane];
      // c0: (row g, token 2t), c1: (row g, token 2t + 1), c2 / c3: row g + 8.
      const int tok = 8 * i + 2 * t + (j & 1);
      const int row = row0 + g + (j >> 1) * 8;
      if (tok < m) {
        const float v = __fmul_rn(sx[tok], __fmul_rn(sw[row], static_cast<float>(sum)));
        out[int64_t(tok) * out_ld + row] = __float2bfloat16_rn(v);
      }
    }
  }
}

// The decode kernel at a chosen (warps, unroll) (w8_decode_mm_cfg / VLLM_EXL3_W8_TUNE).
template <int kTiles, int kW, int kU>
__global__ void __launch_bounds__(32 * kW)
    w8_decode_cfg_kernel(const int8_t* __restrict__ xq, const int8_t* __restrict__ w8,
                         const float* __restrict__ sx, const float* __restrict__ sw,
                         __nv_bfloat16* __restrict__ out, int m, int n, int k) {
  w8_decode_tile<kTiles, kW, kU>(xq, w8, sx, sw, out, n, m, k, blockIdx.x * 16);
}

template <int kTiles>
__global__ void __launch_bounds__(32 * kDecWarps)
    w8_decode_kernel(const int8_t* __restrict__ xq, const int8_t* __restrict__ w8,
                     const float* __restrict__ sx, const float* __restrict__ sw,
                     __nv_bfloat16* __restrict__ out, int m, int n, int k) {
  w8_decode_tile<kTiles>(xq, w8, sx, sw, out, n, m, k, blockIdx.x * 16);
}

// Block-diagonal projection (DeepSeek-V4.1 wo_a): G groups in one launch, grid (n / 16, G). Group i reads rows
// [i m, (i + 1) m) of xq / sx (group-major), its own weight w8p[i] [n, k] and scales swp[i] [n] (device pointers),
// and writes columns [i n, (i + 1) n) of out [m, G n]. Each CTA's arithmetic is w8_decode_kernel's.
template <int kTiles>
__global__ void __launch_bounds__(32 * kDecWarps)
    w8_decode_grouped_kernel(const int8_t* __restrict__ xq, const int64_t* __restrict__ w8p,
                             const int64_t* __restrict__ swp, const float* __restrict__ sx,
                             __nv_bfloat16* __restrict__ out, int m, int n, int k) {
  const int grp = blockIdx.y;
  w8_decode_tile<kTiles>(xq + int64_t(grp) * m * k, reinterpret_cast<const int8_t*>(w8p[grp]),
                         sx + int64_t(grp) * m, reinterpret_cast<const float*>(swp[grp]),
                         out + int64_t(grp) * n, int64_t(gridDim.y) * n, m, k, blockIdx.x * 16);
}

template <int kTiles, int kW, int kU>
__global__ void __launch_bounds__(32 * kW)
    w8_decode_grouped_cfg_kernel(const int8_t* __restrict__ xq, const int64_t* __restrict__ w8p,
                                 const int64_t* __restrict__ swp, const float* __restrict__ sx,
                                 __nv_bfloat16* __restrict__ out, int m, int n, int k) {
  const int grp = blockIdx.y;
  w8_decode_tile<kTiles, kW, kU>(xq + int64_t(grp) * m * k, reinterpret_cast<const int8_t*>(w8p[grp]),
                                 sx + int64_t(grp) * m, reinterpret_cast<const float*>(swp[grp]),
                                 out + int64_t(grp) * n, int64_t(gridDim.y) * n, m, k, blockIdx.x * 16);
}

// VLLM_EXL3_W8_TUNE="[g:]n:k:[tiles:]warps:unroll,...": per-shape (and per row-tile count: 1 for <= 8 rows, 2 for
// <= 16) launch configurations for the decode kernels (bitwise).
struct W8Cfg { int w, u; };
inline W8Cfg w8_tuned(bool grouped, int64_t n, int64_t k, int tiles) {
  static const std::vector<std::array<int64_t, 6>> table = [] {
    std::vector<std::array<int64_t, 6>> t;
    const char* v = std::getenv("VLLM_EXL3_W8_TUNE");
    if (v == nullptr) return t;
    std::string s(v);
    size_t pos = 0;
    while (pos < s.size()) {
      size_t end = s.find(',', pos);
      if (end == std::string::npos) end = s.size();
      std::string item = s.substr(pos, end - pos);
      int64_t g = 0;
      if (item.rfind("g:", 0) == 0) { g = 1; item = item.substr(2); }
      std::vector<int64_t> a;
      size_t p2 = 0;
      while (p2 < item.size()) {
        size_t e2 = item.find(':', p2);
        a.push_back(std::stoll(item.substr(p2, e2 == std::string::npos ? std::string::npos : e2 - p2)));
        p2 = e2 == std::string::npos ? item.size() : e2 + 1;
      }
      TORCH_CHECK(a.size() == 4 || a.size() == 5, "VLLM_EXL3_W8_TUNE item: [g:]n:k:[tiles:]warps:unroll");
      if (a.size() == 4) t.push_back({g, a[0], a[1], 0, a[2], a[3]});
      else t.push_back({g, a[0], a[1], a[2], a[3], a[4]});
      pos = end + 1;
    }
    return t;
  }();
  for (const auto& e : table)
    if (e[0] == (grouped ? 1 : 0) && e[1] == n && e[2] == k && (e[3] == 0 || e[3] == tiles))
      return {int(e[4]), int(e[5])};
  return {kDecWarps, kDecUnroll};
}

#define W8_CFG_SWITCH(W, U, CALL)                                                       \
  switch (W * 16 + U) {                                                                 \
    case 2 * 16 + 1: CALL(2, 1); break;  case 2 * 16 + 2: CALL(2, 2); break;            \
    case 2 * 16 + 4: CALL(2, 4); break;  case 4 * 16 + 1: CALL(4, 1); break;            \
    case 4 * 16 + 2: CALL(4, 2); break;  case 4 * 16 + 4: CALL(4, 4); break;            \
    case 8 * 16 + 1: CALL(8, 1); break;  case 8 * 16 + 2: CALL(8, 2); break;            \
    case 8 * 16 + 4: CALL(8, 4); break;                                                 \
    default: TORCH_CHECK(false, "W8 decode config: warps 2/4/8, unroll 1/2/4");         \
  }

#define LAB_FORMATS(X) X(10, 2) X(12, 2) X(14, 2) X(12, 1)

template <int kMode>
void launch_rebuild(const at::Tensor& trellis, const at::Tensor& svh, const float* inv_sw,
                    int64_t bits_x2, int64_t codebook, int64_t n_begin, int64_t n_count,
                    void* out, float* colmax) {
  TORCH_CHECK(trellis.is_cuda() && trellis.scalar_type() == at::kShort && trellis.dim() == 3 &&
                  trellis.is_contiguous(), "exl3_i8_lab: trellis int16 [K/16, N/16, 8*bits_x2]");
  TORCH_CHECK(trellis.size(2) == 8 * bits_x2, "exl3_i8_lab: trellis stride");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(trellis.data_ptr()) % 4 == 0, "exl3_i8_lab: alignment");
  const int64_t K = trellis.size(0) * 16, N = trellis.size(1) * 16;
  (void)N;
  TORCH_CHECK(K % kKS == 0 && N % 128 == 0, "exl3_i8_lab: K % 32, N % 128");
  TORCH_CHECK(svh.scalar_type() == at::kHalf && svh.numel() == N && svh.is_contiguous(), "svh");
  const c10::cuda::CUDAGuard guard(trellis.device());
  TORCH_CHECK(n_begin >= 0 && n_count > 0 && n_begin % 128 == 0 && n_count % 128 == 0 &&
                  n_begin + n_count <= N, "exl3_i8_lab: column range");
  const dim3 grid(K / kKS, n_count / 128);
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* t = reinterpret_cast<const uint16_t*>(trellis.data_ptr());
  const auto* s = reinterpret_cast<const half*>(svh.data_ptr());
#define LAB_CASE(B, C)                                                                \
  if (bits_x2 == B && codebook == C) {                                                \
    rebuild_kernel<B, C, kMode><<<grid, kThreads, 0, stream>>>(t, s, inv_sw, K, N, n_begin, \
                                                               out, colmax);          \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                   \
    return;                                                                           \
  }
  LAB_FORMATS(LAB_CASE)
#undef LAB_CASE
  TORCH_CHECK(false, "exl3_i8_lab: unsupported format ", bits_x2, "/", codebook);
}

at::Tensor colmax(const at::Tensor& trellis, const at::Tensor& svh, int64_t bits_x2,
                  int64_t codebook) {
  at::Tensor cm = at::zeros({trellis.size(1) * 16}, trellis.options().dtype(at::kFloat));
  launch_rebuild<kColMax>(trellis, svh, nullptr, bits_x2, codebook, 0, cm.numel(), nullptr,
                          cm.data_ptr<float>());
  return cm;
}

void rebuild_int8(const at::Tensor& trellis, const at::Tensor& svh, const at::Tensor& inv_sw,
                  int64_t bits_x2, int64_t codebook, int64_t n_begin, at::Tensor& w8) {
  const int64_t K = trellis.size(0) * 16, N = trellis.size(1) * 16;
  TORCH_CHECK(inv_sw.scalar_type() == at::kFloat && inv_sw.numel() == N && inv_sw.is_contiguous(),
              "inv_sw");
  TORCH_CHECK(w8.scalar_type() == at::kChar && w8.is_contiguous() && w8.dim() == 2 &&
                  w8.size(1) == K, "w8 int8 [n_count, K]");
  launch_rebuild<kInt8>(trellis, svh, inv_sw.data_ptr<float>(), bits_x2, codebook, n_begin,
                        w8.size(0), w8.data_ptr(), nullptr);
}

void rebuild_fp16(const at::Tensor& trellis, const at::Tensor& svh, int64_t bits_x2,
                  int64_t codebook, int64_t n_begin, at::Tensor& w16) {
  const int64_t K = trellis.size(0) * 16, N = trellis.size(1) * 16;
  TORCH_CHECK(w16.scalar_type() == at::kHalf && w16.is_contiguous() && w16.dim() == 2 &&
                  w16.size(1) == K, "w16 fp16 [n_count, K]");
  launch_rebuild<kFp16>(trellis, svh, nullptr, bits_x2, codebook, n_begin, w16.size(0),
                        w16.data_ptr(), nullptr);
}

void check_act(const at::Tensor& x, const at::Tensor& suh) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(),
              "x bf16 [M, logical_in]");
  TORCH_CHECK(suh.scalar_type() == at::kHalf && suh.is_contiguous() && suh.numel() % 128 == 0 &&
                  suh.numel() >= x.size(1), "suh");
}

void act_int8(const at::Tensor& x, const at::Tensor& suh, at::Tensor& xq, at::Tensor& sx) {
  check_act(x, suh);
  const int64_t M = x.size(0), P = suh.numel();
  TORCH_CHECK(xq.scalar_type() == at::kChar && xq.is_contiguous() && xq.size(0) == M &&
                  xq.size(1) == P, "xq int8 [M, padded_in]");
  TORCH_CHECK(sx.scalar_type() == at::kFloat && sx.numel() == M, "sx float [M]");
  if (M == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  const size_t smem = P * sizeof(float);
  auto k = act_kernel<true>;
  // Dynamic plus the static wmax block must fit; raise the opt-in limit for wide K.
  if (smem > 40 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
  k<<<M, kThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const half*>(suh.data_ptr()), x.size(1), P,
      reinterpret_cast<int8_t*>(xq.data_ptr()), sx.data_ptr<float>(), nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void act_int8_dec(const at::Tensor& x, const at::Tensor& suh, at::Tensor& xq, at::Tensor& sx) {
  check_act(x, suh);
  const int64_t M = x.size(0), P = suh.numel();
  TORCH_CHECK(xq.scalar_type() == at::kChar && xq.is_contiguous() && xq.size(0) == M &&
                  xq.size(1) == P, "xq int8 [M, padded_in]");
  TORCH_CHECK(sx.scalar_type() == at::kFloat && sx.numel() == M, "sx float [M]");
  if (M == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (act_reg_enabled() && P / 128 <= 128) {
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const auto* sp = reinterpret_cast<const half*>(suh.data_ptr());
    auto* qp = reinterpret_cast<int8_t*>(xq.data_ptr());
    float* sxp = sx.data_ptr<float>();
    const int64_t nb = P / 128;
    if (nb <= 16) act_reg_kernel<512, 1><<<M, 512, 0, stream>>>(xp, sp, x.size(1), P, qp, sxp);
    else if (nb <= 32) act_reg_kernel<512, 2><<<M, 512, 0, stream>>>(xp, sp, x.size(1), P, qp, sxp);
    else if (nb <= 64) act_reg_kernel<1024, 2><<<M, 1024, 0, stream>>>(xp, sp, x.size(1), P, qp, sxp);
    else act_reg_kernel<1024, 4><<<M, 1024, 0, stream>>>(xp, sp, x.size(1), P, qp, sxp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const size_t smem = P * sizeof(float);
  // Card 4, graph replay, 4 rows: 512 threads fastest below K 8192, 1024 at or above (act_int8: 256).
  auto k = P >= 8192 ? act_kernel<true, 1024> : act_kernel<true, 512>;
  const int threads = P >= 8192 ? 1024 : 512;
  if (smem > 40 * 1024)
    C10_CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
  k<<<M, threads, smem, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                  reinterpret_cast<const half*>(suh.data_ptr()), x.size(1), P,
                                  reinterpret_cast<int8_t*>(xq.data_ptr()), sx.data_ptr<float>(),
                                  nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// w8_decode_mm at (warps, unroll) (bitwise; the sweep op and the tuned path).
void w8_decode_mm_at(at::Tensor& out, const at::Tensor& xq, const at::Tensor& w8, const at::Tensor& sx,
                     const at::Tensor& sw, int64_t warps, int64_t unroll) {
  const int64_t m = xq.size(0), k = xq.size(1), n = w8.size(0);
  TORCH_CHECK(w8.size(1) == k && m >= 1 && m <= 16 && n % 16 == 0 && k % 128 == 0, "w8_decode_mm_at shapes");
  const c10::cuda::CUDAGuard guard(xq.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
#define W8_LAUNCH(W, U)                                                                            \
  if (m <= 8)                                                                                      \
    w8_decode_cfg_kernel<1, W, U><<<n / 16, 32 * W, 0, stream>>>(                                  \
        xq.data_ptr<int8_t>(), w8.data_ptr<int8_t>(), sx.data_ptr<float>(), sw.data_ptr<float>(), o, \
        int(m), int(n), int(k));                                                                   \
  else                                                                                             \
    w8_decode_cfg_kernel<2, W, U><<<n / 16, 32 * W, 0, stream>>>(                                  \
        xq.data_ptr<int8_t>(), w8.data_ptr<int8_t>(), sx.data_ptr<float>(), sw.data_ptr<float>(), o, \
        int(m), int(n), int(k));
  W8_CFG_SWITCH(int(warps), int(unroll), W8_LAUNCH)
#undef W8_LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// act_int8_dec of the group-major rows of x [T, G, K] (row g T + t = x[t, g, :]) without the transposed copy:
// the bytes act_int8 / act_int8_dec write for x.transpose(0, 1).reshape(G T, K) (per-row quantisation).
void act_int8_dec_grouped(const at::Tensor& x, const at::Tensor& suh, at::Tensor& xq, at::Tensor& sx) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 3 && x.is_contiguous(),
              "x bf16 [T, G, K] contiguous");
  const int64_t t = x.size(0), g = x.size(1), k = x.size(2), P = suh.numel();
  TORCH_CHECK(suh.scalar_type() == at::kHalf && suh.is_contiguous() && P % 128 == 0 && P >= k && P / 128 <= 128,
              "suh");
  TORCH_CHECK(xq.scalar_type() == at::kChar && xq.is_contiguous() && xq.size(0) == g * t && xq.size(1) == P,
              "xq int8 [G T, padded_in]");
  TORCH_CHECK(sx.scalar_type() == at::kFloat && sx.numel() == g * t, "sx float [G T]");
  if (t * g == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* sp = reinterpret_cast<const half*>(suh.data_ptr());
  auto* qp = reinterpret_cast<int8_t*>(xq.data_ptr());
  float* sxp = sx.data_ptr<float>();
  const int64_t nb = P / 128, m = g * t;
  if (nb <= 16) act_reg_kernel<512, 1, true><<<m, 512, 0, stream>>>(xp, sp, k, P, qp, sxp, t, g);
  else if (nb <= 32) act_reg_kernel<512, 2, true><<<m, 512, 0, stream>>>(xp, sp, k, P, qp, sxp, t, g);
  else if (nb <= 64) act_reg_kernel<1024, 2, true><<<m, 1024, 0, stream>>>(xp, sp, k, P, qp, sxp, t, g);
  else act_reg_kernel<1024, 4, true><<<m, 1024, 0, stream>>>(xp, sp, k, P, qp, sxp, t, g);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void w8_decode_mm(at::Tensor& out, const at::Tensor& xq, const at::Tensor& w8, const at::Tensor& sx,
                  const at::Tensor& sw) {
  TORCH_CHECK(xq.is_cuda() && xq.scalar_type() == at::kChar && xq.dim() == 2 && xq.is_contiguous(),
              "xq int8 [M, K]");
  TORCH_CHECK(w8.scalar_type() == at::kChar && w8.dim() == 2 && w8.is_contiguous(), "w8 int8 [N, K]");
  const int64_t m = xq.size(0), k = xq.size(1), n = w8.size(0);
  TORCH_CHECK(w8.size(1) == k && m >= 1 && m <= 16 && n % 16 == 0 && k % 128 == 0,
              "w8_decode_mm: 1..16 rows, N % 16, K % 128");
  TORCH_CHECK(sx.scalar_type() == at::kFloat && sx.numel() == m && sw.scalar_type() == at::kFloat &&
                  sw.numel() == n, "scales");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == m &&
                  out.size(1) == n, "out bf16 [M, N]");
  const c10::cuda::CUDAGuard guard(xq.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const W8Cfg cfg = w8_tuned(false, n, k, m <= 8 ? 1 : 2);
  if (cfg.w != kDecWarps || cfg.u != kDecUnroll) {
    w8_decode_mm_at(out, xq, w8, sx, sw, cfg.w, cfg.u);
    return;
  }
  if (m <= 8)
    w8_decode_kernel<1><<<n / 16, 32 * kDecWarps, 0, stream>>>(
        xq.data_ptr<int8_t>(), w8.data_ptr<int8_t>(), sx.data_ptr<float>(), sw.data_ptr<float>(), o,
        int(m), int(n), int(k));
  else
    w8_decode_kernel<2><<<n / 16, 32 * kDecWarps, 0, stream>>>(
        xq.data_ptr<int8_t>(), w8.data_ptr<int8_t>(), sx.data_ptr<float>(), sw.data_ptr<float>(), o,
        int(m), int(n), int(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void w8_decode_grouped_mm(at::Tensor& out, const at::Tensor& xq, const at::Tensor& w8p,
                          const at::Tensor& swp, const at::Tensor& sx, int64_t n, int64_t k) {
  TORCH_CHECK(xq.is_cuda() && xq.scalar_type() == at::kChar && xq.dim() == 2 && xq.is_contiguous() &&
                  xq.size(1) == k, "xq int8 [G * M, K]");
  TORCH_CHECK(w8p.scalar_type() == at::kLong && swp.scalar_type() == at::kLong && w8p.is_cuda() &&
                  swp.is_cuda() && w8p.numel() == swp.numel() && w8p.numel() >= 1, "pointer tables int64 [G]");
  const int64_t g = w8p.numel(), mg = xq.size(0);
  TORCH_CHECK(mg % g == 0, "xq rows must be G * M");
  const int64_t m = mg / g;
  TORCH_CHECK(m >= 1 && m <= 16 && n % 16 == 0 && k % 128 == 0, "w8_decode_grouped_mm: 1..16 rows, N % 16, K % 128");
  TORCH_CHECK(sx.scalar_type() == at::kFloat && sx.numel() == mg, "sx float [G * M]");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.size(0) == m &&
                  out.size(1) == g * n, "out bf16 [M, G * N]");
  const c10::cuda::CUDAGuard guard(xq.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  const dim3 grid(n / 16, g);
  const W8Cfg cfg = w8_tuned(true, n, k, m <= 8 ? 1 : 2);
  if (cfg.w != kDecWarps || cfg.u != kDecUnroll) {
#define W8G_LAUNCH(W, U)                                                                                 \
  if (m <= 8)                                                                                            \
    w8_decode_grouped_cfg_kernel<1, W, U><<<grid, 32 * W, 0, stream>>>(                                  \
        xq.data_ptr<int8_t>(), w8p.data_ptr<int64_t>(), swp.data_ptr<int64_t>(), sx.data_ptr<float>(), o, \
        int(m), int(n), int(k));                                                                         \
  else                                                                                                   \
    w8_decode_grouped_cfg_kernel<2, W, U><<<grid, 32 * W, 0, stream>>>(                                  \
        xq.data_ptr<int8_t>(), w8p.data_ptr<int64_t>(), swp.data_ptr<int64_t>(), sx.data_ptr<float>(), o, \
        int(m), int(n), int(k));
    W8_CFG_SWITCH(cfg.w, cfg.u, W8G_LAUNCH)
#undef W8G_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  if (m <= 8)
    w8_decode_grouped_kernel<1><<<grid, 32 * kDecWarps, 0, stream>>>(
        xq.data_ptr<int8_t>(), w8p.data_ptr<int64_t>(), swp.data_ptr<int64_t>(), sx.data_ptr<float>(), o,
        int(m), int(n), int(k));
  else
    w8_decode_grouped_kernel<2><<<grid, 32 * kDecWarps, 0, stream>>>(
        xq.data_ptr<int8_t>(), w8p.data_ptr<int64_t>(), swp.data_ptr<int64_t>(), sx.data_ptr<float>(), o,
        int(m), int(n), int(k));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void act_fp16(const at::Tensor& x, const at::Tensor& suh, at::Tensor& xr) {
  check_act(x, suh);
  const int64_t M = x.size(0), P = suh.numel();
  TORCH_CHECK(xr.scalar_type() == at::kHalf && xr.is_contiguous() && xr.size(0) == M &&
                  xr.size(1) == P, "xr fp16 [M, padded_in]");
  if (M == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  act_kernel<false><<<M, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const half*>(suh.data_ptr()), x.size(1), P, nullptr, nullptr,
      reinterpret_cast<half*>(xr.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace
}  // namespace exl3_i8_lab

TORCH_LIBRARY(exl3_i8_lab, m) {
  m.def("colmax(Tensor trellis, Tensor svh, int bits_x2, int codebook) -> Tensor");
  m.def("rebuild_int8(Tensor trellis, Tensor svh, Tensor inv_sw, int bits_x2, int codebook, "
        "int n_begin, Tensor(a!) w8) -> ()");
  m.def("rebuild_fp16(Tensor trellis, Tensor svh, int bits_x2, int codebook, int n_begin, "
        "Tensor(a!) w16) -> ()");
  m.def("act_int8(Tensor x, Tensor suh, Tensor(a!) xq, Tensor(b!) sx) -> ()");
  m.def("act_fp16(Tensor x, Tensor suh, Tensor(a!) xr) -> ()");
  m.def("act_int8_dec(Tensor x, Tensor suh, Tensor(a!) xq, Tensor(b!) sx) -> ()");
  m.def("w8_decode_mm(Tensor(a!) out, Tensor xq, Tensor w8, Tensor sx, Tensor sw) -> ()");
  m.def("act_int8_dec_grouped(Tensor x, Tensor suh, Tensor(a!) xq, Tensor(b!) sx) -> ()");
  m.def("w8_decode_mm_at(Tensor(a!) out, Tensor xq, Tensor w8, Tensor sx, Tensor sw, int warps, int unroll) -> ()");
  m.def("w8_decode_grouped_mm(Tensor(a!) out, Tensor xq, Tensor w8p, Tensor swp, Tensor sx, int n, int k) -> ()");
}

TORCH_LIBRARY_IMPL(exl3_i8_lab, CUDA, m) {
  m.impl("colmax", &exl3_i8_lab::colmax);
  m.impl("rebuild_int8", &exl3_i8_lab::rebuild_int8);
  m.impl("rebuild_fp16", &exl3_i8_lab::rebuild_fp16);
  m.impl("act_int8", &exl3_i8_lab::act_int8);
  m.impl("act_fp16", &exl3_i8_lab::act_fp16);
  m.impl("act_int8_dec", &exl3_i8_lab::act_int8_dec);
  m.impl("w8_decode_mm", &exl3_i8_lab::w8_decode_mm);
  m.impl("w8_decode_grouped_mm", &exl3_i8_lab::w8_decode_grouped_mm);
  m.impl("w8_decode_mm_at", &exl3_i8_lab::w8_decode_mm_at);
  m.impl("act_int8_dec_grouped", &exl3_i8_lab::act_int8_dec_grouped);
}
