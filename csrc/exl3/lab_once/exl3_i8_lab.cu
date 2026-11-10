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
#include <torch/library.h>
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

template <bool kInt8>
__global__ void __launch_bounds__(kThreads) act_kernel(
    const __nv_bfloat16* __restrict__ x, const half* __restrict__ suh, int64_t logical_in,
    int64_t padded_in, int8_t* __restrict__ xq, float* __restrict__ sx, half* __restrict__ xr) {
  extern __shared__ float row_buf[];
  __shared__ float wmax[kThreads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t r = blockIdx.x;
  float m = 0.0f;
  for (int64_t b = warp; b < padded_in / 128; b += kThreads / 32) {
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
  for (int w = 0; w < kThreads / 32; ++w) m = fmaxf(m, wmax[w]);
  const float inv = m > 0.0f ? 127.0f / m : 0.0f;
  for (int64_t col = threadIdx.x; col < padded_in; col += kThreads) {
    int q = __float2int_rn(__fmul_rn(row_buf[col], inv));
    q = min(max(q, -127), 127);
    xq[r * padded_in + col] = static_cast<int8_t>(q);
  }
  if (threadIdx.x == 0) sx[r] = m > 0.0f ? m / 127.0f : 1.0f;
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
}

TORCH_LIBRARY_IMPL(exl3_i8_lab, CUDA, m) {
  m.impl("colmax", &exl3_i8_lab::colmax);
  m.impl("rebuild_int8", &exl3_i8_lab::rebuild_int8);
  m.impl("rebuild_fp16", &exl3_i8_lab::rebuild_fp16);
  m.impl("act_int8", &exl3_i8_lab::act_int8);
  m.impl("act_fp16", &exl3_i8_lab::act_fp16);
}
