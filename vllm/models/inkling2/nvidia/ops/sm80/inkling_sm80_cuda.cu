// CUDA kernels for Inkling 2 Mini on sm_80 (Ampere / GA100, e.g. the CMP
// 170HX), loaded as the Python extension _inkling_sm80_cuda (see
// site/inkling_sm80_cuda.py; build with csrc/build.sh inside the vLLM image).
//   - conv/mix: fused decode kernels (288 threads for up to 3 stacked sources,
//   576 for 4-7) and a two-pass prefill kernel
//   - MoE routing: top-k select fused with block alignment for decode, a
//   counting-sort align for long prefills
//   - loop controller: batched bank messages (rank-2 coefficients instead of an
//   fp32 message tensor), pair norm fused
//     with the controller's down projection for one-token decode, fp32 -> bf16
//     RMS norm
//   - DSpark drafter: Markov RNN step, context KV, q/k norm + RoPE + cache
//   write, draft embedding, Markov fold + Gumbel
// Hot paths compile without stack frames or spills; results match the Triton /
// PyTorch paths they replace to bf16 rounding (most bit-exact).
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__device__ __forceinline__ float rsqrt_approx_f32(float x) {
  float r;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, off);
  return v;
}

__device__ __forceinline__ float block_sum_9w(float v, float* s_buf, int wid,
                                              int lane) {
  v = warp_sum(v);
  if (lane == 0) s_buf[wid] = v;
  __syncthreads();
  if (wid == 0) {
    float tot = (lane < 9) ? s_buf[lane] : 0.0f;
    tot += __shfl_xor_sync(0xffffffff, tot, 1);
    tot += __shfl_xor_sync(0xffffffff, tot, 2);
    tot += __shfl_xor_sync(0xffffffff, tot, 4);
    tot += __shfl_xor_sync(0xffffffff, tot, 8);
    if (lane == 0) s_buf[0] = tot;
  }
  __syncthreads();
  return s_buf[0];
}

__device__ __forceinline__ void unpack_bf16x8(const uint4& v, float (&out)[8]) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 f = __bfloat1622float2(p[i]);
    out[i * 2] = f.x;
    out[i * 2 + 1] = f.y;
  }
}

__device__ __forceinline__ uint4 pack_bf16x8(const float (&in)[8]) {
  uint4 v;
  __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    p[i] = __float22bfloat162_rn(make_float2(in[i * 2], in[i * 2 + 1]));
  }
  return v;
}

__device__ __forceinline__ void unpack_bf16x4(const uint2& v, float (&out)[4]) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
  float2 f0 = __bfloat1622float2(p[0]), f1 = __bfloat1622float2(p[1]);
  out[0] = f0.x;
  out[1] = f0.y;
  out[2] = f1.x;
  out[3] = f1.y;
}

__device__ __forceinline__ uint2 pack_bf16x4(const float (&in)[4]) {
  uint2 v;
  __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&v);
  p[0] = __float22bfloat162_rn(make_float2(in[0], in[1]));
  p[1] = __float22bfloat162_rn(make_float2(in[2], in[3]));
  return v;
}

// =========================================================================
// 1A. FUSED conv_mix (288 threads, small n_stack <= 3, D=2304, W=4)
// =========================================================================
template <int MAX_SRC, int NUM_Q, bool APPLY_NORM>
__global__ __launch_bounds__(288, 2) void sm80_conv_mix_2304_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ ConvW, __nv_bfloat16* __restrict__ Cache,
    const int64_t* __restrict__ Positions,
    const int32_t* __restrict__ BlockTable, const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ SlotMapping,
    const int32_t* __restrict__ QueryStart, __nv_bfloat16* __restrict__ ConvOut,
    const __nv_bfloat16* __restrict__ Residual,
    const __nv_bfloat16* __restrict__ Addend,
    const __nv_bfloat16* __restrict__ Stack,
    const __nv_bfloat16* __restrict__ Last, const float* __restrict__ Queries,
    const __nv_bfloat16* __restrict__ NormW, void* __restrict__ Out,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_h,
    int64_t stride_c_n, int64_t stride_bt, int64_t stride_s, int32_t T,
    int32_t n_stack, int32_t max_blocks, int32_t cache_N, int32_t stream_off,
    int32_t stream_w, float scale, float eps, float norm_eps) {
  constexpr int D = 2304;
  const int n_src = n_stack + 1;
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int c0 = tid * 8;

  const int head = c0 / stream_w;
  const int cache_col = head * (int)stride_c_h + stream_off + (c0 % stream_w);

  const int64_t slot = SlotMapping[t];
  const bool valid = slot >= 0;
  const int64_t pos = Positions[t];
  const int32_t req = SeqIdx[t];
  const int32_t qstart = QueryStart[t];

  uint4 cur_u4 =
      __ldg(reinterpret_cast<const uint4*>(X + (int64_t)t * stride_x + c0));
  if (valid) {
    int64_t dst_off = (slot / cache_N) * stride_c_blk +
                      (slot % cache_N) * stride_c_n + cache_col;
    *reinterpret_cast<uint4*>(Cache + dst_off) = cur_u4;
  }
  float cur_f[8];
  unpack_bf16x8(cur_u4, cur_f);

  const uint4* cw_ptr = reinterpret_cast<const uint4*>(ConvW + c0 * 4);
  float cw[8][4];
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    uint4 wu = __ldg(cw_ptr + p);
    const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wu);
    float2 f0 = __bfloat1622float2(wb[0]), f1 = __bfloat1622float2(wb[1]);
    float2 f2 = __bfloat1622float2(wb[2]), f3 = __bfloat1622float2(wb[3]);
    cw[p * 2][0] = f0.x;
    cw[p * 2][1] = f0.y;
    cw[p * 2][2] = f1.x;
    cw[p * 2][3] = f1.y;
    cw[p * 2 + 1][0] = f2.x;
    cw[p * 2 + 1][1] = f2.y;
    cw[p * 2 + 1][2] = f3.x;
    cw[p * 2 + 1][3] = f3.y;
  }

  float conv_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    int lag = 3 - k;
    int64_t src_pos = pos - lag;
    int32_t src_row = t - lag;
    float xt[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (valid && src_pos >= 0) {
      if (lag == 0) {
#pragma unroll
        for (int i = 0; i < 8; ++i) xt[i] = cur_f[i];
      } else if (src_row >= qstart) {
        uint4 xu = __ldg(reinterpret_cast<const uint4*>(
            X + (int64_t)src_row * stride_x + c0));
        unpack_bf16x8(xu, xt);
      } else {
        int32_t lblk = (int32_t)(src_pos / cache_N);
        if (lblk >= max_blocks) lblk = max_blocks - 1;
        int32_t blk = BlockTable[(int64_t)req * stride_bt + lblk];
        int64_t coff = (int64_t)blk * stride_c_blk +
                       (src_pos % cache_N) * stride_c_n + cache_col;
        uint4 cu = *reinterpret_cast<const uint4*>(Cache + coff);
        unpack_bf16x8(cu, xt);
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
      conv_acc[i] = __fmaf_rn(xt[i], cw[i][k], conv_acc[i]);
  }

  float y_f[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    y_f[i] = __bfloat162float(__float2bfloat16_rn(
        __fmul_rn(__fadd_rn(conv_acc[i], cur_f[i]), scale)));
  }
  if (Addend != nullptr) {
    uint4 au =
        __ldg(reinterpret_cast<const uint4*>(Addend + (int64_t)t * D + c0));
    float af[8];
    unpack_bf16x8(au, af);
    if (Residual != nullptr) {
      uint4 ru =
          __ldg(reinterpret_cast<const uint4*>(Residual + (int64_t)t * D + c0));
      float rf[8];
      unpack_bf16x8(ru, rf);
#pragma unroll
      for (int i = 0; i < 8; ++i) af[i] = __fadd_rn(rf[i], af[i]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
      y_f[i] = __bfloat162float(__float2bfloat16_rn(__fadd_rn(af[i], y_f[i])));
  }
  *reinterpret_cast<uint4*>(ConvOut + (int64_t)t * D + c0) = pack_bf16x8(y_f);

  float sx[MAX_SRC][8];
#pragma unroll
  for (int n = 0; n < MAX_SRC; ++n) {
    if (n < n_stack) {
      uint4 su = __ldg(reinterpret_cast<const uint4*>(
          Stack + (int64_t)n * stride_s + (int64_t)t * D + c0));
      unpack_bf16x8(su, sx[n]);
    } else if (n == n_stack) {
      if (Last != nullptr) {
        uint4 lu =
            __ldg(reinterpret_cast<const uint4*>(Last + (int64_t)t * D + c0));
        float lf[8];
        unpack_bf16x8(lu, lf);
#pragma unroll
        for (int i = 0; i < 8; ++i) sx[n][i] = __fadd_rn(lf[i], y_f[i]);
      } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) sx[n][i] = y_f[i];
      }
    }
  }

  float qv[NUM_Q][8];
#pragma unroll
  for (int q = 0; q < NUM_Q; ++q) {
    const float4* qp =
        reinterpret_cast<const float4*>(Queries + (int64_t)q * D + c0);
    float4 q0 = __ldg(qp), q1 = __ldg(qp + 1);
    qv[q][0] = q0.x;
    qv[q][1] = q0.y;
    qv[q][2] = q0.z;
    qv[q][3] = q0.w;
    qv[q][4] = q1.x;
    qv[q][5] = q1.y;
    qv[q][6] = q1.z;
    qv[q][7] = q1.w;
  }

  constexpr int N_RED = MAX_SRC * (1 + NUM_Q);
  __shared__ float s_red[9][N_RED];
  __shared__ float s_probs[NUM_Q][MAX_SRC];
  __shared__ float s_inv_sum[NUM_Q];

#pragma unroll
  for (int n = 0; n < MAX_SRC; ++n) {
    if (n < n_src) {
      float ss = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) ss = __fmaf_rn(sx[n][i], sx[n][i], ss);
      ss = warp_sum(ss);
      if (lane == 0) s_red[wid][n] = ss;
#pragma unroll
      for (int q = 0; q < NUM_Q; ++q) {
        float dt = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) dt = __fmaf_rn(qv[q][i], sx[n][i], dt);
        dt = warp_sum(dt);
        if (lane == 0) s_red[wid][MAX_SRC + q * MAX_SRC + n] = dt;
      }
    }
  }
  __syncthreads();

  if (wid == 0) {
    float my_rstd = 0.0f;
    if (lane < n_src) {
      float ss = 0.0f;
#pragma unroll
      for (int w = 0; w < 9; ++w) ss += s_red[w][lane];
      my_rstd = rsqrt_approx_f32(__fmaf_rn(ss, 1.0f / (float)D, eps));
    }
#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
      float lg = -1e38f;
      if (lane < n_src) {
        float dt = 0.0f;
#pragma unroll
        for (int w = 0; w < 9; ++w)
          dt += s_red[w][MAX_SRC + q * MAX_SRC + lane];
        lg = __fmul_rn(dt, my_rstd);
      }
      float mx = lg;
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      float pr = (lane < n_src) ? expf(lg - mx) : 0.0f;
      float psum = pr;
      psum += __shfl_xor_sync(0xffffffff, psum, 4);
      psum += __shfl_xor_sync(0xffffffff, psum, 2);
      psum += __shfl_xor_sync(0xffffffff, psum, 1);
      if (lane < MAX_SRC)
        s_probs[q][lane] = (lane < n_src) ? __fmul_rn(pr, my_rstd) : 0.0f;
      if (lane == 0) s_inv_sum[q] = __fdiv_rn(1.0f, psum);
    }
  }
  __syncthreads();

  if constexpr (APPLY_NORM) {
    __shared__ float s_norm[NUM_Q][9];
    __shared__ float s_rstd[NUM_Q];
    float acc[NUM_Q][8];
#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[q][i] = 0.0f;
#pragma unroll
      for (int n = 0; n < MAX_SRC; ++n) {
        if (n < n_src) {
          float p = s_probs[q][n];
#pragma unroll
          for (int i = 0; i < 8; ++i)
            acc[q][i] = __fmaf_rn(p, sx[n][i], acc[q][i]);
        }
      }
      float inv_s = s_inv_sum[q];
      float ss = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float v = __fmul_rn(acc[q][i], inv_s);
        acc[q][i] = v;
        ss = __fmaf_rn(v, v, ss);
      }
      ss = warp_sum(ss);
      if (lane == 0) s_norm[q][wid] = ss;
    }
    __syncthreads();
    if (wid == 0) {
#pragma unroll
      for (int q = 0; q < NUM_Q; ++q) {
        float tot = (lane < 9) ? s_norm[q][lane] : 0.0f;
        tot += __shfl_xor_sync(0xffffffff, tot, 1);
        tot += __shfl_xor_sync(0xffffffff, tot, 2);
        tot += __shfl_xor_sync(0xffffffff, tot, 4);
        tot += __shfl_xor_sync(0xffffffff, tot, 8);
        if (lane == 0)
          s_rstd[q] =
              rsqrt_approx_f32(__fadd_rn(__fdiv_rn(tot, (float)D), norm_eps));
      }
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
      float out_rstd = s_rstd[q];
      uint4 nwu =
          __ldg(reinterpret_cast<const uint4*>(NormW + (int64_t)q * D + c0));
      float nw[8];
      unpack_bf16x8(nwu, nw);
#pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[q][i] = __fmul_rn(__fmul_rn(acc[q][i], out_rstd), nw[i]);
      __nv_bfloat16* out_bf16 =
          reinterpret_cast<__nv_bfloat16*>(Out) + ((int64_t)q * T + t) * D + c0;
      *reinterpret_cast<uint4*>(out_bf16) = pack_bf16x8(acc[q]);
    }
  } else {
#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
      float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int n = 0; n < MAX_SRC; ++n) {
        if (n < n_src) {
          float p = s_probs[q][n];
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[i] = __fmaf_rn(p, sx[n][i], acc[i]);
        }
      }
      float inv_s = s_inv_sum[q];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = __fmul_rn(acc[i], inv_s);
      float* out_f32 =
          reinterpret_cast<float*>(Out) + ((int64_t)q * T + t) * D + c0;
      reinterpret_cast<float4*>(out_f32)[0] =
          make_float4(acc[0], acc[1], acc[2], acc[3]);
      reinterpret_cast<float4*>(out_f32)[1] =
          make_float4(acc[4], acc[5], acc[6], acc[7]);
    }
  }
}

// =========================================================================
// 1B. 576-THREAD zero-SPILL conv_mix (18 warps x 4 cols, n_stack = 4..7)
//     1.31-1.34x faster than 288-thread on S=4..7 for both Q=1 and Q=3
// =========================================================================
template <int MAX_SRC, int NUM_Q, bool APPLY_NORM>
__global__ __launch_bounds__(576, 2) void sm80_conv_mix_576_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ ConvW, __nv_bfloat16* __restrict__ Cache,
    const int64_t* __restrict__ Positions,
    const int32_t* __restrict__ BlockTable, const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ SlotMapping,
    const int32_t* __restrict__ QueryStart, __nv_bfloat16* __restrict__ ConvOut,
    const __nv_bfloat16* __restrict__ Residual,
    const __nv_bfloat16* __restrict__ Addend,
    const __nv_bfloat16* __restrict__ Stack,
    const __nv_bfloat16* __restrict__ Last, const float* __restrict__ Queries,
    const __nv_bfloat16* __restrict__ NormW, void* __restrict__ Out,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_h,
    int64_t stride_c_n, int64_t stride_bt, int64_t stride_s, int32_t T,
    int32_t n_stack, int32_t max_blocks, int32_t cache_N, int32_t stream_off,
    int32_t stream_w, float scale, float eps, float norm_eps) {
  constexpr int D = 2304;
  const int n_src = n_stack + 1;
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int c0 = tid * 4;

  const int head = c0 / stream_w;
  const int cache_col = head * (int)stride_c_h + stream_off + (c0 % stream_w);

  const int64_t slot = SlotMapping[t];
  const bool valid = slot >= 0;
  const int64_t pos = Positions[t];
  const int32_t req = SeqIdx[t];
  const int32_t qstart = QueryStart[t];

  uint2 cur_u2 =
      __ldg(reinterpret_cast<const uint2*>(X + (int64_t)t * stride_x + c0));
  if (valid) {
    int64_t dst_off = (slot / cache_N) * stride_c_blk +
                      (slot % cache_N) * stride_c_n + cache_col;
    *reinterpret_cast<uint2*>(Cache + dst_off) = cur_u2;
  }
  float cur_f[4];
  unpack_bf16x4(cur_u2, cur_f);

  const uint4* cw_ptr = reinterpret_cast<const uint4*>(ConvW + c0 * 4);
  float cw[4][4];
#pragma unroll
  for (int p = 0; p < 2; ++p) {
    uint4 wu = __ldg(cw_ptr + p);
    const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wu);
    float2 f0 = __bfloat1622float2(wb[0]), f1 = __bfloat1622float2(wb[1]);
    float2 f2 = __bfloat1622float2(wb[2]), f3 = __bfloat1622float2(wb[3]);
    cw[p * 2][0] = f0.x;
    cw[p * 2][1] = f0.y;
    cw[p * 2][2] = f1.x;
    cw[p * 2][3] = f1.y;
    cw[p * 2 + 1][0] = f2.x;
    cw[p * 2 + 1][1] = f2.y;
    cw[p * 2 + 1][2] = f3.x;
    cw[p * 2 + 1][3] = f3.y;
  }

  float conv_acc[4] = {0, 0, 0, 0};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    int lag = 3 - k;
    int64_t src_pos = pos - lag;
    int32_t src_row = t - lag;
    float xt[4] = {0, 0, 0, 0};
    if (valid && src_pos >= 0) {
      if (lag == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) xt[i] = cur_f[i];
      } else if (src_row >= qstart) {
        uint2 xu = __ldg(reinterpret_cast<const uint2*>(
            X + (int64_t)src_row * stride_x + c0));
        unpack_bf16x4(xu, xt);
      } else {
        int32_t lblk = (int32_t)(src_pos / cache_N);
        if (lblk >= max_blocks) lblk = max_blocks - 1;
        int32_t blk = BlockTable[(int64_t)req * stride_bt + lblk];
        int64_t coff = (int64_t)blk * stride_c_blk +
                       (src_pos % cache_N) * stride_c_n + cache_col;
        uint2 cu = *reinterpret_cast<const uint2*>(Cache + coff);
        unpack_bf16x4(cu, xt);
      }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
      conv_acc[i] = __fmaf_rn(xt[i], cw[i][k], conv_acc[i]);
  }

  float y_f[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    y_f[i] = __bfloat162float(__float2bfloat16_rn(
        __fmul_rn(__fadd_rn(conv_acc[i], cur_f[i]), scale)));
  if (Addend != nullptr) {
    uint2 au =
        __ldg(reinterpret_cast<const uint2*>(Addend + (int64_t)t * D + c0));
    float af[4];
    unpack_bf16x4(au, af);
    if (Residual != nullptr) {
      uint2 ru =
          __ldg(reinterpret_cast<const uint2*>(Residual + (int64_t)t * D + c0));
      float rf[4];
      unpack_bf16x4(ru, rf);
#pragma unroll
      for (int i = 0; i < 4; ++i) af[i] = __fadd_rn(rf[i], af[i]);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
      y_f[i] = __bfloat162float(__float2bfloat16_rn(__fadd_rn(af[i], y_f[i])));
  }
  *reinterpret_cast<uint2*>(ConvOut + (int64_t)t * D + c0) = pack_bf16x4(y_f);

  float sx[MAX_SRC][4];
#pragma unroll
  for (int n = 0; n < MAX_SRC; ++n) {
    if (n < n_stack) {
      uint2 su = __ldg(reinterpret_cast<const uint2*>(
          Stack + (int64_t)n * stride_s + (int64_t)t * D + c0));
      unpack_bf16x4(su, sx[n]);
    } else if (n == n_stack) {
      if (Last != nullptr) {
        uint2 lu =
            __ldg(reinterpret_cast<const uint2*>(Last + (int64_t)t * D + c0));
        float lf[4];
        unpack_bf16x4(lu, lf);
#pragma unroll
        for (int i = 0; i < 4; ++i) sx[n][i] = __fadd_rn(lf[i], y_f[i]);
      } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) sx[n][i] = y_f[i];
      }
    }
  }

  float qv[NUM_Q][4];
#pragma unroll
  for (int q = 0; q < NUM_Q; ++q) {
    float4 q0 =
        __ldg(reinterpret_cast<const float4*>(Queries + (int64_t)q * D + c0));
    qv[q][0] = q0.x;
    qv[q][1] = q0.y;
    qv[q][2] = q0.z;
    qv[q][3] = q0.w;
  }

  constexpr int N_RED = MAX_SRC * (1 + NUM_Q);
  __shared__ float s_red[18][N_RED];
  __shared__ float s_probs[NUM_Q][MAX_SRC];
  __shared__ float s_inv_sum[NUM_Q];

#pragma unroll
  for (int n = 0; n < MAX_SRC; ++n) {
    if (n < n_src) {
      float ss = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) ss = __fmaf_rn(sx[n][i], sx[n][i], ss);
      ss = warp_sum(ss);
      if (lane == 0) s_red[wid][n] = ss;
#pragma unroll
      for (int q = 0; q < NUM_Q; ++q) {
        float dt = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) dt = __fmaf_rn(qv[q][i], sx[n][i], dt);
        dt = warp_sum(dt);
        if (lane == 0) s_red[wid][MAX_SRC + q * MAX_SRC + n] = dt;
      }
    }
  }
  __syncthreads();

  if (wid == 0) {
    float my_rstd = 0.0f;
    if (lane < n_src) {
      float ss = 0.0f;
#pragma unroll
      for (int w = 0; w < 18; ++w) ss += s_red[w][lane];
      my_rstd = rsqrt_approx_f32(__fmaf_rn(ss, 1.0f / (float)D, eps));
    }
#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
      float lg = -1e38f;
      if (lane < n_src) {
        float dt = 0.0f;
#pragma unroll
        for (int w = 0; w < 18; ++w)
          dt += s_red[w][MAX_SRC + q * MAX_SRC + lane];
        lg = __fmul_rn(dt, my_rstd);
      }
      float mx = lg;
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      float pr = (lane < n_src) ? expf(lg - mx) : 0.0f;
      float psum = pr;
      psum += __shfl_xor_sync(0xffffffff, psum, 4);
      psum += __shfl_xor_sync(0xffffffff, psum, 2);
      psum += __shfl_xor_sync(0xffffffff, psum, 1);
      if (lane < MAX_SRC)
        s_probs[q][lane] = (lane < n_src) ? __fmul_rn(pr, my_rstd) : 0.0f;
      if (lane == 0) s_inv_sum[q] = __fdiv_rn(1.0f, psum);
    }
  }
  __syncthreads();

  if constexpr (APPLY_NORM) {
    __shared__ float s_norm[NUM_Q][18];
    __shared__ float s_rstd[NUM_Q];
    float acc[NUM_Q][4];
#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
      acc[q][0] = 0.0f;
      acc[q][1] = 0.0f;
      acc[q][2] = 0.0f;
      acc[q][3] = 0.0f;
#pragma unroll
      for (int n = 0; n < MAX_SRC; ++n) {
        if (n < n_src) {
          float p = s_probs[q][n];
#pragma unroll
          for (int i = 0; i < 4; ++i)
            acc[q][i] = __fmaf_rn(p, sx[n][i], acc[q][i]);
        }
      }
      float inv_s = s_inv_sum[q];
      float ss = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float v = __fmul_rn(acc[q][i], inv_s);
        acc[q][i] = v;
        ss = __fmaf_rn(v, v, ss);
      }
      ss = warp_sum(ss);
      if (lane == 0) s_norm[q][wid] = ss;
    }
    __syncthreads();
    if (wid == 0) {
#pragma unroll
      for (int q = 0; q < NUM_Q; ++q) {
        float tot = (lane < 18) ? s_norm[q][lane] : 0.0f;
        tot = warp_sum(tot);
        if (lane == 0)
          s_rstd[q] =
              rsqrt_approx_f32(__fadd_rn(__fdiv_rn(tot, (float)D), norm_eps));
      }
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
      float out_rstd = s_rstd[q];
      uint2 nwu =
          __ldg(reinterpret_cast<const uint2*>(NormW + (int64_t)q * D + c0));
      float nw[4];
      unpack_bf16x4(nwu, nw);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc[q][i] = __fmul_rn(__fmul_rn(acc[q][i], out_rstd), nw[i]);
      __nv_bfloat16* ob =
          reinterpret_cast<__nv_bfloat16*>(Out) + ((int64_t)q * T + t) * D + c0;
      *reinterpret_cast<uint2*>(ob) = pack_bf16x4(acc[q]);
    }
  } else {
#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
      float acc[4] = {0, 0, 0, 0};
#pragma unroll
      for (int n = 0; n < MAX_SRC; ++n) {
        if (n < n_src) {
          float p = s_probs[q][n];
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[i] = __fmaf_rn(p, sx[n][i], acc[i]);
        }
      }
      float inv_s = s_inv_sum[q];
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[i] = __fmul_rn(acc[i], inv_s);
      float* of = reinterpret_cast<float*>(Out) + ((int64_t)q * T + t) * D + c0;
      *reinterpret_cast<float4*>(of) =
          make_float4(acc[0], acc[1], acc[2], acc[3]);
    }
  }
}

// =========================================================================
// 1C. 2-PASS zero-SPILL 1-BARRIER INTRA-FAST-PATH PREFILL conv_mix
//     Covers both Q == 1 and Q == 3 for T > 256 (1.33-1.73x faster than Triton)
// =========================================================================
template <int NUM_Q, bool APPLY_NORM>
__global__ __launch_bounds__(288, 2) void sm80_prefill_conv_mix_2304_kernel(
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ ConvW, __nv_bfloat16* __restrict__ Cache,
    const int64_t* __restrict__ Positions,
    const int32_t* __restrict__ BlockTable, const int32_t* __restrict__ SeqIdx,
    const int64_t* __restrict__ SlotMapping,
    const int32_t* __restrict__ QueryStart, __nv_bfloat16* __restrict__ ConvOut,
    const __nv_bfloat16* __restrict__ Residual,
    const __nv_bfloat16* __restrict__ Addend,
    const __nv_bfloat16* __restrict__ Stack,
    const __nv_bfloat16* __restrict__ Last, const float* __restrict__ Queries,
    const __nv_bfloat16* __restrict__ NormW, void* __restrict__ Out,
    int64_t stride_x, int64_t stride_c_blk, int64_t stride_c_h,
    int64_t stride_c_n, int64_t stride_bt, int64_t stride_s, int32_t T,
    int32_t n_stack, int32_t max_blocks, int32_t cache_N, int32_t stream_off,
    int32_t stream_w, float scale, float eps, float norm_eps) {
  constexpr int D = 2304;
  const int n_src = n_stack + 1;
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int c0 = tid * 8;

  const int head = c0 / stream_w;
  const int cache_col = head * (int)stride_c_h + stream_off + (c0 % stream_w);

  const int64_t slot = SlotMapping[t];
  const bool valid = slot >= 0;
  const int64_t pos = Positions[t];
  const int32_t req = SeqIdx[t];
  const int32_t qstart = QueryStart[t];

  uint4 cur_u4 =
      __ldg(reinterpret_cast<const uint4*>(X + (int64_t)t * stride_x + c0));
  if (valid) {
    int64_t dst_off = (slot / cache_N) * stride_c_blk +
                      (slot % cache_N) * stride_c_n + cache_col;
    *reinterpret_cast<uint4*>(Cache + dst_off) = cur_u4;
  }
  float cur_f[8];
  unpack_bf16x8(cur_u4, cur_f);

  const uint4* cw_ptr = reinterpret_cast<const uint4*>(ConvW + c0 * 4);
  float conv_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  // Prefill fast path: nearly all tokens have t - 3 >= qstart -> zero cache
  // reads
  if (valid && pos >= 3 && (t - 3 >= qstart)) {
    float x3[8], x2[8], x1[8];
    unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(
                      X + (int64_t)(t - 3) * stride_x + c0)),
                  x3);
    unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(
                      X + (int64_t)(t - 2) * stride_x + c0)),
                  x2);
    unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(
                      X + (int64_t)(t - 1) * stride_x + c0)),
                  x1);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      uint4 wu = __ldg(cw_ptr + p);
      const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wu);
      float2 w0 = __bfloat1622float2(wb[0]), w1 = __bfloat1622float2(wb[1]);
      float2 w2 = __bfloat1622float2(wb[2]), w3 = __bfloat1622float2(wb[3]);
      conv_acc[p * 2] =
          __fmaf_rn(cur_f[p * 2], w1.y,
                    __fmaf_rn(x1[p * 2], w1.x,
                              __fmaf_rn(x2[p * 2], w0.y, x3[p * 2] * w0.x)));
      conv_acc[p * 2 + 1] = __fmaf_rn(
          cur_f[p * 2 + 1], w3.y,
          __fmaf_rn(x1[p * 2 + 1], w3.x,
                    __fmaf_rn(x2[p * 2 + 1], w2.y, x3[p * 2 + 1] * w2.x)));
    }
  } else {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      int lag = 3 - k;
      int64_t src_pos = pos - lag;
      int32_t src_row = t - lag;
      float xt[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      if (valid && src_pos >= 0) {
        if (lag == 0) {
#pragma unroll
          for (int i = 0; i < 8; ++i) xt[i] = cur_f[i];
        } else if (src_row >= qstart) {
          unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(
                            X + (int64_t)src_row * stride_x + c0)),
                        xt);
        } else {
          int32_t lblk = (int32_t)(src_pos / cache_N);
          if (lblk >= max_blocks) lblk = max_blocks - 1;
          int32_t blk = BlockTable[(int64_t)req * stride_bt + lblk];
          int64_t coff = (int64_t)blk * stride_c_blk +
                         (src_pos % cache_N) * stride_c_n + cache_col;
          unpack_bf16x8(*reinterpret_cast<const uint4*>(Cache + coff), xt);
        }
      }
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        uint4 wu = __ldg(cw_ptr + p);
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wu);
        conv_acc[p * 2] =
            __fmaf_rn(xt[p * 2], __bfloat162float(wb[k]), conv_acc[p * 2]);
        conv_acc[p * 2 + 1] = __fmaf_rn(
            xt[p * 2 + 1], __bfloat162float(wb[4 + k]), conv_acc[p * 2 + 1]);
      }
    }
  }

  float y_f[8];
#pragma unroll
  for (int i = 0; i < 8; ++i)
    y_f[i] = __bfloat162float(__float2bfloat16_rn(
        __fmul_rn(__fadd_rn(conv_acc[i], cur_f[i]), scale)));
  if (Addend != nullptr) {
    uint4 au =
        __ldg(reinterpret_cast<const uint4*>(Addend + (int64_t)t * D + c0));
    float af[8];
    unpack_bf16x8(au, af);
    if (Residual != nullptr) {
      uint4 ru =
          __ldg(reinterpret_cast<const uint4*>(Residual + (int64_t)t * D + c0));
      float rf[8];
      unpack_bf16x8(ru, rf);
#pragma unroll
      for (int i = 0; i < 8; ++i) af[i] = __fadd_rn(rf[i], af[i]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
      y_f[i] = __bfloat162float(__float2bfloat16_rn(__fadd_rn(af[i], y_f[i])));
  }
  *reinterpret_cast<uint4*>(ConvOut + (int64_t)t * D + c0) = pack_bf16x8(y_f);

  if (Last != nullptr) {
    uint4 lu =
        __ldg(reinterpret_cast<const uint4*>(Last + (int64_t)t * D + c0));
    float lf[8];
    unpack_bf16x8(lu, lf);
#pragma unroll
    for (int i = 0; i < 8; ++i) y_f[i] = __fadd_rn(lf[i], y_f[i]);
  }

  float qv[NUM_Q][8];
#pragma unroll
  for (int q = 0; q < NUM_Q; ++q) {
    const float4* qp =
        reinterpret_cast<const float4*>(Queries + (int64_t)q * D + c0);
    float4 q0 = __ldg(qp), q1 = __ldg(qp + 1);
    qv[q][0] = q0.x;
    qv[q][1] = q0.y;
    qv[q][2] = q0.z;
    qv[q][3] = q0.w;
    qv[q][4] = q1.x;
    qv[q][5] = q1.y;
    qv[q][6] = q1.z;
    qv[q][7] = q1.w;
  }

  // Pass 1: zero barriers inside source loop. Single __syncthreads() + warp-0
  // parallel softmax
  __shared__ float s_red[9][8 * (1 + NUM_Q)];
  __shared__ float s_probs[NUM_Q][8];
  __shared__ float s_inv_sum[NUM_Q];

  for (int n = 0; n < n_src; ++n) {
    float sx[8];
    if (n < n_stack) {
      uint4 su = __ldg(reinterpret_cast<const uint4*>(
          Stack + (int64_t)n * stride_s + (int64_t)t * D + c0));
      unpack_bf16x8(su, sx);
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) sx[i] = y_f[i];
    }
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) ss = __fmaf_rn(sx[i], sx[i], ss);
    ss = warp_sum(ss);
    if (lane == 0) s_red[wid][n] = ss;
#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
      float dt = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) dt = __fmaf_rn(qv[q][i], sx[i], dt);
      dt = warp_sum(dt);
      if (lane == 0) s_red[wid][8 + q * 8 + n] = dt;
    }
  }
  __syncthreads();

  if (wid == 0) {
    float my_rstd = 0.0f;
    if (lane < n_src) {
      float ss = 0.0f;
#pragma unroll
      for (int w = 0; w < 9; ++w) ss += s_red[w][lane];
      my_rstd = rsqrt_approx_f32(__fmaf_rn(ss, 1.0f / 2304.0f, eps));
    }
#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
      float lg = -1e38f;
      if (lane < n_src) {
        float dt = 0.0f;
#pragma unroll
        for (int w = 0; w < 9; ++w) dt += s_red[w][8 + q * 8 + lane];
        lg = __fmul_rn(dt, my_rstd);
      }
      float mx = lg;
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      float pr = (lane < n_src) ? expf(lg - mx) : 0.0f;
      float psum = pr;
      psum += __shfl_xor_sync(0xffffffff, psum, 4);
      psum += __shfl_xor_sync(0xffffffff, psum, 2);
      psum += __shfl_xor_sync(0xffffffff, psum, 1);
      if (lane < 8)
        s_probs[q][lane] = (lane < n_src) ? __fmul_rn(pr, my_rstd) : 0.0f;
      if (lane == 0) s_inv_sum[q] = __fdiv_rn(1.0f, psum);
    }
  }
  __syncthreads();

  float mix[NUM_Q][8];
#pragma unroll
  for (int q = 0; q < NUM_Q; ++q) {
    float pw = s_probs[q][n_stack];
#pragma unroll
    for (int i = 0; i < 8; ++i) mix[q][i] = pw * y_f[i];
  }
  for (int n = 0; n < n_stack; ++n) {
    uint4 su = __ldg(reinterpret_cast<const uint4*>(
        Stack + (int64_t)n * stride_s + (int64_t)t * D + c0));
    float sx[8];
    unpack_bf16x8(su, sx);
#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
      float pw = s_probs[q][n];
#pragma unroll
      for (int i = 0; i < 8; ++i) mix[q][i] = __fmaf_rn(pw, sx[i], mix[q][i]);
    }
  }

  __shared__ float s_nsq[9];
#pragma unroll
  for (int q = 0; q < NUM_Q; ++q) {
    float inv_s = s_inv_sum[q];
#pragma unroll
    for (int i = 0; i < 8; ++i) mix[q][i] *= inv_s;
    if constexpr (APPLY_NORM) {
      float sq = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) sq = __fmaf_rn(mix[q][i], mix[q][i], sq);
      float tot = block_sum_9w(sq, s_nsq, wid, lane);
      float rstd_o = rsqrt_approx_f32(__fdiv_rn(tot, 2304.0f) + norm_eps);
      uint4 nwu =
          __ldg(reinterpret_cast<const uint4*>(NormW + (int64_t)q * D + c0));
      float nwf[8];
      unpack_bf16x8(nwu, nwf);
#pragma unroll
      for (int i = 0; i < 8; ++i) mix[q][i] = (mix[q][i] * rstd_o) * nwf[i];
      __nv_bfloat16* ob =
          reinterpret_cast<__nv_bfloat16*>(Out) + ((int64_t)q * T + t) * D + c0;
      *reinterpret_cast<uint4*>(ob) = pack_bf16x8(mix[q]);
    } else {
      float* of = reinterpret_cast<float*>(Out) + ((int64_t)q * T + t) * D + c0;
      reinterpret_cast<float4*>(of)[0] =
          make_float4(mix[q][0], mix[q][1], mix[q][2], mix[q][3]);
      reinterpret_cast<float4*>(of)[1] =
          make_float4(mix[q][4], mix[q][5], mix[q][6], mix[q][7]);
    }
  }
}

// =========================================================================
// 2. BATCHED BANK MESSAGES v2 (grid=(48, T), replaces 12x attn_res per loop
// pass)
// =========================================================================
__global__ __launch_bounds__(288, 2) void sm80_batched_bank_messages_v2_kernel(
    const __nv_bfloat16* __restrict__ Bank, const float* __restrict__ Queries,
    float* __restrict__ Out, int64_t stride_bank_0, int32_t T, float eps) {
  constexpr int D = 2304;
  const int q_idx = blockIdx.x;
  const int t = blockIdx.y;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int c0 = tid * 8;

  uint4 u0 = __ldg(reinterpret_cast<const uint4*>(Bank + (int64_t)t * D + c0));
  uint4 u1 = __ldg(reinterpret_cast<const uint4*>(Bank + stride_bank_0 +
                                                  (int64_t)t * D + c0));
  float b0[8], b1[8];
  unpack_bf16x8(u0, b0);
  unpack_bf16x8(u1, b1);

  const float4* qp =
      reinterpret_cast<const float4*>(Queries + (int64_t)q_idx * D + c0);
  float4 qa = __ldg(qp), qb = __ldg(qp + 1);
  float qv[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};

  float ss0 = 0.0f, ss1 = 0.0f, d0 = 0.0f, d1 = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    ss0 = __fmaf_rn(b0[i], b0[i], ss0);
    ss1 = __fmaf_rn(b1[i], b1[i], ss1);
    d0 = __fmaf_rn(qv[i], b0[i], d0);
    d1 = __fmaf_rn(qv[i], b1[i], d1);
  }
  ss0 = warp_sum(ss0);
  ss1 = warp_sum(ss1);
  d0 = warp_sum(d0);
  d1 = warp_sum(d1);

  __shared__ float s_red[9][4];
  if (lane == 0) {
    s_red[wid][0] = ss0;
    s_red[wid][1] = ss1;
    s_red[wid][2] = d0;
    s_red[wid][3] = d1;
  }
  __syncthreads();
  if (tid == 0) {
    float t0 = 0, t1 = 0, td0 = 0, td1 = 0;
#pragma unroll
    for (int w = 0; w < 9; ++w) {
      t0 += s_red[w][0];
      t1 += s_red[w][1];
      td0 += s_red[w][2];
      td1 += s_red[w][3];
    }
    float r0 = rsqrt_approx_f32(__fmaf_rn(t0, 1.0f / (float)D, eps));
    float r1 = rsqrt_approx_f32(__fmaf_rn(t1, 1.0f / (float)D, eps));
    float l0 = __fmul_rn(td0, r0), l1 = __fmul_rn(td1, r1);
    float mx = fmaxf(l0, l1);
    float p0 = expf(l0 - mx), p1 = expf(l1 - mx);
    float inv_s = __fdiv_rn(1.0f, p0 + p1);
    s_red[0][0] = __fmul_rn(__fmul_rn(p0, r0), inv_s);
    s_red[0][1] = __fmul_rn(__fmul_rn(p1, r1), inv_s);
  }
  __syncthreads();
  float w0 = s_red[0][0], w1 = s_red[0][1];
  float4* op =
      reinterpret_cast<float4*>(Out + ((int64_t)q_idx * T + t) * D + c0);
  op[0] = make_float4(w0 * b0[0] + w1 * b1[0], w0 * b0[1] + w1 * b1[1],
                      w0 * b0[2] + w1 * b1[2], w0 * b0[3] + w1 * b1[3]);
  op[1] = make_float4(w0 * b0[4] + w1 * b1[4], w0 * b0[5] + w1 * b1[5],
                      w0 * b0[6] + w1 * b1[6], w0 * b0[7] + w1 * b1[7]);
}

// =========================================================================
// 2B. PREFILL RANK-2 BANK MESSAGES (stats + finalize + pair_rms_norm_rank2)
//     Replaces 7.2 GB/chunk fp32 message tensor with 3 MB rank-2 coefficients
//     38.3x faster bank messages + 1.14-1.23x faster pair_rms_norm (0 spills).
// =========================================================================
__global__ __launch_bounds__(288, 2) void sm80_bank_stats_kernel(
    const __nv_bfloat16* __restrict__ Bank, float* __restrict__ Stats,
    int64_t stride_b0, int32_t T, float eps) {
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int c0 = tid * 8;

  uint4 u0 =
      __ldg(reinterpret_cast<const uint4*>(Bank + (int64_t)t * 2304 + c0));
  uint4 u1 = __ldg(reinterpret_cast<const uint4*>(Bank + stride_b0 +
                                                  (int64_t)t * 2304 + c0));
  float b0[8], b1[8];
  unpack_bf16x8(u0, b0);
  unpack_bf16x8(u1, b1);

  float s00 = 0.0f, s01 = 0.0f, s11 = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    s00 = __fmaf_rn(b0[i], b0[i], s00);
    s01 = __fmaf_rn(b0[i], b1[i], s01);
    s11 = __fmaf_rn(b1[i], b1[i], s11);
  }
  s00 = warp_sum(s00);
  s01 = warp_sum(s01);
  s11 = warp_sum(s11);

  __shared__ float s_red[9][3];
  if (lane == 0) {
    s_red[wid][0] = s00;
    s_red[wid][1] = s01;
    s_red[wid][2] = s11;
  }
  __syncthreads();
  if (tid == 0) {
    float t00 = 0.0f, t01 = 0.0f, t11 = 0.0f;
#pragma unroll
    for (int w = 0; w < 9; ++w) {
      t00 += s_red[w][0];
      t01 += s_red[w][1];
      t11 += s_red[w][2];
    }
    constexpr float invD = 1.0f / 2304.0f;
    float m00 = t00 * invD;
    float m01 = t01 * invD;
    float m11 = t11 * invD;
    float r0 = rsqrt_approx_f32(m00 + eps);
    float r1 = rsqrt_approx_f32(m11 + eps);
    float* dst = Stats + (int64_t)t * 5;
    dst[0] = r0;
    dst[1] = r1;
    dst[2] = m00;
    dst[3] = m01;
    dst[4] = m11;
  }
}

__global__ __launch_bounds__(256) void sm80_bank_rank2_finalize_kernel(
    const float* __restrict__ Dots, const float* __restrict__ Stats,
    float* __restrict__ CRaw, float* __restrict__ CNorm, int32_t T,
    int32_t Q_TOT, float eps) {
  const int t = blockIdx.x * 256 + threadIdx.x;
  const int q = blockIdx.y;
  if (t >= T) return;
  const int64_t stride_d = (int64_t)2 * Q_TOT;
  const float d0 =
      Dots[(int64_t)t * stride_d + q] + Dots[(int64_t)t * stride_d + Q_TOT + q];
  const float d1 = Dots[(int64_t)(T + t) * stride_d + q] +
                   Dots[(int64_t)(T + t) * stride_d + Q_TOT + q];
  const float* st = Stats + (int64_t)t * 5;
  const float r0 = st[0], r1 = st[1], m00 = st[2], m01 = st[3], m11 = st[4];
  const float l0 = __fmul_rn(d0, r0);
  const float l1 = __fmul_rn(d1, r1);
  const float mx = fmaxf(l0, l1);
  const float p0 = expf(l0 - mx);
  const float p1 = expf(l1 - mx);
  const float inv_s = __fdiv_rn(1.0f, p0 + p1);
  const float c0 = __fmul_rn(__fmul_rn(p0, r0), inv_s);
  const float c1 = __fmul_rn(__fmul_rn(p1, r1), inv_s);
  const float msg_ms =
      fmaxf(0.0f, c0 * c0 * m00 + 2.0f * c0 * c1 * m01 + c1 * c1 * m11);
  const float msg_rstd = rsqrt_approx_f32(msg_ms + eps);
  const int64_t out_off = ((int64_t)q * T + t) * 2;
  *reinterpret_cast<float2*>(CRaw + out_off) = make_float2(c0, c1);
  *reinterpret_cast<float2*>(CNorm + out_off) =
      make_float2(c0 * msg_rstd, c1 * msg_rstd);
}

__global__
__launch_bounds__(576, 2) void sm80_pair_rms_norm_rank2_bcast3_kernel(
    const float* __restrict__ Dest, const __nv_bfloat16* __restrict__ Bank,
    const float* __restrict__ CNorm, __nv_bfloat16* __restrict__ Out,
    int64_t stride_b0, int32_t T, float eps) {
  // Specialized for S == 3 with stride_dest_s == 0 (10/12 loop core layers:
  // local attn): Loads dest[t], b0[t], b1[t] once and emits all 3 streams
  // -> 1.23x faster
  const int t = blockIdx.x;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int c0 = tid * 4;

  float4 d4 =
      __ldg(reinterpret_cast<const float4*>(Dest + (int64_t)t * 2304 + c0));
  float sq = d4.x * d4.x + d4.y * d4.y + d4.z * d4.z + d4.w * d4.w;
  sq = warp_sum(sq);
  __shared__ float s_norm[18];
  if (lane == 0) s_norm[wid] = sq;
  __syncthreads();
  if (wid == 0) {
    float tot = (lane < 18) ? s_norm[lane] : 0.0f;
    tot = warp_sum(tot);
    if (lane == 0) s_norm[0] = rsqrt_approx_f32(__fdiv_rn(tot, 2304.0f) + eps);
  }
  __syncthreads();
  const float rd = s_norm[0];
  float dn[4] = {d4.x * rd, d4.y * rd, d4.z * rd, d4.w * rd};
  uint2 dn_u2 = pack_bf16x4(dn);

  uint2 u0 =
      __ldg(reinterpret_cast<const uint2*>(Bank + (int64_t)t * 2304 + c0));
  uint2 u1 = __ldg(reinterpret_cast<const uint2*>(Bank + stride_b0 +
                                                  (int64_t)t * 2304 + c0));
  float b0[4], b1[4];
  unpack_bf16x4(u0, b0);
  unpack_bf16x4(u1, b1);

#pragma unroll
  for (int s = 0; s < 3; ++s) {
    float2 cn =
        *reinterpret_cast<const float2*>(CNorm + ((int64_t)s * T + t) * 2);
    float mn[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) mn[i] = __fmaf_rn(cn.x, b0[i], cn.y * b1[i]);
    __nv_bfloat16* row = Out + ((int64_t)s * T + t) * 4608;
    *reinterpret_cast<uint2*>(row + c0) = dn_u2;
    *reinterpret_cast<uint2*>(row + 2304 + c0) = pack_bf16x4(mn);
  }
}

__global__ __launch_bounds__(576, 2) void sm80_pair_rms_norm_rank2_gen_kernel(
    const float* __restrict__ Dest, const __nv_bfloat16* __restrict__ Bank,
    const float* __restrict__ CNorm, __nv_bfloat16* __restrict__ Out,
    int64_t stride_dest_s, int64_t stride_b0, int32_t T, float eps) {
  const int t = blockIdx.x;
  const int s = blockIdx.y;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int c0 = tid * 4;

  float4 d4 = __ldg(reinterpret_cast<const float4*>(
      Dest + (int64_t)s * stride_dest_s + (int64_t)t * 2304 + c0));
  float sq = d4.x * d4.x + d4.y * d4.y + d4.z * d4.z + d4.w * d4.w;
  sq = warp_sum(sq);
  __shared__ float s_norm[18];
  if (lane == 0) s_norm[wid] = sq;
  __syncthreads();
  if (wid == 0) {
    float tot = (lane < 18) ? s_norm[lane] : 0.0f;
    tot = warp_sum(tot);
    if (lane == 0) s_norm[0] = rsqrt_approx_f32(__fdiv_rn(tot, 2304.0f) + eps);
  }
  __syncthreads();
  const float rd = s_norm[0];
  float dn[4] = {d4.x * rd, d4.y * rd, d4.z * rd, d4.w * rd};

  uint2 u0 =
      __ldg(reinterpret_cast<const uint2*>(Bank + (int64_t)t * 2304 + c0));
  uint2 u1 = __ldg(reinterpret_cast<const uint2*>(Bank + stride_b0 +
                                                  (int64_t)t * 2304 + c0));
  float b0[4], b1[4];
  unpack_bf16x4(u0, b0);
  unpack_bf16x4(u1, b1);

  float2 cn =
      *reinterpret_cast<const float2*>(CNorm + ((int64_t)s * T + t) * 2);
  float mn[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) mn[i] = __fmaf_rn(cn.x, b0[i], cn.y * b1[i]);
  __nv_bfloat16* row = Out + ((int64_t)s * T + t) * 4608;
  *reinterpret_cast<uint2*>(row + c0) = pack_bf16x4(dn);
  *reinterpret_cast<uint2*>(row + 2304 + c0) = pack_bf16x4(mn);
}

// =========================================================================
// 2C. 1u DECODE FUSED pair_rms_norm + ctrl_down (2-pass zero-spill, M <= 8)
//     Eliminates pair buffer, _pair_rms_norm_kernel, _ctrl_down_kernel, and
//     SPLITS partials
// =========================================================================
__global__ __launch_bounds__(288, 2) void sm80_fused_pair_norm_ctrl_down_kernel(
    const float* __restrict__ Dest, const float* __restrict__ Msg,
    const __nv_bfloat16* __restrict__ Down, __nv_bfloat16* __restrict__ H,
    int64_t stride_dest_s, int64_t stride_w_s, int64_t stride_w_r, int32_t M,
    int32_t R, float eps) {
  const int r = blockIdx.x;
  const int s = blockIdx.y;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int c0 = tid * 8;

  __shared__ float s_red[9][16];
  __shared__ float s_rstd[16];

  // Pass 1: compute rstd for dest[s, 0..M-1] and msg[s, 0..M-1] (fits in L1/L2)
  for (int m = 0; m < M; ++m) {
    const float4* dp = reinterpret_cast<const float4*>(
        Dest + (int64_t)s * stride_dest_s + (int64_t)m * 2304 + c0);
    const float4* mp =
        reinterpret_cast<const float4*>(Msg + ((int64_t)s * M + m) * 2304 + c0);
    float4 d0 = __ldg(dp), d1 = __ldg(dp + 1);
    float4 m0 = __ldg(mp), m1 = __ldg(mp + 1);
    float sd = d0.x * d0.x + d0.y * d0.y + d0.z * d0.z + d0.w * d0.w +
               d1.x * d1.x + d1.y * d1.y + d1.z * d1.z + d1.w * d1.w;
    float sm = m0.x * m0.x + m0.y * m0.y + m0.z * m0.z + m0.w * m0.w +
               m1.x * m1.x + m1.y * m1.y + m1.z * m1.z + m1.w * m1.w;
    sd = warp_sum(sd);
    sm = warp_sum(sm);
    if (lane == 0) {
      s_red[wid][m] = sd;
      s_red[wid][8 + m] = sm;
    }
  }
  __syncthreads();
  if (wid == 0 && lane < 16) {
    int m_idx = lane & 7;
    if (m_idx < M) {
      float tot = 0.0f;
#pragma unroll
      for (int w = 0; w < 9; ++w) tot += s_red[w][lane];
      s_rstd[lane] = rsqrt_approx_f32(__fdiv_rn(tot, 2304.0f) + eps);
    }
  }
  __syncthreads();

  // Pass 2: load weight row r once, compute bf16-rounded normalized dot
  // products for m=0..M-1
  const __nv_bfloat16* w_base =
      Down + (int64_t)s * stride_w_s + (int64_t)r * stride_w_r;
  uint4 wu0 = __ldg(reinterpret_cast<const uint4*>(w_base + c0));
  uint4 wu1 = __ldg(reinterpret_cast<const uint4*>(w_base + 2304 + c0));
  float w0[8], w1[8];
  unpack_bf16x8(wu0, w0);
  unpack_bf16x8(wu1, w1);

  for (int m = 0; m < M; ++m) {
    const float rd = s_rstd[m];
    const float rm = s_rstd[8 + m];
    const float4* dp = reinterpret_cast<const float4*>(
        Dest + (int64_t)s * stride_dest_s + (int64_t)m * 2304 + c0);
    const float4* mp =
        reinterpret_cast<const float4*>(Msg + ((int64_t)s * M + m) * 2304 + c0);
    float4 d0 = __ldg(dp), d1 = __ldg(dp + 1);
    float4 m0 = __ldg(mp), m1 = __ldg(mp + 1);
    auto rbf = [](float v) -> float {
      return __bfloat162float(__float2bfloat16_rn(v));
    };
    float acc = 0.0f;
    acc =
        __fmaf_rn(rbf(d0.x * rd), w0[0], __fmaf_rn(rbf(m0.x * rm), w1[0], acc));
    acc =
        __fmaf_rn(rbf(d0.y * rd), w0[1], __fmaf_rn(rbf(m0.y * rm), w1[1], acc));
    acc =
        __fmaf_rn(rbf(d0.z * rd), w0[2], __fmaf_rn(rbf(m0.z * rm), w1[2], acc));
    acc =
        __fmaf_rn(rbf(d0.w * rd), w0[3], __fmaf_rn(rbf(m0.w * rm), w1[3], acc));
    acc =
        __fmaf_rn(rbf(d1.x * rd), w0[4], __fmaf_rn(rbf(m1.x * rm), w1[4], acc));
    acc =
        __fmaf_rn(rbf(d1.y * rd), w0[5], __fmaf_rn(rbf(m1.y * rm), w1[5], acc));
    acc =
        __fmaf_rn(rbf(d1.z * rd), w0[6], __fmaf_rn(rbf(m1.z * rm), w1[6], acc));
    acc =
        __fmaf_rn(rbf(d1.w * rd), w0[7], __fmaf_rn(rbf(m1.w * rm), w1[7], acc));
    acc = warp_sum(acc);
    if (lane == 0) s_red[wid][m] = acc;
  }
  __syncthreads();
  if (wid == 0 && lane < M) {
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < 9; ++w) tot += s_red[w][lane];
    H[((int64_t)s * M + lane) * R + r] = __float2bfloat16_rn(tot);
  }
}

// =========================================================================
// 3. FAST fp32 -> bf16 RMSNorm (for hybrid fused blend, 288 threads, 28 regs)
// =========================================================================
__global__ __launch_bounds__(288) void sm80_rms_norm_f32_to_bf16_kernel(
    const float* __restrict__ X, const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ out, int64_t stride_w_s, int T, float eps) {
  int s = blockIdx.x;
  int t = blockIdx.y;
  int tid = threadIdx.x;
  int lane = tid & 31;
  int wid = tid >> 5;
  int col = tid * 8;
  const float4* x4 =
      reinterpret_cast<const float4*>(X + ((int64_t)s * T + t) * 2304 + col);
  float4 v0 = x4[0], v1 = x4[1];
  float sq = v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w +
             v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w;
  sq = warp_sum(sq);
  __shared__ float sh[9];
  if (lane == 0) sh[wid] = sq;
  __syncthreads();
  if (tid == 0) {
    float tot = 0.0f;
#pragma unroll
    for (int i = 0; i < 9; ++i) tot += sh[i];
    sh[0] = rsqrt_approx_f32(__fdiv_rn(tot, 2304.0f) + eps);
  }
  __syncthreads();
  float rstd = sh[0];
  const __nv_bfloat162* w2 =
      reinterpret_cast<const __nv_bfloat162*>(norm_w + s * stride_w_s + col);
  __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(
      out + ((int64_t)s * T + t) * 2304 + col);
  float2 wf0 = __bfloat1622float2(w2[0]), wf1 = __bfloat1622float2(w2[1]);
  float2 wf2 = __bfloat1622float2(w2[2]), wf3 = __bfloat1622float2(w2[3]);
  o2[0] = __float22bfloat162_rn(
      make_float2((v0.x * rstd) * wf0.x, (v0.y * rstd) * wf0.y));
  o2[1] = __float22bfloat162_rn(
      make_float2((v0.z * rstd) * wf1.x, (v0.w * rstd) * wf1.y));
  o2[2] = __float22bfloat162_rn(
      make_float2((v1.x * rstd) * wf2.x, (v1.y * rstd) * wf2.y));
  o2[3] = __float22bfloat162_rn(
      make_float2((v1.z * rstd) * wf3.x, (v1.w * rstd) * wf3.y));
}

// =========================================================================
// 4A. MoE ROUTER + TOP-8 SELECT + BLOCK_M=8 ALIGN (1-kernel T<=8 & hybrid
// T<=64)
// =========================================================================
__device__ __forceinline__ bool is_better_sel(float s_a, int e_a, float s_b,
                                              int e_b) {
  return (s_a > s_b) || (s_a == s_b && e_a < e_b);
}

template <int BM>
__device__ __forceinline__ void parallel_align_128(
    int tid, const int32_t* s_counts, int32_t* s_starts, int32_t* s_total_mb,
    int32_t* expert_ids, int32_t* num_padded) {
  __shared__ int32_t s_warp_tot[4];
  int c = (tid < 128) ? s_counts[tid] : 0;
  int nmb = (c + (BM - 1)) / BM;
  int lane = tid & 31;
  int wid = tid >> 5;
  if (tid < 128) {
    int v = nmb;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int x = __shfl_up_sync(0xffffffff, v, off);
      if (lane >= off) v += x;
    }
    if (lane == 31) s_warp_tot[wid] = v;
    s_starts[tid] = v - nmb;
  }
  __syncthreads();
  if (tid < 128) {
    int base = 0;
    if (wid >= 1) base += s_warp_tot[0];
    if (wid >= 2) base += s_warp_tot[1];
    if (wid >= 3) base += s_warp_tot[2];
    int mb0 = base + s_starts[tid];
    s_starts[tid] = mb0 * BM;
    if (nmb > 0) {
      for (int b = 0; b < nmb; ++b) expert_ids[mb0 + b] = tid;
    }
    if (tid == 127) {
      int tot = mb0 + nmb;
      *s_total_mb = tot;
      num_padded[0] = tot * BM;
    }
  }
}

// Multi-CTA parallel select + single-BM (8 for 2u/4u, 16 for 8u) align (8 < T
// <= 64): 1.42x faster at T=20 (4u), 2.26x faster at T=40 (8u), ~7.0us CG
// (vs 12.8us dual-BM)
template <int G, int BM>
__global__ __launch_bounds__(128) void sm80_select_align_mcta_kernel(
    const float* __restrict__ partials, const float* __restrict__ bias,
    const bool* __restrict__ is_pad, int32_t* __restrict__ g_counts,
    int32_t* __restrict__ g_row_rank, int32_t* __restrict__ g_done,
    int32_t* __restrict__ out_ids, float* __restrict__ out_weights,
    int32_t* __restrict__ out_src2dst, int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids, int32_t* __restrict__ num_padded, int T,
    float route_scale) {
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int t = blockIdx.x * 4 + wid;

  __shared__ float s_bias[128];
  s_bias[tid] = __ldg(bias + tid);
  __syncthreads();

  if (t < T) {
    const bool pad = (is_pad != nullptr) && is_pad[t];
    const int e0 = lane, e1 = lane + 32, e2 = lane + 64, e3 = lane + 96;
    const int64_t sg = (int64_t)T * 128;
    const int64_t base = (int64_t)t * 128;
    float l0, l1, l2, l3;
    if constexpr (G == 3) {
      l0 = partials[base + e0] + partials[sg + base + e0] +
           partials[2 * sg + base + e0];
      l1 = partials[base + e1] + partials[sg + base + e1] +
           partials[2 * sg + base + e1];
      l2 = partials[base + e2] + partials[sg + base + e2] +
           partials[2 * sg + base + e2];
      l3 = partials[base + e3] + partials[sg + base + e3] +
           partials[2 * sg + base + e3];
    } else {
      l0 = partials[base + e0];
      l1 = partials[base + e1];
      l2 = partials[base + e2];
      l3 = partials[base + e3];
    }
    float s0 = __fdiv_rn(1.0f, 1.0f + expf(-l0)) + s_bias[e0];
    float s1 = __fdiv_rn(1.0f, 1.0f + expf(-l1)) + s_bias[e1];
    float s2 = __fdiv_rn(1.0f, 1.0f + expf(-l2)) + s_bias[e2];
    float s3 = __fdiv_rn(1.0f, 1.0f + expf(-l3)) + s_bias[e3];

    int my_id = 0;
    float my_l = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float bs = s0;
      int be = e0;
      float bl = l0;
      if (is_better_sel(s1, e1, bs, be)) {
        bs = s1;
        be = e1;
        bl = l1;
      }
      if (is_better_sel(s2, e2, bs, be)) {
        bs = s2;
        be = e2;
        bl = l2;
      }
      if (is_better_sel(s3, e3, bs, be)) {
        bs = s3;
        be = e3;
        bl = l3;
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        float os = __shfl_xor_sync(0xffffffff, bs, off);
        int oe = __shfl_xor_sync(0xffffffff, be, off);
        float ol = __shfl_xor_sync(0xffffffff, bl, off);
        if (is_better_sel(os, oe, bs, be)) {
          bs = os;
          be = oe;
          bl = ol;
        }
      }
      if (e0 == be)
        s0 = -1e38f;
      else if (e1 == be)
        s1 = -1e38f;
      else if (e2 == be)
        s2 = -1e38f;
      else if (e3 == be)
        s3 = -1e38f;
      if (lane == k) {
        my_id = be;
        my_l = bl;
      }
    }
    if (lane < 8) {
      float lp = fminf(my_l, 0.0f) - log1pf(expf(-fabsf(my_l)));
      float m = lp;
      m = fmaxf(m, __shfl_xor_sync(0xff, m, 4));
      m = fmaxf(m, __shfl_xor_sync(0xff, m, 2));
      m = fmaxf(m, __shfl_xor_sync(0xff, m, 1));
      float sh = expf(lp - m);
      float r2 = sh + __shfl_xor_sync(0xff, sh, 2);
      float r4 = r2 + __shfl_xor_sync(0xff, r2, 1);
      float tot = r4 + __shfl_xor_sync(0xff, r4, 4);
      float w = expf(lp - (m + logf(tot))) * route_scale;
      int row = t * 8 + lane;
      int eid = pad ? -1 : my_id;
      out_ids[row] = eid;
      out_weights[row] = w;
      out_src2dst[row] = pad ? -1 : row;
      if (!pad) g_row_rank[row] = atomicAdd(&g_counts[eid], 1);
    }
  }
  __syncthreads();
  __threadfence();

  __shared__ bool is_last;
  if (tid == 0) {
    int done = atomicAdd(g_done, 1) + 1;
    is_last = (done == (int)gridDim.x);
    if (is_last) g_done[0] = 0;
  }
  __syncthreads();
  if (!is_last) return;

  __shared__ int32_t s_counts[128];
  __shared__ int32_t s_starts[128];
  __shared__ int32_t s_total_mb;
  s_counts[tid] = __ldcg(g_counts + tid);
  g_counts[tid] = 0;
  __syncthreads();

  parallel_align_128<BM>(tid, s_counts, s_starts, &s_total_mb, expert_ids,
                         num_padded);
  __syncthreads();

  const int total_padded = s_total_mb * BM;
  const int total_rows = T * 8;
  for (int i = tid; i < total_padded; i += 128) sorted_ids[i] = total_rows;
  __syncthreads();
  for (int r = tid; r < total_rows; r += 128) {
    int e = __ldcg(out_ids + r);
    if (e >= 0) sorted_ids[s_starts[e] + __ldcg(g_row_rank + r)] = r;
  }
}

template <int MAX_T>
__global__ __launch_bounds__(288, 2) void sm80_route_align_1k_v6_kernel(
    const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
    const float* __restrict__ bias, const bool* __restrict__ is_pad,
    float* __restrict__ g_logits, int32_t* __restrict__ g_counter,
    int32_t* __restrict__ out_ids, float* __restrict__ out_weights,
    int32_t* __restrict__ out_src2dst, int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ expert_ids, int32_t* __restrict__ num_padded, int T,
    float route_scale) {
  // grid=128 (1 expert/CTA), 288 threads (9 warps x 8 cols = 2304 exact, zero
  // tail branch)
  const int expert = blockIdx.x;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;

  uint4 wv =
      __ldg(reinterpret_cast<const uint4*>(W + (int64_t)expert * 2304) + tid);
  float wf[8];
  unpack_bf16x8(wv, wf);

  __shared__ float s_part[9][MAX_T];
#pragma unroll
  for (int t = 0; t < MAX_T; ++t) {
    if (t < T) {
      uint4 xv =
          __ldg(reinterpret_cast<const uint4*>(X + (int64_t)t * 2304) + tid);
      float xf[8];
      unpack_bf16x8(xv, xf);
      float acc = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) acc = __fmaf_rn(xf[i], wf[i], acc);
      acc = warp_sum(acc);
      if (lane == 0) s_part[wid][t] = acc;
    }
  }
  __syncthreads();
  if (wid == 0 && lane < T) {
    float tot = 0.0f;
#pragma unroll
    for (int w = 0; w < 9; ++w) tot += s_part[w][lane];
    g_logits[lane * 128 + expert] = tot;
  }
  __threadfence();
  __syncthreads();

  __shared__ bool is_last;
  if (tid == 0) {
    int done = atomicAdd(g_counter, 1) + 1;
    is_last = (done == 128);
    if (is_last) g_counter[0] = 0;
  }
  __syncthreads();
  if (!is_last) return;

  __shared__ float s_bias[128];
  __shared__ int32_t s_counts[128];
  __shared__ int32_t s_starts[128];
  __shared__ int32_t s_row_exp[MAX_T * 8];
  __shared__ int32_t s_row_rank[MAX_T * 8];
  __shared__ int32_t s_total_mb;
  if (tid < 128) {
    s_bias[tid] = bias[tid];
    s_counts[tid] = 0;
  }
  __syncthreads();

  if (wid < T) {
    int t = wid;
    bool pad = (is_pad != nullptr) && is_pad[t];
    int e0 = lane, e1 = lane + 32, e2 = lane + 64, e3 = lane + 96;
    int base = t * 128;
    float l0 = __ldcg(g_logits + base + e0);
    float l1 = __ldcg(g_logits + base + e1);
    float l2 = __ldcg(g_logits + base + e2);
    float l3 = __ldcg(g_logits + base + e3);
    float s0 = __fdiv_rn(1.0f, 1.0f + expf(-l0)) + s_bias[e0];
    float s1 = __fdiv_rn(1.0f, 1.0f + expf(-l1)) + s_bias[e1];
    float s2 = __fdiv_rn(1.0f, 1.0f + expf(-l2)) + s_bias[e2];
    float s3 = __fdiv_rn(1.0f, 1.0f + expf(-l3)) + s_bias[e3];
    int my_id = 0;
    float my_l = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float bs = s0;
      int be = e0;
      float bl = l0;
      if (is_better_sel(s1, e1, bs, be)) {
        bs = s1;
        be = e1;
        bl = l1;
      }
      if (is_better_sel(s2, e2, bs, be)) {
        bs = s2;
        be = e2;
        bl = l2;
      }
      if (is_better_sel(s3, e3, bs, be)) {
        bs = s3;
        be = e3;
        bl = l3;
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        float os = __shfl_xor_sync(0xffffffff, bs, off);
        int oe = __shfl_xor_sync(0xffffffff, be, off);
        float ol = __shfl_xor_sync(0xffffffff, bl, off);
        if (is_better_sel(os, oe, bs, be)) {
          bs = os;
          be = oe;
          bl = ol;
        }
      }
      if (e0 == be)
        s0 = -1e38f;
      else if (e1 == be)
        s1 = -1e38f;
      else if (e2 == be)
        s2 = -1e38f;
      else if (e3 == be)
        s3 = -1e38f;
      if (lane == k) {
        my_id = be;
        my_l = bl;
      }
    }
    if (lane < 8) {
      float lp = fminf(my_l, 0.0f) - log1pf(expf(-fabsf(my_l)));
      float m = lp;
      m = fmaxf(m, __shfl_xor_sync(0xff, m, 4));
      m = fmaxf(m, __shfl_xor_sync(0xff, m, 2));
      m = fmaxf(m, __shfl_xor_sync(0xff, m, 1));
      float sh = expf(lp - m);
      float r2 = sh + __shfl_xor_sync(0xff, sh, 2);
      float r4 = r2 + __shfl_xor_sync(0xff, r2, 1);
      float tot = r4 + __shfl_xor_sync(0xff, r4, 4);
      float w = expf(lp - (m + logf(tot))) * route_scale;
      int row = t * 8 + lane;
      int eid = pad ? -1 : my_id;
      out_ids[row] = eid;
      out_weights[row] = w;
      out_src2dst[row] = pad ? -1 : row;
      s_row_exp[row] = eid;
      if (!pad) s_row_rank[row] = atomicAdd(&s_counts[eid], 1);
    }
  }
  __syncthreads();
  parallel_align_128<8>(tid, s_counts, s_starts, &s_total_mb, expert_ids,
                        num_padded);
  __syncthreads();
  int total_padded = s_total_mb * 8;
  int total_rows = T * 8;
  for (int i = tid; i < total_padded; i += 288) sorted_ids[i] = total_rows;
  __syncthreads();
  for (int r = tid; r < total_rows; r += 288) {
    int e = s_row_exp[r];
    if (e >= 0) sorted_ids[s_starts[e] + s_row_rank[r]] = r;
  }
}

// =========================================================================
// 4B. FAST 3-KERNEL COUNTING-SORT moe_align_large (PREFILL, E <= 256)
//     6.8-12.6x faster than moe_align_block_size_torch
// =========================================================================
__global__ __launch_bounds__(256) void sm80_moe_align_hist_kernel(
    const int32_t* __restrict__ ids, int32_t* __restrict__ g_counts, int numel,
    int E) {
  __shared__ int32_t s_cnt[256];
  int tid = threadIdx.x;
  if (tid < E) s_cnt[tid] = 0;
  __syncthreads();
  int stride = gridDim.x * 256;
  for (int i = blockIdx.x * 256 + tid; i < numel; i += stride) {
    int e = ids[i];
    if ((uint32_t)e < (uint32_t)E) atomicAdd(&s_cnt[e], 1);
  }
  __syncthreads();
  if (tid < E && s_cnt[tid] > 0) atomicAdd(&g_counts[tid], s_cnt[tid]);
}

__global__ __launch_bounds__(256) void sm80_moe_align_prefix_kernel(
    const int32_t* __restrict__ g_counts, int32_t* __restrict__ g_offsets,
    int32_t* __restrict__ expert_ids, int32_t* __restrict__ num_padded, int E,
    int block_size) {
  int tid = threadIdx.x;
  int lane = tid & 31;
  int wid = tid >> 5;
  __shared__ int32_t s_wsum[8];
  __shared__ int32_t s_start[256];
  __shared__ int32_t s_nmb[256];
  int mb = 0;
  if (tid < E) {
    int c = g_counts[tid];
    mb = (c + block_size - 1) / block_size;
    s_nmb[tid] = mb;
  }
  int val = mb;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    int o = __shfl_up_sync(0xffffffff, val, d);
    if (lane >= d) val += o;
  }
  if (lane == 31) s_wsum[wid] = val;
  s_start[tid] = val - mb;
  __syncthreads();
  if (tid < 8) {
    int wv = s_wsum[tid];
#pragma unroll
    for (int d = 1; d < 8; d <<= 1) {
      int o = __shfl_up_sync(0xff, wv, d);
      if (tid >= d) wv += o;
    }
    s_wsum[tid] = wv;
  }
  __syncthreads();
  if (tid < E) {
    int base_mb = s_start[tid] + (wid > 0 ? s_wsum[wid - 1] : 0);
    g_offsets[tid] = base_mb * block_size;
    int nmb = s_nmb[tid];
    for (int b = 0; b < nmb; ++b) expert_ids[base_mb + b] = tid;
    if (tid == E - 1) num_padded[0] = (base_mb + nmb) * block_size;
  }
}

__global__ __launch_bounds__(256) void sm80_moe_align_scatter_kernel(
    const int32_t* __restrict__ ids, int32_t* __restrict__ g_offsets,
    int32_t* __restrict__ sorted_ids, int numel, int E) {
  int stride = gridDim.x * 256;
  for (int i = blockIdx.x * 256 + threadIdx.x; i < numel; i += stride) {
    int e = ids[i];
    if ((uint32_t)e < (uint32_t)E) {
      int pos = atomicAdd(&g_offsets[e], 1);
      sorted_ids[pos] = i;
    }
  }
}

// =========================================================================
// 5. DSPARK DRAFT FUSIONS (markov_rnn_step, draft_ctx_kv, draft_qk_kv,
// draft_embed_block)
// =========================================================================
template <typename IdxT>
__global__ __launch_bounds__(128) void sm80_markov_rnn_step_kernel(
    const __nv_bfloat16* __restrict__ State,
    const IdxT* __restrict__ PrevTokens,
    const __nv_bfloat16* __restrict__ W1Embed,
    const __nv_bfloat16* __restrict__ RnnW, const float* __restrict__ Context,
    __nv_bfloat16* __restrict__ OutState, __nv_bfloat16* __restrict__ OutLatent,
    int64_t sw_rnn) {
  constexpr int R = 256;
  const int blk = blockIdx.x;
  const int req = blockIdx.y;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;

  __shared__ float s_in[512];
  int64_t tok = static_cast<int64_t>(PrevTokens[req]);
  for (int i = tid; i < 64; i += 128) {
    int c0 = i * 4;
    const __nv_bfloat162* sp =
        reinterpret_cast<const __nv_bfloat162*>(State + (int64_t)req * R + c0);
    const __nv_bfloat162* ep =
        reinterpret_cast<const __nv_bfloat162*>(W1Embed + tok * R + c0);
    float2 sf0 = __bfloat1622float2(sp[0]), sf1 = __bfloat1622float2(sp[1]);
    float2 ef0 = __bfloat1622float2(ep[0]), ef1 = __bfloat1622float2(ep[1]);
    s_in[c0] = sf0.x;
    s_in[c0 + 1] = sf0.y;
    s_in[c0 + 2] = sf1.x;
    s_in[c0 + 3] = sf1.y;
    s_in[R + c0] = ef0.x;
    s_in[R + c0 + 1] = ef0.y;
    s_in[R + c0 + 2] = ef1.x;
    s_in[R + c0 + 3] = ef1.y;
  }
  __syncthreads();

  const int c_base = blk * 16 + wid * 4;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int c = c_base + j;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
    const uint4* w0 =
        reinterpret_cast<const uint4*>(RnnW + (int64_t)c * sw_rnn);
    const uint4* w1 =
        reinterpret_cast<const uint4*>(RnnW + (int64_t)(c + R) * sw_rnn);
    const uint4* w2 =
        reinterpret_cast<const uint4*>(RnnW + (int64_t)(c + 2 * R) * sw_rnn);
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      int idx = (it * 32 + lane) * 8;
      uint4 u0 = __ldg(w0 + it * 32 + lane);
      uint4 u1 = __ldg(w1 + it * 32 + lane);
      uint4 u2 = __ldg(w2 + it * 32 + lane);
      const __nv_bfloat162* b0 = reinterpret_cast<const __nv_bfloat162*>(&u0);
      const __nv_bfloat162* b1 = reinterpret_cast<const __nv_bfloat162*>(&u1);
      const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&u2);
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float x0 = s_in[idx + k * 2], x1 = s_in[idx + k * 2 + 1];
        float2 f0 = __bfloat1622float2(b0[k]);
        float2 f1 = __bfloat1622float2(b1[k]);
        float2 f2 = __bfloat1622float2(b2[k]);
        a0 = __fmaf_rn(f0.x, x0, __fmaf_rn(f0.y, x1, a0));
        a1 = __fmaf_rn(f1.x, x0, __fmaf_rn(f1.y, x1, a1));
        a2 = __fmaf_rn(f2.x, x0, __fmaf_rn(f2.y, x1, a2));
      }
    }
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    a2 = warp_sum(a2);
    if (lane == 0) {
      auto rbf16 = [](float x) -> float {
        return __bfloat162float(__float2bfloat16_rn(x));
      };
      float raw_g = Context[(int64_t)req * (3 * R) + c] + a0;
      float raw_c = Context[(int64_t)req * (3 * R) + R + c] + a1;
      float raw_o = Context[(int64_t)req * (3 * R) + 2 * R + c] + a2;
      float gate = rbf16(__fdiv_rn(1.0f, 1.0f + expf(-rbf16(raw_g))));
      float cand = rbf16(tanhf(rbf16(raw_c)));
      float st = s_in[c];
      float keep = rbf16(gate * st);
      float upd = rbf16(rbf16(1.0f - gate) * cand);
      OutState[(int64_t)req * R + c] = __float2bfloat16_rn(keep + upd);
      OutLatent[(int64_t)req * R + c] =
          __float2bfloat16_rn(tanhf(rbf16(raw_o)));
    }
  }
}

template <typename PosT>
__global__ __launch_bounds__(64) void sm80_draft_ctx_kv_kernel(
    const __nv_bfloat16* __restrict__ kv_proj,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const float* __restrict__ cos_sin, const PosT* __restrict__ positions,
    const int64_t* __restrict__ slot_mapping,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ val_cache, int64_t kv_stride_t, int kv_size,
    int64_t kc_blk, int64_t kc_pos, int64_t kc_h, int64_t vc_blk,
    int64_t vc_pos, int64_t vc_h, int block_size, float eps) {
  int t = blockIdx.x;
  int h = blockIdx.y;
  int d = threadIdx.x;
  int64_t slot = slot_mapping[t];
  if (slot < 0) return;
  const __nv_bfloat16* k_row = kv_proj + (int64_t)t * kv_stride_t + h * 128;
  const __nv_bfloat16* v_row = k_row + kv_size;
  float x1 = __bfloat162float(k_row[d]);
  float x2 = __bfloat162float(k_row[d + 64]);
  float sq = warp_sum(x1 * x1 + x2 * x2);
  __shared__ float s_w[2];
  if ((d & 31) == 0) s_w[d >> 5] = sq;
  __syncthreads();
  float inv = rsqrtf((s_w[0] + s_w[1]) * (1.0f / 128.0f) + eps);
  float k1 = __bfloat162float(
      __float2bfloat16_rn(x1 * inv * __bfloat162float(k_norm_w[d])));
  float k2 = __bfloat162float(
      __float2bfloat16_rn(x2 * inv * __bfloat162float(k_norm_w[d + 64])));
  int64_t pos = static_cast<int64_t>(positions[t]);
  float c = cos_sin[pos * 128 + d];
  float s = cos_sin[pos * 128 + 64 + d];
  int64_t blk = slot / block_size;
  int64_t off = slot % block_size;
  int64_t k_dst = blk * kc_blk + off * kc_pos + (int64_t)h * kc_h;
  int64_t v_dst = blk * vc_blk + off * vc_pos + (int64_t)h * vc_h;
  key_cache[k_dst + d] = __float2bfloat16_rn(k1 * c - k2 * s);
  key_cache[k_dst + d + 64] = __float2bfloat16_rn(k2 * c + k1 * s);
  val_cache[v_dst + d] = v_row[d];
  val_cache[v_dst + d + 64] = v_row[d + 64];
}

template <typename PosT>
__global__ __launch_bounds__(64) void sm80_draft_qk_norm_rope_cache_kernel(
    __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const float* __restrict__ cos_sin, const PosT* __restrict__ positions,
    const int64_t* __restrict__ slot_mapping,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ val_cache, int64_t stride_t, int q_size,
    int kv_size, int nhq, int slot_len, int64_t kc_blk, int64_t kc_pos,
    int64_t kc_h, int64_t vc_blk, int64_t vc_pos, int64_t vc_h, int block_size,
    float eps) {
  int t = blockIdx.x;
  int h = blockIdx.y;
  int d = threadIdx.x;
  bool is_q = h < nhq;
  __nv_bfloat16* row = qkv + (int64_t)t * stride_t +
                       (is_q ? (h * 128) : (q_size + (h - nhq) * 128));
  const __nv_bfloat16* nw = is_q ? q_norm_w : k_norm_w;

  float x1 = __bfloat162float(row[d]);
  float x2 = __bfloat162float(row[d + 64]);
  float sq = warp_sum(x1 * x1 + x2 * x2);
  __shared__ float s_w[2];
  if ((d & 31) == 0) s_w[d >> 5] = sq;
  __syncthreads();
  float inv = rsqrtf((s_w[0] + s_w[1]) * (1.0f / 128.0f) + eps);
  float n1 =
      __bfloat162float(__float2bfloat16_rn(x1 * inv * __bfloat162float(nw[d])));
  float n2 = __bfloat162float(
      __float2bfloat16_rn(x2 * inv * __bfloat162float(nw[d + 64])));
  int64_t pos = static_cast<int64_t>(positions[t]);
  float c = cos_sin[pos * 128 + d];
  float s = cos_sin[pos * 128 + 64 + d];
  __nv_bfloat16 o1 = __float2bfloat16_rn(n1 * c - n2 * s);
  __nv_bfloat16 o2 = __float2bfloat16_rn(n2 * c + n1 * s);
  if (is_q) {
    row[d] = o1;
    row[d + 64] = o2;
  } else if (t < slot_len) {
    int64_t slot = slot_mapping[t];
    if (slot >= 0) {
      int kh = h - nhq;
      const __nv_bfloat16* v_row = row + kv_size;
      int64_t blk = slot / block_size;
      int64_t off = slot % block_size;
      int64_t k_dst = blk * kc_blk + off * kc_pos + (int64_t)kh * kc_h;
      int64_t v_dst = blk * vc_blk + off * vc_pos + (int64_t)kh * vc_h;
      key_cache[k_dst + d] = o1;
      key_cache[k_dst + d + 64] = o2;
      val_cache[v_dst + d] = v_row[d];
      val_cache[v_dst + d + 64] = v_row[d + 64];
    }
  }
}

template <typename IdxT>
__global__ __launch_bounds__(288) void sm80_draft_embed_block_kernel(
    const IdxT* __restrict__ input_ids,
    const __nv_bfloat16* __restrict__ embed_w,
    const __nv_bfloat16* __restrict__ norm_w,
    const __nv_bfloat16* __restrict__ mask_e, __nv_bfloat16* __restrict__ out,
    int block_size, float eps) {
  int req = blockIdx.x;
  int pos_in_blk = blockIdx.y;
  int tid = threadIdx.x;
  int col = tid * 8;
  int out_row = req * block_size + pos_in_blk;
  if (pos_in_blk > 0) {
    *reinterpret_cast<int4*>(out + (int64_t)out_row * 2304 + col) =
        *reinterpret_cast<const int4*>(mask_e + col);
    return;
  }
  int64_t tok_id = static_cast<int64_t>(input_ids[req * block_size]);
  int4 ev = *reinterpret_cast<const int4*>(embed_w + tok_id * 2304 + col);
  const __nv_bfloat162* e2 = reinterpret_cast<const __nv_bfloat162*>(&ev);
  float2 f0 = __bfloat1622float2(e2[0]), f1 = __bfloat1622float2(e2[1]);
  float2 f2 = __bfloat1622float2(e2[2]), f3 = __bfloat1622float2(e2[3]);
  float sq = warp_sum(f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y +
                      f2.x * f2.x + f2.y * f2.y + f3.x * f3.x + f3.y * f3.y);
  __shared__ float sh[9];
  if ((tid & 31) == 0) sh[tid >> 5] = sq;
  __syncthreads();
  if (tid == 0) {
    float tot = 0.0f;
#pragma unroll
    for (int i = 0; i < 9; ++i) tot += sh[i];
    sh[0] = rsqrt_approx_f32(__fdiv_rn(tot, 2304.0f) + eps);
  }
  __syncthreads();
  float inv = sh[0];
  int4 wv = *reinterpret_cast<const int4*>(norm_w + col);
  const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
  float2 w0 = __bfloat1622float2(w2[0]), w1 = __bfloat1622float2(w2[1]);
  float2 w2f = __bfloat1622float2(w2[2]), w3 = __bfloat1622float2(w2[3]);
  __nv_bfloat162* o2 =
      reinterpret_cast<__nv_bfloat162*>(out + (int64_t)out_row * 2304 + col);
  o2[0] =
      __float22bfloat162_rn(make_float2(f0.x * inv * w0.x, f0.y * inv * w0.y));
  o2[1] =
      __float22bfloat162_rn(make_float2(f1.x * inv * w1.x, f1.y * inv * w1.y));
  o2[2] = __float22bfloat162_rn(
      make_float2(f2.x * inv * w2f.x, f2.y * inv * w2f.y));
  o2[3] =
      __float22bfloat162_rn(make_float2(f3.x * inv * w3.x, f3.y * inv * w3.y));
}

// =========================================================================
// PYTHON ENTRYPOINTS
// =========================================================================
torch::Tensor py_conv_mix(
    torch::Tensor x, torch::Tensor conv_w, torch::Tensor cache,
    torch::Tensor positions, torch::Tensor block_table, torch::Tensor seq_idx,
    torch::Tensor slot_mapping, torch::Tensor query_start, int stream_off,
    int stream_w, torch::Tensor conv_out, c10::optional<torch::Tensor> residual,
    c10::optional<torch::Tensor> addend, torch::Tensor stack,
    c10::optional<torch::Tensor> last, torch::Tensor queries,
    c10::optional<torch::Tensor> norm_w, double norm_eps, double eps) {
  int T = conv_out.size(0);
  int n_stack = stack.size(0);
  int Q = queries.size(0);
  bool hn = norm_w.has_value();
  auto out = torch::empty(
      {Q, T, 2304},
      conv_out.options().dtype(hn ? torch::kBFloat16 : torch::kFloat32));
  auto st = at::cuda::getCurrentCUDAStream();
  const __nv_bfloat16* rp = residual.has_value()
                                ? (const __nv_bfloat16*)residual->data_ptr()
                                : nullptr;
  const __nv_bfloat16* ap =
      addend.has_value() ? (const __nv_bfloat16*)addend->data_ptr() : nullptr;
  const __nv_bfloat16* lp =
      last.has_value() ? (const __nv_bfloat16*)last->data_ptr() : nullptr;
  const __nv_bfloat16* np =
      hn ? (const __nv_bfloat16*)norm_w->data_ptr() : nullptr;

#define CM_ARGS                                                                \
  (const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)conv_w.data_ptr(), \
      (__nv_bfloat16*)cache.data_ptr(), positions.data_ptr<int64_t>(),         \
      block_table.data_ptr<int32_t>(), seq_idx.data_ptr<int32_t>(),            \
      slot_mapping.data_ptr<int64_t>(), query_start.data_ptr<int32_t>(),       \
      (__nv_bfloat16*)conv_out.data_ptr(), rp, ap,                             \
      (const __nv_bfloat16*)stack.data_ptr(), lp, queries.data_ptr<float>(),   \
      np, out.data_ptr(), x.stride(0), cache.stride(0), cache.stride(1),       \
      cache.stride(2), block_table.stride(0), stack.stride(0), T, n_stack,     \
      block_table.size(1), cache.size(2), stream_off, stream_w,                \
      0.4472135954999579f, (float)eps, (float)norm_eps

  if (T > 256) {
    // Prefill 2-pass zero-spill kernel (called for Q == 3, 1.27-1.34x faster
    // than Triton)
    if (Q == 1) {
      if (hn)
        sm80_prefill_conv_mix_2304_kernel<1, true><<<T, 288, 0, st>>>(CM_ARGS);
      else
        sm80_prefill_conv_mix_2304_kernel<1, false><<<T, 288, 0, st>>>(CM_ARGS);
    } else {
      if (hn)
        sm80_prefill_conv_mix_2304_kernel<3, true><<<T, 288, 0, st>>>(CM_ARGS);
      else
        sm80_prefill_conv_mix_2304_kernel<3, false><<<T, 288, 0, st>>>(CM_ARGS);
    }
  } else if (n_stack <= 3) {
    if (Q == 1) {
      if (hn)
        sm80_conv_mix_2304_kernel<4, 1, true><<<T, 288, 0, st>>>(CM_ARGS);
      else
        sm80_conv_mix_2304_kernel<4, 1, false><<<T, 288, 0, st>>>(CM_ARGS);
    } else {
      if (hn)
        sm80_conv_mix_2304_kernel<4, 3, true><<<T, 288, 0, st>>>(CM_ARGS);
      else
        sm80_conv_mix_2304_kernel<4, 3, false><<<T, 288, 0, st>>>(CM_ARGS);
    }
  } else {
    // 576-thread zero-spill kernel: 1.31-1.34x faster than 288t for n_stack
    // = 4..7
    if (Q == 1) {
      if (hn)
        sm80_conv_mix_576_kernel<8, 1, true><<<T, 576, 0, st>>>(CM_ARGS);
      else
        sm80_conv_mix_576_kernel<8, 1, false><<<T, 576, 0, st>>>(CM_ARGS);
    } else {
      if (hn)
        sm80_conv_mix_576_kernel<8, 3, true><<<T, 576, 0, st>>>(CM_ARGS);
      else
        sm80_conv_mix_576_kernel<8, 3, false><<<T, 576, 0, st>>>(CM_ARGS);
    }
  }
#undef CM_ARGS
  return out;
}

torch::Tensor py_batched_bank_messages(torch::Tensor bank,
                                       torch::Tensor queries, double eps) {
  int L = queries.size(0);
  int S = queries.size(1);
  int T = bank.size(1);
  auto out =
      torch::empty({L, S, T, 2304}, bank.options().dtype(torch::kFloat32));
  sm80_batched_bank_messages_v2_kernel<<<dim3(L * S, T), 288, 0,
                                         at::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)bank.data_ptr(), queries.data_ptr<float>(),
      out.data_ptr<float>(), bank.stride(0), T, (float)eps);
  return out;
}

torch::Tensor py_bank_stats(torch::Tensor bank, double eps) {
  int T = bank.size(1);
  auto stats = torch::empty({T, 5}, bank.options().dtype(torch::kFloat32));
  if (T > 0) {
    sm80_bank_stats_kernel<<<T, 288, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)bank.data_ptr(), stats.data_ptr<float>(),
        bank.stride(0), T, (float)eps);
  }
  return stats;
}

std::tuple<torch::Tensor, torch::Tensor> py_bank_rank2_finalize(
    torch::Tensor dots, torch::Tensor stats, int L, int S, double eps) {
  int T = stats.size(0);
  int Q_TOT = L * S;
  auto c_raw = torch::empty({L, S, T, 2}, stats.options());
  auto c_norm = torch::empty({L, S, T, 2}, stats.options());
  if (T > 0) {
    dim3 grid((T + 255) / 256, Q_TOT);
    sm80_bank_rank2_finalize_kernel<<<grid, 256, 0,
                                      at::cuda::getCurrentCUDAStream()>>>(
        dots.data_ptr<float>(), stats.data_ptr<float>(),
        c_raw.data_ptr<float>(), c_norm.data_ptr<float>(), T, Q_TOT,
        (float)eps);
  }
  return {c_raw, c_norm};
}

torch::Tensor py_pair_rms_norm_rank2(torch::Tensor dest, torch::Tensor bank,
                                     torch::Tensor c_norm, double eps) {
  int S = c_norm.size(0);
  int T = c_norm.size(1);
  auto out = torch::empty({S, T, 4608}, bank.options().dtype(torch::kBFloat16));
  if (T == 0) return out;
  auto st = at::cuda::getCurrentCUDAStream();
  if (S == 3 && dest.stride(0) == 0) {
    sm80_pair_rms_norm_rank2_bcast3_kernel<<<T, 576, 0, st>>>(
        dest.data_ptr<float>(), (const __nv_bfloat16*)bank.data_ptr(),
        c_norm.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
        bank.stride(0), T, (float)eps);
  } else {
    sm80_pair_rms_norm_rank2_gen_kernel<<<dim3(T, S), 576, 0, st>>>(
        dest.data_ptr<float>(), (const __nv_bfloat16*)bank.data_ptr(),
        c_norm.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
        dest.stride(0), bank.stride(0), T, (float)eps);
  }
  return out;
}

torch::Tensor py_fused_pair_norm_ctrl_down(torch::Tensor dest,
                                           torch::Tensor msg,
                                           torch::Tensor down, double eps) {
  int S = msg.size(0);
  int M = msg.size(1);
  int R = down.size(1);
  auto H = torch::empty({S, M, R}, down.options());
  if (M > 0) {
    sm80_fused_pair_norm_ctrl_down_kernel<<<dim3(R, S), 288, 0,
                                            at::cuda::getCurrentCUDAStream()>>>(
        dest.data_ptr<float>(), msg.data_ptr<float>(),
        (const __nv_bfloat16*)down.data_ptr(), (__nv_bfloat16*)H.data_ptr(),
        dest.stride(0), down.stride(0), down.stride(1), M, R, (float)eps);
  }
  return H;
}

torch::Tensor py_rms_norm_f32_to_bf16(torch::Tensor x, torch::Tensor norm_w,
                                      double eps) {
  int S = x.size(0), T = x.size(1);
  auto out =
      torch::empty({S, T, 2304}, norm_w.options().dtype(torch::kBFloat16));
  sm80_rms_norm_f32_to_bf16_kernel<<<dim3(S, T), 288, 0,
                                     at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), (const __nv_bfloat16*)norm_w.data_ptr(),
      (__nv_bfloat16*)out.data_ptr(), norm_w.stride(0), T, (float)eps);
  return out;
}

static inline int calc_max_pad_bm(int numel, int bm) {
  int max_pad = numel + 128 * (bm - 1);
  if (numel < 128) max_pad = std::min(numel * bm, max_pad);
  return max_pad;
}

static inline int calc_max_pad_8(int numel) {
  return calc_max_pad_bm(numel, 8);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
py_route_align_1k(torch::Tensor x, torch::Tensor w, torch::Tensor bias,
                  c10::optional<torch::Tensor> is_pad, torch::Tensor g_logits,
                  torch::Tensor g_counter, double route_scale) {
  int T = x.size(0);
  int numel = T * 8;
  int max_pad = calc_max_pad_8(numel);
  int max_mb = (max_pad + 7) / 8;
  auto oi = x.options().dtype(torch::kInt32);
  auto ids = torch::empty({T, 8}, oi);
  auto weights = torch::empty({numel}, x.options().dtype(torch::kFloat32));
  auto src2dst = torch::empty({numel}, oi);
  auto sorted_ids = torch::empty({max_pad}, oi);
  auto expert_ids = torch::empty({max_mb}, oi);
  auto num_padded = torch::empty({1}, oi);
  const bool* pad_p = is_pad.has_value() ? is_pad->data_ptr<bool>() : nullptr;
  auto st = at::cuda::getCurrentCUDAStream();
  if (T <= 5) {
    sm80_route_align_1k_v6_kernel<5><<<128, 288, 0, st>>>(
        (const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)w.data_ptr(),
        bias.data_ptr<float>(), pad_p, g_logits.data_ptr<float>(),
        g_counter.data_ptr<int32_t>(), ids.data_ptr<int32_t>(),
        weights.data_ptr<float>(), src2dst.data_ptr<int32_t>(),
        sorted_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
        num_padded.data_ptr<int32_t>(), T, (float)route_scale);
  } else {
    sm80_route_align_1k_v6_kernel<8><<<128, 288, 0, st>>>(
        (const __nv_bfloat16*)x.data_ptr(), (const __nv_bfloat16*)w.data_ptr(),
        bias.data_ptr<float>(), pad_p, g_logits.data_ptr<float>(),
        g_counter.data_ptr<int32_t>(), ids.data_ptr<int32_t>(),
        weights.data_ptr<float>(), src2dst.data_ptr<int32_t>(),
        sorted_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
        num_padded.data_ptr<int32_t>(), T, (float)route_scale);
  }
  return {ids, weights, src2dst, sorted_ids, expert_ids, num_padded};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
py_select_align(torch::Tensor partials, torch::Tensor bias,
                c10::optional<torch::Tensor> is_pad, torch::Tensor g_counts,
                torch::Tensor g_row_rank, torch::Tensor g_done,
                double route_scale, int block_size) {
  int G = partials.size(0);
  int T = partials.size(1);
  int numel = T * 8;
  int bm = (block_size == 16) ? 16 : 8;
  int max_pad = calc_max_pad_bm(numel, bm);
  int max_mb = (max_pad + bm - 1) / bm;
  auto oi = partials.options().dtype(torch::kInt32);
  auto ids = torch::empty({T, 8}, oi);
  auto weights =
      torch::empty({numel}, partials.options().dtype(torch::kFloat32));
  auto src2dst = torch::empty({numel}, oi);
  auto sorted_ids = torch::empty({max_pad}, oi);
  auto expert_ids = torch::empty({max_mb}, oi);
  auto num_padded = torch::empty({1}, oi);

  const bool* pad_p = is_pad.has_value() ? is_pad->data_ptr<bool>() : nullptr;
  auto st = at::cuda::getCurrentCUDAStream();
  int n_ctas = (T + 3) / 4;
  if (n_ctas > 0) {
#define SA_CALL(GV, BM_V)                                             \
  sm80_select_align_mcta_kernel<GV, BM_V><<<n_ctas, 128, 0, st>>>(    \
      partials.data_ptr<float>(), bias.data_ptr<float>(), pad_p,      \
      g_counts.data_ptr<int32_t>(), g_row_rank.data_ptr<int32_t>(),   \
      g_done.data_ptr<int32_t>(), ids.data_ptr<int32_t>(),            \
      weights.data_ptr<float>(), src2dst.data_ptr<int32_t>(),         \
      sorted_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(), \
      num_padded.data_ptr<int32_t>(), T, (float)route_scale)
    if (G == 3) {
      if (bm == 16) {
        SA_CALL(3, 16);
      } else {
        SA_CALL(3, 8);
      }
    } else {
      if (bm == 16) {
        SA_CALL(1, 16);
      } else {
        SA_CALL(1, 8);
      }
    }
#undef SA_CALL
  }
  return {ids, weights, src2dst, sorted_ids, expert_ids, num_padded};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> py_moe_align_large(
    torch::Tensor ids, int block_size, int num_experts) {
  int numel = ids.numel();
  int max_pad = numel + num_experts * (block_size - 1);
  if (numel < num_experts) max_pad = std::min(numel * block_size, max_pad);
  int max_blocks = (max_pad + block_size - 1) / block_size;
  auto oi = ids.options().dtype(torch::kInt32);
  auto sorted_ids = torch::full({max_pad}, numel, oi);
  auto expert_ids = torch::zeros({max_blocks}, oi);
  auto num_padded = torch::empty({1}, oi);
  auto counts = torch::zeros({num_experts}, oi);
  auto offsets = torch::empty({num_experts}, oi);
  auto st = at::cuda::getCurrentCUDAStream();
  int grid = std::min(148, (numel + 255) / 256);
  if (grid > 0) {
    sm80_moe_align_hist_kernel<<<grid, 256, 0, st>>>(ids.data_ptr<int32_t>(),
                                                     counts.data_ptr<int32_t>(),
                                                     numel, num_experts);
  }
  sm80_moe_align_prefix_kernel<<<1, 256, 0, st>>>(
      counts.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
      expert_ids.data_ptr<int32_t>(), num_padded.data_ptr<int32_t>(),
      num_experts, block_size);
  if (grid > 0) {
    sm80_moe_align_scatter_kernel<<<grid, 256, 0, st>>>(
        ids.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
        sorted_ids.data_ptr<int32_t>(), numel, num_experts);
  }
  return {sorted_ids, expert_ids, num_padded};
}

std::tuple<torch::Tensor, torch::Tensor> py_markov_rnn_step(
    torch::Tensor state, torch::Tensor prev_tokens, torch::Tensor w1_embed,
    torch::Tensor rnn_w, torch::Tensor context) {
  int B = state.size(0);
  auto out_state = torch::empty_like(state);
  auto out_latent = torch::empty_like(state);
  auto st = at::cuda::getCurrentCUDAStream();
  if (prev_tokens.scalar_type() == torch::kInt64) {
    sm80_markov_rnn_step_kernel<int64_t><<<dim3(16, B), 128, 0, st>>>(
        (const __nv_bfloat16*)state.data_ptr(), prev_tokens.data_ptr<int64_t>(),
        (const __nv_bfloat16*)w1_embed.data_ptr(),
        (const __nv_bfloat16*)rnn_w.data_ptr(), context.data_ptr<float>(),
        (__nv_bfloat16*)out_state.data_ptr(),
        (__nv_bfloat16*)out_latent.data_ptr(), rnn_w.stride(0));
  } else {
    sm80_markov_rnn_step_kernel<int32_t><<<dim3(16, B), 128, 0, st>>>(
        (const __nv_bfloat16*)state.data_ptr(), prev_tokens.data_ptr<int32_t>(),
        (const __nv_bfloat16*)w1_embed.data_ptr(),
        (const __nv_bfloat16*)rnn_w.data_ptr(), context.data_ptr<float>(),
        (__nv_bfloat16*)out_state.data_ptr(),
        (__nv_bfloat16*)out_latent.data_ptr(), rnn_w.stride(0));
  }
  return {out_state, out_latent};
}

void py_draft_ctx_kv(torch::Tensor kv_proj, torch::Tensor k_norm_w,
                     torch::Tensor cos_sin, torch::Tensor positions,
                     torch::Tensor slot_mapping, torch::Tensor key_cache,
                     torch::Tensor val_cache, double eps) {
  int T = slot_mapping.size(0);
  if (T == 0) return;
  int hkv = key_cache.size(2);
  auto st = at::cuda::getCurrentCUDAStream();
  if (positions.scalar_type() == torch::kInt64) {
    sm80_draft_ctx_kv_kernel<int64_t><<<dim3(T, hkv), 64, 0, st>>>(
        (const __nv_bfloat16*)kv_proj.data_ptr(),
        (const __nv_bfloat16*)k_norm_w.data_ptr(), cos_sin.data_ptr<float>(),
        positions.data_ptr<int64_t>(), slot_mapping.data_ptr<int64_t>(),
        (__nv_bfloat16*)key_cache.data_ptr(),
        (__nv_bfloat16*)val_cache.data_ptr(), kv_proj.stride(0), hkv * 128,
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        val_cache.stride(0), val_cache.stride(1), val_cache.stride(2),
        key_cache.size(1), (float)eps);
  } else {
    sm80_draft_ctx_kv_kernel<int32_t><<<dim3(T, hkv), 64, 0, st>>>(
        (const __nv_bfloat16*)kv_proj.data_ptr(),
        (const __nv_bfloat16*)k_norm_w.data_ptr(), cos_sin.data_ptr<float>(),
        positions.data_ptr<int32_t>(), slot_mapping.data_ptr<int64_t>(),
        (__nv_bfloat16*)key_cache.data_ptr(),
        (__nv_bfloat16*)val_cache.data_ptr(), kv_proj.stride(0), hkv * 128,
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        val_cache.stride(0), val_cache.stride(1), val_cache.stride(2),
        key_cache.size(1), (float)eps);
  }
}

void py_draft_qk_norm_rope_cache(torch::Tensor qkv, torch::Tensor q_norm_w,
                                 torch::Tensor k_norm_w, torch::Tensor cos_sin,
                                 torch::Tensor positions,
                                 torch::Tensor slot_mapping,
                                 torch::Tensor key_cache,
                                 torch::Tensor val_cache, int nhq, int nhk,
                                 double eps) {
  int T = qkv.size(0);
  if (T == 0) return;
  auto st = at::cuda::getCurrentCUDAStream();
  if (positions.scalar_type() == torch::kInt64) {
    sm80_draft_qk_norm_rope_cache_kernel<int64_t>
        <<<dim3(T, nhq + nhk), 64, 0, st>>>(
            (__nv_bfloat16*)qkv.data_ptr(),
            (const __nv_bfloat16*)q_norm_w.data_ptr(),
            (const __nv_bfloat16*)k_norm_w.data_ptr(),
            cos_sin.data_ptr<float>(), positions.data_ptr<int64_t>(),
            slot_mapping.data_ptr<int64_t>(),
            (__nv_bfloat16*)key_cache.data_ptr(),
            (__nv_bfloat16*)val_cache.data_ptr(), qkv.stride(0), nhq * 128,
            nhk * 128, nhq, slot_mapping.size(0), key_cache.stride(0),
            key_cache.stride(1), key_cache.stride(2), val_cache.stride(0),
            val_cache.stride(1), val_cache.stride(2), key_cache.size(1),
            (float)eps);
  } else {
    sm80_draft_qk_norm_rope_cache_kernel<int32_t>
        <<<dim3(T, nhq + nhk), 64, 0, st>>>(
            (__nv_bfloat16*)qkv.data_ptr(),
            (const __nv_bfloat16*)q_norm_w.data_ptr(),
            (const __nv_bfloat16*)k_norm_w.data_ptr(),
            cos_sin.data_ptr<float>(), positions.data_ptr<int32_t>(),
            slot_mapping.data_ptr<int64_t>(),
            (__nv_bfloat16*)key_cache.data_ptr(),
            (__nv_bfloat16*)val_cache.data_ptr(), qkv.stride(0), nhq * 128,
            nhk * 128, nhq, slot_mapping.size(0), key_cache.stride(0),
            key_cache.stride(1), key_cache.stride(2), val_cache.stride(0),
            val_cache.stride(1), val_cache.stride(2), key_cache.size(1),
            (float)eps);
  }
}

torch::Tensor py_draft_embed_block(torch::Tensor input_ids,
                                   torch::Tensor embed_w, torch::Tensor norm_w,
                                   torch::Tensor mask_e, int block_size,
                                   double eps) {
  int total = input_ids.size(0);
  int n_reqs = total / block_size;
  auto out = torch::empty({total, 2304}, embed_w.options());
  if (n_reqs == 0) return out;
  auto st = at::cuda::getCurrentCUDAStream();
  if (input_ids.scalar_type() == torch::kInt64) {
    sm80_draft_embed_block_kernel<int64_t>
        <<<dim3(n_reqs, block_size), 288, 0, st>>>(
            input_ids.data_ptr<int64_t>(),
            (const __nv_bfloat16*)embed_w.data_ptr(),
            (const __nv_bfloat16*)norm_w.data_ptr(),
            (const __nv_bfloat16*)mask_e.data_ptr(),
            (__nv_bfloat16*)out.data_ptr(), block_size, (float)eps);
  } else {
    sm80_draft_embed_block_kernel<int32_t>
        <<<dim3(n_reqs, block_size), 288, 0, st>>>(
            input_ids.data_ptr<int32_t>(),
            (const __nv_bfloat16*)embed_w.data_ptr(),
            (const __nv_bfloat16*)norm_w.data_ptr(),
            (const __nv_bfloat16*)mask_e.data_ptr(),
            (__nv_bfloat16*)out.data_ptr(), block_size, (float)eps);
  }
  return out;
}

// =========================================================================
// 11. FUSED B=1 markov_fold + add_softcap + draft_logits_cache + gumbel_argmax
//     4-row-unrolled evict_first, 0 spills -> 1.17-1.22x faster (-60..78us/step
//     at 1u)
// =========================================================================
__device__ __forceinline__ uint4 mfg_ldg_ef_u4(const uint4* ptr) {
  uint4 r;
  asm volatile("ld.global.nc.L1::evict_first.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(ptr));
  return r;
}
__device__ __forceinline__ float mfg_fast_tanh(float x) {
  float r;
  asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ uint32_t mfg_rotl32(uint32_t v, int s) {
  return (v << s) | (v >> (32 - s));
}
__device__ __forceinline__ uint32_t mfg_m3_mix(uint32_t h, uint32_t k) {
  k *= 0xCC9E2D51u;
  k = mfg_rotl32(k, 15);
  k *= 0x1B873593u;
  h ^= k;
  h = mfg_rotl32(h, 13);
  return h * 5u + 0xE6546B64u;
}
__device__ __forceinline__ uint32_t mfg_m3_fmix32(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85EBCA6Bu;
  h ^= h >> 13;
  h *= 0xC2B2AE35u;
  return h ^ (h >> 16);
}
__device__ __forceinline__ float mfg_murmur3_uniform32(int64_t seed,
                                                       int64_t pos,
                                                       uint32_t offset) {
  uint32_t h = 0u;
  h = mfg_m3_mix(h, (uint32_t)(seed & 0xFFFFFFFFLL));
  h = mfg_m3_mix(h, (uint32_t)(((uint64_t)seed >> 32) & 0xFFFFFFFFULL));
  h = mfg_m3_mix(h, (uint32_t)(pos & 0xFFFFFFFFLL));
  h = mfg_m3_mix(h, offset);
  uint32_t r = mfg_m3_fmix32(h ^ 16u);
  return (float)(int32_t)(r >> 16) * 1.52587890625e-05f +
         ((float)(int32_t)(r & 0xFFFFu) + 0.5f) * 2.3283064365386963e-10f;
}
__device__ __forceinline__ float mfg_log1p_neg_stable(float v) {
  if (v < 0.25f) {
    float p = 1.0f / 8.0f;
    p = 1.0f / 7.0f + v * p;
    p = 1.0f / 6.0f + v * p;
    p = 1.0f / 5.0f + v * p;
    p = 1.0f / 4.0f + v * p;
    p = 1.0f / 3.0f + v * p;
    p = 1.0f / 2.0f + v * p;
    p = 1.0f + v * p;
    return -v * p;
  }
  return __logf(fmaxf(1.0f - v, 5.960464477539063e-08f));
}

template <typename IdxT, typename PosT>
__global__ __launch_bounds__(256, 4) void sm80_mfg_b1_fast_kernel(
    const __nv_bfloat16* __restrict__ Latent,
    const __nv_bfloat16* __restrict__ FoldW,
    const __nv_bfloat16* __restrict__ FoldBias,
    const __nv_bfloat16* __restrict__ BaseLogits,
    __nv_bfloat16* __restrict__ LogitsCache, int64_t* __restrict__ LocalArgmax,
    float* __restrict__ LocalMax, const IdxT* __restrict__ IdxMap,
    const int64_t* __restrict__ Seeds, const PosT* __restrict__ Pos,
    const float* __restrict__ Temps, const int32_t* __restrict__ LcColPtr,
    const int32_t* __restrict__ LcRowMap, int64_t lc_s0, int64_t lc_s1,
    int32_t V, float softcap) {
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int v_block0 = blockIdx.x * 256;
  const int v_warp0 = v_block0 + wid * 32;

  uint4 lu = __ldg(reinterpret_cast<const uint4*>(Latent + lane * 8));
  const __nv_bfloat162* lb = reinterpret_cast<const __nv_bfloat162*>(&lu);
  float2 lf0 = __bfloat1622float2(lb[0]), lf1 = __bfloat1622float2(lb[1]);
  float2 lf2 = __bfloat1622float2(lb[2]), lf3 = __bfloat1622float2(lb[3]);

  int32_t col = *LcColPtr;
  int64_t req = (int64_t)IdxMap[0];
  int64_t crow =
      (req >= 0) ? ((LcRowMap != nullptr) ? (int64_t)LcRowMap[req] : req) : -1;
  float temp = (req >= 0) ? Temps[req] : 0.0f;
  int64_t seed = (req >= 0) ? (Seeds[req] ^ ((int64_t)col << 32)) : 0;
  int64_t pos = (req >= 0) ? ((int64_t)Pos[0] - 1 + (1LL << 30)) : 0;
  __nv_bfloat16* cache_row_ptr =
      (crow >= 0) ? (LogitsCache + crow * lc_s0 + (int64_t)col * lc_s1)
                  : nullptr;

  float best_v = -INFINITY;
  int32_t best_i = v_block0;
  const float inv_sc = 1.0f / softcap;
  __shared__ float s_dot[8][32];

#pragma unroll
  for (int r = 0; r < 32; r += 4) {
    int v0 = v_warp0 + r;
    uint4 wu0 = (v0 < V) ? mfg_ldg_ef_u4(reinterpret_cast<const uint4*>(
                               FoldW + (int64_t)(v0) * 256 + lane * 8))
                         : make_uint4(0, 0, 0, 0);
    uint4 wu1 = (v0 + 1 < V) ? mfg_ldg_ef_u4(reinterpret_cast<const uint4*>(
                                   FoldW + (int64_t)(v0 + 1) * 256 + lane * 8))
                             : make_uint4(0, 0, 0, 0);
    uint4 wu2 = (v0 + 2 < V) ? mfg_ldg_ef_u4(reinterpret_cast<const uint4*>(
                                   FoldW + (int64_t)(v0 + 2) * 256 + lane * 8))
                             : make_uint4(0, 0, 0, 0);
    uint4 wu3 = (v0 + 3 < V) ? mfg_ldg_ef_u4(reinterpret_cast<const uint4*>(
                                   FoldW + (int64_t)(v0 + 3) * 256 + lane * 8))
                             : make_uint4(0, 0, 0, 0);

    auto dot8 = [&](uint4 wu) -> float {
      const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wu);
      float2 w0 = __bfloat1622float2(wb[0]), w1 = __bfloat1622float2(wb[1]);
      float2 w2 = __bfloat1622float2(wb[2]), w3 = __bfloat1622float2(wb[3]);
      return __fmaf_rn(
          lf0.x, w0.x,
          __fmaf_rn(
              lf0.y, w0.y,
              __fmaf_rn(
                  lf1.x, w1.x,
                  __fmaf_rn(lf1.y, w1.y,
                            __fmaf_rn(lf2.x, w2.x,
                                      __fmaf_rn(lf2.y, w2.y,
                                                __fmaf_rn(lf3.x, w3.x,
                                                          lf3.y * w3.y)))))));
    };
    float d0 = warp_sum(dot8(wu0));
    float d1 = warp_sum(dot8(wu1));
    float d2 = warp_sum(dot8(wu2));
    float d3 = warp_sum(dot8(wu3));
    if (lane == 0) {
      s_dot[wid][r] = d0;
      s_dot[wid][r + 1] = d1;
      s_dot[wid][r + 2] = d2;
      s_dot[wid][r + 3] = d3;
    }
  }
  __syncwarp();

  int v = v_warp0 + lane;
  if (v < V) {
    float fb = __bfloat162float(__ldg(FoldBias + v));
    float bias_f = __bfloat162float(__float2bfloat16_rn(s_dot[wid][lane] + fb));
    float base_f = __bfloat162float(__ldg(BaseLogits + v));
    float s_bf = __bfloat162float(__float2bfloat16_rn(base_f + bias_f));
    __nv_bfloat16 s_cap_bf =
        __float2bfloat16_rn(mfg_fast_tanh(s_bf * inv_sc) * softcap);
    if (cache_row_ptr != nullptr) cache_row_ptr[v] = s_cap_bf;
    float lg = __bfloat162float(s_cap_bf);
    if (temp != 0.0f) {
      lg = lg / temp + (-__logf(-mfg_log1p_neg_stable(
                           mfg_murmur3_uniform32(seed, pos, (uint32_t)v))));
    }
    best_v = lg;
    best_i = v;
  }

#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    float ov = __shfl_xor_sync(0xffffffff, best_v, o);
    int32_t oi = __shfl_xor_sync(0xffffffff, best_i, o);
    if (ov > best_v || (ov == best_v && oi < best_i)) {
      best_v = ov;
      best_i = oi;
    }
  }
  __shared__ float s_bv[8];
  __shared__ int32_t s_bi[8];
  if (lane == 0) {
    s_bv[wid] = best_v;
    s_bi[wid] = best_i;
  }
  __syncthreads();
  if (wid == 0) {
    float bv = (lane < 8) ? s_bv[lane] : -INFINITY;
    int32_t bi = (lane < 8) ? s_bi[lane] : v_block0;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffff, bv, o);
      int32_t oi = __shfl_xor_sync(0xffffffff, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if (lane == 0) {
      LocalMax[blockIdx.x] = bv;
      LocalArgmax[blockIdx.x] = (int64_t)bi;
    }
  }
}

void py_fused_markov_fold_gumbel_b1(
    torch::Tensor latent, torch::Tensor fold_w, torch::Tensor fold_b,
    torch::Tensor base_logits, torch::Tensor logits_cache, torch::Tensor la,
    torch::Tensor lm, torch::Tensor idx_map, torch::Tensor seeds,
    torch::Tensor pos, torch::Tensor temps, torch::Tensor lc_col,
    c10::optional<torch::Tensor> lc_row_map, int64_t vocab_size,
    double softcap) {
  int V = (int)vocab_size;
  int grid = (V + 255) / 256;
  const int32_t* rmap =
      lc_row_map.has_value() ? lc_row_map->data_ptr<int32_t>() : nullptr;
  auto st = at::cuda::getCurrentCUDAStream();
  bool idx64 = (idx_map.scalar_type() == torch::kInt64);
  bool pos64 = (pos.scalar_type() == torch::kInt64);
#define MFG_CALL(IT, PT)                                                       \
  sm80_mfg_b1_fast_kernel<IT, PT><<<grid, 256, 0, st>>>(                       \
      (const __nv_bfloat16*)latent.data_ptr(),                                 \
      (const __nv_bfloat16*)fold_w.data_ptr(),                                 \
      (const __nv_bfloat16*)fold_b.data_ptr(),                                 \
      (const __nv_bfloat16*)base_logits.data_ptr(),                            \
      (__nv_bfloat16*)logits_cache.data_ptr(), la.data_ptr<int64_t>(),         \
      lm.data_ptr<float>(), idx_map.data_ptr<IT>(), seeds.data_ptr<int64_t>(), \
      pos.data_ptr<PT>(), temps.data_ptr<float>(), lc_col.data_ptr<int32_t>(), \
      rmap, logits_cache.stride(0), logits_cache.stride(1), V, (float)softcap)
  if (!idx64 && pos64) {
    MFG_CALL(int32_t, int64_t);
  } else if (!idx64 && !pos64) {
    MFG_CALL(int32_t, int32_t);
  } else if (idx64 && pos64) {
    MFG_CALL(int64_t, int64_t);
  } else {
    MFG_CALL(int64_t, int32_t);
  }
#undef MFG_CALL
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_mix", &py_conv_mix);
  m.def("batched_bank_messages", &py_batched_bank_messages);
  m.def("bank_stats", &py_bank_stats);
  m.def("bank_rank2_finalize", &py_bank_rank2_finalize);
  m.def("pair_rms_norm_rank2", &py_pair_rms_norm_rank2);
  m.def("fused_pair_norm_ctrl_down", &py_fused_pair_norm_ctrl_down);
  m.def("rms_norm_f32_to_bf16", &py_rms_norm_f32_to_bf16);
  m.def("route_align_1k", &py_route_align_1k);
  m.def("select_align", &py_select_align);
  m.def("moe_align_large", &py_moe_align_large);
  m.def("markov_rnn_step", &py_markov_rnn_step);
  m.def("draft_ctx_kv", &py_draft_ctx_kv);
  m.def("draft_qk_norm_rope_cache", &py_draft_qk_norm_rope_cache);
  m.def("draft_embed_block", &py_draft_embed_block);
  m.def("fused_markov_fold_gumbel_b1", &py_fused_markov_fold_gumbel_b1);
}
