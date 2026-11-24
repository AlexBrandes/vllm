// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// DeepSeek-V4.1 sparse attention with sinks, all 64 heads per CTA (sm_80 prototype).
//
// One CTA per (query row, segment of KV tiles). The row's Q [64, 512] is staged in shared memory once; each tile of
// 32 selected entries (window slots first, then compressed positions) is decoded once from its packed record
// (MXFP8 window / NVFP4 compressed, sm80_kv_pack) or loaded as BF16 into shared memory, instead of once per
// 16-head program. S = Q KV^T: warp w computes heads [16 (w & 3), +16) x entries [16 (w >> 2), +16) over D = 512.
// Online softmax per head in shared memory (fp32 scores x scale, finite floor -1e30, p rounded to BF16 for the PV
// product, running sum in fp32). O = P KV: warp w owns output dims [64 w, +64) for all 64 heads (128 fp32
// accumulators per thread). Segments > 1 write fp32 partials (m, l, acc) merged by a combine kernel with the sink.
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

namespace {

constexpr int H = 64, D = 512, TN = 32, NT = 256;
constexpr int QLD = D + 8;     // padded row strides (bf16 elements) against bank conflicts
constexpr int KLD = D + 8;
constexpr int SLD = TN + 1;    // fp32
constexpr int PLD = TN + 8;    // bf16
constexpr size_t SMEM = size_t(H) * QLD * 2 + size_t(TN) * KLD * 2 + size_t(H) * SLD * 4 + size_t(H) * PLD * 2 +
                        3 * H * 4 + TN * 4;

__device__ __forceinline__ float e4m3_to_f(uint32_t b) {
  // OCP E4M3FN: no inf; 0x7f / 0xff NaN (never produced by the encoder).
  const uint32_t s = (b & 0x80u) << 24, e = (b >> 3) & 0xFu, m = b & 0x7u;
  if (e == 0) {
    const float v = __uint_as_float(0x3B000000u) * float(m);  // m * 2^-9
    return s ? -v : v;
  }
  return __uint_as_float(s | ((e + 120u) << 23) | (m << 20));
}
__device__ __forceinline__ float e2m1_to_f(uint32_t n) {
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float v = mag[n & 7u];
  return (n & 8u) ? -v : v;
}
__device__ __forceinline__ float e8m0_to_f(uint32_t b) { return __uint_as_float(b << 23); }  // 2^(b - 127)

__device__ __forceinline__ void mma_bf16(const uint32_t (&a)[4], uint32_t b0, uint32_t b1, float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4(const void* p, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(const void* p, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
  return uint32_t(__bfloat16_as_ushort(lo)) | (uint32_t(__bfloat16_as_ushort(hi)) << 16);
}

struct Args {
  const __nv_bfloat16* q; int64_t sq_t, sq_h;
  const uint8_t* swa; int64_t s_sb, s_se; int swa_e; bool swa_packed;
  const int32_t* swa_idx; int64_t s_wi; int n_window;
  const uint8_t* comp; int64_t s_cb, s_ce; int comp_e; bool comp_packed;
  const int32_t* bt; int64_t s_bt;
  const int32_t* comp_idx; int64_t s_ci; int n_comp;
  const int32_t* req;
  const float* sink;
  float scale;
  int tiles_per_seg, n_seg;
  __nv_bfloat16* out; int64_t so_t, so_h;
  float* pm; float* pl; float* pacc;  // partials [rows, n_seg, H] / [rows, n_seg, H, D]
};

// Decode 64 contiguous values [64 c, 64 c + 64) of one entry into shared memory (BF16).
__device__ __forceinline__ void load_entry_chunk(const Args& a, bool window, int64_t off, bool ok, int c,
                                                 __nv_bfloat16* dst) {
  if (!ok) {
#pragma unroll
    for (int i = 0; i < 64; i += 8) *reinterpret_cast<uint4*>(dst + i) = make_uint4(0, 0, 0, 0);
    return;
  }
  const bool packed = window ? a.swa_packed : a.comp_packed;
  const uint8_t* base = (window ? a.swa : a.comp) + off;
  if (!packed) {
    const uint4* src = reinterpret_cast<const uint4*>(base) + c * 8;  // 64 bf16 = 8 x 16 B
#pragma unroll
    for (int i = 0; i < 8; ++i) reinterpret_cast<uint4*>(dst)[i] = src[i];
    return;
  }
  if (window) {  // MXFP8: 512 e4m3 codes, then 16 UE8M0 (one per 32 values)
    uint32_t codes[16];
#pragma unroll
    for (int i = 0; i < 4; ++i) *reinterpret_cast<uint4*>(codes + 4 * i) = reinterpret_cast<const uint4*>(base + 64 * c)[i];
    const float s0 = e8m0_to_f(base[512 + 2 * c]), s1 = e8m0_to_f(base[512 + 2 * c + 1]);
#pragma unroll
    for (int w = 0; w < 16; ++w) {
      const float s = w < 8 ? s0 : s1;
      const uint32_t x = codes[w];
      dst[4 * w + 0] = __float2bfloat16_rn(e4m3_to_f(x & 0xFF) * s);
      dst[4 * w + 1] = __float2bfloat16_rn(e4m3_to_f((x >> 8) & 0xFF) * s);
      dst[4 * w + 2] = __float2bfloat16_rn(e4m3_to_f((x >> 16) & 0xFF) * s);
      dst[4 * w + 3] = __float2bfloat16_rn(e4m3_to_f(x >> 24) * s);
    }
  } else {  // NVFP4: 256 bytes of e2m1 pairs (even element low nibble), then 32 e4m3 scales (one per 16 values)
    uint32_t codes[8];
#pragma unroll
    for (int i = 0; i < 2; ++i) *reinterpret_cast<uint4*>(codes + 4 * i) = reinterpret_cast<const uint4*>(base + 32 * c)[i];
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      const float s = e4m3_to_f(base[256 + 4 * c + (w >> 1)]);
      const uint32_t x = codes[w];
#pragma unroll
      for (int j = 0; j < 8; ++j) dst[8 * w + j] = __float2bfloat16_rn(e2m1_to_f((x >> (4 * j)) & 0xF) * s);
    }
  }
}

__global__ void __launch_bounds__(NT, 1) sparse_mla_kernel(Args a) {
  extern __shared__ __align__(16) uint8_t smem[];
  __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Ks = Qs + H * QLD;
  float* Ss = reinterpret_cast<float*>(Ks + TN * KLD);
  __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss + H * SLD);
  float* Ms = reinterpret_cast<float*>(Ps + H * PLD);
  float* Ls = Ms + H;
  float* As = Ls + H;
  int* Vs = reinterpret_cast<int*>(As + H);  // per-entry validity of the current tile

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
  const int64_t row = blockIdx.x;
  const int seg = blockIdx.y;
  const int req = a.req[row];

  // Q -> shared (64 heads x 512, 16 B per thread step)
  for (int i = tid; i < H * D / 8; i += NT) {
    const int h = i / (D / 8), c = i % (D / 8);
    *reinterpret_cast<uint4*>(Qs + h * QLD + 8 * c) =
        *reinterpret_cast<const uint4*>(a.q + row * a.sq_t + h * a.sq_h + 8 * c);
  }
  if (tid < H) { Ms[tid] = -1.0e30f; Ls[tid] = 0.f; }

  float acc[4][8][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;

  const int win_tiles = (a.n_window + TN - 1) / TN;
  const int comp_tiles = (a.n_comp + TN - 1) / TN;
  const int total = win_tiles + comp_tiles;
  const int tb = seg * a.tiles_per_seg, te = min(total, tb + a.tiles_per_seg);

  for (int tile = tb; tile < te; ++tile) {
    const bool window = tile < win_tiles;
    // ---- decode the tile's entries into Ks: thread -> (entry tid / 8, 64-value chunk tid % 8)
    {
      const int e = tid >> 3, c = tid & 7;
      bool ok;
      int64_t off = 0;
      if (window) {
        const int col = tile * TN + e;
        const int slot = col < a.n_window ? a.swa_idx[row * a.s_wi + col] : -1;
        ok = slot >= 0;
        if (ok) off = int64_t(slot / a.swa_e) * a.s_sb + int64_t(slot % a.swa_e) * a.s_se;
      } else {
        const int col = (tile - win_tiles) * TN + e;
        const int j = col < a.n_comp ? a.comp_idx[row * a.s_ci + col] : -1;
        ok = j >= 0;
        if (ok) {
          const int blk = a.bt[int64_t(req) * a.s_bt + j / a.comp_e];
          off = int64_t(blk) * a.s_cb + int64_t(j % a.comp_e) * a.s_ce;
        }
      }
      load_entry_chunk(a, window, off, ok, c, Ks + e * KLD + 64 * c);
      if (c == 0) Vs[e] = ok;
    }
    __syncthreads();
    // ---- S = Q KV^T * scale (warp: 16 heads x 16 entries)
    {
      const int h0 = 16 * (warp & 3), n0 = 16 * (warp >> 2);
      float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
      // ldmatrix addresses: lane l feeds row (l % 8) of 8x8 matrix l / 8.
      const __nv_bfloat16* qa = Qs + (h0 + (lane & 7) + ((lane >> 3) & 1) * 8) * QLD + (lane >> 4) * 8;
      const __nv_bfloat16* ka = Ks + (n0 + (lane & 7) + (lane >> 4) * 8) * KLD + ((lane >> 3) & 1) * 8;
#pragma unroll 4
      for (int k0 = 0; k0 < D; k0 += 16) {
        uint32_t af[4], b00, b01, b10, b11;
        ldsm_x4(qa + k0, af[0], af[1], af[2], af[3]);
        ldsm_x4(ka + k0, b00, b01, b10, b11);
        mma_bf16(af, b00, b01, s[0]);
        mma_bf16(af, b10, b11, s[1]);
      }
      const int* valid = Vs;
#pragma unroll
      for (int nb = 0; nb < 2; ++nb) {
        const int e0 = n0 + 8 * nb + 2 * t4;
        Ss[(h0 + g) * SLD + e0] = valid[e0] ? s[nb][0] * a.scale : -INFINITY;
        Ss[(h0 + g) * SLD + e0 + 1] = valid[e0 + 1] ? s[nb][1] * a.scale : -INFINITY;
        Ss[(h0 + g + 8) * SLD + e0] = valid[e0] ? s[nb][2] * a.scale : -INFINITY;
        Ss[(h0 + g + 8) * SLD + e0 + 1] = valid[e0 + 1] ? s[nb][3] * a.scale : -INFINITY;
      }
    }
    __syncthreads();
    // ---- online softmax: 4 threads per head, 8 entries each
    {
      const int h = tid >> 2, part = tid & 3;
      float mx = -INFINITY;
#pragma unroll
      for (int i = 0; i < 8; ++i) mx = fmaxf(mx, Ss[h * SLD + 8 * part + i]);
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float m_old = Ms[h];
      const float m_new = fmaxf(m_old, mx);
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float p = expf(Ss[h * SLD + 8 * part + i] - m_new);
        sum += p;
        Ps[h * PLD + 8 * part + i] = __float2bfloat16_rn(p);
      }
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      sum += __shfl_xor_sync(0xffffffffu, sum, 2);
      __syncwarp();
      if (part == 0) {
        const float alpha = expf(m_old - m_new);
        As[h] = alpha;
        Ls[h] = Ls[h] * alpha + sum;
        Ms[h] = m_new;
      }
    }
    __syncthreads();
    // ---- O = O * alpha + P KV (warp: all heads x dims [64 warp, +64))
    {
      const int d0 = 64 * warp;
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        const float a0 = As[16 * mt + g], a1 = As[16 * mt + g + 8];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          acc[mt][nt][0] *= a0; acc[mt][nt][1] *= a0; acc[mt][nt][2] *= a1; acc[mt][nt][3] *= a1;
        }
      }
#pragma unroll
      for (int k0 = 0; k0 < TN; k0 += 16) {
        // B fragments of all 8 n-tiles for this k step: KV rows k0.. (entries) x dims, transposed 8x8 loads;
        // lane l: entry k0 + (l % 8) + ((l / 8) % 2) * 8, dims + (l / 16) * 8 -> {b0, b1} of n-tiles nt, nt + 1.
        uint32_t bf[8][2];
#pragma unroll
        for (int nt = 0; nt < 8; nt += 2) {
          const __nv_bfloat16* kb = Ks + (k0 + (lane & 7) + ((lane >> 3) & 1) * 8) * KLD + d0 + 8 * nt + (lane >> 4) * 8;
          ldsm_x4_t(kb, bf[nt][0], bf[nt][1], bf[nt + 1][0], bf[nt + 1][1]);
        }
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          uint32_t af[4];
          ldsm_x4(Ps + (16 * mt + (lane & 7) + ((lane >> 3) & 1) * 8) * PLD + k0 + (lane >> 4) * 8,
                  af[0], af[1], af[2], af[3]);
#pragma unroll
          for (int nt = 0; nt < 8; ++nt) mma_bf16(af, bf[nt][0], bf[nt][1], acc[mt][nt]);
        }
      }
    }
    __syncthreads();
  }

  // ---- epilogue
  const int d0 = 64 * warp;
  if (a.n_seg == 1) {
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int h = 16 * mt + g + 8 * half;
        const float m = Ms[h];
        const float l = Ls[h] + expf(a.sink[h] - m);
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const int d = d0 + 8 * nt + 2 * t4;
          __nv_bfloat16* o = a.out + row * a.so_t + h * a.so_h + d;
          o[0] = __float2bfloat16_rn(acc[mt][nt][2 * half] / l);
          o[1] = __float2bfloat16_rn(acc[mt][nt][2 * half + 1] / l);
        }
      }
    }
  } else {
    const int64_t base = (row * a.n_seg + seg) * H;
    if (tid < H) { a.pm[base + tid] = Ms[tid]; a.pl[base + tid] = Ls[tid]; }
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int h = 16 * mt + g + 8 * half;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const int d = d0 + 8 * nt + 2 * t4;
          float* p = a.pacc + (base + h) * D + d;
          p[0] = acc[mt][nt][2 * half];
          p[1] = acc[mt][nt][2 * half + 1];
        }
      }
  }
}

// 32 heads per CTA (grid (rows, segments, 2)): 74 KB of shared memory, so two CTAs share an SM. Every output value
// is computed by the same operations in the same order as sparse_mla_kernel's: each S element is the same k16 MMA
// chain, each head's softmax the same 4 x 8-entry partial sums and shuffles, each O element the same alpha scaling
// and PV MMAs; only how the work is split across warps and CTAs differs (bit-identical results).
constexpr int HC = 32;
constexpr size_t SMEM32 = size_t(HC) * QLD * 2 + size_t(TN) * KLD * 2 + size_t(HC) * SLD * 4 + size_t(HC) * PLD * 2 +
                          3 * HC * 4 + TN * 4;

__global__ void __launch_bounds__(NT, 2) sparse_mla_kernel32(Args a) {
  extern __shared__ __align__(16) uint8_t smem[];
  __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Ks = Qs + HC * QLD;
  float* Ss = reinterpret_cast<float*>(Ks + TN * KLD);
  __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss + HC * SLD);
  float* Ms = reinterpret_cast<float*>(Ps + HC * PLD);
  float* Ls = Ms + HC;
  float* As = Ls + HC;
  int* Vs = reinterpret_cast<int*>(As + HC);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
  const int64_t row = blockIdx.x;
  const int seg = blockIdx.y;
  const int hg = blockIdx.z * HC;  // first global head of this CTA
  const int req = a.req[row];

  for (int i = tid; i < HC * D / 8; i += NT) {
    const int h = i / (D / 8), c = i % (D / 8);
    *reinterpret_cast<uint4*>(Qs + h * QLD + 8 * c) =
        *reinterpret_cast<const uint4*>(a.q + row * a.sq_t + (hg + h) * a.sq_h + 8 * c);
  }
  if (tid < HC) { Ms[tid] = -1.0e30f; Ls[tid] = 0.f; }

  float acc[2][8][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;

  const int win_tiles = (a.n_window + TN - 1) / TN;
  const int comp_tiles = (a.n_comp + TN - 1) / TN;
  const int total = win_tiles + comp_tiles;
  const int tb = seg * a.tiles_per_seg, te = min(total, tb + a.tiles_per_seg);

  for (int tile = tb; tile < te; ++tile) {
    const bool window = tile < win_tiles;
    {
      const int e = tid >> 3, c = tid & 7;
      bool ok;
      int64_t off = 0;
      if (window) {
        const int col = tile * TN + e;
        const int slot = col < a.n_window ? a.swa_idx[row * a.s_wi + col] : -1;
        ok = slot >= 0;
        if (ok) off = int64_t(slot / a.swa_e) * a.s_sb + int64_t(slot % a.swa_e) * a.s_se;
      } else {
        const int col = (tile - win_tiles) * TN + e;
        const int j = col < a.n_comp ? a.comp_idx[row * a.s_ci + col] : -1;
        ok = j >= 0;
        if (ok) {
          const int blk = a.bt[int64_t(req) * a.s_bt + j / a.comp_e];
          off = int64_t(blk) * a.s_cb + int64_t(j % a.comp_e) * a.s_ce;
        }
      }
      load_entry_chunk(a, window, off, ok, c, Ks + e * KLD + 64 * c);
      if (c == 0) Vs[e] = ok;
    }
    __syncthreads();
    // ---- S (warp: 16 heads x 8 entries; k steps in order, two per ldmatrix of K)
    {
      const int h0 = 16 * (warp & 1), n0 = 8 * (warp >> 1);
      float s4[4] = {0.f, 0.f, 0.f, 0.f};
      const __nv_bfloat16* qa = Qs + (h0 + (lane & 7) + ((lane >> 3) & 1) * 8) * QLD + (lane >> 4) * 8;
      const __nv_bfloat16* ka = Ks + (n0 + (lane & 7)) * KLD + (lane >> 3) * 8;
#pragma unroll 4
      for (int k0 = 0; k0 < D; k0 += 32) {
        uint32_t af0[4], af1[4], b0, b1, b2, b3;
        ldsm_x4(qa + k0, af0[0], af0[1], af0[2], af0[3]);
        ldsm_x4(qa + k0 + 16, af1[0], af1[1], af1[2], af1[3]);
        ldsm_x4(ka + k0, b0, b1, b2, b3);
        mma_bf16(af0, b0, b1, s4);
        mma_bf16(af1, b2, b3, s4);
      }
      const int e0 = n0 + 2 * t4;
      Ss[(h0 + g) * SLD + e0] = Vs[e0] ? s4[0] * a.scale : -INFINITY;
      Ss[(h0 + g) * SLD + e0 + 1] = Vs[e0 + 1] ? s4[1] * a.scale : -INFINITY;
      Ss[(h0 + g + 8) * SLD + e0] = Vs[e0] ? s4[2] * a.scale : -INFINITY;
      Ss[(h0 + g + 8) * SLD + e0 + 1] = Vs[e0 + 1] ? s4[3] * a.scale : -INFINITY;
    }
    __syncthreads();
    if (tid < HC * 4) {  // the same 4 threads per head, 8 entries each
      const int h = tid >> 2, part = tid & 3;
      float mx = -INFINITY;
#pragma unroll
      for (int i = 0; i < 8; ++i) mx = fmaxf(mx, Ss[h * SLD + 8 * part + i]);
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float m_old = Ms[h];
      const float m_new = fmaxf(m_old, mx);
      float sum = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float p = expf(Ss[h * SLD + 8 * part + i] - m_new);
        sum += p;
        Ps[h * PLD + 8 * part + i] = __float2bfloat16_rn(p);
      }
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      sum += __shfl_xor_sync(0xffffffffu, sum, 2);
      __syncwarp();
      if (part == 0) {
        const float alpha = expf(m_old - m_new);
        As[h] = alpha;
        Ls[h] = Ls[h] * alpha + sum;
        Ms[h] = m_new;
      }
    }
    __syncthreads();
    {
      const int d0 = 64 * warp;
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const float a0 = As[16 * mt + g], a1 = As[16 * mt + g + 8];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          acc[mt][nt][0] *= a0; acc[mt][nt][1] *= a0; acc[mt][nt][2] *= a1; acc[mt][nt][3] *= a1;
        }
      }
#pragma unroll
      for (int k0 = 0; k0 < TN; k0 += 16) {
        uint32_t bf[8][2];
#pragma unroll
        for (int nt = 0; nt < 8; nt += 2) {
          const __nv_bfloat16* kb = Ks + (k0 + (lane & 7) + ((lane >> 3) & 1) * 8) * KLD + d0 + 8 * nt + (lane >> 4) * 8;
          ldsm_x4_t(kb, bf[nt][0], bf[nt][1], bf[nt + 1][0], bf[nt + 1][1]);
        }
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          uint32_t af[4];
          ldsm_x4(Ps + (16 * mt + (lane & 7) + ((lane >> 3) & 1) * 8) * PLD + k0 + (lane >> 4) * 8,
                  af[0], af[1], af[2], af[3]);
#pragma unroll
          for (int nt = 0; nt < 8; ++nt) mma_bf16(af, bf[nt][0], bf[nt][1], acc[mt][nt]);
        }
      }
    }
    __syncthreads();
  }

  const int d0 = 64 * warp;
  if (a.n_seg == 1) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int hl = 16 * mt + g + 8 * half, h = hg + hl;
        const float m = Ms[hl];
        const float l = Ls[hl] + expf(a.sink[h] - m);
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const int d = d0 + 8 * nt + 2 * t4;
          __nv_bfloat16* o = a.out + row * a.so_t + h * a.so_h + d;
          o[0] = __float2bfloat16_rn(acc[mt][nt][2 * half] / l);
          o[1] = __float2bfloat16_rn(acc[mt][nt][2 * half + 1] / l);
        }
      }
    }
  } else {
    const int64_t base = (row * a.n_seg + seg) * H + hg;
    if (tid < HC) { a.pm[base + tid] = Ms[tid]; a.pl[base + tid] = Ls[tid]; }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int hl = 16 * mt + g + 8 * half;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const int d = d0 + 8 * nt + 2 * t4;
          float* p = a.pacc + (base + hl) * D + d;
          p[0] = acc[mt][nt][2 * half];
          p[1] = acc[mt][nt][2 * half + 1];
        }
      }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Batch-invariant form (op sparse_mla_inv). A row's entries are cut into fixed segments of seg_tiles 32-entry tiles
// (the same boundaries for every call). Each segment runs the online softmax from a fresh state; segments are merged
// in segment order by merge_seg (explicitly rounded FP32 ops, so every caller compiles it alike):
//   M' = max(M, m); a = e^(M - M'); b = e^(m - M'); L = L a + l b; O = O a + o b
// from (M, L, O) = (-1e30, 0, 0), then out = O / (L + e^(sink - M)). Split mode (grid rows x segments x 2) writes each
// segment's (m, l, o) and sparse_mla_inv_combine merges them; fused mode (grid rows x 1 x 2) merges in shared memory
// as it goes. Both apply the same operations in the same order, so a row's result does not depend on the call.
__device__ __forceinline__ void merge_coeffs(float M, float m, float& Mn, float& a, float& b) {
  Mn = fmaxf(M, m);
  a = expf(M - Mn);
  b = expf(m - Mn);
}
__device__ __forceinline__ float merge_val(float X, float a, float x, float b) {
  return __fadd_rn(__fmul_rn(X, a), __fmul_rn(x, b));
}
__device__ __forceinline__ float final_val(float O, float L, float M, float sink) {
  return __fdiv_rn(O, __fadd_rn(L, expf(sink - M)));
}

constexpr size_t SMEM_INV = SMEM32 + size_t(HC) * D * 4 + 2 * HC * 4;  // + running O, M, L (fused mode)

template <bool kFused>
__global__ void __launch_bounds__(NT, kFused ? 1 : 2) sparse_mla_inv_kernel(Args a, int seg_tiles, int n_segs) {
  extern __shared__ __align__(16) uint8_t smem[];
  __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Ks = Qs + HC * QLD;
  float* Ss = reinterpret_cast<float*>(Ks + TN * KLD);
  __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss + HC * SLD);
  float* Ms = reinterpret_cast<float*>(Ps + HC * PLD);
  float* Ls = Ms + HC;
  float* As = Ls + HC;
  int* Vs = reinterpret_cast<int*>(As + HC);
  float* Os = reinterpret_cast<float*>(Vs + TN);  // fused: running O [HC][D]
  float* Mr = Os + HC * D;                         // fused: running M, L
  float* Lr = Mr + HC;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
  const int64_t row = blockIdx.x;
  const int hg = blockIdx.z * HC;
  const int req = a.req[row];
  for (int i = tid; i < HC * D / 8; i += NT) {
    const int h = i / (D / 8), c = i % (D / 8);
    *reinterpret_cast<uint4*>(Qs + h * QLD + 8 * c) =
        *reinterpret_cast<const uint4*>(a.q + row * a.sq_t + (hg + h) * a.sq_h + 8 * c);
  }
  if constexpr (kFused) {
    for (int i = tid; i < HC * D; i += NT) Os[i] = 0.f;
    if (tid < HC) { Mr[tid] = -1.0e30f; Lr[tid] = 0.f; }
  }
  const int win_tiles = (a.n_window + TN - 1) / TN;
  const int total = win_tiles + (a.n_comp + TN - 1) / TN;
  const int s_begin = kFused ? 0 : blockIdx.y, s_end = kFused ? n_segs : blockIdx.y + 1;
  const int d0 = 64 * warp;

  for (int sg = s_begin; sg < s_end; ++sg) {
    if (tid < HC) { Ms[tid] = -1.0e30f; Ls[tid] = 0.f; }
    float acc[2][8][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;
    __syncthreads();
    const int tb = sg * seg_tiles, te = min(total, tb + seg_tiles);
    for (int tile = tb; tile < te; ++tile) {
      const bool window = tile < win_tiles;
      {
        const int e = tid >> 3, c = tid & 7;
        bool ok;
        int64_t off = 0;
        if (window) {
          const int col = tile * TN + e;
          const int slot = col < a.n_window ? a.swa_idx[row * a.s_wi + col] : -1;
          ok = slot >= 0;
          if (ok) off = int64_t(slot / a.swa_e) * a.s_sb + int64_t(slot % a.swa_e) * a.s_se;
        } else {
          const int col = (tile - win_tiles) * TN + e;
          const int j = col < a.n_comp ? a.comp_idx[row * a.s_ci + col] : -1;
          ok = j >= 0;
          if (ok) {
            const int blk = a.bt[int64_t(req) * a.s_bt + j / a.comp_e];
            off = int64_t(blk) * a.s_cb + int64_t(j % a.comp_e) * a.s_ce;
          }
        }
        load_entry_chunk(a, window, off, ok, c, Ks + e * KLD + 64 * c);
        if (c == 0) Vs[e] = ok;
      }
      __syncthreads();
      {
        const int h0 = 16 * (warp & 1), n0 = 8 * (warp >> 1);
        float s4[4] = {0.f, 0.f, 0.f, 0.f};
        const __nv_bfloat16* qa = Qs + (h0 + (lane & 7) + ((lane >> 3) & 1) * 8) * QLD + (lane >> 4) * 8;
        const __nv_bfloat16* ka = Ks + (n0 + (lane & 7)) * KLD + (lane >> 3) * 8;
#pragma unroll 4
        for (int k0 = 0; k0 < D; k0 += 32) {
          uint32_t af0[4], af1[4], b0, b1, b2, b3;
          ldsm_x4(qa + k0, af0[0], af0[1], af0[2], af0[3]);
          ldsm_x4(qa + k0 + 16, af1[0], af1[1], af1[2], af1[3]);
          ldsm_x4(ka + k0, b0, b1, b2, b3);
          mma_bf16(af0, b0, b1, s4);
          mma_bf16(af1, b2, b3, s4);
        }
        const int e0 = n0 + 2 * t4;
        Ss[(h0 + g) * SLD + e0] = Vs[e0] ? s4[0] * a.scale : -INFINITY;
        Ss[(h0 + g) * SLD + e0 + 1] = Vs[e0 + 1] ? s4[1] * a.scale : -INFINITY;
        Ss[(h0 + g + 8) * SLD + e0] = Vs[e0] ? s4[2] * a.scale : -INFINITY;
        Ss[(h0 + g + 8) * SLD + e0 + 1] = Vs[e0 + 1] ? s4[3] * a.scale : -INFINITY;
      }
      __syncthreads();
      if (tid < HC * 4) {
        const int h = tid >> 2, part = tid & 3;
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; ++i) mx = fmaxf(mx, Ss[h * SLD + 8 * part + i]);
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        const float m_old = Ms[h];
        const float m_new = fmaxf(m_old, mx);
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float pr = expf(Ss[h * SLD + 8 * part + i] - m_new);
          sum += pr;
          Ps[h * PLD + 8 * part + i] = __float2bfloat16_rn(pr);
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        __syncwarp();
        if (part == 0) {
          const float alpha = expf(m_old - m_new);
          As[h] = alpha;
          Ls[h] = __fadd_rn(__fmul_rn(Ls[h], alpha), sum);
          Ms[h] = m_new;
        }
      }
      __syncthreads();
      {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          const float a0 = As[16 * mt + g], a1 = As[16 * mt + g + 8];
#pragma unroll
          for (int nt = 0; nt < 8; ++nt) {
            acc[mt][nt][0] *= a0; acc[mt][nt][1] *= a0; acc[mt][nt][2] *= a1; acc[mt][nt][3] *= a1;
          }
        }
#pragma unroll
        for (int k0 = 0; k0 < TN; k0 += 16) {
          uint32_t bf[8][2];
#pragma unroll
          for (int nt = 0; nt < 8; nt += 2) {
            const __nv_bfloat16* kb = Ks + (k0 + (lane & 7) + ((lane >> 3) & 1) * 8) * KLD + d0 + 8 * nt + (lane >> 4) * 8;
            ldsm_x4_t(kb, bf[nt][0], bf[nt][1], bf[nt + 1][0], bf[nt + 1][1]);
          }
#pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            uint32_t af[4];
            ldsm_x4(Ps + (16 * mt + (lane & 7) + ((lane >> 3) & 1) * 8) * PLD + k0 + (lane >> 4) * 8,
                    af[0], af[1], af[2], af[3]);
#pragma unroll
            for (int nt = 0; nt < 8; ++nt) mma_bf16(af, bf[nt][0], bf[nt][1], acc[mt][nt]);
          }
        }
      }
      __syncthreads();
    }
    // segment done: (Ms, Ls, acc) is this segment's (m, l, o)
    if constexpr (kFused) {
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int hl = 16 * mt + g + 8 * half;
          float Mn, ca, cb;
          merge_coeffs(Mr[hl], Ms[hl], Mn, ca, cb);
#pragma unroll
          for (int nt = 0; nt < 8; ++nt) {
            const int d = d0 + 8 * nt + 2 * t4;
            float* o = Os + hl * D + d;
            o[0] = merge_val(o[0], ca, acc[mt][nt][2 * half], cb);
            o[1] = merge_val(o[1], ca, acc[mt][nt][2 * half + 1], cb);
          }
        }
      __syncthreads();  // every thread read Mr / Ms before they change
      if (tid < HC) {
        float Mn, ca, cb;
        merge_coeffs(Mr[tid], Ms[tid], Mn, ca, cb);
        Lr[tid] = merge_val(Lr[tid], ca, Ls[tid], cb);
        Mr[tid] = Mn;
      }
      __syncthreads();
    } else {
      const int64_t base = (row * n_segs + sg) * H + hg;
      if (tid < HC) { a.pm[base + tid] = Ms[tid]; a.pl[base + tid] = Ls[tid]; }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int hl = 16 * mt + g + 8 * half;
#pragma unroll
          for (int nt = 0; nt < 8; ++nt) {
            const int d = d0 + 8 * nt + 2 * t4;
            float* pp = a.pacc + (base + hl) * D + d;
            pp[0] = acc[mt][nt][2 * half];
            pp[1] = acc[mt][nt][2 * half + 1];
          }
        }
    }
  }
  if constexpr (kFused) {
    for (int i = tid; i < HC * D; i += NT) {
      const int hl = i / D, d = i % D, h = hg + hl;
      a.out[row * a.so_t + h * a.so_h + d] = __float2bfloat16_rn(final_val(Os[i], Lr[hl], Mr[hl], a.sink[h]));
    }
  }
}

// 16 heads per CTA (op sparse_mla_inv with heads_per_cta 16): the same segment / merge definition as above with the
// accumulator (32 FP32 per thread) and, in fused mode, the running O (32 more) in registers; 54 KB of shared memory.
constexpr int HC16 = 16;
constexpr size_t SMEM16 = size_t(HC16) * QLD * 2 + size_t(TN) * KLD * 2 + size_t(HC16) * SLD * 4 +
                          size_t(HC16) * PLD * 2 + 5 * HC16 * 4 + TN * 4;

template <bool kFused>
__global__ void __launch_bounds__(NT, 2) sparse_mla_inv16_kernel(Args a, int seg_tiles, int n_segs) {
  extern __shared__ __align__(16) uint8_t smem[];
  __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Ks = Qs + HC16 * QLD;
  float* Ss = reinterpret_cast<float*>(Ks + TN * KLD);
  __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss + HC16 * SLD);
  float* Ms = reinterpret_cast<float*>(Ps + HC16 * PLD);
  float* Ls = Ms + HC16;
  float* As = Ls + HC16;
  float* Mr = As + HC16;
  float* Lr = Mr + HC16;
  int* Vs = reinterpret_cast<int*>(Lr + HC16);

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
  const int64_t row = blockIdx.x;
  const int hg = blockIdx.z * HC16;
  const int req = a.req[row];
  for (int i = tid; i < HC16 * D / 8; i += NT) {
    const int h = i / (D / 8), c = i % (D / 8);
    *reinterpret_cast<uint4*>(Qs + h * QLD + 8 * c) =
        *reinterpret_cast<const uint4*>(a.q + row * a.sq_t + (hg + h) * a.sq_h + 8 * c);
  }
  if (tid < HC16) { Mr[tid] = -1.0e30f; Lr[tid] = 0.f; }
  float O[8][4];
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int k = 0; k < 4; ++k) O[j][k] = 0.f;
  const int win_tiles = (a.n_window + TN - 1) / TN;
  const int total = win_tiles + (a.n_comp + TN - 1) / TN;
  const int s_begin = kFused ? 0 : blockIdx.y, s_end = kFused ? n_segs : blockIdx.y + 1;
  const int d0 = 64 * warp;

  for (int sg = s_begin; sg < s_end; ++sg) {
    if (tid < HC16) { Ms[tid] = -1.0e30f; Ls[tid] = 0.f; }
    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[j][k] = 0.f;
    __syncthreads();
    const int tb = sg * seg_tiles, te = min(total, tb + seg_tiles);
    for (int tile = tb; tile < te; ++tile) {
      const bool window = tile < win_tiles;
      {
        const int e = tid >> 3, c = tid & 7;
        bool ok;
        int64_t off = 0;
        if (window) {
          const int col = tile * TN + e;
          const int slot = col < a.n_window ? a.swa_idx[row * a.s_wi + col] : -1;
          ok = slot >= 0;
          if (ok) off = int64_t(slot / a.swa_e) * a.s_sb + int64_t(slot % a.swa_e) * a.s_se;
        } else {
          const int col = (tile - win_tiles) * TN + e;
          const int j = col < a.n_comp ? a.comp_idx[row * a.s_ci + col] : -1;
          ok = j >= 0;
          if (ok) {
            const int blk = a.bt[int64_t(req) * a.s_bt + j / a.comp_e];
            off = int64_t(blk) * a.s_cb + int64_t(j % a.comp_e) * a.s_ce;
          }
        }
        load_entry_chunk(a, window, off, ok, c, Ks + e * KLD + 64 * c);
        if (c == 0) Vs[e] = ok;
      }
      __syncthreads();
      if (warp < 4) {  // S: warp = 16 heads x 8 entries
        const int n0 = 8 * warp;
        float s4[4] = {0.f, 0.f, 0.f, 0.f};
        const __nv_bfloat16* qa = Qs + ((lane & 7) + ((lane >> 3) & 1) * 8) * QLD + (lane >> 4) * 8;
        const __nv_bfloat16* ka = Ks + (n0 + (lane & 7)) * KLD + (lane >> 3) * 8;
#pragma unroll 4
        for (int k0 = 0; k0 < D; k0 += 32) {
          uint32_t af0[4], af1[4], b0, b1, b2, b3;
          ldsm_x4(qa + k0, af0[0], af0[1], af0[2], af0[3]);
          ldsm_x4(qa + k0 + 16, af1[0], af1[1], af1[2], af1[3]);
          ldsm_x4(ka + k0, b0, b1, b2, b3);
          mma_bf16(af0, b0, b1, s4);
          mma_bf16(af1, b2, b3, s4);
        }
        const int e0 = n0 + 2 * t4;
        Ss[g * SLD + e0] = Vs[e0] ? s4[0] * a.scale : -INFINITY;
        Ss[g * SLD + e0 + 1] = Vs[e0 + 1] ? s4[1] * a.scale : -INFINITY;
        Ss[(g + 8) * SLD + e0] = Vs[e0] ? s4[2] * a.scale : -INFINITY;
        Ss[(g + 8) * SLD + e0 + 1] = Vs[e0 + 1] ? s4[3] * a.scale : -INFINITY;
      }
      __syncthreads();
      if (tid < HC16 * 4) {
        const int h = tid >> 2, part = tid & 3;
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; ++i) mx = fmaxf(mx, Ss[h * SLD + 8 * part + i]);
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        const float m_old = Ms[h];
        const float m_new = fmaxf(m_old, mx);
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float pr = expf(Ss[h * SLD + 8 * part + i] - m_new);
          sum += pr;
          Ps[h * PLD + 8 * part + i] = __float2bfloat16_rn(pr);
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        __syncwarp();
        if (part == 0) {
          const float alpha = expf(m_old - m_new);
          As[h] = alpha;
          Ls[h] = __fadd_rn(__fmul_rn(Ls[h], alpha), sum);
          Ms[h] = m_new;
        }
      }
      __syncthreads();
      {
        const float a0 = As[g], a1 = As[g + 8];
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          acc[nt][0] *= a0; acc[nt][1] *= a0; acc[nt][2] *= a1; acc[nt][3] *= a1;
        }
#pragma unroll
        for (int k0 = 0; k0 < TN; k0 += 16) {
          uint32_t bf[8][2];
#pragma unroll
          for (int nt = 0; nt < 8; nt += 2) {
            const __nv_bfloat16* kb = Ks + (k0 + (lane & 7) + ((lane >> 3) & 1) * 8) * KLD + d0 + 8 * nt + (lane >> 4) * 8;
            ldsm_x4_t(kb, bf[nt][0], bf[nt][1], bf[nt + 1][0], bf[nt + 1][1]);
          }
          uint32_t af[4];
          ldsm_x4(Ps + ((lane & 7) + ((lane >> 3) & 1) * 8) * PLD + k0 + (lane >> 4) * 8, af[0], af[1], af[2], af[3]);
#pragma unroll
          for (int nt = 0; nt < 8; ++nt) mma_bf16(af, bf[nt][0], bf[nt][1], acc[nt]);
        }
      }
      __syncthreads();
    }
    if constexpr (kFused) {
      float Mn0, ca0, cb0, Mn1, ca1, cb1;
      merge_coeffs(Mr[g], Ms[g], Mn0, ca0, cb0);
      merge_coeffs(Mr[g + 8], Ms[g + 8], Mn1, ca1, cb1);
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        O[nt][0] = merge_val(O[nt][0], ca0, acc[nt][0], cb0);
        O[nt][1] = merge_val(O[nt][1], ca0, acc[nt][1], cb0);
        O[nt][2] = merge_val(O[nt][2], ca1, acc[nt][2], cb1);
        O[nt][3] = merge_val(O[nt][3], ca1, acc[nt][3], cb1);
      }
      __syncthreads();
      if (tid < HC16) {
        float Mn, ca, cb;
        merge_coeffs(Mr[tid], Ms[tid], Mn, ca, cb);
        Lr[tid] = merge_val(Lr[tid], ca, Ls[tid], cb);
        Mr[tid] = Mn;
      }
      __syncthreads();
    } else {
      const int64_t base = (row * n_segs + sg) * H + hg;
      if (tid < HC16) { a.pm[base + tid] = Ms[tid]; a.pl[base + tid] = Ls[tid]; }
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int hl = g + 8 * half;
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const int d = d0 + 8 * nt + 2 * t4;
          float* pp = a.pacc + (base + hl) * D + d;
          pp[0] = acc[nt][2 * half];
          pp[1] = acc[nt][2 * half + 1];
        }
      }
    }
  }
  if constexpr (kFused) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int hl = g + 8 * half, h = hg + hl;
      const float Lh = Lr[hl], Mh = Mr[hl], sk = a.sink[h];
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int d = d0 + 8 * nt + 2 * t4;
        __nv_bfloat16* o = a.out + row * a.so_t + h * a.so_h + d;
        o[0] = __float2bfloat16_rn(final_val(O[nt][2 * half], Lh, Mh, sk));
        o[1] = __float2bfloat16_rn(final_val(O[nt][2 * half + 1], Lh, Mh, sk));
      }
    }
  }
}

// 64 heads, 16 warps (op sparse_mla_inv64): the segment / merge definition of sparse_mla_inv with every head of a row
// in one CTA (each tile's records decoded once) and both the segment accumulator and the fused running O in
// registers (64 + 64 FP32 per thread): fused mode for many rows (one CTA per row), split mode + the combine for few.
constexpr int NT16 = 512;
constexpr int kMaxOffTiles = 24;  // offset table capacity per CTA (served layers: 4 window + 16 compressed tiles)

// 32 contiguous values [32 c, 32 c + 32) of one entry into shared memory (BF16), c in 0..15.
__device__ __forceinline__ void load_entry_chunk32(const Args& a, bool window, int64_t off, bool ok, int c,
                                                   __nv_bfloat16* dst) {
  if (!ok) {
#pragma unroll
    for (int i = 0; i < 32; i += 8) *reinterpret_cast<uint4*>(dst + i) = make_uint4(0, 0, 0, 0);
    return;
  }
  const bool packed = window ? a.swa_packed : a.comp_packed;
  const uint8_t* base = (window ? a.swa : a.comp) + off;
  if (!packed) {
    const uint4* src = reinterpret_cast<const uint4*>(base) + c * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) reinterpret_cast<uint4*>(dst)[i] = src[i];
    return;
  }
  // The 32 BF16 values are packed in pairs and each 8-value chunk is written with one 16-byte store as soon as it
  // is formed (4 stores instead of 32 two-byte stores, 7.9-way bank conflicts in the prefill profile; only 4 packed
  // words live, so no extra register pressure); the same values and bytes.
  auto pack = [](__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return uint32_t(__bfloat16_as_ushort(lo)) | (uint32_t(__bfloat16_as_ushort(hi)) << 16);
  };
  if (window) {  // MXFP8: 32 codes from byte 32 c, scale byte 512 + c
    uint32_t codes[8];
#pragma unroll
    for (int i = 0; i < 2; ++i) *reinterpret_cast<uint4*>(codes + 4 * i) = reinterpret_cast<const uint4*>(base + 32 * c)[i];
    const float sc = e8m0_to_f(base[512 + c]);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      uint32_t pk[4];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t x = codes[2 * q + h];
        pk[2 * h] = pack(__float2bfloat16_rn(e4m3_to_f(x & 0xFF) * sc),
                         __float2bfloat16_rn(e4m3_to_f((x >> 8) & 0xFF) * sc));
        pk[2 * h + 1] = pack(__float2bfloat16_rn(e4m3_to_f((x >> 16) & 0xFF) * sc),
                             __float2bfloat16_rn(e4m3_to_f(x >> 24) * sc));
      }
      reinterpret_cast<uint4*>(dst)[q] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    }
  } else {  // NVFP4: 16 code bytes from byte 16 c, scales 256 + 2 c (+1)
    uint32_t codes[4];
    *reinterpret_cast<uint4*>(codes) = reinterpret_cast<const uint4*>(base + 16 * c)[0];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      const float sc = e4m3_to_f(base[256 + 2 * c + (w >> 1)]);
      const uint32_t x = codes[w];
      uint32_t pk[4];
#pragma unroll
      for (int j = 0; j < 8; j += 2)
        pk[j / 2] = pack(__float2bfloat16_rn(e2m1_to_f((x >> (4 * j)) & 0xF) * sc),
                         __float2bfloat16_rn(e2m1_to_f((x >> (4 * j + 4)) & 0xF) * sc));
      reinterpret_cast<uint4*>(dst)[w] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    }
  }
}

template <bool kFused>
__global__ void __launch_bounds__(NT16, 1) sparse_mla_inv64_kernel(Args a, int seg_tiles, int n_segs) {
  extern __shared__ __align__(16) uint8_t smem[];
  __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Ks = Qs + H * QLD;
  float* Ss = reinterpret_cast<float*>(Ks + TN * KLD);
  __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss + H * SLD);
  float* Ms = reinterpret_cast<float*>(Ps + H * PLD);
  float* Ls = Ms + H;
  float* As = Ls + H;
  float* Mr = As + H;
  float* Lr = Mr + H;
  int* Vs = reinterpret_cast<int*>(Lr + H);
  int64_t* Off = reinterpret_cast<int64_t*>(Vs + TN);  // entry byte offsets of this CTA's tiles (-1: invalid)

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
  const int64_t row = blockIdx.x;
  const int req = a.req[row];
  for (int i = tid; i < H * D / 8; i += NT16) {
    const int h = i / (D / 8), c = i % (D / 8);
    *reinterpret_cast<uint4*>(Qs + h * QLD + 8 * c) =
        *reinterpret_cast<const uint4*>(a.q + row * a.sq_t + h * a.sq_h + 8 * c);
  }
  if (tid < H) { Mr[tid] = -1.0e30f; Lr[tid] = 0.f; }
  float O[4][4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) O[i][j][k] = 0.f;
  const int win_tiles = (a.n_window + TN - 1) / TN;
  const int total = win_tiles + (a.n_comp + TN - 1) / TN;
  const int s_begin = kFused ? 0 : blockIdx.y, s_end = kFused ? n_segs : blockIdx.y + 1;
  const int d0 = 32 * warp;
  // The index chain (window slot / compressed position -> block table -> record) of every entry this CTA reads,
  // resolved once and in parallel instead of once per tile on the critical path.
  const int t_first = s_begin * seg_tiles, t_last = min(total, s_end * seg_tiles);
  for (int i = tid; i < (t_last - t_first) * TN; i += NT16) {
    const int tile = t_first + i / TN, e = i % TN;
    int64_t off = -1;
    if (tile < win_tiles) {
      const int col = tile * TN + e;
      const int slot = col < a.n_window ? a.swa_idx[row * a.s_wi + col] : -1;
      if (slot >= 0) off = int64_t(slot / a.swa_e) * a.s_sb + int64_t(slot % a.swa_e) * a.s_se;
    } else {
      const int col = (tile - win_tiles) * TN + e;
      const int j = col < a.n_comp ? a.comp_idx[row * a.s_ci + col] : -1;
      if (j >= 0) {
        const int blk = a.bt[int64_t(req) * a.s_bt + j / a.comp_e];
        off = int64_t(blk) * a.s_cb + int64_t(j % a.comp_e) * a.s_ce;
      }
    }
    Off[i] = off;
  }

  for (int sg = s_begin; sg < s_end; ++sg) {
    if (tid < H) { Ms[tid] = -1.0e30f; Ls[tid] = 0.f; }
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;
    __syncthreads();
    const int tb = sg * seg_tiles, te = min(total, tb + seg_tiles);
    for (int tile = tb; tile < te; ++tile) {
      const bool window = tile < win_tiles;
      {
        const int e = tid >> 4, c = tid & 15;
        const int64_t off = Off[(tile - t_first) * TN + e];
        const bool ok = off >= 0;
        load_entry_chunk32(a, window, ok ? off : 0, ok, c, Ks + e * KLD + 32 * c);
        if (c == 0) Vs[e] = ok;
      }
      __syncthreads();
      {  // S: warp = 16 heads x 8 entries
        const int h0 = 16 * (warp & 3), n0 = 8 * (warp >> 2);
        float s4[4] = {0.f, 0.f, 0.f, 0.f};
        const __nv_bfloat16* qa = Qs + (h0 + (lane & 7) + ((lane >> 3) & 1) * 8) * QLD + (lane >> 4) * 8;
        const __nv_bfloat16* ka = Ks + (n0 + (lane & 7)) * KLD + (lane >> 3) * 8;
#pragma unroll 4
        for (int k0 = 0; k0 < D; k0 += 32) {
          uint32_t af0[4], af1[4], b0, b1, b2, b3;
          ldsm_x4(qa + k0, af0[0], af0[1], af0[2], af0[3]);
          ldsm_x4(qa + k0 + 16, af1[0], af1[1], af1[2], af1[3]);
          ldsm_x4(ka + k0, b0, b1, b2, b3);
          mma_bf16(af0, b0, b1, s4);
          mma_bf16(af1, b2, b3, s4);
        }
        const int e0 = n0 + 2 * t4;
        Ss[(h0 + g) * SLD + e0] = Vs[e0] ? s4[0] * a.scale : -INFINITY;
        Ss[(h0 + g) * SLD + e0 + 1] = Vs[e0 + 1] ? s4[1] * a.scale : -INFINITY;
        Ss[(h0 + g + 8) * SLD + e0] = Vs[e0] ? s4[2] * a.scale : -INFINITY;
        Ss[(h0 + g + 8) * SLD + e0 + 1] = Vs[e0 + 1] ? s4[3] * a.scale : -INFINITY;
      }
      __syncthreads();
      if (tid < H * 4) {
        const int h = tid >> 2, part = tid & 3;
        float mx = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; ++i) mx = fmaxf(mx, Ss[h * SLD + 8 * part + i]);
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        const float m_old = Ms[h];
        const float m_new = fmaxf(m_old, mx);
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float pr = expf(Ss[h * SLD + 8 * part + i] - m_new);
          sum += pr;
          Ps[h * PLD + 8 * part + i] = __float2bfloat16_rn(pr);
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        __syncwarp();
        if (part == 0) {
          const float alpha = expf(m_old - m_new);
          As[h] = alpha;
          Ls[h] = __fadd_rn(__fmul_rn(Ls[h], alpha), sum);
          Ms[h] = m_new;
        }
      }
      __syncthreads();
      {  // PV: warp = all 64 heads x dims [32 warp, +32)
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          const float a0 = As[16 * mt + g], a1 = As[16 * mt + g + 8];
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            acc[mt][nt][0] *= a0; acc[mt][nt][1] *= a0; acc[mt][nt][2] *= a1; acc[mt][nt][3] *= a1;
          }
        }
#pragma unroll
        for (int k0 = 0; k0 < TN; k0 += 16) {
          uint32_t bf[4][2];
#pragma unroll
          for (int nt = 0; nt < 4; nt += 2) {
            const __nv_bfloat16* kb = Ks + (k0 + (lane & 7) + ((lane >> 3) & 1) * 8) * KLD + d0 + 8 * nt + (lane >> 4) * 8;
            ldsm_x4_t(kb, bf[nt][0], bf[nt][1], bf[nt + 1][0], bf[nt + 1][1]);
          }
#pragma unroll
          for (int mt = 0; mt < 4; ++mt) {
            uint32_t af[4];
            ldsm_x4(Ps + (16 * mt + (lane & 7) + ((lane >> 3) & 1) * 8) * PLD + k0 + (lane >> 4) * 8,
                    af[0], af[1], af[2], af[3]);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) mma_bf16(af, bf[nt][0], bf[nt][1], acc[mt][nt]);
          }
        }
      }
      __syncthreads();
    }
    if constexpr (kFused) {
#pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        float Mn0, ca0, cb0, Mn1, ca1, cb1;
        merge_coeffs(Mr[16 * mt + g], Ms[16 * mt + g], Mn0, ca0, cb0);
        merge_coeffs(Mr[16 * mt + g + 8], Ms[16 * mt + g + 8], Mn1, ca1, cb1);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          O[mt][nt][0] = merge_val(O[mt][nt][0], ca0, acc[mt][nt][0], cb0);
          O[mt][nt][1] = merge_val(O[mt][nt][1], ca0, acc[mt][nt][1], cb0);
          O[mt][nt][2] = merge_val(O[mt][nt][2], ca1, acc[mt][nt][2], cb1);
          O[mt][nt][3] = merge_val(O[mt][nt][3], ca1, acc[mt][nt][3], cb1);
        }
      }
      __syncthreads();
      if (tid < H) {
        float Mn, ca, cb;
        merge_coeffs(Mr[tid], Ms[tid], Mn, ca, cb);
        Lr[tid] = merge_val(Lr[tid], ca, Ls[tid], cb);
        Mr[tid] = Mn;
      }
      __syncthreads();
    } else {
      const int64_t base = (row * n_segs + sg) * H;
      if (tid < H) { a.pm[base + tid] = Ms[tid]; a.pl[base + tid] = Ls[tid]; }
#pragma unroll
      for (int mt = 0; mt < 4; ++mt)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int h = 16 * mt + g + 8 * half;
#pragma unroll
          for (int nt = 0; nt < 4; ++nt) {
            const int d = d0 + 8 * nt + 2 * t4;
            float* pp = a.pacc + (base + h) * D + d;
            pp[0] = acc[mt][nt][2 * half];
            pp[1] = acc[mt][nt][2 * half + 1];
          }
        }
    }
  }
  if constexpr (kFused) {
#pragma unroll
    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int h = 16 * mt + g + 8 * half;
        const float Lh = Lr[h], Mh = Mr[h], sk = a.sink[h];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          const int d = d0 + 8 * nt + 2 * t4;
          __nv_bfloat16* o = a.out + row * a.so_t + h * a.so_h + d;
          o[0] = __float2bfloat16_rn(final_val(O[mt][nt][2 * half], Lh, Mh, sk));
          o[1] = __float2bfloat16_rn(final_val(O[mt][nt][2 * half + 1], Lh, Mh, sk));
        }
      }
  }
}

// Merge a row's segment partials in segment order (grid (rows, H), 128 threads x 4 dims).
__global__ void __launch_bounds__(128) sparse_mla_inv_combine(const float* __restrict__ pm, const float* __restrict__ pl,
    const float* __restrict__ pacc, const float* __restrict__ sink, int n_segs, __nv_bfloat16* __restrict__ out,
    int64_t so_t, int64_t so_h) {
  const int64_t row = blockIdx.x;
  const int h = blockIdx.y, d = threadIdx.x * 4;
  float M = -1.0e30f, L = 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
  for (int s = 0; s < n_segs; ++s) {
    const int64_t b = (row * n_segs + s) * H + h;
    float Mn, ca, cb;
    merge_coeffs(M, pm[b], Mn, ca, cb);
    L = merge_val(L, ca, pl[b], cb);
    const float4 v = *reinterpret_cast<const float4*>(pacc + b * D + d);
    o[0] = merge_val(o[0], ca, v.x, cb); o[1] = merge_val(o[1], ca, v.y, cb);
    o[2] = merge_val(o[2], ca, v.z, cb); o[3] = merge_val(o[3], ca, v.w, cb);
    M = Mn;
  }
  __nv_bfloat16* dst = out + row * so_t + h * so_h + d;
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[i] = __float2bfloat16_rn(final_val(o[i], L, M, sink[h]));
}

// grid (rows, H): one head per CTA, 128 threads x 4 dims of D = 512. The segments are merged in segment order with
// the same FP32 operations as before (running max, then L and the weighted sums accumulated in segment order); the
// loads of a row's segments are independent, so the loop is unrolled to keep several in flight (the 4-heads-per-CTA
// form was latency-bound: few CTAs, one dependent load chain per thread).
__global__ void __launch_bounds__(128) sparse_mla_combine(const float* __restrict__ pm, const float* __restrict__ pl,
                                   const float* __restrict__ pacc, const float* __restrict__ sink, int n_seg,
                                   __nv_bfloat16* __restrict__ out, int64_t so_t, int64_t so_h) {
  const int64_t row = blockIdx.x;
  const int h = blockIdx.y, d = threadIdx.x * 4;
  float M = -1.0e30f;
  for (int s = 0; s < n_seg; ++s) M = fmaxf(M, pm[(row * n_seg + s) * H + h]);
  float L = expf(sink[h] - M), o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
  for (int s = 0; s < n_seg; ++s) {
    const int64_t b = (row * n_seg + s) * H + h;
    const float w = expf(pm[b] - M);
    L += pl[b] * w;
    const float4 v = *reinterpret_cast<const float4*>(pacc + b * D + d);
    o[0] += v.x * w; o[1] += v.y * w; o[2] += v.z * w; o[3] += v.w * w;
  }
  __nv_bfloat16* dst = out + row * so_t + h * so_h + d;
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[i] = __float2bfloat16_rn(o[i] / L);
}

}  // namespace

namespace dsv41_mla {
void sparse_mla(at::Tensor q, at::Tensor swa, at::Tensor swa_idx, at::Tensor comp, at::Tensor bt,
                at::Tensor comp_idx, at::Tensor req, at::Tensor sink, double scale, bool swa_packed,
                bool comp_packed, int64_t tiles_per_seg, at::Tensor out, at::Tensor pm, at::Tensor pl,
                at::Tensor pacc, int64_t heads_per_cta) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.size(1) == H && q.size(2) == D && q.stride(2) == 1,
              "q bf16 [T, 64, 512]");
  TORCH_CHECK(swa_idx.scalar_type() == at::kInt && swa_idx.stride(1) == 1 && comp_idx.scalar_type() == at::kInt &&
                  comp_idx.stride(1) == 1 && bt.scalar_type() == at::kInt && req.scalar_type() == at::kInt &&
                  sink.scalar_type() == at::kFloat, "int32 indices / block table / req, fp32 sink");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.stride(2) == 1, "out bf16 [T, 64, 512]");
  TORCH_CHECK(swa.stride(2) == 1 && comp.stride(2) == 1, "cache rows contiguous");
  const c10::cuda::CUDAGuard guard(q.device());
  Args a;
  a.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); a.sq_t = q.stride(0); a.sq_h = q.stride(1);
  const int64_t esz_s = swa.element_size(), esz_c = comp.element_size();
  a.swa = reinterpret_cast<const uint8_t*>(swa.data_ptr()); a.s_sb = swa.stride(0) * esz_s;
  a.s_se = swa.stride(1) * esz_s; a.swa_e = swa.size(1); a.swa_packed = swa_packed;
  a.swa_idx = swa_idx.data_ptr<int32_t>(); a.s_wi = swa_idx.stride(0); a.n_window = swa_idx.size(1);
  a.comp = reinterpret_cast<const uint8_t*>(comp.data_ptr()); a.s_cb = comp.stride(0) * esz_c;
  a.s_ce = comp.stride(1) * esz_c; a.comp_e = comp.size(1); a.comp_packed = comp_packed;
  a.bt = bt.data_ptr<int32_t>(); a.s_bt = bt.stride(0);
  a.comp_idx = comp_idx.data_ptr<int32_t>(); a.s_ci = comp_idx.stride(0); a.n_comp = comp_idx.size(1);
  a.req = req.data_ptr<int32_t>(); a.sink = sink.data_ptr<float>(); a.scale = float(scale);
  const int total = (a.n_window + TN - 1) / TN + (a.n_comp + TN - 1) / TN;
  a.tiles_per_seg = int(tiles_per_seg); a.n_seg = (total + a.tiles_per_seg - 1) / a.tiles_per_seg;
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr()); a.so_t = out.stride(0); a.so_h = out.stride(1);
  a.pm = pm.data_ptr<float>(); a.pl = pl.data_ptr<float>(); a.pacc = pacc.data_ptr<float>();
  auto st = c10::cuda::getCurrentCUDAStream();
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(sparse_mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(sparse_mla_kernel32, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM32)));
    attr = true;
  }
  TORCH_CHECK(heads_per_cta == 64 || heads_per_cta == 32, "heads_per_cta 64 or 32");
  if (heads_per_cta == 32)
    sparse_mla_kernel32<<<dim3(q.size(0), a.n_seg, H / HC), NT, SMEM32, st>>>(a);
  else
    sparse_mla_kernel<<<dim3(q.size(0), a.n_seg), NT, SMEM, st>>>(a);
  if (a.n_seg > 1)
    sparse_mla_combine<<<dim3(q.size(0), H), 128, 0, st>>>(a.pm, a.pl, a.pacc, a.sink, a.n_seg, a.out, a.so_t, a.so_h);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_mla_inv(at::Tensor q, at::Tensor swa, at::Tensor swa_idx, at::Tensor comp, at::Tensor bt,
                    at::Tensor comp_idx, at::Tensor req, at::Tensor sink, double scale, bool swa_packed,
                    bool comp_packed, int64_t seg_tiles, bool fused, at::Tensor out, at::Tensor pm, at::Tensor pl,
                    at::Tensor pacc, int64_t heads_per_cta) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.size(1) == H && q.size(2) == D && q.stride(2) == 1,
              "q bf16 [T, 64, 512]");
  TORCH_CHECK(seg_tiles >= 1, "seg_tiles >= 1");
  const c10::cuda::CUDAGuard guard(q.device());
  Args a;
  a.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()); a.sq_t = q.stride(0); a.sq_h = q.stride(1);
  const int64_t esz_s = swa.element_size(), esz_c = comp.element_size();
  a.swa = reinterpret_cast<const uint8_t*>(swa.data_ptr()); a.s_sb = swa.stride(0) * esz_s;
  a.s_se = swa.stride(1) * esz_s; a.swa_e = swa.size(1); a.swa_packed = swa_packed;
  a.swa_idx = swa_idx.data_ptr<int32_t>(); a.s_wi = swa_idx.stride(0); a.n_window = swa_idx.size(1);
  a.comp = reinterpret_cast<const uint8_t*>(comp.data_ptr()); a.s_cb = comp.stride(0) * esz_c;
  a.s_ce = comp.stride(1) * esz_c; a.comp_e = comp.size(1); a.comp_packed = comp_packed;
  a.bt = bt.data_ptr<int32_t>(); a.s_bt = bt.stride(0);
  a.comp_idx = comp_idx.data_ptr<int32_t>(); a.s_ci = comp_idx.stride(0); a.n_comp = comp_idx.size(1);
  a.req = req.data_ptr<int32_t>(); a.sink = sink.data_ptr<float>(); a.scale = float(scale);
  const int total = (a.n_window + TN - 1) / TN + (a.n_comp + TN - 1) / TN;
  const int n_segs = (total + int(seg_tiles) - 1) / int(seg_tiles);
  a.tiles_per_seg = int(seg_tiles); a.n_seg = n_segs;
  a.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr()); a.so_t = out.stride(0); a.so_h = out.stride(1);
  a.pm = pm.data_ptr<float>(); a.pl = pl.data_ptr<float>(); a.pacc = pacc.data_ptr<float>();
  auto st = c10::cuda::getCurrentCUDAStream();
  static bool attr = false;
  if (!attr) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(sparse_mla_inv_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM32)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(sparse_mla_inv_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM_INV)));
    attr = true;
  }
  static bool attr16 = false;
  if (!attr16) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(sparse_mla_inv16_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM16)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(sparse_mla_inv16_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM16)));
    attr16 = true;
  }
  TORCH_CHECK(heads_per_cta == 64 || heads_per_cta == 32 || heads_per_cta == 16, "heads_per_cta 64, 32 or 16");
  if (!fused) TORCH_CHECK(pacc.numel() >= q.size(0) * n_segs * H * D, "pacc too small");
  constexpr size_t SMEM64 = size_t(H) * QLD * 2 + size_t(TN) * KLD * 2 + size_t(H) * SLD * 4 + size_t(H) * PLD * 2 +
                            5 * H * 4 + TN * 4 + size_t(kMaxOffTiles) * TN * 8;
  static bool attr64 = false;
  if (!attr64) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(sparse_mla_inv64_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM64)));
    C10_CUDA_CHECK(cudaFuncSetAttribute(sparse_mla_inv64_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(SMEM64)));
    attr64 = true;
  }
  if (heads_per_cta == 64) {
    TORCH_CHECK((fused ? total : int(seg_tiles)) <= kMaxOffTiles, "sparse_mla_inv64: tiles per CTA exceed the offset table");
    if (fused)
      sparse_mla_inv64_kernel<true><<<dim3(q.size(0), 1, 1), NT16, SMEM64, st>>>(a, int(seg_tiles), n_segs);
    else
      sparse_mla_inv64_kernel<false><<<dim3(q.size(0), n_segs, 1), NT16, SMEM64, st>>>(a, int(seg_tiles), n_segs);
  } else if (heads_per_cta == 16) {
    if (fused)
      sparse_mla_inv16_kernel<true><<<dim3(q.size(0), 1, H / HC16), NT, SMEM16, st>>>(a, int(seg_tiles), n_segs);
    else
      sparse_mla_inv16_kernel<false><<<dim3(q.size(0), n_segs, H / HC16), NT, SMEM16, st>>>(a, int(seg_tiles), n_segs);
  } else if (fused) {
    sparse_mla_inv_kernel<true><<<dim3(q.size(0), 1, H / HC), NT, SMEM_INV, st>>>(a, int(seg_tiles), n_segs);
  } else {
    sparse_mla_inv_kernel<false><<<dim3(q.size(0), n_segs, H / HC), NT, SMEM32, st>>>(a, int(seg_tiles), n_segs);
  }
  if (!fused)
    sparse_mla_inv_combine<<<dim3(q.size(0), H), 128, 0, st>>>(a.pm, a.pl, a.pacc, a.sink, n_segs, a.out, a.so_t, a.so_h);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace dsv41_mla

TORCH_LIBRARY(dsv41_mla, m) {
  m.def("sparse_mla(Tensor q, Tensor swa, Tensor swa_idx, Tensor comp, Tensor bt, Tensor comp_idx, Tensor req, "
        "Tensor sink, float scale, bool swa_packed, bool comp_packed, int tiles_per_seg, Tensor(a!) out, "
        "Tensor(b!) pm, Tensor(c!) pl, Tensor(d!) pacc, int heads_per_cta=64) -> ()");
  m.def("sparse_mla_inv(Tensor q, Tensor swa, Tensor swa_idx, Tensor comp, Tensor bt, Tensor comp_idx, Tensor req, "
        "Tensor sink, float scale, bool swa_packed, bool comp_packed, int seg_tiles, bool fused, Tensor(a!) out, "
        "Tensor(b!) pm, Tensor(c!) pl, Tensor(d!) pacc, int heads_per_cta=32) -> ()");
}
TORCH_LIBRARY_IMPL(dsv41_mla, CUDA, m) {
  m.impl("sparse_mla", &dsv41_mla::sparse_mla);
  m.impl("sparse_mla_inv", &dsv41_mla::sparse_mla_inv);
}
