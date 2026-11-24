#pragma once
// Q4N codec (CMP 170HX native-quant campaign): V=4 bitshift trellis, 16-bit state, KV = 2 * bits_x2 bits per state,
// zero-prefix start (bits before a tile's first bit are zero), nibble-sum codebook over the LCG product
// state * 0x83DCD12D: value_c = hi(byte_c) + lo(byte_c) - 15, c = 0..3, scaled by kQ4NAlpha = 1 / sqrt(2 * 255 / 12).
// Encoder lane L (g = L / 4, t = L % 4) owns states 2L and 2L + 1; state j covers W^T fragment row g + 8j and k
// columns 2t + (c & 1) + 8 (c >> 1): exactly the m16n8k16 A fragment, a[j] = (c0, c1), a[2 + j] = (c2, c3).
#include <cstdint>
#include <cuda_fp16.h>

namespace exl3 {

constexpr int kQ4NFormatOffset = 64;  // bits_x2 key = bits_x2 + 64 marks a Q4N tensor
constexpr float kQ4NAlpha = 0.153392998f;  // float32(1 / 6.519202405202649) = 0x3e1d130e (was 0.15339208f, 62 ulps low)

// Codebook 3 (Q4N): 16-bit state. Codebook 4 (Q4N-X): the longest state whose two lane states fit one 32-bit
// window, L = 32 - KV (22 at 2.5 bpw, 20 at 3.0 bpw). Everything else is shared.
template <int BitsX2, int Codebook = 3>
struct Q4NRing {
  static constexpr int kKV = 2 * BitsX2;
  static constexpr int kL = Codebook == 4 ? 32 - kKV : 16;
  static constexpr int kWords = 4 * BitsX2;  // 64 * kKV / 32
  // First bit of lane L's window (states 2L, 2L + 1): (2L + 1) KV - L; negative only for lane 0.
  __device__ static __forceinline__ int start(int lane) { return (2 * lane + 1) * kKV - kL; }
};

// hi:lo = the two ring words around the window start (hi = 0 before the tile); shift = start mod 32.
template <int BitsX2, int Codebook = 3>
__device__ __forceinline__ void q4n_fragment(uint32_t hi, uint32_t lo, int shift, uint32_t (&a)[4]) {
  using R = Q4NRing<BitsX2, Codebook>;
  const uint32_t u = __funnelshift_l(lo, hi, shift);
  const uint32_t s0 = u >> (32 - R::kL);
  const uint32_t s1 = (u >> (32 - R::kL - R::kKV)) & ((1u << R::kL) - 1u);
  const uint32_t off = 0xe40fe40fu;  // FP16 -1039 (= -(1024 + 15)) in both halves
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const uint32_t p = (j == 0 ? s0 : s1) * 0x83DCD12Du;
    const uint32_t w = ((p >> 4) & 0x0f0f0f0fu) + (p & 0x0f0f0f0fu);
    uint32_t h01 = __byte_perm(w, 0x6464u, 0x5140);
    uint32_t h23 = __byte_perm(w, 0x6464u, 0x5342);
    const __half2 r01 = __hadd2(*reinterpret_cast<const __half2*>(&h01), *reinterpret_cast<const __half2*>(&off));
    const __half2 r23 = __hadd2(*reinterpret_cast<const __half2*>(&h23), *reinterpret_cast<const __half2*>(&off));
    a[j] = *reinterpret_cast<const uint32_t*>(&r01);
    a[2 + j] = *reinterpret_cast<const uint32_t*>(&r23);
  }
}

}  // namespace exl3
