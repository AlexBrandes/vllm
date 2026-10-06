#pragma once

/*
 * EXL3 codec adapted from turboderp-org/exllamav3, revision
 * d3739fd393337b1ff4d6c2a342b12f0c87a9592f:
 * quant/{pack.cu,frac.cu,codebook.cuh,exl3_dq.cuh,hadamard_inner.cuh}
 * and modules/quant/exl3_lib/quantize.py.
 * Modifications: scalar generic ring decoder, inverse lane permutation,
 * BF16 operator integration, and FP32 normalized H128 arithmetic.
 *
 * MIT License
 *
 * Copyright (c) 2025 Turboderp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace exl3 {

constexpr float kH128Scale = 0.08838834764831845f;

struct Matrix {
  const uint16_t* trellis;
  const half* suh;
  const half* svh;
  int64_t padded_in;
  int64_t padded_out;
  int bits_x2;
  int codebook;
};

// The codebook is stored/procedurally generated in FP16 by the format. Only
// its decoded value is widened; activation transforms and dots remain FP32.
__device__ __forceinline__ float codebook_value(uint32_t state, int codebook) {
  if (codebook == 1) {
    uint32_t product = state * 0xCBAC1FEDu;
    product = (product & 0x8fff8fffu) ^ 0x3b603b60u;
    return __half2float(__hadd(__ushort_as_half(product & 0xffffu),
                              __ushort_as_half(product >> 16)));
  }
  const uint32_t product = state * 0x83DCD12Du;
  const uint32_t sum = (product & 255u) + ((product >> 8) & 255u) +
                       ((product >> 16) & 255u) + (product >> 24);
  return __half2float(__hfma(__ushort_as_half(0x6400u + sum),
                            __ushort_as_half(0x1eee),
                            __ushort_as_half(0xc931)));
}

__device__ __forceinline__ float core_value(const Matrix& matrix,
                                            int64_t row, int64_t col) {
  const int r = row & 15;
  const int c = col & 15;
  // Invert tensor_core_perm: lane owns eight interleaved row/column values.
  const int lane = ((c & 7) * 4) + ((r & 7) >> 1);
  const int index = lane * 8 + (r & 1) + ((r >> 3) * 2) + ((c >> 3) * 4);
  const int whole = matrix.bits_x2 / 2;
  const int end = (index + 1) * whole +
                  ((matrix.bits_x2 & 1) ? (index + 1) / 2 : 0);
  const int ring_bits = 128 * matrix.bits_x2;
  const int start = (end - 16 + ring_bits) % ring_bits;
  const int words = ring_bits / 32;
  const int64_t tile = (row / 16) * (matrix.padded_out / 16) + col / 16;
  const uint16_t* packed = matrix.trellis + tile * (8 * matrix.bits_x2);
  // pack_trellis swaps 16-bit pairs, leaving MSB-first uint32 ring words.
  // Build words explicitly so an I16 view need not have uint32 alignment.
  const int lo = start / 32;
  const int hi = (lo + 1) % words;
  const uint32_t a = uint32_t(packed[2 * lo]) |
                     (uint32_t(packed[2 * lo + 1]) << 16);
  const uint32_t b = uint32_t(packed[2 * hi]) |
                     (uint32_t(packed[2 * hi + 1]) << 16);
  const uint64_t window = (uint64_t(a) << 32) | b;
  const uint32_t state = (window >> (48 - (start & 31))) & 0xffffu;
  return codebook_value(state, matrix.codebook);
}

// One 128-thread block. Standard symmetric Sylvester ordering, not sequency
// ordering. Every caller participates even for masked routes/padded channels.
__device__ __forceinline__ float hadamard128(float value, float* shared) {
  const int lane = threadIdx.x;
  shared[lane] = value;
  __syncthreads();
  #pragma unroll
  for (int distance = 1; distance < 128; distance <<= 1) {
    const float other = shared[lane ^ distance];
    const float next = (lane & distance) ? __fsub_rn(other, value)
                                         : __fadd_rn(value, other);
    __syncthreads();
    shared[lane] = next;
    value = next;
    __syncthreads();
  }
  return __fmul_rn(value, kH128Scale);
}

}  // namespace exl3
