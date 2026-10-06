#include "ops.h"
#include "common.h"
#include "codec.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace exl3 {
namespace {

__global__ void linear_input(const __nv_bfloat16* x, Matrix matrix,
                             int64_t rows, int64_t logical_in, float* transformed) {
  __shared__ float shared[128];
  const int64_t blocks = matrix.padded_in / 128;
  for (int64_t task = blockIdx.x; task < rows * blocks; task += gridDim.x) {
    const int64_t row = task / blocks;
    const int64_t col = (task % blocks) * 128 + threadIdx.x;
    const float value = col < logical_in
        ? __fmul_rn(__bfloat162float(x[row * logical_in + col]),
                     __half2float(matrix.suh[col])) : 0.0f;
    transformed[row * matrix.padded_in + col] = hadamard128(value, shared);
  }
}

__global__ void linear_core(const float* input, Matrix matrix, int64_t rows,
                            float* result) {
  __shared__ float shared_input[128];
  const int64_t blocks = matrix.padded_out / 128;
  for (int64_t task = blockIdx.x; task < rows * blocks; task += gridDim.x) {
    const int64_t row = task / blocks;
    const int64_t col = (task % blocks) * 128 + threadIdx.x;
    float sum = 0.0f;
    for (int64_t base = 0; base < matrix.padded_in; base += 128) {
      shared_input[threadIdx.x] = input[row * matrix.padded_in + base + threadIdx.x];
      __syncthreads();
      for (int k = 0; k < 128; ++k) {
        sum = __fmaf_rn(shared_input[k], core_value(matrix, base + k, col), sum);
      }
      __syncthreads();
    }
    result[row * matrix.padded_out + col] = sum;
  }
}

__global__ void linear_output(const float* result, Matrix matrix, int64_t rows,
                              int64_t logical_out, __nv_bfloat16* out) {
  __shared__ float shared[128];
  const int64_t blocks = matrix.padded_out / 128;
  for (int64_t task = blockIdx.x; task < rows * blocks; task += gridDim.x) {
    const int64_t row = task / blocks;
    const int64_t col = (task % blocks) * 128 + threadIdx.x;
    const float value = hadamard128(result[row * matrix.padded_out + col], shared);
    if (col < logical_out) {
      out[row * logical_out + col] = __float2bfloat16_rn(
          __fmul_rn(value, __half2float(matrix.svh[col])));
    }
  }
}

}  // namespace

int64_t linear_workspace_size(int64_t max_rows, int64_t padded_in,
                              int64_t padded_out) {
  TORCH_CHECK(max_rows >= 0, "EXL3 max_rows must be nonnegative");
  check_padded(padded_in);
  check_padded(padded_out);
  return workspace_elements(checked_mul(max_rows, checked_add(padded_in, padded_out)));
}

void linear_out(const torch::Tensor& x, const torch::Tensor& trellis,
                const torch::Tensor& suh, const torch::Tensor& svh,
                int64_t bits_x2, int64_t codebook, int64_t logical_in,
                int64_t logical_out, torch::Tensor& out,
                torch::Tensor& workspace) {
  TORCH_CHECK(x.is_cuda(), "EXL3 x must be CUDA");
  check_tensor(x, x, at::kBFloat16, "x");
  check_tensor(trellis, x, at::kShort, "trellis");
  check_tensor(suh, x, at::kHalf, "suh");
  check_tensor(svh, x, at::kHalf, "svh");
  check_tensor(out, x, at::kBFloat16, "out");
  check_tensor(workspace, x, at::kFloat, "workspace");
  TORCH_CHECK(valid_format(bits_x2, codebook), "Unsupported EXL3 rate/codebook combination");
  TORCH_CHECK(x.dim() == 2 && trellis.dim() == 3 && suh.dim() == 1 &&
                  svh.dim() == 1 && out.dim() == 2,
              "EXL3 linear requires rank-2 x/out, rank-3 trellis and rank-1 scales");
  TORCH_CHECK(trellis.size(2) == 8 * bits_x2, "EXL3 packed word stride mismatch");
  const int64_t padded_in = checked_mul(trellis.size(0), 16);
  const int64_t padded_out = checked_mul(trellis.size(1), 16);
  check_padded(padded_in);
  check_padded(padded_out);
  TORCH_CHECK(logical_in > 0 && logical_in <= padded_in &&
                  logical_out > 0 && logical_out <= padded_out,
              "EXL3 logical dimensions must lie within the packed geometry");
  TORCH_CHECK(x.size(1) == logical_in && out.size(0) == x.size(0) &&
                  out.size(1) == logical_out && suh.numel() == padded_in &&
                  svh.numel() == padded_out,
              "EXL3 linear input/output/scale geometry mismatch");
  const int64_t rows = x.size(0);
  TORCH_CHECK(workspace.numel() >= linear_workspace_size(rows, padded_in, padded_out),
              "EXL3 linear workspace is too small");
  check_mutations(out, workspace, {&x, &trellis, &suh, &svh});
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const Matrix matrix{reinterpret_cast<const uint16_t*>(trellis.data_ptr<int16_t>()),
                      reinterpret_cast<const half*>(suh.data_ptr<at::Half>()),
                      reinterpret_cast<const half*>(svh.data_ptr<at::Half>()),
                      padded_in, padded_out, static_cast<int>(bits_x2),
                      static_cast<int>(codebook)};
  float* transformed = workspace.data_ptr<float>();
  float* result = transformed + rows * padded_in;
  linear_input<<<launch_blocks(rows * (padded_in / 128)), 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
      matrix, rows, logical_in, transformed);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  linear_core<<<launch_blocks(rows * (padded_out / 128)), 128, 0, stream>>>(
      transformed, matrix, rows, result);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  linear_output<<<launch_blocks(rows * (padded_out / 128)), 128, 0, stream>>>(
      result, matrix, rows, logical_out,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace exl3
