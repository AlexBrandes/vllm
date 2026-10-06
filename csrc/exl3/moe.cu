#include "ops.h"
#include "common.h"
#include "codec.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace exl3 {
namespace {

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
__device__ __forceinline__ Matrix expert_matrix(const Routing& routing,
                                               int64_t expert, int projection) {
  const int64_t* ptrs = routing.pointers + expert * 9 + projection * 3;
  const int64_t* meta = routing.metadata + expert * 12 + projection * 4;
  return {reinterpret_cast<const uint16_t*>(ptrs[0]),
          reinterpret_cast<const half*>(ptrs[1]),
          reinterpret_cast<const half*>(ptrs[2]), meta[0], meta[1],
          static_cast<int>(meta[2]), static_cast<int>(meta[3])};
}

__global__ void expert_input(const __nv_bfloat16* x, Routing routing,
                             int64_t assignments, int64_t hidden,
                             int64_t hpad, int projection, float* input) {
  __shared__ float shared[128];
  const int64_t blocks = hpad / 128;
  for (int64_t task = blockIdx.x; task < assignments * blocks; task += gridDim.x) {
    const int64_t assignment = task / blocks;
    const int64_t col = task % blocks * 128 + threadIdx.x;
    const int64_t expert = local_expert(routing, assignment);
    float value = 0.0f;
    if (expert >= 0 && col < hidden) {
      const Matrix matrix = expert_matrix(routing, expert, projection);
      value = __fmul_rn(__bfloat162float(x[(assignment / routing.topk) * hidden + col]),
                         __half2float(matrix.suh[col]));
    }
    input[assignment * hpad + col] = hadamard128(value, shared);
  }
}

__global__ void expert_core(const float* input, Routing routing,
                            int64_t assignments, int64_t kpad, int64_t npad,
                            int projection, float* result) {
  __shared__ float shared_input[128];
  const int64_t blocks = npad / 128;
  for (int64_t task = blockIdx.x; task < assignments * blocks; task += gridDim.x) {
    const int64_t assignment = task / blocks;
    const int64_t col = task % blocks * 128 + threadIdx.x;
    const int64_t expert = local_expert(routing, assignment);
    float sum = 0.0f;
    if (expert >= 0) {
      const Matrix matrix = expert_matrix(routing, expert, projection);
      // H128 is block-local: additional entirely padded input/output blocks
      // are zero or cropped. Use actual metadata strides to address the core.
      for (int64_t base = 0; base < kpad; base += 128) {
        shared_input[threadIdx.x] = input[assignment * kpad + base + threadIdx.x];
        __syncthreads();
        for (int k = 0; k < 128; ++k) {
          sum = __fmaf_rn(shared_input[k], core_value(matrix, base + k, col), sum);
        }
        __syncthreads();
      }
    }
    result[assignment * npad + col] = sum;
  }
}

__global__ void expert_output(Routing routing, int64_t assignments,
                              int64_t npad, int projection, float* result) {
  __shared__ float shared[128];
  const int64_t blocks = npad / 128;
  for (int64_t task = blockIdx.x; task < assignments * blocks; task += gridDim.x) {
    const int64_t assignment = task / blocks;
    const int64_t col = task % blocks * 128 + threadIdx.x;
    const int64_t expert = local_expert(routing, assignment);
    float value = hadamard128(result[assignment * npad + col], shared);
    if (expert >= 0) {
      const Matrix matrix = expert_matrix(routing, expert, projection);
      value = __fmul_rn(value, __half2float(matrix.svh[col]));
    } else {
      value = 0.0f;
    }
    result[assignment * npad + col] = value;
  }
}

__global__ void activate_down_input(Routing routing, int64_t assignments,
                                    int64_t intermediate, int64_t ipad,
                                    float* gate, const float* up) {
  __shared__ float shared[128];
  const int64_t blocks = ipad / 128;
  for (int64_t task = blockIdx.x; task < assignments * blocks; task += gridDim.x) {
    const int64_t assignment = task / blocks;
    const int64_t col = task % blocks * 128 + threadIdx.x;
    const int64_t expert = local_expert(routing, assignment);
    float value = 0.0f;
    if (expert >= 0 && col < intermediate) {
      const int64_t index = assignment * ipad + col;
      const float g = gate[index];
      const float silu = g / __fadd_rn(1.0f, expf(-g));
      const Matrix down = expert_matrix(routing, expert, 2);
      value = __fmul_rn(__fmul_rn(silu, up[index]), __half2float(down.suh[col]));
    }
    gate[assignment * ipad + col] = hadamard128(value, shared);
  }
}

__global__ void combine_routes(const float* down, const float* weights,
                               Routing routing, int64_t rows, int64_t hidden,
                               int64_t hpad, __nv_bfloat16* out) {
  const int64_t elements = rows * hidden;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = index / hidden;
    const int64_t col = index % hidden;
    float sum = 0.0f;
    for (int64_t route = 0; route < routing.topk; ++route) {
      const int64_t assignment = row * routing.topk + route;
      if (local_expert(routing, assignment) >= 0) {
        // Exactly one routing-weight application, after the complete down op.
        sum = __fmaf_rn(down[assignment * hpad + col], weights[assignment], sum);
      }
    }
    out[index] = __float2bfloat16_rn(sum);
  }
}

}  // namespace

// Setup-only CPU validation. The loader validates these construction tables
// before copying them to CUDA; pointed tensors and immutable table ownership
// remain with the caller. This function never initializes or queries CUDA.
void validate_moe_tables(const torch::Tensor& packed_ptrs,
                         const torch::Tensor& packed_meta,
                         int64_t hidden_size, int64_t intermediate_size) {
  TORCH_CHECK(packed_ptrs.device().is_cpu() && packed_meta.device().is_cpu(),
              "EXL3 setup pointer/metadata tables must be CPU tensors");
  TORCH_CHECK(packed_ptrs.scalar_type() == at::kLong &&
                  packed_meta.scalar_type() == at::kLong &&
                  packed_ptrs.is_contiguous() && packed_meta.is_contiguous(),
              "EXL3 setup tables must be contiguous int64 tensors");
  TORCH_CHECK(packed_ptrs.dim() == 3 && packed_meta.dim() == 3 &&
                  packed_ptrs.size(1) == 3 && packed_ptrs.size(2) == 3 &&
                  packed_meta.size(0) == packed_ptrs.size(0) &&
                  packed_meta.size(1) == 3 && packed_meta.size(2) == 4,
              "EXL3 setup pointer/metadata geometry mismatch");
  const int64_t hpad = padded_size(hidden_size);
  const int64_t ipad = padded_size(intermediate_size);
  const int64_t* pointers = packed_ptrs.data_ptr<int64_t>();
  const int64_t* metadata = packed_meta.data_ptr<int64_t>();
  for (int64_t expert = 0; expert < packed_ptrs.size(0); ++expert) {
    for (int projection = 0; projection < 3; ++projection) {
      const int64_t* ptrs = pointers + expert * 9 + projection * 3;
      const int64_t* meta = metadata + expert * 12 + projection * 4;
      check_padded(meta[0]);
      check_padded(meta[1]);
      TORCH_CHECK(meta[0] >= (projection == 2 ? ipad : hpad) &&
                      meta[1] >= (projection == 2 ? hpad : ipad),
                  "EXL3 expert packed geometry is smaller than logical geometry");
      TORCH_CHECK(valid_format(meta[2], meta[3]),
                  "Unsupported EXL3 expert rate/codebook combination");
      // Checked multiplication after positive-dimension validation: no
      // division by untrusted metadata, including on error paths.
      checked_mul(checked_mul(meta[0] / 16, meta[1] / 16),
                  checked_mul(16, meta[2]));
      TORCH_CHECK(ptrs[0] > 0 && ptrs[1] > 0 && ptrs[2] > 0 &&
                      ptrs[0] % 2 == 0 && ptrs[1] % 2 == 0 && ptrs[2] % 2 == 0,
                  "EXL3 expert pointers must be nonnull and 2-byte aligned");
    }
  }
}

int64_t moe_workspace_size(int64_t max_rows, int64_t hidden_size,
                           int64_t intermediate_size, int64_t topk,
                           int64_t local_experts) {
  TORCH_CHECK(max_rows >= 0 && topk >= 0 && local_experts >= 0,
              "EXL3 rows/topk/local_experts must be nonnegative");
  const int64_t hpad = padded_size(hidden_size);
  const int64_t ipad = padded_size(intermediate_size);
  if (local_experts == 0) return 0;
  return workspace_elements(checked_mul(checked_mul(max_rows, topk),
                                         checked_add(hpad, checked_mul(2, ipad))));
}

void moe_out(const torch::Tensor& x, const torch::Tensor& topk_ids,
             const torch::Tensor& topk_weights,
             const torch::Tensor& packed_ptrs,
             const torch::Tensor& packed_meta,
             const torch::Tensor& expert_map, int64_t intermediate_size,
             torch::Tensor& out, torch::Tensor& workspace) {
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
  const int64_t assignments = checked_mul(rows, topk);
  const int64_t required = moe_workspace_size(rows, hidden, intermediate_size, topk, experts);
  TORCH_CHECK(workspace.numel() >= required, "EXL3 MoE workspace is too small");
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
  if (experts > 0 && assignments > 0) {
    float* gate = input + assignments * hpad;
    float* up = gate + assignments * ipad;
    const auto* bf16_input = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    for (int projection = 0; projection < 2; ++projection) {
      float* result = projection == 0 ? gate : up;
      expert_input<<<launch_blocks(assignments * (hpad / 128)), 128, 0, stream>>>(
          bf16_input, routing, assignments, hidden, hpad, projection, input);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      expert_core<<<launch_blocks(assignments * (ipad / 128)), 128, 0, stream>>>(
          input, routing, assignments, hpad, ipad, projection, result);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      expert_output<<<launch_blocks(assignments * (ipad / 128)), 128, 0, stream>>>(
          routing, assignments, ipad, projection, result);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    activate_down_input<<<launch_blocks(assignments * (ipad / 128)), 128, 0, stream>>>(
        routing, assignments, intermediate_size, ipad, gate, up);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    expert_core<<<launch_blocks(assignments * (hpad / 128)), 128, 0, stream>>>(
        gate, routing, assignments, ipad, hpad, 2, input);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    expert_output<<<launch_blocks(assignments * (hpad / 128)), 128, 0, stream>>>(
        routing, assignments, hpad, 2, input);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    routing.topk = 0;
  }
  combine_routes<<<launch_blocks(checked_add(checked_mul(rows, hidden), 127) / 128), 128, 0, stream>>>(
      input, topk_weights.data_ptr<float>(), routing, rows, hidden, hpad,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace exl3
