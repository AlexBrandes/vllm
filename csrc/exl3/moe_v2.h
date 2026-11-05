#pragma once

#include <torch/types.h>
#include <cstdint>

namespace exl3 {

// moe_out on the tensor cores for one served shape (moe_v2.cu): padded hidden
// 2560, padded intermediate 640, every routed matrix bits_x2 10 / codebook 2,
// at most 1024 routed assignments per call. Same arguments and output as
// moe_out; results differ from it in the last bits (FP64 side stages, hi/lo
// FP16 MMAs).
int64_t moe_workspace_size_v2(int64_t max_rows, int64_t hidden_size,
                              int64_t intermediate_size, int64_t topk,
                              int64_t local_experts);
void moe_out_v2(const torch::Tensor& x, const torch::Tensor& topk_ids,
                const torch::Tensor& topk_weights,
                const torch::Tensor& packed_ptrs,
                const torch::Tensor& packed_meta,
                const torch::Tensor& expert_map, int64_t intermediate_size,
                torch::Tensor& out, torch::Tensor& workspace);

}  // namespace exl3
