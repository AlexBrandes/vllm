#pragma once

#include <torch/types.h>
#include <cstdint>

namespace exl3 {

int64_t linear_workspace_size(int64_t max_rows, int64_t padded_in,
                              int64_t padded_out);
int64_t moe_workspace_size(int64_t max_rows, int64_t hidden_size,
                           int64_t intermediate_size, int64_t topk,
                           int64_t local_experts);

void linear_out(const torch::Tensor& x, const torch::Tensor& trellis,
                const torch::Tensor& suh, const torch::Tensor& svh,
                int64_t bits_x2, int64_t codebook, int64_t logical_in,
                int64_t logical_out, torch::Tensor& out,
                torch::Tensor& workspace);
// Call once on CPU construction tables before copying them to CUDA. Forward
// requires immutable, validated tables and retained CUDA weight owners.
void validate_moe_tables(const torch::Tensor& packed_ptrs,
                         const torch::Tensor& packed_meta,
                         int64_t hidden_size, int64_t intermediate_size);

void moe_out(const torch::Tensor& x, const torch::Tensor& topk_ids,
             const torch::Tensor& topk_weights,
             const torch::Tensor& packed_ptrs,
             const torch::Tensor& packed_meta,
             const torch::Tensor& expert_map, int64_t intermediate_size,
             torch::Tensor& out, torch::Tensor& workspace);

}  // namespace exl3
