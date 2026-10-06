// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

#include <cstdint>

// All tensors are contiguous CUDA tensors on one device. packed_rows contains
// selected rows, in global_ids order, rather than the full embedding table.
// A mapped pinned-host buffer is supported through its CUDA/UVA tensor view;
// its CPU owner and transfer/readiness lifetime remain the caller's responsibility.
// head_offsets must be sorted and describe nonoverlapping heads; the loader
// validates metadata values before capture. owner_hi is exclusive.
// No tensor allocation, host synchronization, or codebook table is needed.
void ple_decode_out(const torch::Tensor& packed_rows,
                    const torch::Tensor& global_ids,
                    const torch::Tensor& head_bias,
                    const torch::Tensor& head_offsets,
                    const torch::Tensor& head_vocab_sizes, int64_t owner_lo,
                    int64_t owner_hi, int64_t bits, torch::Tensor& out);
