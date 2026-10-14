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

// Host only. Reads rows ids[0:count) of the compressed table in file fd
// (trellis at byte base, row_bytes per row) into dest, row i at byte
// i * row_bytes: the same bytes as Exl3PLERowReader.read_rows. Rows outside
// [owner_lo, owner_hi) or after their head's vocabulary are zero-filled.
// Rows in the page cache are read on the calling thread; the others on up
// to max_threads threads. A row that still fails after retries (1 ms
// backoff, doubling) is left unspecified and returned as one [index, errno
// (0: end of file), byte offset reached] row of the int64 result, sorted by
// index.
torch::Tensor ple_read_rows(int64_t fd, const torch::Tensor& ids,
                            torch::Tensor& dest, int64_t count, int64_t base,
                            int64_t row_bytes, int64_t owner_lo,
                            int64_t owner_hi, const torch::Tensor& head_offsets,
                            const torch::Tensor& head_vocab_sizes,
                            int64_t retries, int64_t max_threads);
