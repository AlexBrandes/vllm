// SPDX-License-Identifier: Apache-2.0
// Host reader for selected rows of the compressed PLE table on storage.
#include "ple.h"

#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <system_error>
#include <thread>
#include <vector>

namespace {

struct Failure {
  int64_t index;
  int64_t error;  // errno, or 0 for end of file
  int64_t offset;
};

struct Table {
  int fd;
  int64_t base;
  int64_t row_bytes;
  int64_t owner_lo;
  int64_t owner_hi;
  const int64_t* head_offsets;
  const int64_t* head_ends;  // offsets[h] + vocab_sizes[h]
  int64_t heads;
  int64_t retries;
};

// Exl3PLERowReader.owns: inside the owner range and inside its head's
// vocabulary (padding rows after a head belong to nobody).
bool owns(const Table& t, int64_t row) {
  if (row < t.owner_lo || row >= t.owner_hi) return false;
  const int64_t* end = t.head_offsets + t.heads;
  const int64_t* after = std::upper_bound(t.head_offsets, end, row);
  if (after == t.head_offsets) return false;
  return row < t.head_ends[after - t.head_offsets - 1];
}

// pread_exact_into: partial reads continue; EINTR is retried as os.preadv
// retries it; any other error or end of file is returned with the offset
// reached. Returns -1 on success.
int64_t read_exact(int fd, uint8_t* dest, int64_t length, int64_t offset,
                   int64_t* reached) {
  int64_t done = 0;
  while (done < length) {
    ssize_t count = pread(fd, dest + done, length - done, offset + done);
    if (count < 0) {
      if (errno == EINTR) continue;
      *reached = offset + done;
      return errno;
    }
    if (count == 0) {
      *reached = offset + done;
      return 0;
    }
    done += count;
  }
  return -1;
}

// Exl3PLERowReader.read_exact: the same bounded retries and backoff (1 ms,
// doubling).
bool read_row(const Table& t, uint8_t* dest, int64_t row, Failure* failure) {
  const int64_t offset = t.base + row * t.row_bytes;
  int64_t reached = offset;
  int64_t error = -1;
  for (int64_t attempt = 0; attempt <= t.retries; ++attempt) {
    error = read_exact(t.fd, dest, t.row_bytes, offset, &reached);
    if (error < 0) return true;
    if (attempt == t.retries) break;
    const int64_t ms = int64_t{1} << std::min<int64_t>(attempt, 40);
    timespec wait{static_cast<time_t>(ms / 1000),
                  static_cast<long>(ms % 1000) * 1000000L};
    while (nanosleep(&wait, &wait) != 0 && errno == EINTR) {
    }
  }
  failure->error = error;
  failure->offset = reached;
  return false;
}

void read_rows(const Table& t, uint8_t* dest, const int64_t* ids,
               const int64_t* indices, int64_t begin, int64_t end,
               std::vector<Failure>* failures) {
  for (int64_t i = begin; i < end; ++i) {
    const int64_t index = indices[i];
    Failure failure{index, 0, 0};
    if (!read_row(t, dest + index * t.row_bytes, ids[index], &failure)) {
      failures->push_back(failure);
    }
  }
}

}  // namespace

torch::Tensor ple_read_rows(int64_t fd, const torch::Tensor& ids,
                            torch::Tensor& dest, int64_t count, int64_t base,
                            int64_t row_bytes, int64_t owner_lo,
                            int64_t owner_hi, const torch::Tensor& head_offsets,
                            const torch::Tensor& head_vocab_sizes,
                            int64_t retries, int64_t max_threads) {
  TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type() == torch::kInt64 &&
                  ids.dim() == 1 && ids.is_contiguous(),
              "PLE row ids must be a contiguous 1-D CPU int64 tensor");
  TORCH_CHECK(dest.device().is_cpu() && dest.is_contiguous(),
              "PLE row destination must be contiguous CPU memory");
  TORCH_CHECK(
      head_offsets.device().is_cpu() && head_vocab_sizes.device().is_cpu() &&
          head_offsets.scalar_type() == torch::kInt64 &&
          head_vocab_sizes.scalar_type() == torch::kInt64 &&
          head_offsets.dim() == 1 &&
          head_offsets.sizes() == head_vocab_sizes.sizes() &&
          head_offsets.is_contiguous() && head_vocab_sizes.is_contiguous(),
      "PLE heads must be matching 1-D CPU int64 tensors");
  TORCH_CHECK(row_bytes > 0 && base >= 0 && 0 <= owner_lo &&
                  owner_lo <= owner_hi && retries >= 0 && max_threads > 0,
              "Invalid PLE read arguments");
  TORCH_CHECK(owner_hi <= (INT64_MAX - base) / row_bytes,
              "PLE file offset exceeds signed 64-bit range");
  TORCH_CHECK(0 <= count && count <= ids.numel() &&
                  count <= static_cast<int64_t>(dest.nbytes()) / row_bytes,
              "PLE row destination is too small");

  const int64_t heads = head_offsets.numel();
  const int64_t* offsets = head_offsets.data_ptr<int64_t>();
  const int64_t* sizes = head_vocab_sizes.data_ptr<int64_t>();
  std::vector<int64_t> ends(heads);
  for (int64_t h = 0; h < heads; ++h) ends[h] = offsets[h] + sizes[h];
  const Table table{
      static_cast<int>(fd), base,  row_bytes, owner_lo, owner_hi, offsets,
      ends.data(),          heads, retries};
  const int64_t* id = ids.data_ptr<int64_t>();
  uint8_t* out = static_cast<uint8_t*>(dest.data_ptr());

  // Rows already in the page cache are read here without blocking on
  // storage; the rest are read below, on up to max_threads threads, so their
  // storage latencies overlap.
  std::vector<int64_t> pending;
  bool nowait = true;
  for (int64_t index = 0; index < count; ++index) {
    uint8_t* row = out + index * row_bytes;
    if (!owns(table, id[index])) {
      std::memset(row, 0, row_bytes);
      continue;
    }
    if (nowait) {
      iovec vector{row, static_cast<size_t>(row_bytes)};
      ssize_t done = preadv2(table.fd, &vector, 1, base + id[index] * row_bytes,
                             RWF_NOWAIT);
      if (done == row_bytes) continue;
      if (done < 0 &&
          (errno == EOPNOTSUPP || errno == EINVAL || errno == ENOSYS)) {
        nowait = false;
      }
    }
    pending.push_back(index);
  }

  const int64_t rows = static_cast<int64_t>(pending.size());
  const int64_t threads = std::min(max_threads, rows);
  std::vector<std::vector<Failure>> failures(std::max<int64_t>(threads, 1));
  if (rows > 0) {
    const int64_t step = (rows + threads - 1) / threads;
    std::vector<std::thread> workers;
    std::vector<int64_t> inline_chunks;
    for (int64_t chunk = 1; chunk < threads; ++chunk) {
      const int64_t begin = chunk * step;
      const int64_t end = std::min(rows, begin + step);
      if (begin >= end) break;
      try {
        workers.emplace_back(read_rows, std::cref(table), out, id,
                             pending.data(), begin, end, &failures[chunk]);
      } catch (const std::system_error&) {
        inline_chunks.push_back(chunk);
      }
    }
    read_rows(table, out, id, pending.data(), 0, std::min(rows, step),
              &failures[0]);
    for (int64_t chunk : inline_chunks) {
      const int64_t begin = chunk * step;
      read_rows(table, out, id, pending.data(), begin,
                std::min(rows, begin + step), &failures[chunk]);
    }
    for (std::thread& worker : workers) worker.join();
  }

  std::vector<Failure> all;
  for (const auto& part : failures)
    all.insert(all.end(), part.begin(), part.end());
  std::sort(all.begin(), all.end(), [](const Failure& a, const Failure& b) {
    return a.index < b.index;
  });
  torch::Tensor result =
      torch::empty({static_cast<int64_t>(all.size()), 3}, torch::kInt64);
  int64_t* data = result.data_ptr<int64_t>();
  for (const Failure& failure : all) {
    *data++ = failure.index;
    *data++ = failure.error;
    *data++ = failure.offset;
  }
  return result;
}
