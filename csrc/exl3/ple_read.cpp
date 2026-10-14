// SPDX-License-Identifier: Apache-2.0
// Host reader for selected rows of the compressed PLE table on storage.
#include "ple.h"

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <torch/custom_class.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <ctime>
#include <memory>
#include <thread>
#include <utility>
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

// The arguments both entry points share; returns the heads' ends.
std::vector<int64_t> check_table(int64_t base, int64_t row_bytes,
                                 int64_t owner_lo, int64_t owner_hi,
                                 const torch::Tensor& head_offsets,
                                 const torch::Tensor& head_vocab_sizes,
                                 int64_t retries, int64_t max_threads) {
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
  const int64_t heads = head_offsets.numel();
  const int64_t* offsets = head_offsets.data_ptr<int64_t>();
  const int64_t* sizes = head_vocab_sizes.data_ptr<int64_t>();
  std::vector<int64_t> ends(heads);
  for (int64_t h = 0; h < heads; ++h) ends[h] = offsets[h] + sizes[h];
  return ends;
}

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

// Reads the rows at indices[begin, end); stores their failures in order from
// failures[0] and returns how many there are.
int64_t read_rows(const Table& t, uint8_t* dest, const int64_t* ids,
                  const int64_t* indices, int64_t begin, int64_t end,
                  Failure* failures) {
  int64_t failed = 0;
  for (int64_t i = begin; i < end; ++i) {
    const int64_t index = indices[i];
    Failure failure{index, 0, 0};
    if (!read_row(t, dest + index * t.row_bytes, ids[index], &failure)) {
      failures[failed++] = failure;
    }
  }
  return failed;
}

// What a batch read needs besides its rows, allocated for up to `rows` rows
// and max_threads threads before the read starts.
struct Scratch {
  Scratch(int64_t rows, int64_t max_threads)
      : failures(new Failure[std::max<int64_t>(rows, 1)]),
        failed(new int64_t[max_threads]) {
    pending.reserve(rows);
    workers.reserve(max_threads);
    inline_chunks.reserve(max_threads);
  }
  std::vector<int64_t> pending;
  std::unique_ptr<Failure[]> failures;
  std::unique_ptr<int64_t[]> failed;  // per chunk
  std::vector<std::thread> workers;
  std::vector<int64_t> inline_chunks;
};

// Reads rows ids[0:count) into out as ple_read_rows (ple.h) describes and
// returns how many failed; their failures are scratch.failures[0, n), sorted
// by index. Only starting a thread allocates, and a thread that cannot start
// leaves its chunk to the calling thread, so no started thread is left
// joinable by an exception.
int64_t read_batch(const Table& t, uint8_t* out, const int64_t* id,
                   int64_t count, int64_t max_threads, Scratch& s) {
  // Rows already in the page cache are read here without waiting for
  // storage; the rest are read below, on up to max_threads threads, so their
  // storage latencies overlap. For a row that is not cached the probe also
  // starts the read of its page, so such a row gets one more read attempt
  // than Exl3PLERowReader.read_rows makes.
  s.pending.clear();
  bool nowait = true;
  for (int64_t index = 0; index < count; ++index) {
    uint8_t* row = out + index * t.row_bytes;
    if (!owns(t, id[index])) {
      std::memset(row, 0, t.row_bytes);
      continue;
    }
    if (nowait) {
      iovec vector{row, static_cast<size_t>(t.row_bytes)};
      ssize_t done = preadv2(t.fd, &vector, 1, t.base + id[index] * t.row_bytes,
                             RWF_NOWAIT);
      if (done == t.row_bytes) continue;
      if (done < 0 &&
          (errno == EOPNOTSUPP || errno == EINVAL || errno == ENOSYS)) {
        nowait = false;
      }
    }
    s.pending.push_back(index);
  }

  const int64_t rows = static_cast<int64_t>(s.pending.size());
  if (rows == 0) return 0;
  const int64_t threads = std::min(max_threads, rows);
  const int64_t step = (rows + threads - 1) / threads;
  const int64_t* pending = s.pending.data();
  Failure* failures = s.failures.get();
  int64_t* failed = s.failed.get();
  s.workers.clear();
  s.inline_chunks.clear();
  int64_t chunks = 1;
  for (; chunks < threads && chunks * step < rows; ++chunks) {
    const int64_t chunk = chunks;
    const int64_t begin = chunk * step;
    const int64_t end = std::min(rows, begin + step);
    try {
      s.workers.emplace_back(
          [&t, out, id, pending, begin, end, failures, failed, chunk] {
            failed[chunk] =
                read_rows(t, out, id, pending, begin, end, failures + begin);
          });
    } catch (...) {
      s.inline_chunks.push_back(chunk);
    }
  }
  failed[0] = read_rows(t, out, id, pending, 0, std::min(rows, step), failures);
  for (int64_t chunk : s.inline_chunks) {
    const int64_t begin = chunk * step;
    failed[chunk] = read_rows(t, out, id, pending, begin,
                              std::min(rows, begin + step), failures + begin);
  }
  for (std::thread& worker : s.workers) worker.join();

  // Chunk c stored its failures from failures[c * step]; chunks hold
  // increasing indices, so packing them in chunk order keeps index order.
  int64_t total = failed[0];
  for (int64_t chunk = 1; chunk < chunks; ++chunk) {
    std::memmove(failures + total, failures + chunk * step,
                 failed[chunk] * sizeof(Failure));
    total += failed[chunk];
  }
  return total;
}

// exl3_ple.poison_rows: an FP16 quiet NaN as the row's scale.
constexpr int16_t kPoisonedScale = 0x7E00;

void poison(uint8_t* row) {
  std::memcpy(row, &kPoisonedScale, sizeof kPoisonedScale);
}

// Job phases: the caller claims a free job and queues it; the host function
// marks it done; take() returns its result and frees it.
enum : int { kFree, kQueued, kDone };

// Job status: read (failed rows, if any, are poisoned), or stopped by an
// unexpected error, with every row of the job poisoned.
enum : int64_t { kRead = 0, kStopped = 1 };

// Columns of PleHostReader::take().
constexpr int64_t kResultColumns = 9;

struct HostReadState;

// One queued read: its argument, then its result, written by the host
// function before it marks the job done.
struct HostReadJob {
  HostReadState* state = nullptr;
  int64_t sequence = 0;
  int64_t count = 0;
  int64_t status = kRead;
  int64_t failed = 0;
  Failure first{0, 0, 0};
  int64_t first_row = 0;
  int64_t read_ns = 0;
  std::atomic<int> phase{kFree};
};

// Everything a host function touches. It owns a duplicate of the table's
// descriptor and references to the staging buffers, so neither can go away
// under a queued read.
struct HostReadState {
  HostReadState(int fd, torch::Tensor ids, torch::Tensor rows,
                std::vector<int64_t> offsets, std::vector<int64_t> ends,
                const Table& shape, int64_t capacity, int64_t max_threads,
                int64_t slots)
      : fd(fd),
        ids(std::move(ids)),
        rows(std::move(rows)),
        offsets(std::move(offsets)),
        ends(std::move(ends)),
        table(shape),
        capacity(capacity),
        max_threads(max_threads),
        slots(slots),
        scratch(capacity, max_threads),
        jobs(new HostReadJob[slots]) {
    table.fd = fd;
    table.head_offsets = this->offsets.data();
    table.head_ends = this->ends.data();
    ids_data = this->ids.data_ptr<int64_t>();
    rows_data = static_cast<uint8_t*>(this->rows.data_ptr());
    for (int64_t i = 0; i < slots; ++i) jobs[i].state = this;
  }
  ~HostReadState() { ::close(fd); }

  int fd;
  torch::Tensor ids;
  torch::Tensor rows;
  std::vector<int64_t> offsets;
  std::vector<int64_t> ends;
  Table table;
  int64_t capacity;
  int64_t max_threads;
  int64_t slots;
  Scratch scratch;
  std::unique_ptr<HostReadJob[]> jobs;
  const int64_t* ids_data = nullptr;
  uint8_t* rows_data = nullptr;
  // The caller's thread only: the sequence of the next job, and of the
  // oldest job take() has not returned.
  int64_t next = 0;
  int64_t oldest = 0;
};

// The host function. It runs without the GIL on a thread of the CUDA driver,
// makes no CUDA call and lets no exception out.
void execute(HostReadJob& job) noexcept {
  // Pairs with the release that queued the job: its arguments are visible.
  static_cast<void>(job.phase.load(std::memory_order_acquire));
  HostReadState& s = *job.state;
  const auto start = std::chrono::steady_clock::now();
  const int64_t row_bytes = s.table.row_bytes;
  try {
    const int64_t failed = read_batch(s.table, s.rows_data, s.ids_data,
                                      job.count, s.max_threads, s.scratch);
    const Failure* failures = s.scratch.failures.get();
    for (int64_t i = 0; i < failed; ++i) {
      poison(s.rows_data + failures[i].index * row_bytes);
    }
    job.failed = failed;
    if (failed > 0) {
      job.first = failures[0];
      job.first_row = s.ids_data[failures[0].index];
    }
  } catch (...) {
    for (std::thread& worker : s.scratch.workers) {
      if (worker.joinable()) {
        try {
          worker.join();
        } catch (...) {
        }
      }
    }
    for (int64_t index = 0; index < job.count; ++index) {
      poison(s.rows_data + index * row_bytes);
    }
    job.status = kStopped;
    job.failed = job.count;
  }
  job.read_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count();
  job.phase.store(kDone, std::memory_order_release);
}

void CUDART_CB run_queued(void* job) {
  execute(*static_cast<HostReadJob*>(job));
}

// Reads the PLE rows of a step in a host function queued on the current
// CUDA stream, so the stream itself orders the read between the copy of the
// ids to the host and the copy of the rows to the device, and the caller
// never waits. Failed rows are poisoned as exl3_ple.poison_rows does; the
// caller learns of them through take(), one or more steps later. The caller
// uses one thread and queues every read on one stream.
class PleHostReader : public torch::CustomClassHolder {
 public:
  PleHostReader(int64_t fd, torch::Tensor ids, torch::Tensor rows, int64_t base,
                int64_t row_bytes, int64_t owner_lo, int64_t owner_hi,
                torch::Tensor head_offsets, torch::Tensor head_vocab_sizes,
                int64_t retries, int64_t max_threads, int64_t slots) {
    TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type() == torch::kInt64 &&
                    ids.dim() == 1 && ids.is_contiguous(),
                "PLE row ids must be a contiguous 1-D CPU int64 tensor");
    TORCH_CHECK(rows.device().is_cpu() && rows.is_contiguous(),
                "PLE row destination must be contiguous CPU memory");
    std::vector<int64_t> ends =
        check_table(base, row_bytes, owner_lo, owner_hi, head_offsets,
                    head_vocab_sizes, retries, max_threads);
    const int64_t capacity = ids.numel();
    TORCH_CHECK(capacity <= static_cast<int64_t>(rows.nbytes()) / row_bytes,
                "PLE row destination is too small");
    TORCH_CHECK(slots > 0, "PLE host reads need at least one slot");
    const int64_t* first = head_offsets.data_ptr<int64_t>();
    std::vector<int64_t> offsets(first, first + head_offsets.numel());
    const Table shape{
        -1,       base,    row_bytes, owner_lo,
        owner_hi, nullptr, nullptr,   static_cast<int64_t>(offsets.size()),
        retries};
    const int own = fcntl(static_cast<int>(fd), F_DUPFD_CLOEXEC, 0);
    TORCH_CHECK(own >= 0, "Cannot duplicate the PLE table descriptor: ",
                std::strerror(errno));
    try {
      state_ = std::make_unique<HostReadState>(
          own, std::move(ids), std::move(rows), std::move(offsets),
          std::move(ends), shape, capacity, max_threads, slots);
    } catch (...) {
      ::close(own);
      throw;
    }
  }

  // A queued host function still uses the state, so it is left allocated
  // rather than freed under that function. Callers drain the stream first,
  // so this happens only when that drain failed.
  ~PleHostReader() override {
    for (int64_t i = 0; i < state_->slots; ++i) {
      if (state_->jobs[i].phase.load(std::memory_order_acquire) == kQueued) {
        static_cast<void>(state_.release());
        return;
      }
    }
  }

  // Queue a read of ids[0:count) into the rows on the current CUDA stream.
  // Refused under stream capture: the job's slot is chosen at queue time.
  void enqueue(int64_t count) {
    HostReadJob& job = claim(count);
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    cudaError_t error = cudaStreamIsCapturing(stream, &capture);
    TORCH_CHECK(error != cudaSuccess || capture == cudaStreamCaptureStatusNone,
                "PLE host reads cannot be captured");
    if (error == cudaSuccess) {
      job.phase.store(kQueued, std::memory_order_release);
      error = cudaLaunchHostFunc(stream, &run_queued, &job);
      if (error != cudaSuccess) {
        job.phase.store(kFree, std::memory_order_relaxed);
      }
    }
    if (error != cudaSuccess) {
      static_cast<void>(cudaGetLastError());
      TORCH_CHECK(
          false, "Cannot queue the PLE host read: ", cudaGetErrorString(error));
    }
    ++state_->next;
  }

  // The host function of a read of ids[0:count), run on this thread.
  void run(int64_t count) {
    HostReadJob& job = claim(count);
    job.phase.store(kQueued, std::memory_order_release);
    ++state_->next;
    execute(job);
  }

  // The results of the finished reads not yet taken, in queue order, one
  // row each: [sequence, count, status (0 read, 1 stopped), failed rows,
  // first failed index, its row id, its errno (0: end of file), the byte
  // offset reached, read time in ns]. Stops at the first unfinished read.
  torch::Tensor take() {
    HostReadState& s = *state_;
    int64_t n = 0;
    while (s.oldest + n < s.next && s.jobs[(s.oldest + n) % s.slots].phase.load(
                                        std::memory_order_acquire) == kDone) {
      ++n;
    }
    torch::Tensor result = torch::empty({n, kResultColumns}, torch::kInt64);
    int64_t* data = result.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
      HostReadJob& job = s.jobs[(s.oldest + i) % s.slots];
      *data++ = job.sequence;
      *data++ = job.count;
      *data++ = job.status;
      *data++ = job.failed;
      *data++ = job.first.index;
      *data++ = job.first_row;
      *data++ = job.first.error;
      *data++ = job.first.offset;
      *data++ = job.read_ns;
      job.phase.store(kFree, std::memory_order_relaxed);
    }
    s.oldest += n;
    return result;
  }

 private:
  HostReadJob& claim(int64_t count) {
    HostReadState& s = *state_;
    TORCH_CHECK(0 <= count && count <= s.capacity,
                "PLE host read exceeds the staging capacity");
    HostReadJob& job = s.jobs[s.next % s.slots];
    TORCH_CHECK(job.phase.load(std::memory_order_acquire) == kFree,
                "Every PLE host read slot is in flight or not taken");
    job.sequence = s.next;
    job.count = count;
    job.status = kRead;
    job.failed = 0;
    job.first = Failure{0, 0, 0};
    job.first_row = 0;
    job.read_ns = 0;
    return job;
  }

  std::unique_ptr<HostReadState> state_;
};

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
  const std::vector<int64_t> ends =
      check_table(base, row_bytes, owner_lo, owner_hi, head_offsets,
                  head_vocab_sizes, retries, max_threads);
  TORCH_CHECK(0 <= count && count <= ids.numel() &&
                  count <= static_cast<int64_t>(dest.nbytes()) / row_bytes,
              "PLE row destination is too small");

  const Table table{static_cast<int>(fd),
                    base,
                    row_bytes,
                    owner_lo,
                    owner_hi,
                    head_offsets.data_ptr<int64_t>(),
                    ends.data(),
                    head_offsets.numel(),
                    retries};
  Scratch scratch(count, max_threads);
  const int64_t failed =
      read_batch(table, static_cast<uint8_t*>(dest.data_ptr()),
                 ids.data_ptr<int64_t>(), count, max_threads, scratch);
  torch::Tensor result = torch::empty({failed, 3}, torch::kInt64);
  int64_t* data = result.data_ptr<int64_t>();
  for (int64_t i = 0; i < failed; ++i) {
    const Failure& failure = scratch.failures[i];
    *data++ = failure.index;
    *data++ = failure.error;
    *data++ = failure.offset;
  }
  return result;
}

void register_ple_host_reader(torch::Library& m) {
  m.class_<PleHostReader>("PleHostReader")
      .def(torch::init<int64_t, torch::Tensor, torch::Tensor, int64_t, int64_t,
                       int64_t, int64_t, torch::Tensor, torch::Tensor, int64_t,
                       int64_t, int64_t>())
      .def("enqueue", &PleHostReader::enqueue)
      .def("run", &PleHostReader::run)
      .def("take", &PleHostReader::take);
}
