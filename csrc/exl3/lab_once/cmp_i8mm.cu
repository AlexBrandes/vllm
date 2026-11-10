// cmp_i8mm: INT8 x INT8 -> BF16 scaled GEMM for the 27B's prefill W4A8 path
// (overlay cmp_w4a8_prefill.prefill_gemm), with a mainloop chosen for the CMP
// 170HX under its 170 W cap.
//
// Same math as vLLM's cutlass_scaled_mm on sm80 (c2x ScaledEpilogue):
//   D[m, n] = bf16( scale_a[m] * ( scale_b[n] * float(acc[m, n]) ) )
// with per-token scale_a [M] and per-channel scale_b [N] (both fp32), so the
// output is bit-identical. Only the mainloop differs: 128x256x64 tiles with
// 64x64x64 warps (8 warps), 3 stages (73,728 B smem) and an 8-wide identity
// swizzle, instead of vLLM's 128x128x64 / 5 stages / stream-K swizzle.
// Synthetic, sustained at the cap (tests/i8gemm): +9 to +13% on the 27B's
// prefill shapes (fewer shared-memory and L2 bytes per MAC -> higher clock).
//
// Built inside the engine image (build_i8mm.sh) against its torch and
// flashinfer's bundled CUTLASS; loaded with torch.ops.load_library.
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"

using namespace cute;

namespace {

template <typename TileShape, typename WarpShape, int Stages, int Swizzle>
struct I8Gemm {
  using ElementD = cutlass::bfloat16_t;
  using OutputTileThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<TileShape, WarpShape, float, 4, 1>;
  using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
  using ScaleA = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputTileThreadMap, float, Stride<Int<1>, Int<0>, Int<0>>>;
  using ScaleB = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputTileThreadMap, float, Stride<Int<0>, Int<1>, Int<0>>>;
  using Compute0 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT0 = cutlass::epilogue::threadblock::Sm80EVT<Compute0, ScaleB, Accum>;
  using Compute1 = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, ElementD, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTCompute = cutlass::epilogue::threadblock::Sm80EVT<Compute1, ScaleA, EVT0>;
  using D = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap, ElementD, cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, Int<1>, Int<0>>>;
  using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute>;

  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
      int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
      float, cutlass::layout::RowMajor, 4,
      int32_t, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      TileShape, WarpShape, cutlass::gemm::GemmShape<16, 8, 32>, EVTD,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>, Stages,
      cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
  using Op = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static void run(at::Tensor& out, const at::Tensor& a, const at::Tensor& b,
                  const at::Tensor& sa, const at::Tensor& sb) {
    int m = a.size(0), k = a.size(1), n = b.size(0);
    const int64_t ldd = out.stride(0);
    using StrideD = Stride<int64_t, Int<1>, Int<0>>;
    typename D::Arguments d_args{reinterpret_cast<ElementD*>(out.data_ptr()), StrideD{ldd, Int<1>{}, Int<0>{}}};
    typename EVT0::Arguments evt0{{sb.data_ptr<float>()}, {}, {}};
    typename EVTCompute::Arguments evtc{{sa.data_ptr<float>()}, evt0, {}};
    typename EVTD::Arguments epi{evtc, d_args};
    typename Op::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, 1, epi,
                                a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), nullptr, nullptr,
                                0, 0, 0, 0, (int64_t)k, (int64_t)k, ldd, ldd};
    Op op;
    TORCH_CHECK(op.can_implement(args) == cutlass::Status::kSuccess, "cmp_i8mm: cannot implement");
    size_t ws = op.get_workspace_size(args);
    at::Tensor workspace;
    if (ws) workspace = at::empty({(int64_t)ws}, a.options().dtype(at::kByte));
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    TORCH_CHECK(op(args, ws ? workspace.data_ptr() : nullptr, stream) == cutlass::Status::kSuccess,
                "cmp_i8mm: launch failed");
  }
};

using Wide = I8Gemm<cutlass::gemm::GemmShape<128, 256, 64>, cutlass::gemm::GemmShape<64, 64, 64>, 3, 8>;

// a [M, K] int8 row-major; b [N, K] int8 row-major (the rebuilt weight);
// scale_a [M] or [M, 1] fp32; scale_b [N] or [1, N] fp32 -> [M, N] bf16
at::Tensor scaled_mm(const at::Tensor& a, const at::Tensor& b, const at::Tensor& scale_a,
                     const at::Tensor& scale_b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.dim() == 2 && b.dim() == 2, "cmp_i8mm: 2-D CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::kChar && b.scalar_type() == at::kChar, "cmp_i8mm: int8 operands");
  TORCH_CHECK(a.size(1) == b.size(1), "cmp_i8mm: K mismatch");
  TORCH_CHECK(a.stride(1) == 1 && a.stride(0) == a.size(1) && b.is_contiguous(), "cmp_i8mm: contiguous operands");
  TORCH_CHECK(a.size(1) % 16 == 0, "cmp_i8mm: K % 16");
  TORCH_CHECK(scale_a.scalar_type() == at::kFloat && scale_a.is_contiguous() && scale_a.numel() == a.size(0),
              "cmp_i8mm: per-token fp32 scale_a");
  TORCH_CHECK(scale_b.scalar_type() == at::kFloat && scale_b.is_contiguous() && scale_b.numel() == b.size(0),
              "cmp_i8mm: per-channel fp32 scale_b");
  TORCH_CHECK(b.size(0) % 4 == 0, "cmp_i8mm: N % 4");
  const at::cuda::OptionalCUDAGuard guard(a.device());
  at::Tensor out = at::empty({a.size(0), b.size(0)}, a.options().dtype(at::kBFloat16));
  if (a.size(0) == 0) return out;
  Wide::run(out, a, b, scale_a, scale_b);
  return out;
}

// GLM lab: write into `out` [M, N] bf16 with unit column stride and any row
// stride (a column slice of a wider output), so N-chunked calls need no copy.
void scaled_mm_out(at::Tensor& out, const at::Tensor& a, const at::Tensor& b,
                   const at::Tensor& scale_a, const at::Tensor& scale_b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.dim() == 2 && b.dim() == 2, "cmp_i8mm: 2-D CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::kChar && b.scalar_type() == at::kChar, "cmp_i8mm: int8 operands");
  TORCH_CHECK(a.size(1) == b.size(1), "cmp_i8mm: K mismatch");
  TORCH_CHECK(a.stride(1) == 1 && a.stride(0) == a.size(1) && b.is_contiguous(), "cmp_i8mm: contiguous operands");
  TORCH_CHECK(a.size(1) % 16 == 0 && b.size(0) % 8 == 0, "cmp_i8mm: K % 16, N % 8");
  TORCH_CHECK(scale_a.scalar_type() == at::kFloat && scale_a.is_contiguous() && scale_a.numel() == a.size(0),
              "cmp_i8mm: per-token fp32 scale_a");
  TORCH_CHECK(scale_b.scalar_type() == at::kFloat && scale_b.is_contiguous() && scale_b.numel() == b.size(0),
              "cmp_i8mm: per-channel fp32 scale_b");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(0) == a.size(0) &&
                  out.size(1) == b.size(0) && out.stride(1) == 1 && out.stride(0) % 8 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "cmp_i8mm: out bf16 [M, N], unit column stride, 16-byte aligned rows");
  if (a.size(0) == 0) return;
  const at::cuda::OptionalCUDAGuard guard(a.device());
  Wide::run(out, a, b, scale_a, scale_b);
}

}  // namespace

TORCH_LIBRARY(cmp_i8mm, m) {
  m.def("scaled_mm(Tensor a, Tensor b, Tensor scale_a, Tensor scale_b) -> Tensor");
  m.def("scaled_mm_out(Tensor(a!) out, Tensor a, Tensor b, Tensor scale_a, Tensor scale_b) -> ()");
}
TORCH_LIBRARY_IMPL(cmp_i8mm, CUDA, m) {
  m.impl("scaled_mm", &scaled_mm);
  m.impl("scaled_mm_out", &scaled_mm_out);
}
