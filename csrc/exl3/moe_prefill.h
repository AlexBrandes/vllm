#pragma once

#include <torch/types.h>

#include <cstdint>

namespace exl3 {

// moe_out on the tensor cores for prefill-sized calls of the served shapes
// (moe_prefill.cu): hidden 2560 / intermediate 640 with every routed matrix
// bits_x2 10 / codebook 2, or hidden 4096 / intermediate 2048 with every
// routed matrix bits_x2 5 or 7 (the 3.5-bit layer family) / codebook 2 and
// the SwiGLU fixed limit 10.0 (the caller keeps that family to models whose
// limit is exactly 10.0), at least kMoePrefillMinRows rows per call. The
// caller passes the layer's uniform bits_x2; the family selection is by
// (shape, bits_x2). Results differ from moe_out in the last bits (hi/lo FP16
// MMAs, FP64 route sums).
constexpr int64_t kMoePrefillMinRows = 65;
constexpr int64_t kMoePrefillMaxExperts = 4096;

// moe_out_decode_tc: prefill tensor-core GEMMs on an 8-row route schedule,
// with original FP32 route-weight FMAs rather than the prefill FP64 sum.
// GLM only: hidden 4096 / intermediate 2048, every routed matrix bits_x2
// 5 or 7 / codebook 2, SwiGLU fixed limit 10.0, and at most
// kMoeDecodeTcMaxRows rows. The gate/up role mode
// (VLLM_EXL3_MOE_DECODE_TC_MODE, read once per process) selects which
// projections the tensor-core schedule owns: full (default), down (scalar
// gate/up), gate_down, or up_down; an unknown value or the obsolete DOWN_ONLY
// flag is a hard error. Prefill and other calls are unchanged; defaults stay
// off behind the master decode tensor-core opt-in.
constexpr int64_t kMoeDecodeTcMaxRows = 8;
constexpr int64_t kMoeDecodeTcMaxExperts = 4096;

int64_t moe_workspace_size_prefill(int64_t max_rows, int64_t hidden_size,
                                   int64_t intermediate_size, int64_t topk,
                                   int64_t local_experts);

void moe_out_prefill(const torch::Tensor& x, const torch::Tensor& topk_ids,
                     const torch::Tensor& topk_weights,
                     const torch::Tensor& packed_ptrs,
                     const torch::Tensor& packed_meta,
                     const torch::Tensor& expert_map, int64_t intermediate_size,
                     int64_t bits_x2, torch::Tensor& out,
                     torch::Tensor& workspace);

int64_t moe_workspace_size_decode_tc(int64_t max_rows, int64_t hidden_size,
                                     int64_t intermediate_size, int64_t topk,
                                     int64_t local_experts);

void moe_out_decode_tc(const torch::Tensor& x, const torch::Tensor& topk_ids,
                       const torch::Tensor& topk_weights,
                       const torch::Tensor& packed_ptrs,
                       const torch::Tensor& packed_meta,
                       const torch::Tensor& expert_map,
                       int64_t intermediate_size, int64_t bits_x2,
                       torch::Tensor& out, torch::Tensor& workspace);

// Internal stages for the default-off decode role split
// (VLLM_EXL3_MOE_DECODE_TC_MODE). No new torch operator or workspace contract:
// the existing TC decode workspace capacity (moe_workspace_size_decode_tc)
// covers every mode. The gate/up role mode names the projections the
// tensor-core schedule owns:
//   full      TC gate + TC up + TC down (moe_out_decode_tc's own run),
//   down      scalar gate + scalar up + TC down (moe_out_decode_tc_down),
//   gate_down TC gate + scalar up + TC down,
//   up_down   scalar gate + TC up + TC down.
// The mixed modes run on moe.cu's scalar spine (assignment order): one scalar
// projection, one tensor-core projection scattered to assignment order, the
// shipped activation, then the same TC-down finish as the down mode.
//
// Scalar gate/up and the SwiGLU/activation for the down mode: the scalar prefix
// fills the workspace and its activated gate rows survive until down staging
// ends.
void moe_out_decode_tc_down(
    const torch::Tensor& x, const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights, const torch::Tensor& packed_ptrs,
    const torch::Tensor& packed_meta, const torch::Tensor& expert_map,
    int64_t intermediate_size, int64_t bits_x2, torch::Tensor& out,
    torch::Tensor& workspace, double swiglu_limit);

// The mixed-role counterfactuals (gate_down: tensor-core gate; up_down:
// tensor-core up). `gate_on_tc` selects which projection the tensor-core
// schedule computes; the other projection and the activation are scalar.
// Default-off through the existing master decode opt-in and the static mode.
void moe_out_decode_tc_mixed(
    const torch::Tensor& x, const torch::Tensor& topk_ids,
    const torch::Tensor& topk_weights, const torch::Tensor& packed_ptrs,
    const torch::Tensor& packed_meta, const torch::Tensor& expert_map,
    int64_t intermediate_size, int64_t bits_x2, torch::Tensor& out,
    torch::Tensor& workspace, double swiglu_limit, bool gate_on_tc);

// One gate/up projection on the tensor-core decode schedule, its raw log
// written to `out_assignment` (assignments * intermediate floats, assignment
// order, so moe.cu's scalar activation consumes it directly). Runs its own
// route table at `table`, stages the operand at `operand` and its per-position
// power-of-two at `unscale`; all three are caller-reserved regions of the
// decode workspace. Diagnostics-only internal helper: no torch operator, no new
// workspace contract.
void moe_decode_tc_projection(
    const torch::Tensor& x, const torch::Tensor& topk_ids,
    const torch::Tensor& expert_map, const torch::Tensor& packed_ptrs,
    const torch::Tensor& packed_meta, int64_t intermediate_size,
    int64_t bits_x2, int projection, torch::Tensor& workspace,
    float* out_assignment, int* table, float* unscale);

void moe_decode_tc_down_finish(const torch::Tensor& topk_ids,
                               const torch::Tensor& topk_weights,
                               const torch::Tensor& packed_ptrs,
                               const torch::Tensor& packed_meta,
                               const torch::Tensor& expert_map,
                               int64_t intermediate_size, int64_t bits_x2,
                               torch::Tensor& out, torch::Tensor& workspace);

}  // namespace exl3
