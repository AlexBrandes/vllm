#include "ops.h"
#include "ple.h"
#include "core/registration.h"
#include <torch/library.h>

TORCH_LIBRARY(_exl3_C, m) {
  m.def("linear_out(Tensor x, Tensor trellis, Tensor suh, Tensor svh, "
        "int bits_x2, int codebook, int logical_in, int logical_out, "
        "Tensor(a!) out, Tensor(b!) workspace) -> ()");
  m.def("moe_out(Tensor x, Tensor topk_ids, Tensor topk_weights, "
        "Tensor packed_ptrs, Tensor packed_meta, Tensor expert_map, "
        "int intermediate_size, Tensor(a!) out, Tensor(b!) workspace) -> ()");
  m.def("ple_decode_out(Tensor packed_rows, Tensor global_ids, Tensor head_bias, "
        "Tensor head_offsets, Tensor head_vocab_sizes, int owner_lo, "
        "int owner_hi, int bits, Tensor(a!) out) -> ()");
  m.def("validate_moe_tables(Tensor packed_ptrs, Tensor packed_meta, "
        "int hidden_size, int intermediate_size) -> ()");
  // No tensor arguments: these are catch-all CPU arithmetic functions, not
  // CUDA-dispatched ops, and never inspect a device or initialize CUDA.
  m.def("linear_workspace_size(int max_rows, int padded_in, int padded_out) -> int",
        &exl3::linear_workspace_size);
  m.def("moe_workspace_size(int max_rows, int hidden_size, int intermediate_size, "
        "int topk, int local_experts) -> int", &exl3::moe_workspace_size);
}

TORCH_LIBRARY_IMPL(_exl3_C, CPU, m) {
  m.impl("validate_moe_tables", &exl3::validate_moe_tables);
}

TORCH_LIBRARY_IMPL(_exl3_C, CUDA, m) {
  m.impl("linear_out", &exl3::linear_out);
  m.impl("moe_out", &exl3::moe_out);
  m.impl("ple_decode_out", &ple_decode_out);
}

REGISTER_EXTENSION(_exl3_C)
