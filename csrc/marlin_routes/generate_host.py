#!/usr/bin/env python3
"""K3 Marlin routes for the vllm-cmp170hx engine (upstream vLLM a0d7356b98).

Generates marlin_host.cpp from upstream's marlin.cu at the engine's wheel
commit (marlin.upstream.cu, checked by hash): a fenced host-only dispatcher that
reuses the wheel's installed device kernels (get_marlin_kernel and the rest
of the selector are exported by _C_stable_libtorch.abi3.so) and adds a
measured (M, K, N) -> tile / threads / CTAs-per-SM table for int8-activation
(W4A8) calls. Every other call keeps the stock selector at one CTA per SM.
Upstream dropped activation order (no g_idx/perm) and already sizes the
Marlin workspace for 4 CTAs per SM (MARLIN_MAX_BLOCKS_PER_SM).
"""
import hashlib
import json
import os
from pathlib import Path

# Experimental: STAGES=1 adds a pipeline-stage argument and routes stage
# counts other than the wheel's 4 to our own kernels (../marlin-stages,
# libcmp_marlin_stages.so). Without it the output is the v2 dispatcher.
STAGES = os.environ.get("STAGES") == "1"

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'marlin.upstream.cu').read_text()
assert hashlib.sha256(source.encode()).hexdigest() == \
    'a6e91b6281759b45fd9ca4e781889b072e2566f73612e13a1c0694d49d46bcc2'
kernel_h = (ROOT / 'kernel.h').read_text()


def change(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('pinned source boundary changed: ' + old[:90])
    return text.replace(old, new, 1)


def declaration(prefix):
    start = source.index(prefix)
    return source[start:source.index(' {', start)] + ';\n'


# MARLIN_KERNEL_PARAMS, verbatim from kernel.h (a multi-line #define).
_ks = kernel_h.index('#define MARLIN_KERNEL_PARAMS')
_ke = _ks
while kernel_h[_ke:kernel_h.index('\n', _ke)].rstrip().endswith('\\'):
    _ke = kernel_h.index('\n', _ke) + 1
kernel_params = kernel_h[_ks:kernel_h.index('\n', _ke)] + '\n'

license_header = source[:source.index('#ifndef MARLIN_NAMESPACE_NAME')]
struct_start = source.index('typedef struct {')
thread_struct = source[struct_start:source.index('thread_config_t;', struct_start)
                       + len('thread_config_t;')]
exec_start = source.index('typedef struct {', struct_start + len(thread_struct))
exec_struct = source[exec_start:source.index('exec_config_t;', exec_start)
                     + len('exec_config_t;')]
abi = '''#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include "core/scalar_type.hpp"
''' + kernel_params + '''namespace marlin {
static constexpr int default_threads = 256;
static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;
static constexpr int max_thread_n = 256;
static constexpr int tile_size = 16;
constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }
using std::min;
using std::max;
using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);
'''
abi += thread_struct + '\n' + exec_struct + '\n'
for prefix in ('bool is_valid_config(', 'MarlinFuncPtr get_marlin_kernel(',
               'exec_config_t determine_exec_config('):
    abi += declaration(prefix)
abi += '}  // namespace marlin\n'
(ROOT / 'marlin_host_abi.h').write_text(license_header + abi)

start = source.index('void marlin_mm(')
end = source.index('\n}  // namespace marlin\n\ntorch::stable::Tensor marlin_gemm(', start)
dispatch = source[start:end]
dispatch = change(dispatch, 'bool use_fp32_reduce, bool is_zp_float) {',
                  'bool use_fp32_reduce, bool is_zp_float, int num_threads_init, '
                  'int blocks_per_sm) {')
dispatch = change(dispatch, '  int max_par = 16;',
    '''  // MarlinDefault (the selector's "no kernel" sentinel) is not exported;
  // ask the installed selector for an impossible tile once and cache it.
  static const MarlinFuncPtr unsupported_kernel = get_marlin_kernel(
      vllm::kS8, vllm::kU4B8, vllm::kBFloat16, vllm::kBFloat16,
      0, 0, 0, false, false, 0, 0, false, 4);
  int max_par = 16;''')
dispatch = change(dispatch, '    if (kernel == MarlinDefault) {',
                  '    if (kernel == unsupported_kernel) {')
dispatch = change(dispatch,
    '      thread_tfg = thread_config_t{thread_k, thread_n, default_threads};',
    '      thread_tfg = thread_config_t{thread_k, thread_n, num_threads_init};')
dispatch = change(dispatch,
    '    int num_threads = thread_tfg.num_threads;',
    '''    exec_cfg.blocks_per_sm = blocks_per_sm;
    int num_threads = thread_tfg.num_threads;''')
dispatch = change(dispatch, '  int max_shared_mem_new = max_shared_mem;',
    '''  int max_shared_mem_new = blocks_per_sm > 1
      ? max_shared_mem / blocks_per_sm - 1024 : max_shared_mem;''')
dispatch = change(dispatch,
    '          is_zp_float, is_a_8bit, stages, max_shared_mem, sms);',
    '          is_zp_float, is_a_8bit, stages, max_shared_mem_new, sms);')
dispatch = change(dispatch,
    '    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,\n'
    '                         max_shared_mem_new);',
    '''    auto attribute_status = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel),
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);
    STD_TORCH_CHECK(attribute_status == cudaSuccess,
                    cudaGetErrorString(attribute_status));''')
dispatch = change(dispatch,
    '''    kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr,
        prob_m_split, prob_n, prob_k, lda, locks, has_bias, part_use_atomic_add,
        use_fp32_reduce, max_shared_mem_new);''',
    '''    void* kernel_args[] = {&A_ptr, &B_ptr, &C_ptr, &C_tmp_ptr,
        &bias_ptr, &a_s_ptr, &b_s_ptr, &g_s_ptr, &zp_ptr,
        &prob_m_split, &prob_n, &prob_k, &lda, &locks,
        &has_bias, &part_use_atomic_add, &use_fp32_reduce, &max_shared_mem_new};
    auto launch_status = cudaLaunchKernel(reinterpret_cast<const void*>(kernel),
        dim3(blocks), dim3(num_threads), kernel_args, max_shared_mem_new, stream);
    STD_TORCH_CHECK(launch_status == cudaSuccess, cudaGetErrorString(launch_status));''')

if STAGES:
    dispatch = change(dispatch, 'int num_threads_init, int blocks_per_sm) {',
                      'int num_threads_init, int blocks_per_sm, int stages_override) {')
    dispatch = change(dispatch, '  int stages = 4;',
                      '  int stages = stages_override > 0 ? stages_override : 4;')
    dispatch = change(dispatch, """    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
        thread_k_blocks, m_block_size_8, has_zp, group_blocks, num_threads,
        is_zp_float, stages);""", """    MarlinFuncPtr kernel;
    if (stages == 4) {
      kernel = get_marlin_kernel(
          a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
          thread_k_blocks, m_block_size_8, has_zp, group_blocks, num_threads,
          is_zp_float, stages);
    } else {
      // STAGES=1: stage-variant kernels (W4A8 uint4b8, BF16, group 128 only)
      STD_TORCH_CHECK(a_type == vllm::kS8 && b_type == vllm::kU4B8 &&
                      c_type == vllm::kBFloat16 && group_blocks == 8 &&
                      !has_zp && !m_block_size_8,
                      "CMP stage kernels cover W4A8 uint4b8 BF16 group 128 only");
      auto stage_kernel = cmp_marlin_dev::cmp_stage_kernel(
          num_threads, thread_m_blocks, thread_n_blocks, thread_k_blocks, stages);
      kernel = stage_kernel ? reinterpret_cast<MarlinFuncPtr>(stage_kernel)
                            : unsupported_kernel;
    }""")

start = source.rindex('torch::stable::Tensor marlin_gemm(')
end = source.index('\n#endif', start)
wrapper = source[start:end]
wrapper = change(wrapper, 'torch::stable::Tensor marlin_gemm(',
                 'torch::stable::Tensor cmp_marlin_gemm(')
wrapper = change(wrapper, '    bool use_fp32_reduce, bool is_zp_float) {',
    '''    bool use_fp32_reduce, bool is_zp_float,
    int64_t thread_k, int64_t thread_n, int64_t num_threads, int64_t blocks_per_sm) {
  STD_TORCH_CHECK(blocks_per_sm >= 1 && blocks_per_sm <= 4,
                  "CMP blocks_per_sm must be between one and four");
  const bool automatic = thread_k == -1 && thread_n == -1 && num_threads == -1;
  STD_TORCH_CHECK(automatic ||
      ((thread_k == 64 || thread_k == 128) &&
       (thread_n == 64 || thread_n == 128 || thread_n == 256) &&
       (num_threads == 128 || num_threads == 256)),
      "CMP requires a complete supported tile or the unmodified automatic selector");''')
wrapper = change(wrapper, '  int thread_k = -1;\n', '')
wrapper = change(wrapper, '  int thread_n = -1;\n', '')
wrapper = change(wrapper,
    '  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_index);',
    '''  STD_TORCH_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                  device_index) == cudaSuccess);
  int major = 0, minor = 0;
  STD_TORCH_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                  device_index) == cudaSuccess);
  STD_TORCH_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                  device_index) == cudaSuccess);
  STD_TORCH_CHECK(major == 8 && minor == 0, "CMP host tuner requires SM80");''')
wrapper = change(wrapper,
    '        sms * max_m_block_size * MARLIN_NAMESPACE_NAME::max_thread_n;',
    '        sms * blocks_per_sm * max_m_block_size * MARLIN_NAMESPACE_NAME::max_thread_n;')
wrapper = change(wrapper, '  int min_workspace_size = sms;',
                 '  int min_workspace_size = sms * blocks_per_sm;')
wrapper = change(wrapper, '  marlin::marlin_mm(', '  cmp_marlin::marlin_mm(')
wrapper = change(wrapper, '      use_atomic_add, use_fp32_reduce, is_zp_float);',
    '      use_atomic_add, use_fp32_reduce, is_zp_float, num_threads, blocks_per_sm);')

if STAGES:
    wrapper = change(wrapper, 'int64_t thread_k, int64_t thread_n, int64_t num_threads, int64_t blocks_per_sm) {',
                     'int64_t thread_k, int64_t thread_n, int64_t num_threads, int64_t blocks_per_sm,\n    int64_t stages) {\n  STD_TORCH_CHECK(stages >= 2 && stages <= 8, "CMP stages must be 2..8");')
    wrapper = change(wrapper, 'is_zp_float, num_threads, blocks_per_sm);',
                     'is_zp_float, num_threads, blocks_per_sm, stages);')

routes = json.loads((ROOT / 'routes.json').read_text())["routes"]
rows = []
for r in routes:
    rows.append(
        f"    if (size_m == {r['m']} && size_k == {r['k']} && size_n == {r['n']}) {{\n"
        f"      thread_k = {r['thread_k']}; thread_n = {r['thread_n']}; "
        f"num_threads = {r['num_threads']}; blocks_per_sm = {r['blocks_per_sm']};"
        + (f" stages = {r.get('stages', 4)};" if STAGES else "") + "\n"
        f"    }}")
tuned_wrapper = r'''
torch::stable::Tensor cmp_marlin_gemm_tuned(
    torch::stable::Tensor& a, std::optional<torch::stable::Tensor> c_or_none,
    torch::stable::Tensor& b_q_weight,
    std::optional<torch::stable::Tensor> const& b_bias_or_none,
    torch::stable::Tensor& b_scales,
    std::optional<torch::stable::Tensor> const& a_scales_or_none,
    std::optional<torch::stable::Tensor> const& global_scale_or_none,
    std::optional<torch::stable::Tensor> const& b_zeros_or_none,
    torch::stable::Tensor& workspace, vllm::ScalarTypeId const& b_type_id,
    int64_t size_m, int64_t size_n, int64_t size_k, bool use_atomic_add,
    bool use_fp32_reduce, bool is_zp_float) {
  int64_t thread_k = -1;
  int64_t thread_n = -1;
  int64_t num_threads = -1;
  int64_t blocks_per_sm = 1;
  if (a.scalar_type() == torch::headeronly::ScalarType::Char) {
@ROWS@
  }
  return cmp_marlin_gemm(
      a, c_or_none, b_q_weight, b_bias_or_none, b_scales, a_scales_or_none,
      global_scale_or_none, b_zeros_or_none, workspace, b_type_id,
      size_m, size_n, size_k, use_atomic_add, use_fp32_reduce, is_zp_float,
      thread_k, thread_n, num_threads, blocks_per_sm@STAGEARG@);
}
'''.replace('@ROWS@', '\n'.join(rows))
tuned_wrapper = tuned_wrapper.replace('@STAGEARG@', ', stages' if STAGES else '')
if STAGES:
    tuned_wrapper = change(tuned_wrapper, '  int64_t blocks_per_sm = 1;\n',
                           '  int64_t blocks_per_sm = 1;\n  int64_t stages = 4;\n')

header = '''// Host dispatcher derived from upstream vLLM a0d7356b98 marlin.cu; device code
// is NOT rebuilt (the wheel's installed kernels are called through its selector).
#define MARLIN_NAMESPACE_NAME marlin
#include "marlin_host_abi.h"
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include "torch_utils.h"
using std::min;
using std::max;
namespace cmp_marlin {
using namespace marlin;
'''
if STAGES:
    header = header.replace('namespace cmp_marlin {', '''namespace cmp_marlin_dev {
using StageFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);
StageFuncPtr cmp_stage_kernel(int threads, int thread_m_blocks, int thread_n_blocks,
                              int thread_k_blocks, int stages);
}  // namespace cmp_marlin_dev
namespace cmp_marlin {''', 1)
registration = '''
// The schema is defined from Python before this implementation library loads.
STABLE_TORCH_LIBRARY_IMPL(_CMP_marlin, CUDA, m) {
  m.impl("marlin_gemm", TORCH_BOX(&cmp_marlin_gemm));
  m.impl("marlin_gemm_tuned", TORCH_BOX(&cmp_marlin_gemm_tuned));
}
'''
(ROOT / 'marlin_host.cpp').write_text(license_header + header + dispatch +
    '\n}  // namespace cmp_marlin\n\n' + wrapper + tuned_wrapper + registration)
print('Generated host-only dispatcher (upstream a0d7356b98); no kernels changed.')
