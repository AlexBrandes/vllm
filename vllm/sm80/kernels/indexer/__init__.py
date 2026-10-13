# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer top-k tie fix: canonical (lowest-index) choice on exact ties at the
top-k boundary, optionally sorting each row. The kernels in ``topk_tiefix.py``
are shared by the GLM kpool sparse indexer
(``vllm/models/glm5next/nvidia/sparse_indexer.py``, flags and dispatch in
``topk_tiefix_gate.py``) and the Qwen QSA indexer
(``vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py``, VLLM_QWEN4_QSA_TOPK_CANONICAL
and dispatch in ``qsa_topk_gate.py``).
"""
