# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer top-k tie fix: canonical (lowest-index) choice on exact ties at the
top-k boundary, optionally sorting each row: kernels in ``topk_tiefix.py``,
flags and dispatch in ``topk_tiefix_gate.py`` (called from the GLM kpool sparse
indexer, ``vllm/models/glm5next/nvidia/sparse_indexer.py``).
"""
