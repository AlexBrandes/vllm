# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""sm_80 KDA pieces called from shared engine code.

  ``recover``  RecoverSSM for KDA layers: the enable check, the GDN metadata
               builder hook, the per-token verify records and the commit
               kernel. The fork's GLM KDA layer, its GDN metadata builder
               (vllm/models/glm5next/nvidia/sm80_kda.py) and the GLM model
               state call it; the KDA decode and prefill kernels themselves
               stay in vllm/models/glm5next/nvidia/ops/.
"""
