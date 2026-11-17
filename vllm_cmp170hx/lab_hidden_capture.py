# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lab (default off): dump the final-norm hidden states of eager forwards for fidelity replay.

With ``VLLM_SM80_CAPTURE_HIDDEN_DIR=<dir>`` the last pipeline rank's ``Glm5NextModel.norm`` gets a
forward hook. While ``<dir>/START`` exists, every eager forward (prefill chunks; never a CUDA-graph
capture or replay) writes its post-norm output, BF16 ``[rows, hidden]``, to
``<dir>/chunk-<n>.pt`` in call order. A client that sends one teacher-forced request at a time
(``max_tokens=1``) can therefore concatenate the chunks written during each request: every
forward between two requests belongs to that request's prompt.

This is the cut point of the malaiwah GLM-5.3-Flash fidelity suite (after the final RMSNorm, before
the LM head): replaying the dump through the suite's shared BF16 head gives full-vocabulary logits
comparable to its BF16 / FP8 captures. Diagnostic boots only (synchronous host copies).
"""

import os

import torch

DIR = os.environ.get("VLLM_SM80_CAPTURE_HIDDEN_DIR", "").strip()
_installed = False
_counter = [0]


def _hook(_module, _inputs, output):
    if torch.cuda.is_current_stream_capturing():
        return
    if not os.path.exists(os.path.join(DIR, "START")):
        return
    t = output[0] if isinstance(output, tuple) else output
    if not isinstance(t, torch.Tensor) or t.dim() != 2:
        return
    n = _counter[0]
    _counter[0] += 1
    torch.save(t.detach().to("cpu", copy=True).contiguous(), os.path.join(DIR, f"chunk-{n:07d}.pt"))


def install() -> None:
    """Wrap Glm5NextModel.__init__ so the last rank's final norm carries the hook."""
    global _installed
    if not DIR or _installed:
        return
    _installed = True
    os.makedirs(DIR, exist_ok=True)
    from vllm.models.glm5next.common import model as glm

    orig_init = glm.Glm5NextModel.__init__

    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        norm = getattr(self, "norm", None)
        if isinstance(norm, torch.nn.Module) and type(norm).__name__ != "PPMissingLayer":
            norm.register_forward_hook(_hook)
            from vllm.logger import init_logger

            init_logger(__name__).info("lab hidden capture: final norm hooked, writing to %s", DIR)

    glm.Glm5NextModel.__init__ = init
