# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Code the CMP 170HX fork adds to vLLM.

- ``plugin`` and ``guard``: the ``vllm.general_plugins`` entry point
  ``cmp170hx``, and the startup check that it loaded.

Importing this package does no work beyond placing its loggers: the upstream
files that use a module here import that module directly.
"""

import logging as _logging

# vLLM configures only the "vllm" logger (handler, format, level). The
# package's loggers ("vllm_cmp170hx.*", from vllm.logger.init_logger(__name__))
# get it as their ancestor, so their records reach vLLM's handler at vLLM's
# level.
_logging.getLogger(__name__).parent = _logging.getLogger("vllm")
