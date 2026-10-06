# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np

from vllm.v1.worker.gpu.async_utils import failed_request_ids


def test_failed_tokens_map_to_their_requests():
    # Requests a=[0,3), b empty, c=[3,5), d=[5,7); tokens 7.. are padding.
    query_start_loc = np.array([0, 3, 3, 5, 7], dtype=np.int32)
    req_ids = ["a", "b", "c", "d"]
    mask = np.zeros(9, dtype=bool)
    assert failed_request_ids(mask, query_start_loc, req_ids) == set()
    mask[[3, 4, 8]] = True
    assert failed_request_ids(mask, query_start_loc, req_ids) == {"c"}
    mask[[0, 6]] = True
    assert failed_request_ids(mask, query_start_loc, req_ids) == {"a", "c", "d"}
    mask[:] = False
    mask[2] = True
    assert failed_request_ids(mask, query_start_loc, req_ids) == {"a"}
