# SPDX-License-Identifier: Apache-2.0
# GLM-5.3-Flash EXL3 on two CMP 170HX cards: the engine is three pins, as
# docs/development.md describes. Upstream vLLM's prebuilt wheel for the base
# commit, the vLLM branch's Python files laid over it (the branch changes no
# native code), and the vllm-cmp170hx plugin installed beside it with its EXL3
# extension compiled here. Building needs network access and CPU/RAM, not a GPU.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266

ARG UPSTREAM_COMMIT=a0d7356b98e929c17051bac44f5a086e4117974d
ARG VLLM_WHEEL=https://wheels.vllm.ai/a0d7356b98e929c17051bac44f5a086e4117974d/vllm-0.33.0rc1.dev659%2Bga0d7356b9-cp38-abi3-manylinux_2_28_x86_64.whl
ARG VLLM_WHEEL_SHA256=5e293d985d0e3964538c989d6123d1b073c2c12d5384c5429e7357868e9e47e2
ARG VLLM_BRANCH_COMMIT=ea2ef399a614ea72b65b15804000342f629a850f
ARG VLLM_BRANCH_SHA256=f67a1f89f71d6ff1d17f309bf7845f6dfa6ed80ba57379561cd2d6f6b0db4548
ARG PLUGIN_COMMIT=d5cb84150bb0327224d7caa72aae1135f2bd2d92
ARG PLUGIN_SHA256=c479fc1e4fe8871031d9b840906ef25769043ffedb9ae8cb4ac015c427e32daa

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:${PATH}
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.12 python3.12-venv python3.12-dev ca-certificates curl ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir uv==0.9.8

# 1. Upstream vLLM at the base commit, with its pinned dependencies.
RUN curl -fsSL -o /tmp/vllm.whl "${VLLM_WHEEL}" \
    && echo "${VLLM_WHEEL_SHA256}  /tmp/vllm.whl" | sha256sum -c - \
    && mv /tmp/vllm.whl "/tmp/$(basename "${VLLM_WHEEL}" | sed 's/%2B/+/')" \
    && uv pip install --no-cache /tmp/vllm-*.whl \
    && rm -f /tmp/vllm-*.whl

# 2. The vLLM branch (upstream + the fork's commits): its Python files over the
#    installed package. The wheel keeps its compiled extensions and _version.py.
RUN curl -fsSL -o /tmp/vllm-branch.tar.gz \
       "https://codeload.github.com/AlexBrandes/vllm/tar.gz/${VLLM_BRANCH_COMMIT}" \
    && echo "${VLLM_BRANCH_SHA256}  /tmp/vllm-branch.tar.gz" | sha256sum -c - \
    && mkdir /tmp/vllm-branch \
    && tar -xzf /tmp/vllm-branch.tar.gz --strip-components=1 -C /tmp/vllm-branch \
    && SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
    && cd /tmp/vllm-branch \
    && find vllm -type f -name '*.py' ! -name '_version.py' -exec cp --parents '{}' "$SP/" \; \
    && rm -rf /tmp/vllm-branch /tmp/vllm-branch.tar.gz

# 3. The plugin, with the EXL3 extension built against this image's PyTorch.
RUN curl -fsSL -o /tmp/plugin.tar.gz \
       "https://codeload.github.com/AlexBrandes/vllm/tar.gz/${PLUGIN_COMMIT}" \
    && echo "${PLUGIN_SHA256}  /tmp/plugin.tar.gz" | sha256sum -c - \
    && mkdir /opt/vllm-cmp170hx \
    && tar -xzf /tmp/plugin.tar.gz --strip-components=1 -C /opt/vllm-cmp170hx \
    && rm /tmp/plugin.tar.gz \
    && cd /opt/vllm-cmp170hx \
    && MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.0 python csrc/exl3/build_standalone.py --out vllm_cmp170hx \
    && uv pip install --no-cache --no-deps . \
    && python -c "import importlib.metadata as m; \
assert [e.value for e in m.entry_points(group='vllm.general_plugins') if e.name == 'cmp170hx'] == ['vllm_cmp170hx.plugin:register']; \
assert [e.value for e in m.entry_points(group='vllm.platform_plugins') if e.name == 'cmp170hx'] == ['vllm_cmp170hx.plugin:platform_plugin']"

COPY serve.sh /opt/cmp170hx/serve.sh
LABEL org.opencontainers.image.source="https://github.com/AlexBrandes/vllm" \
      org.opencontainers.image.title="GLM-5.3-Flash EXL3 PP2 on CMP 170HX" \
      cmp170hx.upstream_commit="${UPSTREAM_COMMIT}" \
      cmp170hx.vllm_commit="${VLLM_BRANCH_COMMIT}" \
      cmp170hx.plugin_commit="${PLUGIN_COMMIT}"
WORKDIR /cache
ENTRYPOINT ["/bin/bash", "/opt/cmp170hx/serve.sh"]
