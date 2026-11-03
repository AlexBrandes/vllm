# SPDX-License-Identifier: Apache-2.0
# Retain Morrowmake's pinned runtime/dependencies and upstream binary extensions;
# replace the engine source and rebuild EXL3 against this image's PyTorch.
FROM ghcr.io/morrowmake/vllm-cmp170hx@sha256:5b45c4d6feafefe2eba962375b0a9a0b64fd22b9e05abb4d1966f14a32a2db79
USER root
ADD https://codeload.github.com/AlexBrandes/vllm/tar.gz/344e5c252fbdda1670676f3a2ccfb1a3914b0274 /tmp/engine.tar.gz
RUN echo '88efe6285ea33dd0e8aed029bc9913fa6dc4f8c4c59a656668a1a2d9692f12f2  /tmp/engine.tar.gz' | sha256sum -c -
RUN mv /opt/vllm-src /opt/vllm-base \
    && mkdir /opt/vllm-src \
    && tar -xzf /tmp/engine.tar.gz --strip-components=1 -C /opt/vllm-src \
    && cd /opt/vllm-base \
    && find vllm -type f \( -name '*.so' -o -name '_version.py' \) -exec cp --parents '{}' /opt/vllm-src/ \; \
    && cp -an vllm/third_party/. /opt/vllm-src/vllm/third_party/ \
    && cp -an vllm/vllm_flash_attn/. /opt/vllm-src/vllm/vllm_flash_attn/ \
    && rm -rf /opt/vllm-base /tmp/engine.tar.gz
WORKDIR /opt/vllm-src
ENV PATH="/opt/venv/bin:${PATH}"
RUN MAX_JOBS=2 python csrc/exl3/build_standalone.py --out vllm
COPY serve.sh /opt/cmp170hx/serve.sh
LABEL org.opencontainers.image.source="https://github.com/AlexBrandes/vllm" \
      org.opencontainers.image.revision="344e5c252fbdda1670676f3a2ccfb1a3914b0274" \
      org.opencontainers.image.title="Qwen3.8-Flash-Next EXL3 PP2 on CMP 170HX"
ENTRYPOINT ["/bin/bash", "/opt/cmp170hx/serve.sh"]
