# DeepSeek-V4.1-Flash-1017 lab image (PP3 on CMP 170HX cards 1, 4, 2): an overlay on the
# GLM-5.3-Flash kf lab image (upstream vLLM a0d7356b98 + vLLM branch + vllm_cmp170hx plugin,
# torch 2.13 cu130, /opt/exl3-once lab libraries for the dense W8 sidecar).
#
# Build context (made by build.sh):
#   plugin/vllm_cmp170hx/   the dsv41/plugin package (python), with _exl3_C.abi3.so built from
#                           its csrc/exl3 (engram_fetch + V4.1 Q4N-X families) inside the base image
#   vllm-overlay/vllm/...   the files dsv41/vllm changes against glm53/vllm-a0d7356b98
#                           (models/deepseek_v41/nvidia/model.py, common/engram.py)
ARG BASE=lab-glm53-q4nx-kf:6734e9fe03
FROM ${BASE}
COPY plugin/vllm_cmp170hx/ /tmp/dsv41-plugin/
COPY vllm-overlay/ /tmp/dsv41-vllm/
RUN set -e; SP=/opt/venv/lib/python3.12/site-packages; \
    test -f /tmp/dsv41-plugin/_exl3_C.abi3.so; \
    rm -rf $SP/vllm_cmp170hx.dsv41-old; mv $SP/vllm_cmp170hx $SP/vllm_cmp170hx.dsv41-old; \
    cp -a /tmp/dsv41-plugin $SP/vllm_cmp170hx; \
    for f in $(cd /tmp/dsv41-vllm && find vllm -type f); do cp /tmp/dsv41-vllm/$f $SP/$f; done; \
    rm -rf $SP/vllm_cmp170hx.dsv41-old /tmp/dsv41-plugin /tmp/dsv41-vllm; \
    find $SP/vllm_cmp170hx $SP/vllm/models/deepseek_v41 -name __pycache__ -prune -exec rm -rf {} +; \
    python -c "import vllm_cmp170hx.models.deepseek_v41.sm80_model"
ARG PLUGIN_COMMIT=unknown
ARG VLLM_COMMIT=unknown
LABEL dsv41.overlay.plugin_commit=${PLUGIN_COMMIT} dsv41.overlay.vllm_commit=${VLLM_COMMIT} \
      dsv41.overlay.note="DeepSeek-V4.1-Flash-1017 lab engine (branches dsv41/plugin, dsv41/vllm)"
