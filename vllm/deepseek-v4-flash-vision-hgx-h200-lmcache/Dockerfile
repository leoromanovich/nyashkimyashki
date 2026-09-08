FROM vllm/vllm-openai:deepseekv4-flash-vision@sha256:65f80b25dfb7b6a419fa8054a5da22a35bec798ef82e625354d1f708448c9b6c

USER root
ARG MAX_JOBS=4
ENV MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=1 \
    TORCH_CUDA_ARCH_LIST=9.0 \
    BUILD_WITH_CUDA=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*
COPY versions.json build_lmcache.py image_check.py /opt/recipe/
RUN python3 /opt/recipe/build_lmcache.py
LABEL local.recipe.lmcache.revision="21a5db10f1fc280fc7a5d59ce2660697669a0e1d"
ENTRYPOINT []
