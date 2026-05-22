services:
  qwen3-coder-nvfp4-dpa:
    image: ${SGLANG_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260520-425dffbd}
    container_name: qwen3-coder-nvfp4-dpa
    ipc: host
    shm_size: "32g"
    restart: unless-stopped
    gpus: all
    ports:
      - "${PORT:-30000}:30000"
    volumes:
      - ${HF_CACHE}:/root/.cache/huggingface
      - ./logs:/logs
    environment:
      HF_TOKEN: ${HF_TOKEN:-}
      HUGGING_FACE_HUB_TOKEN: ${HF_TOKEN:-}
      HF_HOME: /root/.cache/huggingface
      CUDA_DEVICE_ORDER: PCI_BUS_ID
      OMP_NUM_THREADS: ${OMP_NUM_THREADS:-8}
      NCCL_NVLS_ENABLE: "0"
    command: >
      python3 -m sglang.launch_server
      --model-path ${MODEL_PATH:-NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4}
      --served-model-name ${SERVED_MODEL_NAME:-qwen3-coder-nvfp4-proxy}
      --host 0.0.0.0
      --port 30000
      --trust-remote-code
      --tensor-parallel-size 2
      --data-parallel-size 2
      --enable-dp-attention
      --load-balance-method total_tokens
      --quantization modelopt_fp4
      --fp4-gemm-backend ${FP4_GEMM_BACKEND:-flashinfer_cutlass}
      --moe-runner-backend ${MOE_RUNNER_BACKEND:-flashinfer_cutlass}
      --attention-backend ${ATTENTION_BACKEND:-trtllm_mha}
      --kv-cache-dtype ${KV_CACHE_DTYPE:-fp8_e4m3}
      --page-size ${PAGE_SIZE:-32}
      --context-length ${CONTEXT_LENGTH:-32768}
      --tool-call-parser qwen3_coder
      --chunked-prefill-size ${CHUNKED_PREFILL_SIZE:-8192}
      --schedule-conservativeness ${SCHEDULE_CONSERVATIVENESS:-0.8}
      --cuda-graph-max-bs ${CUDA_GRAPH_MAX_BS:-16}
      --max-running-requests ${MAX_RUNNING_REQUESTS:-16}
      --mem-fraction-static ${MEM_FRACTION_STATIC:-0.78}
      --enable-hierarchical-cache
      --hicache-size ${HICACHE_SIZE_GB_PER_RANK:-16}
      --hicache-io-backend kernel
      --hicache-mem-layout page_first
      --hicache-write-policy write_through_selective
      --hicache-storage-prefetch-policy timeout
      --enable-metrics
      --enable-metrics-for-all-schedulers
      --enable-cache-report
      --log-level info
