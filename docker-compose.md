kimi-k2.5

```
version: "3.9"

  services:
    vllm-kimi-k25:
      image: vllm/vllm-openai:v0.18.1-cu130
      container_name: vllm-kimi-k25
      restart: unless-stopped

      network_mode: host
      ipc: host
      privileged: true
      shm_size: "32gb"

      environment:
        VLLM_USE_FLASHINFER_MOE_FP4: "1"
        CUDA_DEVICE_MAX_CONNECTIONS: "1"
        NCCL_DEBUG: "WARN"
        NCCL_IB_DISABLE: "0"
        NCCL_P2P_DISABLE: "0"
        TORCH_NCCL_AVOID_RECORD_STREAMS: "1"

      volumes:
        - /home/models/Kimi-K2.5-NVFP4:/models/Kimi-K2.5-NVFP4:ro
        - ./vllm-cache:/root/.cache/vllm

      ulimits:
        memlock: -1
        stack: 67108864

      command: >
        vllm serve /models/Kimi-K2.5-NVFP4
        --trust-remote-code
        --tensor-parallel-size 4
        --data-parallel-size 2
        --enable-expert-parallel
        --enable-ep-weight-filter
        --mm-encoder-tp-mode data
        --compilation_config.pass_config.fuse_allreduce_rms true
        --max-model-len 131072
        --gpu-memory-utilization 0.95
        --enable-chunked-prefill
        --max-num-batched-tokens 32768
        --max-num-seqs 8
        --enable-prefix-caching
        --kv-cache-dtype fp8_e4m3
        --kv-offloading-size 512
        --kv-offloading-backend native
        --tool-call-parser kimi_k2
        --reasoning-parser kimi_k2
        --enable-auto-tool-choice
        --served-model-name kimi-k2.5
        --port 8000

      healthcheck:
        test: ["CMD-SHELL", "curl -fsS http://127.0.0.1:8000/health || exit 1"]
        interval: 30s
        timeout: 10s
        retries: 10
        start_period: 300s

      logging:
        driver: json-file
        options:
          max-size: "100m"
          max-file: "5"


```



```
export VLLM_USE_FLASHINFER_MOE_FP4=1

vllm serve moonshotai/Kimi-K2.5-NVFP4 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --mm-encoder-tp-mode data \
    --compilation_config.pass_config.fuse_allreduce_rms true \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_e5m2 \
    --kv-offloading-size 512 \
    --kv-offloading-backend native \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --served-model-name kimi-k2.5

```



```bash
  export VLLM_USE_FLASHINFER_MOE_FP4=1

  vllm serve moonshotai/Kimi-K2.5-NVFP4 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --mm-encoder-tp-mode data \
    --compilation_config.pass_config.fuse_allreduce_rms true \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_e5m2 \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --served-model-name kimi-k2.5

```





Как можно заранить компоуз с glm-5
```docker-compose
version: "3.9"

services:
  sglang-glm5:
    image: lmsysorg/sglang:glm5-hopper
    container_name: sglang-glm5
    restart: unless-stopped

    network_mode: host
    ipc: host
    privileged: true
    shm_size: "32gb"

    environment:
      CUDA_DEVICE_MAX_CONNECTIONS: "1"
      NCCL_DEBUG: "WARN"
      NCCL_IB_DISABLE: "0"
      NCCL_P2P_DISABLE: "0"
      TORCH_NCCL_AVOID_RECORD_STREAMS: "1"

    volumes:
      - /home/models/GLM-5-FP8:/home/models/GLM-5-FP8:ro
      - ./logs:/logs

    ulimits:
      memlock: -1
      stack: 67108864

    command: >
      python3 -m sglang.launch_server
      --model-path /home/models/GLM-5-FP8
      --served-model-name glm
      --host 0.0.0.0
      --port 30000
      --tp-size 8
      --attention-backend fa3
      --tool-call-parser glm47
      --reasoning-parser glm45
      --speculative-algorithm EAGLE
      --speculative-num-steps 3
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 4
      --mem-fraction-static 0.80
      --schedule-conservativeness 0.3
      --chunked-prefill-size 4096
      --max-running-requests 512
      --page-size 1
      --enable-hierarchical-cache
      --hicache-size 1536
      --hicache-mem-layout page_first_direct
      --hicache-io-backend direct
      --hicache-write-policy write_through
      --enable-cache-report

    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://127.0.0.1:30000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 10
      start_period: 180s

    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "5"
```



```
version: "3.9"

services:
  sglang-glm5:
    image: lmsysorg/sglang:glm5-hopper
    container_name: sglang-glm5
    restart: unless-stopped

    network_mode: host
    ipc: host
    privileged: true
    shm_size: "32gb"

    environment:
      CUDA_DEVICE_MAX_CONNECTIONS: "1"
      NCCL_DEBUG: "WARN"
      NCCL_IB_DISABLE: "0"
      NCCL_P2P_DISABLE: "0"
      TORCH_NCCL_AVOID_RECORD_STREAMS: "1"

    volumes:
      - /home/models/GLM-5-FP8:/home/models/GLM-5-FP8:ro
      - ./logs:/logs

    ulimits:
      memlock: -1
      stack: 67108864

    command: >
      python3 -m sglang.launch_server
      --model-path /home/models/GLM-5-FP8
      --served-model-name glm
      --host 0.0.0.0
      --port 30000
      --tp-size 8
      --attention-backend fa3
      --tool-call-parser glm47
      --reasoning-parser glm45
      --speculative-algorithm EAGLE
      --speculative-num-steps 3
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 4
      --mem-fraction-static 0.80
      --schedule-conservativeness 0.3
      --chunked-prefill-size 4096
      --max-running-requests 512
      --page-size 64
      --enable-hierarchical-cache
      --hicache-size 1536
      --hicache-mem-layout page_first_direct
      --hicache-io-backend direct
      --hicache-write-policy write_through
      --enable-cache-report

    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://127.0.0.1:30000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 10
      start_period: 180s

    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "5"

```




  Фаза 1: Стабильный запуск с FP8

  python -m sglang.launch_server \
    --model-path <path-to-glm5> \
    --tp 8 \
    --attention-backend nsa \
    --page-size 64 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 4096 \
    --cuda-graph-max-bs 64 \
    --enable-hierarchical-cache \
    --hicache-size 400 \
    --hicache-write-policy write_through \
    --hicache-io-backend kernel \
    --max-running-requests 48




Что НЕ указываем (и почему):
  - --nsa-prefill-backend / --nsa-decode-backend — не указываем явно. При fp8_e4m3 автодетект сам выберет flashmla_auto / flashmla_kv для Hopper. Явное указание может конфликтовать с
  будущими оптимизациями
  - --context-length — оставляем 200k по умолчанию. FP8 даёт достаточно GPU KV для одного полного контекста

  Фаза 2: Если OOM — пошаговая эскалация

  Шаг 1: --cuda-graph-max-bs 32          # ещё меньше графов
  Шаг 2: --mem-fraction-static 0.82      # больше headroom
  Шаг 3: --disable-cuda-graph            # убираем графы совсем
  Шаг 4: --chunked-prefill-size 2048     # минимальные activations

  Фаза 3: После стабильного запуска — тюнинг производительности

  # Постепенно увеличиваем:
  --cuda-graph-max-bs 128    # больше графов → лучше decode latency
  --mem-fraction-static 0.87 # больше KV cache
  --chunked-prefill-size 8192 # быстрее prefill

  Что даёт FP8 в конкретных цифрах для вашего сетапа

  При mem_fraction_static=0.85, ~24 GB/GPU на KV:


```





```
export VLLM_USE_FLASHINFER_MOE_FP4=1

vllm serve moonshotai/Kimi-K2.5-NVFP4 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --mm-encoder-tp-mode data \
    --compilation_config.pass_config.fuse_allreduce_rms true \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_e5m2 \
    --kv-offloading-size 512 \
    --kv-offloading-backend native \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --served-model-name kimi-k2.5

```



```bash
  export VLLM_USE_FLASHINFER_MOE_FP4=1

  vllm serve moonshotai/Kimi-K2.5-NVFP4 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --data-parallel-size 2 \
    --enable-expert-parallel \
    --enable-ep-weight-filter \
    --mm-encoder-tp-mode data \
    --compilation_config.pass_config.fuse_allreduce_rms true \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_e5m2 \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --served-model-name kimi-k2.5

```





Как можно заранить компоуз с glm-5
```docker-compose
version: "3.9"

services:
  sglang-glm5:
    image: lmsysorg/sglang:glm5-hopper
    container_name: sglang-glm5
    restart: unless-stopped

    network_mode: host
    ipc: host
    privileged: true
    shm_size: "32gb"

    environment:
      CUDA_DEVICE_MAX_CONNECTIONS: "1"
      NCCL_DEBUG: "WARN"
      NCCL_IB_DISABLE: "0"
      NCCL_P2P_DISABLE: "0"
      TORCH_NCCL_AVOID_RECORD_STREAMS: "1"

    volumes:
      - /home/models/GLM-5-FP8:/home/models/GLM-5-FP8:ro
      - ./logs:/logs

    ulimits:
      memlock: -1
      stack: 67108864

    command: >
      python3 -m sglang.launch_server
      --model-path /home/models/GLM-5-FP8
      --served-model-name glm
      --host 0.0.0.0
      --port 30000
      --tp-size 8
      --attention-backend fa3
      --tool-call-parser glm47
      --reasoning-parser glm45
      --speculative-algorithm EAGLE
      --speculative-num-steps 3
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 4
      --mem-fraction-static 0.80
      --schedule-conservativeness 0.3
      --chunked-prefill-size 4096
      --max-running-requests 512
      --page-size 1
      --enable-hierarchical-cache
      --hicache-size 1536
      --hicache-mem-layout page_first_direct
      --hicache-io-backend direct
      --hicache-write-policy write_through
      --enable-cache-report

    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://127.0.0.1:30000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 10
      start_period: 180s

    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "5"
```



```
version: "3.9"

services:
  sglang-glm5:
    image: lmsysorg/sglang:glm5-hopper
    container_name: sglang-glm5
    restart: unless-stopped

    network_mode: host
    ipc: host
    privileged: true
    shm_size: "32gb"

    environment:
      CUDA_DEVICE_MAX_CONNECTIONS: "1"
      NCCL_DEBUG: "WARN"
      NCCL_IB_DISABLE: "0"
      NCCL_P2P_DISABLE: "0"
      TORCH_NCCL_AVOID_RECORD_STREAMS: "1"

    volumes:
      - /home/models/GLM-5-FP8:/home/models/GLM-5-FP8:ro
      - ./logs:/logs

    ulimits:
      memlock: -1
      stack: 67108864

    command: >
      python3 -m sglang.launch_server
      --model-path /home/models/GLM-5-FP8
      --served-model-name glm
      --host 0.0.0.0
      --port 30000
      --tp-size 8
      --attention-backend fa3
      --tool-call-parser glm47
      --reasoning-parser glm45
      --speculative-algorithm EAGLE
      --speculative-num-steps 3
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 4
      --mem-fraction-static 0.80
      --schedule-conservativeness 0.3
      --chunked-prefill-size 4096
      --max-running-requests 512
      --page-size 64
      --enable-hierarchical-cache
      --hicache-size 1536
      --hicache-mem-layout page_first_direct
      --hicache-io-backend direct
      --hicache-write-policy write_through
      --enable-cache-report

    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://127.0.0.1:30000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 10
      start_period: 180s

    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "5"

```




  Фаза 1: Стабильный запуск с FP8

  python -m sglang.launch_server \
    --model-path <path-to-glm5> \
    --tp 8 \
    --attention-backend nsa \
    --page-size 64 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 4096 \
    --cuda-graph-max-bs 64 \
    --enable-hierarchical-cache \
    --hicache-size 400 \
    --hicache-write-policy write_through \
    --hicache-io-backend kernel \
    --max-running-requests 48




Что НЕ указываем (и почему):
  - --nsa-prefill-backend / --nsa-decode-backend — не указываем явно. При fp8_e4m3 автодетект сам выберет flashmla_auto / flashmla_kv для Hopper. Явное указание может конфликтовать с
  будущими оптимизациями
  - --context-length — оставляем 200k по умолчанию. FP8 даёт достаточно GPU KV для одного полного контекста

  Фаза 2: Если OOM — пошаговая эскалация

  Шаг 1: --cuda-graph-max-bs 32          # ещё меньше графов
  Шаг 2: --mem-fraction-static 0.82      # больше headroom
  Шаг 3: --disable-cuda-graph            # убираем графы совсем
  Шаг 4: --chunked-prefill-size 2048     # минимальные activations

  Фаза 3: После стабильного запуска — тюнинг производительности

  # Постепенно увеличиваем:
  --cuda-graph-max-bs 128    # больше графов → лучше decode latency
  --mem-fraction-static 0.87 # больше KV cache
  --chunked-prefill-size 8192 # быстрее prefill

  Что даёт FP8 в конкретных цифрах для вашего сетапа

  При mem_fraction_static=0.85, ~24 GB/GPU на KV:
