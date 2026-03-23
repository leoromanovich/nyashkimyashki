

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
