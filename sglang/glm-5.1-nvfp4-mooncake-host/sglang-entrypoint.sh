#!/usr/bin/env bash
set -euo pipefail

export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/mooncake_cache
export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend
export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$(python3 - <<'PY'
import os
print(int(os.environ.get("MOONCAKE_OFFLOAD_GB", "7000")) * 1000000000)
PY
)
export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=$MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES
export MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=lru
export MOONCAKE_OFFLOAD_USE_URING=1

echo "Mooncake SSD offload: $MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES bytes at $MOONCAKE_OFFLOAD_FILE_STORAGE_PATH"

exec python3 -m sglang.launch_server \
  --enable-trace \
  --otlp-traces-endpoint "${OTLP_TRACES_ENDPOINT}" \
  --api-key "${SGLANG_API_KEY:?set key}" \
  --model-path nvidia/GLM-5.1-NVFP4 \
  --served-model-name glm51 \
  --host 127.0.0.1 \
  --port "${ENGINE_PORT}" \
  --tensor-parallel-size 8 \
  --quantization modelopt_fp4 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --trust-remote-code \
  --context-length 131072 \
  --chunked-prefill-size 131072 \
  --mem-fraction-static 0.80 \
  --page-size 64 \
  --enable-cache-report \
  --enable-metrics \
  --schedule-policy lpm \
  --enable-hierarchical-cache \
  --hicache-size 256 \
  --hicache-io-backend direct \
  --hicache-mem-layout page_first_direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend mooncake \
  --hicache-storage-prefetch-policy best_effort \
  --file-storage-path /mooncake_cache \
  --hicache-storage-backend-extra-config '{"local_hostname":"127.0.0.1","metadata_server":"http://127.0.0.1:8080/metadata","master_server_address":"127.0.0.1:50051","protocol":"tcp","device_name":"","global_segment_size":"64gb","check_server":false,"enable_ssd_offload":true,"ssd_offload_path":"/mooncake_cache"}'
