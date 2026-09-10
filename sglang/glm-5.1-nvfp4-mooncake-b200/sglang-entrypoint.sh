#!/usr/bin/env bash
set -euo pipefail

export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/mooncake_cache
export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend
export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$(python3 - <<'PY'
import os
print(int(os.environ.get("MOONCAKE_OFFLOAD_GB", "100")) * 1000000000)
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
  --enable-trace \
  --otlp-traces-endpoint "${OTLP_TRACES_ENDPOINT}" \
  --api-key "${SGLANG_API_KEY:?set key}" \
  --model-path /models/GLM-5.1-NVFP4 \
  --quantization modelopt_fp4 \
  --served-model-name glm \
  --host 0.0.0.0 \
  --port 30000 \
  --admin-api-key "${SGLANG_ADMIN_API_KEY:?set key}" \
  --attention-backend nsa \
  --nsa-prefill-backend trtllm \
  --nsa-decode-backend trtllm \
  --moe-runner-backend flashinfer_trtllm \
  --enable-flashinfer-allreduce-fusion \
  --context-length 202752 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 \
  --enable-hierarchical-cache \
  --hicache-io-backend kernel \
  --hicache-size 150 \
  --hicache-write-policy write_through_selective \
  --hicache-storage-prefetch-policy timeout \
  --hicache-storage-backend mooncake \
  --file-storage-path /mooncake_cache \
  --hicache-storage-backend-extra-config '{"local_hostname":"sglang","metadata_server":"http://mooncake_master:8080/metadata","master_server_address":"mooncake_master:50051","protocol":"tcp","device_name":"","global_segment_size":"64gb","check_server":false,"enable_ssd_offload":true,"ssd_offload_path":"/mooncake_cache"}' \
  --page-size 64 \
  --chunked-prefill-size 65536 \
  --max-prefill-tokens 32768 \
  --schedule-conservativeness 3.3333333333 \
  --cuda-graph-max-bs 18 \
  --max-running-requests 128 \
  --tp 8 \
  --dp 8 \
  --enable-dp-attention \
  --enable-flashinfer-allreduce-fusion \
  --enable-dp-lm-head \
  --load-balance-method total_tokens \
  --schedule-policy lpm \
  --mem-fraction-static 0.85 \
  --enable-metrics \
  --enable-custom-logit-processor \
  --enable-cache-report \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --json-model-override-args '{"index_topk_pattern": "FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}' \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}'
