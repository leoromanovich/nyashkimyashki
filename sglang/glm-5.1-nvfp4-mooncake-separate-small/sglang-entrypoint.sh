#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import socket, time

for host, port in [("172.29.0.3", 50052)]:
    for _ in range(180):
        try:
            with socket.create_connection((host, port), timeout=1):
                break
        except OSError:
            time.sleep(1)
    else:
        raise SystemExit(f"{host}:{port} is not ready")
PY

echo "Starting SGLang with Mooncake Dummy Client -> 172.29.0.3:50052"

exec python3 -m sglang.launch_server \
  --enable-trace \
  --otlp-traces-endpoint "${OTLP_TRACES_ENDPOINT}" \
  --api-key "${SGLANG_API_KEY:?set key}" \
  --model-path=/models/GLM-5.1-NVFP4 \
  --quantization=modelopt_fp4 \
  --served-model-name=glm \
  --host=0.0.0.0 \
  --port=30000 \
  --admin-api-key "${SGLANG_ADMIN_API_KEY:?set key}" \
  --attention-backend=nsa \
  --nsa-prefill-backend=trtllm \
  --nsa-decode-backend=trtllm \
  --moe-runner-backend=flashinfer_trtllm \
  --enable-flashinfer-allreduce-fusion \
  --context-length=202752 \
  --reasoning-parser=glm45 \
  --tool-call-parser=glm47 \
  --trust-remote-code \
  --kv-cache-dtype=fp8_e4m3 \
  --enable-hierarchical-cache \
  --hicache-io-backend=kernel \
  --hicache-mem-layout=page_first \
  --hicache-size=100 \
  --hicache-write-policy=write_through_selective \
  --hicache-storage-prefetch-policy=timeout \
  --hicache-storage-backend=mooncake \
  --hicache-storage-backend-extra-config='"'"'{"standalone_storage":true,"client_server_address":"172.29.0.3:50052"}'"'"' \
  --page-size=64 \
  --chunked-prefill-size=65536 \
  --max-prefill-tokens=32768 \
  --schedule-conservativeness=3.3333333333 \
  --cuda-graph-max-bs=18 \
  --max-running-requests=128 \
  --tp=8 \
  --dp=8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --load-balance-method=total_tokens \
  --schedule-policy=lpm \
  --mem-fraction-static=0.85 \
  --enable-metrics \
  --enable-custom-logit-processor \
  --enable-cache-report \
  --speculative-algorithm=EAGLE \
  --speculative-num-steps=3 \
  --speculative-eagle-topk=1 \
  --speculative-num-draft-tokens=4 \
  --json-model-override-args='{"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}' \
  --model-loader-extra-config='{"enable_multithread_load":true,"num_threads":128}'
