#!/usr/bin/env bash
set -euo pipefail

test -d /mooncake_cache
test -w /mooncake_cache

python3 - <<'PY'
import socket, time

for host, port in [("172.29.0.2", 50051), ("172.29.0.2", 8080)]:
    for _ in range(120):
        try:
            with socket.create_connection((host, port), timeout=1):
                break
        except OSError:
            time.sleep(1)
    else:
        raise SystemExit(f"{host}:{port} is not ready")
PY

export MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=$((MOONCAKE_OFFLOAD_GB * 1024 * 1024 * 1024))
export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE="${MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES}"

echo "Starting Mooncake standalone real client"
echo "  protocol=${MOONCAKE_PROTOCOL}"
echo "  global_segment_size=${MOONCAKE_GLOBAL_SEGMENT_SIZE}"
echo "  offload_path=${MOONCAKE_OFFLOAD_FILE_STORAGE_PATH}"
echo "  offload_limit_bytes=${MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES}"

CLIENT_ARGS=(
  mooncake_client
  --master_server_address=172.29.0.2:50051
  --metadata_server=http://172.29.0.2:8080/metadata
  --host=172.29.0.3
  --port=50052
  --protocol="${MOONCAKE_PROTOCOL}"
  --global_segment_size="${MOONCAKE_GLOBAL_SEGMENT_SIZE}"
  --enable_offload=true
  --threads="${MOONCAKE_THREADS}"
)

if [ -n "${MOONCAKE_DEVICE:-}" ]; then
  CLIENT_ARGS+=(--device_names="${MOONCAKE_DEVICE}")
fi

exec numactl --interleave=all "${CLIENT_ARGS[@]}"
