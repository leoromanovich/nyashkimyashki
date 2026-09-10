#!/usr/bin/env bash
set -euo pipefail

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${1:-$recipe_dir/.env}"

if [[ ! -r "$env_file" ]]; then
  printf 'missing readable env file: %s\n' "$env_file" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

for command_name in docker nvidia-smi python3; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'missing command: %s\n' "$command_name" >&2
    exit 2
  fi
done
docker compose version >/dev/null

model_dir="${MODEL_DIR:?set MODEL_DIR}"
if [[ ! -d "$model_dir" || ! -r "$model_dir" ]]; then
  printf 'model directory must exist and be readable: %s\n' "$model_dir" >&2
  exit 2
fi

model_shards="$(python3 - "$model_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit("missing model files: " + ", ".join(missing))

config = json.loads((root / "config.json").read_text())
if config.get("model_type") != "diffusion_gemma":
    raise SystemExit("config.json model_type must be diffusion_gemma")
quant = config.get("quantization_config", {})
if quant.get("quant_method") != "compressed-tensors" or quant.get("format") != "int-quantized":
    raise SystemExit("config.json must describe compressed-tensors INT8")

index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(index.get("weight_map", {}).values()))
missing_shards = [name for name in shards if not (root / name).is_file()]
if not shards or missing_shards:
    raise SystemExit("missing safetensors shards: " + ", ".join(missing_shards))
print(len(shards))
PY
)"

gpu_row="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits -i "${GPU_DEVICE_ID:-0}" | head -n 1)"
IFS=',' read -r gpu_name gpu_memory_mib driver_version <<<"$gpu_row"
gpu_name="${gpu_name# }"
gpu_memory_mib="${gpu_memory_mib// /}"
driver_version="${driver_version// /}"

if [[ "$gpu_name" != *"${EXPECTED_GPU_NAME:-A100}"* ]]; then
  printf 'GPU %s is outside this profile; expected name containing %s\n' \
    "$gpu_name" "${EXPECTED_GPU_NAME:-A100}" >&2
  exit 2
fi
if (( gpu_memory_mib < ${MIN_GPU_MEMORY_MIB:-79000} )); then
  printf 'GPU memory is too small: %s MiB\n' "$gpu_memory_mib" >&2
  exit 2
fi

version_ge() {
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" == "$2" ]]
}

if ! version_ge "$driver_version" "525.60.13"; then
  printf 'driver %s is below the CUDA 12 baseline 525.60.13\n' "$driver_version" >&2
  exit 2
fi
if ! version_ge "$driver_version" "575.51.03"; then
  if [[ "${VLLM_ENABLE_CUDA_COMPATIBILITY:-0}" != "1" ]]; then
    printf 'driver %s requires VLLM_ENABLE_CUDA_COMPATIBILITY=1 for the cu129 image\n' "$driver_version" >&2
    exit 2
  fi
  printf 'warning: driver %s uses CUDA forward compatibility\n' "$driver_version" >&2
fi

kv_bytes="$(python3 - "${KV_CPU_GIB:?set KV_CPU_GIB}" "${KV_TRANSFER_CONFIG:?set KV_TRANSFER_CONFIG}" <<'PY'
import json
import sys

try:
    gib = float(sys.argv[1])
except ValueError as error:
    raise SystemExit("KV_CPU_GIB must be a positive number") from error
if gib <= 0:
    raise SystemExit("KV_CPU_GIB must be a positive number")

config = json.loads(sys.argv[2])
extra = config.get("kv_connector_extra_config", {})
if config.get("kv_connector") != "OffloadingConnector":
    raise SystemExit("KV_TRANSFER_CONFIG must use OffloadingConnector")
if extra.get("spec_name") != "TieringOffloadingSpec":
    raise SystemExit("KV_TRANSFER_CONFIG must use TieringOffloadingSpec")
print(int(gib * 1024**3))
PY
)"
if [[ ! "${KV_CPU_GIB}" =~ ^[0-9]+$ ]]; then
  printf 'KV_CPU_GIB must be a positive integer\n' >&2
  exit 2
fi
if [[ ! "${VLLM_SHM_GIB:?set VLLM_SHM_GIB}" =~ ^[0-9]+$ ]]; then
  printf 'VLLM_SHM_GIB must be a positive integer\n' >&2
  exit 2
fi
minimum_shm_gib="$((KV_CPU_GIB + 16))"
if (( VLLM_SHM_GIB < minimum_shm_gib )); then
  printf 'private /dev/shm is too small: configured=%s GiB, required>=%s GiB\n' \
    "$VLLM_SHM_GIB" "$minimum_shm_gib" >&2
  exit 2
fi
available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
required_kib="$((kv_bytes / 1024 + 32 * 1024 * 1024))"
if (( available_kib < required_kib )); then
  printf 'host RAM headroom is too small: available=%s GiB, required=%s GiB\n' \
    "$((available_kib / 1024 / 1024))" "$((required_kib / 1024 / 1024))" >&2
  exit 2
fi

for cache_dir in "${VLLM_CACHE_DIR:?set VLLM_CACHE_DIR}" "${KV_CACHE_DIR:?set KV_CACHE_DIR}"; do
  if [[ ! -d "$cache_dir" || ! -w "$cache_dir" ]]; then
    printf 'cache directory must exist and be writable: %s\n' "$cache_dir" >&2
    exit 2
  fi
done

kv_disk_free_kib="$(df -Pk "$KV_CACHE_DIR" | awk 'NR == 2 {print $4}')"
kv_disk_required_kib="$((${KV_DISK_MIN_FREE_GB:-512} * 1024 * 1024))"
if (( kv_disk_free_kib < kv_disk_required_kib )); then
  printf 'KV disk needs at least %s GiB free; found %s GiB\n' \
    "${KV_DISK_MIN_FREE_GB:-512}" "$((kv_disk_free_kib / 1024 / 1024))" >&2
  exit 2
fi

printf 'ok: %s, %s MiB, driver %s, model shards %s, RAM KV %s GiB, private shm %s GiB, disk KV >= %s GiB free\n' \
  "$gpu_name" "$gpu_memory_mib" "$driver_version" \
  "$model_shards" "${KV_CPU_GIB}" "$VLLM_SHM_GIB" \
  "${KV_DISK_MIN_FREE_GB:-512}"
