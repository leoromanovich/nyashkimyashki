#!/usr/bin/env bash
set -euo pipefail

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${1:-$recipe_dir/.env}"
mode="${2:-base}"

if [[ "$mode" != "base" && "$mode" != "mtp" ]]; then
  printf 'mode must be base or mtp: %s\n' "$mode" >&2
  exit 2
fi
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
docker compose --env-file "$env_file" --file "$recipe_dir/docker-compose.yaml" \
  --profile mtp config --quiet

model_dir="${MODEL_DIR:?set MODEL_DIR}"
if [[ ! -d "$model_dir" || ! -r "$model_dir" ]]; then
  printf 'model directory must exist and be readable: %s\n' "$model_dir" >&2
  exit 2
fi

model_summary="$(python3 - "$model_dir" <<'PY'
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
text = config.get("text_config", {})
if config.get("model_type") != "gemma4":
    raise SystemExit("config.json model_type must be gemma4")
if "Gemma4ForConditionalGeneration" not in config.get("architectures", []):
    raise SystemExit("config.json must describe Gemma4ForConditionalGeneration")
if not text.get("enable_moe_block"):
    raise SystemExit("target model must enable the Gemma 4 MoE block")
if text.get("num_experts") != 128 or text.get("top_k_experts") != 8:
    raise SystemExit("target model must be the 128-expert top-8 26B-A4B variant")
if config.get("quantization_config"):
    raise SystemExit("default A100 profile requires the unquantized BF16 checkpoint")
if config.get("dtype") not in ("bfloat16", "bf16"):
    raise SystemExit("target model dtype must be bfloat16")

index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(index.get("weight_map", {}).values()))
missing_shards = [name for name in shards if not (root / name).is_file()]
if not shards or missing_shards:
    raise SystemExit("missing safetensors shards: " + ", ".join(missing_shards))
weight_bytes = sum((root / name).stat().st_size for name in shards)
if weight_bytes < 45 * 1024**3:
    raise SystemExit("target BF16 checkpoint is unexpectedly smaller than 45 GiB")
print(f"{len(shards)} shards, {weight_bytes / 1024**3:.1f} GiB")
PY
)"

assistant_summary="disabled"
if [[ "$mode" == "mtp" ]]; then
  assistant_dir="${ASSISTANT_MODEL_DIR:?set ASSISTANT_MODEL_DIR for mtp}"
  if [[ ! -d "$assistant_dir" || ! -r "$assistant_dir" ]]; then
    printf 'assistant model directory must exist and be readable: %s\n' "$assistant_dir" >&2
    exit 2
  fi
  if [[ "$assistant_dir" == "$model_dir" ]]; then
    printf 'assistant and target model directories must differ\n' >&2
    exit 2
  fi
  assistant_summary="$(python3 - "$assistant_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
missing = [name for name in required if not (root / name).is_file()]
if missing:
    raise SystemExit("missing assistant files: " + ", ".join(missing))
config = json.loads((root / "config.json").read_text())
if config.get("model_type") != "gemma4_assistant":
    raise SystemExit("assistant config model_type must be gemma4_assistant")
if (root / "model.safetensors").stat().st_size < 700 * 1024**2:
    raise SystemExit("assistant checkpoint is unexpectedly smaller than 700 MiB")
print(f"{(root / 'model.safetensors').stat().st_size / 1024**2:.0f} MiB")
PY
)"
  if (( ${MTP_NUM_SPECULATIVE_TOKENS:-0} < 1 || ${MTP_NUM_SPECULATIVE_TOKENS:-0} > 4 )); then
    printf 'MTP_NUM_SPECULATIVE_TOKENS must be between 1 and 4\n' >&2
    exit 2
  fi
fi

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
if config.get("kv_role") != "kv_both":
    raise SystemExit("KV_TRANSFER_CONFIG must use kv_both")
if extra.get("spec_name") != "TieringOffloadingSpec":
    raise SystemExit("KV_TRANSFER_CONFIG must use TieringOffloadingSpec")
secondary = extra.get("secondary_tiers", [])
if len(secondary) != 1 or secondary[0].get("type") != "fs":
    raise SystemExit("KV_TRANSFER_CONFIG must contain one filesystem tier")
if secondary[0].get("root_dir") != "/kv-cache":
    raise SystemExit("filesystem tier must use /kv-cache")
print(int(gib * 1024**3))
PY
)"
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

printf 'ok: mode=%s, %s, %s MiB, driver %s, target=%s, assistant=%s, RAM KV=%s GiB, disk KV>=%s GiB\n' \
  "$mode" "$gpu_name" "$gpu_memory_mib" "$driver_version" "$model_summary" \
  "$assistant_summary" "${KV_CPU_GIB}" "${KV_DISK_MIN_FREE_GB:-512}"
