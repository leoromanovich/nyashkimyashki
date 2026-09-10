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

results_dir="${BENCH_RESULTS_DIR:-$recipe_dir/results}"
if [[ "$results_dir" != /* ]]; then
  results_dir="$recipe_dir/${results_dir#./}"
fi
mkdir -p "$results_dir"
chmod 700 "$results_dir"

compose=(
  docker compose
  --project-directory "$recipe_dir"
  --file "$recipe_dir/docker-compose.yaml"
  --env-file "$env_file"
)
if [[ "$mode" == "mtp" ]]; then
  compose+=(--profile mtp)
  service=gemma4-mtp
else
  service=gemma4
fi

if ! "${compose[@]}" ps --status running --services | grep -Fx "$service" >/dev/null; then
  printf 'service is not running: %s\n' "$service" >&2
  exit 2
fi

exec "${compose[@]}" exec -T -e "BENCH_MODE=$mode" \
  "$service" /opt/gemma4/bench-serve-matrix.sh
