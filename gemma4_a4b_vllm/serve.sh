#!/usr/bin/env bash
set -uo pipefail

cleanup_offload_mmaps() {
  local path
  shopt -s nullglob
  for path in /dev/shm/vllm_offload_*.mmap; do
    if [[ -f "$path" && ! -L "$path" ]]; then
      rm -f -- "$path"
      printf 'removed private stale offload mmap: %s\n' "$path" >&2
    fi
  done
  shopt -u nullglob
}

cleanup_offload_mmaps
vllm serve "$@" &
server_pid=$!

forward_term() {
  kill -TERM "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}

trap forward_term INT TERM
wait "$server_pid"
status=$?
trap - INT TERM
cleanup_offload_mmaps
exit "$status"
