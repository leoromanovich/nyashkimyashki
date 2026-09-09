#!/usr/bin/env bash
set -euo pipefail
args=("$@")
if [[ "${ENABLE_MTP:-0}" == 1 ]]; then
  args+=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
  )
  if [[ "${MTP_REPLAY_SSM:-0}" == 1 ]]; then
    args+=(--enable-linear-replayssm-spec)
  fi
fi
if [[ "${ENABLE_TRACING:-0}" == 1 ]]; then
  args+=(--enable-trace --otlp-traces-endpoint "${OTLP_TRACES_ENDPOINT:-otel-collector:4317}")
fi
exec python3 -m sglang.launch_server "${args[@]}"
