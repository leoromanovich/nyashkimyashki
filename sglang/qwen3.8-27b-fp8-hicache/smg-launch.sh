#!/usr/bin/env sh
set -eu
if [ "${ENABLE_TRACING:-0}" = 1 ]; then
  set -- "$@" --enable-trace --otlp-traces-endpoint "${OTLP_TRACES_ENDPOINT:-otel-collector:4317}"
fi
exec smg "$@"
