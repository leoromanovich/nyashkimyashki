#!/usr/bin/env bash
set -euo pipefail
model_dir="${MODEL_DIR:-/data/scratch/qwen38-sglang-smg/models/Qwen3.8-27B-FP8}"
exec "${HF_CLI:-hf}" download Qwen/Qwen3.8-27B-FP8 \
  --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a --local-dir "$model_dir" "$@"
