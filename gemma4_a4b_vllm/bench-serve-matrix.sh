#!/usr/bin/env bash
set -uo pipefail

results_root="${BENCH_RESULTS_ROOT:-/bench-results}"
mode="${BENCH_MODE:-unknown}"
output_len="${BENCH_OUTPUT_LEN:-1024}"
requests_per_user="${BENCH_REQUESTS_PER_USER:-2}"
warmups="${BENCH_NUM_WARMUPS:-1}"
read -r -a concurrencies <<<"${BENCH_CONCURRENCIES:-2 4 8 16 32}"
read -r -a input_lengths <<<"${BENCH_INPUT_LENGTHS:-8192 16384 32768}"

for value in "$output_len" "$requests_per_user" "$warmups" "${concurrencies[@]}" "${input_lengths[@]}"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    printf 'benchmark values must be non-negative integers: %s\n' "$value" >&2
    exit 2
  fi
done
if (( output_len < 1 || requests_per_user < 1 )); then
  printf 'BENCH_OUTPUT_LEN and BENCH_REQUESTS_PER_USER must be positive\n' >&2
  exit 2
fi
if [[ -z "${VLLM_API_KEY:-}" || -z "${VLLM_SERVED_MODEL:-}" ]]; then
  printf 'VLLM_API_KEY and VLLM_SERVED_MODEL must be set in the container\n' >&2
  exit 2
fi

export OPENAI_API_KEY="$VLLM_API_KEY"
run_id="${BENCH_RUN_ID:-${mode}-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'BENCH_RUN_ID contains unsupported characters: %s\n' "$run_id" >&2
  exit 2
fi
run_dir="$results_root/$run_id"
mkdir -p "$run_dir"

failures=0
for input_len in "${input_lengths[@]}"; do
  for concurrency in "${concurrencies[@]}"; do
    num_prompts=$((concurrency * requests_per_user))
    stem="c${concurrency}-in${input_len}-out${output_len}"
    printf '\n=== %s: concurrency=%s input=%s output=%s requests=%s ===\n' \
      "$mode" "$concurrency" "$input_len" "$output_len" "$num_prompts"
    command=(
      vllm bench serve
      --backend vllm
      --host 127.0.0.1
      --port 8000
      --endpoint /v1/completions
      --model /models
      --served-model-name "$VLLM_SERVED_MODEL"
      --tokenizer /models
      --dataset-name random
      --random-input-len "$input_len"
      --random-output-len "$output_len"
      --random-range-ratio 0
      --num-prompts "$num_prompts"
      --num-warmups "$warmups"
      --request-rate inf
      --max-concurrency "$concurrency"
      --temperature 0
      --ignore-eos
      --disable-tqdm
      --percentile-metrics ttft,tpot,itl,e2el
      --metric-percentiles 50,95,99
      --save-result
      --result-dir "$run_dir"
      --result-filename "$stem.json"
      --metadata
      "mode=$mode"
      "concurrency=$concurrency"
      "input_tokens=$input_len"
      "output_tokens=$output_len"
      "requests_per_user=$requests_per_user"
    )
    if "${command[@]}" 2>&1 | tee "$run_dir/$stem.log"; then
      printf 'completed: %s\n' "$stem"
    else
      status=$?
      printf '%s\n' "$status" >"$run_dir/$stem.failed"
      failures=$((failures + 1))
      printf 'failed: %s, exit=%s\n' "$stem" "$status" >&2
    fi
  done
done

python3 /opt/gemma4/summarize-bench.py "$run_dir"
summary_status=$?
if (( failures > 0 || summary_status != 0 )); then
  printf 'matrix finished with %s command failures; results: %s\n' \
    "$failures" "$run_dir" >&2
  exit 1
fi
printf 'matrix complete: %s\n' "$run_dir"
