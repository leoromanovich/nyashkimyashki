docker run --gpus all \
  --privileged --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/models/google:/models \
  vllm/vllm-openai:v0.22.1 --model /models/gemma-4-31B-it-qat-w4a16-ct \
  --tensor-parallel-size 1 \
  --max-model-len auto \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8_e4m3 \
  --kv-cache-dtype-skip-layers sliding_window \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --chat-template examples/tool_chat_template_gemma4.jinja \
  --reasoning-parser gemma4 \
  --speculative-config '{"model":"/models/gemma-4-31B-it-assistant","num_speculative_tokens":4}'
