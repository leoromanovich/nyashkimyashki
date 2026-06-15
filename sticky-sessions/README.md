# Sticky sessions: LiteLLM -> SMG -> SGLang DPA

Схема:

```text
client -> LiteLLM :4000 -> SMG :30000 -> SGLang worker :30000 internal
```

LiteLLM сам выставляет `X-SMG-Routing-Key`; пользователям его задавать не нужно.

## Запуск

```bash
cd /Users/leo/Projects/opensource/llm_engines/sglang/sticky-sessions
docker compose up -d
```

Порты:

- LiteLLM OpenAI API: `http://127.0.0.1:4000/v1`
- SMG OpenAI API: `http://127.0.0.1:30000/v1`
- SMG Prometheus: `http://127.0.0.1:29000/metrics`

## Проверка

Два запроса с одним `x-litellm-session-id` должны идти в один sticky bucket:

```bash
for i in 1 2 3; do
  curl -s http://127.0.0.1:4000/v1/chat/completions \
    -H 'Authorization: Bearer sk-sticky-test' \
    -H 'Content-Type: application/json' \
    -H 'x-litellm-session-id: sticky-session-a' \
    -H 'x-litellm-agent-id: test-agent' \
    -H 'x-repo-id: test/repo' \
    -d '{"model":"glm","messages":[{"role":"user","content":"Reply with one word: ok"}],"max_tokens":4,"temperature":0}'
  echo
done
```

Метрики SMG:

```bash
curl -s http://127.0.0.1:29000/metrics | grep 'smg_manual_policy'
```

Ожидание: первый запрос даст assignment, повторные - hit. Если растёт `no_routing_id`, LiteLLM не донёс `X-SMG-Routing-Key`.

## Переменные

- `LITELLM_MASTER_KEY`, default `sk-sticky-test`
- `LITELLM_PORT`, default `4000`
- `SMG_PORT`, default `30000`
- `SMG_PROMETHEUS_PORT`, default `29000`
- `SMG_ASSIGNMENT_MODE`, default `min_group`
- `SMG_MAX_IDLE_SECS`, default `14400`
- `SGLANG_HICACHE_SIZE`, default `150`
