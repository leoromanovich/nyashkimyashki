# Sticky sessions: LiteLLM -> SMG -> SGLang DPA

База: Qwen3-8B-FP8, `tp=2`, `dp=2`, `--enable-dp-attention`.

```text
client -> LiteLLM :4000 -> SMG :30000 -> SGLang worker :30000 internal
```

LiteLLM сам выставляет `X-SMG-Routing-Key`; пользователям его задавать не нужно.

Важно: для sticky по `X-SMG-Routing-Key` у SMG стоит `--policy=manual`. `cache_aware` маршрутизирует по prefix/cache locality, а не по session header.

## Запуск

```bash
cd /Users/leo/Projects/opensource/llm_engines/nyashkimyashki/sticky-sessions
docker compose up -d
```

Порты:

- LiteLLM OpenAI API: `http://127.0.0.1:4000/v1`
- SMG OpenAI API: `http://127.0.0.1:30000/v1`
- SGLang worker direct debug: `http://127.0.0.1:30001/v1`
- SMG Prometheus: `http://127.0.0.1:29000/metrics`

## Проверка

```bash
for i in 1 2 3; do
  curl -s http://127.0.0.1:4000/v1/chat/completions \
    -H 'Authorization: Bearer sk-sticky-test' \
    -H 'Content-Type: application/json' \
    -H 'x-litellm-session-id: sticky-session-a' \
    -H 'x-litellm-agent-id: test-agent' \
    -H 'x-repo-id: test/repo' \
    -d '{"model":"qwen3-8b","messages":[{"role":"user","content":"Reply with one word: ok"}],"max_tokens":4,"temperature":0}'
  echo
done
```

Метрики:

```bash
curl -s http://127.0.0.1:29000/metrics | grep 'smg_manual_policy'
```

Ожидание: первый запрос создаёт assignment, повторы дают hit. Если растёт `no_routing_id`, LiteLLM не донёс `X-SMG-Routing-Key`.

## Переменные

- `LITELLM_MASTER_KEY`, default `sk-sticky-test`
- `LITELLM_PORT`, default `4000`
- `LITELLM_UPSTREAM_MODEL`, default `openai/qwen3-8b`
- `SMG_PROMETHEUS_PORT`, default `29000`
- `SMG_ASSIGNMENT_MODE`, default `min_group`
- `SMG_MAX_IDLE_SECS`, default `14400`
