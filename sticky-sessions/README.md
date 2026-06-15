# Sticky sessions: LiteLLM -> SMG -> SGLang DPA

База: Qwen3-8B-FP8, `tp=2`, `dp=2`, `--enable-dp-attention`.

```text
client -> Open WebUI :8080 -> Open WebUI /api/chat/completions
       -> LiteLLM :4000 internal -> SMG :30000 internal -> SGLang worker internal
```

LiteLLM сам выставляет `X-SMG-Routing-Key`; пользователям его задавать не нужно.

Важно: для sticky по `X-SMG-Routing-Key` у SMG стоит `--policy=manual`. `cache_aware` маршрутизирует по prefix/cache locality, а не по session header.

## Open WebUI

Compose поднимает Open WebUI и задаёт базовый OpenAI upstream на LiteLLM:

- URL: `http://litellm:4000/v1`
- Auth: `Bearer`
- Key: значение `LITELLM_MASTER_KEY`

Sticky headers задаются автоматически через `ENABLE_FORWARD_USER_INFO_HEADERS=true`.

Open WebUI отправляет в LiteLLM `X-OpenWebUI-User-*` и `X-OpenWebUI-Chat-Id`. LiteLLM hook читает эти headers и отправляет в SMG только hashed `X-SMG-Routing-Key`.

Обычный UI Open WebUI ходит через `/api/chat/completions`. В этом варианте без edge-proxy `/openai/chat/completions` не блокируется на сетевом уровне; не отдавай этот endpoint внешним клиентам или режь его существующим nginx/ingress, если нужен жёсткий `/api`-only.

LiteLLM и SMG inference-порт не опубликованы на host. Наружу открыт только Open WebUI и SMG metrics.

Если в Open WebUI есть другие external OpenAI backends, `ENABLE_FORWARD_USER_INFO_HEADERS=true` будет слать эти headers и туда. В таком случае лучше вернуться к per-connection Headers.

Если `./openwebui-data` уже содержит сохранённый Open WebUI config, database config может перекрыть `OPENAI_API_BASE_URL`/`OPENAI_API_KEY` из env. Тогда проверь connection в UI или стартуй с пустым `openwebui-data`.

## API clients

Для programmatic вызовов через Open WebUI используй `/api/chat/completions` или `/api/v1/chat/completions` и передавай стабильный `chat_id`:

```json
{
  "model": "qwen3-8b",
  "chat_id": "stable-session-1",
  "session_id": "api-client-1",
  "id": "assistant-message-1",
  "messages": [{"role": "user", "content": "ok"}],
  "stream": false
}
```

Если `chat_id` не передан, sticky будет не per-chat: либо fallback по user, либо новый generated `chat_id` для new-chat запроса.

## Запуск

```bash
cd /Users/leo/Projects/opensource/llm_engines/nyashkimyashki/sticky-sessions
docker compose up -d
```

Порты:

- Open WebUI: `http://127.0.0.1:8080`
- SMG Prometheus: `http://127.0.0.1:29000/metrics`

Default admin:

- email: `admin@example.com`
- password: `admin-password`

## Проверка

```bash
TOKEN=$(curl -s http://127.0.0.1:8080/api/v1/auths/signin \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@example.com","password":"admin-password"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')

for i in 1 2 3; do
  curl -s http://127.0.0.1:8080/api/chat/completions \
    -H "Authorization: Bearer $TOKEN" \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3-8b","chat_id":"sticky-api-test","session_id":"curl-test","id":"assistant-msg-1","messages":[{"role":"user","content":"Reply with one word: ok"}],"stream":false,"max_tokens":4,"temperature":0}'
  echo
done
```

Проверка, что прямые host-порты LiteLLM/SMG inference закрыты:

```bash
curl -sS http://127.0.0.1:4000/v1/models
curl -sS http://127.0.0.1:30000/v1/models
```

Ожидание: connection refused. `http://127.0.0.1:8080/openai/chat/completions` в этом варианте не блокируется compose-ом.

Метрики:

```bash
curl -s http://127.0.0.1:29000/metrics | grep 'smg_manual_policy'
```

Ожидание: первый запрос создаёт assignment, повторы дают hit. Если растёт `no_routing_id`, LiteLLM не донёс `X-SMG-Routing-Key`.

## Переменные

- `LITELLM_MASTER_KEY`, default `sk-sticky-test`
- `LITELLM_UPSTREAM_MODEL`, default `openai/qwen3-8b`
- `OPENWEBUI_PORT`, default `8080`
- `WEBUI_ADMIN_EMAIL`, default `admin@example.com`
- `WEBUI_ADMIN_PASSWORD`, default `admin-password`
- `SMG_PROMETHEUS_PORT`, default `29000`
- `SMG_ASSIGNMENT_MODE`, default `min_group`
- `SMG_MAX_IDLE_SECS`, default `14400`
