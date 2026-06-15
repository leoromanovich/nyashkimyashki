# Sticky sessions: LiteLLM -> SMG -> SGLang DPA

База: Qwen3-8B-FP8, `tp=2`, `dp=2`, `--enable-dp-attention`.

```text
client -> Open WebUI edge :8080 -> Open WebUI /api/chat/completions
       -> LiteLLM :4000 internal -> SMG :30000 internal -> SGLang worker internal
```

LiteLLM сам выставляет `X-SMG-Routing-Key`; пользователям его задавать не нужно.

Важно: для sticky по `X-SMG-Routing-Key` у SMG стоит `--policy=manual`. `cache_aware` маршрутизирует по prefix/cache locality, а не по session header.

## Open WebUI

Compose сам настраивает Open WebUI connection на LiteLLM через `openwebui-bootstrap`.

Итоговая connection:

- URL: `http://litellm:4000/v1`
- Auth: `Bearer`
- Key: значение `LITELLM_MASTER_KEY`
- Headers:

```json
{
  "x-litellm-session-id": "{{CHAT_ID}}",
  "x-litellm-agent-id": "openwebui",
  "x-openwebui-chat-id": "{{CHAT_ID}}",
  "x-openwebui-message-id": "{{MESSAGE_ID}}",
  "x-openwebui-user-id": "{{USER_ID}}",
  "x-openwebui-user-email": "{{USER_EMAIL}}",
  "x-openwebui-user-role": "{{USER_ROLE}}"
}
```

Open WebUI умеет подставлять `{{CHAT_ID}}`, `{{MESSAGE_ID}}`, `{{USER_ID}}`, `{{USER_NAME}}`, `{{USER_EMAIL}}`, `{{USER_ROLE}}` в per-connection headers. LiteLLM hook читает эти headers и отправляет в SMG только hashed `X-SMG-Routing-Key`.

Все обычные UI-запросы Open WebUI идут через `/api/chat/completions`. Для принуждения:

- наружу опубликован только `openwebui-edge`;
- LiteLLM и SMG inference-порт не опубликованы на host;
- edge режет `/openai/chat/completions` и `/openai/responses`;
- `ENABLE_OPENAI_API_PASSTHROUGH=false`.

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

Проверка, что внешний `/openai` generation path закрыт:

```bash
curl -i http://127.0.0.1:8080/openai/chat/completions
```

Ожидание: `404`.

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
