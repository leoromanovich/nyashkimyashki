# Sticky session live check: 192.168.0.59

Дата: 2026-06-17.

Папка стенда:

```bash
/data/scratch/hf/sglang_hicache_research/sticky_owui_opencode
```

Цепочка:

```text
opencode-like client -> Open WebUI :8090 -> LiteLLM :4010 -> SMG :30100 -> SGLang :30000
```

Модель: существующий SGLang `qwen` (`/data/scratch/hf/Qwen/Qwen3.6-27B-FP8`), one GPU. DP-aware routing не тестировался.

Первичный прогон был opencode-like HTTP. После установки `opencode 1.17.7` на remote дополнительно проверен настоящий `opencode run`.

## Запущено

- SGLang: existing `sglang-qwen36-server-mooncake-ssd`, `0.0.0.0:30000`.
- SMG: `sticky-test-smg`, `127.0.0.1:30100`, metrics `127.0.0.1:29100`, `--policy=manual`.
- LiteLLM: host venv, `127.0.0.1:4010`, config `litellm-config-host.yaml`, hook `smg_sticky.py`.
- Open WebUI: `sticky-test-openwebui`, `127.0.0.1:8090`, `ENABLE_FORWARD_USER_INFO_HEADERS=true`.
- OWUI sticky wrapper: `sticky-test-owui-sticky-wrapper`, `127.0.0.1:8091`, proxies only chat completion endpoints.

## Результат

### Real opencode

`opencode 1.17.7` для `@ai-sdk/openai-compatible` реально отправляет:

```http
User-Agent: opencode/1.17.7 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14
x-session-affinity: ses_...
x-session-id: ses_...
```

Body остаётся обычным OpenAI Chat Completions, без `metadata.chat_id` и без top-level `chat_id`.

| Real opencode path | Two sessions / four LLM calls | SMG result | Вывод |
|---|---:|---|---|
| `opencode -> LiteLLM /v1` | 2 sessions | `vacant=2`, `occupied_hit=2`, `cache_entries=2` | Sticky per opencode session работает. |
| `opencode -> OWUI /openai` | 2 sessions | `vacant=1`, `occupied_hit=3`, `cache_entries=1` | OWUI не доносит opencode session headers до LiteLLM; sticky схлопнулся в user/API-key fallback. |
| `opencode -> OWUI /api` | 2 sessions | `vacant=1`, `occupied_hit=3`, `cache_entries=1` | Сам path `/api` не помогает, если request body не содержит стабильный `chat_id`. |
| `opencode -> wrapper /openai -> OWUI` | 2 sessions | `vacant=2`, `occupied_hit=2`, `cache_entries=2` | Wrapper добавляет `metadata.chat_id`; sticky per opencode session работает. Rechecked after production exact-route mode. |

### Wrapper `/api`

`owui_sticky_wrapper.py` в production-mode принимает только:

- `/openai/chat/completions`
- `/openai/v1/chat/completions`
- `/api/chat/completions`
- `/api/v1/chat/completions`

Остальные `/api/*`, `/openai/*`, uploads/files/models должны идти напрямую в Open WebUI через ingress.

Проверено: `/openai/models` на wrapper возвращает `404`; `/openai/models` напрямую в Open WebUI с auth возвращает `200`.

For `/api/chat/completions`:

- читает session id из `x-session-affinity`/`x-session-id`/aliases;
- если `chat_id` отсутствует, добавляет `chat_id=<session>`;
- не добавляет `session_id`: в текущем Open WebUI это переводит `/api/chat/completions` в task-envelope branch (`{"status": true, "task_ids": []}`), а не в обычный `chat.completion`.

Проверка `/api` через wrapper в production exact-route mode: A/A/B requests с `x-session-affinity` дали `vacant=2`, `occupied_hit=1`, `cache_entries=2` и обычный `chat.completion` response.

### HTTP probe

| Scenario | A1/A2/B1 SMG manual metrics | LiteLLM sticky source | Вывод |
|---|---:|---|---|
| Direct LiteLLM + `x-opencode-session` | `vacant=2`, `occupied_hit=1`, `cache_entries=2` | `x-opencode-session` | Работает per-session. |
| OWUI `/api/chat/completions` + explicit `chat_id` | `vacant=2`, `occupied_hit=1`, `cache_entries=2` | `x-openwebui-chat-id` | Работает per-chat. |
| OWUI `/api/chat/completions` без `chat_id` | `vacant=1`, `occupied_hit=2`, `cache_entries=1` | fallback `api_key.user_id` | Не per-chat; все запросы схлопнулись в user/API-key sticky. |
| OWUI `/openai/chat/completions` + opencode headers | `vacant=1`, `occupied_hit=2`, `cache_entries=1` | fallback `api_key.user_id` | OWUI не форвардит client session headers в LiteLLM. |
| OWUI `/openai/chat/completions` + `metadata.chat_id` | `vacant=2`, `occupied_hit=1`, `cache_entries=2` | `x-openwebui-chat-id` | Работает per-chat: OWUI превращает `metadata.chat_id` в `X-OpenWebUI-Chat-Id`. |
| OWUI `/openai/chat/completions` + top-level `chat_id` | `vacant=1`, `occupied_hit=2`, `cache_entries=1` | fallback `api_key.user_id` | Top-level `chat_id` не является sticky signal на `/openai`. |
| OWUI `/openai/chat/completions` + `x-litellm-session-id` | `vacant=1`, `occupied_hit=2`, `cache_entries=1` | fallback `api_key.user_id` | Header не доходит до LiteLLM через OWUI `/openai`. |

## Вывод

Для opencode через Open WebUI нельзя рассчитывать, что native headers дойдут до LiteLLM. Рабочие варианты:

1. Подключать opencode напрямую к LiteLLM/ingress и слать `X-LiteLLM-Session-Id` или поддержанный alias (`x-opencode-session`, `x-session-affinity`).
2. Если обязательно через OWUI `/openai`, нужен wrapper/adapter, который читает `x-session-affinity`/`x-session-id` от opencode и кладёт session id в body `metadata.chat_id`.
3. Для `/api/chat/completions` wrapper должен добавлять только top-level `chat_id=<session id>`, не `session_id`. Просто сменить base URL opencode на `/api` недостаточно.

`X-SMG-Routing-Key` остаётся internal-only: его строит LiteLLM hook.

Дополнительно: в этой конфигурации OWUI `GET /openai/models` работает, а `GET /openai/v1/models` возвращает 403 (`ENABLE_OPENAI_API_PASSTHROUGH=false`). Для OpenAI-compatible clients base URL должен быть `http://<owui>/openai`, если client не требует `/v1`.
