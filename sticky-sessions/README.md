# Sticky sessions: OWUI -> LiteLLM -> SMG -> SGLang

Цель: все клиенты используют привычный Open WebUI endpoint, а sticky routing до SGLang делается внутренне и прозрачно.

```text
browser / agents
  -> public Open WebUI URL
  -> ingress
       exact chat-completion paths -> owui-sticky-wrapper
       everything else             -> Open WebUI
  -> Open WebUI
  -> LiteLLM hook
  -> SMG --policy=manual
  -> SGLang
```

Пользователи и инструменты не должны слать `X-SMG-Routing-Key`. Его строит только LiteLLM hook.

## Files

- `smg_sticky.py` - LiteLLM hook: строит internal `X-SMG-Routing-Key`.
- `remote-opencode-owui/owui_sticky_wrapper.py` - production wrapper перед OWUI chat-completion ручками.
- `remote-opencode-owui/docker-compose.host-litellm.yaml` - remote стенд: existing SGLang + SMG + OWUI + wrapper + host LiteLLM.
- `remote-opencode-owui/docker-compose.2x4090-dpa.yaml` - full stack для 2x4090 DPA: SGLang `TP=2 DP=2` + SMG `--dp-aware` + LiteLLM + OWUI + wrapper.
- `remote-opencode-owui/env.2x4090-dpa.example` - env template для 2x4090 DPA compose.
- `remote-opencode-owui/INGRESS.md` - production ingress routes.
- `remote-opencode-owui/RESULTS.md` - live checks на `192.168.0.59`.
- `remote-opencode-owui/run_sticky_probe.py` - HTTP probe.
- `remote-opencode-owui/run_opencode_real_probe.py` - real opencode probe.

## Current Tested State

Remote folder:

```bash
/data/scratch/hf/sglang_hicache_research/sticky_owui_opencode
```

Remote services:

```text
Open WebUI            127.0.0.1:8090
OWUI sticky wrapper   127.0.0.1:8091
LiteLLM               127.0.0.1:4010
SMG                   127.0.0.1:30100, metrics :29100
SGLang                0.0.0.0:30000
```

Production wrapper mode:

```text
/api/chat/completions        -> wrapper
/api/v1/chat/completions     -> wrapper
/openai/chat/completions     -> wrapper
/openai/v1/chat/completions  -> wrapper
all other paths              -> real Open WebUI
```

Wrapper rejects non-chat paths with `404`; uploads/files/images/videos/models/UI do not pass through it.

Live result:

```text
real opencode -> wrapper /openai -> OWUI:
  2 sessions / 4 LLM calls
  SMG: vacant=2, occupied_hit=2, cache_entries=2

/api/chat/completions via wrapper, A/A/B:
  SMG: vacant=2, occupied_hit=1, cache_entries=2
```

## LiteLLM Hook

`smg_sticky.py`:

1. Drops any incoming/proxied `X-SMG-Routing-Key`.
2. Reads semantic identity from headers/body/LiteLLM metadata.
3. Builds tuple `tenant + agent + workspace + user + stable_identity`.
4. Hashes it.
5. Sends only internal `X-SMG-Routing-Key: smg:<hash>` to SMG.

Stable identity priority:

1. `X-LiteLLM-Session-Id`
2. `X-OpenWebUI-Chat-Id`
3. `metadata.chat_id`, `metadata.session_id`
4. Coding-agent aliases: Codex, Claude Code, Continue, opencode, Aider, Roo, Kilo
5. Generic aliases: `x-session-affinity`, `x-session-id`, `session_id`, `x-client-request-id`
6. Fallback: repo/workspace, then user, then LiteLLM call id

Canonical client headers:

```http
X-LiteLLM-Agent-Id: <agent-name>
X-LiteLLM-Session-Id: <chat/thread/task/run/session id>
X-Repo-Id: <repo id>
X-Workspace-Id: <workspace id>
X-User-Id: <user id>
```

## OWUI Sticky Wrapper

Why it exists:

- Coding agents can send session headers.
- Open WebUI does not forward those client headers to LiteLLM.
- Open WebUI does forward `X-OpenWebUI-Chat-Id` when chat id is present in the right place.

Patch rules:

```text
/openai/chat/completions:
  body.metadata.chat_id = <session id>

/api/chat/completions:
  body.chat_id = <session id>
```

For `/api/chat/completions`, do not synthesize `session_id`. Current OWUI treats `session_id` as task-style mode and returns:

```json
{"status": true, "task_ids": []}
```

instead of a normal `chat.completion`.

## Production Ingress

Use exact routes. Do not send all `/api/*` or `/openai/*` through the wrapper.

```nginx
location = /api/chat/completions {
    proxy_pass http://owui-sticky-wrapper:8080;
    proxy_http_version 1.1;
    proxy_buffering off;
    proxy_request_buffering on;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

location = /api/v1/chat/completions {
    proxy_pass http://owui-sticky-wrapper:8080;
    proxy_http_version 1.1;
    proxy_buffering off;
    proxy_request_buffering on;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

location = /openai/chat/completions {
    proxy_pass http://owui-sticky-wrapper:8080;
    proxy_http_version 1.1;
    proxy_buffering off;
    proxy_request_buffering on;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

location = /openai/v1/chat/completions {
    proxy_pass http://owui-sticky-wrapper:8080;
    proxy_http_version 1.1;
    proxy_buffering off;
    proxy_request_buffering on;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

location / {
    proxy_pass http://open-webui:8080;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

In a real nginx config, add separate direct locations for `/api/` and `/openai/` if needed by your location ordering. The important rule is: only exact chat completion paths go to wrapper.

## Agent Authoring Guidelines

### OpenAI-compatible agents

Use the public OWUI URL and the `/openai` base URL:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://owui.example.com/openai",
    api_key="OWUI_TOKEN",
)
```

For sticky, send a stable session id on every request belonging to the same agent run/thread/chat:

```python
session_id = "agent-run-123"

response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "hi"}],
    extra_headers={
        "X-LiteLLM-Agent-Id": "custom-python-agent",
        "X-LiteLLM-Session-Id": session_id,
        "X-User-Id": "user-42",
        "X-Repo-Id": "repo-a",
    },
)
```

Equivalent body-only fallback:

```python
response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "hi"}],
    extra_body={
        "metadata": {
            "chat_id": session_id,
        },
    },
)
```

Prefer headers. They are provider-agnostic and match the LiteLLM hook convention.

Without session headers/body metadata, sticky falls back to user/API-key level and is not per agent run.

### OpenAI Python specifics

Recommended:

```python
client = OpenAI(
    base_url="https://owui.example.com/openai",
    api_key="OWUI_TOKEN",
)
```

Avoid using `https://owui.example.com/openai/v1` as the client base URL. Current OWUI returns `403` for `/openai/v1/models` when passthrough is disabled. The wrapper still accepts `/openai/v1/chat/completions`, but model discovery should hit `/openai/models` directly through OWUI.

Supported for sticky:

```python
client.chat.completions.create(...)
```

Not covered by this wrapper:

```python
client.responses.create(...)
client.embeddings.create(...)
client.audio.*
client.files.*
```

Those paths should go directly to OWUI unless separate sticky support is added.

### opencode

Real `opencode 1.17.7` sends:

```http
x-session-affinity: ses_...
x-session-id: ses_...
```

So opencode works through:

```text
baseURL = https://owui.example.com/openai
```

The wrapper reads those headers and writes `metadata.chat_id`.

### Custom HTTP agents

Use:

```http
POST /openai/chat/completions
Authorization: Bearer <OWUI token>
Content-Type: application/json
X-LiteLLM-Agent-Id: my-agent
X-LiteLLM-Session-Id: stable-run-id
X-User-Id: user-id
X-Repo-Id: repo-id
```

Do not use:

```http
X-SMG-Routing-Key: ...
```

That header is internal-only.

### Choosing IDs

Recommended sticky identity:

```text
session/thread/run/task id > repo/workspace id > user id
```

Use one stable `X-LiteLLM-Session-Id` for the full logical conversation or agent task. Do not generate a new id per HTTP request.

Recommended `X-LiteLLM-Agent-Id` values:

```text
open-webui
opencode
codex
aider
continue
claude-code
roo-code
kilo-code
pi
custom-python-agent
```

## Compose

Remote compose with existing host LiteLLM:

```bash
cd /data/scratch/hf/sglang_hicache_research/sticky_owui_opencode
docker compose -f docker-compose.host-litellm.yaml up -d
```

Important ports:

```text
Open WebUI           127.0.0.1:8090
OWUI sticky wrapper  127.0.0.1:8091
LiteLLM              127.0.0.1:4010
SMG                  127.0.0.1:30100
SMG metrics          127.0.0.1:29100
SGLang               127.0.0.1:30000
```

2x4090 DPA compose:

```bash
git clone https://github.com/leoromanovich/nyashkimyashki.git
cd nyashkimyashki/sticky-sessions/remote-opencode-owui

cp env.2x4090-dpa.example .env.2x4090-dpa
sed -i "s/change-me-random-hex/$(openssl rand -hex 32)/" .env.2x4090-dpa
# edit HOST_MODEL_PATH, WEBUI_ADMIN_EMAIL, WEBUI_ADMIN_PASSWORD if needed

docker compose --env-file .env.2x4090-dpa -f docker-compose.2x4090-dpa.yaml up -d
```

This starts one SGLang worker across both GPUs with DPA:

```text
SGLang TP=2 DP=2       127.0.0.1:30000
SMG manual --dp-aware  127.0.0.1:30100
SMG metrics            127.0.0.1:29100
LiteLLM                127.0.0.1:4010
Open WebUI             127.0.0.1:8090
OWUI sticky wrapper    127.0.0.1:8091
```

In this mode SMG discovers SGLang DP ranks and pins `X-SMG-Routing-Key` to rank-specific logical workers. With `SGLANG_DP=2`, SGLang divides `CHUNKED_PREFILL_SIZE` and `MAX_RUNNING_REQUESTS` by 2 internally.

## Verification

Wrapper should reject non-chat paths:

```bash
curl -i http://127.0.0.1:8091/openai/models
```

Expected:

```text
404
```

Exact `/api/chat/completions` via wrapper should patch:

```bash
TOKEN=$(curl -sS http://127.0.0.1:8090/api/v1/auths/signin \
  -H 'Content-Type: application/json' \
  -d '{"email":"admin@example.com","password":"admin-password"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')

curl -sS -D /tmp/headers.txt http://127.0.0.1:8091/api/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -H 'x-session-affinity: curl-api-A' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Reply with one word: ok"}],"stream":false,"max_tokens":4,"temperature":0}'

grep -i 'X-Sticky-Wrapper' /tmp/headers.txt
```

Expected headers:

```text
X-Sticky-Wrapper-Patched: 1
X-Sticky-Wrapper-Source: x-session-affinity
```

SMG metrics:

```bash
curl -sS http://127.0.0.1:29100/metrics | grep 'smg_manual_policy'
```

Expected pattern for A/A/B:

```text
vacant=2
occupied_hit=1
cache_entries=2
```

## Caveats

- One-GPU remote checks prove header/session propagation and SMG manual-policy cache entries, not DP-aware rank routing.
- `cache_aware` in SMG is prefix/cache locality, not session sticky. Use `--policy=manual`.
- OpenAI-compatible `/openai` is the best path for coding agents.
- `/api/chat/completions` is mainly for OWUI UI/API clients; wrapper patches it for sticky, but OpenAI-compatible agents should prefer `/openai`.
