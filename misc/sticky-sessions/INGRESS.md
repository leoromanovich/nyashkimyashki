# OWUI sticky wrapper ingress

Production idea:

```text
external owui URL
  /api/chat/completions        -> owui-sticky-wrapper -> real Open WebUI
  /api/v1/chat/completions     -> owui-sticky-wrapper -> real Open WebUI
  /openai/chat/completions     -> owui-sticky-wrapper -> real Open WebUI
  /openai/v1/chat/completions  -> owui-sticky-wrapper -> real Open WebUI
  everything else              -> real Open WebUI
```

Users and coding agents keep the same public URL. Only ingress upstreams change.

Nginx-style sketch:

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

location ^~ /api/ {
    proxy_pass http://open-webui:8080;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}

location ^~ /openai/ {
    proxy_pass http://open-webui:8080;
    proxy_http_version 1.1;
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

Wrapper patches only:

- `/openai/chat/completions`
- `/openai/v1/chat/completions`
- `/api/chat/completions`
- `/api/v1/chat/completions`

All other `/api/*` and `/openai/*` requests go directly to Open WebUI. The wrapper itself rejects non-chat-completion paths, so uploads/files/models do not get buffered by it even if ingress is misrouted.

Patch rules:

- `/openai/*`: `body.metadata.chat_id = <session id>` when missing.
- `/api/*`: `body.chat_id = <session id>` when missing.
- `/api/*`: do not synthesize `session_id`; current Open WebUI treats `session_id` on `/api/chat/completions` as a task-style response path.

Session id sources are checked in this order: `x-litellm-session-id`, `x-session-affinity`, `x-session-id`, opencode/codex/continue/claude/roo/kilo aliases, then existing body `metadata.chat_id`/`chat_id`/`session_id`.
