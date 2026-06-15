# LiteLLM + SMG Sticky Session Test Stack

This stack assumes SGLang is already running and reachable from Docker.

## Configure

```bash
cd litellm-smg-sticky
cp .env.example .env
```

Set `SGLANG_BASE_URL` and `LITELLM_UPSTREAM_MODEL` in `.env`.

For SGLang running on the host:

```env
SGLANG_BASE_URL=http://host.docker.internal:30000
```

## Start

```bash
docker compose up -d
```

Endpoints:

- LiteLLM: `http://127.0.0.1:4000/v1`
- SMG: `http://127.0.0.1:8000/v1`
- SMG Prometheus: `http://127.0.0.1:29000/metrics`

## Test Same Sticky Key

```bash
for i in 1 2 3; do
  curl -s http://127.0.0.1:4000/v1/chat/completions \
    -H 'Authorization: Bearer sk-sticky-test' \
    -H 'Content-Type: application/json' \
    -H 'x-litellm-session-id: sticky-session-a' \
    -H 'x-litellm-agent-id: test-agent' \
    -H 'x-repo-id: test/repo' \
    -d '{
      "model": "sglang-dpa",
      "messages": [{"role": "user", "content": "Say one short sentence."}],
      "max_tokens": 16,
      "temperature": 0
    }' | jq '{id, usage}'
done
```

SMG should show one new key and then hits:

```bash
curl -s http://127.0.0.1:29000/metrics | grep 'smg_manual_policy'
```

Expected pattern:

```text
smg_manual_policy_branch_total{branch="vacant"} 1
smg_manual_policy_branch_total{branch="occupied_hit"} 2
smg_manual_policy_cache_entries 1
```

If `branch="no_routing_id"` increases, LiteLLM did not inject/pass `X-SMG-Routing-Key`.

## Test Different Sticky Keys

Change only `x-litellm-session-id`:

```bash
for s in sticky-session-a sticky-session-b sticky-session-c; do
  curl -s http://127.0.0.1:4000/v1/chat/completions \
    -H 'Authorization: Bearer sk-sticky-test' \
    -H 'Content-Type: application/json' \
    -H "x-litellm-session-id: ${s}" \
    -H 'x-litellm-agent-id: test-agent' \
    -H 'x-repo-id: test/repo' \
    -d '{
      "model": "sglang-dpa",
      "messages": [{"role": "user", "content": "Say one short sentence."}],
      "max_tokens": 16,
      "temperature": 0
    }' >/dev/null
done
curl -s http://127.0.0.1:29000/metrics | grep 'smg_manual_policy_cache_entries'
```

`smg_manual_policy_cache_entries` should grow with distinct sessions, then expire after `SMG_MAX_IDLE_SECS`.

## Optional Cache Check

If SGLang was started with `--enable-cache-report`, repeated same-session prompts should eventually show non-zero cached tokens in `usage.prompt_tokens_details.cached_tokens`.
