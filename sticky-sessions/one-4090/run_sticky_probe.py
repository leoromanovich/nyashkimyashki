#!/usr/bin/env python3
import json
import re
import subprocess
import time
import urllib.error
import urllib.request


LITELLM = "http://127.0.0.1:4010"
OWUI = "http://127.0.0.1:8090"
SMG = "http://127.0.0.1:30100"
SMG_METRICS = "http://127.0.0.1:29100/metrics"
LITELLM_KEY = "sk-sticky-test"
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "admin-password"


def request_json(url, payload=None, headers=None, timeout=180):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers or {}, method="GET" if payload is None else "POST")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode()
        return json.loads(raw) if raw else {}


def wait_url(url, timeout=240):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            return request_json(url, timeout=10)
        except Exception as exc:
            last = exc
            time.sleep(2)
    raise RuntimeError(f"{url} did not become ready: {last}")


def metric_snapshot():
    text = urllib.request.urlopen(SMG_METRICS, timeout=10).read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or "manual" not in line.lower():
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([-+0-9.eE]+)$", line)
        if match:
            out[match.group(1)] = float(match.group(2))
    return out


def metric_delta(before, after):
    keys = sorted(set(before) | set(after))
    return {key: after.get(key, 0.0) - before.get(key, 0.0) for key in keys if after.get(key, 0.0) != before.get(key, 0.0)}


def chat_payload(text, chat_id=None, metadata=None):
    payload = {
        "model": "qwen",
        "messages": [{"role": "user", "content": text}],
        "stream": False,
        "max_tokens": 4,
        "temperature": 0,
    }
    if chat_id:
        payload["chat_id"] = chat_id
    if metadata:
        payload["metadata"] = metadata
    return payload


def litellm_chat(session_id):
    return request_json(
        f"{LITELLM}/v1/chat/completions",
        chat_payload("Reply with one word: ok"),
        {
            "Authorization": f"Bearer {LITELLM_KEY}",
            "User-Agent": "opencode/1.0 sticky-probe",
            "x-session-affinity": session_id,
            "x-opencode-session": session_id,
            "x-repo-id": "repo-sticky-probe",
            "x-user-id": "user-sticky-probe",
        },
    )


def owui_api_chat(token, chat_id):
    return request_json(
        f"{OWUI}/api/chat/completions",
        chat_payload("Reply with one word: ok", chat_id=chat_id),
        {"Authorization": f"Bearer {token}"},
    )


def owui_openai_chat(token, session_id, metadata=None, litellm_session_id=None):
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "opencode/1.0 sticky-probe",
        "x-session-affinity": session_id,
        "x-opencode-session": session_id,
        "x-repo-id": "repo-sticky-probe",
        "x-user-id": "user-sticky-probe",
    }
    if litellm_session_id:
        headers["x-litellm-session-id"] = litellm_session_id
    return request_json(
        f"{OWUI}/openai/chat/completions",
        chat_payload("Reply with one word: ok", metadata=metadata),
        headers,
    )


def owui_openai_top_level_chat(token, chat_id):
    return request_json(
        f"{OWUI}/openai/chat/completions",
        chat_payload("Reply with one word: ok", chat_id=chat_id),
        {"Authorization": f"Bearer {token}"},
    )


def login():
    data = request_json(
        f"{OWUI}/api/v1/auths/signin",
        {"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
    )
    return data["token"]


def restart_smg():
    subprocess.run(["docker", "compose", "restart", "smg"], check=True)
    wait_url(f"{SMG}/v1/models", timeout=240)
    time.sleep(2)


def run_scenario(name, calls):
    restart_smg()
    before = metric_snapshot()
    results = []
    for label, fn in calls:
        try:
            fn()
            results.append((label, "ok"))
        except urllib.error.HTTPError as exc:
            results.append((label, f"http-{exc.code}:{exc.read().decode(errors='ignore')[:300]}"))
        except Exception as exc:
            results.append((label, f"error:{exc!r}"))
    after = metric_snapshot()
    print(json.dumps({"scenario": name, "calls": results, "metric_delta": metric_delta(before, after)}, indent=2, sort_keys=True))


def main():
    wait_url(f"{SMG}/v1/models")
    wait_url(f"{LITELLM}/health/liveliness")
    wait_url(f"{OWUI}/health")
    token = login()

    run_scenario(
        "direct_litellm_opencode_headers",
        [
            ("A1", lambda: litellm_chat("ses_direct_A")),
            ("A2", lambda: litellm_chat("ses_direct_A")),
            ("B1", lambda: litellm_chat("ses_direct_B")),
        ],
    )
    run_scenario(
        "owui_api_chat_id",
        [
            ("A1", lambda: owui_api_chat(token, "chat_api_A")),
            ("A2", lambda: owui_api_chat(token, "chat_api_A")),
            ("B1", lambda: owui_api_chat(token, "chat_api_B")),
        ],
    )
    run_scenario(
        "owui_api_no_chat_id",
        [
            ("A1", lambda: owui_api_chat(token, None)),
            ("A2", lambda: owui_api_chat(token, None)),
            ("B1", lambda: owui_api_chat(token, None)),
        ],
    )
    run_scenario(
        "owui_openai_opencode_headers",
        [
            ("A1", lambda: owui_openai_chat(token, "ses_owui_openai_A")),
            ("A2", lambda: owui_openai_chat(token, "ses_owui_openai_A")),
            ("B1", lambda: owui_openai_chat(token, "ses_owui_openai_B")),
        ],
    )
    run_scenario(
        "owui_openai_metadata_chat_id",
        [
            ("A1", lambda: owui_openai_chat(token, "ses_ignored_A", {"chat_id": "meta_chat_A"})),
            ("A2", lambda: owui_openai_chat(token, "ses_ignored_A", {"chat_id": "meta_chat_A"})),
            ("B1", lambda: owui_openai_chat(token, "ses_ignored_B", {"chat_id": "meta_chat_B"})),
        ],
    )
    run_scenario(
        "owui_openai_top_level_chat_id",
        [
            ("A1", lambda: owui_openai_top_level_chat(token, "top_chat_A")),
            ("A2", lambda: owui_openai_top_level_chat(token, "top_chat_A")),
            ("B1", lambda: owui_openai_top_level_chat(token, "top_chat_B")),
        ],
    )
    run_scenario(
        "owui_openai_litellm_session_header",
        [
            ("A1", lambda: owui_openai_chat(token, "ses_ignored_A", litellm_session_id="llm_ses_A")),
            ("A2", lambda: owui_openai_chat(token, "ses_ignored_A", litellm_session_id="llm_ses_A")),
            ("B1", lambda: owui_openai_chat(token, "ses_ignored_B", litellm_session_id="llm_ses_B")),
        ],
    )


if __name__ == "__main__":
    main()
