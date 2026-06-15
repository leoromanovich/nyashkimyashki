import json
import os
import time
import urllib.error
import urllib.request


BASE_URL = os.getenv("OPENWEBUI_URL", "http://open-webui:8080").rstrip("/")
ADMIN_EMAIL = os.getenv("WEBUI_ADMIN_EMAIL", "admin@example.com")
ADMIN_PASSWORD = os.getenv("WEBUI_ADMIN_PASSWORD", "admin-password")
LITELLM_BASE_URL = os.getenv("OPENWEBUI_LITELLM_BASE_URL", "http://litellm:4000/v1")
LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-sticky-test")


def request_json(method, path, payload=None, token=None, timeout=10):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
        if not body:
            return None
        return json.loads(body.decode("utf-8"))


def wait_for_openwebui():
    deadline = time.time() + 300
    while time.time() < deadline:
        try:
            request_json("GET", "/health", timeout=3)
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError("Open WebUI did not become healthy")


def signin():
    payload = {"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    for _ in range(60):
        try:
            response = request_json("POST", "/api/v1/auths/signin", payload=payload)
            token = response.get("token")
            if token:
                return token
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            print(f"signin failed: HTTP {error.code}: {detail}", flush=True)
        except Exception as error:
            print(f"signin failed: {error}", flush=True)
        time.sleep(2)
    raise RuntimeError("Could not sign in to Open WebUI")


def configure_openai(token):
    payload = {
        "ENABLE_OPENAI_API": True,
        "OPENAI_API_BASE_URLS": [LITELLM_BASE_URL],
        "OPENAI_API_KEYS": [LITELLM_MASTER_KEY],
        "OPENAI_API_CONFIGS": {
            "0": {
                "auth_type": "bearer",
                "headers": {
                    "x-litellm-session-id": "{{CHAT_ID}}",
                    "x-litellm-agent-id": "openwebui",
                    "x-openwebui-chat-id": "{{CHAT_ID}}",
                    "x-openwebui-message-id": "{{MESSAGE_ID}}",
                    "x-openwebui-user-id": "{{USER_ID}}",
                    "x-openwebui-user-email": "{{USER_EMAIL}}",
                    "x-openwebui-user-role": "{{USER_ROLE}}",
                },
            }
        },
    }
    request_json("POST", "/openai/config/update", payload=payload, token=token)


def main():
    wait_for_openwebui()
    token = signin()
    configure_openai(token)
    print("Open WebUI LiteLLM connection configured", flush=True)


if __name__ == "__main__":
    main()
