#!/usr/bin/env python3
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request


ROOT = "/data/scratch/hf/sglang_hicache_research/sticky_owui_opencode"
SMG = "http://127.0.0.1:30100"
SMG_METRICS = "http://127.0.0.1:29100/metrics"
OWUI = "http://127.0.0.1:8090"
LITELLM_KEY = "sk-sticky-test"
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = "admin-password"


def request_json(url, payload=None, headers=None, timeout=60):
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
        if line.startswith("#") or not (
            "smg_manual_policy" in line or "smg_worker_selection_total" in line
        ):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([-+0-9.eE]+)$", line)
        if match:
            out[match.group(1)] = float(match.group(2))
    return out


def metric_delta(before, after):
    keys = sorted(set(before) | set(after))
    return {key: after.get(key, 0.0) - before.get(key, 0.0) for key in keys if after.get(key, 0.0) != before.get(key, 0.0)}


def restart_smg():
    subprocess.run(["docker", "compose", "restart", "smg"], cwd=ROOT, check=True)
    wait_url(f"{SMG}/v1/models", timeout=240)
    time.sleep(2)


def login_owui():
    data = request_json(
        f"{OWUI}/api/v1/auths/signin",
        {"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD},
    )
    return data["token"]


def write_config(project_dir, provider, base_url, api_key):
    os.makedirs(project_dir, exist_ok=True)
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            provider: {
                "name": provider,
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "baseURL": base_url,
                    "apiKey": api_key,
                    "timeout": 180000,
                    "chunkTimeout": 180000,
                },
                "models": {
                    "qwen": {
                        "id": "qwen",
                        "name": "qwen",
                        "limit": {"context": 131072, "output": 64},
                        "cost": {"input": 0, "output": 0},
                    }
                },
            }
        },
        "model": f"{provider}/qwen",
        "small_model": f"{provider}/qwen",
        "agent": {
            "build": {
                "model": f"{provider}/qwen",
                "tools": {},
                "steps": 1,
            },
            "title": {
                "model": f"{provider}/qwen",
                "tools": {},
                "steps": 1,
            },
        },
    }
    with open(os.path.join(project_dir, "opencode.json"), "w") as f:
        json.dump(config, f, indent=2)


def run_opencode(project_dir, provider, label):
    cmd = [
        "opencode",
        "run",
        "--pure",
        "--model",
        f"{provider}/qwen",
        "--agent",
        "build",
        "--format",
        "json",
        "Reply with exactly ok",
    ]
    stdout_path = os.path.join(ROOT, f"{label}.stdout.jsonl")
    stderr_path = os.path.join(ROOT, f"{label}.stderr.log")
    with open(stdout_path, "w") as stdout, open(stderr_path, "w") as stderr:
        proc = subprocess.run(
            cmd,
            cwd=project_dir,
            text=True,
            stdout=stdout,
            stderr=stderr,
            timeout=240,
        )
    with open(stdout_path) as f:
        stdout_text = f.read()
    with open(stderr_path) as f:
        stderr_text = f.read()
    session_ids = sorted(set(re.findall(r"ses_[A-Za-z0-9]+", stdout_text + "\n" + stderr_text)))
    return {
        "label": label,
        "returncode": proc.returncode,
        "session_ids": session_ids,
        "stdout_tail": stdout_text[-800:],
        "stderr_tail": stderr_text[-1200:],
    }


def run_scenario(name, provider, base_url, api_key):
    project_dir = os.path.join(ROOT, f"opencode_real_{name}")
    shutil.rmtree(project_dir, ignore_errors=True)
    write_config(project_dir, provider, base_url, api_key)
    restart_smg()
    before = metric_snapshot()
    runs = [run_opencode(project_dir, provider, f"{name}_A"), run_opencode(project_dir, provider, f"{name}_B")]
    after = metric_snapshot()
    print(json.dumps({"scenario": name, "runs": runs, "metric_delta": metric_delta(before, after)}, indent=2, sort_keys=True))


def main():
    wait_url(f"{SMG}/v1/models")
    token = login_owui()
    run_scenario("direct_litellm", "litellm", "http://127.0.0.1:4010/v1", LITELLM_KEY)
    run_scenario("owui_openai", "owuiopenai", "http://127.0.0.1:8090/openai", token)
    run_scenario("owui_api", "owuiapi", "http://127.0.0.1:8090/api", token)


if __name__ == "__main__":
    main()
