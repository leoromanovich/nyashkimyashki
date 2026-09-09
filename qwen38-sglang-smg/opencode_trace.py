#!/usr/bin/env python3
"""Finite OpenCode smoke test with metadata-only OTLP traces (Python stdlib)."""
import argparse
import collections
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid


SERVICES = {"opencode-smoke", "open-webui", "litellm", "smg", "sglang"}
FORBIDDEN = {"gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.prompt",
             "gen_ai.completion", "http.request.body", "http.response.body",
             "db.statement", "authorization", "exception.message", "exception.stacktrace"}


def http_json(url, payload=None):
    request = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def run_cli(argv, workspace, env, timeout):
    # A separate process group also lets timeout/interrupt close CLI child processes.
    process = subprocess.Popen(argv, cwd=workspace, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    timed_out = False
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, _ = process.communicate()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return process.returncode, timed_out, events


def export_root(endpoint, trace_id, span_id, start, end, mode, success):
    span = {"traceId": trace_id, "spanId": span_id, "name": "opencode.run", "kind": 1,
            "startTimeUnixNano": str(start), "endTimeUnixNano": str(end),
            "attributes": [{"key": "probe.mode", "value": {"stringValue": mode}}],
            "status": {"code": 1 if success else 2}}
    response = http_json(endpoint, {"resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "opencode-smoke"}}]},
        "scopeSpans": [{"scope": {"name": "opencode-cli-test-wrapper"}, "spans": [span]}]}]})
    if int(response.get("partialSuccess", {}).get("rejectedSpans", 0)):
        raise ValueError("Collector rejected the root span")


def verify_trace(jaeger, trace_id, minimum_requests):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            record = http_json(jaeger.rstrip("/") + "/api/traces/" + trace_id)["data"][0]
            spans, processes = record["spans"], record["processes"]
            by_id = {s["spanID"]: s for s in spans}
            roots = [s for s in spans if s["operationName"] == "sglang.generate"]
            ready = (SERVICES <= {p["serviceName"] for p in processes.values()}
                     and len(roots) >= max(1, minimum_requests)
                     and any(s["operationName"].startswith("Scheduler") for s in spans)
                     and all(r["spanID"] in by_id for s in spans for r in s["references"]
                             if r["refType"] == "CHILD_OF"))
            if ready:
                break
        except (OSError, IndexError):
            pass
        time.sleep(1)
    else:
        raise AssertionError("Complete trace did not arrive within 30 seconds")
    for root in roots:
        services, seen, cursor = set(), set(), root
        while cursor["spanID"] not in seen:
            seen.add(cursor["spanID"])
            services.add(processes[cursor["processID"]]["serviceName"])
            parents = [r["spanID"] for r in cursor["references"] if r["refType"] == "CHILD_OF"]
            if not parents:
                break
            cursor = by_id[parents[0]]
        assert SERVICES <= services, "Broken parent chain"
    assert all(s["traceID"].lstrip("0") == trace_id.lstrip("0") for s in spans)
    keys = {a["key"].lower() for s in spans for a in s.get("tags", [])}
    keys.update(a["key"].lower() for s in spans for log in s.get("logs", []) for a in log["fields"])
    assert not keys & FORBIDDEN, "Unexpected payload attributes in trace"
    wire = json.dumps(record)
    assert all(marker not in wire for marker in
               ("Compute 17 times 19", "Reply with only this JSON object", "OPENCODE_TOOL_OK")), "Prompt marker in trace"
    return {"parentage": "verified", "services": sorted(SERVICES), "span_count": len(spans),
            "inference_requests": len(roots), "prompt_absent": True,
            "inference_ms": [round(s["duration"] / 1000, 3) for s in roots]}


def probe(args, mode):
    trace_id, span_id = uuid.uuid4().hex, secrets.token_hex(8)
    report = {"case": mode, "trace_id": trace_id, "jaeger_path": "/trace/" + trace_id}
    with tempfile.TemporaryDirectory(prefix="opencode-trace-", dir=args.work_root) as directory:
        root = Path(directory).resolve()
        workspace = root / "workspace"
        workspace.mkdir()
        subprocess.run(["git", "init", "-q", str(workspace)], check=True, capture_output=True)
        proof = workspace / "proof.txt"
        permissions = {"*": "deny"}
        if mode == "tools":
            for tool in ("read", "write", "edit"):
                permissions[tool] = {"*": "deny", "proof.txt": "allow", str(proof): "allow"}
        config = {
            "model": "trace-local/Qwen3.8-27B", "small_model": "trace-local/Qwen3.8-27B",
            "enabled_providers": ["trace-local"], "share": "disabled", "autoupdate": False,
            "permission": permissions,
            "agent": {"smoke": {"mode": "primary", "permission": permissions,
                                "prompt": "Perform only the requested synthetic test. Use the exact file path supplied. Be concise."}},
            "provider": {"trace-local": {
                "npm": "@ai-sdk/openai-compatible", "name": "Local trace test",
                "options": {"baseURL": args.base_url.rstrip("/"), "apiKey": "{env:TRACE_API_KEY}",
                            "headers": {"traceparent": f"00-{trace_id}-{span_id}-01"},
                            "timeout": int(args.timeout * 1000)},
                "models": {"Qwen3.8-27B": {"name": "Qwen3.8-27B", "limit": {"context": 32768, "output": 1024}}}}}}
        env = dict(os.environ)
        env.pop("OPENCODE_CONFIG", None)
        for name in ("CONFIG", "CACHE", "DATA", "STATE"):
            path = root / name.lower()
            path.mkdir()
            env[f"XDG_{name}_HOME"] = str(path)
        env.update(OPENCODE_CONFIG_DIR=str(root / "config"), OPENCODE_CONFIG_CONTENT=json.dumps(config),
                   OPENCODE_DISABLE_MODELS_FETCH="true", OPENCODE_DISABLE_AUTOUPDATE="true")
        prompt = {"arithmetic": "Compute 17 times 19. Reply with only the integer.",
                  "json": 'Reply with only this JSON object, without markdown: {"ok":true,"value":42}',
                  "tools": f"Use write to create {proof} containing OPENCODE_TOOL_OK. Read that exact file using read, then reply with only DONE."}[mode]
        print(f"Running {mode}; trace_id={trace_id}", file=sys.stderr, flush=True)
        start = time.time_ns()
        code, timed_out, events = run_cli(
            [args.opencode, "run", "--dir", str(workspace), "--pure", "--format", "json",
             "--agent", "smoke", "--model", "trace-local/Qwen3.8-27B", "--title", "Trace smoke " + mode, prompt],
            workspace, env, args.timeout)
        end = time.time_ns()
        counts = collections.Counter(e.get("type") for e in events)
        answer = "".join(e.get("part", {}).get("text", "") for e in events if e.get("type") == "text").strip()
        tool_events = [e.get("part", {}) for e in events if e.get("type") == "tool_use"]
        if mode == "arithmetic":
            correct = answer == "323"
        elif mode == "json":
            try:
                correct = json.loads(answer) == {"ok": True, "value": 42}
            except ValueError:
                correct = False
        else:
            completed = {e.get("tool") for e in tool_events if e.get("state", {}).get("status") == "completed"}
            correct = (proof.is_file() and proof.read_text().strip() == "OPENCODE_TOOL_OK"
                       and {"write", "read"} <= completed and answer.endswith("DONE")
                       and all(e.get("state", {}).get("input", {}).get("filePath") == str(proof) for e in tool_events))
        success = correct and code == 0 and not timed_out and not counts["error"]
        report.update(exit_code=code, timed_out=timed_out, response_correct=success,
                      client_seconds=round((end - start) / 1e9, 3), event_types=dict(counts),
                      tools=[{"tool": e.get("tool"), "status": e.get("state", {}).get("status")} for e in tool_events])
    # Temporary CLI history, credentials/config and synthetic files are gone here.
    try:
        export_root(args.otlp_http_endpoint, trace_id, span_id, start, end, mode, success)
        report["root_exported"] = True
        if not args.no_verify:
            report.update(verify_trace(args.jaeger_url, trace_id, counts["step_finish"]))
    except (OSError, AssertionError, ValueError, KeyError, IndexError) as exc:
        # Error text can contain upstream response bodies. Report the class only.
        report["trace_error"] = type(exc).__name__
        success = False
    report["passed"] = success
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="OpenWebUI API root, e.g. http://127.0.0.1:18090/api")
    parser.add_argument("--opencode", default="opencode")
    parser.add_argument("--mode", choices=["all", "arithmetic", "json", "tools"], default="all")
    parser.add_argument("--work-root", help="Existing directory for temporary CLI state/workspace")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--otlp-http-endpoint", default="http://127.0.0.1:4318/v1/traces")
    parser.add_argument("--jaeger-url", default="http://127.0.0.1:16686")
    parser.add_argument("--no-verify", action="store_true", help="Skip Jaeger inspection for an external backend")
    args = parser.parse_args()
    if not os.getenv("TRACE_API_KEY"):
        parser.error("Set TRACE_API_KEY to an OpenWebUI API token")
    args.opencode = shutil.which(args.opencode)
    if not args.opencode or not shutil.which("git"):
        parser.error("OpenCode and git must be installed on the probe host")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    modes = ["arithmetic", "json", "tools"] if args.mode == "all" else [args.mode]
    passed = True
    for mode in modes:
        report = probe(args, mode)
        print(json.dumps(report), flush=True)
        passed = passed and report["passed"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
