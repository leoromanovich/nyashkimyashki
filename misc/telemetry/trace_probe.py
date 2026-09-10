#!/usr/bin/env python3
"""Synthetic request + metadata-only OTLP root, optionally verify Jaeger ancestry."""
import argparse
import json
import os
import secrets
import time
import urllib.error
import urllib.request


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def export_root(endpoint, tid, sid, start, end, mode):
    body = {"resourceSpans": [{"resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "trace-probe"}}]},
             "scopeSpans": [{"scope": {"name": "recipes.trace-probe"}, "spans": [{
                 "traceId": tid, "spanId": sid, "name": "probe." + mode, "kind": 1,
                 "startTimeUnixNano": str(start), "endTimeUnixNano": str(end),
                 "attributes": [{"key": "probe.mode", "value": {"stringValue": mode}}]}]}]}]}
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as response:
        data = json.load(response)
    assert not int(data.get("partialSuccess", {}).get("rejectedSpans", 0)), "Collector rejected root"


def verify_trace(record, tid, engine, expected, marker):
    spans = record["spans"]
    ids = {s["spanID"]: s for s in spans}
    service = lambda s: record["processes"][s["processID"]]["serviceName"]
    assert all(s["traceID"].lstrip("0") == tid.lstrip("0") and s["duration"] >= 0 for s in spans), "Invalid span identity/timing"
    assert marker not in json.dumps(record), "Synthetic payload leaked into trace"
    forbidden = ("prompt", "completion", "message", "content", "authorization", "cookie", "http.request.header", "http.response.header", "exception.stacktrace", "exception.message", "db.statement")
    for span in spans:
        for tag in span.get("tags", []):
            key = tag["key"].lower()
            # Token counts are metadata, including gen_ai.usage.prompt_tokens.
            if key.startswith("gen_ai.usage.") and key.endswith("tokens"):
                continue
            assert not any(word in key for word in forbidden), "Payload attribute present"
        for parent in span.get("references", []):
            if parent["refType"] == "CHILD_OF":
                assert parent["spanID"] in ids, "Incomplete parent chain"
    roots = [s for s in spans if service(s) == engine and
             (s["operationName"] == "llm_request" if engine == "vllm" else
              s["operationName"] == "sglang.generate" or "Req " in s["operationName"])]
    assert roots, "Engine request span missing"
    for root in roots:
        current = root
        ancestors = set()
        seen = set()
        while current["spanID"] not in seen:
            seen.add(current["spanID"])
            ancestors.add(service(current))
            parents = [x["spanID"] for x in current.get("references", []) if x["refType"] == "CHILD_OF"]
            if not parents:
                break
            current = ids[parents[0]]
        assert expected <= ancestors, "Missing expected service in request ancestry"
    return {"services": sorted({service(s) for s in spans}), "spans": len(spans),
            "parentage": "verified", "privacy": "verified", "inference_ms": round(max(s["duration"] for s in roots) / 1000, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://smg:30000")
    parser.add_argument("--api-path", default="/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", choices=["sglang", "vllm"], required=True)
    parser.add_argument("--mode", choices=["stream", "nonstream", "unsampled"], default="stream")
    parser.add_argument("--otlp-http-endpoint", default="http://otel-collector:4318/v1/traces")
    parser.add_argument("--jaeger-url", default="http://jaeger:16686")
    parser.add_argument("--expect-services", default="")
    parser.add_argument("--no-verify", action="store_true", help="External backend without Jaeger query API")
    args = parser.parse_args()
    tid, sid = secrets.token_hex(16), secrets.token_hex(8)
    sampled = args.mode != "unsampled"
    marker = "TRACE_PRIVACY_" + secrets.token_hex(8)
    body = {"model": args.model, "messages": [{"role": "user", "content": "Say OK. Ignore this test identifier: " + marker}],
            "stream": args.mode != "nonstream", "max_tokens": 32, "temperature": 0}
    headers = {"Content-Type": "application/json", "traceparent": f"00-{tid}-{sid}-{'01' if sampled else '00'}"}
    if os.getenv("TRACE_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["TRACE_API_KEY"]
    start = time.time_ns()
    req = urllib.request.Request(args.base_url.rstrip("/") + args.api_path, data=json.dumps(body).encode(), headers=headers)
    chunks, done = 0, False
    with urllib.request.urlopen(req, timeout=180) as response:
        if body["stream"]:
            for raw in response:
                if raw.strip() == b"data: [DONE]":
                    done = True
                elif raw.startswith(b"data: "):
                    part = json.loads(raw[6:])
                    assert "error" not in part, "Inference stream failed"
                    chunks += bool(part.get("choices"))
            assert done and chunks, "Incomplete inference stream"
        else:
            assert json.load(response).get("choices"), "Inference result missing choices"
    end = time.time_ns()
    if sampled:
        export_root(args.otlp_http_endpoint, tid, sid, start, end, args.mode)
    report = {"trace_id": tid, "mode": args.mode, "request": "passed", "client_seconds": round((end-start)/1e9, 3), "jaeger_path": "/trace/"+tid}
    if args.no_verify:
        report["verification"] = "inspect external backend; export acceptance does not prove delivery"
    elif not sampled:
        time.sleep(8)
        try:
            assert not get_json(args.jaeger_url.rstrip("/")+"/api/traces/"+tid).get("data"), "Unsampled parent ignored"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404, "Jaeger query failed"
        report["sampling"] = "unsampled parent respected"
    else:
        expected = {"trace-probe", "smg", args.engine} | set(filter(None, args.expect_services.split(",")))
        deadline = time.monotonic()+30
        while True:
            try:
                records = get_json(args.jaeger_url.rstrip("/")+"/api/traces/"+tid).get("data", [])
                assert records, "Trace not found"
                details = verify_trace(records[0], tid, args.engine, expected, marker)
                report.update(details)
                break
            except (AssertionError, urllib.error.URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # HTTP error bodies, engine output and URLs can contain private data.
        print(json.dumps({"result": "failed", "error_type": type(exc).__name__}))
        raise SystemExit(1)
