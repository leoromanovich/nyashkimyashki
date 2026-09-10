#!/usr/bin/env python3
"""Finite request/trace probe. Uses Jaeger's read API; no request content is saved."""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ParentBased
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


def fetch(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://smg:30000")
    parser.add_argument("--api-path", default="/v1/chat/completions")
    parser.add_argument("--expect-services", default="")
    parser.add_argument("--jaeger-url", default="http://jaeger:16686")
    parser.add_argument("--otlp-endpoint", default="otel-collector:4317")
    parser.add_argument("--mode", choices=["stream", "nonstream", "cancel", "unsampled"], default="stream")
    parser.add_argument("--no-verify", action="store_true", help="Use with an external trace backend")
    args = parser.parse_args()
    provider = TracerProvider(resource=Resource.create({"service.name": "trace-probe"}),
                              sampler=ParentBased(ALWAYS_OFF) if args.mode == "unsampled" else None)
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=args.otlp_endpoint, insecure=True)))
    tracer = provider.get_tracer("qwen38.trace-probe")
    stream = args.mode != "nonstream"
    headers = {"Content-Type": "application/json"}
    if os.getenv("TRACE_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["TRACE_API_KEY"]
    body = {"model": "Qwen3.8-27B", "messages": [{"role": "user", "content": "Count from one to ten."}],
            "stream": stream, "max_tokens": 1024 if args.mode == "cancel" else 32,
            "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0}
    content_chunks = 0
    started = time.monotonic()
    with tracer.start_as_current_span("probe." + args.mode) as span:
        tid = format(span.get_span_context().trace_id, "032x")
        TraceContextTextMapPropagator().inject(headers)
        req = urllib.request.Request(args.base_url.rstrip("/") + args.api_path,
                                     data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=180) as response:
            if stream:
                for raw in response:
                    if not raw.startswith(b"data: ") or raw.strip() == b"data: [DONE]":
                        continue
                    part = json.loads(raw[6:])
                    if any(c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content")
                           for c in part.get("choices", [])):
                        content_chunks += 1
                    if args.mode == "cancel" and content_chunks >= 3:
                        break
            else:
                result = json.load(response)
                assert result.get("choices"), result
        span.set_attribute("probe.mode", args.mode)
    elapsed = time.monotonic() - started
    report = {"mode": args.mode, "trace_id": tid, "client_seconds": round(elapsed, 3),
              "jaeger_path": "/trace/" + tid}
    expected = {"smg", "sglang", "trace-probe"} | set(filter(None, args.expect_services.split(",")))
    if not args.no_verify and args.mode == "unsampled":
        time.sleep(8)
        try:
            data = fetch(args.jaeger_url.rstrip("/") + "/api/traces/" + tid)
            assert not data.get("data"), "Downstream ignored unsampled parent"
        except urllib.error.HTTPError as exc:
            assert exc.code == 404, exc.code
        report["sampling"] = "unsampled parent respected"
        provider.shutdown()
        print(json.dumps(report, ensure_ascii=False))
        return
    if not args.no_verify:
        deadline = time.monotonic() + 30
        data = None
        complete_seen = None
        while time.monotonic() < deadline:
            try:
                data = fetch(args.jaeger_url.rstrip("/") + "/api/traces/" + tid)
                traces = data.get("data") or []
                ready = False
                if traces:
                    record = traces[0]
                    spans = record["spans"]
                    ids = {s["spanID"] for s in spans}
                    services = {v["serviceName"] for v in record["processes"].values()}
                    ready = (expected <= services
                             and any(s["operationName"] == "sglang.generate" for s in spans)
                             and any(s["operationName"].startswith("Scheduler") for s in spans)
                             and all(ref["spanID"] in ids for s in spans for ref in s["references"]
                                     if ref["refType"] == "CHILD_OF"))
                if ready:
                    # Wait for all service exporters and the complete parent chain.
                    if complete_seen is None:
                        complete_seen = time.monotonic()
                    if time.monotonic() - complete_seen >= 2:
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(1)
        assert data and data.get("data"), "Trace not found in Jaeger"
        record = data["data"][0]
        spans = record["spans"]
        by_id = {s["spanID"]: s for s in spans}
        services = {v["serviceName"] for v in record["processes"].values()}
        assert expected <= services, f"Trace {tid}: missing services {expected - services}"
        roots = [s for s in spans if s["operationName"] == "sglang.generate"]
        assert roots, "No SGLang request span"
        for root in roots:
            parents = [r["spanID"] for r in root["references"] if r["refType"] == "CHILD_OF"]
            assert parents and parents[0] in by_id, "SGLang parent span missing"
            parent = by_id[parents[0]]
            assert record["processes"][parent["processID"]]["serviceName"] == "smg", parent["operationName"]
            tags = {t["key"]: t["value"] for t in root.get("tags", [])}
            if args.mode == "cancel":
                assert tags.get("error.type") == "Cancelled", tags
        for root in roots:
            ancestor_services = set()
            cursor = root
            seen = set()
            while cursor["spanID"] not in seen:
                seen.add(cursor["spanID"])
                ancestor_services.add(record["processes"][cursor["processID"]]["serviceName"])
                parent_ids = [r["spanID"] for r in cursor["references"] if r["refType"] == "CHILD_OF"]
                if not parent_ids:
                    break
                assert parent_ids[0] in by_id, "Missing ancestor in full trace"
                cursor = by_id[parent_ids[0]]
            assert expected <= ancestor_services, ancestor_services
        assert any(s["operationName"].startswith("Scheduler") for s in spans), "No scheduler spans"
        assert all(s["traceID"].lstrip("0") == tid.lstrip("0") and s["duration"] >= 0 for s in spans)
        http_spans = [s for s in spans if s["operationName"] == "http_request"
                      and record["processes"][s["processID"]]["serviceName"] == "smg"]
        assert http_spans, "No SMG HTTP span"
        if args.mode != "cancel":
            assert max(s["duration"] for s in http_spans) + 1000 >= max(s["duration"] for s in roots), "HTTP span ends before generation"
        report.update(services=sorted(services), span_count=len(spans), parentage="verified",
                      inference_ms=round(max(s["duration"] for s in roots) / 1000, 3))
    provider.shutdown()
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
