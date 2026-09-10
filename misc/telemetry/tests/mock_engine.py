"""Synthetic protocol backend. No model, no GPU, no request logging."""
import json, os, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
engine = os.environ["ENGINE"]
p = TracerProvider(resource=Resource.create({"service.name": engine}))
p.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint="otel-collector:4317", insecure=True)))
tracer = p.get_tracer("mock-protocol")
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def reply(self, obj):
        raw=json.dumps(obj).encode();self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_GET(self):
        if self.path == "/v1/models": self.reply({"object":"list","data":[{"id":"synthetic","object":"model","owned_by":"test"}]})
        elif self.path == "/get_server_info": self.reply({"model_path":"synthetic","served_model_name":"synthetic","dp_size":1,"tp_size":1,"version":"0.5.12"})
        elif self.path == "/get_model_info": self.reply({"model_path":"synthetic","is_generation":True})
        else: self.reply({"status":"ok"})
    def do_POST(self):
        body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        ctx=TraceContextTextMapPropagator().extract(dict(self.headers.items()))
        # HTTP header names can arrive in any case.
        ctx=TraceContextTextMapPropagator().extract({k.lower():v for k,v in self.headers.items()})
        with tracer.start_as_current_span("llm_request" if engine=="vllm" else " Req synthetic",context=ctx) as span:
            span.set_attribute("gen_ai.latency.time_in_queue",0.012)
            span.set_attribute("gen_ai.latency.time_in_model_prefill",0.015)
            span.set_attribute("gen_ai.latency.time_in_model_decode",0.01)
            span.set_attribute("gen_ai.usage.prompt_tokens",12)
            # Deliberate synthetic leaks exercise the Collector allowlist.
            marker=body['messages'][0]['content']
            span.set_attribute("gen_ai.prompt",marker)
            span.set_attribute("http.request.header.authorization","synthetic-secret")
            span.add_event(marker,{"exception.message":marker,"exception.type":"SyntheticError"})
            span.set_status(trace.Status(trace.StatusCode.ERROR,marker))
            time.sleep(.04)
            if body.get('stream'):
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                self.wfile.write(b'data: {"id":"synthetic","object":"chat.completion.chunk","model":"synthetic","choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}\n\n');self.wfile.flush()
                time.sleep(.02)
                self.wfile.write(b'data: {"id":"synthetic","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
            else:self.reply({"id":"synthetic","object":"chat.completion","model":"synthetic","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}]})
ThreadingHTTPServer(('0.0.0.0',8000),Handler).serve_forever()
