"""LiteLLM OTel v2 callback: propagate the active proxy span to SMG.

Mount into LiteLLM and append trace_headers to existing callbacks; see TRACING.md.
Only W3C context is forwarded. Requires an active instrumented request context.
"""
from litellm.integrations.custom_logger import CustomLogger
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


def inject_current_context(data):
    if not trace.get_current_span().get_span_context().is_valid:
        return data
    carrier = {}
    TraceContextTextMapPropagator().inject(carrier)
    headers = {k: v for k, v in (data.get("extra_headers") or {}).items()
               if k.lower() not in ("traceparent", "tracestate")}
    headers.update(carrier)
    data["extra_headers"] = headers
    return data


class TraceHeaders(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        return inject_current_context(data)


trace_headers = TraceHeaders()
