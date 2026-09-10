"""W3C tracing adapter for smg-grpc-servicer 0.9.1 / SGLang 0.5.19.

Only identifiers, timings and token counts are recorded. Request content stays out.
"""


def initialize(server_args):
    if not server_args.enable_trace:
        return
    from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info

    process_tracing_init(server_args.otlp_traces_endpoint, "sglang",
                         trace_modules=server_args.trace_modules)
    trace_set_thread_info("gRPC bridge")


def prepare(time_stats, obj, grpc_context):
    carrier = {}
    if grpc_context is not None:
        carrier = {key: value for key, value in grpc_context.invocation_metadata()
                   if key in ("traceparent", "tracestate") and isinstance(value, str)}
    time_stats.init_trace_ctx(obj.rid, getattr(obj, "bootstrap_room", None), carrier)


def attach(time_stats, obj):
    # Tokenization already happened in SMG. End the native zero-work tokenize slice.
    time_stats.set_tokenize_finish_time()
    ctx = time_stats.trace_ctx
    if ctx.tracing_enable and ctx.root_span is not None:
        ctx.root_span.update_name("sglang.generate")
        ctx.trace_set_root_attrs({"request.id": obj.rid, "rpc.system": "grpc",
                                 "gen_ai.operation.name": "chat",
                                 "gen_ai.usage.input_tokens": len(obj.input_ids)})
    # The bridge uses pickle ZMQ IPC; SGLang reconstructs this in scheduler processes.
    obj.time_stats = time_stats


def finish(state, error_type=None):
    stats = state.time_stats
    if stats.finished_time:
        return
    ctx = stats.trace_ctx
    if ctx.tracing_enable and ctx.root_span is not None:
        if error_type:
            from opentelemetry.trace import StatusCode
            ctx.root_span.set_status(StatusCode.ERROR)
            ctx.root_span.set_attribute("error.type", error_type)
        ctx.root_span.set_attribute("gen_ai.usage.output_tokens", len(state.output_ids))
    stats.set_finished_time()
