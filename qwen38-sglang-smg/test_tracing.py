"""Run in the SGLang image: python3 /opt/qwen38/test_tracing.py."""
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TraceFlags
from sglang.srt.observability import trace as native
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats, calibrate_time_diff
from smg_grpc_servicer.sglang import qwen38_tracing as bridge


class BridgeTracingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exporter = InMemorySpanExporter()
        cls.provider = TracerProvider()
        cls.provider.add_span_processor(SimpleSpanProcessor(cls.exporter))
        native.tracer = cls.provider.get_tracer("bridge-test")
        native.opentelemetry_initialized = True
        native.trace_set_thread_info("bridge-test")
        calibrate_time_diff()

    def setUp(self):
        self.exporter.clear()
        self.level = patch.object(native, "get_global_trace_level", return_value=1)
        self.level.start()
        self.addCleanup(self.level.stop)

    def request(self, trace_id="1" * 32, sampled=True):
        parent = "2" * 16
        headers = [("traceparent", f"00-{trace_id}-{parent}-{int(sampled):02x}"),
                   ("tracestate", "vendor=value"), ("authorization", "private")]
        stats = APIServerReqTimeStats()
        obj = SimpleNamespace(rid="test-request", input_ids=[1, 2], bootstrap_room=None)
        bridge.prepare(stats, obj, SimpleNamespace(invocation_metadata=lambda: headers))
        stats.set_created_time()
        bridge.attach(stats, obj)
        return stats, obj

    def test_parent_and_pickle_scheduler_handoff(self):
        stats, obj = self.request()
        self.assertIs(obj.time_stats, stats)
        remote = pickle.loads(pickle.dumps(stats))
        self.assertTrue(remote.trace_ctx.is_copy)
        self.assertIsNone(remote.trace_ctx.root_span)
        self.assertEqual(native.trace.get_current_span(remote.trace_ctx.root_span_context)
                         .get_span_context().trace_id, int("1" * 32, 16))
        self.assertEqual(stats.trace_ctx.root_span.get_span_context().trace_state.get("vendor"), "value")
        state = SimpleNamespace(time_stats=stats, output_ids=[3, 4])
        bridge.finish(state)
        bridge.finish(state)  # normal completion + finally must be idempotent
        roots = [s for s in self.exporter.get_finished_spans() if s.name == "sglang.generate"]
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].parent.span_id, int("2" * 16, 16))
        self.assertEqual(roots[0].attributes["gen_ai.usage.output_tokens"], 2)
        self.assertNotIn("authorization", str(roots[0].attributes))

    def test_unsampled_parent(self):
        stats, _ = self.request(sampled=False)
        self.assertFalse(stats.trace_ctx.root_span.get_span_context().trace_flags & TraceFlags.SAMPLED)
        bridge.finish(SimpleNamespace(time_stats=stats, output_ids=[]))
        self.assertEqual(len(self.exporter.get_finished_spans()), 0)

    def test_cancellation_and_independent_contexts(self):
        first, _ = self.request("3" * 32)
        second, _ = self.request("4" * 32)
        bridge.finish(SimpleNamespace(time_stats=first, output_ids=[]), "Cancelled")
        bridge.finish(SimpleNamespace(time_stats=second, output_ids=[]))
        roots = [s for s in self.exporter.get_finished_spans() if s.name == "sglang.generate"]
        self.assertEqual({s.context.trace_id for s in roots}, {int("3" * 32, 16), int("4" * 32, 16)})
        cancelled = next(s for s in roots if s.context.trace_id == int("3" * 32, 16))
        self.assertEqual(cancelled.status.status_code.name, "ERROR")
        self.assertEqual(cancelled.attributes["error.type"], "Cancelled")

    def test_disabled_initialization(self):
        with patch.object(native, "process_tracing_init") as initialize:
            bridge.initialize(SimpleNamespace(enable_trace=False))
            initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
