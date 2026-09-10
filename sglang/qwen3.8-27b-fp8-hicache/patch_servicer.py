#!/usr/bin/env python3
"""Version-bound load-snapshot and tracing fixes for SGLang 0.5.19 / bridge 0.9.1."""
import ast
from pathlib import Path
from importlib.metadata import distribution, version

assert version("sglang") == "0.5.19"
assert version("smg-grpc-servicer") == "0.9.1"
path = distribution("smg-grpc-servicer").locate_file("smg_grpc_servicer/sglang/servicer.py")
source = path.read_text()
# SGLang renamed the prefill bootstrap queue; the proto retains its old name.
before = "result.disaggregation.prefill_prealloc_queue_reqs"
after = "result.disaggregation.prefill_bootstrap_queue_reqs"
assert source.count(before) == 1, "Unexpected servicer source; review compatibility patch"
patched = source.replace(before, after)
ast.parse(patched)
path.write_text(patched)
print("Applied SGLang 0.5.19 prefill queue compatibility patch")

# Request tracing is missing from this bridge release. Keep edits version-bound
# and fail the image build if upstream source changes.
base = path.parent
(base / "qwen38_tracing.py").write_text(
    Path(__file__).with_name("tracing_bridge.py").read_text()
)

def replace_once(source, before, after):
    assert source.count(before) == 1, f"Unexpected tracing patch anchor: {before!r}"
    return source.replace(before, after)

server = base / "server.py"
s = server.read_text()
s = replace_once(s, "    # Load model config to get HF config info",
    "    from smg_grpc_servicer.sglang.qwen38_tracing import initialize\n"
    "    initialize(server_args)\n\n    # Load model config to get HF config info")
ast.parse(s)
server.write_text(s)

manager = base / "request_manager.py"
s = manager.read_text()
s = replace_once(s, "logger = logging.getLogger(__name__)",
    "from smg_grpc_servicer.sglang import qwen38_tracing\n\nlogger = logging.getLogger(__name__)")
s = replace_once(s, "        # TODO: support request tracing\n", "")
s = replace_once(s,
    "        time_stats = APIServerReqTimeStats(disagg_mode=self.disaggregation_mode)\n",
    "        time_stats = APIServerReqTimeStats(disagg_mode=self.disaggregation_mode)\n"
    "        if self.server_args.enable_trace:\n"
    "            qwen38_tracing.prepare(time_stats, obj, grpc_context)\n")
s = replace_once(s, "        time_stats.set_created_time()\n",
    "        time_stats.set_created_time()\n"
    "        if self.server_args.enable_trace:\n"
    "            qwen38_tracing.attach(time_stats, obj)\n")
s = replace_once(s,
    "        if request_id in self.rid_to_state:\n            del self.rid_to_state[request_id]\n",
    "        if request_id in self.rid_to_state:\n"
    "            if self.server_args.enable_trace:\n"
    "                qwen38_tracing.finish(self.rid_to_state[request_id], 'RequestTerminated')\n"
    "            del self.rid_to_state[request_id]\n")
s = replace_once(s, "        if state:\n            state.finished = True\n",
    "        if state:\n"
    "            if self.server_args.enable_trace:\n"
    "                qwen38_tracing.finish(state, 'Cancelled')\n"
    "            state.finished = True\n")
s = replace_once(s,
    "            if output_data[\"finished\"]:\n                state.finished = True\n",
    "            if output_data[\"finished\"]:\n"
    "                if self.server_args.enable_trace:\n"
    "                    reason = output_data[\"meta_info\"].get(\"finish_reason\") or {}\n"
    "                    qwen38_tracing.finish(state, 'SchedulerAbort' if reason.get('type') == 'abort' else None)\n"
    "                state.finished = True\n")
ast.parse(s)
manager.write_text(s)
print("Applied W3C request tracing bridge patch")
