#!/usr/bin/env python3
"""Version-specific compatibility fix for the SGLang 0.5.19 load snapshot."""
import ast
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
