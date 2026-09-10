#!/usr/bin/env python3
"""Connect SMG 1.10.1's existing OTel injector to the SGLang client factory."""
from pathlib import Path
import subprocess

expected = "2f945936b5be75d1a8bbb8a7dbb2f6585ac76e8e"
assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == expected
path = Path("model_gateway/src/routers/grpc/client.rs")
source = path.read_text()
before = '"sglang" => Ok(Self::Sglang(SglangSchedulerClient::connect(url).await?)),'
after = '''"sglang" => Ok(Self::Sglang(
                SglangSchedulerClient::connect_with_trace_injector(
                    url,
                    std::sync::Arc::new(crate::observability::otel_trace::OtelTraceInjector),
                ).await?,
            )),'''
assert source.count(before) == 1, "Review SMG gRPC client factory patch"
path.write_text(source.replace(before, after))
print("Connected SGLang gRPC client to SMG native OTel injector")
