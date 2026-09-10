#!/usr/bin/env python3
"""Inspect the worker without storing prompts, generated text or token IDs."""
import argparse
import collections
import json
import time
import urllib.request

import grpc
from google.protobuf.json_format import MessageToDict
from smg_grpc_proto import sglang_scheduler_pb2 as pb, sglang_scheduler_pb2_grpc as rpc
from smg_grpc_proto.generated import common_pb2 as common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["info", "loads", "metrics", "flush", "events", "watch-loads"])
    parser.add_argument("--worker", default="localhost:19051")
    parser.add_argument("--metrics-url", default="http://localhost:19052/metrics")
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    channel = grpc.insecure_channel(args.worker)
    stub = rpc.SglangSchedulerStub(channel)
    convert = lambda x: MessageToDict(x, preserving_proto_field_name=True)
    if args.mode == "info":
        info = convert(stub.GetServerInfo(pb.GetServerInfoRequest(), timeout=30))
        fields = ["model_path", "served_model_name", "tp_size", "dp_size", "enable_dp_attention",
                  "page_size", "kv_cache_dtype", "mamba_ssm_dtype", "max_running_requests",
                  "max_mamba_cache_size", "mamba_full_memory_ratio", "context_length",
                  "mem_fraction_static", "chunked_prefill_size", "hicache_size", "hicache_ratio",
                  "hicache_write_policy", "hicache_storage_backend", "hicache_storage_prefetch_policy",
                  "mamba_radix_cache_strategy", "mamba_track_interval", "attention_backend"]
        cfg = info.pop("server_args", {})
        info["server_args"] = {k: cfg.get(k) for k in fields}
        print(json.dumps(info, indent=2))
    elif args.mode == "loads":
        print(json.dumps(convert(stub.GetLoads(pb.GetLoadsRequest(include=["all"]), timeout=30)), indent=2))
    elif args.mode == "watch-loads":
        start = time.monotonic()
        rows = []
        while time.monotonic() - start < args.seconds:
            reply = stub.GetLoads(pb.GetLoadsRequest(include=["all"]), timeout=5)
            rows.append({"seconds": time.monotonic() - start,
                         "running": sum(x.num_running_reqs for x in reply.loads),
                         "waiting": sum(x.num_waiting_reqs for x in reply.loads),
                         "used_tokens": sum(x.num_used_tokens for x in reply.loads),
                         "capacity": sum(x.max_total_num_tokens for x in reply.loads)})
            time.sleep(.5)
        print(json.dumps({"max_running": max((r["running"] for r in rows), default=0),
                          "max_used_tokens": max((r["used_tokens"] for r in rows), default=0),
                          "samples": rows}))
    elif args.mode == "flush":
        result = stub.FlushCache(common.FlushCacheRequest(timeout_s=30), timeout=45)
        print(json.dumps(convert(result)))
        if not result.success:
            raise SystemExit(1)
    elif args.mode == "metrics":
        with urllib.request.urlopen(args.metrics_url, timeout=30) as response:
            for line in response.read().decode().splitlines():
                if not line.startswith("#") and any(k in line for k in
                    ("hicache", "storage_", "cache_hit", "cached_tokens", "prefix_cache", "token_usage", "prefetched_tokens", "backuped_tokens")):
                    print(line)
    else:
        counts = collections.Counter()
        sizes = collections.Counter()
        ranks = set()
        first = last = None
        call = stub.SubscribeKvEvents(common.SubscribeKvEventsRequest(), timeout=args.seconds)
        try:
            for batch in call:
                first = first if first is not None else batch.sequence_number
                last = batch.sequence_number
                ranks.add(batch.dp_rank)
                counts["batches"] += 1
                for event in batch.events:
                    kind = event.WhichOneof("data")
                    counts[kind] += 1
                    if kind == "stored":
                        for block in event.stored.blocks:
                            sizes[block.block_size] += 1
                            counts["stored_tokens"] += len(block.token_ids)
        except grpc.RpcError as error:
            if error.code() != grpc.StatusCode.DEADLINE_EXCEEDED:
                raise
        print(json.dumps({"counts": counts, "block_sizes": sizes, "ranks": sorted(ranks),
                          "first_sequence": first, "last_sequence": last}))


if __name__ == "__main__":
    main()
