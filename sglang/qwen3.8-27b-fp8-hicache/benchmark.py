#!/usr/bin/env python3
"""Synthetic streaming benchmark; persist timings and usage, never model text."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import statistics
import time
import urllib.request
import uuid


def post_json(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def stream(base, model, prompt, output_tokens, expected_answer=None):
    body = {"model": model, "prompt": prompt, "max_tokens": output_tokens,
            "temperature": 0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    usage = None
    chunks = 0
    digest = hashlib.sha256()
    answer_text = ""
    with urllib.request.urlopen(req, timeout=900) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            data = json.loads(payload)
            if data.get("error"):
                raise RuntimeError(str(data["error"]))
            if data.get("usage"):
                usage = data["usage"]
            for choice in data.get("choices", []):
                if choice.get("text"):
                    first = first or time.perf_counter()
                    chunks += 1
                    digest.update(choice["text"].encode())
                    if expected_answer is not None:
                        answer_text += choice["text"]
    end = time.perf_counter()
    if first is None or usage is None:
        raise RuntimeError("Missing generated tokens or streaming usage")
    output = usage["completion_tokens"]
    details = usage.get("prompt_tokens_details") or {}
    result = {"ttft_s": first - start, "latency_s": end - start,
            "decode_tokens_s": (output - 1) / max(end - first, 1e-9),
            "tpot_ms": 1000 * (end - first) / max(output - 1, 1),
            "prompt_tokens": usage["prompt_tokens"], "output_tokens": output,
            "cached_tokens": details.get("cached_tokens"), "text_chunks": chunks,
            "output_sha256": digest.hexdigest()}
    if expected_answer is not None:
        numbers = re.findall(r"\d+", answer_text)
        result["quality_pass"] = bool(numbers) and numbers[0] == expected_answer
    return result


def percentile(values, fraction):
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (pos - lower)


def prompt_for(tokenizer, size, key):
    # Different first blocks prevent accidental common-prefix warming.
    prefix = f"Synthetic test document {key}. "
    paragraph = "A storage server keeps numbered records in memory and on disk. Each record contains a key, a value, and a checksum. "
    ids = tokenizer.encode(prefix + paragraph * (size // 10 + 10), add_special_tokens=False)
    return tokenizer.decode(ids[:size], skip_special_tokens=False)


def run_wave(args, prompts, concurrency, label):
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(lambda p: stream(args.base_url, args.model, p, args.output_tokens), prompts))
    elapsed = time.perf_counter() - start
    result = {"label": label, "prefix_id": args.prefix_id,
              "concurrency": concurrency, "requests": len(rows), "elapsed_s": elapsed,
              "output_throughput_tokens_s": sum(r["output_tokens"] for r in rows) / elapsed,
              "prompt_throughput_tokens_s": sum(r["prompt_tokens"] for r in rows) / elapsed,
              "ttft_median_s": statistics.median(r["ttft_s"] for r in rows),
              "ttft_p95_s": percentile([r["ttft_s"] for r in rows], .95),
              "tpot_median_ms": statistics.median(r["tpot_ms"] for r in rows),
              "rows": rows}
    print(json.dumps(result), flush=True)


def quality(args, tokenizer):
    cases = [("arithmetic", f"Compute {a} * {b} + {c}. Reply only with the integer.", str(a * b + c))
             for a, b, c in [(17, 25, 13), (123, 7, 19), (83, 42, 51), (51, 19, 29),
                             (204, 3, 72), (97, 11, 6), (35, 16, 20), (72, 24, 15)]]
    for size in [2048, 8192, 24576]:
        filler = prompt_for(tokenizer, size, "retrieval-" + args.prefix_id)
        middle = len(filler) // 2
        needle = " The unique access code for project Kestrel is 739125. "
        question = "\nWhat is the access code for project Kestrel? Reply only with its six digits."
        cases.append((f"retrieval{size}", filler[:middle] + needle + filler[middle:] + question, "739125"))
    rows = []
    for kind, prompt, answer in cases:
        result = post_json(args.base_url, "/v1/chat/completions", {
            "model": args.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}})
        content = result["choices"][0]["message"].get("content") or ""
        numbers = re.findall(r"\d+", content)
        rows.append({"kind": kind, "pass": numbers == [answer], "usage": result.get("usage")})
    passed = sum(row["pass"] for row in rows)
    print(json.dumps({"quality_passed": passed, "quality_total": len(rows), "rows": rows}), flush=True)
    if passed != len(rows):
        raise SystemExit(1)



def precision_check(args, tokenizer):
    """Long-input known-answer probes plus a bounded 2048-token decode.

    Decode completion checks stability only, not semantic correctness.
    """
    rows = []
    for size in [8192, 16384, 28672]:
        for position in [.1, .5, .9]:
            key = f"precision-{size}-{position}"
            filler = prompt_for(tokenizer, size, key)
            offset = int(len(filler) * position)
            answer = str(100000 + (size * 17 + int(position * 1000)) % 900000)
            content = (filler[:offset] + f" The unique access code for project Kestrel is {answer}. "
                       + filler[offset:] + "\nWhat is the access code for project Kestrel? Reply only with its six digits.")
            start = time.perf_counter()
            result = post_json(args.base_url, "/v1/chat/completions", {
                "model": args.model, "messages": [{"role": "user", "content": content}],
                "temperature": 0, "max_tokens": 128,
                "chat_template_kwargs": {"enable_thinking": False}})
            text = result["choices"][0]["message"].get("content") or ""
            row = {"kind": key, "pass": re.findall(r"\d+", text) == [answer],
                   "latency_s": time.perf_counter() - start, "usage": result.get("usage"),
                   "output_sha256": hashlib.sha256(text.encode()).hexdigest()}
            rows.append(row)
            print(json.dumps(row), flush=True)
    decode = stream(args.base_url, args.model,
                    prompt_for(tokenizer, 8192, "precision-long-decode"), 2048)
    print(json.dumps({"kind": "long_decode", "pass": decode["output_tokens"] == 2048,
                      "semantic_accuracy_checked": False, **decode}), flush=True)
    passed = sum(r["pass"] for r in rows)
    print(json.dumps({"kind": "precision_summary", "known_answer_passed": passed,
                      "known_answer_total": len(rows), "long_decode_completed": decode["output_tokens"] == 2048}), flush=True)
    if passed != len(rows) or decode["output_tokens"] != 2048:
        raise SystemExit(1)


def cache_metrics(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        lines = response.read().decode().splitlines()
    result = {}
    for line in lines:
        if line.startswith("#") or not any(k in line for k in ("cache", "prefetch", "backup")):
            continue
        name, value = line.rsplit(" ", 1)
        if "_bucket{" not in name and "_created{" not in name:
            result[name] = float(value)
    return result


def cache_files(path):
    result = {"kv": {"files": 0, "bytes": 0}, "mamba": {"files": 0, "bytes": 0}}
    for entry in os.scandir(path):
        if entry.is_file() and entry.name.endswith(".bin"):
            kind = "mamba" if ".mamba" in entry.name else "kv"
            result[kind]["files"] += 1
            result[kind]["bytes"] += entry.stat().st_size
    return result


def ssd_restore(args, tokenizer):
    import grpc
    from smg_grpc_proto import sglang_scheduler_pb2_grpc as rpc
    from smg_grpc_proto.generated import common_pb2 as common
    stub = rpc.SglangSchedulerStub(grpc.insecure_channel(args.worker))
    if args.restore_record:
        with open(args.restore_record) as file:
            result = json.load(file)
        args.prefix_id = result["prefix_id"]
        args.ssd_retrieval = result.get("ssd_retrieval", False)
        if args.ssd_retrieval and result.get("probe_version") != 3:
            raise RuntimeError("Retrieval seed uses an older prompt format; create a new SSD seed")
    prompt = prompt_for(tokenizer, 8192, "ssd-" + args.prefix_id)
    if args.ssd_retrieval:
        middle = len(prompt) // 2
        content = (prompt[:middle] + " The unique access code for project Kestrel is 739125. "
                   + prompt[middle:] + "\nWhat is the access code for project Kestrel? Reply only with its six digits.")
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                                 tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        excess = len(ids) - 8192
        # Remove only filler, preserving the chat template, needle and question.
        for offset in range(128, 256):
            candidate = tokenizer.decode(ids[:offset] + ids[offset + excess:], skip_special_tokens=False)
            if len(tokenizer.encode(candidate, add_special_tokens=False)) == 8192:
                prompt = candidate
                break
        else:
            raise RuntimeError("Could not construct the exact 8192-token retrieval probe")
    args.expected_answer = "739125" if args.ssd_retrieval else None
    if args.restore_record:
        return finish_ssd_restore(args, prompt, result, args.reset_method)
    result = {"prefix_id": args.prefix_id, "ssd_retrieval": args.ssd_retrieval, "probe_version": 3,
              "before": cache_metrics(args.metrics_url),
              "files_before": cache_files(args.cache_dir)}
    result["cold"] = stream(args.base_url, args.model, prompt, args.output_tokens, args.expected_answer)
    # A repeated insertion gives the lazy GDN cache a stable backup boundary.
    result["prime"] = stream(args.base_url, args.model, prompt, 16)
    previous = None
    stable = 0
    for _ in range(40):
        current = cache_files(args.cache_dir)
        stable = stable + 1 if current == previous else 0
        previous = current
        if stable >= 4:
            break
        time.sleep(.5)
    result["files_backed_up"] = current
    result["after_cold"] = cache_metrics(args.metrics_url)
    if args.mode == "ssd-seed":
        print(json.dumps(result), flush=True)
        if args.expected_answer and not result["cold"]["quality_pass"]:
            raise SystemExit("SSD seed failed its known-answer check")
        return
    flush = stub.FlushCache(common.FlushCacheRequest(timeout_s=30), timeout=45)
    if not flush.success:
        result["reset_failed"] = "FlushCache timed out waiting for idle"
        print(json.dumps(result), flush=True)
        raise SystemExit("FlushCache failed; the JSON can seed a check after a full worker restart")
    return finish_ssd_restore(args, prompt, result, "FlushCache")


def finish_ssd_restore(args, prompt, result, reset_method):
    result["reset_method"] = reset_method
    result["files_after_flush"] = cache_files(args.cache_dir)
    result["after_flush"] = cache_metrics(args.metrics_url)
    def io_reads():
        with open("/sys/fs/cgroup/io.stat") as file:
            return {parts[0]: int(next(v.split("=", 1)[1] for v in parts if v.startswith("rbytes=")))
                    for line in file if (parts := line.split())}
    if args.drop_file_cache:
        count = 0
        for entry in os.scandir(args.cache_dir):
            if entry.is_file() and entry.name.endswith(".bin"):
                with open(entry.path, "rb") as file:
                    os.fsync(file.fileno())
                    os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                count += 1
        result["fadvise_files"] = count
    reads_before = io_reads()
    result["ssd_restore"] = stream(args.base_url, args.model, prompt, args.output_tokens, args.expected_answer)
    result["physical_read_bytes_by_device"] = {device: count - reads_before.get(device, 0)
                                               for device, count in io_reads().items()}
    result["after_restore"] = cache_metrics(args.metrics_url)
    result["warm"] = stream(args.base_url, args.model, prompt, args.output_tokens, args.expected_answer)
    result["after_warm"] = cache_metrics(args.metrics_url)
    total = lambda values: sum(v for k, v in values.items() if k.startswith(("sglang:prefetched_tokens_total{", "sglang_prefetched_tokens_total{")))
    result["prefetched_tokens_delta"] = total(result["after_restore"]) - total(result["after_flush"])
    result["ssd_restore_verified"] = result["prefetched_tokens_delta"] > 0 and result["ssd_restore"]["cached_tokens"] > 0
    result["output_consistent"] = len({result[k]["output_sha256"] for k in ["cold", "ssd_restore", "warm"]}) == 1
    if args.expected_answer is not None:
        result["quality_consistent"] = all(result[k]["quality_pass"] for k in ["cold", "ssd_restore", "warm"])
    print(json.dumps(result), flush=True)
    if not result["ssd_restore_verified"] or not result.get("quality_consistent", result["output_consistent"]):
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://smg:30000")
    p.add_argument("--model", default="Qwen3.8-27B")
    p.add_argument("--tokenizer", default="/model")
    p.add_argument("--mode", choices=["smoke", "benchmark", "quality", "precision", "ssd-seed", "ssd-restore"], default="benchmark")
    p.add_argument("--restore-record", help="SSD seed JSON; use only immediately after a successful GPU/RAM cache reset")
    p.add_argument("--reset-method", choices=["process_restart", "external_FlushCache"], default="process_restart")
    p.add_argument("--ssd-retrieval", action="store_true", help="Check a known retrieval answer across SSD restore")
    p.add_argument("--worker", default="localhost:19051")
    p.add_argument("--metrics-url", default="http://localhost:19052/metrics")
    p.add_argument("--cache-dir", default="/hicache")
    p.add_argument("--drop-file-cache", action="store_true")
    p.add_argument("--lengths", default="2048,8192")
    p.add_argument("--concurrencies", default="1,4,8")
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--waves", type=int, default=2)
    p.add_argument("--requests-per-wave", type=int, default=0)
    p.add_argument("--prefix-id", default=uuid.uuid4().hex)
    args = p.parse_args()
    if args.mode == "smoke":
        result = post_json(args.base_url, "/v1/chat/completions", {
            "model": args.model, "messages": [{"role": "user", "content": "What is 17 + 25? Reply only with the number."}],
            "temperature": 0, "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}})
        content = result["choices"][0]["message"].get("content", "")
        print(json.dumps({"smoke": "pass" if content.strip() == "42" else "fail",
                          "usage": result.get("usage")}))
        if content.strip() != "42":
            raise SystemExit(1)
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if args.mode == "quality":
        quality(args, tokenizer)
        return
    if args.mode == "precision":
        precision_check(args, tokenizer)
        return
    if args.mode in ("ssd-restore", "ssd-seed"):
        ssd_restore(args, tokenizer)
        return
    for size in map(int, args.lengths.split(",")):
        for concurrency in map(int, args.concurrencies.split(",")):
            count = args.requests_per_wave or max(3, concurrency * 2)
            prompts = [prompt_for(tokenizer, size, f"{args.prefix_id}-{size}-{concurrency}-{i}")
                       for i in range(count)]
            for wave in range(args.waves):
                run_wave(args, prompts, concurrency, f"input{size}-wave{wave}")


if __name__ == "__main__":
    main()
